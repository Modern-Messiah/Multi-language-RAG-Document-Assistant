import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi import (
    Path as PathParam,
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from openai import APITimeoutError, RateLimitError

from app import byok
from app.activity import ActivityTracker, is_owner_name
from app.config import Settings, get_settings
from app.feedback import FeedbackStorageFull, store_from_settings
from app.humanize import human_size
from app.models.schemas import (
    ClearResponse,
    DeleteResponse,
    DocumentListResponse,
    FeedbackRequest,
    FeedbackResponse,
    OrphanEntry,
    QueryRequest,
    QueryResponse,
    QuotaUsage,
    SweepEntry,
    SweepFailure,
    SweepResponse,
    UploadResponse,
)
from app.observability import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    configure_logging,
    readiness,
    request_id_of,
)
from app.rag.chain import RAGChain
from app.rag.document_loader import SUPPORTED_EXTENSIONS, DocumentLoader
from app.rag.embeddings import EmbeddingsManager
from app.rag.text_splitter import TextChunker

logger = logging.getLogger(__name__)

API_VERSION = "0.2.0"

# user_id doubles as a directory name and a metadata filter value,
# so it is restricted to filesystem- and filter-safe characters.
USER_ID_PATTERN = r"^[A-Za-z0-9_-]+$"
USER_ID_QUERY = Query(..., min_length=1, max_length=64, pattern=USER_ID_PATTERN)

# The content hash is the public handle for a document: 16 hex characters, cut
# from a sha256 in upload_document.
FILE_HASH_PATH = PathParam(..., min_length=16, max_length=16, pattern=r"^[0-9a-f]{16}$")

# Copying .env.template verbatim leaves a "valid" key that anyone can read off
# the repository, which is worse than no key at all because it looks secure.
PLACEHOLDER_API_KEYS = frozenset({
    "change-me-to-a-long-random-string",
    "your-backend-api-key-here",
})

# Stored name is "{16-char hash}_{stem}{ext}"; keep the whole component under
# the 255-byte filename limit of common filesystems.
MAX_STEM_BYTES = 150
MAX_EXT_LEN = 10


def _openai_timeout(request: Request) -> float:
    """The configured OpenAI timeout, for log messages."""
    return request.app.state.settings.openai_timeout


def safe_filename(name: str) -> str:
    """Reduce a client-supplied filename to a safe, length-bounded basename."""
    name = os.path.basename((name or "").replace("\\", "/"))
    name = re.sub(r"[^\w. -]", "_", name).strip()
    stem, ext = os.path.splitext(name)
    stem = stem.encode("utf-8")[:MAX_STEM_BYTES].decode("utf-8", errors="ignore").strip()
    name = stem + ext[:MAX_EXT_LEN]
    if not name or name.startswith(".") or set(name) <= {".", " ", "_"}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


# =========================
# Authentication
# =========================
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    request: Request,
    key: Optional[str] = Depends(api_key_header),
):
    expected = request.app.state.settings.backend_api_key
    if not expected:
        return  # auth disabled (development mode)
    # Compare bytes: secrets.compare_digest raises TypeError on str operands
    # containing non-ASCII, and uvicorn decodes headers as latin-1 - so a
    # non-ASCII BACKEND_API_KEY turned every request into a 500.
    if not key or not secrets.compare_digest(
        key.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


router = APIRouter(dependencies=[Depends(require_api_key)])


# =========================
# App factory & lifespan
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings

    configure_logging()

    if not settings.backend_api_key:
        logger.warning(
            "BACKEND_API_KEY is not set - API authentication is DISABLED"
        )
    elif settings.backend_api_key in PLACEHOLDER_API_KEYS:
        logger.warning(
            "BACKEND_API_KEY is still the .env.template placeholder - "
            "authentication is effectively public. Set a random secret."
        )

    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    # None when FEEDBACK_ENABLED is off, which is what the endpoint checks.
    app.state.feedback = store_from_settings(settings)

    # Who was here when - what lets the sweep tell an abandoned namespace from
    # a quiet one.
    app.state.activity = ActivityTracker(settings.upload_dir)
    # Owners from before markers existed get one now, so "idle" means "idle
    # since the upgrade" rather than "has not uploaded lately".
    app.state.activity.seed_missing()
    app.state.upload_locks = OwnerLocks()

    app.state.loader = DocumentLoader()
    app.state.chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    app.state.embeddings = EmbeddingsManager(
        persist_directory=str(settings.chroma_persist_dir),
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
        base_url=settings.openai_base_url,
    )
    app.state.vectorstore = app.state.embeddings.get_vectorstore(
        settings.collection_name
    )
    app.state.rag_chain = RAGChain(
        app.state.vectorstore,
        model=settings.model_name,
        top_k=settings.top_k_results,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
        max_answer_tokens=settings.max_answer_tokens,
        relevance_threshold=settings.relevance_threshold,
        max_history_turns=settings.max_history_turns,
        mmr_lambda=settings.mmr_lambda,
        embeddings_manager=app.state.embeddings,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
        base_url=settings.openai_base_url,
    )
    yield


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    application = FastAPI(
        title="RAG Assistant API",
        description="Upload documents and ask questions using RAG",
        version=API_VERSION,
        lifespan=lifespan,
    )
    application.state.settings = settings or get_settings()
    application.include_router(router)
    application.add_middleware(RequestContextMiddleware)

    @application.get("/health")
    async def health():
        """Liveness only: the process is answering. Says nothing about whether
        it can serve a query - that is what /ready is for."""
        return {"status": "ok", "version": application.version}

    # Deliberately `def`: the readiness check reads from ChromaDB, which blocks.
    # As a coroutine it would run on the event loop and a slow disk would stall
    # every other request while answering a probe.
    @application.get("/ready")
    def ready():
        """Whether the components a request needs are usable.

        Unauthenticated, like /health: an orchestrator's probe has no API key.
        It therefore reports check names and pass/fail, never why.
        """
        ok, checks = readiness(application.state)
        body = {"status": "ready" if ok else "not ready", "checks": checks}
        return JSONResponse(body, status_code=200 if ok else 503)

    @application.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        """Answer a crash in JSON, naming the request.

        Without this, an unhandled exception produced a plain-text 500 from the
        server itself, outside this app's middleware - so the response carried
        no request id, and the one failure a user most needs to report was the
        one they could not point at. The exception still propagates for the
        server to log and for tests to see.
        """
        request_id = request_id_of(request)
        return JSONResponse(
            {"detail": "Internal server error", "request_id": request_id},
            status_code=500,
            headers={REQUEST_ID_HEADER: request_id},
        )

    return application


def __getattr__(name):
    # Lazy module attribute so `uvicorn app.main:app` works while plain
    # imports (tests, tooling) don't trigger Settings validation.
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =========================
# Upload endpoint
# =========================
# Deliberately `def`, not `async def`: the body is fully synchronous - it reads
# the upload, calls OpenAI to embed and writes to ChromaDB, none of which yield.
# As a coroutine it ran ON the event loop, so one slow upload froze every other
# request in the process, /health included, and the compose healthcheck then
# restarted a backend that was merely busy. FastAPI runs a plain `def` handler
# in a threadpool instead.
@router.post("/upload", response_model=UploadResponse)
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = USER_ID_QUERY,
):
    state = request.app.state
    settings: Settings = state.settings

    safe_name = safe_filename(file.filename)

    if not safe_name.lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. Supported: "
                + ", ".join(e.lstrip(".").upper() for e in SUPPORTED_EXTENSIONS)
            ),
        )

    def too_large() -> HTTPException:
        return HTTPException(
            status_code=400,
            detail=(
                "File too large. Maximum allowed size is "
                f"{human_size(settings.max_file_size)}."
            )
        )

    # 🔒 Reject on the size starlette already knows, BEFORE pulling the body
    # into one bytes object. The multipart parser spools to a temp file, so
    # this is what keeps an oversized upload from becoming resident memory.
    if file.size is not None and file.size > settings.max_file_size:
        raise too_large()

    # file.file is the SpooledTemporaryFile starlette already parsed the body
    # into; reading it directly is the synchronous equivalent of await
    # file.read() and is what lets this handler stay off the event loop.
    contents = file.file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    # Backstop for a client that sent no size metadata.
    if len(contents) > settings.max_file_size:
        raise too_large()

    file_hash = hashlib.sha256(contents).hexdigest()[:16]

    # One upload at a time per owner from here on, and the dedup lookup is
    # inside the fence with everything else. The quota check is a read followed
    # by a write, and two uploads from the same owner arriving together - the
    # web UI sends a multi-file selection back to back, the bot handles updates
    # concurrently - would both read "room for one more" and both proceed. Held
    # through indexing, so the second sees the first's document in the count and
    # not merely its file on disk.
    #
    # The duplicate answer needs the fence too: outside it, an upload could
    # confirm a document is indexed and answer 200 while a sweep, already past
    # its own re-check, deletes the namespace underneath - leaving the user told
    # their document is there when it no longer is.
    with state.upload_locks.for_owner(user_id):
        return _store_and_index(state, settings, user_id, safe_name, file_hash, contents)


def _store_and_index(state, settings, user_id, safe_name, file_hash, contents) -> UploadResponse:
    """The half of an upload that changes what the owner holds."""

    # ♻️ Identical content already indexed for this owner - skip re-embedding
    try:
        already_indexed = state.embeddings.has_file_hash(file_hash, user_id)
    except Exception:
        logger.exception("Dedup lookup failed")
        raise HTTPException(status_code=503, detail="Vector store unavailable")

    if already_indexed:
        state.activity.touch(user_id)
        return UploadResponse(
            message="Document already indexed (identical content)",
            filename=safe_name,
            chunks=0,
            duplicate=True,
            file_hash=file_hash,
        )

    # 📏 Quota, checked after dedup (an identical re-upload adds nothing and
    # must succeed even at the limit) and before anything is written. A new
    # revision of an existing filename replaces the old one, so what the old
    # revision holds is not counted against this upload.
    try:
        superseded = state.embeddings.file_hashes_for_source(safe_name, user_id)
        documents, total_bytes = _owner_usage(
            state, settings, user_id, exclude_hashes=superseded
        )
    except Exception:
        logger.exception("Quota lookup failed")
        raise HTTPException(status_code=503, detail="Vector store unavailable")

    problem = _quota_problem(settings, documents, total_bytes, len(contents), user_id)
    if problem:
        # 413, not 403 or 429: both clients hide 401/403 as an operator
        # problem, and 429 tells them to retry, which will not help. 413 is
        # shown to the user verbatim, and it says what to do.
        raise HTTPException(status_code=413, detail=problem)

    # 💾 Save file under the user's own directory, keyed by content hash.
    # user_id is validated ([A-Za-z0-9_-]{1,64}), so it is path-safe as-is.
    owner_dir = settings.upload_dir / user_id
    file_path = owner_dir / f"{file_hash}_{safe_name}"
    try:
        owner_dir.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(contents)
    except OSError:
        # A full or read-only volume is a server-side condition, not a bad
        # request, and it used to surface as a bare 500 with a traceback.
        logger.exception("Could not store the uploaded file")
        raise HTTPException(status_code=503, detail="Storage unavailable")

    # 📄 Load & chunk. A corrupt or text-free PDF used to escape as an
    # unhandled exception (bare 500 + traceback); it is a bad request, and the
    # unusable file must not be left behind on disk.
    try:
        docs = state.loader.load_document(str(file_path))
        chunks = state.chunker.split_documents(docs)
    except Exception:
        logger.exception("Failed to parse uploaded document")
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Could not read the document. It may be corrupted or empty."
        )

    if not chunks:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Document contains no extractable text"
        )

    # 🏷 Metadata: human-readable source, content hash, owner
    for chunk in chunks:
        chunk.metadata["source"] = safe_name
        chunk.metadata["file_hash"] = file_hash
        chunk.metadata["user_id"] = user_id

    # 📦 Deterministic per-owner IDs. Chroma UPSERTS on an existing ID
    # (it does not skip), but user_id is validated and used raw, so IDs
    # cannot collide across distinct owners.
    ids = [f"{user_id}-{file_hash}-{i}" for i in range(len(chunks))]

    try:
        state.embeddings.add_documents(chunks, ids=ids)
    except Exception:
        logger.exception("Failed to index document")
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Vector store unavailable")

    # ♻️ Same filename, different content: this is a new revision, so retire
    # the old one. Keeping both meant two versions of "report.pdf" answering
    # the same question with no way for the reader to tell which sentence came
    # from which revision. Done after the new one is safely indexed, so a
    # failure above never leaves the owner with neither.
    replaced = _retire_older_revisions(state, settings, safe_name, file_hash, user_id)
    state.activity.touch(user_id)

    return UploadResponse(
        message="Document processed successfully",
        filename=safe_name,
        chunks=len(chunks),
        duplicate=False,
        file_hash=file_hash,
        replaced=replaced,
    )


def _retire_older_revisions(state, settings, source: str, keep_hash: str, user_id: str) -> bool:
    """Drop other revisions stored under the same filename for this owner."""
    try:
        stale = [
            h for h in state.embeddings.file_hashes_for_source(source, user_id)
            if h != keep_hash
        ]
    except Exception:
        logger.exception("Could not look up earlier revisions of %s", source)
        return False

    removed = False
    for old_hash in stale:
        try:
            state.embeddings.delete_by_file_hash(old_hash, user_id)
            _remove_stored_file(settings, user_id, old_hash)
            removed = True
            logger.info(
                "Replaced revision %s of %s for user_id=%s", old_hash, source, user_id
            )
        except Exception:
            # The new revision is already live; a stale leftover is worth a log,
            # not a failed upload.
            logger.exception("Could not retire revision %s of %s", old_hash, source)
    return removed


def _remove_stored_file(settings: Settings, user_id: str, file_hash: str) -> None:
    """Delete the raw upload whose name starts with this content hash."""
    owner_dir = settings.upload_dir / user_id
    if not owner_dir.is_dir():
        return
    for path in owner_dir.glob(f"{file_hash}_*"):
        path.unlink(missing_ok=True)


def _caller_model(request: Request) -> tuple:
    """The key and model this caller asked to answer with, if any.

    A malformed pair is a 400 rather than a silent fall back to the operator's
    key: someone who sent a key meant to use it, and quietly spending the
    operator's money instead is the one outcome nobody asked for.
    """
    try:
        return byok.wanted(request.headers)
    except byok.BringYourOwnKeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# =========================
# Per-owner quotas
# =========================
class OwnerLocks:
    """One lock per owner, so an owner's uploads happen one at a time.

    Striped rather than one lock per user_id: the web UI mints a new owner per
    browser session, and a dictionary keyed on those would grow for as long as
    the process lived. 64 stripes means two unrelated owners occasionally wait
    on each other, which costs a moment; a lock per owner would cost memory
    forever.
    """

    STRIPES = 64

    def __init__(self):
        self._locks = [threading.Lock() for _ in range(self.STRIPES)]

    def for_owner(self, user_id: str) -> threading.Lock:
        # sha256 rather than hash(): the latter is salted per process, which
        # does not matter here but makes debugging a stripe collision odd.
        digest = hashlib.sha256(user_id.encode("utf-8")).digest()
        return self._locks[digest[0] % self.STRIPES]


def _owner_usage(state, settings: Settings, user_id: str, exclude_hashes=()) -> tuple:
    """(documents, bytes) this owner holds, leaving out `exclude_hashes`.

    Documents are counted in the vector store and bytes on disk, because those
    are the two things the limits are about: what the store has to search
    through, and what the volume has to hold. Stored files are named
    "<hash>_<name>", which is how a file is matched to a document.

    Only files that back a listed document are counted. A file with no vectors
    behind it - left by a crash between write and index, or by the /clear of
    an earlier version that deleted vectors only - is invisible in /documents
    and cannot be deleted through the API, so counting it would present the
    user with a limit they have no way to get under. Such orphans are the
    sweep's to reconcile.
    """
    excluded = set(exclude_hashes)
    held = {
        d["file_hash"] for d in state.embeddings.list_documents(user_id)
        if d["file_hash"] not in excluded
    }

    return len(held), sum(
        _size_of(path) for path in _stored_files(settings, user_id)
        if path.name.split("_", 1)[0] in held
    )


def _stored_files(settings: Settings, user_id: str) -> list:
    owner_dir = settings.upload_dir / user_id
    if not owner_dir.is_dir():
        return []
    return [path for path in owner_dir.iterdir() if path.is_file()]


def _size_of(path) -> int:
    """Bytes on disk, or 0 if the file is already gone.

    Every caller lists a directory and then stats what it found, and a delete
    or a retired revision can remove a file in between - the listing is not
    always under the owner's lock. A vanished file is worth nothing towards a
    quota, not a 500.
    """
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _quota_problem(settings: Settings, documents: int, total_bytes: int,
                   adding_bytes: int, user_id: str = "") -> Optional[str]:
    """Why an upload of `adding_bytes` may not proceed, or None if it may.

    The wording follows the other user-facing limits: it says what the limit
    is and what to do, and names no environment variable - that goes to the
    log, where the person who can change it is reading.
    """
    if settings.max_documents_per_user and documents + 1 > settings.max_documents_per_user:
        logger.warning(
            "Quota refused upload for user_id=%s: %d documents at "
            "MAX_DOCUMENTS_PER_USER=%d",
            user_id, documents, settings.max_documents_per_user,
        )
        return (
            f"Document limit reached: you hold {documents} of "
            f"{settings.max_documents_per_user}. Remove documents you no longer "
            "need, or ask the operator to raise the limit."
        )
    if settings.max_bytes_per_user and total_bytes + adding_bytes > settings.max_bytes_per_user:
        logger.warning(
            "Quota refused upload for user_id=%s: %d + %d bytes at "
            "MAX_BYTES_PER_USER=%d",
            user_id, total_bytes, adding_bytes, settings.max_bytes_per_user,
        )
        return (
            f"Storage limit reached: this upload would take you to "
            f"{human_size(total_bytes + adding_bytes)} of "
            f"{human_size(settings.max_bytes_per_user)}. Remove documents you no "
            "longer need, or ask the operator to raise the limit."
        )
    return None


def _quota_usage(state, settings: Settings, user_id: str) -> QuotaUsage:
    documents, total_bytes = _owner_usage(state, settings, user_id)
    return QuotaUsage(
        documents=documents,
        max_documents=settings.max_documents_per_user,
        bytes=total_bytes,
        max_bytes=settings.max_bytes_per_user,
    )


# `def`, not `async def` - see the note on upload_document. The chain's OpenAI
# calls are synchronous, so as a coroutine this blocked the whole event loop.
@router.post("/query", response_model=QueryResponse)
def query_rag(request: Request, payload: QueryRequest):
    settings: Settings = request.app.state.settings
    key, model = _caller_model(request)
    client = byok.client_for(key, settings) if key else None

    try:
        answer = request.app.state.rag_chain.ask(
            question=payload.question,
            language=payload.language,
            user_id=payload.user_id,
            history=payload.history,
            client=client,
            model=model,
        )
    except RateLimitError as exc:
        if client:
            # Their key, their quota: "retry shortly" would be wrong advice,
            # and a 429 tells a client to do exactly that.
            byok.close_quietly(client)
            raise HTTPException(
                status_code=400, detail=byok.describe_upstream_refusal(exc)
            )
        # A quota or rate-limit rejection is retryable and temporary; saying so
        # lets a client back off instead of treating it as a broken backend.
        logger.warning("OpenAI rate limited the request: %s", exc)
        raise HTTPException(
            status_code=429,
            detail="Upstream model is rate limited. Please retry shortly.",
            headers={"Retry-After": "20"},
        )
    except APITimeoutError:
        logger.warning("OpenAI timed out after %ss", _openai_timeout(request))
        raise HTTPException(
            status_code=504, detail="The model did not answer in time. Please retry."
        )
    except Exception as exc:
        refusal = byok.describe_upstream_refusal(exc) if client else None
        if refusal:
            # The caller's own key was refused. Every other 401 here is the
            # operator's problem and both clients hide it as one; this one the
            # caller can fix, so it must not travel that path.
            raise HTTPException(status_code=400, detail=refusal)
        logger.exception("Query failed")
        raise HTTPException(status_code=503, detail="Query failed")
    finally:
        byok.close_quietly(client)

    request.app.state.activity.touch(payload.user_id)
    return answer


# `def`, not `async def`: deleting from ChromaDB and rmtree-ing the upload
# directory are both blocking calls.
@router.post("/clear", response_model=ClearResponse)
def clear_documents(request: Request, user_id: str = USER_ID_QUERY):
    state = request.app.state

    try:
        _clear_namespace(state, state.settings, user_id)
    except Exception:
        logger.exception("Error clearing documents")
        raise HTTPException(status_code=500, detail="Failed to clear documents")

    return ClearResponse(message="Documents cleared successfully")


def _clear_namespace(state, settings: Settings, user_id: str) -> None:
    """Remove everything one owner has: vectors, raw uploads, activity marker.

    Shared by /clear and the idle-namespace sweep so the two cannot drift
    into deleting different things. Raises if the vectors cannot be deleted;
    a leftover file is logged, since the data the user cares about is gone.

    Takes the owner's upload lock. Without it a sweep could run between an
    upload's write_bytes and its add_documents and remove the file from under
    it - the upload then answers 200 with nothing stored, or a spurious 400.
    The exact owner a sweep targets is one who has just come back.
    """
    with state.upload_locks.for_owner(user_id):
        _wipe_namespace(state, settings, user_id)


def _wipe_namespace(state, settings: Settings, user_id: str) -> None:
    """_clear_namespace without the lock, for a caller that already holds it."""
    state.embeddings.delete_documents(filter={"user_id": user_id})

    # Drop the raw uploads too. Deleting only the vectors left every file the
    # user ever sent on disk forever - unbounded volume growth, and "cleared"
    # documents that are still sitting there.
    # user_id is validated ([A-Za-z0-9_-]{1,64}), so it cannot escape upload_dir.
    owner_dir = settings.upload_dir / user_id
    if owner_dir.is_dir():
        shutil.rmtree(owner_dir, ignore_errors=True)
        if owner_dir.exists():
            logger.warning("Could not fully remove upload dir for %s", user_id)

    state.activity.forget(user_id)


def _sse(event: dict) -> str:
    """One Server-Sent Event. ensure_ascii=False keeps the payload small."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# `def`, not `async def` - see the note on upload_document.
@router.post("/query/stream")
def query_rag_stream(request: Request, payload: QueryRequest):
    """Answer as the tokens arrive, instead of after the last one.

    /query is unchanged; this is an addition. Five to fifteen seconds of a
    motionless spinner was the most visible latency in the product.

    The generator is primed here on purpose. Retrieval, and the condensing call
    a follow-up needs, both happen on that first step - so their failures are
    still ordinary status codes, rather than something a client has to dig out
    of a stream whose headers already said 200.
    """
    settings: Settings = request.app.state.settings
    key, model = _caller_model(request)
    client = byok.client_for(key, settings) if key else None

    stream = request.app.state.rag_chain.ask_stream(
        question=payload.question,
        language=payload.language,
        user_id=payload.user_id,
        history=payload.history,
        client=client,
        model=model,
    )

    try:
        first_event = next(stream)
    except RateLimitError as exc:
        if client:
            # Their key, their quota: "retry shortly" would be wrong advice,
            # and a 429 tells a client to do exactly that.
            byok.close_quietly(client)
            raise HTTPException(
                status_code=400, detail=byok.describe_upstream_refusal(exc)
            )
        logger.warning("OpenAI rate limited the request: %s", exc)
        raise HTTPException(
            status_code=429,
            detail="Upstream model is rate limited. Please retry shortly.",
            headers={"Retry-After": "20"},
        )
    except APITimeoutError:
        logger.warning("OpenAI timed out after %ss", _openai_timeout(request))
        raise HTTPException(
            status_code=504, detail="The model did not answer in time. Please retry."
        )
    except Exception as exc:
        refusal = byok.describe_upstream_refusal(exc) if client else None
        byok.close_quietly(client)
        if refusal:
            raise HTTPException(status_code=400, detail=refusal)
        logger.exception("Query failed")
        raise HTTPException(status_code=503, detail="Query failed")

    # Recorded once retrieval has succeeded, not when the stream ends: the
    # owner was here whether or not the model finished its sentence.
    request.app.state.activity.touch(payload.user_id)

    def events():
        # The caller's client has to outlive this handler - the stream is read
        # after it returns - so it is closed here, when the last token has gone
        # out or the connection has died. Closing it in the handler would have
        # ended the stream before its first token.
        try:
            yield _sse(first_event)
            try:
                for event in stream:
                    yield _sse(event)
            except Exception:
                # The status line is long gone by now, so the only honest way
                # to report this is in the stream itself.
                logger.exception("Streaming failed after the response started")
                yield _sse({"type": "error", "detail": "The answer was cut short."})
        finally:
            byok.close_quietly(client)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tells nginx not to buffer, which would defeat the point.
            "X-Accel-Buffering": "no",
        },
    )


# =========================
# Document inventory
# =========================
@router.get("/documents", response_model=DocumentListResponse)
def list_documents(request: Request, user_id: str = USER_ID_QUERY):
    """What this owner has indexed.

    There was no way to find this out before: the Streamlit sidebar only knew
    about uploads made in the current browser session, and the bot knew
    nothing at all, so the only way to fix one stale file was to clear
    everything and re-upload.
    """
    state = request.app.state
    try:
        documents = state.embeddings.list_documents(user_id)
        quota = _quota_usage(state, state.settings, user_id)
    except Exception:
        logger.exception("Could not list documents")
        raise HTTPException(status_code=503, detail="Vector store unavailable")

    # Only a namespace that holds something is worth keeping alive. The web UI
    # lists on every rerun under a fresh per-session owner, and a marker per
    # page view would fill the activity directory with nothing.
    if documents:
        state.activity.touch(user_id)
    return DocumentListResponse(
        documents=documents,
        total_chunks=sum(doc["chunks"] for doc in documents),
        quota=quota,
    )


@router.delete("/documents/{file_hash}", response_model=DeleteResponse)
def delete_document(
    request: Request,
    file_hash: str = FILE_HASH_PATH,
    user_id: str = USER_ID_QUERY,
):
    """Remove one document: its chunks and the raw file behind it.

    Owner-scoped, so the same content held by another tenant is untouched -
    the hash is identical across owners by construction.
    """
    state = request.app.state
    settings: Settings = state.settings

    try:
        removed = state.embeddings.delete_by_file_hash(file_hash, user_id)
    except Exception:
        logger.exception("Could not delete document %s", file_hash)
        raise HTTPException(status_code=503, detail="Vector store unavailable")

    if not removed:
        raise HTTPException(status_code=404, detail="No such document")

    _remove_stored_file(settings, user_id, file_hash)
    state.activity.touch(user_id)

    return DeleteResponse(
        message="Document deleted",
        file_hash=file_hash,
        chunks_removed=removed,
    )


# =========================
# Answer feedback
# =========================
# Deliberately `def`: it appends to a file, which blocks.
@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: Request, payload: FeedbackRequest):
    """Record one rating of one answer.

    The golden set the evaluation harness measures against was written by
    guessing what people would ask. A rating carries the real question, the
    answer and the sources behind it, so a complaint becomes a case that can be
    measured instead of a story.
    """
    store = getattr(request.app.state, "feedback", None)
    if store is None:
        # FEEDBACK_ENABLED is off: the feature does not exist in this
        # deployment, which is what 404 says. Nothing is created on disk.
        raise HTTPException(status_code=404, detail="Feedback collection is disabled")

    try:
        store.record(
            rating=payload.rating,
            user_id=payload.user_id,
            question=payload.question,
            answer=payload.answer,
            sources=payload.sources,
            request_id=payload.request_id,
            comment=payload.comment,
            language=payload.language,
            client=payload.client,
        )
    except FeedbackStorageFull:
        logger.warning(
            "Feedback storage is full at %s bytes; rotate the file",
            request.app.state.settings.feedback_max_bytes,
        )
        raise HTTPException(
            status_code=507,
            detail="Feedback storage is full. The operator needs to rotate it.",
        )
    except OSError:
        logger.exception("Could not write feedback")
        raise HTTPException(status_code=503, detail="Feedback storage unavailable")

    request.app.state.activity.touch(payload.user_id)
    return FeedbackResponse(message="Thanks - recorded.")


# =========================
# Idle-namespace sweep
# =========================
# The web UI mints a fresh owner per browser session, so its documents are
# orphaned the moment the tab closes and can never be reached again - and since
# Stage 14 they are faithfully copied into every backup. This finds namespaces
# nobody has touched in `idle_days` and, when asked twice, removes them.
#
# In-process rather than an offline script because ChromaDB must have a single
# writer: the same reason backup and restore insist the backend is stopped.
# Everything is a dry run unless apply=true, and apply refuses on its own in
# the situations where the activity data is most likely to be wrong.

# Below this, apply needs force. A typo in a cron line must not sweep
# yesterday's users; a web session left open over a weekend is not abandoned.
MIN_IDLE_DAYS_TO_APPLY = 7

# Deliberately `def`: a full metadata scan of the collection plus a stat per
# owner, none of which yields.
@router.post("/maintenance/sweep", response_model=SweepResponse)
def sweep_idle_namespaces(
    request: Request,
    idle_days: int = Query(..., ge=1, le=3650),
    # Required, with the empty string allowed and meaning every owner: the
    # choice to sweep every tenant has to be written into the request, never
    # arrived at by leaving something out.
    prefix: str = Query(..., max_length=64, pattern=r"^[A-Za-z0-9_-]*$"),
    apply: bool = False,
    force: bool = False,
):
    state = request.app.state
    settings: Settings = state.settings
    activity: ActivityTracker = state.activity

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=idle_days)

    # Every name the data knows: owners with vectors, owners with a directory,
    # owners with a marker. A name that could not be a user_id is reported and
    # never acted on - ".activity" itself is one, and so is whatever an
    # operator or a filesystem left there.
    try:
        with_vectors = set(state.embeddings.list_owners())
    except Exception:
        logger.exception("Could not enumerate owners")
        raise HTTPException(status_code=503, detail="Vector store unavailable")
    names = with_vectors | activity.owner_dirs() | activity.known_owners()

    foreign = sorted(name for name in names if not is_owner_name(name))
    valid = sorted(name for name in names if is_owner_name(name))

    # "Anyone active at all" is judged over every owner, not only the prefix:
    # after a restore or a long stop everyone looks idle at once, and that is
    # stale data, not a mass departure. A prefix sweep of web sessions while
    # Telegram users are busy is fine; a sweep where nobody has been seen is
    # the one that needs a second look.
    dated = {name: activity.last_seen(name) for name in valid}
    known_dates = [seen for seen in dated.values() if seen is not None]
    newest_seen = max(known_dates) if known_dates else None
    anyone_active = newest_seen is not None and newest_seen >= cutoff

    candidates, empty, unknown, orphans = [], [], [], []
    for owner in valid:
        # Orphan files are looked for under EVERY owner, not only the prefix.
        # They are most likely under the stable ids the default prefix excludes -
        # left by a crash between write and index, or by an older /clear that
        # deleted vectors only - and reporting them costs one lookup. Removing
        # them stays inside the prefix: an operator who asked to sweep web
        # sessions should not have files deleted elsewhere.
        in_scope = owner.startswith(prefix)
        try:
            held = {d["file_hash"] for d in state.embeddings.list_documents(owner)}
        except Exception:
            logger.exception("Could not list documents for %s during sweep", owner)
            raise HTTPException(status_code=503, detail="Vector store unavailable")

        files = _stored_files(settings, owner)
        backed = sum(_size_of(p) for p in files if p.name.split("_", 1)[0] in held)
        stray = [p for p in files if p.name.split("_", 1)[0] not in held]
        if stray:
            orphans.append(OrphanEntry(
                user_id=owner, files=len(stray),
                bytes=sum(_size_of(p) for p in stray),
                in_scope=in_scope,
            ))

        if not in_scope:
            continue

        seen = dated[owner]
        entry = SweepEntry(
            user_id=owner, documents=len(held), bytes=backed,
            last_seen=seen.isoformat() if seen else None,
        )
        if seen is None:
            unknown.append(owner)
        elif seen >= cutoff:
            # Inclusive: exactly idle_days is not yet idle. No test pins the
            # difference between >= and >, because cutoff is derived from this
            # request's own clock and no marker's mtime can be made equal to
            # it; the choice is here so it is at least deliberate.
            continue  # active: not this sweep's business
        elif held or files:
            candidates.append(entry)
        else:
            # A marker or an empty directory left after every document was
            # deleted. Nothing to lose, so cleaned without ceremony.
            empty.append(entry)

    refused = None
    if apply and not force:
        if not prefix:
            refused = (
                "apply without a prefix would sweep every tenant, stable Telegram "
                "ids included. Pass prefix=web- for the web UI's per-session "
                "namespaces, or force=true to mean it."
            )
        elif idle_days < MIN_IDLE_DAYS_TO_APPLY:
            refused = (
                f"apply with idle_days below {MIN_IDLE_DAYS_TO_APPLY} would sweep "
                "namespaces a returning user still expects. Pass force=true to mean it."
            )
        elif activity.touch_failures:
            refused = (
                f"{activity.touch_failures} activity update(s) failed since startup, "
                "so some live owners may look idle. Fix the volume (it is probably "
                "full), restart, then sweep - or pass force=true."
            )
        elif not anyone_active and (candidates or empty):
            refused = (
                f"no owner at all has been active in the last {idle_days} days, which "
                "looks like stale activity data (a restore, a long stop) rather than "
                "everyone leaving. Pass force=true if it really is everyone."
            )

    swept, became_active, failed = [], [], []
    orphans_removed = 0
    if apply and not refused:
        for entry in candidates + empty:
            owner = entry.user_id
            try:
                with state.upload_locks.for_owner(owner):
                    # Re-read under the lock: the owner may have come back while
                    # the list above was being built.
                    seen_now = activity.last_seen(owner)
                    if seen_now is not None and seen_now >= cutoff:
                        became_active.append(owner)
                        continue
                    # Logged BEFORE deleting, so a crash or a failure mid-loop
                    # still leaves a record of what was about to go; the response
                    # only exists if the loop ends. The re-check above comes
                    # first so an owner who came back is not logged as swept.
                    logger.warning(
                        "Sweeping idle namespace user_id=%s documents=%d bytes=%d last_seen=%s",
                        owner, entry.documents, entry.bytes, entry.last_seen,
                    )
                    _wipe_namespace(state, settings, owner)
                swept.append(owner)
            except Exception as exc:
                logger.exception("Could not sweep %s", owner)
                failed.append(SweepFailure(user_id=owner, error=f"{type(exc).__name__}: {exc}"[:200]))

        for orphan in orphans:
            if not orphan.in_scope or orphan.user_id in swept:
                continue  # out of scope, or already gone with the namespace
            try:
                with state.upload_locks.for_owner(orphan.user_id):
                    # Recomputed under the lock: an upload since the listing may
                    # have given a file its vectors.
                    held = {d["file_hash"] for d in state.embeddings.list_documents(orphan.user_id)}
                    for path in _stored_files(settings, orphan.user_id):
                        if path.name.split("_", 1)[0] not in held:
                            path.unlink()
                            orphans_removed += 1
            except Exception as exc:
                logger.exception("Could not remove orphan files for %s", orphan.user_id)
                failed.append(SweepFailure(user_id=orphan.user_id, error=f"{type(exc).__name__}: {exc}"[:200]))

    return SweepResponse(
        idle_days=idle_days,
        prefix=prefix,
        dry_run=not (apply and not refused),
        cutoff=cutoff.isoformat(),
        newest_seen=newest_seen.isoformat() if newest_seen else None,
        candidates=candidates,
        empty=empty,
        unknown=unknown,
        foreign=foreign,
        orphans=orphans,
        swept=swept,
        became_active=became_active,
        failed=failed,
        orphans_removed=orphans_removed,
        refused=refused,
    )
