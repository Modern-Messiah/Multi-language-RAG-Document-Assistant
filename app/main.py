import hashlib
import logging
import os
import re
import secrets
import shutil
from contextlib import asynccontextmanager
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
from fastapi.security import APIKeyHeader
from openai import APITimeoutError, RateLimitError

from app.config import Settings, get_settings
from app.models.schemas import (
    ClearResponse,
    DeleteResponse,
    DocumentListResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.rag.chain import RAGChain
from app.rag.document_loader import DocumentLoader
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


def human_size(num_bytes: int) -> str:
    """Render a byte count in the largest unit that keeps it readable.

    Formatting MB with "{:.0f}" alone reported a 1 KB limit as "0 MB".
    """
    for unit, size in (("MB", 1024 * 1024), ("KB", 1024)):
        if num_bytes >= size:
            # One decimal, but "5.0 MB" reads worse than "5 MB".
            value = f"{num_bytes / size:.1f}".rstrip("0").rstrip(".")
            return f"{value} {unit}"
    return f"{num_bytes} bytes"


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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

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

    @application.get("/health")
    async def health():
        return {"status": "ok", "version": application.version}

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

    if not safe_name.lower().endswith((".txt", ".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Only TXT and PDF files are supported"
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

    # ♻️ Identical content already indexed for this owner - skip re-embedding
    try:
        already_indexed = state.embeddings.has_file_hash(file_hash, user_id)
    except Exception:
        logger.exception("Dedup lookup failed")
        raise HTTPException(status_code=503, detail="Vector store unavailable")

    if already_indexed:
        return UploadResponse(
            message="Document already indexed (identical content)",
            filename=safe_name,
            chunks=0,
            duplicate=True,
            file_hash=file_hash,
        )

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


# `def`, not `async def` - see the note on upload_document. The chain's OpenAI
# calls are synchronous, so as a coroutine this blocked the whole event loop.
@router.post("/query", response_model=QueryResponse)
def query_rag(request: Request, payload: QueryRequest):
    try:
        return request.app.state.rag_chain.ask(
            question=payload.question,
            language=payload.language,
            user_id=payload.user_id,
        )
    except RateLimitError as exc:
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
    except Exception:
        logger.exception("Query failed")
        raise HTTPException(status_code=503, detail="Query failed")


# `def`, not `async def`: deleting from ChromaDB and rmtree-ing the upload
# directory are both blocking calls.
@router.post("/clear", response_model=ClearResponse)
def clear_documents(request: Request, user_id: str = USER_ID_QUERY):
    state = request.app.state
    settings: Settings = state.settings

    try:
        state.embeddings.delete_documents(filter={"user_id": user_id})
    except Exception:
        logger.exception("Error clearing documents")
        raise HTTPException(status_code=500, detail="Failed to clear documents")

    # Drop the raw uploads too. Deleting only the vectors left every file the
    # user ever sent on disk forever - unbounded volume growth, and "cleared"
    # documents that are still sitting there.
    # user_id is validated ([A-Za-z0-9_-]{1,64}), so it cannot escape upload_dir.
    owner_dir = settings.upload_dir / user_id
    if owner_dir.is_dir():
        shutil.rmtree(owner_dir, ignore_errors=True)
        if owner_dir.exists():
            logger.warning("Could not fully remove upload dir for %s", user_id)

    return ClearResponse(message="Documents cleared successfully")


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
    try:
        documents = request.app.state.embeddings.list_documents(user_id)
    except Exception:
        logger.exception("Could not list documents")
        raise HTTPException(status_code=503, detail="Vector store unavailable")

    return DocumentListResponse(
        documents=documents,
        total_chunks=sum(doc["chunks"] for doc in documents),
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

    return DeleteResponse(
        message="Document deleted",
        file_hash=file_hash,
        chunks_removed=removed,
    )
