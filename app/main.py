import hashlib
import logging
import os
import re
import secrets
import shutil
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings
from app.models.schemas import QueryRequest, QueryResponse
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
    # containing non-ASCII, and uvicorn decodes headers as latin-1 — so a
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
            "BACKEND_API_KEY is not set — API authentication is DISABLED"
        )
    elif settings.backend_api_key in PLACEHOLDER_API_KEYS:
        logger.warning(
            "BACKEND_API_KEY is still the .env.template placeholder — "
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
@router.post("/upload")
async def upload_document(
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
        limit_mb = settings.max_file_size / (1024 * 1024)
        return HTTPException(
            status_code=400,
            detail=f"File too large. Maximum allowed size is {limit_mb:.0f} MB."
        )

    # 🔒 Reject on the size starlette already knows, BEFORE pulling the body
    # into one bytes object. The multipart parser spools to a temp file, so
    # this is what keeps an oversized upload from becoming resident memory.
    if file.size is not None and file.size > settings.max_file_size:
        raise too_large()

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    # Backstop for a client that sent no size metadata.
    if len(contents) > settings.max_file_size:
        raise too_large()

    file_hash = hashlib.sha256(contents).hexdigest()[:16]

    # ♻️ Identical content already indexed for this owner — skip re-embedding
    if state.embeddings.has_file_hash(file_hash, user_id):
        return {
            "message": "Document already indexed (identical content)",
            "filename": safe_name,
            "chunks": 0,
            "duplicate": True
        }

    # 💾 Save file under the user's own directory, keyed by content hash.
    # user_id is validated ([A-Za-z0-9_-]{1,64}), so it is path-safe as-is.
    owner_dir = settings.upload_dir / user_id
    owner_dir.mkdir(parents=True, exist_ok=True)
    file_path = owner_dir / f"{file_hash}_{safe_name}"
    file_path.write_bytes(contents)

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

    return {
        "message": "Document processed successfully",
        "filename": safe_name,
        "chunks": len(chunks),
        "duplicate": False
    }


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: Request, payload: QueryRequest):
    try:
        return request.app.state.rag_chain.ask(
            question=payload.question,
            language=payload.language,
            user_id=payload.user_id,
        )
    except Exception:
        logger.exception("Query failed")
        raise HTTPException(status_code=503, detail="Query failed")


@router.post("/clear")
async def clear_documents(request: Request, user_id: str = USER_ID_QUERY):
    state = request.app.state
    settings: Settings = state.settings

    try:
        state.embeddings.delete_documents(filter={"user_id": user_id})
    except Exception:
        logger.exception("Error clearing documents")
        raise HTTPException(status_code=500, detail="Failed to clear documents")

    # Drop the raw uploads too. Deleting only the vectors left every file the
    # user ever sent on disk forever — unbounded volume growth, and "cleared"
    # documents that are still sitting there.
    # user_id is validated ([A-Za-z0-9_-]{1,64}), so it cannot escape upload_dir.
    owner_dir = settings.upload_dir / user_id
    if owner_dir.is_dir():
        shutil.rmtree(owner_dir, ignore_errors=True)
        if owner_dir.exists():
            logger.warning("Could not fully remove upload dir for %s", user_id)

    return {"message": "Documents cleared successfully"}
