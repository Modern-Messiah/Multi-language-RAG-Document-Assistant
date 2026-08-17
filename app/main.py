from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
import hashlib
import logging
import os
import re
from typing import Optional

from app.rag.document_loader import DocumentLoader
from app.rag.text_splitter import TextChunker
from app.rag.embeddings import EmbeddingsManager
from app.rag.chain import RAGChain
from app.models.schemas import QueryRequest, QueryResponse

# =========================
# Globals & config
# =========================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

vectorstore = None
rag_chain = None

MAX_FILE_SIZE = 30 * 1024 * 1024  # 30 MB

UPLOAD_DIR = "data/uploads"
VECTOR_DIR = "data/chroma_db"
COLLECTION_NAME = "documents"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# App
# =========================
app = FastAPI(
    title="RAG Assistant API",
    description="Upload documents and ask questions using RAG",
    version="0.1.0"
)

# =========================
# Init components
# =========================
loader = DocumentLoader()
chunker = TextChunker()
embeddings = EmbeddingsManager(persist_directory=VECTOR_DIR)


# Anonymous uploads share one namespace; a client sending this literal value
# simply joins that namespace (user_id is unauthenticated client input anyway).
ANON_USER = "__anon__"

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


def safe_user_dir(user_id: Optional[str]) -> str:
    """Directory name for a user's uploads; user_id is client input, so sanitize."""
    if not user_id:
        return "anon"
    return re.sub(r"[^\w-]", "_", user_id)[:64] or "anon"


def get_vectorstore():
    """Open (get-or-create) the collection, translating failures to 503."""
    global vectorstore
    try:
        vectorstore = embeddings.load_vectorstore(COLLECTION_NAME)
        return vectorstore
    except Exception:
        logger.exception("Failed to open vector store")
        raise HTTPException(status_code=503, detail="Vector store unavailable")


# =========================
# Health endpoint
# =========================
@app.get("/health")
async def health():
    return {"status": "ok", "version": app.version}


# =========================
# Upload endpoint
# =========================
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: Optional[str] = None
):
    global vectorstore, rag_chain

    safe_name = safe_filename(file.filename)

    if not safe_name.lower().endswith((".txt", ".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Only TXT and PDF files are supported"
        )

    # 🔒 Read file to check size
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum allowed size is 30 MB."
        )

    file_hash = hashlib.sha256(contents).hexdigest()[:16]
    owner = user_id or ANON_USER

    get_vectorstore()

    # ♻️ Identical content already indexed for this owner — skip re-embedding
    if embeddings.has_file_hash(file_hash, owner):
        if rag_chain is None:
            rag_chain = RAGChain(vectorstore)
        return {
            "message": "Document already indexed (identical content)",
            "filename": safe_name,
            "chunks": 0,
            "duplicate": True
        }

    # 💾 Save file under the user's own directory, keyed by content hash
    owner_dir = os.path.join(UPLOAD_DIR, safe_user_dir(user_id))
    os.makedirs(owner_dir, exist_ok=True)
    file_path = os.path.join(owner_dir, f"{file_hash}_{safe_name}")
    with open(file_path, "wb") as f:
        f.write(contents)

    # 📄 Load & chunk
    docs = loader.load_document(file_path)
    chunks = chunker.split_documents(docs)

    # 🏷 Metadata: human-readable source, content hash, owner. The owner is
    # always stamped: Chroma cannot filter on a missing metadata key, so the
    # dedup check needs an explicit value even for anonymous uploads.
    for chunk in chunks:
        chunk.metadata["source"] = safe_name
        chunk.metadata["file_hash"] = file_hash
        chunk.metadata["user_id"] = owner

    # 📦 Deterministic per-owner IDs. Chroma UPSERTS on an existing ID
    # (it does not skip), so the ID namespace must be collision-free across
    # owners — hash the raw owner value, never a sanitized/truncated form.
    owner_key = hashlib.sha256(owner.encode("utf-8")).hexdigest()[:16]
    ids = [f"{owner_key}-{file_hash}-{i}" for i in range(len(chunks))]
    embeddings.add_documents(chunks, ids=ids)

    if rag_chain is None:
        rag_chain = RAGChain(vectorstore)

    return {
        "message": "Document processed successfully",
        "filename": safe_name,
        "chunks": len(chunks),
        "duplicate": False
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    global rag_chain, vectorstore

    # Lazy initial load if not done yet
    if vectorstore is None:
        get_vectorstore()

    if rag_chain is None:
        rag_chain = RAGChain(vectorstore)

    return rag_chain.ask(
        question=request.question,
        language=request.language,
        user_id=request.user_id
    )


@app.post("/clear")
async def clear_documents(user_id: Optional[str] = None):
    global vectorstore, rag_chain
    try:
        if vectorstore is None:
            vectorstore = embeddings.load_vectorstore(COLLECTION_NAME)

        embeddings.delete_documents(
            filter={"user_id": user_id or "streamlit_user"}
        )

        return {"message": "Documents cleared successfully"}
    except Exception:
        logger.exception("Error clearing documents")
        raise HTTPException(status_code=500, detail="Failed to clear documents")
