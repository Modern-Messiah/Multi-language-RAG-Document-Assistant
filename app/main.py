from fastapi import FastAPI, UploadFile, File, HTTPException
from app.rag.document_loader import DocumentLoader
from app.rag.text_splitter import TextChunker
from app.rag.embeddings import EmbeddingsManager
from app.rag.chain import RAGChain
from dotenv import load_dotenv
import shutil
import os

load_dotenv()

app = FastAPI(title="RAG Assistant API")

UPLOAD_DIR = "data/uploads"
VECTOR_DIR = "data/chroma_db"
COLLECTION_NAME = "documents"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Init components
loader = DocumentLoader()
chunker = TextChunker()
embeddings = EmbeddingsManager(persist_directory=VECTOR_DIR)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="Only TXT and PDF supported")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load → Chunk
    docs = loader.load_document(file_path)
    chunks = chunker.split_documents(docs)

    # Create or update vectorstore
    if os.path.exists(VECTOR_DIR) and os.listdir(VECTOR_DIR):
        vs = embeddings.load_vectorstore(COLLECTION_NAME)
        embeddings.add_documents(chunks)
    else:
        embeddings.create_vectorstore(chunks, COLLECTION_NAME)

    return {
        "message": "Document processed successfully",
        "chunks": len(chunks)
    }


@app.post("/query")
async def query_rag(question: str):
    vs = embeddings.load_vectorstore(COLLECTION_NAME)
    rag = RAGChain(vs)

    result = rag.ask(question)
    return result
