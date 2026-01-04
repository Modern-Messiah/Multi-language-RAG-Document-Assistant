vectorstore = None
rag_chain = None
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.rag.document_loader import DocumentLoader
from app.rag.text_splitter import TextChunker
from app.rag.embeddings import EmbeddingsManager
from app.rag.chain import RAGChain
from dotenv import load_dotenv
import shutil
import os
from app.models.schemas import QueryRequest, QueryResponse


load_dotenv()

app = FastAPI(
    title="RAG Assistant API",
    description="Upload documents and ask questions using RAG",
    version="0.1.0"
)

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
    global vectorstore, rag_chain

    if not file.filename.endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="Only TXT and PDF supported")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    docs = loader.load_document(file_path)
    chunks = chunker.split_documents(docs)

    try:
        vectorstore = embeddings.load_vectorstore(COLLECTION_NAME)

    
        if vectorstore._collection.count() == 0:
            vectorstore = embeddings.create_vectorstore(chunks, COLLECTION_NAME)
        else:
            embeddings.add_documents(chunks)

    except Exception:
        vectorstore = embeddings.create_vectorstore(chunks, COLLECTION_NAME)

    rag_chain = RAGChain(vectorstore)

    return {
        "message": "Document processed successfully",
        "chunks": len(chunks)
    }




@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet"
        )

    return rag_chain.ask(
        question=request.question,
        language=request.language
    )

