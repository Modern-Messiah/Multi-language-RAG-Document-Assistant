from app.rag.embeddings import EmbeddingsManager
from app.rag.chain import RAGChain
from dotenv import load_dotenv

load_dotenv()

em = EmbeddingsManager(persist_directory="./data/test_pipeline")
vs = em.load_vectorstore(collection_name="pipeline_test")

rag = RAGChain(vs)

result = rag.ask("What is RAG?")
print(result["answer"])
print(result["sources"])
