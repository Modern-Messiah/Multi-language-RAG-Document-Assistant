"""
Embeddings Manager for RAG Assistant
Handles vector generation and ChromaDB operations
"""
import httpx
from openai import OpenAI
from typing import List
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Optional
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIEmbeddingFunction:
    def __init__(self, model: str):
        self.client = OpenAI(
            http_client=httpx.Client(trust_env=False)
        )
        self.model = model

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            all_embeddings.extend(
                [item.embedding for item in response.data]
            )

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        return response.data[0].embedding



class EmbeddingsManager:
    """
    Manage embeddings generation and vector store operations
    
    Features:
    - Generate embeddings using OpenAI
    - Store vectors in ChromaDB
    - Perform similarity search
    - Manage collections
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        embedding_model: str = "text-embedding-3-small"
 ):
        """
        Initialize embeddings manager
        
        Args:
            persist_directory: Path to ChromaDB storage
            embedding_model: OpenAI embedding model to use
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in .env file"
            )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddingFunction(
            model=embedding_model)

        
        self.vectorstore: Optional[Chroma] = None
        
        logger.info(
            f"✅ Initialized EmbeddingsManager: "
            f"model={embedding_model}, dir={persist_directory}"
        )
    
    def create_vectorstore(
        self,
        documents: List[Document],
        collection_name: str = "documents"
    ) -> Chroma:
        """
        Create new vector store from documents
        
        Args:
            documents: List of Document objects to embed
            collection_name: Name for the ChromaDB collection
            
        Returns:
            Chroma vectorstore instance
        """
        if not documents:
            raise ValueError("No documents provided for embedding")
        
        logger.info(
            f"🔄 Creating vectorstore with {len(documents)} documents..."
        )
        
        try:
            # Create vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
                collection_name=collection_name
            )
            
            # Get collection info
            collection = self.vectorstore._collection
            count = collection.count()
            
            logger.info(
                f"✅ Created vectorstore '{collection_name}' "
                f"with {count} vectors"
            )
            
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"❌ Error creating vectorstore: {str(e)}")
            raise
    
    def load_vectorstore(
        self,
        collection_name: str = "documents"
    ) -> Chroma:
        """
        Load existing vector store
        
        Args:
            collection_name: Name of the ChromaDB collection to load
            
        Returns:
            Chroma vectorstore instance
        """
        try:
            logger.info(f"🔄 Loading vectorstore '{collection_name}'...")
            
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            
            # Verify collection exists and has data
            collection = self.vectorstore._collection
            count = collection.count()
            
            if count == 0:
                logger.warning(
                    f"⚠️ Collection '{collection_name}' exists but is empty"
                )
            else:
                logger.info(
                    f"✅ Loaded vectorstore '{collection_name}' "
                    f"with {count} vectors"
                )
            
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"❌ Error loading vectorstore: {str(e)}")
            raise
    
    def add_documents(
        self,
        documents: List[Document]
    ) -> List[str]:
        """
        Add documents to existing vectorstore
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        if not self.vectorstore:
            raise ValueError("No vectorstore loaded. Create or load one first.")
        
        if not documents:
            raise ValueError("No documents provided")
        
        logger.info(f"🔄 Adding {len(documents)} documents to vectorstore...")
        
        try:
            ids = self.vectorstore.add_documents(documents)
            
            logger.info(f"✅ Added {len(ids)} documents")
            
            return ids
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of most similar Document objects
        """
        if not self.vectorstore:
            raise ValueError("No vectorstore loaded. Create or load one first.")
        
        logger.info(f"🔍 Searching for: '{query[:50]}...' (top {k})")
        
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            logger.info(f"✅ Found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching: {str(e)}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3
    ) -> List[tuple[Document, float]]:
        """
        Search with relevance scores
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.vectorstore:
            raise ValueError("No vectorstore loaded")
        
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )
            
            logger.info(
                f"✅ Found {len(results)} results with scores: "
                f"{[f'{score:.3f}' for _, score in results]}"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching with scores: {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str = "documents"):
        """
        Delete a collection from ChromaDB
        
        Args:
            collection_name: Name of collection to delete
        """
        try:
            if self.vectorstore and self.vectorstore._collection.name == collection_name:
                self.vectorstore = None
            
            # Delete from disk
            import shutil
            collection_path = self.persist_directory / collection_name
            if collection_path.exists():
                shutil.rmtree(collection_path)
            
            logger.info(f"✅ Deleted collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"❌ Error deleting collection: {str(e)}")
            raise
    
    def get_collection_info(self) -> dict:
        """
        Get information about current vectorstore
        
        Returns:
            Dictionary with collection information
        """
        if not self.vectorstore:
            return {"error": "No vectorstore loaded"}
        
        try:
            collection = self.vectorstore._collection
            
            info = {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Error getting collection info: {str(e)}")
            return {"error": str(e)}


# Testing and example usage
if __name__ == "__main__":
    """
    Test the EmbeddingsManager
    Note: Requires OPENAI_API_KEY in environment
    """
    from dotenv import load_dotenv
    
    print("\n" + "="*60)
    print("Testing EmbeddingsManager")
    print("="*60 + "\n")
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("   Please create a .env file with:")
        print("   OPENAI_API_KEY=your-key-here\n")
        exit(1)
    
    # Sample documents (from previous chunking example)
    sample_docs = [
        Document(
            page_content="RAG systems combine retrieval and generation for better AI responses.",
            metadata={"source": "intro.txt", "page": 1, "chunk_id": 0}
        ),
        Document(
            page_content="ChromaDB is a vector database optimized for embeddings storage and similarity search.",
            metadata={"source": "tech.txt", "page": 1, "chunk_id": 1}
        ),
        Document(
            page_content="LangChain provides tools for building LLM applications with retrieval capabilities.",
            metadata={"source": "tools.txt", "page": 1, "chunk_id": 2}
        )
    ]
    
    try:
        print("1️⃣ Initializing EmbeddingsManager...")
        print("-" * 60)
        
        manager = EmbeddingsManager(
            persist_directory="./data/test_chroma_db"
        )
        
        print("\n2️⃣ Creating vectorstore...")
        print("-" * 60)
        
        vectorstore = manager.create_vectorstore(
            documents=sample_docs,
            collection_name="test_collection"
        )
        
        # Get collection info
        info = manager.get_collection_info()
        print(f"\n📊 Collection Info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        print("\n3️⃣ Testing similarity search...")
        print("-" * 60)
        
        queries = [
            "What is a vector database?",
            "How does RAG work?",
            "Tell me about LangChain"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: '{query}'")
            results = manager.similarity_search(query, k=2)
            
            for i, doc in enumerate(results):
                print(f"\n   Result {i+1}:")
                print(f"   Source: {doc.metadata.get('source', 'unknown')}")
                print(f"   Content: {doc.page_content[:80]}...")
        
        print("\n4️⃣ Testing search with scores...")
        print("-" * 60)
        
        results_with_scores = manager.similarity_search_with_score(
            "vector embeddings", 
            k=3
        )
        
        print("\n📊 Results with relevance scores:")
        for doc, score in results_with_scores:
            print(f"\n   Score: {score:.4f}")
            print(f"   Source: {doc.metadata.get('source', 'unknown')}")
            print(f"   Content: {doc.page_content[:60]}...")
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60 + "\n")
        
        print("💡 Next steps:")
        print("   1. Integrate with document_loader and text_splitter")
        print("   2. Build the RAG chain (Day 3)")
        print("   3. Create FastAPI endpoints\n")
        
    except ValueError as e:
        print(f"\n❌ Configuration error: {str(e)}\n")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}\n")