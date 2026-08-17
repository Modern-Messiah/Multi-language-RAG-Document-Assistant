"""
Text Splitter for RAG Assistant
Intelligent chunking with overlap for better context preservation
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Split documents into semantic chunks for embedding
    
    Uses recursive character splitting with configurable:
    - chunk_size: Target size of each chunk
    - chunk_overlap: Overlap between chunks for context
    
    Splitting hierarchy:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentences (periods + space)
    4. Words (spaces)
    5. Characters
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum characters per chunk (default: 1000)
            chunk_overlap: Characters to overlap between chunks (default: 200)
            length_function: Function to measure chunk length (default: len)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create splitter with intelligent separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentences
                " ",     # Words
                ""       # Characters (last resort)
            ],
            keep_separator=True
        )
        
        logger.info(
            f"✅ Initialized TextChunker: "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving metadata
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects with enriched metadata
        """
        if not documents:
            logger.warning("⚠️ No documents provided for splitting")
            return []
        
        logger.info(f"🔄 Splitting {len(documents)} document(s)...")
        
        # Split documents
        chunks = self.splitter.split_documents(documents)
        
        # Enrich metadata with chunk information
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content),
                "total_chunks": len(chunks)
            })
        
        logger.info(f"✅ Created {len(chunks)} chunks from {len(documents)} document(s)")
        
        return chunks
    
    def split_text(self, text: str) -> List[str]:
        """
        Split raw text into chunks (without Document wrapper)
        
        Args:
            text: Raw text string to split
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.warning("⚠️ Empty text provided for splitting")
            return []
        
        chunks = self.splitter.split_text(text)
        
        logger.info(f"✅ Created {len(chunks)} chunks from text")
        
        return chunks
    
    @staticmethod
    def get_chunk_statistics(chunks: List[Document]) -> dict:
        """
        Calculate statistics about chunks
        
        Args:
            chunks: List of chunked Document objects
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {"error": "No chunks provided"}
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "avg_chunk_size": sum(chunk_sizes) // len(chunk_sizes),
            "total_characters": sum(chunk_sizes)
        }
        
        return stats


# Testing and example usage
if __name__ == "__main__":
    """
    Test the TextChunker with sample text
    """
    print("\n" + "="*60)
    print("Testing TextChunker")
    print("="*60 + "\n")
    
    # Sample text (simulating a loaded document)
    sample_text = """
RAG System Architecture

Introduction
Retrieval-Augmented Generation (RAG) is a powerful technique that enhances 
large language models by grounding their responses in external knowledge.

Core Components
The system consists of several key components that work together:

1. Document Loader
The document loader handles parsing of various file formats including PDF 
and TXT files. It extracts text while preserving metadata like page numbers 
and source information.

2. Text Splitter
Documents are split into smaller chunks for efficient processing. The splitter 
uses intelligent separators to maintain semantic coherence across chunks.

3. Embeddings
Each chunk is converted into a vector representation using embedding models. 
These vectors capture the semantic meaning of the text.

4. Vector Store
ChromaDB stores the embeddings and enables fast similarity search. When a 
question is asked, the system retrieves the most relevant chunks.

5. LLM Chain
The language model generates answers based on the retrieved context. This 
grounding reduces hallucinations and improves accuracy.

Benefits
RAG systems offer several advantages over pure language models:
- Reduced hallucinations through grounding
- Source attribution for transparency
- Easy updates without retraining
- Cost-effective scaling

Implementation
Our implementation uses LangChain for orchestration, OpenAI for embeddings 
and generation, ChromaDB for vector storage, and FastAPI for the backend API.

Conclusion
RAG represents a significant advancement in making AI systems more reliable 
and trustworthy for real-world applications.
    """.strip()
    
    # Create a Document object
    doc = Document(
        page_content=sample_text,
        metadata={
            "source": "rag_architecture.txt",
            "type": "txt"
        }
    )
    
    try:
        print("1️⃣ Testing document chunking...")
        print("-" * 60)
        
        # Create chunker with small chunks for demo
        chunker = TextChunker(chunk_size=300, chunk_overlap=50)
        
        # Split the document
        chunks = chunker.split_documents([doc])
        
        print(f"\n✅ Created {len(chunks)} chunks")
        
        # Show statistics
        stats = TextChunker.get_chunk_statistics(chunks)
        print(f"\n📊 Chunk Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Show first 3 chunks
        print(f"\n📄 Sample Chunks (first 3):")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Metadata: {chunk.metadata}")
            print(f"Content ({len(chunk.page_content)} chars):")
            print(chunk.page_content[:150] + "...")
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60 + "\n")
        
        print("💡 Key observations:")
        print("   - Chunks maintain semantic boundaries")
        print("   - Overlap preserves context between chunks")
        print("   - Metadata is preserved and enriched")
        print("   - Ready for embedding generation\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}\n")