"""
Embeddings Manager for RAG Assistant
Handles vector generation and ChromaDB operations
"""
import logging
import os
from pathlib import Path
from typing import List, Optional

import chromadb
import httpx
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document
from langchain_chroma import Chroma
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIEmbeddingFunction:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        base_url: str = "",
    ):
        options = {}
        if timeout is not None:
            # Without this the SDK waits up to 600 s while the caller has long
            # since given up, holding a worker thread for nothing.
            options["timeout"] = timeout
        if max_retries is not None:
            options["max_retries"] = max_retries
        if base_url:
            options["base_url"] = base_url

        self.client = OpenAI(
            api_key=api_key,  # None -> read from OPENAI_API_KEY env
            http_client=httpx.Client(trust_env=False),
            **options,
        )
        self.model = model

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        all_embeddings: List[List[float]] = []

        billed_tokens = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            usage = getattr(response, "usage", None)
            billed_tokens += getattr(usage, "total_tokens", 0) or 0

            all_embeddings.extend(
                [item.embedding for item in response.data]
            )

        if billed_tokens:
            # Indexing is the other half of the bill, and it was invisible:
            # the usage field came back on every batch and was discarded.
            logger.info(
                "embedded %d texts in %d batch(es), model=%s total_tokens=%d",
                len(texts),
                (len(texts) + batch_size - 1) // batch_size,
                self.model,
                billed_tokens,
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
    Manage embeddings generation and vector store operations.

    Owns a chromadb.PersistentClient so every operation (count, delete by
    filter, drop collection) goes through the supported public API instead
    of the langchain wrapper's private attributes.
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        embedding_model: str = "text-embedding-3-small",
        embedding_fn=None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        base_url: str = "",
    ):
        """
        Args:
            persist_directory: Path to ChromaDB storage
            embedding_model: OpenAI embedding model to use
            embedding_fn: Injectable embeddings object (tests); defaults to
                OpenAIEmbeddingFunction over the OpenAI API
            api_key: OpenAI API key; defaults to the OPENAI_API_KEY env var
            timeout / max_retries: passed to the OpenAI client, whose own
                defaults (600 s read timeout) outlast every caller we have
            base_url: for Azure or an OpenAI-compatible endpoint
        """
        self.persist_directory = Path(persist_directory)
        self.embedding_model_name = embedding_model

        if embedding_fn is not None:
            self.embeddings = embedding_fn
        else:
            if not (api_key or os.getenv("OPENAI_API_KEY")):
                raise ValueError(
                    "OPENAI_API_KEY not found in environment. "
                    "Please set it in .env file"
                )
            self.embeddings = OpenAIEmbeddingFunction(
                model=embedding_model,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                base_url=base_url,
            )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.vectorstore: Optional[Chroma] = None
        self.collection = None

        logger.info(
            f"✅ Initialized EmbeddingsManager: "
            f"model={embedding_model}, dir={persist_directory}"
        )

    # Recorded in the collection's metadata so a later EMBEDDING_MODEL change
    # is caught at startup instead of corrupting search results.
    MODEL_METADATA_KEY = "embedding_model"

    # =========================
    # Collection lifecycle
    # =========================
    def _assert_embedding_model_matches(self, collection_name: str) -> None:
        metadata = self.collection.metadata or {}
        recorded = metadata.get(self.MODEL_METADATA_KEY)

        if recorded == self.embedding_model_name:
            return

        if recorded is None:
            # Either brand new, or created before this check existed: record
            # the current model rather than guess at an older one.
            if self.collection.count():
                logger.warning(
                    f"⚠️ Collection '{collection_name}' has no recorded "
                    f"embedding model; assuming '{self.embedding_model_name}'"
                )
            try:
                self.collection.modify(
                    metadata={
                        **metadata,
                        self.MODEL_METADATA_KEY: self.embedding_model_name,
                    }
                )
            except Exception:
                logger.warning("Could not record the embedding model")
            return

        raise ValueError(
            f"Collection '{collection_name}' was built with embedding model "
            f"'{recorded}', but EMBEDDING_MODEL is now "
            f"'{self.embedding_model_name}'. Vectors from different models are "
            f"not comparable. Either restore EMBEDDING_MODEL={recorded} or "
            f"delete the collection and re-index "
            f"(remove {self.persist_directory})."
        )

    def get_vectorstore(self, collection_name: str = "documents") -> Chroma:
        """Open (get-or-create) a collection and bind the langchain wrapper.

        Refuses to open a collection that was built with a different embedding
        model. Vectors from two models are not comparable, and the dimensions
        usually differ outright (1536 vs 3072 for text-embedding-3-small vs
        -large), so silently mixing them corrupts every subsequent search.
        """
        # Deliberately no metadata= here: in chromadb 0.4.24
        # get_or_create_collection OVERWRITES the metadata of an existing
        # collection, which would stamp the new model over the recorded one
        # and defeat the very check below.
        self.collection = self.client.get_or_create_collection(collection_name)
        self._assert_embedding_model_matches(collection_name)

        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

        count = self.collection.count()
        if count == 0:
            logger.info(f"ℹ️ Collection '{collection_name}' is empty")
        else:
            logger.info(
                f"✅ Opened collection '{collection_name}' with {count} vectors"
            )
        return self.vectorstore

    def load_vectorstore(self, collection_name: str = "documents") -> Chroma:
        """Backwards-compatible alias for get_vectorstore."""
        return self.get_vectorstore(collection_name)

    def create_vectorstore(
        self,
        documents: List[Document],
        collection_name: str = "documents"
    ) -> Chroma:
        """Open a collection and add the given documents (compat wrapper)."""
        if not documents:
            raise ValueError("No documents provided for embedding")

        self.get_vectorstore(collection_name)
        self.add_documents(documents)
        return self.vectorstore

    def delete_collection(self, collection_name: str = "documents"):
        """Drop a collection and all its vectors."""
        try:
            self.client.delete_collection(collection_name)
            self.vectorstore = None
            self.collection = None
            logger.info(f"✅ Deleted collection '{collection_name}'")
        except Exception as e:
            logger.error(f"❌ Error deleting collection: {str(e)}")
            raise

    def count(self) -> int:
        """Number of vectors in the currently open collection."""
        if self.collection is None:
            return 0
        return self.collection.count()

    # =========================
    # Documents
    # =========================
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the open collection.

        Args:
            documents: List of Document objects to add
            ids: Optional stable IDs. NOTE: Chroma UPSERTS on an existing ID
                 (overwrites the record, it does not skip), so IDs must be
                 unique per owner+content; dedup is the caller's job.

        Returns:
            List of document IDs
        """
        if self.vectorstore is None:
            raise ValueError("No vectorstore loaded. Create or load one first.")

        if not documents:
            raise ValueError("No documents provided")

        logger.info(f"🔄 Adding {len(documents)} documents to vectorstore...")

        try:
            if ids is not None:
                added = self.vectorstore.add_documents(documents, ids=ids)
            else:
                added = self.vectorstore.add_documents(documents)

            logger.info(f"✅ Added {len(added)} documents")

            return added

        except Exception as e:
            logger.error(f"❌ Error adding documents: {str(e)}")
            raise

    def has_file_hash(
        self,
        file_hash: str,
        owner: str
    ) -> bool:
        """
        Check whether this owner already indexed a document with this content
        hash. Always owner-scoped: an unscoped hash lookup would leak whether
        OTHER users possess a given file and would drop legitimate uploads.
        """
        if self.collection is None:
            return False

        where = {"$and": [
            {"file_hash": {"$eq": file_hash}},
            {"user_id": {"$eq": owner}}
        ]}

        results = self.collection.get(where=where, limit=1)
        return bool(results.get("ids"))

    def delete_documents(self, filter: dict):
        """
        Delete documents from the open collection by metadata filter.

        Args:
            filter: Metadata filter (e.g. {"user_id": "123"})
        """
        if self.collection is None:
            return

        try:
            logger.info(f"🗑️ Deleting documents with filter: {filter}")
            self.collection.delete(where=filter)
            logger.info("✅ Documents deleted")
        except Exception as e:
            logger.error(f"❌ Error deleting documents: {str(e)}")
            raise

    # =========================
    # Search
    # =========================
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
        if self.vectorstore is None:
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
    ) -> List[tuple]:
        """
        Search with relevance scores

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of (Document, score) tuples
        """
        if self.vectorstore is None:
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

    def get_collection_info(self) -> dict:
        """
        Get information about the currently open collection.

        Returns:
            Dictionary with collection information
        """
        if self.collection is None:
            return {"error": "No vectorstore loaded"}

        try:
            return {
                "name": self.collection.name,
                "count": self.collection.count(),
                "metadata": self.collection.metadata,
                "embedding_model": self.embedding_model_name,
            }
        except Exception as e:
            logger.error(f"❌ Error getting collection info: {str(e)}")
            return {"error": str(e)}


# Testing and example usage
if __name__ == "__main__":
    """
    Manual smoke run for the EmbeddingsManager.
    Note: requires OPENAI_API_KEY in the environment and makes real API calls.
    """
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("Testing EmbeddingsManager")
    print("=" * 60 + "\n")

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("   Please create a .env file with:")
        print("   OPENAI_API_KEY=your-key-here\n")
        exit(1)

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
        manager = EmbeddingsManager(persist_directory="./data/test_chroma_db")
        manager.create_vectorstore(
            documents=sample_docs,
            collection_name="test_collection"
        )

        info = manager.get_collection_info()
        print("\n📊 Collection Info:")
        for key, value in info.items():
            print(f"   {key}: {value}")

        for query in ["What is a vector database?", "How does RAG work?"]:
            print(f"\n🔍 Query: '{query}'")
            for i, doc in enumerate(manager.similarity_search(query, k=2)):
                print(f"   Result {i + 1}: {doc.metadata.get('source')}: "
                      f"{doc.page_content[:60]}...")

        print("\n✅ Smoke run finished\n")

    except ValueError as e:
        print(f"\n❌ Configuration error: {str(e)}\n")
    except Exception as e:
        print(f"\n❌ Smoke run failed: {str(e)}\n")
