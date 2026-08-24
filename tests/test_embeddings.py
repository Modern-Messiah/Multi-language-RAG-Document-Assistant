"""EmbeddingsManager: PersistentClient-backed store operations."""
from langchain.schema import Document

from app.rag.embeddings import EmbeddingsManager


def _manager(tmp_path):
    return EmbeddingsManager(persist_directory=str(tmp_path / "chroma"))


def test_delete_collection_actually_deletes(tmp_path):
    manager = _manager(tmp_path)
    manager.get_vectorstore("col")
    manager.add_documents(
        [Document(page_content="hello world", metadata={"user_id": "u"})],
        ids=["u-x-0"],
    )
    assert manager.count() == 1

    manager.delete_collection("col")
    manager.get_vectorstore("col")

    assert manager.count() == 0


def test_delete_documents_by_filter(tmp_path):
    manager = _manager(tmp_path)
    manager.get_vectorstore("col")
    manager.add_documents(
        [
            Document(page_content="doc one", metadata={"user_id": "a"}),
            Document(page_content="doc two", metadata={"user_id": "b"}),
        ],
        ids=["a-1", "b-1"],
    )

    manager.delete_documents(filter={"user_id": "a"})

    assert manager.count() == 1
    remaining = manager.collection.get()
    assert remaining["metadatas"][0]["user_id"] == "b"
