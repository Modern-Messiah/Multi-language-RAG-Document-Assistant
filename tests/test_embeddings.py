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


# =========================
# Embedding-model changes must not corrupt an existing collection
# =========================

def test_new_collection_records_its_embedding_model(tmp_path):
    manager = EmbeddingsManager(
        persist_directory=str(tmp_path / "chroma"),
        embedding_model="text-embedding-3-small",
    )
    manager.get_vectorstore("col")

    assert manager.collection.metadata["embedding_model"] == "text-embedding-3-small"


def test_reopening_with_the_same_model_is_fine(tmp_path):
    path = str(tmp_path / "chroma")
    EmbeddingsManager(persist_directory=path, embedding_model="m-a").get_vectorstore("col")

    reopened = EmbeddingsManager(persist_directory=path, embedding_model="m-a")
    reopened.get_vectorstore("col")  # must not raise

    assert reopened.count() == 0


def test_changing_the_embedding_model_is_refused(tmp_path):
    import pytest

    path = str(tmp_path / "chroma")
    first = EmbeddingsManager(persist_directory=path, embedding_model="text-embedding-3-small")
    first.get_vectorstore("col")
    first.add_documents(
        [Document(page_content="indexed with the small model", metadata={"user_id": "u"})],
        ids=["u-x-0"],
    )

    swapped = EmbeddingsManager(persist_directory=path, embedding_model="text-embedding-3-large")

    with pytest.raises(ValueError) as exc:
        swapped.get_vectorstore("col")

    message = str(exc.value)
    assert "text-embedding-3-small" in message, message
    assert "text-embedding-3-large" in message, message
    assert "not comparable" in message


def test_legacy_collection_without_metadata_is_adopted(tmp_path):
    """A collection created before this check must still open."""
    manager = EmbeddingsManager(
        persist_directory=str(tmp_path / "chroma"), embedding_model="m-a"
    )
    # Create the collection the old way: no metadata at all.
    manager.client.create_collection("legacy")

    manager.get_vectorstore("legacy")  # must not raise

    assert manager.collection.metadata["embedding_model"] == "m-a"


def test_a_different_collection_may_use_a_different_model(tmp_path):
    path = str(tmp_path / "chroma")
    EmbeddingsManager(persist_directory=path, embedding_model="m-a").get_vectorstore("first")

    other = EmbeddingsManager(persist_directory=path, embedding_model="m-b")
    other.get_vectorstore("second")  # separate collection, no conflict

    assert other.collection.metadata["embedding_model"] == "m-b"
