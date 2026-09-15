"""Unit tests for Hybrid retrieval and Reciprocal Rank Fusion (RRF)."""

from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document

from app.rag.bm25 import BM25Index
from app.rag.hybrid import HybridRetriever, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_basic():
    doc_a = Document(page_content="Content A", metadata={"source": "a.txt", "file_hash": "h_a"})
    doc_b = Document(page_content="Content B", metadata={"source": "b.txt", "file_hash": "h_b"})
    doc_c = Document(page_content="Content C", metadata={"source": "c.txt", "file_hash": "h_c"})

    # Dense ranked: A (rank 1), B (rank 2)
    dense_list = [doc_a, doc_b]
    # Sparse ranked: B (rank 1), C (rank 2)
    sparse_list = [doc_b, doc_c]

    # B appears in both lists: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = ~0.0325
    # A appears only in list 1: 1/(60+1) = ~0.01639
    # C appears only in list 2: 1/(60+2) = ~0.01613
    fused = reciprocal_rank_fusion([(dense_list, 1.0), (sparse_list, 1.0)], k_rrf=60)

    assert len(fused) == 3
    # B must be ranked first because it is present in both lists
    assert fused[0].metadata["source"] == "b.txt"
    assert fused[1].metadata["source"] == "a.txt"
    assert fused[2].metadata["source"] == "c.txt"
    assert "rrf_score" in fused[0].metadata


def test_reciprocal_rank_fusion_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([([], 1.0)]) == []


def test_hybrid_retriever_coordination():
    mock_vectorstore = MagicMock()
    doc_dense = Document(page_content="Dense match", metadata={"source": "dense.txt", "user_id": "u1"})
    mock_vectorstore.similarity_search.return_value = [doc_dense]

    bm25 = BM25Index()
    doc_sparse = Document(page_content="Sparse keyword match", metadata={"source": "sparse.txt", "user_id": "u1"})
    bm25.add_documents([doc_sparse], owner="u1")

    retriever = HybridRetriever(
        vectorstore=mock_vectorstore,
        bm25_index=bm25,
        dense_weight=1.0,
        bm25_weight=1.0,
    )

    results = retriever.retrieve(
        query="keyword match",
        filter_dict={"user_id": "u1"},
        k=2,
    )

    assert len(results) == 2
    sources = [d.metadata["source"] for d in results]
    assert "dense.txt" in sources
    assert "sparse.txt" in sources


def test_hybrid_retriever_fallback_on_dense_failure():
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.side_effect = RuntimeError("Chroma down")

    bm25 = BM25Index()
    doc_sparse = Document(page_content="Only BM25 works", metadata={"source": "sparse.txt", "user_id": "u1"})
    bm25.add_documents([doc_sparse], owner="u1")

    retriever = HybridRetriever(
        vectorstore=mock_vectorstore,
        bm25_index=bm25,
    )

    results = retriever.retrieve(
        query="works",
        filter_dict={"user_id": "u1"},
        k=2,
    )

    assert len(results) == 1
    assert results[0].metadata["source"] == "sparse.txt"
