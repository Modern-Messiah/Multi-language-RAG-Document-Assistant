"""Tests for cross-encoder reranker functionality.

All tests run fully offline in compliance with tests/conftest.py network isolation.
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.byok import BringYourOwnKeyError, wanted_reranker
from app.config import Settings
from app.rag.chain import RAGChain
from app.rag.reranker import (
    BaseReranker,
    CohereReranker,
    FlashRankReranker,
    NoOpReranker,
    get_reranker,
)


# =========================
# NoOpReranker
# =========================

def test_noop_reranker_preserves_order_and_slices():
    docs = [
        Document(page_content=f"Doc {i}", metadata={"id": i})
        for i in range(10)
    ]
    reranker = NoOpReranker()
    res = reranker.rerank("query", docs, top_n=3)
    assert len(res) == 3
    assert [d.metadata["id"] for d in res] == [0, 1, 2]


def test_noop_reranker_handles_empty_or_zero_top_n():
    reranker = NoOpReranker()
    assert reranker.rerank("query", [], top_n=5) == []
    docs = [Document(page_content="A")]
    assert reranker.rerank("query", docs, top_n=0) == []
    assert reranker.rerank("query", docs, top_n=-1) == []


# =========================
# FlashRankReranker
# =========================

def test_flashrank_reranker_reorders_and_attaches_scores():
    docs = [
        Document(page_content="Bananas and apples in a basket", metadata={"id": 0}),
        Document(page_content="Walmart total revenue in FY2020 was $524B", metadata={"id": 1}),
        Document(page_content="The weather today is rainy and cold", metadata={"id": 2}),
    ]
    reranker = FlashRankReranker()
    # Mock ranker to stay 100% fast and deterministic offline
    mock_ranker = MagicMock()
    mock_ranker.rerank.return_value = [
        {"id": 1, "score": 0.985, "text": docs[1].page_content},
        {"id": 0, "score": 0.120, "text": docs[0].page_content},
        {"id": 2, "score": 0.001, "text": docs[2].page_content},
    ]
    reranker._ranker = mock_ranker

    results = reranker.rerank("What was Walmart revenue in 2020?", docs, top_n=2)
    assert len(results) == 2
    assert results[0].metadata["id"] == 1
    assert results[0].metadata["rerank_score"] == 0.985
    assert results[1].metadata["id"] == 0
    assert results[1].metadata["rerank_score"] == 0.120


def test_flashrank_reranker_handles_exception_gracefully():
    docs = [Document(page_content="A", metadata={"id": 0})]
    reranker = FlashRankReranker()
    mock_ranker = MagicMock()
    mock_ranker.rerank.side_effect = RuntimeError("ONNX error")
    reranker._ranker = mock_ranker

    # Should fall back to vector order without raising
    results = reranker.rerank("query", docs, top_n=1)
    assert len(results) == 1
    assert results[0].page_content == "A"


def test_flashrank_reranker_handles_missing_ranker():
    docs = [Document(page_content="A", metadata={"id": 0})]
    reranker = FlashRankReranker()
    reranker._get_ranker = MagicMock(return_value=None)

    results = reranker.rerank("query", docs, top_n=1)
    assert len(results) == 1
    assert results[0].page_content == "A"


# =========================
# CohereReranker
# =========================

def test_cohere_reranker_success_with_mocked_http():
    docs = [
        Document(page_content="Alpha", metadata={"id": 0}),
        Document(page_content="Beta", metadata={"id": 1}),
        Document(page_content="Gamma", metadata={"id": 2}),
    ]
    reranker = CohereReranker(api_key="test-cohere-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.70},
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        results = reranker.rerank("query", docs, top_n=2)
        assert len(results) == 2
        assert results[0].metadata["id"] == 2
        assert results[0].metadata["rerank_score"] == 0.95
        assert results[1].metadata["id"] == 0
        assert results[1].metadata["rerank_score"] == 0.70

        # Verify payload sent to Cohere
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-cohere-key"
        assert kwargs["json"]["query"] == "query"
        assert kwargs["json"]["top_n"] == 2


def test_cohere_reranker_without_api_key_falls_back():
    docs = [Document(page_content="Doc1", metadata={"id": 1})]
    reranker = CohereReranker(api_key="")
    results = reranker.rerank("query", docs, top_n=1)
    assert len(results) == 1
    assert results[0].page_content == "Doc1"


def test_cohere_reranker_http_error_falls_back_gracefully():
    docs = [Document(page_content="Doc1", metadata={"id": 1})]
    reranker = CohereReranker(api_key="bad-key")

    with patch("httpx.Client.post", side_effect=Exception("Connection refused")):
        results = reranker.rerank("query", docs, top_n=1)
        assert len(results) == 1
        assert results[0].page_content == "Doc1"


# =========================
# get_reranker Factory
# =========================

def test_get_reranker_returns_noop_when_disabled():
    settings = Settings(
        openai_api_key="test",
        reranker_enabled=False,
        reranker_provider="flashrank",
    )
    reranker = get_reranker(settings)
    assert isinstance(reranker, NoOpReranker)


def test_get_reranker_instantiates_flashrank_when_enabled():
    settings = Settings(
        openai_api_key="test",
        reranker_enabled=True,
        reranker_provider="flashrank",
        reranker_model="custom-model",
    )
    reranker = get_reranker(settings)
    assert isinstance(reranker, FlashRankReranker)
    assert reranker.model_name == "custom-model"


def test_get_reranker_instantiates_cohere_when_enabled():
    settings = Settings(
        openai_api_key="test",
        reranker_enabled=True,
        reranker_provider="cohere",
        reranker_api_key="cohere-key-123",
        reranker_model="rerank-multilingual-v3.0",
    )
    reranker = get_reranker(settings)
    assert isinstance(reranker, CohereReranker)
    assert reranker.api_key == "cohere-key-123"
    assert reranker.model == "rerank-multilingual-v3.0"


def test_get_reranker_respects_provider_override():
    settings = Settings(
        openai_api_key="test",
        reranker_enabled=False,
    )
    reranker = get_reranker(settings, provider_override="flashrank")
    assert isinstance(reranker, FlashRankReranker)


# =========================
# BYOK Reranker Headers
# =========================

def test_wanted_reranker_none_when_no_header():
    assert wanted_reranker({}) is None


def test_wanted_reranker_flashrank():
    headers = {"X-Reranker-Provider": "flashrank"}
    reranker = wanted_reranker(headers)
    assert isinstance(reranker, FlashRankReranker)


def test_wanted_reranker_cohere_with_key():
    headers = {
        "X-Reranker-Provider": "cohere",
        "X-Reranker-Key": "my-cohere-key",
        "X-Reranker-Model": "rerank-v3.5",
    }
    reranker = wanted_reranker(headers)
    assert isinstance(reranker, CohereReranker)
    assert reranker.api_key == "my-cohere-key"
    assert reranker.model == "rerank-v3.5"


def test_wanted_reranker_cohere_without_key_raises():
    headers = {"X-Reranker-Provider": "cohere"}
    with pytest.raises(BringYourOwnKeyError, match="Cohere reranker requires an API key"):
        wanted_reranker(headers)


def test_wanted_reranker_unknown_provider_raises():
    headers = {"X-Reranker-Provider": "magic-ranker"}
    with pytest.raises(BringYourOwnKeyError, match="Unknown reranker provider"):
        wanted_reranker(headers)


# =========================
# RAGChain Integration
# =========================

class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search_with_score(self, query, k, filter=None):
        return [(d, 0.2) for d in self.docs[:k]]

    def similarity_search(self, query, k, filter=None):
        return self.docs[:k]


class FakeReranker(BaseReranker):
    def rerank(self, query, documents, top_n=5):
        # Reverse order to verify reranker effect
        reversed_docs = list(reversed(documents))[:top_n]
        for idx, doc in enumerate(reversed_docs):
            doc.metadata["rerank_score"] = 0.99 - (idx * 0.1)
        return reversed_docs


def test_rag_chain_ask_invokes_reranker_and_attaches_score():
    candidates = [
        Document(page_content=f"Candidate chunk {i}", metadata={"source": f"doc_{i}.txt"})
        for i in range(10)
    ]
    store = FakeVectorStore(candidates)

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Mocked answer [1]"))]
    mock_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=20, total_tokens=120)
    mock_client.chat.completions.create.return_value = mock_resp

    chain = RAGChain(
        vectorstore=store,
        client=mock_client,
        top_k=3,
        retrieval_candidates=8,
        reranker=FakeReranker(),
    )

    answer = chain.ask(question="Test question?", user_id="test-user")
    assert answer["answer"] == "Mocked answer"
    assert len(answer["sources"]) == 3
    # First source should be doc_7 because candidates were 8 and FakeReranker reversed them
    assert answer["sources"][0]["source"] == "doc_7.txt"
    assert "rerank_score" in answer["sources"][0]
    assert answer["sources"][0]["rerank_score"] == 0.99
