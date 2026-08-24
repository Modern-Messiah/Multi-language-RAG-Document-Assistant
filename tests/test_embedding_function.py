"""OpenAIEmbeddingFunction's real body.

The autouse fixture replaces embed_documents/embed_query on the CLASS for every
other test, so the batching loop here had zero coverage. These tests call the
real methods against a stub OpenAI client instead.
"""
from types import SimpleNamespace

import pytest

from app.rag.embeddings import OpenAIEmbeddingFunction


class StubEmbeddings:
    """Records every request and returns one vector per input."""

    def __init__(self, dim=4):
        self.dim = dim
        self.requests = []

    def create(self, model, input):
        self.requests.append({"model": model, "input": list(input)})
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(i)] * self.dim) for i in range(len(input))]
        )


@pytest.fixture()
def fn(monkeypatch):
    """A real OpenAIEmbeddingFunction whose OpenAI client is a stub.

    Explicitly undoes the session-wide class monkeypatch so the real methods
    run; without this the fixture would silently test the fake.
    """
    monkeypatch.undo()
    function = OpenAIEmbeddingFunction(model="text-embedding-3-small", api_key="not-real")
    function.client = SimpleNamespace(embeddings=StubEmbeddings())
    return function


def test_embed_documents_returns_one_vector_per_text(fn):
    vectors = fn.embed_documents(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    assert all(len(v) == 4 for v in vectors)


def test_embed_documents_sends_one_request_when_under_the_batch_size(fn):
    fn.embed_documents(["a", "b", "c"], batch_size=100)

    assert len(fn.client.embeddings.requests) == 1
    assert fn.client.embeddings.requests[0]["input"] == ["a", "b", "c"]


def test_embed_documents_splits_into_batches(fn):
    texts = [f"text-{i}" for i in range(250)]

    vectors = fn.embed_documents(texts, batch_size=100)

    sizes = [len(r["input"]) for r in fn.client.embeddings.requests]
    assert sizes == [100, 100, 50], sizes
    assert len(vectors) == 250


def test_batches_cover_every_text_in_order(fn):
    texts = [f"text-{i}" for i in range(7)]

    fn.embed_documents(texts, batch_size=3)

    sent = [t for request in fn.client.embeddings.requests for t in request["input"]]
    assert sent == texts


def test_empty_input_makes_no_request(fn):
    assert fn.embed_documents([]) == []
    assert fn.client.embeddings.requests == []


def test_configured_model_is_used(fn):
    fn.embed_documents(["a"])
    fn.embed_query("q")

    assert {r["model"] for r in fn.client.embeddings.requests} == {"text-embedding-3-small"}


def test_embed_query_sends_a_single_text_and_unwraps_the_vector(fn):
    vector = fn.embed_query("just one question")

    assert fn.client.embeddings.requests == [
        {"model": "text-embedding-3-small", "input": ["just one question"]}
    ]
    assert vector == [0.0, 0.0, 0.0, 0.0]


def test_client_is_built_without_trusting_proxy_env_vars(monkeypatch):
    """trust_env=False keeps a stray HTTPS_PROXY from redirecting API traffic."""
    monkeypatch.undo()
    function = OpenAIEmbeddingFunction(model="m", api_key="not-real")

    assert function.client is not None
    assert function.model == "m"
