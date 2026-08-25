"""Runtime resilience: cost caps, transport limits, and honest upstream errors.

Every gap here was live in production shape: the chat call carried no
max_tokens, response.usage was read off the wire and discarded, the OpenAI
client kept the SDK's 600 s default timeout while our own clients gave up after
60 s, an OpenAI rate-limit came back as an anonymous 503, and the request
handlers were coroutines wrapping fully synchronous work.
"""
import inspect

import pytest
from openai import APITimeoutError, RateLimitError

from app import main as app_main
from app.config import Settings
from app.rag.chain import RAGChain
from tests.conftest import FakeChatClient, make_settings


class _Store:
    def __init__(self, docs=None):
        self.docs = docs or []

    def similarity_search(self, query, k=3, filter=None):
        return self.docs


def _doc(text="body", source="a.txt"):
    from langchain.schema import Document

    return Document(page_content=text, metadata={"source": source})


# =========================
# Answer length cap
# =========================

def test_max_tokens_is_sent_when_configured():
    client = FakeChatClient()
    chain = RAGChain(_Store([_doc()]), client=client, max_answer_tokens=256)

    chain.ask("q", user_id="u1")

    assert client.calls[0]["max_tokens"] == 256


def test_max_tokens_is_omitted_when_not_configured():
    """Omitted rather than sent as None: the API rejects an explicit null."""
    client = FakeChatClient()
    chain = RAGChain(_Store([_doc()]), client=client)

    chain.ask("q", user_id="u1")

    assert "max_tokens" not in client.calls[0]


def test_max_answer_tokens_reaches_the_chain_from_settings(tmp_path, fake_openai_embeddings):
    from fastapi.testclient import TestClient

    app = app_main.create_app(make_settings(tmp_path, max_answer_tokens=321))

    with TestClient(app):
        assert app.state.rag_chain.max_answer_tokens == 321


@pytest.mark.parametrize("bad", [0, -1])
def test_max_answer_tokens_must_be_positive(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key="k", max_answer_tokens=bad)


# =========================
# Cost accounting
# =========================

def test_token_usage_is_logged_with_the_tenant(caplog):
    client = FakeChatClient()
    chain = RAGChain(_Store([_doc()]), client=client, model="gpt-4o-mini")

    with caplog.at_level("INFO", logger="app.rag.chain"):
        chain.ask("q", user_id="alice")

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "user_id=alice" in logged
    assert "total_tokens=" in logged
    assert "model=gpt-4o-mini" in logged


def test_a_truncated_answer_is_flagged(caplog):
    """finish_reason=length means MAX_ANSWER_TOKENS cut the answer off."""
    client = FakeChatClient(finish_reason="length")
    chain = RAGChain(_Store([_doc()]), client=client, max_answer_tokens=16)

    with caplog.at_level("WARNING", logger="app.rag.chain"):
        chain.ask("q", user_id="u1")

    assert any("truncated" in r.getMessage() for r in caplog.records)


def test_a_complete_answer_is_not_flagged(caplog):
    client = FakeChatClient(finish_reason="stop")
    chain = RAGChain(_Store([_doc()]), client=client)

    with caplog.at_level("WARNING", logger="app.rag.chain"):
        chain.ask("q", user_id="u1")

    assert not any("truncated" in r.getMessage() for r in caplog.records)


def test_usage_logging_survives_a_response_without_usage(caplog):
    """Not every OpenAI-compatible endpoint returns a usage block."""
    client = FakeChatClient(usage=None)
    chain = RAGChain(_Store([_doc()]), client=client)

    with caplog.at_level("INFO", logger="app.rag.chain"):
        result = chain.ask("q", user_id="u1")

    assert result["answer"]  # the answer still came back


# =========================
# Transport limits
# =========================

def test_openai_client_gets_an_explicit_timeout():
    """The SDK default is a 600 s read timeout - longer than any client waits."""
    chain = RAGChain(_Store(), api_key="k", timeout=45.0, max_retries=1)

    assert chain.client.timeout == 45.0
    assert chain.client.max_retries == 1


def test_embedding_client_gets_an_explicit_timeout():
    from app.rag.embeddings import OpenAIEmbeddingFunction

    fn = OpenAIEmbeddingFunction(model="m", api_key="k", timeout=30.0, max_retries=0)

    assert fn.client.timeout == 30.0
    assert fn.client.max_retries == 0


def test_base_url_override_reaches_the_client():
    """Azure and OpenAI-compatible servers need this; pydantic-settings does
    not export .env into os.environ, so the SDK's own fallback never fires."""
    chain = RAGChain(_Store(), api_key="k", base_url="https://example.invalid/v1")

    assert "example.invalid" in str(chain.client.base_url)


def test_transport_settings_reach_both_clients(tmp_path, fake_openai_embeddings):
    from fastapi.testclient import TestClient

    app = app_main.create_app(
        make_settings(tmp_path, openai_timeout=12.5, openai_max_retries=4)
    )

    with TestClient(app):
        assert app.state.rag_chain.client.timeout == 12.5
        assert app.state.rag_chain.client.max_retries == 4


@pytest.mark.parametrize(
    "overrides", [{"openai_timeout": 0}, {"openai_timeout": -1}, {"openai_max_retries": -1}]
)
def test_transport_settings_are_range_checked(overrides):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key="k", **overrides)


# =========================
# Upstream failures are distinguishable
# =========================

def _raise(exc):
    def boom(*args, **kwargs):
        raise exc

    return boom


def test_rate_limit_becomes_429_with_retry_after(api, monkeypatch):
    error = RateLimitError("slow down", response=_FakeHTTPResponse(429), body=None)
    monkeypatch.setattr(api.app_state.rag_chain, "ask", _raise(error))

    response = api.post("/query", json={"question": "hi", "user_id": "u1"})

    assert response.status_code == 429
    assert response.headers.get("Retry-After")


def test_timeout_becomes_504(api, monkeypatch):
    import httpx

    error = APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1"))
    monkeypatch.setattr(api.app_state.rag_chain, "ask", _raise(error))

    response = api.post("/query", json={"question": "hi", "user_id": "u1"})

    assert response.status_code == 504


def test_other_failures_still_return_503(api, monkeypatch):
    monkeypatch.setattr(api.app_state.rag_chain, "ask", _raise(RuntimeError("chroma died")))

    response = api.post("/query", json={"question": "hi", "user_id": "u1"})

    assert response.status_code == 503


def test_upstream_errors_do_not_leak_provider_detail(api, monkeypatch):
    error = RateLimitError(
        "You exceeded your quota for org-SECRET123", response=_FakeHTTPResponse(429), body=None
    )
    monkeypatch.setattr(api.app_state.rag_chain, "ask", _raise(error))

    response = api.post("/query", json={"question": "hi", "user_id": "u1"})

    assert "SECRET123" not in response.text


class _FakeHTTPResponse:
    """Minimal stand-in for the httpx.Response the openai errors carry."""

    def __init__(self, status_code):
        self.status_code = status_code
        self.headers = {}
        self.request = None


# =========================
# The event loop stays free
# =========================

@pytest.mark.parametrize("name", ["upload_document", "query_rag", "clear_documents"])
def test_request_handlers_are_not_coroutines(name):
    """These call OpenAI and ChromaDB synchronously.

    As `async def` they ran on the event loop, so one slow upload froze every
    other request in the process - /health included, which made the compose
    healthcheck restart a backend that was merely busy. A plain `def` handler
    is dispatched to FastAPI's threadpool instead.
    """
    handler = getattr(app_main, name)

    assert not inspect.iscoroutinefunction(handler), (
        f"{name} is async but its body is blocking - it will stall the event loop"
    )


def test_upload_still_works_after_the_sync_switch(api):
    """The body had to stop awaiting file.read(); prove the path still runs."""
    response = api.post(
        "/upload",
        params={"user_id": "u1"},
        files={"file": ("doc.txt", b"RAG grounds answers in retrieved text.", "text/plain")},
    )

    assert response.status_code == 200, response.text
    assert response.json()["chunks"] >= 1
