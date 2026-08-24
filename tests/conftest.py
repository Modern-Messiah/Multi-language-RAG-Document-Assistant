"""Shared pytest fixtures.

All tests run fully offline, and that is *enforced*, not just intended:

- OpenAI embedding calls are replaced with deterministic local vectors.
- The chat completion client is replaced with a canned responder.
- Outbound socket connections raise, so a future test that forgets either of
  the above fails loudly instead of quietly billing a real API key.

Each test gets its own app instance built from a Settings override, with its
own temporary upload dir and ChromaDB directory.
"""
import hashlib
import random
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_API_KEY = "test-backend-key"
FAKE_ANSWER = "This is a canned offline answer."


_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}


def _is_loopback(address) -> bool:
    # asyncio's Windows event loop builds its self-pipe over a real 127.0.0.1
    # socket, so only off-box destinations may be blocked.
    if isinstance(address, (tuple, list)) and address:
        return str(address[0]) in _LOOPBACK
    return True  # AF_UNIX and friends never leave the machine


@pytest.fixture(autouse=True, scope="session")
def _no_real_network():
    """Fail any attempt to reach a host outside this machine.

    OPENAI_API_KEY is set unconditionally rather than via setdefault: a
    developer with a real key exported would otherwise run the suite with it.
    """
    import os

    os.environ["OPENAI_API_KEY"] = "test-key-not-real"

    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def guarded_connect(self, address, *args, **kwargs):
        if not _is_loopback(address):
            raise RuntimeError(
                f"A test tried to connect to {address!r}. The suite must stay "
                "offline — stub the client instead."
            )
        return real_connect(self, address, *args, **kwargs)

    def guarded_create(address, *args, **kwargs):
        if not _is_loopback(address):
            raise RuntimeError(
                f"A test tried to connect to {address!r}. The suite must stay "
                "offline — stub the client instead."
            )
        return real_create(address, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.create_connection = guarded_create
    try:
        yield
    finally:
        socket.socket.connect = real_connect
        socket.create_connection = real_create


def _fake_vector(text: str, dim: int = 32):
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


@pytest.fixture(autouse=True)
def fake_openai_embeddings(monkeypatch):
    """Replace OpenAI embedding calls with deterministic local vectors."""
    from app.rag.embeddings import OpenAIEmbeddingFunction

    def fake_embed_documents(self, texts, batch_size=100):
        return [_fake_vector(t) for t in texts]

    def fake_embed_query(self, text):
        return _fake_vector(text)

    monkeypatch.setattr(OpenAIEmbeddingFunction, "embed_documents", fake_embed_documents)
    monkeypatch.setattr(OpenAIEmbeddingFunction, "embed_query", fake_embed_query)


class FakeChatClient:
    """Stands in for openai.OpenAI, recording what the chain asked for."""

    def __init__(self, answer: str = FAKE_ANSWER):
        self.answer = answer
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=outer.answer))]
                )

        self.chat = SimpleNamespace(completions=_Completions())


def make_settings(tmp_path, **overrides):
    from app.config import Settings

    defaults = dict(
        openai_api_key="test-key-not-real",
        backend_api_key=TEST_API_KEY,
        upload_dir=tmp_path / "uploads",
        chroma_persist_dir=tmp_path / "chroma",
    )
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


@pytest.fixture()
def api(tmp_path, fake_openai_embeddings):
    """Authenticated TestClient over a fresh app with isolated dirs.

    The chain's chat client is swapped for a fake after startup, so /query can
    be exercised end to end without reaching OpenAI.
    """
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings = make_settings(tmp_path)
    app = create_app(settings)

    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        app.state.rag_chain.client = FakeChatClient()

        client.upload_dir = Path(settings.upload_dir)  # convenience for assertions
        client.app_state = app.state
        client.chat = app.state.rag_chain.client
        yield client
