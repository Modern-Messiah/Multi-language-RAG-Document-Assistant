"""
Shared pytest fixtures.

All tests run fully offline: OpenAI embedding calls are replaced with a
deterministic fake, and each test gets its own temporary upload dir and
ChromaDB persist dir.
"""
import hashlib
import os
import random
import sys
from pathlib import Path

import pytest

# Must be set before app.main is imported (EmbeddingsManager checks it at init).
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


@pytest.fixture()
def api(tmp_path, monkeypatch):
    """TestClient over app.main with isolated upload/vector dirs and reset globals."""
    import app.main as main
    from app.rag.embeddings import EmbeddingsManager
    from fastapi.testclient import TestClient

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(main, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(
        main, "embeddings", EmbeddingsManager(persist_directory=str(tmp_path / "chroma"))
    )
    monkeypatch.setattr(main, "vectorstore", None)
    monkeypatch.setattr(main, "rag_chain", None)

    client = TestClient(main.app, raise_server_exceptions=False)
    client.upload_dir = upload_dir  # convenience for assertions
    return client
