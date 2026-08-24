"""Guard the guard: the suite's offline promise must itself be tested.

Without these, the fixtures could silently stop stubbing OpenAI and the only
symptom would be a real API bill.
"""
import socket

import pytest

from app.rag.chain import RAGChain


def test_outbound_connections_are_blocked():
    with pytest.raises(RuntimeError, match="must stay"):
        socket.create_connection(("api.openai.com", 443), timeout=1)


def test_outbound_socket_connect_is_blocked():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="must stay"):
            sock.connect(("1.1.1.1", 443))
    finally:
        sock.close()


def test_loopback_is_still_allowed():
    """TestClient and asyncio's Windows self-pipe both need loopback."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect(listener.getsockname())  # must not raise
    finally:
        client.close()
        listener.close()


def test_embedding_calls_are_stubbed(fake_openai_embeddings):
    from app.rag.embeddings import OpenAIEmbeddingFunction

    fn = OpenAIEmbeddingFunction(model="text-embedding-3-small", api_key="not-real")

    vectors = fn.embed_documents(["hello", "world"])

    assert len(vectors) == 2
    assert all(len(v) == 32 for v in vectors), "not the deterministic fake vector"
    # Deterministic: the same text must always embed identically.
    assert fn.embed_query("hello") == vectors[0]


def test_api_fixture_stubs_the_chat_client(api):
    from tests.conftest import FakeChatClient

    assert isinstance(api.app_state.rag_chain.client, FakeChatClient)


def test_openai_api_key_is_never_a_real_one():
    import os

    assert os.environ["OPENAI_API_KEY"] == "test-key-not-real"


# =========================
# Retrieval must always be scoped
# =========================

class _Store:
    def __init__(self):
        self.calls = []

    def similarity_search(self, query, k=3, filter=None):
        self.calls.append(filter)
        return []


@pytest.mark.parametrize("bad", [None, ""])
def test_ask_refuses_an_unscoped_search(bad):
    """A falsy user_id used to mean 'search every tenant'."""
    store = _Store()
    chain = RAGChain(store, client=object())

    with pytest.raises(ValueError, match="user_id is required"):
        chain.ask("anything", user_id=bad)

    assert store.calls == [], "an unscoped search reached the vector store"


def test_ask_without_user_id_argument_also_refuses():
    chain = RAGChain(_Store(), client=object())

    with pytest.raises(ValueError, match="user_id is required"):
        chain.ask("anything")
