"""Score-gated retrieval.

Retrieval used to hand the model TOP_K_RESULTS chunks whether or not the corpus
had anything to do with the question, so asking about a topic the user never
uploaded still filled the prompt with their nearest unrelated paragraphs -
leaving "I don't know" entirely to the prompt's good behaviour.

The scores ChromaDB returns are distances, not similarities: for the default
squared-L2 space 0.0 means identical. Reading them the wrong way round would
discard the best matches, so the conversion gets its own tests.
"""
from types import SimpleNamespace

import pytest
from langchain.schema import Document

from app.config import Settings
from app.rag.chain import RAGChain
from app.rag.embeddings import DEFAULT_SPACE, distance_to_similarity


class ScoredStore:
    """A vector store that reports distances, like langchain_chroma does."""

    def __init__(self, scored, space=DEFAULT_SPACE):
        self.scored = scored
        self.calls = []
        self._collection = SimpleNamespace(metadata={"hnsw:space": space} if space else {})

    def similarity_search_with_score(self, query, k=4, filter=None):
        self.calls.append({"query": query, "k": k, "filter": filter})
        return self.scored[:k]

    def similarity_search(self, query, k=4, filter=None):  # pragma: no cover
        raise AssertionError("the scored path should have been used")


class UnscoredStore:
    """An older store, or an injected double, with no scored search."""

    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def similarity_search(self, query, k=4, filter=None):
        self.calls.append({"query": query, "k": k, "filter": filter})
        return self.docs


class FakeChat:
    def __init__(self):
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="an answer"),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )

        self.chat = SimpleNamespace(completions=_Completions())


def _doc(source):
    return Document(page_content=f"text of {source}", metadata={"source": source})


def _context_of(chat):
    return chat.calls[0]["messages"][1]["content"]


# =========================
# Distance to similarity
# =========================

@pytest.mark.parametrize(
    "distance,expected",
    [(0.0, 1.0), (1.0, 0.5), (2.0, 0.0), (4.0, -1.0)],
)
def test_squared_l2_converts_to_cosine(distance, expected):
    """d = 2 - 2*cos for unit-normed embeddings, so cos = 1 - d/2."""
    assert distance_to_similarity(distance, "l2") == pytest.approx(expected)


def test_identical_content_is_similarity_one():
    assert distance_to_similarity(0.0) == 1.0


def test_orthogonal_content_is_similarity_zero():
    assert distance_to_similarity(2.0) == 0.0


@pytest.mark.parametrize("distance,expected", [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)])
def test_cosine_space_is_one_minus_distance(distance, expected):
    assert distance_to_similarity(distance, "cosine") == pytest.approx(expected)


def test_inner_product_space_negates():
    assert distance_to_similarity(-0.8, "ip") == pytest.approx(0.8)


def test_an_unknown_space_has_no_conversion():
    """Returning a number here would compare against a scale that does not
    apply, which silently throws away the best matches."""
    assert distance_to_similarity(0.5, "hamming") is None


def test_the_default_space_is_what_chroma_uses():
    assert DEFAULT_SPACE == "l2"


# =========================
# Filtering
# =========================

def _chain(scored, threshold, space=DEFAULT_SPACE, top_k=3):
    store = ScoredStore(scored, space=space)
    chat = FakeChat()
    return RAGChain(
        store, client=chat, top_k=top_k, relevance_threshold=threshold
    ), store, chat


SCORED = [
    (_doc("exact.txt"), 0.0),      # cos 1.00
    (_doc("close.txt"), 0.6),      # cos 0.70
    (_doc("unrelated.txt"), 2.0),  # cos 0.00
]


def test_threshold_zero_keeps_every_candidate():
    """The behaviour before this feature existed."""
    chain, _, chat = _chain(SCORED, threshold=0.0)

    result = chain.ask("q", user_id="u1")

    assert len(result["sources"]) == 3
    assert "unrelated.txt" in _context_of(chat)


def test_a_threshold_drops_the_unrelated_chunk():
    chain, _, chat = _chain(SCORED, threshold=0.5)

    result = chain.ask("q", user_id="u1")

    kept = [s["source"] for s in result["sources"]]
    assert kept == ["exact.txt", "close.txt"]
    assert "unrelated.txt" not in _context_of(chat)


def test_a_high_threshold_keeps_only_the_best():
    chain, _, chat = _chain(SCORED, threshold=0.9)

    assert [s["source"] for s in chain.ask("q", user_id="u1")["sources"]] == ["exact.txt"]


def test_the_boundary_is_inclusive():
    """cos exactly at the threshold is relevant enough."""
    chain, _, _ = _chain([(_doc("edge.txt"), 1.0)], threshold=0.5)  # cos 0.5

    assert len(chain.ask("q", user_id="u1")["sources"]) == 1


def test_when_everything_is_dropped_the_model_is_not_called():
    """The whole point: no context means no answer to invent one from."""
    chain, _, chat = _chain([(_doc("unrelated.txt"), 2.0)], threshold=0.5)

    result = chain.ask("q", user_id="u1")

    assert result["sources"] == []
    assert "No relevant information" in result["answer"]
    assert chat.calls == [], "the model was asked to answer from nothing"


def test_filtering_does_not_reorder_survivors():
    chain, _, _ = _chain(SCORED, threshold=0.0)

    sources = [s["source"] for s in chain.ask("q", user_id="u1")["sources"]]

    assert sources == ["exact.txt", "close.txt", "unrelated.txt"]


# =========================
# The tenant filter still reaches the store
# =========================

def test_the_scored_search_is_tenant_scoped():
    """EmbeddingsManager.similarity_search_with_score had no filter parameter
    at all; a caller would have searched across every tenant."""
    chain, store, _ = _chain(SCORED, threshold=0.0)

    chain.ask("q", user_id="alice")

    assert store.calls[0]["filter"] == {"user_id": "alice"}


def test_top_k_reaches_the_scored_search():
    chain, store, _ = _chain(SCORED, threshold=0.0, top_k=2)

    chain.ask("q", user_id="u1")

    assert store.calls[0]["k"] == 2


def test_manager_scored_search_accepts_a_filter(tmp_path):
    from app.rag.embeddings import EmbeddingsManager

    manager = EmbeddingsManager(persist_directory=str(tmp_path / "chroma"))
    manager.get_vectorstore("documents")
    manager.add_documents(
        [
            Document(page_content="alice private note", metadata={"user_id": "alice"}),
            Document(page_content="bob private note", metadata={"user_id": "bob"}),
        ],
        ids=["alice-1", "bob-1"],
    )

    results = manager.similarity_search_with_score(
        "note", k=5, filter={"user_id": "alice"}
    )

    owners = {doc.metadata["user_id"] for doc, _ in results}
    assert owners == {"alice"}, "the scored search crossed tenants"


# =========================
# Stores that cannot score
# =========================

def test_a_store_without_scores_still_works():
    docs = [_doc("a.txt")]
    chain = RAGChain(UnscoredStore(docs), client=FakeChat(), relevance_threshold=0.5)

    result = chain.ask("q", user_id="u1")

    assert [s["source"] for s in result["sources"]] == ["a.txt"]


def test_an_unknown_index_space_disables_filtering(caplog):
    """Better to include too much than to silently discard the best matches."""
    chain, _, chat = _chain(SCORED, threshold=0.9, space="hamming")

    with caplog.at_level("WARNING", logger="app.rag.chain"):
        result = chain.ask("q", user_id="u1")

    assert len(result["sources"]) == 3, "filtered against a scale that does not apply"
    assert any("Unknown index space" in r.getMessage() for r in caplog.records)
    assert "unrelated.txt" in _context_of(chat)


# =========================
# Observability
# =========================

def test_scores_are_logged_even_when_filtering_is_off(caplog):
    """The number has to come from data, so the data has to be visible."""
    chain, _, _ = _chain(SCORED, threshold=0.0)

    with caplog.at_level("INFO", logger="app.rag.chain"):
        chain.ask("q", user_id="u1")

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "similarity_best=1.000" in logged
    assert "similarity_worst=0.000" in logged
    assert "candidates=3" in logged


def test_dropped_chunks_are_reported(caplog):
    chain, _, _ = _chain(SCORED, threshold=0.5)

    with caplog.at_level("INFO", logger="app.rag.chain"):
        chain.ask("q", user_id="u1")

    assert any("dropped 1/3" in r.getMessage() for r in caplog.records)


# =========================
# Settings
# =========================

def test_the_threshold_defaults_to_disabled():
    """A guessed default would risk discarding relevant context, which is much
    harder to notice than including too much."""
    settings = Settings(_env_file=None, openai_api_key="k")

    assert settings.relevance_threshold == 0.0


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_the_threshold_must_be_a_similarity(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key="k", relevance_threshold=bad)


def test_the_threshold_reaches_the_chain(tmp_path, fake_openai_embeddings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import make_settings

    app = create_app(make_settings(tmp_path, relevance_threshold=0.42))

    with TestClient(app):
        assert app.state.rag_chain.relevance_threshold == 0.42
