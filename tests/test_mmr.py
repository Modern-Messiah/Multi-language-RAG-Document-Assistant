"""Diversity re-ranking.

Overlapping chunks make the neighbours of a strong match strong matches too, so
the top k can be one passage repeated. Measured on a nine-chunk corpus where one
document supplied six of them, a question spanning generation *and* storage
returned five chunks about generation and nothing about storage.

MMR trades a little relevance for coverage. It ships off, because the same
measurement shows the trade is not free - see the table in DOCUMENTATION.
"""
import math

import pytest
from langchain.schema import Document

from app.config import Settings
from app.rag.chain import MMR_FETCH_MULTIPLIER, RAGChain
from app.rag.embeddings import EmbeddingsManager, cosine_similarity, select_mmr

# =========================
# cosine_similarity
# =========================

def test_identical_vectors_are_one():
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_are_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_opposite_vectors_are_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_magnitude_does_not_matter():
    assert cosine_similarity([1.0, 0.0], [9.0, 0.0]) == pytest.approx(1.0)


@pytest.mark.parametrize("pair", [([0.0, 0.0], [1.0, 1.0]), ([1.0, 1.0], [0.0, 0.0])])
def test_a_zero_vector_has_no_direction(pair):
    """A chunk whose words were all outside the vocabulary; not an error."""
    assert cosine_similarity(*pair) == 0.0


# =========================
# select_mmr
# =========================

def _candidate(name, relevance, vector):
    return (name, relevance, vector)


# Three near-duplicates pointing one way, one different document.
CROWD = [
    _candidate("dup-1", 0.90, [1.0, 0.0]),
    _candidate("dup-2", 0.88, [0.99, 0.01]),
    _candidate("dup-3", 0.86, [0.98, 0.02]),
    _candidate("other", 0.50, [0.0, 1.0]),
]


def test_lambda_one_is_pure_relevance():
    """Which is exactly the behaviour before MMR existed."""
    assert select_mmr(CROWD, k=3, lambda_mult=1.0) == ["dup-1", "dup-2", "dup-3"]


def test_diversity_breaks_up_the_duplicates():
    selected = select_mmr(CROWD, k=3, lambda_mult=0.5)

    assert selected[0] == "dup-1", "the most relevant must still come first"
    assert "other" in selected, "the different document never got a slot"


def test_the_most_relevant_is_always_chosen_first():
    """Nothing displaces the best match, at any lambda."""
    for lambda_mult in (0.0, 0.3, 0.5, 0.9, 1.0):
        assert select_mmr(CROWD, k=2, lambda_mult=lambda_mult)[0] == "dup-1"


def test_lambda_zero_ignores_relevance_after_the_first_pick():
    selected = select_mmr(CROWD, k=2, lambda_mult=0.0)

    assert selected == ["dup-1", "other"]


def test_k_bounds_the_selection():
    assert len(select_mmr(CROWD, k=2, lambda_mult=0.5)) == 2


def test_asking_for_more_than_exists_returns_everything():
    selected = select_mmr(CROWD, k=99, lambda_mult=0.5)

    assert sorted(selected) == sorted(name for name, _, _ in CROWD)


@pytest.mark.parametrize("k", [0, -1])
def test_a_non_positive_k_selects_nothing(k):
    assert select_mmr(CROWD, k=k, lambda_mult=0.5) == []


def test_no_candidates_selects_nothing():
    assert select_mmr([], k=3, lambda_mult=0.5) == []


def test_nothing_is_selected_twice():
    selected = select_mmr(CROWD, k=4, lambda_mult=0.5)

    assert len(selected) == len(set(selected))


# =========================
# Wired into retrieval
# =========================

def _tokens(text):
    import re

    return re.findall(r"\w+", text.lower())


class BagOfWords:
    def __init__(self, vocabulary):
        self.vocabulary = sorted(vocabulary)
        self.index = {word: i for i, word in enumerate(self.vocabulary)}

    def _vector(self, text):
        vector = [0.0] * len(self.vocabulary)
        for word in _tokens(text):
            position = self.index.get(word)
            if position is not None:
                vector[position] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed_documents(self, texts, batch_size=100):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


# A faithful reproduction of the measured condition, not merely a similar one:
# one document whose chunks are near-duplicates of each other AND dense in the
# query's own words, so they outrank documents that mention those words once.
# The others each answer part of the question, and storage only appears in one.
CROWDED_CORPUS = (
    [(
        "solar.txt",
        f"Electricity generation electricity generation electricity generation. "
        f"Solar photovoltaic panels generate electricity, section {i}.",
    ) for i in range(6)]
    + [("wind.txt", "Offshore turbines and moving air, with rated capacity in megawatts.")]
    + [("battery.txt", "Storage in lithium cells, discharged after sunset each evening.")]
    + [("grid.txt", "Balancing supply against demand across interconnected regions.")]
)

SPANNING_QUESTION = "electricity generation and storage"


@pytest.fixture()
def crowded(tmp_path):
    vocabulary = set(_tokens(SPANNING_QUESTION))
    for _, text in CROWDED_CORPUS:
        vocabulary.update(_tokens(text))

    manager = EmbeddingsManager(
        persist_directory=str(tmp_path / "chroma"),
        embedding_fn=BagOfWords(vocabulary),
    )
    store = manager.get_vectorstore("documents")
    manager.add_documents(
        [
            Document(page_content=text, metadata={"source": source, "user_id": "u1"})
            for source, text in CROWDED_CORPUS
        ],
        ids=[f"u1-{i}" for i in range(len(CROWDED_CORPUS))],
    )
    return manager, store


def _sources(manager, store, lambda_mult, threshold=0.0, top_k=5):
    chain = RAGChain(
        store,
        client=object(),
        top_k=top_k,
        mmr_lambda=lambda_mult,
        relevance_threshold=threshold,
        embeddings_manager=manager,
    )
    docs = chain._retrieve(SPANNING_QUESTION, {"user_id": "u1"})
    return [d.metadata.get("source") for d in docs]


def test_without_mmr_one_document_crowds_out_the_rest(crowded):
    """The problem, reproduced. If this ever stops holding, MMR's rationale
    has changed and the default should be revisited."""
    manager, store = crowded

    sources = _sources(manager, store, lambda_mult=1.0)

    assert len(set(sources)) < 3, f"expected crowding, got {sources}"


def test_mmr_brings_in_the_other_documents(crowded):
    manager, store = crowded

    sources = _sources(manager, store, lambda_mult=0.5)

    assert len(set(sources)) > 2, f"still crowded: {sources}"


def test_mmr_finds_the_storage_document_the_question_needs(crowded):
    """The concrete failure: "generation and storage" returned no storage."""
    manager, store = crowded

    assert "battery.txt" not in _sources(manager, store, lambda_mult=1.0)
    assert "battery.txt" in _sources(manager, store, lambda_mult=0.5)


def test_the_best_match_still_ranks_first(crowded):
    manager, store = crowded

    assert _sources(manager, store, lambda_mult=0.5)[0] == "solar.txt"


def test_top_k_is_still_respected(crowded):
    manager, store = crowded

    assert len(_sources(manager, store, lambda_mult=0.5, top_k=3)) == 3


def test_the_threshold_is_applied_before_diversity(crowded):
    """Otherwise MMR spends a slot on a chunk that is merely different.

    The threshold is derived from the scores this embedder actually produces
    rather than hardcoded: a fixed number would be testing the bag of words,
    not the order of the two steps.
    """
    manager, store = crowded
    from app.rag.embeddings import distance_to_similarity

    scored = [
        distance_to_similarity(distance)
        for _, distance, _ in manager.search_candidates(SPANNING_QUESTION, 20, "u1")
    ]
    best, worst = max(scored), min(scored)
    assert best > worst, "every candidate scored the same, so this proves nothing"

    # High enough to admit only the strongest chunks, which all belong to the
    # crowded document; MMR then cannot reach past it for variety.
    cutoff = worst + (best - worst) * 0.95

    sources = _sources(manager, store, lambda_mult=0.5, threshold=cutoff)

    assert sources, "the threshold removed everything, so this proves nothing"
    assert set(sources) == {"solar.txt"}, sources


def test_retrieval_stays_within_the_tenant(crowded):
    manager, store = crowded
    manager.add_documents(
        [Document(
            page_content="Battery electricity storage lithium for another tenant",
            metadata={"source": "intruder.txt", "user_id": "u2"},
        )],
        ids=["u2-intruder"],
    )

    assert "intruder.txt" not in _sources(manager, store, lambda_mult=0.5)


def test_mmr_fetches_more_candidates_than_it_returns(crowded):
    """With no slack there is nothing to trade relevance against."""
    manager, store = crowded
    seen = {}
    original = manager.search_candidates

    def spy(query, k, owner):
        seen["k"] = k
        return original(query, k, owner)

    manager.search_candidates = spy
    _sources(manager, store, lambda_mult=0.5, top_k=2)

    assert seen["k"] == 2 * MMR_FETCH_MULTIPLIER


# =========================
# Falling back safely
# =========================

def test_without_a_manager_mmr_is_inactive_and_says_so(caplog):
    """A knob that looks set but is not is worse than one that is off."""
    class PlainStore:
        def similarity_search(self, query, k=4, filter=None):
            return [Document(page_content="x", metadata={"source": "a.txt"})]

    chain = RAGChain(PlainStore(), client=object(), mmr_lambda=0.5)

    with caplog.at_level("WARNING", logger="app.rag.chain"):
        docs = chain._retrieve("q", {"user_id": "u1"})

    assert [d.metadata["source"] for d in docs] == ["a.txt"]
    assert any("MMR_LAMBDA" in r.getMessage() for r in caplog.records)


def test_the_warning_is_not_repeated_per_query(caplog):
    class PlainStore:
        def similarity_search(self, query, k=4, filter=None):
            return []

    chain = RAGChain(PlainStore(), client=object(), mmr_lambda=0.5)

    with caplog.at_level("WARNING", logger="app.rag.chain"):
        for _ in range(5):
            chain._retrieve("q", {"user_id": "u1"})

    assert sum("MMR_LAMBDA" in r.getMessage() for r in caplog.records) == 1


def test_lambda_one_needs_no_manager(caplog):
    """The default configuration must not warn about anything."""
    class PlainStore:
        def similarity_search(self, query, k=4, filter=None):
            return []

    chain = RAGChain(PlainStore(), client=object(), mmr_lambda=1.0)

    with caplog.at_level("WARNING", logger="app.rag.chain"):
        chain._retrieve("q", {"user_id": "u1"})

    assert not any("MMR" in r.getMessage() for r in caplog.records)


# =========================
# Settings
# =========================

def test_mmr_ships_disabled():
    """Measured, not cautious: the recall gain arrives only after precision has
    already fallen from 0.60 to 0.35."""
    assert Settings(_env_file=None, openai_api_key="k").mmr_lambda == 1.0


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_lambda_must_be_a_fraction(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key="k", mmr_lambda=bad)


def test_the_setting_reaches_the_chain(tmp_path, fake_openai_embeddings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import make_settings

    app = create_app(make_settings(tmp_path, mmr_lambda=0.6))

    with TestClient(app):
        assert app.state.rag_chain.mmr_lambda == 0.6
        assert app.state.rag_chain.embeddings_manager is app.state.embeddings, (
            "MMR would be silently inactive without the manager"
        )
