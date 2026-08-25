"""Does retrieval still find the right document?

This is a guard on the **pipeline**, not on embedding quality. The embedding
function here is a deterministic bag of words, so it says nothing about how
good text-embedding-3-small is - and it is not trying to. What it catches is a
change that breaks filtering, ordering, chunking or tenant scoping badly enough
that the right chunk stops coming back, which the rest of the suite cannot see
because its fake vectors are random and every ranking is therefore arbitrary.

For the real thing - picking RELEVANCE_THRESHOLD, comparing TOP_K_RESULTS - run
evaluation/run_eval.py against a live backend.
"""
import math
import re

import pytest
from langchain.schema import Document

from app.rag.chain import RAGChain
from app.rag.embeddings import EmbeddingsManager
from evaluation.golden import CORPUS, GOLDEN_CASES, UNANSWERABLE
from evaluation.metrics import aggregate

TOP_K = 3

# Floors, not targets. They are set below what the pipeline currently achieves
# so that ordinary tuning does not trip them, but far enough above chance that
# a real regression does.
MIN_RECALL = 0.85
MIN_MRR = 0.85
MIN_HIT_RATE = 0.85


def _tokens(text: str):
    """Words, lowercased. Handles Cyrillic as well as Latin."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BagOfWordsEmbedding:
    """A deterministic embedder where overlap in words means closeness.

    Real embeddings understand paraphrase; this does not, which is why the
    golden questions reuse vocabulary from the documents. What it does give is
    a stable, offline, meaningful ranking - random vectors would make every
    metric here noise.
    """

    def __init__(self, vocabulary):
        self.vocabulary = sorted(vocabulary)
        self.index = {word: i for i, word in enumerate(self.vocabulary)}

    def _vector(self, text):
        vector = [0.0] * len(self.vocabulary)
        for word in _tokens(text):
            position = self.index.get(word)
            if position is not None:
                vector[position] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            # Orthogonal to everything, so it simply ranks last.
            return [0.0] * len(self.vocabulary)
        return [value / norm for value in vector]

    def embed_documents(self, texts, batch_size=100):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture()
def indexed(tmp_path):
    """The golden corpus in a real ChromaDB collection."""
    vocabulary = set()
    for text in CORPUS.values():
        vocabulary.update(_tokens(text))
    for case in GOLDEN_CASES:
        vocabulary.update(_tokens(case["question"]))

    manager = EmbeddingsManager(
        persist_directory=str(tmp_path / "chroma"),
        embedding_fn=BagOfWordsEmbedding(vocabulary),
    )
    store = manager.get_vectorstore("documents")
    manager.add_documents(
        [
            Document(page_content=text, metadata={"source": name, "user_id": "u1"})
            for name, text in CORPUS.items()
        ],
        ids=[f"u1-{name}" for name in CORPUS],
    )
    return manager, store


def _retrieved_sources(store, question, top_k=TOP_K, threshold=0.0):
    """What the real retrieval path returns, best first."""
    chain = RAGChain(
        store, client=object(), top_k=top_k, relevance_threshold=threshold
    )
    docs = chain._retrieve(question, {"user_id": "u1"})
    return [doc.metadata.get("source", "unknown") for doc in docs]


def _run_golden(store, top_k=TOP_K, threshold=0.0):
    return [
        {
            "expected": case["expected"],
            "retrieved": _retrieved_sources(store, case["question"], top_k, threshold),
        }
        for case in GOLDEN_CASES
    ]


# =========================
# The corpus is usable
# =========================

def test_the_golden_set_references_only_real_documents():
    """A typo in an expected filename would make a case unpassable forever."""
    for case in GOLDEN_CASES:
        for source in case["expected"]:
            assert source in CORPUS, f"{source} is not in the corpus"


def test_every_case_expects_something():
    for case in GOLDEN_CASES:
        assert case["expected"], f"{case['question']!r} expects nothing"


def test_the_corpus_covers_more_than_one_language():
    """The product's whole point is multilingual, so the set must be too."""
    has_cyrillic = any(
        re.search(r"[а-яё]", text, flags=re.IGNORECASE) for text in CORPUS.values()
    )
    assert has_cyrillic


# =========================
# The pipeline finds the right document
# =========================

def test_retrieval_meets_the_quality_floor(indexed):
    _, store = indexed

    results = aggregate(_run_golden(store), k=TOP_K)

    assert results[f"recall@{TOP_K}"] >= MIN_RECALL, results
    assert results["mrr"] >= MIN_MRR, results
    assert results[f"hit_rate@{TOP_K}"] >= MIN_HIT_RATE, results


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c["note"])
def test_each_golden_question_finds_its_document(indexed, case):
    """Named per case, so a failure says which property broke."""
    _, store = indexed

    retrieved = _retrieved_sources(store, case["question"])

    assert set(case["expected"]) & set(retrieved), (
        f"{case['question']!r} returned {retrieved}, expected one of {case['expected']}"
    )


def test_the_best_match_is_ranked_first(indexed):
    """Rank matters: the model reads context top down, and a threshold cuts
    from the bottom."""
    _, store = indexed

    top_hits = 0
    for case in GOLDEN_CASES:
        retrieved = _retrieved_sources(store, case["question"])
        if retrieved and retrieved[0] in case["expected"]:
            top_hits += 1

    assert top_hits >= len(GOLDEN_CASES) - 1, (
        f"only {top_hits}/{len(GOLDEN_CASES)} questions ranked the right document first"
    )


def test_a_question_about_storage_does_not_prefer_the_solar_document(indexed):
    """The case that distinguishes real retrieval from keyword overlap: the
    question mentions solar, but it is about storage."""
    _, store = indexed

    retrieved = _retrieved_sources(store, "storage smooths solar generation against evening demand")

    assert retrieved[0] == "battery.txt", f"ranked {retrieved} - solar won on a shared word"


# =========================
# Tenant scoping survives
# =========================

def test_retrieval_stays_within_the_tenant(indexed):
    manager, store = indexed
    manager.add_documents(
        [Document(
            page_content="Solar panels photovoltaic efficiency for another tenant",
            metadata={"source": "intruder.txt", "user_id": "u2"},
        )],
        ids=["u2-intruder"],
    )

    retrieved = _retrieved_sources(store, "How efficient are photovoltaic modules?")

    assert "intruder.txt" not in retrieved


# =========================
# The knobs behave as advertised
# =========================

def test_top_k_bounds_what_comes_back(indexed):
    _, store = indexed

    assert len(_retrieved_sources(store, "solar photovoltaic", top_k=1)) == 1
    assert len(_retrieved_sources(store, "solar photovoltaic", top_k=3)) <= 3


def test_a_threshold_improves_precision(indexed):
    """The reason RELEVANCE_THRESHOLD exists: fewer, better chunks."""
    _, store = indexed

    unfiltered = aggregate(_run_golden(store, threshold=0.0), k=TOP_K)
    filtered = aggregate(_run_golden(store, threshold=0.25), k=TOP_K)

    assert filtered[f"precision@{TOP_K}"] > unfiltered[f"precision@{TOP_K}"], (
        f"filtering did not help precision: {unfiltered} -> {filtered}"
    )


def test_raising_the_threshold_never_improves_recall(indexed):
    """Filtering can only remove candidates, so recall is monotone downward.

    No absolute number is asserted, and deliberately so. Whether a given
    threshold keeps the right chunks is a property of the *embedding space*,
    and the bag of words used here is not that space - an early version of this
    test demanded recall at 0.25 and failed for reasons that said nothing about
    the pipeline. Picking a real value is what evaluation/run_eval.py is for.
    """
    _, store = indexed

    recalls = [
        aggregate(_run_golden(store, threshold=t), k=TOP_K)[f"recall@{TOP_K}"]
        for t in (0.0, 0.2, 0.5, 0.9)
    ]

    assert recalls == sorted(recalls, reverse=True), recalls
    assert recalls[0] >= MIN_RECALL, "unfiltered retrieval is below the floor"


def test_the_shipped_threshold_is_still_disabled():
    """Nothing here justifies a non-zero default; run_eval.py is what would."""
    from app.config import Settings

    assert Settings(_env_file=None, openai_api_key="k").relevance_threshold == 0.0


def test_unanswerable_questions_exist_for_the_manual_run():
    """They are not asserted on here.

    Separating answerable from unanswerable questions by score is exactly the
    semantic judgement a bag-of-words embedder cannot make - its vectors are
    dominated by shared stopwords. The set is defined so run_eval.py can report
    that separation against real embeddings.
    """
    assert UNANSWERABLE
    assert not any(q in {c["question"] for c in GOLDEN_CASES} for q in UNANSWERABLE)
