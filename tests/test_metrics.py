"""The retrieval metrics themselves.

A harness whose arithmetic is wrong is worse than no harness: it reports a
number that looks like evidence. These are all hand-checkable.
"""
import pytest

from evaluation.metrics import (
    aggregate,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# =========================
# recall@k
# =========================

def test_recall_is_one_when_everything_wanted_is_in_the_top_k():
    assert recall_at_k(["a", "b"], ["a", "b", "c"], k=3) == 1.0


def test_recall_is_the_share_found():
    assert recall_at_k(["a", "b"], ["a", "z"], k=2) == 0.5


def test_recall_is_zero_when_nothing_wanted_is_returned():
    assert recall_at_k(["a"], ["x", "y"], k=2) == 0.0


def test_recall_respects_k():
    """The wanted document is returned, but below the cutoff."""
    assert recall_at_k(["a"], ["x", "y", "a"], k=2) == 0.0
    assert recall_at_k(["a"], ["x", "y", "a"], k=3) == 1.0


def test_recall_with_no_expectations_is_one():
    """Nothing was wanted, so nothing was missed."""
    assert recall_at_k([], ["x"], k=3) == 1.0


def test_recall_ignores_duplicate_expectations():
    assert recall_at_k(["a", "a"], ["a"], k=1) == 1.0


# =========================
# precision@k
# =========================

def test_precision_is_the_share_of_the_top_k_that_was_wanted():
    assert precision_at_k(["a"], ["a", "x", "y"], k=3) == pytest.approx(1 / 3)


def test_precision_is_one_when_everything_returned_was_wanted():
    assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0


def test_precision_of_an_empty_result_is_zero():
    assert precision_at_k(["a"], [], k=3) == 0.0


def test_precision_divides_by_what_was_returned_not_by_k():
    """Two results at k=5 is 50 percent precise, not 20."""
    assert precision_at_k(["a"], ["a", "x"], k=5) == 0.5


# =========================
# reciprocal rank
# =========================

@pytest.mark.parametrize(
    "retrieved,expected_rr",
    [(["a", "x", "y"], 1.0), (["x", "a", "y"], 0.5), (["x", "y", "a"], pytest.approx(1 / 3))],
)
def test_reciprocal_rank_follows_the_position(retrieved, expected_rr):
    assert reciprocal_rank(["a"], retrieved) == expected_rr


def test_reciprocal_rank_is_zero_when_nothing_wanted_appears():
    assert reciprocal_rank(["a"], ["x", "y"]) == 0.0


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank(["a", "b"], ["x", "b", "a"]) == 0.5


def test_reciprocal_rank_of_an_empty_result_is_zero():
    assert reciprocal_rank(["a"], []) == 0.0


# =========================
# hit@k
# =========================

def test_hit_is_true_when_anything_wanted_is_in_range():
    assert hit_at_k(["a", "b"], ["x", "b"], k=2) is True


def test_hit_is_false_beyond_k():
    assert hit_at_k(["a"], ["x", "y", "a"], k=2) is False


# =========================
# Degenerate k
# =========================

@pytest.mark.parametrize("k", [0, -1])
def test_a_non_positive_k_returns_nothing(k):
    assert recall_at_k(["a"], ["a"], k=k) == 0.0
    assert precision_at_k(["a"], ["a"], k=k) == 0.0
    assert hit_at_k(["a"], ["a"], k=k) is False


# =========================
# aggregate
# =========================

def test_aggregate_averages_across_cases():
    cases = [
        {"expected": ["a"], "retrieved": ["a", "x"]},   # recall 1, rr 1
        {"expected": ["b"], "retrieved": ["x", "b"]},   # recall 1, rr 0.5
    ]

    results = aggregate(cases, k=2)

    assert results["cases"] == 2
    assert results["recall@2"] == 1.0
    assert results["mrr"] == 0.75


def test_aggregate_names_the_metrics_after_k():
    results = aggregate([{"expected": ["a"], "retrieved": ["a"]}], k=5)

    assert "recall@5" in results and "precision@5" in results and "hit_rate@5" in results


def test_aggregate_of_nothing_is_zeros_not_an_error():
    """A report of nothing is still a report."""
    results = aggregate([], k=3)

    assert results["cases"] == 0
    assert results["mrr"] == 0.0
    assert results["recall@3"] == 0.0


def test_aggregate_hit_rate_counts_cases_not_documents():
    cases = [
        {"expected": ["a", "b"], "retrieved": ["a"]},  # partial hit, still a hit
        {"expected": ["c"], "retrieved": ["x"]},       # miss
    ]

    assert aggregate(cases, k=3)["hit_rate@3"] == 0.5


def test_a_worse_ranking_scores_worse():
    """The property every metric here exists for."""
    good = aggregate([{"expected": ["a"], "retrieved": ["a", "x", "y"]}], k=3)
    bad = aggregate([{"expected": ["a"], "retrieved": ["x", "y", "a"]}], k=3)

    assert good["mrr"] > bad["mrr"]
    assert good["recall@3"] == bad["recall@3"], "recall ignores rank, by definition"
