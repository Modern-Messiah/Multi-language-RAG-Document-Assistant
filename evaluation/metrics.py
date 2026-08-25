"""Retrieval metrics, as plain functions over lists of source names.

Deliberately not ragas or deepeval: those bring an LLM judge, a dependency
tree and a bill, to answer a question this project can answer with arithmetic.
What is measured here is whether retrieval put the right document in front of
the model, which is exactly the part that breaks silently.

Everything takes `expected` (the sources that should have been found) and
`retrieved` (what retrieval actually returned, best first) and is order-aware
where that matters.
"""
from typing import Dict, Iterable, List, Sequence

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "hit_at_k",
    "aggregate",
]


def _top(retrieved: Sequence[str], k: int) -> List[str]:
    if k <= 0:
        return []
    return list(retrieved)[:k]


def recall_at_k(expected: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Share of the expected sources that appear in the top k.

    1.0 means everything that should have been found was. Undefined with no
    expectations, which is treated as 1.0 - nothing was missed.
    """
    wanted = set(expected)
    if not wanted:
        return 1.0
    found = wanted & set(_top(retrieved, k))
    return len(found) / len(wanted)


def precision_at_k(expected: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Share of the top k that was actually wanted.

    Low precision is what fills a prompt with noise; it is the number a
    relevance threshold is supposed to improve.
    """
    top = _top(retrieved, k)
    if not top:
        return 0.0
    wanted = set(expected)
    return sum(1 for source in top if source in wanted) / len(top)


def reciprocal_rank(expected: Iterable[str], retrieved: Sequence[str]) -> float:
    """1/rank of the first wanted source, or 0.0 if none was returned.

    Rank matters because the model reads the context top down and the weakest
    chunk is the one most likely to be dropped by a threshold.
    """
    wanted = set(expected)
    for position, source in enumerate(retrieved, start=1):
        if source in wanted:
            return 1.0 / position
    return 0.0


def hit_at_k(expected: Iterable[str], retrieved: Sequence[str], k: int) -> bool:
    """Whether anything wanted made it into the top k at all."""
    return bool(set(expected) & set(_top(retrieved, k)))


def aggregate(cases: Iterable[Dict], k: int) -> Dict[str, float]:
    """Mean metrics over cases of {"expected": [...], "retrieved": [...]}.

    Returns recall@k, precision@k, MRR, hit-rate@k and the case count. An empty
    set returns zeros rather than raising: a report of nothing is still a
    report.
    """
    cases = list(cases)
    if not cases:
        return {
            "cases": 0,
            f"recall@{k}": 0.0,
            f"precision@{k}": 0.0,
            "mrr": 0.0,
            f"hit_rate@{k}": 0.0,
        }

    total = len(cases)
    return {
        "cases": total,
        f"recall@{k}": sum(
            recall_at_k(c["expected"], c["retrieved"], k) for c in cases
        ) / total,
        f"precision@{k}": sum(
            precision_at_k(c["expected"], c["retrieved"], k) for c in cases
        ) / total,
        "mrr": sum(
            reciprocal_rank(c["expected"], c["retrieved"]) for c in cases
        ) / total,
        f"hit_rate@{k}": sum(
            1 for c in cases if hit_at_k(c["expected"], c["retrieved"], k)
        ) / total,
    }
