"""Retrieval evaluation.

Every retrieval knob shipped so far - TOP_K_RESULTS, CHUNK_SIZE,
RELEVANCE_THRESHOLD - was chosen by reasoning about it rather than measuring
it. RELEVANCE_THRESHOLD in particular ships disabled precisely because there
was no way to pick a number honestly.

This package is not part of the running service; the image excludes it. It has
two halves, and they measure different things:

- `metrics` plus `tests/test_retrieval_quality.py`: an offline guard that runs
  in CI. It uses a deterministic embedding function, so it measures whether the
  **pipeline** still finds the right chunk - not how good real embeddings are.
  It catches "someone broke filtering, ordering, chunking or tenant scoping".
- `golden` plus `run_eval.py`: a small labelled set run by hand against a live
  backend with real embeddings. That is what a threshold should be tuned from.
"""
