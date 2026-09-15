"""Benchmark evaluation specifically targeting previously failed cases from SberQuAD.

Extracts hard cases where baseline dense retrieval failed (Hit Rate = 0.0%)
and evaluates them against the new Hybrid (Dense + BM25 + RRF) search pipeline
and multilingual Cross-Encoder reranker.
"""

import argparse
import os
import re
import time
from typing import List, Optional

import requests
from dotenv import load_dotenv

from evaluation.langfuse_eval import (
    get_corpus_and_expectations,
    hit_at_k,
)


def extract_failed_indices(log_path: str, max_items: int = 5166) -> List[int]:
    """Parse log file to find indices of items that previously had [FAIL]."""
    if not os.path.exists(log_path):
        return []
    fail_pattern = re.compile(r"^\s*\[(\d+)/" + str(max_items) + r"\]\s*\[FAIL\]")
    failed_indices = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = fail_pattern.match(line)
            if m:
                failed_indices.append(int(m.group(1)))
    return failed_indices


def run_hard_cases_benchmark(
    backend_url: str = "http://127.0.0.1:8000",
    tenant_id: str = "eval-528d05007556",
    limit: int = 50,
    top_k: int = 5,
    reranker: Optional[str] = None,
    reranker_model: Optional[str] = None,
    log_file: Optional[str] = None,
):
    load_dotenv()
    from langfuse import Langfuse

    langfuse_client = Langfuse()
    dataset = langfuse_client.get_dataset("eval-sberquad-ru")
    items = dataset.items

    # Find failure indices from log or dataset
    failed_indices = []
    if log_file and os.path.exists(log_file):
        failed_indices = extract_failed_indices(log_file, len(items))

    if not failed_indices:
        # Default known hard failures if no log path provided
        failed_indices = [6, 20, 28, 30, 44, 47, 49, 52, 54, 61, 62, 67, 72, 77, 91, 103, 112, 118, 125, 134]

    selected_indices = failed_indices[:limit]
    corpus, item_expectations = get_corpus_and_expectations("eval-sberquad-ru", items)

    backend_key = os.getenv("BACKEND_API_KEY", "")
    headers = {"X-API-Key": backend_key} if backend_key else {}
    if reranker and reranker != "none":
        headers["X-Reranker-Provider"] = reranker
        if reranker_model:
            headers["X-Reranker-Model"] = reranker_model

    print("\n=======================================================", flush=True)
    print("🚀 SberQuAD Hard Failures Recovery Benchmark", flush=True)
    print(f"  Target Cases    : {len(selected_indices)} (previously 100% failed in baseline)", flush=True)
    print(f"  Tenant ID       : {tenant_id}", flush=True)
    print("  Pipeline        : Hybrid Search (ChromaDB + Okapi BM25 + RRF)", flush=True)
    print(f"  Reranker        : {reranker or 'None'} (model={reranker_model or 'default'})", flush=True)
    print("=======================================================\n", flush=True)

    recovered_at_3 = 0
    recovered_at_5 = 0
    latencies = []

    for i, idx in enumerate(selected_indices, start=1):
        item = items[idx - 1]
        expected_sources = item_expectations[idx - 1]
        question = item.input.get("question")

        t0 = time.perf_counter()
        try:
            payload = {
                "question": question,
                "language": "ru",
                "user_id": tenant_id,
            }
            resp = requests.post(f"{backend_url}/query", json=payload, headers=headers, timeout=30.0)
            data = resp.json()
        except Exception as err:
            print(f"  [{i:2d}/{len(selected_indices)}] ERROR: {err}", flush=True)
            continue

        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        retrieved_sources = [s["source"] for s in data.get("sources", [])]

        hit3 = hit_at_k(expected_sources, retrieved_sources, 3)
        hit5 = hit_at_k(expected_sources, retrieved_sources, 5)

        if hit3:
            recovered_at_3 += 1
        if hit5:
            recovered_at_5 += 1

        status = "RECOVERED" if hit5 else "FAIL"
        rank_str = "N/A"
        for r_idx, src in enumerate(retrieved_sources, start=1):
            if src in expected_sources:
                rank_str = f"Rank {r_idx}"
                break

        print(
            f"  [{i:2d}/{len(selected_indices)}] [{status:9s}] {question[:45]:47} "
            f"-> {rank_str:7s} ({latency_ms:.0f}ms)",
            flush=True,
        )

    print("\n=======================================================", flush=True)
    print("🎯 Benchmark Summary on Hard Failure Subset:", flush=True)
    print(f"  Cases Evaluated    : {len(selected_indices)}", flush=True)
    print("  Baseline Hit Rate  : 0.0% (all were failures)", flush=True)
    print(f"  New Hit Rate@3     : {recovered_at_3}/{len(selected_indices)} ({recovered_at_3/len(selected_indices)*100:.1f}%)", flush=True)
    print(f"  New Hit Rate@5     : {recovered_at_5}/{len(selected_indices)} ({recovered_at_5/len(selected_indices)*100:.1f}%)", flush=True)
    if latencies:
        print(f"  Average Latency    : {sum(latencies)/len(latencies):.0f}ms", flush=True)
    print("=======================================================\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate recovery on previously failed SberQuAD cases.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Backend URL")
    parser.add_argument("--tenant", default="eval-528d05007556", help="Tenant with pre-indexed corpus")
    parser.add_argument("--limit", type=int, default=50, help="Number of failed cases to evaluate")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K evaluation")
    parser.add_argument("--reranker", default=None, choices=[None, "none", "flashrank", "cohere"], help="Reranker provider")
    parser.add_argument("--reranker-model", default=None, help="Reranker model override")
    parser.add_argument("--log-file", default=None, help="Path to previous evaluation log file to extract fails")
    args = parser.parse_args()

    run_hard_cases_benchmark(
        backend_url=args.url,
        tenant_id=args.tenant,
        limit=args.limit,
        top_k=args.top_k,
        reranker=args.reranker,
        reranker_model=args.reranker_model,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
