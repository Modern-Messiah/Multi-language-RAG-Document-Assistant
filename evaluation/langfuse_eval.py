"""Run golden evaluation dataset against backend and record experiment in Langfuse.

This script links the local golden test suite (multilingual questions and corpus)
with Langfuse's "Datasets & Experiments" feature.

Usage:
    python -m evaluation.langfuse_eval --run-name baseline-v1
"""
import argparse
import datetime
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Optional

from dotenv import load_dotenv

from evaluation.golden import CORPUS, GOLDEN_CASES, UNANSWERABLE
from evaluation.metrics import aggregate, hit_at_k, precision_at_k, recall_at_k

logger = logging.getLogger("langfuse_eval")
DEFAULT_URL = "http://127.0.0.1:8000"
DATASET_NAME = "multilingual-rag-golden"


class BackendClient:
    """Simple HTTP client for backend interaction."""

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.timeout = timeout

    def _request(self, method: str, path: str, params=None, body=None, content_type=None, extra_headers=None):
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = dict(self.headers)
        if content_type:
            headers["Content-Type"] = content_type
        if extra_headers:
            headers.update(extra_headers)

        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read()
            resp_headers = dict(response.headers)
        return (json.loads(payload) if payload else {}), resp_headers

    def upload(self, user_id: str, filename: str, text: str) -> dict:
        boundary = f"----eval{uuid.uuid4().hex}"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: text/plain\r\n\r\n"
            f"{text}\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        data, _ = self._request(
            "POST",
            "/upload",
            {"user_id": user_id},
            body,
            f"multipart/form-data; boundary={boundary}",
        )
        return data

    def query(
        self,
        user_id: str,
        question: str,
        trace_id: str,
        language: str = "Auto",
        provider: str = None,
        model: str = None,
        model_key: str = None,
    ) -> dict:
        headers = {"X-Request-ID": trace_id}
        # Only attach BYOK headers when an explicit client key is provided
        if model_key:
            if provider:
                headers["X-Model-Provider"] = provider
            if model:
                headers["X-Model"] = model
            headers["X-Model-Key"] = model_key

        body = json.dumps(
            {"question": question, "language": language, "user_id": user_id}
        ).encode("utf-8")
        data, _ = self._request(
            "POST",
            "/query",
            None,
            body,
            "application/json",
            extra_headers=headers,
        )
        return data

    def clear(self, user_id: str) -> dict:
        data, _ = self._request("POST", "/clear", {"user_id": user_id})
        return data


def sync_dataset_items(langfuse_client, dataset_name: str = DATASET_NAME):
    """Ensure dataset exists and is populated."""
    try:
        dataset = langfuse_client.get_dataset(dataset_name)
    except Exception:
        dataset = langfuse_client.create_dataset(
            name=dataset_name,
            description=f"Evaluation dataset {dataset_name}",
        )

    if dataset_name != DATASET_NAME:
        return dataset

    existing_questions = {
        item.input.get("question")
        for item in (dataset.items or [])
        if isinstance(item.input, dict)
    }

    added = 0
    for case in GOLDEN_CASES:
        q = case["question"]
        if q not in existing_questions:
            langfuse_client.create_dataset_item(
                dataset_name=dataset_name,
                input={"question": q},
                expected_output={"expected_sources": case["expected"]},
                metadata={
                    "type": "answerable",
                    "note": case.get("note", ""),
                    "language": "ru" if any(c in q for c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя") else "en",
                },
            )
            existing_questions.add(q)
            added += 1

    for q in UNANSWERABLE:
        if q not in existing_questions:
            langfuse_client.create_dataset_item(
                dataset_name=dataset_name,
                input={"question": q},
                expected_output={"expected_sources": []},
                metadata={
                    "type": "unanswerable",
                    "note": "Negative case - should retrieve nothing or fall back",
                    "language": "ru" if any(c in q for c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя") else "en",
                },
            )
            existing_questions.add(q)
            added += 1

    if added > 0:
        langfuse_client.flush()
        print(f"Added {added} new item(s) to dataset '{dataset_name}'.")

    return langfuse_client.get_dataset(dataset_name)


def get_corpus_and_expectations(dataset_name: str, items: list):
    """Return {filename: content} and list of expected sources per item."""
    if dataset_name == "eval-financebench":
        corpus = {}
        item_expected = []
        for item in items:
            doc_name = (item.metadata or {}).get("doc_name") or "report"
            evidence_list = (item.expected_output or {}).get("evidence", [])
            doc_sources = []
            if not evidence_list:
                filename = f"{doc_name}.txt"
                corpus[filename] = f"Financial filing for {doc_name}."
                doc_sources.append(filename)
            else:
                for idx, ev in enumerate(evidence_list):
                    text = ev.get("evidence_text", "")
                    if text:
                        fname = f"{doc_name}_p{ev.get('evidence_page_num', idx)}.txt"
                        corpus[fname] = text
                        doc_sources.append(fname)
            item_expected.append(doc_sources)
        return corpus, item_expected
    elif dataset_name == "eval-sberquad-ru":
        corpus = {}
        item_expected = []
        for idx, item in enumerate(items):
            title = (item.metadata or {}).get("title") or f"doc_{idx}"
            context = (item.metadata or {}).get("context") or ""
            safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).strip()[:30]
            fname = f"doc_{idx}_{safe_title}.txt"
            corpus[fname] = context
            item_expected.append([fname])
        return corpus, item_expected
    else:
        item_expected = [
            item.expected_output.get("expected_sources", [])
            if isinstance(item.expected_output, dict) else []
            for item in items
        ]
        return CORPUS, item_expected


def run_experiment(
    backend: BackendClient,
    langfuse_client,
    dataset_name: str = DATASET_NAME,
    run_name: Optional[str] = None,
    top_k: int = 3,
    limit: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    model_key: Optional[str] = None,
    keep_tenant: bool = False,
):
    dataset = sync_dataset_items(langfuse_client, dataset_name)
    if not run_name:
        model_tag = model or "default"
        run_name = f"eval-{model_tag}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    items = dataset.items
    if limit:
        items = items[:limit]

    tenant_id = f"eval-{uuid.uuid4().hex[:12]}"
    print(f"\n🚀 Starting Langfuse Experiment Run: '{run_name}' on '{dataset_name}'")
    if provider:
        print(f"  Provider: {provider} | Model: {model or 'default'}")

    corpus, item_expectations = get_corpus_and_expectations(dataset_name, items)
    print(f"Uploading {len(corpus)} documents to tenant: {tenant_id}")
    for filename, text in corpus.items():
        res = backend.upload(tenant_id, filename, text)
        print(f"  ✓ {filename}: {res.get('chunks', 0)} chunk(s)")

    print(f"\nEvaluating {len(items)} dataset items...")
    eval_cases = []
    start_all = time.time()

    for idx, item in enumerate(items, start=1):
        q = item.input.get("question") if isinstance(item.input, dict) else str(item.input)
        expected_sources = item_expectations[idx - 1]
        is_negative = not bool(expected_sources)

        trace_id = uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        try:
            answer = backend.query(
                tenant_id,
                q,
                trace_id=trace_id,
                provider=provider,
                model=model,
                model_key=model_key,
            )
        except Exception as err:
            print(f"  [{idx}/{len(items)}] ERROR: {err}")
            continue
        latency_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_sources = [s["source"] for s in answer.get("sources", [])]
        eval_cases.append({"expected": expected_sources, "retrieved": retrieved_sources})

        # Calculate scores
        hit = hit_at_k(expected_sources, retrieved_sources, top_k) if not is_negative else (len(retrieved_sources) == 0)
        prec = precision_at_k(expected_sources, retrieved_sources, top_k) if not is_negative else 1.0
        rec = recall_at_k(expected_sources, retrieved_sources, top_k) if not is_negative else 1.0

        hit_label = "PASS" if hit else "FAIL"
        print(f"  [{idx}/{len(items)}] [{hit_label}] {q[:50]:52} -> {retrieved_sources} ({latency_ms:.0f}ms)")

        # Link trace to Langfuse dataset item and run
        item.link(
            trace_or_observation=None,
            trace_id=trace_id,
            run_name=run_name,
            run_description=f"Evaluation run {run_name} (model={model or 'default'})",
            run_metadata={"provider": provider or "default", "model": model or "default"},
        )

        # Log boolean passed status with colored True/False badge in Langfuse UI:
        langfuse_client.score(
            trace_id=trace_id,
            name="passed",
            value=1 if hit else 0,
            data_type="BOOLEAN",
            comment=f"expected: {expected_sources}, retrieved: {retrieved_sources}",
        )

        # Log numeric metrics for statistical averages:
        langfuse_client.score(
            trace_id=trace_id,
            name="retrieval_hit",
            value=1.0 if hit else 0.0,
            data_type="NUMERIC",
            comment=f"expected={expected_sources}, retrieved={retrieved_sources}",
        )
        if not is_negative:
            langfuse_client.score(
                trace_id=trace_id,
                name=f"precision@{top_k}",
                value=float(prec),
            )
            langfuse_client.score(
                trace_id=trace_id,
                name=f"recall@{top_k}",
                value=float(rec),
            )
        langfuse_client.score(
            trace_id=trace_id,
            name="latency_ms",
            value=float(latency_ms),
        )

    langfuse_client.flush()
    total_time = time.time() - start_all

    print("\n--- Aggregate Metrics ---")
    summary = aggregate(eval_cases, k=top_k)
    for k, v in summary.items():
        print(f"  {k:16}: {v:.3f}" if isinstance(v, float) else f"  {k:16}: {v}")
    print(f"  Total Duration  : {total_time:.2f}s")

    if not keep_tenant:
        backend.clear(tenant_id)
        print(f"\nCleared temporary tenant {tenant_id}.")
    else:
        print(f"\nKept tenant {tenant_id} as requested.")

    print(f"\n✅ Experiment '{run_name}' completed and synced to Langfuse!")
    print(f"Open Langfuse UI -> Datasets -> '{dataset_name}' -> Runs to view full results.")


def main(argv=None):
    load_dotenv()
    from langfuse import Langfuse

    parser = argparse.ArgumentParser(description=__doc__)
    default_key = os.getenv("BACKEND_API_KEY", "")
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend URL")
    parser.add_argument("--api-key", default=default_key, help="X-API-Key for backend (defaults to BACKEND_API_KEY from .env)")
    parser.add_argument("--dataset", default=DATASET_NAME, help="Dataset name in Langfuse")
    parser.add_argument("--run-name", default=None, help="Name of experiment run")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K for precision and recall")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dataset items to evaluate")
    parser.add_argument("--provider", default=None, help="Model provider (e.g. deepseek, openai, anthropic)")
    parser.add_argument("--model", default=None, help="Model name (e.g. deepseek-flash, gpt-4o)")
    parser.add_argument("--model-key", default=None, help="API key for the custom provider")
    parser.add_argument("--keep", action="store_true", help="Keep scratch tenant documents")
    parser.add_argument("--sync-only", action="store_true", help="Only sync dataset items without running")
    args = parser.parse_args(argv)

    client = Langfuse()
    if args.sync_only:
        sync_dataset_items(client, args.dataset)
        return 0

    backend = BackendClient(args.url, args.api_key)
    try:
        run_experiment(
            backend=backend,
            langfuse_client=client,
            dataset_name=args.dataset,
            run_name=args.run_name,
            top_k=args.top_k,
            limit=args.limit,
            provider=args.provider,
            model=args.model,
            model_key=args.model_key,
            keep_tenant=args.keep,
        )
    except urllib.error.URLError as err:
        print(f"Failed to connect to backend at {args.url}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
