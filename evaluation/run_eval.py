"""Measure retrieval against a live backend, with real embeddings.

This is the half that can answer "what should RELEVANCE_THRESHOLD be?". The
offline test suite cannot: its embedder is a bag of words, so it guards the
pipeline but knows nothing about meaning.

    python -m evaluation.run_eval --url http://127.0.0.1:8000 --api-key <key>

It uploads the golden corpus under a scratch tenant, asks every golden
question, reports recall/precision/MRR, and prints the score gap between
questions the corpus can answer and questions it cannot - which is the number a
threshold has to sit inside. It cleans up after itself unless told not to.

Nothing here runs in CI: it costs real embedding and completion calls.
"""
import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

from evaluation.golden import CORPUS, GOLDEN_CASES, UNANSWERABLE
from evaluation.metrics import aggregate

DEFAULT_URL = "http://127.0.0.1:8000"


class Backend:
    """The few calls this script needs, over stdlib only."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.timeout = timeout

    def _request(self, method, path, params=None, body=None, content_type=None):
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = dict(self.headers)
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read()
        return json.loads(payload) if payload else {}

    def upload(self, user_id, filename, text):
        boundary = f"----eval{uuid.uuid4().hex}"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: text/plain\r\n\r\n"
            f"{text}\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        return self._request(
            "POST", "/upload", {"user_id": user_id}, body,
            f"multipart/form-data; boundary={boundary}",
        )

    def query(self, user_id, question, language="Auto"):
        body = json.dumps(
            {"question": question, "language": language, "user_id": user_id}
        ).encode("utf-8")
        return self._request("POST", "/query", None, body, "application/json")

    def clear(self, user_id):
        return self._request("POST", "/clear", {"user_id": user_id})


def _sources_of(answer: dict):
    return [source["source"] for source in answer.get("sources", [])]


def run(backend: Backend, user_id: str, top_k: int, keep: bool):
    print(f"Uploading {len(CORPUS)} documents as tenant {user_id}")
    for filename, text in CORPUS.items():
        result = backend.upload(user_id, filename, text)
        print(f"  {filename}: {result['chunks']} chunk(s)")

    print("\nAnswerable questions")
    cases = []
    for case in GOLDEN_CASES:
        answer = backend.query(user_id, case["question"])
        retrieved = _sources_of(answer)
        cases.append({"expected": case["expected"], "retrieved": retrieved})
        hit = "ok " if set(case["expected"]) & set(retrieved) else "MISS"
        print(f"  [{hit}] {case['question'][:58]:60} -> {retrieved}")

    print("\nUnanswerable questions (these should retrieve nothing worth using)")
    for question in UNANSWERABLE:
        answer = backend.query(user_id, question)
        retrieved = _sources_of(answer)
        print(f"        {question[:58]:60} -> {retrieved}")

    results = aggregate(cases, k=top_k)
    print("\nMetrics")
    for name, value in results.items():
        print(f"  {name:16} {value:.3f}" if isinstance(value, float) else f"  {name:16} {value}")

    print(
        "\nTo choose RELEVANCE_THRESHOLD, read the backend's per-query log lines:\n"
        "  retrieval space=l2 candidates=5 similarity_best=... similarity_worst=...\n"
        "The answerable questions above give the values a real match reaches;\n"
        "the unanswerable ones give what noise reaches. Put the threshold between\n"
        "them, closer to the noise end - a threshold set too high silently drops\n"
        "relevant context, which is much harder to notice than including too much."
    )

    if keep:
        print(f"\nLeaving tenant {user_id} in place as requested.")
    else:
        backend.clear(user_id)
        print(f"\nCleared tenant {user_id}.")

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help=f"backend base URL (default {DEFAULT_URL})")
    parser.add_argument("--api-key", default="", help="value for the X-API-Key header")
    parser.add_argument("--top-k", type=int, default=3, help="k for recall@k and precision@k")
    parser.add_argument(
        "--user-id",
        default=None,
        help="tenant to use; a scratch one is generated and cleared by default",
    )
    parser.add_argument("--keep", action="store_true", help="do not clear the tenant afterwards")
    args = parser.parse_args(argv)

    user_id = args.user_id or f"eval-{uuid.uuid4().hex[:12]}"
    backend = Backend(args.url, args.api_key)

    try:
        run(backend, user_id, args.top_k, args.keep or bool(args.user_id))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        print(f"\nBackend returned HTTP {error.code}: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as error:
        print(f"\nCould not reach {args.url}: {error.reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
