# Recurring bug classes

The defect classes this repository's history repeats, each with the commits that fixed an instance,
the instances still open (by KNOWN_ISSUES.md section), what to look for in a diff and the test that
catches it. Read a commit's full message with `git show <hash>` (`git.exe show <hash>` from WSL with
Windows Git): the Stage-era messages record what was verified and how.

Contents:

1. Tenant scope missing on one path
2. A side index skipped on one mutation path
3. Blocking work on the event loop, and races
4. Wrong status class, or error text that says too much
5. Streaming failures after the 200
6. Settings that never reach the code, and environment leaking into tests
7. Docs drifting from the code
8. Dependencies and lock files
9. Library behaviour assumed instead of checked
10. Client limits and duplicated client logic
11. Tests passing for the wrong reason
12. Heuristics and defaults that silently narrow retrieval
13. Lint and import errors committed
14. Runtime data directories, volumes and secrets
15. Untested evaluation scripts
16. Open known issues by file

## 1. Tenant scope missing on one path

The class found most often. One collection and one BM25 index hold every tenant, so one unscoped call
is a cross-tenant read or delete.

- c05358d: dedup by content hash was not scoped per owner, and Chroma ids built from a lossy sanitised
  user_id collided, so tenants overwrote each other's vectors (Chroma upserts on an existing id).
- 137b46a: `RAGChain.ask` with a falsy user_id searched every tenant; it raises now.
- fc0c656: document lookups and deletes scoped through one `EmbeddingsManager._owned` helper.
- 94d5d79: `EmbeddingsManager.similarity_search_with_score` had no filter parameter at all.
- 4c37696: MMR's candidate query had to carry the tenant clause (a mutation test pins it).
- 59bb70b: the dedup short-circuit answered 200 outside the owner lock, so a sweep could delete the
  document it had just confirmed.

Still open (Retrieval and search): BM25 searches every tenant when no owner is passed (latent); `$and`
filters skip BM25; an `_owned`-shaped filter crashes the chain and the hybrid retriever; BM25
statistics are global, so one tenant's uploads reorder another's keyword results.

Look for:

- a Chroma `get`, `query`, `delete` or similarity call without an owner clause, or with a filter built
  other than by `_owned` (EmbeddingsManager internals) or `build_chroma_filter` (chain and retriever);
- a `BM25Index.search` or `HybridRetriever.retrieve` call whose owner can be `None` or `""`;
- a lookup keyed by content hash or filename without the owner;
- a new parameter naming an owner without the shared user_id validation.

Catch it with an API test where two owners hold the same file: one owner's action leaves the other's
chunks, sources and counts untouched.

## 2. A side index skipped on one mutation path

- 53aa168 put the in-memory BM25 index on `app.state` and wired it into upload, revision retirement,
  delete and `/clear` in `app/main.py`, but not into `app/sweep.py`, which removes namespaces through
  `storage.wipe_namespace`. Its lazy re-sync from Chroma (`HybridRetriever.retrieve`) runs only for an
  owner with zero BM25 entries.

Still open: swept documents stay searchable until the backend restarts (Operations: backup, restore and
sweep); older documents drop out of keyword search after a restart if the user uploads first, and the
keyword index is filled even with hybrid search off (Retrieval and search).

Look for every place that changes Chroma or the upload files (`add_documents`, `delete_by_file_hash`,
`delete_documents`, `wipe_namespace`, `unlink`, `rmtree`) and check the index changes there too, then
restore and startup.

Catch it with an API test asserting `app_state.bm25_index.count_owner(owner)` after each mutation (the
`api` fixture exposes `app_state`); no API test asserts the index after any mutation path.

## 3. Blocking work on the event loop, and races

- 2f3a3b6: `upload_document`, `query_rag` and `clear_documents` were `async def` around synchronous
  OpenAI and Chroma calls; one slow upload froze every request, `/health` included, and the Compose
  healthcheck restarted a backend that was merely busy.
- 137b46a: the bot handled one update at a time, so one slow upload blocked every user (it runs with
  `concurrent_updates(True)` now).
- 59bb70b: the quota check-then-write raced between two uploads of one owner (hence the striped owner
  locks); the activity `touch()` did `exists()` then `utime()` while `forget()` ran unlocked; listing
  then stat-ing files crashed during a concurrent delete (`storage.size_of` returns 0 for a vanished file).

Look for `async def` on a route, a check-then-act outside `for_owner`, nested lock acquisition, and a
stat after a listing that does not handle a vanished file.

Catch it with an `inspect.iscoroutinefunction` pin per route and a `threading.Barrier`-forced
interleaving (`tests/test_quotas.py`).

## 4. Wrong status class, or error text that says too much

- 137b46a, 7fb55cd: a corrupt PDF and unguarded store or disk calls ended as bare 500s.
- 2f3a3b6: rate limits and timeouts looked like 503 until they became 429 with Retry-After and 504.
- 771f1a3: an unhandled exception gave a plain-text 500 without a request id; it is JSON with
  `request_id` now (`unhandled_error`).
- ad141a0: a provider refusing the caller's own key must reach the caller as a 400, not the clients'
  "contact the operator" path.
- 1abb51b: Gemini answers a bad key with 400, which fell through to a 503 "Query failed" blaming the
  operator; any provider status now maps to caller wording, DeepSeek's 402 (out of balance) by name.
- 82d046f, f4e7947, 3f8f41c: model alias normalisation, detecting DeepSeek from LLM_BASE_URL, and an
  evaluation runner that sent `X-Model` and `X-Model-Provider` without `X-Model-Key` (a 400).

Still open: reranker headers bypass RERANKER_ENABLED and can spend the operator's Cohere key, and a
streamed answer that times out leaves the caller's client to garbage collection (Models, keys and
reranking); a malformed `metadata_filter` is a 503 (Retrieval and search).

Look for `except Exception` mapped to 503 for something the caller caused, a 401 or 403 for a
caller-fixable problem, a `detail` built from `str(exc)`, a 4xx for an outage, and a new error branch
without `byok.close_quietly`.

Catch it with tests that raise each upstream error class through a fake and assert status and detail,
as `tests/test_resilience.py` and `tests/test_byok.py` do.

## 5. Streaming failures after the 200

- e031ab2: citation markers split across chunks slipped past a per-chunk regex (hence
  `CitationStripper`), and failures before the first byte needed generator priming to keep their codes.
- ad141a0: a refused key showed up mid-stream until `ask_stream` opened the completion before yielding
  the sources event; a provider stand-in that answered `stream=True` with plain JSON made a streamed
  probe with no tokens look like a pass.
- 771f1a3: the request middleware had to be raw ASGI to suit streaming.

Still open: streamed answers record no token usage (Tracing and observability).

Look for a `yield` moved above `create(..., stream=True)`, work in `events()` that can fail before the
first byte, a fake that ignores `stream=True`, and text processing applied per chunk.

Catch it with `tests/test_streaming.py`-style tests: the status code of a priming failure, and the
event sequence of a stream.

## 6. Settings that never reach the code, and environment leaking into tests

- 137b46a: `RAGChain` read OPENAI_API_KEY from `os.environ`, which pydantic-settings never fills from
  `.env`, so the documented `.env` flow crashed at startup; invalid combinations were accepted until
  validators were added.
- 2f3a3b6: OPENAI_BASE_URL needed an explicit Settings field for the same reason.
- 7fb55cd: `Settings(_env_file=None)` suppresses only `.env`; exported variables still changed test
  results, so `tests/conftest.py` strips them for the session.
- 82d046f: the same leak reappeared in `tests/test_docs_consistency.py`, whose `_settings_defaults`
  strips the variables too.

Still open: a root `.env` changes test results (Development and tests); host-run backup and restore
ignore `.env` (Operations: backup, restore and sweep).

Look for `os.getenv` or `os.environ` in `app/`, a new helper that builds `Settings` without stripping
the environment, and a value read in a client that the backend also needs.

## 7. Docs drifting from the code

- bb6f00c pinned the contract with `tests/test_docs_consistency.py`; later commits extended it: 9200bd0
  (the `/help` parser), 3b8df92 (every `python -m` command resolves, every package is linted), 59bb70b
  (a heading per route), aa63f87 (digits in variable names).
- 53aa168 added four HYBRID_* settings to `app/config.py` only; efff62e added the DOCUMENTATION.md rows
  and aa63f87 the `.env.template` lines.

Look for a setting, route, command, header, request field or behaviour change without the matching
README.md, DOCUMENTATION.md, CLAUDE.md, skill and KNOWN_ISSUES.md updates. Request fields and headers
are checked by no test.

Catch it with `pytest tests/test_docs_consistency.py tests/test_agent_docs.py -q`.

## 8. Dependencies and lock files

- 137b46a: chromadb left posthog unbounded and posthog 4 broke its telemetry; pinned `posthog<4`.
- 728af8e: transitive dependencies floated until the hashed lock files.
- 04eeba2: uv stripped extras, after which `--require-hashes` refused the whole lock
  (`--no-strip-extras` since).
- 7fb55cd, 84dab23: a local `telegram/` directory shadowed the installed package and made two ruff
  versions disagree; the bot moved to `clients/telegram_bot.py`.
- c43fa91: flashrank is in no requirements file, and `RerankRequest` was imported inside the rerank
  call, so without the package even a test's injected ranker fell back silently to vector order and
  the test failed; a module-level fallback class fixed that path, but the real ranker still never loads.

Still open: choosing the flashrank reranker does nothing (Models, keys and reranking).

Look for requirements edits without regenerated locks, an import of a package in no requirements file,
and a module or directory named like an installed package.

## 9. Library behaviour assumed instead of checked

- c05358d: LangChain's Chroma defines `__len__`, so an empty store looked missing (compare with
  `is None`); Chroma upserts on an existing id.
- 7fb55cd: chromadb 0.4.24's `get_or_create_collection` overwrote collection metadata, defeating the
  embedding-model guard.
- 94d5d79: similarity search returns squared-L2 distances, and the relevance-score API returns
  negative values.
- 771f1a3: `count()` returns 0 for a collection that was never opened, so `/ready` uses `ping()`.
- 59bb70b: chromadb's shared system clients leaked file descriptors ("Too many open files" at about 800
  tests on Windows); `_release_chroma` in `tests/conftest.py` stops them.
- 3b8df92: `with sqlite3.connect(...)` commits but does not close the connection.

Still open: collections created outside the app score wrongly and skip the embedding-model check
(Retrieval and search).

Look for a library call whose semantics the diff assumes, and check it against the pinned version in
the venv rather than the latest documentation.

## 10. Client limits and duplicated client logic

- Streamlit: reruns re-uploaded every file and Clear re-indexed it all (c05358d); model output was
  rendered as unescaped HTML and `.env` was never loaded (137b46a); uploads were keyed by name and size
  (7fb55cd); a deleted file's failed state skipped its re-upload (59bb70b).
- Telegram: a document without a file name crashed the bot and "<" in an answer broke the message
  (137b46a); there was no error handler (7fb55cd); messages are capped at 4096 characters, measured
  after escaping (2f3a3b6); callback data is capped at 64 bytes (6550750).
- 84dab23: both clients carried their own error wording, which diverged; it lives in
  `clients/backend.py` now.

Still open: an empty or cut-short answer breaks the session, ratings after Clear, failed uploads not
retried, the filter boxes (Web UI); `/model` argument forms that leave, echo or misroute the key, an
empty answer, an unsplit `/documents` (Telegram bot).

Look for a new reply without `split_for_telegram`, HTML built from model or document text without
escaping, client-side copies of lists or wording, and history updated after a failure or with an empty
answer.

## 11. Tests passing for the wrong reason

- 137b46a: the "fully offline" claim held only by luck (the one `/query` test expecting 200 queried an
  empty corpus), and the path-traversal test passed with sanitisation disabled.
- 7fb55cd: the autouse embedding patch left `OpenAIEmbeddingFunction` with no coverage at all.
- d5e9eb4: doubles of `RAGChain.ask` with fixed signatures turned a new parameter into a TypeError
  that surfaced as an unrelated 503; doubles take `**kwargs` since.
- 9f1362b, 4c37696: assertions measured the bag-of-words test embedder instead of the pipeline.
- 6550750: tests wrote outside `tmp_path` (the feedback tests appended to the repository's own data).
- 59bb70b: two sweep tests aged only the activity marker (`last_seen` takes the newer of marker and
  files), a concurrency test hoped for an interleaving instead of forcing one, and a test compared
  `id()` of objects nothing kept alive.
- 6828a34: moving functions into `app/storage.py` left patches aimed at the old module; patch
  `app.storage.<fn>`, which every caller goes through.
- 37bea16: a test assertion was replaced rather than a new one added.

Still open: `test_uploading_stays_on_the_operators_key` in `tests/test_frontend_feedback.py` cannot
fail; the coverage gaps listed in the Tests section of SKILL.md.

Look for assertions that also hold on the failure path (`x is None or ...`, `>= 0`, a loop over an
empty list), doubles with fixed signatures, fakes that ignore `stream=True`, ranking assertions through
the fake vectors, replaced or loosened assertions, and writes outside `tmp_path`.

Catch it by mutation-checking: revert the protected line and see a named test fail.

## 12. Heuristics and defaults that silently narrow retrieval

- `app/rag/metadata_extractor.py` needed a burst of follow-ups right after 4dc05e9 introduced it:
  37bea16 (multi-year comparisons were hard-filtered to one year), 95a8f4e (company names not
  normalised), 78b83c7 (8-K came out as "8K-"), f2daf78 and 7e64708 (year spans, prior-year guidance).
- 4c37696: a 0.7 MMR default was written into the config before measuring; the measurement said 1.0.

Still open (Retrieval and search): automatic filtering hides the right document in general
collections; "annual report" discards the filters; spans over five years filter on the end years only;
METADATA_FILTERING_ENABLED off does not stop explicit filters; a filter that matches nothing falls back
to tenant-wide search without telling anyone. Document ingestion: a hyphenated filing type in a file
name is not tagged.

Look for a regex or alias table applied to every question or upload, a new default of `True` for
something that can narrow results, and a fallback that hides that the filter missed.

Catch it with false-positive cases in `tests/test_metadata_filtering.py` and a before/after run
(rag-evaluation skill).

## 13. Lint and import errors committed

- 137b46a: CI shipped red on ruff I001.
- b2f6e2a used `Dict` and `Any` in `app/models/schemas.py` without importing them; importing the app
  raised NameError until f11cb2a.
- 09e93ef, 21d795d: unsorted and unused imports fixed after the fact.

Catch it with `ruff check app frontend clients evaluation scripts tests` and
`python -c "import app.main"` before every commit.

## 14. Runtime data directories, volumes and secrets

- 3b8df92: `data/feedback` was on no Compose volume, so collected ratings were lost on every rebuild;
  bb5b794: it was not git-ignored either.
- df88d5b: the Langfuse Compose file shipped without ENCRYPTION_KEY; the fix committed literal values
  for it, NEXTAUTH_SECRET and SALT, so those are public and fit a local stack only.
- 771f1a3: the Compose healthcheck probed `/health` instead of `/ready`.

Still open (Operations: backup, restore and sweep): restoring into the Compose volumes fails after
emptying the index volume; the tracked uploads placeholder counts as data; archives store most files
two or three times; unverified volume risks (feedback volume ownership, one-off containers that cannot
see the backend).

Look for a new directory without the full set (Settings path, `make_settings` override, volume in both
Compose files, `.gitignore`, `.dockerignore`, Dockerfile `mkdir` and `chown`, backup member), and
secrets with working defaults in Compose files.

## 15. Untested evaluation scripts

- `evaluation/langfuse_eval.py` took a long run of follow-up commits after cdddc0a: 3f8f41c (model and
  provider headers sent without a key), ef692e8 (duplicate contexts needed a hash map), bcf9b98 (unbuffered output, regular
  Langfuse flushes), and two redesigns of the score format (1362ac0, d676096).

Still open (Evaluation and benchmarks): a crash on a dataset that does not exist yet, the author-specific
`eval_sberquad_failures`, the RAGTruth file name, aggregate metrics, the 200-document quota, and
`scripts/download_benchmarks.py` downloading on any invocation.

Look for a network script changed without a test or a recorded run, and output that reports success
after skipping work.

## 16. Open known issues by file

What is open where a diff lands (section names as in KNOWN_ISSUES.md). A fix removes the entry and adds
a regression test that fails on the old behaviour; a change nearby must not make one worse.

| The diff touches | Open issues there (KNOWN_ISSUES.md section) |
| --- | --- |
| `app/sweep.py`, `app/storage.py` | an applied sweep leaves the owner in BM25 (Operations: backup, restore and sweep) |
| `app/rag/hybrid.py`, `app/rag/bm25.py` | `$and` filters skip BM25; lazy sync only at zero entries; owner-less search is every tenant; `_owned`-shaped filters crash; global BM25 statistics; index filled with hybrid off (Retrieval and search) |
| `app/rag/chain.py` | MMR plus a metadata filter is a 503; a filter that matches nothing falls back silently; BM25 hits bypass RELEVANCE_THRESHOLD; METADATA_FILTERING_ENABLED does not stop explicit filters (Retrieval and search); streamed answers record no token usage (Tracing and observability) |
| `app/rag/metadata_extractor.py`, `app/models/schemas.py` | false-positive filters, "annual report", wide year spans, untyped `metadata_filter` (Retrieval and search); hyphenated 10-K file names (Document ingestion) |
| `app/rag/embeddings.py` | collections created elsewhere score wrongly (Retrieval and search); oversized documents fail after embedding (Document ingestion) |
| `app/main.py` `query_rag_stream`, `app/byok.py`, `app/rag/reranker.py` | the 504 branch leaves the client open; reranker headers bypass RERANKER_ENABLED; flashrank missing (Models, keys and reranking) |
| `app/tracing.py`, `app/observability.py`, `submit_feedback` | feedback without `request_id` scores the wrong trace; a caller's `X-Request-ID` becomes the trace id (Tracing and observability) |
| `app/backup.py`, `scripts/backup.py`, `scripts/restore.py`, `Dockerfile`, Compose files | restore into Compose volumes, `.env` ignored on the host, duplicated archive members, the uploads placeholder counted as data, unverified volume risks (Operations: backup, restore and sweep) |
| `app/rag/document_loader.py`, `app/rag/table_splitter.py` | DOCX cells and table order, table detection, `.doc` advice (Document ingestion) |
| `frontend/streamlit_app.py` | empty answer, ratings after Clear, failed uploads, filter boxes (Web UI) |
| `clients/telegram_bot.py` | `/model` argument forms, empty answer, unsplit `/documents` (Telegram bot) |
| `tests/conftest.py` | a root `.env` changes test results (Development and tests) |
| `evaluation/`, benchmark scripts in `scripts/` | dataset creation, aggregate metrics, quota, RAGTruth file name, author-specific `eval_sberquad_failures` defaults, `download_benchmarks` downloading even on `--help` (Evaluation and benchmarks) |
