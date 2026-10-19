---
name: rag-change-review
description: Project-specific review checklist for a diff, branch or pull request in this repository - tenant scoping on every vector and BM25 path, plain def handlers for blocking work, owner locks, BYOK key lifecycle, SSE priming, status-code conventions, side-index and backup consistency, offline tests and the doc-drift contract, plus the bug classes that recur here. Use when reviewing or self-checking a change before commit or merge, alongside general code review.
---

# Reviewing a change

The project layer of a code review: what a diff here can break without any test noticing, and the
defect classes this repository's history keeps repeating. Most were silent - a tenant filter missing
on one path, an index not updated on one mutation path, a test passing for the wrong reason. Run a
general review (logic, edge cases, readability) alongside it.

## Use this skill for / not for

- For: reviewing a branch or pull request, re-reviewing after fixes, self-checking a working tree
  before a commit.
- Not for: how to build the change (rag-api-endpoints, rag-retrieval-pipeline, rag-document-ingestion,
  rag-llm-providers, rag-clients, rag-config-and-docs), writing tests (rag-testing), lint, lock and CI
  commands (rag-development-guide), operating a deployment (rag-operations).

## Get the change

```bash
git fetch origin                         # compare with origin/main if local main is behind
git log --oneline main..HEAD             # the commits; read their messages
git diff --stat main...HEAD              # files changed since the merge base (three dots)
git diff main...HEAD -- app/             # then read area by area
git status --short                       # self-check: staged, unstaged and untracked files
git diff HEAD                            # self-check: everything not yet committed
```

- WSL with only Windows Git: run the same commands as `git.exe ...` from the repository root.
- A pull request, if the GitHub CLI is installed: `gh pr view <number>` (description),
  `gh pr diff <number>`, `gh pr checks <number>` (CI), `gh pr checkout <number>` to run checks locally.
- Untracked files appear in no `git diff`: read them whole.
- Read each changed function in full and grep its callers
  (`grep -rn "<symbol>" app clients frontend scripts evaluation tests`). Defects here usually sit on
  the path the diff did not touch: the sweep beside `/clear`, `/query/stream` beside `/query`.

Grep the added lines for patterns that broke things before (a hit is a place to look, not a verdict):

```bash
git diff main...HEAD -- app clients frontend scripts evaluation | grep -nE '^\+.*(async def|os\.getenv|os\.environ|@application\.|detail=(f"|str\()|_owned\(|filter=None|owner=None|similarity_search|bm25_index\.|wipe_namespace|for_owner\(|client_for\()'
git diff main...HEAD | grep -nP '^\+.*[\x{2013}\x{2014}]'    # en and em dashes (GNU grep)
```

## Procedure

1. Read the commit messages or PR description: what is claimed, and what was verified.
2. Triage with the table, then walk the named checklist sections against the code, not only the diff.
3. Run the checks under Verify. Passing tests are worth what they assert.
4. Confirm each suspected defect: a failing test, a scratch probe (Verify) or the exact code path.
   Mark the findings you only read.
5. Check the open known issues for the touched files (Known issues below): does the change fix one,
   or make one worse?
6. Report.

| The diff touches | Walk |
| --- | --- |
| `app/main.py` handlers, `app/models/schemas.py` | Handlers, Owner locks, Status codes, SSE, Tenancy, Tests |
| `app/storage.py`, `app/sweep.py`, `app/activity.py`, `app/backup.py`, `scripts/` | Side indexes and data, Owner locks, Tenancy |
| `app/rag/` | Tenancy, Side indexes and data, SSE, Settings and docs, Tests |
| `app/byok.py`, `app/tracing.py`, `app/observability.py` | BYOK and secrets, Status codes, SSE |
| `app/config.py`, `.env.template`, README.md, DOCUMENTATION.md, CLAUDE.md, `.claude/skills/` | Settings and docs |
| `clients/`, `frontend/` | Clients, Status codes, BYOK and secrets |
| requirements files, locks, `Dockerfile`, Compose files, `.github/workflows/ci.yml` | Dependencies, Side indexes and data, Lint |
| `tests/` | Tests |

## Report findings

One entry per defect, most severe first. If a general review tool or skill prescribes an output
format, use it and keep these four fields:

```text
[blocker|major|minor|nit] <path> <Class.method or function> - the defect in one sentence
  Scenario: concrete input or state -> the wrong result (status, leaked data, crash, lost file)
  Evidence: the failing test or probe, or the code path; "read, not run" if not executed
  Fix: the smallest change that closes it, and the test that would have caught it
```

- blocker: a tenant can read, change or delete another's data; a key is stored, logged or echoed;
  data loss; the app cannot import or start; CI turns red (ruff, a doc-drift test, a stale lock).
- major: the wrong status class (5xx for a caller mistake, 401/403/429 for something the caller can
  fix, 4xx for an outage); blocking work on the event loop; a mutation path that skips BM25, disk or
  backups; a default that fails silently; a test that passes for the wrong reason.
- minor: drift no test catches (API reference fields, prose); a new branch without a test.
- nit: style. Lint-level issues are ruff's job.

End with the commands you ran and their results, what you could not verify (Docker, a real provider,
Windows-only behaviour), and the known issues the change fixes or touches.

## Checklist

### Tenancy

One ChromaDB collection and one in-memory BM25 index hold every tenant; the unauthenticated `user_id`
is the only separation, so every path carries an owner and fails closed without one.

- EmbeddingsManager's own get, query and delete calls scope through `EmbeddingsManager._owned`;
  `list_owners` (the sweep's owner scan) is the one deliberately unscoped lookup.
  `delete_documents(filter)` deletes whatever it is given (its one app caller, `storage.wipe_namespace`,
  passes `{"user_id": user_id}`). EmbeddingsManager's `similarity_search` and
  `similarity_search_with_score`, and the LangChain store on `app.state.vectorstore`, default to
  `filter=None`, which is every tenant: new code passes an owner filter or calls
  `search_candidates(query, k, owner)`.
- `RAGChain` and `HybridRetriever` take filters from `build_chroma_filter(user_id, metadata)`
  (`app/rag/metadata_extractor.py`): `{"user_id": ...}` or `{"$and": [{"user_id": ...}, ...]}`. It
  raises on an empty user_id and drops a caller-sent `user_id` key; `RAGChain._prepare` refuses a falsy
  user_id. An `_owned`-shaped filter there crashes every query while hybrid search is on (TypeError in
  `BM25Index.count_owner`, a 503).
- BM25 fails open: `BM25Index.search` without an owner searches every tenant, and
  `HybridRetriever.retrieve` takes the owner only from a top-level `user_id`. `$and` filters are safe
  today only because BM25's filter loop rejects every chunk on that key; a change that teaches BM25
  `$and` must take the owner from the clauses. Code that cannot determine an owner raises, as
  `_prepare` and `build_chroma_filter` do, rather than passing `None` or `""` on.
- Chunk ids stay `f"{user_id}-{file_hash}-{i}"` (`_store_and_index` in `app/main.py`): Chroma upserts
  on an existing id. Dedup, revision and delete lookups (`has_file_hash`, `file_hashes_for_source`,
  `delete_by_file_hash`) take the owner, because a content hash is the same for every tenant.
- The user_id rule (`[A-Za-z0-9_-]`, 1-64 characters) is copied in `USER_ID_PATTERN`/`USER_ID_QUERY`
  (`app/main.py`), `USER_ID_FIELD` (`app/models/schemas.py`), `OWNER_NAME` (`app/activity.py`) and
  `_USER_ID_PATTERN` (`app/observability.py`), and the sweep's `prefix` pattern follows it. Change all
  or none: the value is used raw as a directory name and inside chunk ids.

### Side indexes and data

- BM25 (`app.state.bm25_index`) changes with Chroma in `_store_and_index`, `_retire_older_revisions`,
  `delete_document` and `_clear_namespace`, but not in `sweep.execute`, which calls
  `storage.wipe_namespace` directly (open defect). Check a new per-owner index, cache or file on every
  path: upload, revision retirement, single delete, `/clear`, sweep wipe and orphan removal, restore,
  startup. In-memory state resets only on a restart (BM25 starts empty and syncs an owner from Chroma
  lazily, on a hybrid query, only while that owner has no entries); `python -m scripts.restore --live`
  leaves a running process stale.
- A purge both `/clear` and the sweep need belongs in `storage.wipe_namespace`, which both call
  (`app.main` imports `app.sweep`, so the sweep cannot reach `_clear_namespace`).
- `_store_and_index` writes Chroma, then BM25, in one `try` whose `except` unlinks the file but leaves
  vectors already written: nothing added to that block may fail on its own.
- A new runtime directory needs a `Settings` path under `data/`, a `make_settings` override in
  `tests/conftest.py`, a named volume in `docker-compose.yml` repeated in `docker-compose.override.yml`,
  entries in `.gitignore` and `.dockerignore`, and the `Dockerfile` `mkdir -p ... && chown` step (today
  only `data/uploads` and `data/chroma_db`).
- Backups cover exactly the three directories in `storage_locations` (`app/backup.py`): data that must
  survive a restore gets a member there, a setting a restore must agree with goes into
  `_settings_summary`, and in-memory state must be rebuildable from Chroma.
- `UPLOAD_DIR/<user_id>/` holds only uploads named `<hash>_<name>`: quotas and the sweep read every file
  there, an applied sweep deletes any file without vectors (inside its prefix), and a directory under
  UPLOAD_DIR with a valid user_id name counts as an owner (it gets an activity marker at startup).
  Other per-owner state lives elsewhere, like the markers in `UPLOAD_DIR/.activity/<user_id>`, and is
  removed in `storage.wipe_namespace`.

### Handlers and blocking work

- Handlers that touch Chroma, disk, the OpenAI SDK, parsing or BM25 are plain `def`, so FastAPI runs
  them in a threadpool; `/health` is the only `async def` route. A coroutine that blocks stalls every
  request, `/health` included.
- Pinned by tests: `upload_document`, `query_rag`, `clear_documents`
  (`test_request_handlers_are_not_coroutines` in `tests/test_resilience.py`), `/ready`, `/feedback` and
  `/maintenance/sweep`. `query_rag_stream`, `list_documents` and `delete_document` rely on review;
  give a new blocking route a pin.
- Authenticated routes go on the module-level `router`, which carries `require_api_key`;
  `@application.get(...)` inside `create_app` is public and reserved for `/health` and `/ready`.
- Domain modules (`app/sweep.py`, `app/backup.py`) raise `SweepError` or `BackupError`, never
  HTTPException; their caller maps them (the sweep endpoint, or `scripts/backup.py` and
  `scripts/restore.py`).

### Owner locks

- Whatever changes what an owner holds runs under `state.upload_locks.for_owner(user_id)`
  (`OwnerLocks`: 64 striped, non-reentrant `threading.Lock`s): the upload's dedup, quota, write and
  index, `/clear` (`_clear_namespace`) and each removal in `sweep.execute`. A check-then-act on an
  owner's data sits inside it.
- Never take it twice in one thread (calling `_clear_namespace` while holding it deadlocks, which is
  why the sweep calls `storage.wipe_namespace`) and never hold two owners' locks: they can share a stripe.
- `delete_document` takes no lock; a race with an identical re-upload is plausible but not reproduced,
  so a change there should take it. Keep slow new work out of the lock: it blocks the whole stripe.
- Locks and the BM25 index live in one process: nothing may assume or introduce several workers.

### BYOK and secrets

Details: rag-llm-providers.

- Outbound URLs come only from `PROVIDERS` in `app/byok.py` (`openai` means OPENAI_BASE_URL); no
  header, body field or other caller value may carry a URL.
- A model or provider without `X-Model-Key` is a 400 (`byok.wanted`); a new per-request option that
  spends the operator's money follows that rule. The `X-Reranker-*` headers break it (open defect).
- Validate headers (`_caller_model`, `_caller_reranker`) before `byok.client_for(...)`: the client
  exists before the `try` that closes it, so nothing in between may raise (`start_trace` catches its own errors).
- `byok.close_quietly(client)` on every exit: `finally` in `query_rag`; each priming `except` branch
  and the `finally` of `events()` in `query_rag_stream`, whose `APITimeoutError` branch misses it (open
  defect). A new early return or error branch needs it too.
- Pass `request.app.state.settings` into `byok.wanted` (without it ALLOWED_MODEL_PROVIDERS is ignored)
  and into `byok.wanted_reranker`.
- A key never reaches a log line, an HTTP detail, an exception message, a trace, a file, a cache or a
  module global. `BringYourOwnKeyError` messages never quote it; `describe_upstream_refusal` maps
  provider statuses to fixed wording. `RAGChain.ask` and `ask_stream` already send a failed
  completion's `str(exc)` to the tracer; do not widen that.
- Embeddings always run on the operator's key (one collection, one embedding model); a caller's key
  pays only for chat calls, the condensing call included. Both clients send BYOK headers only with
  questions (each client's `asking_headers`: `X-Model-*`, plus `X-Reranker-*` from the web UI), never
  with uploads; the bot keeps a key only in `context.user_data` and deletes the `/model` message on
  success (KNOWN_ISSUES.md, "Telegram bot", lists the argument forms that leave, echo or misroute it).

### Status codes and error mapping

Both clients replace a 401 or 403 with "contact the operator" and show every other `detail` as sent
(validation lists flattened; `describe_error` in `clients/backend.py`), so the code decides who is
told to act. Exact wording: rag-api-endpoints.

| Code | Meaning here |
| --- | --- |
| 400 | caller-fixable: filename, type, size, empty or unreadable file, BYOK or reranker headers, the provider refusing the caller's own key (its 429 included) |
| 401 | missing or wrong `X-API-Key` |
| 404 | a file_hash this owner does not hold; `/feedback` with feedback disabled |
| 413 | quota, with numbers and a remedy and no setting name (`_quota_problem`); not 403 (hidden) or 429 (invites a useless retry) |
| 422 | request validation (user_id, file_hash, body fields such as `ChatTurn` lengths) |
| 429 | the operator's key rate-limited, with `Retry-After: 20` |
| 500 | `/clear` failed; any unhandled exception, as JSON with `request_id` (`unhandled_error`) |
| 503 | store, disk or model failure: `Vector store unavailable`, `Storage unavailable`, `Query failed`, `Feedback storage unavailable`; `/ready` not ready |
| 504 | the model timed out (`APITimeoutError`) |
| 507 | the record would take the feedback file past FEEDBACK_MAX_BYTES |

- `detail` is fixed, user-safe text: log the exception with `logger.exception`, never put `str(exc)` or
  provider wording into a detail. `BringYourOwnKeyError` messages are the one deliberate exception.
- An upload failure after `write_bytes` unlinks the new file; an older revision is retired only after
  the new one is indexed.
- A malformed `metadata_filter` is a 503 today instead of a 422 (open defect); do not copy that.

### SSE

- `query_rag_stream` primes the generator (`next(stream)`) before returning `StreamingResponse`, so
  failures in condensing, retrieval, reranking and opening the completion keep their status codes.
- `RAGChain.ask_stream` calls `create(..., stream=True)` before its first `yield`; a yield moved above
  it turns a refused key into a 200 with a broken stream.
- Events: one `sources`, then `token` events, then `done`. Nothing retrieved: sources `[]`, one token
  "No relevant information found." and `done`, with no answer call. After the first byte a failure is a
  final `{"type": "error", "detail": "The answer was cut short."}`, with no exception text and no `done`.
- Request and prompt changes go in `RAGChain._prepare`, shared by `ask` and `ask_stream`
  (`test_both_paths_build_the_same_request` in `tests/test_streaming.py`).
- `/query` passes its result through `QueryResponse`; the stream sends the chain's dicts as they are.
  A new answer or source field goes into both paths and into the clients that read it.
- `RequestContextMiddleware` (`app/observability.py`) stays raw ASGI; `@app.middleware("http")` wraps
  the response in an extra task, a poor fit for a stream.

### Settings and docs

Every rule: rag-config-and-docs.

- A new `Settings` field has a default (`tests/test_docs_consistency.py` builds Settings from
  `openai_api_key` alone at collection time, so a required field stops the whole run), `Field` bounds or
  a validator, an uncommented `NAME=` line in `.env.template` and a DOCUMENTATION.md Configuration row
  whose first backticked value is exactly `str(default)`.
- Backend values reach components through the `app/main.py` lifespan or `request.app.state.settings`,
  never a new `os.getenv`: pydantic-settings reads `.env` without exporting it. The `os.getenv`
  fallbacks in the `RAGChain` constructor serve direct construction only; `app/backup.py` reads the
  environment on purpose, to work without an OpenAI key, and so ignores `.env` on the host (open defect).
- A path setting also gets a `make_settings` override; a value the clients need is mirrored in
  `clients/backend.py` (`DEFAULT_MAX_FILE_SIZE`, `feedback_enabled`); a variable only the clients read
  joins `CLIENT_ONLY_VARS` in `tests/test_docs_consistency.py`.
- A knob whose wrong value fails silently ships off until measured (RELEVANCE_THRESHOLD `0.0`,
  MMR_LAMBDA `1.0`); a changed retrieval default comes with a before/after measurement (rag-evaluation).
- A new route needs a ``### `METHOD /path` `` heading in DOCUMENTATION.md; its request fields and
  headers belong there too, which no test checks. Changed behaviour updates README.md, DOCUMENTATION.md,
  CLAUDE.md, the matching skill and KNOWN_ISSUES.md, in plain hyphens.

### Tests

How to write them: rag-testing.

- Offline by construction: `tests/conftest.py` blocks non-loopback sockets, strips Settings variables,
  fakes embeddings and forces a dummy OPENAI_API_KEY. Use the `api` fixture, or
  `create_app(make_settings(tmp_path, ...))` inside `with TestClient(...)` with a `FakeChatClient`
  installed after startup. Every written path lands under `tmp_path`.
- Doubles for `RAGChain.ask`, `ask_stream` and other seams accept `**kwargs`; a fake model answers
  `stream=True` with chunks, as `FakeChatClient` does.
- Not vacuous: assert outcomes (status and body; Chroma, BM25 and disk state; the request in
  `api.chat.calls`). `assert sent is None or ...` passes when nothing ran:
  `test_uploading_stays_on_the_operators_key` in `tests/test_frontend_feedback.py` cannot fail, since its
  AppTest never drives `st.file_uploader`. Fake vectors are seeded per text and carry no meaning, so no
  ranking or similarity assertion may rest on them.
- Mutation-checked: revert the protected line and see a named test fail (on a clean tree,
  `git checkout main -- <file>`, run the test, `git checkout HEAD -- <file>`). Deleting or loosening an
  assertion to get green is a finding.
- Concurrency is forced with a `threading.Barrier`, never sleeps; sweep tests age both the marker and
  the stored files (`last_seen` takes the newer); patches target `app.storage.<fn>`, since callers go
  through the module.
- New behaviour gets `tests/test_<feature>.py`. Gaps a change in the area should close: no API-level
  test asserts the BM25 index after any mutation (upload, delete, `/clear`, sweep); no test sets
  HYBRID_SEARCH_ENABLED or METADATA_FILTERING_ENABLED to false (table-aware chunking is off only in a
  `TextChunker` unit test); none combines MMR with a metadata filter.

### Clients

Details: rag-clients.

- Shared logic lives in `clients/backend.py` (headers, limits, `error_from_response`),
  `app/rag/languages.py`, `app/humanize.py`, `PROVIDERS` and `SUPPORTED_EXTENSIONS`: no second copy of
  a list or of error wording, and no client imports `app.main`.
- Telegram: split raw text with `split_for_telegram` (the 4096-character limit is measured after HTML
  escaping), escape after splitting, send sources as their own message, keep callback data within 64
  bytes. Every long reply needs this; `/documents` is not split (open defect). Keep
  `concurrent_updates(True)` and the `on_error` handler.
- History holds only successful, non-empty answers: `ChatTurn.answer` needs at least one character,
  and both clients store an empty answer today (open defects).
- A new bot command updates `CommandHandler`, `BOT_COMMANDS` in `tests/test_docs_consistency.py`,
  README.md and the `/help` text.

### Dependencies

Commands: rag-development-guide.

- A changed `requirements.txt` or `requirements-dev.txt` ships with both locks regenerated, not
  hand-edited, with the uv version CI pins (`UV_VERSION` in `.github/workflows/ci.yml`) and CI's flags,
  into the existing files. CI's lock job recompiles and fails on any diff; `tests/test_lockfile.py` is
  only the fast local half.
- Direct runtime packages are pinned with `==` in `requirements.txt` (`posthog<4` is the deliberate
  bound for chromadb telemetry); dev tools live only in `requirements-dev.txt`.
- A package in no requirements file is in neither the image nor CI: import it guarded, so startup
  survives without it, and make choosing the feature fail loudly (flashrank degrades silently, an open
  defect). Keep langfuse 2.x (`app/tracing.py` uses the v2 `trace` API) and chromadb 0.4.24
  (`app/rag/embeddings.py` relies on its distances; `_release_chroma` in `tests/conftest.py` on a private attribute).

### Lint and import

- `ruff check app frontend clients evaluation scripts tests` (rules F, E9 and I). A commit made without
  it shipped a NameError in `app/models/schemas.py` that stopped the app from importing.
- `python -c "import app.main"` needs no key or `.env` and catches import-time breakage.
- A new top-level package joins CI's ruff command and every documented copy of it
  (`test_the_linter_covers_every_package_of_ours`, `tests/test_agent_docs.py`).

## Recurring bug classes

The classes that keep coming back, with the commits that fixed one (read any with `git show <hash>`):
tenant scope missing on one path (c05358d, 137b46a, 94d5d79, 59bb70b); a side index skipped on one
mutation path (53aa168); blocking work on the event loop, and races (2f3a3b6, 59bb70b); the wrong status
class or leaked error text (2f3a3b6, 771f1a3, ad141a0, 1abb51b); streaming failures after the 200
(e031ab2, ad141a0); settings that never reach the code, and environment leaking into tests (137b46a,
7fb55cd, 82d046f); docs drifting from the code (53aa168, repaired in efff62e and aa63f87); dependencies
and locks (04eeba2, c43fa91); library behaviour assumed (7fb55cd, 94d5d79, 771f1a3); client limits and
duplicated client logic (2f3a3b6, 6550750, 84dab23); tests passing for the wrong reason (137b46a,
d5e9eb4, 9f1362b, 59bb70b); heuristics silently narrowing retrieval (37bea16, 78b83c7); lint and import
errors committed (b2f6e2a, repaired in f11cb2a); data directories, volumes and secrets (3b8df92,
df88d5b); untested evaluation scripts (3f8f41c, ef692e8). Each class with its history, the instances
still open and what to look for in a diff: [references/recurring-bugs.md](references/recurring-bugs.md)
(read it when a finding looks familiar, or the diff touches tenancy, side indexes, streaming or tests).

## Known issues

KNOWN_ISSUES.md lists the verified defects by area, and the "(open defect)" notes above mark the ones on
reviewed paths; the per-file lookup is the last section of
[references/recurring-bugs.md](references/recurring-bugs.md). A fix removes the entry and adds a
regression test that fails on the old behaviour; a change near one must not make it worse (for example,
a new mutation path that also skips BM25, or a new streaming branch without `close_quietly`).

## Verify

```bash
ruff check app frontend clients evaluation scripts tests
python -c "import app.main"                                         # import-time breakage
pytest tests/test_<area>.py -q                                      # what the diff touches (rag-testing maps modules to tests)
pytest tests/test_docs_consistency.py tests/test_agent_docs.py -q   # after config, route, command or doc changes
pytest -q                                                           # the whole offline suite, before a merge
```

Move a root `.env` aside before running the suite (KNOWN_ISSUES.md, "Development and tests"). From
WSL, use a Linux venv or call a Windows venv as `.venv/Scripts/python.exe -m pytest ...`
(rag-development-guide).

Probe a suspected defect without touching the checkout: put a test file in a scratch directory outside
the repository and load the suite's fixtures (`api`, `make_settings`) as a plugin, from the repository root:

```bash
python -m pytest -p tests.conftest <scratch-dir>/test_probe_<topic>.py -q -p no:cacheprovider
```

Prefer a probe to starting the backend, which writes its index and uploads under `data/` in the checkout.

## Related skills

- rag-architecture: components, request lifecycles and the invariants in one place.
- rag-api-endpoints, rag-retrieval-pipeline, rag-document-ingestion, rag-llm-providers, rag-clients:
  how each area works and how to change it.
- rag-config-and-docs: every doc-drift rule. rag-testing: fixtures, the module-to-test map, honest tests.
- rag-development-guide: lint, locks, CI, commits. rag-operations: backup, restore, sweep.
- rag-evaluation: the before/after measurement a retrieval change needs.
