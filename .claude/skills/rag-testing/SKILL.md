---
name: rag-testing
description: Writes and runs the offline pytest suite - the conftest fixtures and fakes (api, make_settings, FakeChatClient, fake embeddings, the socket guard), which test file covers which module, how to run a targeted subset or everything (also from WSL), and the habits that keep tests honest. Use when adding or fixing tests, deciding what to run after a change, or debugging a failing, flaky or network-blocked test (a red doc-drift test is rag-config-and-docs).
---

# Testing

The suite in `tests/` is offline by construction: `tests/conftest.py` strips every app setting
from the environment, refuses sockets to anything but loopback, fakes the OpenAI embeddings and
chat client, and closes the ChromaDB handles each test opened. CI runs the ruff command below,
then `pytest -q`, on Linux with Python 3.10. This skill covers writing tests that fit the
harness, choosing what to run after a change, running pytest on any OS and debugging a red test.

## Use this skill for / not for

- For: a new or failing test, a test that hits the socket guard or depends on order, choosing
  the tests for a change, running pytest from WSL or Windows.
- Not for:
  - what each doc-drift test enforces and how to fix the docs: rag-config-and-docs.
  - installing the environment, lint fixes, locks, CI jobs, commit messages: rag-development-guide.
  - what a change must keep true in its area (and so must test): rag-api-endpoints,
    rag-retrieval-pipeline, rag-document-ingestion, rag-llm-providers, rag-clients,
    rag-operations. Live evaluation and benchmarks: rag-evaluation.

## The harness: `tests/conftest.py`

- `_isolated_environment` (autouse, session): pops the UPPER_CASE variable of every Settings field
  at session start and restores them at the end, because `Settings(_env_file=None)` still reads
  os.environ. BACKEND_URL and TELEGRAM_BOT_TOKEN are not Settings fields and are not stripped:
  set or delete them with monkeypatch.
- `_no_real_network` (autouse, session): sets OPENAI_API_KEY to `test-key-not-real`, overriding a
  real one, and patches `socket.socket.connect` and `socket.create_connection` so a non-loopback
  address raises `RuntimeError("A test tried to connect to ... The suite must stay offline - stub
  the client instead.")`. Loopback (127.0.0.1, ::1, localhost, 0.0.0.0, empty host) and non-tuple
  addresses (AF_UNIX) pass. `tests/test_offline_guarantee.py` matches the words "must stay".
- `_release_chroma` (autouse, function): stops every ChromaDB System the test created, through
  chromadb's private `SharedSystemClient._identifer_to_system`; without it Windows ran out of file
  descriptors after about 800 tests. Warns when a stop() fails.
- `fake_openai_embeddings` (autouse, function): patches `OpenAIEmbeddingFunction.embed_documents`
  and `embed_query` on the class with `_fake_vector`: 32 floats in [-1, 1] seeded from the text's
  sha256, not normalised. Same text, same vector; different texts, unrelated vectors.
- `api` (function): `create_app(make_settings(tmp_path))` inside
  `with TestClient(app, raise_server_exceptions=False)`: the lifespan has run, and a crash comes
  back as a JSON 500 rather than an exception. It sends X-API-Key `TEST_API_KEY` and replaces
  `app.state.rag_chain.client` with `FakeChatClient()`. Extras: `api.chat` (that fake),
  `api.app_state` (app.state), `api.upload_dir` (a Path); TestClient's own `api.app` is the app.
- `make_settings(tmp_path, **overrides)`: `Settings(_env_file=None, ...)` with openai_api_key
  `test-key-not-real`, backend_api_key `TEST_API_KEY`, and upload_dir, chroma_persist_dir and
  feedback_dir at tmp_path / "uploads", "chroma" and "feedback", then your overrides.
- `FakeChatClient(answer=FAKE_ANSWER, finish_reason="stop", usage=<default>)`: stands in for
  openai.OpenAI. `chat.completions.create(**kwargs)` appends the kwargs to `.calls` and returns the
  answer, finish_reason and usage (120 prompt, 40 completion, 160 total tokens; `usage=None`
  imitates an endpoint that omits the block). With `stream=True` it yields the answer in two delta
  chunks, finish_reason on the last only. It has no close().
- Constants: `TEST_API_KEY = "test-backend-key"`, `FAKE_ANSWER = "This is a canned offline answer."`.
- `tests/__init__.py` exists so conftest loads once, as `tests.conftest`: import helpers with
  `from tests.conftest import TEST_API_KEY, FakeChatClient, make_settings`, never `import conftest`
  (a second module copy makes isinstance checks fail).
- `pyproject.toml`: `testpaths = ["tests"]`, `--strict-markers --strict-config`. Only
  `pytest.mark.parametrize` is used; register any other marker there. No pytest-asyncio.

## Writing a test

1. Extend the `tests/test_<area>.py` that covers the code
   ([references/test-index.md](references/test-index.md) says which; read it to find the file), or
   start one with a module docstring naming the behaviour or regression it guards.
2. Test at the narrowest level that shows the behaviour: a pure function directly, chain logic
   through `RAGChain` with a fake store, HTTP behaviour through `api`, a client through AppTest or
   handler fakes.
3. Assert on what a user or the model sees: status and body (`assert r.status_code == 200, r.text`),
   what reached the model (`api.chat.calls`), what is on disk under `api.upload_dir`.
4. Mutation-check it (Habits below), then run it with the area's tests and ruff.

By area. Skeletons that run as written: [references/patterns.md](references/patterns.md); read it
before writing one of these.

- HTTP endpoints: the `api` fixture. Upload with `api.post("/upload", params={"user_id": "u1"},
  files={"file": ("doc.txt", b"...", "text/plain")})`, ask with
  `api.post("/query", json={"question": "...", "user_id": "u1"})`, then inspect
  `api.chat.calls[0]["messages"]`, `api.app_state` or `api.upload_dir`. Streams: enter
  `api.stream("POST", "/query/stream", json=...)` and parse the `data: ` lines. Non-default
  settings: your own fixture modelled on `limited` in `tests/test_quotas.py`.
- RAGChain: `RAGChain(store, client=FakeChatClient(), top_k=..., ...)` with a store double that has
  `similarity_search(query, k, filter)` (`tests/test_chain.py`; `ScoredStore` in
  `tests/test_relevance.py` for the scored path). Pass `metadata_filtering_enabled=False` unless
  the test is about filters (see Debugging).
- Ranking or similarity: never assert an order, score or threshold effect from the autouse
  vectors. Inject a bag-of-words embedder with `EmbeddingsManager(persist_directory=str(tmp_path /
  "chroma"), embedding_fn=...)` (`indexed` in `tests/test_retrieval_quality.py`, `crowded` in
  `tests/test_mmr.py`).
- Web UI: `AppTest.from_file("frontend/streamlit_app.py", default_timeout=30).run()` with
  `requests.get`, `requests.post` and `requests.delete` patched on the `requests` module itself,
  then `assert not at.exception` (`tests/test_frontend_feedback.py`).
- Telegram bot: SimpleNamespace update and context with async methods, `monkeypatch.setattr(
  bot.httpx, "AsyncClient", FakeClient)`, then `asyncio.run(bot.handle_message(update, context))`
  in a plain test (`tests/test_feedback.py`, `tests/test_byok.py`). The bot reads BACKEND_URL,
  BACKEND_HEADERS and FEEDBACK_ENABLED at import: patch them on the module, not in the environment.
- DOCX and PDF: build them in the test, with python-docx (`_docx` in `tests/test_docx_markdown.py`)
  or `from tests.test_upload_errors import _pdf`. No binary fixture files.
- Concurrency: wrap the function between check and write so each request waits on a
  `threading.Barrier(2)` with a timeout, and send requests from threads on one TestClient
  (`tests/test_quotas.py`); without the barrier it passed on a fast machine, lock or no lock.
- Logs: `caplog.at_level("WARNING", logger="app.main")`. Modules under `app/` and `clients/` log
  through `logging.getLogger(__name__)`, so the logger is the module path (`app.rag.chain`,
  `app.observability`, `app.sweep`, `clients.telegram_bot`).
- A caller's own key: `monkeypatch.setattr(byok, "client_for", lambda key, settings,
  provider=None: FakeChatClient())`. Cohere and Langfuse: `unittest.mock.patch` on
  `httpx.Client.post` and `langfuse.Langfuse` (`tests/test_reranker.py`, `tests/test_tracing.py`).
- The real embedding body: `monkeypatch.undo()` first (it also drops every other monkeypatch made
  so far in the test), then stub `.client` (`tests/test_embedding_function.py`).

## Rules

- Write only under `tmp_path`: tests once appended to the repository's data/feedback and read
  each other's records. `make_settings` puts the three data directories there, and a new
  path-like Settings field needs a `make_settings` default too. `EmbeddingsManager` persists to
  `./data/chroma_db` unless given `persist_directory`; the backup and restore CLIs resolve
  `storage_locations()` from the environment unless patched.
- Enter the TestClient as a context manager whenever the test needs lifespan state
  (`app.state.rag_chain`, `embeddings`, `bm25_index`, `feedback`): a bare `TestClient(app)` never
  starts up, which only the before-startup /ready and /health tests want. Install FakeChatClient
  after entering, because the lifespan builds the chain.
- Build apps with `create_app(make_settings(tmp_path, ...))`, never `from app.main import app`: that
  lazy attribute builds an app from `get_settings()`, which reads the real environment and a
  repo-root .env, is cached for the process, and uses the data directories in the repository.
- Doubles take `**kwargs` (or `*args, **kwargs`): with a fixed signature, a new parameter in the
  real call becomes a TypeError that the route reports as an unrelated 503.
- Patch a name where the code under test looks it up. `app/main.py` and `app/sweep.py` call
  `storage.stored_files(...)` and its siblings through the module, so one `monkeypatch.setattr(
  storage, "stored_files", ...)` reaches every caller; `byok.client_for` works the same way. A
  name taken with `from x import y` is a separate reference per module: `scripts/restore.py`
  imports `backend_is_up` from `scripts/backup.py`, and both import `storage_locations` from
  `app/backup.py`, so patch `scripts.backup` and `scripts.restore` separately (`live_backend` in
  `tests/test_backup.py`). Keep the default `raising=True`, so a renamed target fails loudly.
- Put throwaway routes on a fresh `APIRouter()` and `api.app.include_router(router)`
  (`tests/test_observability.py`). `app.main.router` is one module-level object that every
  `create_app()` includes, so a route added to it shows up, behind the API key, in every app built
  later in the session.
- Name fixture files so git keeps them. `.gitignore` drops test_*.pdf, test_*.txt, sample_*.pdf,
  sample_*.txt, *.bin, *.sqlite3, *.log and lib/ anywhere in the tree, so such a file passes
  locally and is missing in CI. Prefer building fixtures in code; check a new file with
  `git check-ignore -v <path>`.
- Move a .env in the repository root (or a parent folder) aside before running tests. The web
  UI runs `load_dotenv()` on every AppTest run and the bot at import, which puts its values back
  into os.environ after the session strip, so later tests see them (`test_quotas_ship_on` then
  fails); see KNOWN_ISSUES.md, section Development and tests. Stub `load_dotenv` on the module
  of code you call that runs it (`scripts/sweep.py` `main`), as `via_api` in `tests/test_sweep.py` does.

## Running

Always from the repository root: `tests/test_frontend_feedback.py` opens
`frontend/streamlit_app.py` relative to the working directory; from `tests/`, AppTest falls back
to the test file's folder and fails with "File not found at <checkout>/tests/frontend/...".

```bash
pytest -q                                                   # everything, as CI runs it
pytest tests/test_api.py -q                                 # one file
pytest tests/test_api.py::test_health_needs_no_api_key -q   # one test
pytest "tests/test_docs_consistency.py::test_documented_default_matches_the_code[CHUNK_SIZE]" -q
pytest -k "quota and not bot" -q                            # by keyword
pytest tests/test_<area>.py -x -q                           # stop at the first failure
```

No environment variables are needed (conftest sets the dummy OpenAI key). `-p no:cacheprovider`
keeps .pytest_cache out of the checkout but removes `--lf`. The full suite takes well under a
minute on Linux and a few minutes on Windows; the contract group below takes seconds.

From WSL, call the venv's own interpreter. Which venv to build, and how (a Linux venv inside WSL
is exactly CI's set and the only option where Device Guard / WDAC blocks the Windows interpreter),
is in rag-development-guide:

```bash
~/.venvs/<name>/bin/python -m pytest -q -p no:cacheprovider                      # Linux venv inside WSL
.venv/Scripts/python.exe -m pytest tests/test_<area>.py -q -p no:cacheprovider   # Windows venv
WSLENV=NAME NAME=value .venv/Scripts/python.exe -m pytest -q   # Linux variables reach Windows only via WSLENV
```

Give a Windows interpreter repo-relative paths (it cannot see Linux paths such as /tmp); in WSL,
git may exist only as git.exe (`git.exe check-ignore -v <path>`).

## What to run after a change

Verify in this order: the files for what you touched, the contract group below,
`ruff check app frontend clients evaluation scripts tests`, and `pytest -q` before a pull request.

| You changed | Run |
| --- | --- |
| `tests/conftest.py` | `tests/test_offline_guarantee.py`, then everything |
| `app/config.py` | `tests/test_config.py`, `tests/test_docs_consistency.py`, the default pins below |
| `app/main.py`: auth, user_id, upload, dedup, /clear | `tests/test_api.py`, `tests/test_upload_errors.py`, `tests/test_document_management.py` |
| `app/main.py`: /query, /query/stream, error mapping | `tests/test_query_e2e.py`, `tests/test_streaming.py`, `tests/test_resilience.py`, `tests/test_conversations.py` |
| quotas, `OwnerLocks`, `app/humanize.py` | `tests/test_quotas.py`, `tests/test_upload_errors.py` |
| `app/observability.py`, /health, /ready | `tests/test_observability.py` |
| `app/feedback.py`, /feedback, `evaluation/from_feedback.py` | `tests/test_feedback.py`, `tests/test_tracing.py`, `tests/test_backup.py` (ratings in the archive) |
| `app/sweep.py`, `app/activity.py`, `app/storage.py`, `scripts/sweep.py` | `tests/test_sweep.py`, `tests/test_quotas.py`, `tests/test_backup.py` |
| `app/backup.py`, `scripts/backup.py`, `scripts/restore.py` | `tests/test_backup.py` |
| `app/byok.py` | `tests/test_byok.py`, `tests/test_reranker.py`, `tests/test_frontend_feedback.py` |
| `app/tracing.py` | `tests/test_tracing.py` |
| `app/models/schemas.py` | the endpoint files that use the model; `tests/test_conversations.py` for history |
| `app/rag/chain.py` | `tests/test_chain.py`, `tests/test_relevance.py`, `tests/test_mmr.py`, `tests/test_conversations.py`, `tests/test_streaming.py`, `tests/test_resilience.py`, `tests/test_metadata_filtering.py`, `tests/test_reranker.py`, `tests/test_retrieval_quality.py`, `tests/test_query_e2e.py`, `tests/test_offline_guarantee.py` |
| `app/rag/embeddings.py` | `tests/test_embeddings.py`, `tests/test_embedding_function.py`, `tests/test_relevance.py`, `tests/test_mmr.py`, `tests/test_retrieval_quality.py`, `tests/test_resilience.py`, `tests/test_offline_guarantee.py`, then the API files |
| `app/rag/bm25.py`, `app/rag/hybrid.py` | `tests/test_bm25.py`, `tests/test_hybrid.py`, `tests/test_query_e2e.py` (hybrid is on in `api`) |
| `app/rag/reranker.py` | `tests/test_reranker.py` |
| `app/rag/metadata_extractor.py` | `tests/test_metadata_filtering.py`, then the chain row |
| `app/rag/document_loader.py` | `tests/test_document_loader.py`, `tests/test_docx_markdown.py`, `tests/test_upload_errors.py` |
| `app/rag/text_splitter.py`, `app/rag/table_splitter.py` | `tests/test_text_splitter.py`, `tests/test_table_splitter.py`, `tests/test_api.py`, `tests/test_config.py` |
| `app/rag/languages.py` | `tests/test_language_consistency.py`, `tests/test_chain.py`, `tests/test_query_e2e.py` |
| `clients/backend.py` | `tests/test_clients_backend.py`, `tests/test_observability.py`, `tests/test_feedback.py`, `tests/test_byok.py`, `tests/test_quotas.py`, `tests/test_language_consistency.py` |
| `clients/telegram_bot.py` | `tests/test_telegram_splitting.py`, `tests/test_byok.py`, `tests/test_feedback.py`, `tests/test_quotas.py`, `tests/test_sweep.py`, `tests/test_language_consistency.py`, `tests/test_docs_consistency.py`, `tests/test_docx_markdown.py` |
| `frontend/streamlit_app.py` | `tests/test_frontend_feedback.py`, `tests/test_language_consistency.py`, `tests/test_docx_markdown.py` |
| `evaluation/metrics.py`, `evaluation/golden.py` | `tests/test_metrics.py`, `tests/test_retrieval_quality.py` |
| requirements files or locks | `tests/test_lockfile.py` |
| README.md, DOCUMENTATION.md, `.env.template`, compose, CI workflow | `tests/test_docs_consistency.py`, `tests/test_agent_docs.py` (links, anchors, KNOWN_ISSUES.md section citations, lint commands) |
| CLAUDE.md, KNOWN_ISSUES.md, `.claude/skills/` | `tests/test_agent_docs.py` |

`evaluation/run_eval.py`, `evaluation/langfuse_eval.py`, `evaluation/eval_sberquad_failures.py`,
`scripts/download_benchmarks.py` and `scripts/upload_to_langfuse.py` have no tests: ruff and a
real run are the only checks (rag-evaluation).

The contract group, whenever docs, settings, routes, languages, providers, formats, bot commands
or dependencies change (what each rule checks: rag-config-and-docs):

```bash
pytest tests/test_docs_consistency.py tests/test_language_consistency.py tests/test_lockfile.py tests/test_offline_guarantee.py tests/test_agent_docs.py -q
```

Add `tests/test_byok.py` and `tests/test_frontend_feedback.py` for a provider (the first pins the
PROVIDERS set, the second checks that the web UI picker offers it), and `tests/test_docx_markdown.py` for a format (SUPPORTED_EXTENSIONS must appear in
`app/main.py`, `frontend/streamlit_app.py` and `clients/telegram_bot.py`).

Tests that pin a default. Changing the default means changing them on purpose, with the reason or
measurement:

- MAX_DOCUMENTS_PER_USER 200 and MAX_BYTES_PER_USER 1 GiB: `test_quotas_ship_on` in
  `tests/test_quotas.py`.
- MMR_LAMBDA 1.0: `test_mmr_ships_disabled` in `tests/test_mmr.py`.
- RELEVANCE_THRESHOLD 0.0: `test_the_threshold_defaults_to_disabled` in `tests/test_relevance.py`
  and `test_the_shipped_threshold_is_still_disabled` in `tests/test_retrieval_quality.py`.
- FEEDBACK_ENABLED True: `test_feedback_ships_enabled`, and for the client mirror in
  `clients/backend.py` `test_the_buttons_are_offered_by_default` and
  `test_an_empty_value_falls_back_to_the_backends_default`, all in `tests/test_feedback.py`.
- MAX_FILE_SIZE: `test_max_file_size_default_matches_the_backend` (`tests/test_clients_backend.py`)
  holds `DEFAULT_MAX_FILE_SIZE` in `clients/backend.py` equal to the Settings default.
- Every default: `test_documented_default_matches_the_code[<NAME>]` in
  `tests/test_docs_consistency.py` compares it with the DOCUMENTATION.md Configuration table.

## Habits that keep tests honest

- Mutation-check new code: undo the key line of the change (comment it out, invert the
  condition, drop the lock), run the new test and watch it fail on your assertion, then restore
  the line and watch it pass. A test that still passes is testing something else. Commit bodies
  record the result in their closing `Tests:` paragraph: how many mutations of the new code a
  named test caught, and why any survivor is acceptable.
- Put the response in the assertion message, `assert r.status_code == 200, r.text`: the status
  alone says nothing about why.
- Assert on the outcome, not on the absence of one guessed failure: the traversal test in
  `tests/test_api.py` checks where every stored file resolved to, because a version checking one
  escape path passed with sanitising removed. Give "every record" assertions a positive control
  (`test_the_id_reaches_log_lines_from_other_modules` first asserts app.rag logged anything).
- Keep the suite offline and deterministic: stub the client (FakeChatClient, `byok.client_for`,
  `bot.httpx.AsyncClient`, the `requests` module, `httpx.Client.post`) rather than leaning on the
  guard, which patches only connect and create_connection; no real keys, no sleeps, no timing
  races (force them with a barrier), no reliance on test order.

## Debugging a red test

1. Rerun it alone from the repository root: `pytest "<file>::<test>" -q`.
2. 503 `{"detail": "Query failed"}` from an API test, slow because the OpenAI SDK retries: a real
   client was called, and "Captured log call" shows APIConnectionError caused by "The suite must
   stay offline". A custom fixture forgot FakeChatClient (`api_without_feedback` in
   `tests/test_feedback.py` never queries, so it has none) or a per-request client was not stubbed.
3. A bare 500 `{"detail": "Internal server error", ...}` is an unhandled exception that
   `raise_server_exceptions=False` swallowed without logging it: enter the client as
   `with TestClient(app) as client:` (the default re-raises) to see the exception in the test.
4. Passes alone, fails in the full run: a repo-root .env (above), a patch made without
   monkeypatch, a route added to `app.main.router`, or a module constant changed in place.
5. A chain test sees `{"$and": [...]}` instead of `{"user_id": ...}`: a bare `RAGChain` takes
   arguments left as None from os.environ and then the code defaults, so automatic metadata
   filtering is on for any question naming a year, quarter or company.
6. Every test errors at setup after a chromadb upgrade, or Windows reports "Too many open files"
   late in a run: `_release_chroma` and its private chromadb attribute.
7. "Interrupted: 1 error during collection" in `tests/test_docs_consistency.py`: a Settings field
   has no default; that module builds Settings at import.
8. Red only in CI: a fixture file swallowed by `.gitignore`, or transitive versions that differ
   from the lock. Red only on Windows: file handles or path separators; CI runs Linux only.

Coverage gaps worth knowing: `api` runs every default, so hybrid search, metadata filtering and
table-aware chunking are on in every API test; no test drives the API with them off, and none
checks the in-memory BM25 index after an API delete, /clear, sweep or restart (only its own
delete and clear are unit-tested, in `tests/test_bm25.py`).

## Related skills

rag-config-and-docs (the rules behind the contract tests), rag-development-guide (environment,
lint, locks, CI, commits), the area skills rag-api-endpoints, rag-retrieval-pipeline,
rag-document-ingestion, rag-llm-providers, rag-clients and rag-operations (what each area must keep
true), rag-evaluation (live benchmarks) and rag-change-review (the pre-merge checklist).
