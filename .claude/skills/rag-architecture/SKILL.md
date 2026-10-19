---
name: rag-architecture
description: Maps how the Multi-language RAG Document Assistant fits together - the FastAPI app factory and lifespan, the components on app.state, the upload, query, streaming, feedback and sweep request lifecycles, on-disk data layout, the user_id tenant model, and the invariants every change must keep. Use when explaining or locating code, planning a change that spans modules (a feature across backend, clients and docs, or moving code out of app/main.py), onboarding, or choosing which rag-* skill applies.
---

# RAG Document Assistant architecture

The big picture: which process owns which state, how a request travels from a client
through `app/main.py` into `app/rag/`, where data lands, and the rules that keep tenants
apart. Every module with its key symbols is in [references/modules.md](references/modules.md) -
read it when you need to find a function or decide where new code belongs.

- Use it for: locating code, explaining a request flow, planning a change that spans modules (recipes after "Where to change what": a feature end to end, extracting a module from `app/main.py`), onboarding, choosing a skill.
- Not for: the detail of one area - each row of "Where to change what" names the skill that owns it.

## Component map

```text
Streamlit UI   frontend/streamlit_app.py   user_id web-<uuid4 hex>, answers via POST /query/stream
Telegram bot   clients/telegram_bot.py     user_id = Telegram user id, answers via POST /query
HTTP tools     scripts/sweep.py, evaluation/run_eval.py, evaluation/langfuse_eval.py
      |  HTTP, X-API-Key (one shared secret), user_id in the query string or JSON body
      v
uvicorn app.main:app   one process, the FastAPI app built by create_app()
  RequestContextMiddleware   X-Request-ID in and out, one access log line (probes only on failure)
  on the app object (public)   GET /health, GET /ready (and FastAPI's /docs, /redoc, /openapi.json)
  router = APIRouter(dependencies=[Depends(require_api_key)])
      POST /upload  POST /query  POST /query/stream  GET /documents
      DELETE /documents/{file_hash}  POST /clear  POST /feedback  POST /maintenance/sweep
  app.state (built by lifespan)
      settings tracer feedback activity upload_locks loader chunker
      embeddings vectorstore bm25_index reranker rag_chain
      |
      +-- Chroma PersistentClient in CHROMA_PERSIST_DIR: one collection (COLLECTION_NAME)
      |     shared by every tenant; each chunk carries user_id metadata
      +-- BM25Index in process memory (state.bm25_index), owner-scoped, never persisted
      +-- UPLOAD_DIR/<user_id>/<hash>_<name> raw uploads, UPLOAD_DIR/.activity/<user_id> markers
      +-- FEEDBACK_DIR/feedback.jsonl ratings (append-only)
      +-- embeddings API: OPENAI_BASE_URL with OPENAI_API_KEY (always the operator's key)
      +-- chat API: LLM_BASE_URL or OPENAI_BASE_URL, LLM_API_KEY or OPENAI_API_KEY;
      |     per request a caller's own key at a PROVIDERS URL from app/byok.py
      |     (OPENAI_BASE_URL when no provider or "openai" is named)
      +-- optional: Cohere rerank API (app/rag/reranker.py), Langfuse (app/tracing.py)
```

The clients reach the backend only over HTTP. They import shared constants from
`app.byok`, `app.humanize`, `app.observability`, `app.rag.languages` and
`app.rag.document_loader`, never `app.main`; the last pulls in langchain and python-docx, so
the clients need the full requirements set.

## App factory and lifespan (`app/main.py`)

`create_app(settings=None)`: builds `FastAPI(version=API_VERSION, lifespan=lifespan)`, sets
`app.state.settings = settings or get_settings()` and a placeholder `NoOpTracer`, includes
the module-level `router`, adds `RequestContextMiddleware`, and registers `/health`
(`async def`, liveness only), `/ready` (`def`, `observability.readiness`: startup done and
`embeddings.ping()`; 200 or 503 with check names, never reasons) and a catch-all exception
handler (500 `{"detail": "Internal server error", "request_id": ...}`).

`lifespan(app)` builds `app.state` in this order, passing Settings values explicitly:

| Step | `app.state` attribute | Built by | Settings used |
| --- | --- | --- | --- |
| 1 | - | `configure_logging()` | - |
| 2 | - | warning if auth is off or the key is the `.env.template` placeholder | BACKEND_API_KEY |
| 3 | - | `upload_dir.mkdir()` | UPLOAD_DIR |
| 4 | `feedback` | `store_from_settings`: `FeedbackStore`, or None when disabled | FEEDBACK_ENABLED, FEEDBACK_DIR, FEEDBACK_MAX_BYTES |
| 5 | `activity` | `ActivityTracker(upload_dir)`, then `seed_missing()` | UPLOAD_DIR |
| 6 | `upload_locks` | `OwnerLocks()`: 64 striped `threading.Lock` | - |
| 7 | `loader` | `DocumentLoader()` | - |
| 8 | `chunker` | `TextChunker` | CHUNK_SIZE, CHUNK_OVERLAP, TABLE_AWARE_CHUNKING |
| 9 | `embeddings` | `EmbeddingsManager` (opens the Chroma client) | CHROMA_PERSIST_DIR, EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_TIMEOUT, OPENAI_MAX_RETRIES, OPENAI_BASE_URL |
| 10 | `tracer` | `get_tracer`: `LangfuseTracer` or `NoOpTracer` | LANGFUSE_ENABLED, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST |
| 11 | `vectorstore` | `embeddings.get_vectorstore`: get-or-create plus the embedding-model guard (a mismatch aborts startup) | COLLECTION_NAME |
| 12 | - | chat key, URL and model; a URL containing "deepseek" rewrites the model name | LLM_API_KEY or OPENAI_API_KEY, LLM_BASE_URL or OPENAI_BASE_URL, MODEL_NAME |
| 13 | `bm25_index` | `BM25Index()`, empty; created even when hybrid search is off | - |
| 14 | `reranker` | `get_reranker`: `NoOpReranker` unless enabled with flashrank or cohere | RERANKER_ENABLED, RERANKER_PROVIDER, RERANKER_MODEL, RERANKER_API_KEY |
| 15 | `rag_chain` | `RAGChain(vectorstore, ...)` wired to embeddings, tracer, reranker, bm25_index and the step-12 model, key and URL | TOP_K_RESULTS, TEMPERATURE, MAX_ANSWER_TOKENS, RELEVANCE_THRESHOLD, MAX_HISTORY_TURNS, MMR_LAMBDA, OPENAI_TIMEOUT, OPENAI_MAX_RETRIES, RETRIEVAL_CANDIDATES, METADATA_FILTERING_ENABLED, HYBRID_SEARCH_ENABLED, HYBRID_DENSE_WEIGHT, HYBRID_BM25_WEIGHT, HYBRID_RRF_K |

Shutdown runs `tracer.flush()` and `tracer.shutdown()`; nothing else is closed. Handlers also
read `request.app.state.settings` at request time: BACKEND_API_KEY (`require_api_key`),
MAX_FILE_SIZE, MAX_DOCUMENTS_PER_USER, MAX_BYTES_PER_USER, UPLOAD_DIR, and through
`app/byok.py` ALLOWED_MODEL_PROVIDERS, the OpenAI timeout, retries and base URL for a
caller's client, and RERANKER_API_KEY and RERANKER_MODEL as fallbacks for X-Reranker-* overrides.

`uvicorn app.main:app` works through a module `__getattr__`: every access to `app.main.app`
calls `create_app()` and returns a new app built from `get_settings()` (cached per process,
reads `.env` from the current directory, requires OPENAI_API_KEY), while a plain
`import app.main` builds nothing. Tests and tools therefore never touch `app.main.app`: they call
`create_app(make_settings(tmp_path))` (`tests/conftest.py`) and use `TestClient` as a
context manager, because the lifespan only runs inside `with`. Before it runs, `app.state`
holds only `settings` and `tracer`, and `/ready` answers 503 with `"startup": "failed"`. The
module-level `router` is shared by every app in the process, so test-only routes go on a
fresh `APIRouter`.

## Request lifecycles

Upload - `POST /upload`, `upload_document` then `_store_and_index`:
1. FastAPI parses the multipart body, then `require_api_key` (401), then `USER_ID_QUERY` (422).
2. `safe_filename` (400 "Invalid filename"), extension in `SUPPORTED_EXTENSIONS` (400), size against MAX_FILE_SIZE before and after reading (400), empty file (400).
3. `file_hash = sha256(contents)[:16]`; everything below runs inside `state.upload_locks.for_owner(user_id)`.
4. Dedup: `embeddings.has_file_hash(file_hash, user_id)` true means 200 with `duplicate=True, chunks=0` and an activity touch.
5. Quota: `file_hashes_for_source(safe_name, user_id)` are superseded and excluded; `_owner_usage`, then `_quota_problem` gives 413.
6. Write `UPLOAD_DIR/<user_id>/<file_hash>_<safe_name>` (OSError gives 503 "Storage unavailable").
7. `loader.load_document` and `chunker.split_documents`; a failure or zero chunks gives 400 and unlinks the file.
8. `extract_document_metadata(safe_name, first chunk text)`; every chunk gets `source`, `file_hash`, `user_id` and the extracted tags; ids are `<user_id>-<file_hash>-<i>`.
9. `embeddings.add_documents(chunks, ids=ids)`, then `bm25_index.add_documents(chunks, owner=user_id)`; any error gives 503 "Vector store unavailable" and unlinks the file.
10. `_retire_older_revisions`: other hashes stored under the same `source` for this owner leave Chroma, BM25 and disk (`storage.remove_stored_file`); failures are only logged; `replaced` is true when one was removed.
11. `activity.touch(user_id)` and an `UploadResponse`.

Query - `POST /query`, `query_rag`:
1. `QueryRequest` (422); `_caller_model` and `_caller_reranker` (400 for bad BYOK headers); `byok.client_for` only when X-Model-Key is sent.
2. `tracer.start_trace(request_id=..., user_id=..., ...)`.
3. `rag_chain.ask`, which calls `RAGChain._prepare`: refuse a falsy user_id; `extract_query_metadata` (if METADATA_FILTERING_ENABLED) overlaid by `metadata_filter`; `build_chroma_filter`; `_condense` a follow-up; `_retrieve` (dense with threshold or MMR, then, with HYBRID_SEARCH_ENABLED, `HybridRetriever` with BM25 and RRF); a tenant-wide retry when a metadata filter found nothing; rerank or cut to TOP_K_RESULTS; prompt with `rule_for(language)`.
4. No chunks: "No relevant information found." with no model call. Otherwise `chat.completions.create`, `_log_usage`, `[n]` markers stripped.
5. `RateLimitError` 429 with Retry-After 20 (400 on a caller key), `APITimeoutError` 504, a refused caller key 400, anything else 503 "Query failed"; `byok.close_quietly(client)` in `finally`.
6. `activity.touch(user_id)` (also for owners with no documents); `response_model=QueryResponse` filters the dict.

Stream - `POST /query/stream`, `query_rag_stream`:
1. Same setup as /query, then `stream = rag_chain.ask_stream(...)`; nothing has run yet.
2. Priming: `next(stream)` runs `_prepare` and opens the model stream (`create(..., stream=True)`) before the first yield, so those failures still map to 429/504/400/503 instead of a broken 200.
3. `activity.touch`, then `StreamingResponse(events())` with `text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`.
4. Events are `data: <json>` lines: one `sources`, `token` events (`CitationStripper` removes markers split across chunks), then `done`; an exception after the start becomes a final `{"type": "error", "detail": "The answer was cut short."}`.
5. `events()` closes the caller's client in `finally`. The 504 priming branch does not close it (see KNOWN_ISSUES.md, section Models, keys and reranking).

Documents - `GET /documents` (`list_documents`) and `DELETE /documents/{file_hash}` (`delete_document`):
- List: `embeddings.list_documents(user_id)` folds chunks by `file_hash` into `{file_hash, source, type, pages, chunks}`, plus `_quota_usage`; a store error is 503; activity is touched only when the list is non-empty.
- Delete: `FILE_HASH_PATH` (16 lowercase hex); owner-scoped `delete_by_file_hash` on Chroma and BM25; nothing removed is 404 "No such document"; then `storage.remove_stored_file` and an activity touch. It does not take the owner lock.

Clear - `POST /clear`, `clear_documents` then `_clear_namespace`: under the owner lock,
`storage.wipe_namespace` (Chroma delete where `{"user_id": user_id}`, rmtree of the owner
directory, `activity.forget`) and then `bm25_index.clear_owner`. Any error is 500 "Failed to
clear documents".

Feedback - `POST /feedback`, `submit_feedback`: `state.feedback` None (FEEDBACK_ENABLED=false)
is 404; `FeedbackStore.record` appends one JSON line (`at` plus the request fields) to
`FEEDBACK_DIR/feedback.jsonl`, creating it on the first rating; past FEEDBACK_MAX_BYTES is
507, an OSError 503; then `tracer.record_feedback` (a Langfuse score keyed by `request_id`)
and an activity touch. With Langfuse on, a rating without `request_id` is scored on the last
trace the process created (see KNOWN_ISSUES.md, section Tracing and observability).

Sweep - `POST /maintenance/sweep`, `sweep_idle_namespaces`:
1. Query params `idle_days` (1-3650), `prefix` (required; "" means every owner), `apply`, `force`.
2. `sweep.plan` never deletes: owners come from `embeddings.list_owners()` (the one unscoped scan), owner directories and markers; each is dated by `activity.last_seen`; it lists candidates, empty, unknown, foreign and orphan files, and records a refusal for apply (no prefix, idle_days under 7, failed activity writes, nobody active) unless `force`. A store error is `SweepError`, answered 503.
3. With `apply` and no refusal, `sweep.execute`: per owner, under its lock, re-check `last_seen`, then `storage.wipe_namespace` - which does not touch BM25; then orphan files inside the prefix.
4. `sweep.report` builds the `SweepResponse`. `python -m scripts.sweep` is only an HTTP client of this endpoint, so the backend must be running.

Startup and restore: every start creates UPLOAD_DIR, gives each owner directory that has no
marker one dated now (`ActivityTracker.seed_missing`, so owners from before markers existed
are not swept at once), opens or creates the collection and checks its recorded embedding
model, and starts with an empty BM25 index; `feedback.jsonl` is not created. `restore_backup`
(`app/backup.py`) calls `ActivityTracker.reset_all()` so restored owners do not look idle.

## Data layout and identifiers

| Data | Location (defaults from `app/config.py`) | Notes |
| --- | --- | --- |
| Vectors | CHROMA_PERSIST_DIR (`./data/chroma_db`), collection COLLECTION_NAME (`documents`) | One collection for all tenants. Its metadata records `embedding_model`, which must equal EMBEDDING_MODEL or startup refuses. |
| Raw uploads | UPLOAD_DIR (`data/uploads`), `<user_id>/<file_hash>_<safe name>` | Quota bytes count only files whose hash backs a listed document. |
| Activity markers | `UPLOAD_DIR/.activity/<user_id>` | Empty files; `last_seen` is the newer of the marker's mtime and the owner's newest upload. `.activity` can never be a user_id. |
| Ratings | FEEDBACK_DIR (`data/feedback`), `feedback.jsonl` | Append-only JSONL, never rotated by the app. |
| BM25 index | process memory | Not persisted, not in backups. |
| Compose | named volumes `data_uploads`, `data_chroma`, `data_feedback` | Mounted at /app/data/uploads, /app/data/chroma_db, /app/data/feedback. |

- user_id: `^[A-Za-z0-9_-]{1,64}$`, used raw as a directory name, chunk-id prefix and filter value.
- file_hash: the first 16 hex characters of the sha256 of the uploaded bytes. The public handle of a document; identical files have the same hash for every owner.
- Chunk id: `<user_id>-<file_hash>-<i>`. Chroma upserts on an existing id.
- Request id: `X-Request-ID`. A caller value matching `[A-Za-z0-9._-]{1,64}` is kept, otherwise 16 hex characters are minted. It is echoed in the response header, stamped on every log line, put in 500 bodies, and used as the Langfuse trace id.
- Chunk metadata on every chunk: `source` (sanitised filename), `file_hash`, `user_id`, `type`, `file_path` (path of the stored upload), `chunk_id`, `chunk_size`, `total_chunks`, `contains_table`, `is_table`, plus `company`, `year`, `quarter` and `doc_type` when `extract_document_metadata` finds them. PDF adds `page` (1-based) and `total_pages`; TXT and Markdown add `char_count` and `encoding`; DOCX adds `char_count`, `paragraphs` and `tables`. Values must be str, int, float or bool: Chroma rejects None and lists, and the upload then answers 503.

## Tenant model and trust boundary

- Authentication is one shared secret: X-API-Key against BACKEND_API_KEY, compared as bytes with `secrets.compare_digest`. An empty BACKEND_API_KEY disables auth (startup warning); the `.env.template` placeholder keeps auth on with a warning.
- user_id is not authenticated. Whoever holds the secret can upload, read and delete as any user_id; validation only makes the value safe as a path, chunk-id prefix and filter. Treat the secret as backend-wide access.
- The web UI mints `web-<uuid4 hex>` per browser session (a reload starts a new, empty namespace; the old one waits for the sweep, whose CLI default prefix is `web-`). The bot uses `str(update.effective_user.id)`: digits only and stable, so `web-` never matches it. `evaluation.run_eval` uses a scratch `eval-<12 hex>` tenant.
- Compose binds the backend to 127.0.0.1:8000. The Streamlit UI on port 8501 is published on all interfaces, has no login and holds the secret. `/docs`, `/redoc` and `/openapi.json` are served without a key.
- Tenant separation lives in three places: Chroma `where` filters (`_owned`, `build_chroma_filter`, and a bare `{"user_id": owner}` in `storage.wipe_namespace` and `BM25Index.sync_from_chroma`), BM25's per-owner document sets (its term statistics are shared by all tenants, see KNOWN_ISSUES.md, section Retrieval and search), and the per-owner upload directory. The striped owner lock only serialises one owner's changes.

## Invariants

1. Blocking handlers are plain `def`; only `/health` is `async def` - an async handler that blocks freezes every request, `/health` included; tests pin only `upload_document`, `query_rag`, `clear_documents`, `/ready`, `/feedback` and the sweep endpoint. Details: rag-api-endpoints.
2. Authenticated routes go on the module-level `router`; only probes go on the app object inside `create_app` - auth is a router dependency, so anything on the app is public. Details: rag-api-endpoints.
3. The user_id rule exists in four copies that must agree: `USER_ID_QUERY` (`app/main.py`), `USER_ID_FIELD` (`app/models/schemas.py`), `OWNER_NAME` (`app/activity.py`), `_USER_ID_PATTERN` (`app/observability.py`) - the value is used raw in paths, ids and filters. Details: rag-api-endpoints.
4. Tenancy helpers: `EmbeddingsManager._owned(owner, ...)` is only for EmbeddingsManager's own collection get/query/delete calls; `RAGChain` and `HybridRetriever` take a filter whose top level is `{"user_id": owner}` or the `$and` clause built by `build_chroma_filter(user_id, metadata)` in `app/rag/metadata_extractor.py` - an `_owned`-shaped filter there crashes (TypeError with hybrid on), and `BM25Index.search` with a falsy owner searches every tenant. Details: rag-retrieval-pipeline.
5. Every change to what an owner holds runs inside `state.upload_locks.for_owner(user_id)` (single-document DELETE currently does not) - quota is check-then-write and the sweep re-checks activity under the lock; the locks are non-reentrant and striped, so never nest them or call `_clear_namespace` while holding one. Details: rag-api-endpoints.
6. Chroma, BM25 and disk change together on every add or delete (the applied sweep is the known gap, below), and a new revision is indexed before the old one is retired - hybrid search answers from BM25, and a failed upload must not leave the owner with neither version. Details: rag-change-review.
7. Components get configuration only from Settings, passed in `lifespan` or read from `request.app.state.settings` - pydantic-settings reads `.env` without exporting it, so `os.getenv` misses values set only there. Details: rag-config-and-docs.
8. Call storage helpers through the module (`storage.wipe_namespace(...)`), never import them by name - tests patch `app.storage.<fn>` once and every caller must see it. Details: rag-testing.
9. HTTP error details are fixed strings or messages written for the caller (`BringYourOwnKeyError`), never raw exception text - exceptions can carry paths or SQL, and `/ready` is public. Details: rag-api-endpoints.
10. Prompt and request changes go in `RAGChain._prepare`, shared by `ask` and `ask_stream`, and the stream handler primes the generator - otherwise /query and /query/stream drift, or upstream failures arrive as a 200 with a broken stream. Details: rag-retrieval-pipeline.
11. Run exactly one backend process (no `--workers`, no replicas) - owner locks and BM25 live in process memory and Chroma needs a single writer; backup and restore run with the backend stopped, the sweep runs inside it. Details: rag-operations.
12. A caller's own model key lives for one request, its client is closed on every path (the stream's 504 branch is the known gap) and it is never stored or logged; indexing always uses the operator's key, because one collection is bound to one embedding model. Details: rag-llm-providers.

## BM25 lifecycle gaps

`state.bm25_index` lives only in process memory: it starts empty on every start, every
upload feeds it (even with HYBRID_SEARCH_ENABLED=false), and revision retirement, single
delete and /clear keep it in step with Chroma. Two gaps remain: an applied sweep
(`sweep.execute` calls `storage.wipe_namespace`, which skips BM25) leaves the swept
owner's chunks retrievable until restart, and the lazy rebuild from Chroma
(`HybridRetriever.retrieve` calling `sync_from_chroma`) runs only for an owner with zero
BM25 entries, so an upload before the first question after a restart leaves the older
documents out of keyword matching. Separately, a metadata filter (the `$and` shape) makes
BM25 contribute nothing unless the tenant-wide fallback runs. See KNOWN_ISSUES.md, sections
Operations: backup, restore and sweep (the sweep gap) and Retrieval and search (the other
two). A fix for the sweep gap belongs in one purge path shared by /clear and
the sweep; whichever gap you touch, assert `bm25_index.count_owner(owner)` in a test of that
mutation path.

## Where to change what

| Task | Files | Skill |
| --- | --- | --- |
| Add or change an endpoint, a request or response field, a status code | `app/main.py`, `app/models/schemas.py`, `clients/backend.py`, `DOCUMENTATION.md` | rag-api-endpoints |
| Add or change a setting or its default | `app/config.py`, `.env.template`, `DOCUMENTATION.md`, `app/main.py` (lifespan) | rag-config-and-docs |
| Retrieval: threshold, MMR, hybrid BM25 and RRF, reranking and reranker providers, prompt, answer languages | `app/rag/chain.py`, `app/rag/embeddings.py`, `app/rag/bm25.py`, `app/rag/hybrid.py`, `app/rag/reranker.py`, `app/rag/languages.py` | rag-retrieval-pipeline |
| How a metadata filter is applied at question time: merged with `metadata_filter`, the zero-hit fallback, MMR and BM25 under a filter | `app/rag/chain.py` (`RAGChain._prepare`, `_retrieve`), `app/rag/hybrid.py` | rag-retrieval-pipeline |
| File formats, parsing, encodings, chunking; the metadata extractor's rules for uploads and questions (aliases, years, doc types, `build_chroma_filter`) | `app/rag/document_loader.py`, `app/rag/text_splitter.py`, `app/rag/table_splitter.py`, `app/rag/metadata_extractor.py` | rag-document-ingestion |
| Server LLM endpoint, BYOK LLM providers and model aliases, who may send reranker headers | `app/byok.py`, `app/main.py` (lifespan), `app/rag/reranker.py` | rag-llm-providers |
| Web UI or Telegram bot behaviour | `frontend/streamlit_app.py`, `clients/telegram_bot.py`, `clients/backend.py` | rag-clients |
| Quotas and owner locks in handlers, deleting or replacing an owner's data | `app/main.py` (`OwnerLocks`, `_owner_usage`, `_quota_problem`, `_clear_namespace`), `app/storage.py` | rag-api-endpoints |
| Idle sweep, activity markers, backup and restore, the feedback log | `app/sweep.py`, `app/activity.py`, `app/backup.py`, `app/feedback.py`, `scripts/` | rag-operations |
| Request ids, logging, /health and /ready, Langfuse tracing | `app/observability.py`, `app/tracing.py`, `docker-compose.langfuse.yml` | rag-operations |
| Measure retrieval quality, golden cases, benchmarks | `evaluation/`, `scripts/download_benchmarks.py`, `scripts/upload_to_langfuse.py`, `tests/test_retrieval_quality.py` | rag-evaluation |
| Tests, fixtures and fakes | `tests/conftest.py`, `tests/test_<area>.py` | rag-testing |
| Setup, running services, Docker, CI, dependencies, commits | `Dockerfile`, `docker-compose.yml`, `docker-compose.override.yml`, `requirements.txt`, `.github/workflows/ci.yml` | rag-development-guide |
| README, DOCUMENTATION, KNOWN_ISSUES, CLAUDE.md and these skills | `README.md`, `DOCUMENTATION.md`, `KNOWN_ISSUES.md`, `CLAUDE.md`, `.claude/skills/` | rag-config-and-docs |
| A feature spanning backend, both clients and docs; moving code out of `app/main.py` | the recipes below | this skill, then each area's |
| Review a diff before commit or merge | the diff | rag-change-review |

## Recipes for cross-module changes

### Ship a user-facing feature end to end

The usual shape here is one pull request with backend, both clients, docs and tests; each step
names the skill with the detail.

1. Domain logic in its own `app/<feature>.py`: no FastAPI imports, its own exception class (as
   `SweepError` and `BackupError` are), no module-level import of `app.main`. Build it in
   `lifespan` onto `app.state`, configured from Settings (a new field: rag-config-and-docs).
2. Route and models in `app/main.py` and `app/models/schemas.py`: plain `def`, the owner lock
   around any change to what an owner holds, the status-code conventions, and a
   ``### `METHOD /path` `` heading in DOCUMENTATION.md (rag-api-endpoints).
3. Per-owner state follows every mutation path: upload, revision retirement, single delete,
   `/clear` and the sweep (purge it in `app/storage.py` `wipe_namespace`, which both reach),
   backup and restore (`app/backup.py` `storage_locations`), and the activity marker.
4. A new data directory: the checklist in rag-development-guide's Docker reference (path setting,
   `make_settings` override, a volume in both Compose files, `.gitignore`, `.dockerignore`,
   the Dockerfile `mkdir`).
5. Clients (rag-clients): shared non-transport pieces in `clients/backend.py`, the control in
   `frontend/streamlit_app.py`, a command in `clients/telegram_bot.py` with `BOT_COMMANDS` and `/help`.
6. Docs: a README.md Key Features bullet (no `/query` or `/upload` token), a DOCUMENTATION.md
   section, and any CLAUDE.md line or skill the change makes untrue.
7. Tests: a new `tests/test_<feature>.py` for a sizeable feature (small changes extend the module
   that already covers the area), its docstring naming what it guards, every write under
   `tmp_path` (rag-testing); mutation-check each new rule. Then Verify below and rag-change-review.

### Extract a module out of `app/main.py`

1. Move a cohesive group of helpers into `app/<name>.py`. It must not import `app.main` at module
   level (a cycle: `app.main` imports it) and raises its own exception, which the calling
   handler maps to a status.
2. Keep routes, `create_app` and `lifespan` in `app/main.py`: the API-reference test reads route
   decorators from that file only, so a route moved elsewhere silently loses its check.
3. Call the moved functions through the module (`storage.wipe_namespace(...)`), never import them
   by name, so one `monkeypatch.setattr` on the module reaches every caller.
4. Repoint the tests: a patch of a name that left `app.main` fails loudly (keep `raising=True`),
   and a `caplog.at_level` naming the `app.main` logger must name the new module. Never delete or
   loosen a test to get green.
5. Update [references/modules.md](references/modules.md), every skill Map naming the moved
   symbols (`grep -rn "<symbol>" .claude/skills CLAUDE.md`) and DOCUMENTATION.md's Directory Structure.
6. `python -c "import app.main"`, the ruff command and the full suite.

## Verify

Run from the repository root in the project's Python 3.10 environment (setup per OS,
including a Linux venv under WSL, is in rag-development-guide):

```bash
python -m pytest tests/test_api.py tests/test_resilience.py tests/test_observability.py tests/test_streaming.py tests/test_query_e2e.py tests/test_document_management.py tests/test_quotas.py tests/test_sweep.py tests/test_config.py -q
python -m pytest tests/test_docs_consistency.py tests/test_agent_docs.py -q
ruff check app frontend clients evaluation scripts tests
```

To check a wiring fact, write a test with the `api` fixture from `tests/conftest.py`: it runs
the real lifespan over `tmp_path` with fake embeddings and a fake chat client, so
`api.app_state` can be inspected offline. Starting uvicorn instead reads `.env`, writes to
the configured data directories and calls the real embedding API on the first upload.

## Related skills

- rag-api-endpoints - routes, schemas, auth, owner locks and quotas in handlers, status codes, SSE, error mapping.
- rag-retrieval-pipeline - `RAGChain` from question to answer, including the known retrieval defects.
- rag-document-ingestion - loaders, chunking, table-aware splitting, document metadata.
- rag-llm-providers - server LLM endpoint, BYOK headers and providers, reranker headers.
- rag-clients - Streamlit UI, Telegram bot, `clients/backend.py`.
- rag-config-and-docs - Settings end to end and the doc-drift tests.
- rag-operations - backup, restore, sweep, quotas, logs, /ready, Langfuse stack.
- rag-testing - fixtures, fakes, which tests to run.
- rag-evaluation - offline metrics, live harness, benchmarks.
- rag-development-guide - setup, services, Docker, CI, dependencies, commits.
- rag-change-review - project review checklist and recurring bug classes.
