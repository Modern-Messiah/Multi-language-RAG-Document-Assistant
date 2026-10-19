---
name: rag-api-endpoints
description: Changes the HTTP API in app/main.py and app/models/schemas.py - adding or modifying routes and request or response fields, X-API-Key auth, user_id validation, owner locks and quotas, status-code conventions, SSE streaming, request ids and error mapping, plus the matching client and API-reference updates. Use when adding an endpoint or parameter, changing a response or error, or touching the upload, query, documents, feedback or sweep handlers.
---

# HTTP API: routes, schemas and their contract

Every route, the auth dependency, the owner locks, the quota helpers, the SSE framing and the error mapping
live in `app/main.py`; every request and response model lives in `app/models/schemas.py`. This skill is the
contract an endpoint change has to keep, plus the clients, tests and docs that move with it. Before editing
a route, read its entry in [references/routes.md](references/routes.md): handler, auth, lock, side effects,
activity touch, every status code and detail, covering tests.

## Use this skill for / not for

Use it for: a new route, parameter, header or response field; a changed status code or error detail; edits
to `upload_document`, `query_rag`, `query_rag_stream`, `clear_documents`, `list_documents`,
`delete_document`, `submit_feedback`, `sweep_idle_namespaces`, `health` or `ready`; auth, user_id
validation, `OwnerLocks`, quotas, SSE, request ids, the catch-all 500.

Not for:
- What `RAGChain` does between handler and answer (filters, hybrid search, reranking, prompt, citation
  stripping): rag-retrieval-pipeline. Loaders, chunking and metadata behind uploads: rag-document-ingestion.
- BYOK and reranker header rules, `PROVIDERS`, refusal wording in `app/byok.py`: rag-llm-providers.
- Streamlit and Telegram code: rag-clients. New settings and the doc-drift tests: rag-config-and-docs.
- Running the sweep, backups, tracing a request id through the logs, Langfuse: rag-operations. Fixtures and
  fakes: rag-testing. Where the API sits in the system: rag-architecture.

## Map

| File | Responsibility | Key symbols |
| --- | --- | --- |
| `app/main.py` | routes, auth, locks, quotas, SSE, error mapping | `router`, `require_api_key`, `PLACEHOLDER_API_KEYS`, `create_app`, `USER_ID_QUERY`, `FILE_HASH_PATH`, `safe_filename`, `OwnerLocks`, `_store_and_index`, `_retire_older_revisions`, `_owner_usage`, `_quota_problem`, `_quota_usage`, `_caller_model`, `_caller_reranker`, `_clear_namespace`, `_sse`, `unhandled_error` |
| `app/models/schemas.py` | request and response models | `USER_ID_FIELD`, `QueryRequest`, `ChatTurn`, `QueryResponse`, `Source`, `UploadResponse`, `DocumentListResponse`, `DocumentSummary`, `QuotaUsage`, `DeleteResponse`, `ClearResponse`, `FeedbackRequest`, `FeedbackResponse`, `SweepResponse` |
| `app/observability.py` | request ids, access line, readiness | `RequestContextMiddleware`, `request_id_of`, `REQUEST_ID_HEADER`, `readiness`, `_QUIET_PATHS` |
| `app/storage.py` | an owner's stored files and the namespace wipe (vectors, files, marker); call as `storage.<fn>` so one test patch reaches every caller | `stored_files`, `size_of`, `remove_stored_file`, `wipe_namespace` |
| `app/activity.py` | last-seen markers the sweep reads | `ActivityTracker.touch`, `ActivityTracker.forget`, `OWNER_NAME` |
| `app/sweep.py`, `app/feedback.py` | logic behind the sweep and feedback routes (no HTTP in them) | `plan`, `execute`, `report`, `SweepError`; `FeedbackStore.record`, `FeedbackStorageFull` |
| `app/byok.py` | caller key, model and reranker headers | `wanted`, `wanted_reranker`, `client_for`, `describe_upstream_refusal`, `close_quietly`, `BringYourOwnKeyError` |
| `app/humanize.py` | sizes in 400/413 text and client usage lines | `human_size`, `describe_quota` |
| `clients/backend.py` | what both clients do with a status and detail | `describe_error`, `error_from_response`, `api_headers`, `OPERATOR_ERROR` |

## Request pipeline and auth

1. `RequestContextMiddleware` (raw ASGI) assigns the request id and echoes it on every response.
2. FastAPI reads the body before any dependency: malformed JSON is 422 even with a wrong key, and a
   multipart upload is parsed and spooled before its 401 (cap body size at the proxy).
3. `require_api_key`, the dependency of the module-level `router`, answers 401 `Invalid or missing API key`
   before field errors are raised (a bad user_id with a wrong key is 401). Then 422 for invalid query,
   path or body fields, with `detail` as a list of pydantic error objects.
4. The handler (plain `def`, FastAPI threadpool) reads its components from `request.app.state`, takes the
   owner lock if it changes what the owner holds (upload and clear; delete takes none), raises
   `HTTPException` with a fixed detail on failure, and touches the activity marker on success where the
   route counts as activity. The middleware writes one access line when the response, or the whole
   stream, has ended.

- Every `@router.<verb>(...)` route needs `X-API-Key`. Anything registered on `application` inside
  `create_app` is public, and only the probes belong there: `GET /health` (`async def`, liveness,
  `{"status": "ok", "version": API_VERSION}`; the CI image smoke test greps it) and `GET /ready` (`def`;
  200 or 503 naming checks, never reasons; the Compose healthcheck probes it). FastAPI's `/docs`, `/redoc`
  and `/openapi.json` are public too; `create_app` does not disable them.
- An empty BACKEND_API_KEY disables auth (startup warning). A key in `PLACEHOLDER_API_KEYS` (the current and
  a former `.env.template` value) only logs a warning: auth stays on with a publicly known key. The check
  is `secrets.compare_digest` on UTF-8 bytes; `Settings` refuses a non-ASCII key at startup.
- `unhandled_error` answers 500 `{"detail": "Internal server error", "request_id": "<id>"}` and sets
  `X-Request-ID` itself (it runs outside the middleware); the exception still reaches the server log. An
  `HTTPException` body is only `{"detail": ...}`, with the id in the header.

## Handler conventions

- **Plain `def` for blocking work** (Chroma, disk, a model call, parsing): an `async def` that never awaits
  runs on the event loop, and one slow upload once froze every request in the process, probes included.
  Only `health` is async. Pinned by
  `tests/test_resilience.py::test_request_handlers_are_not_coroutines` (upload_document, query_rag,
  clear_documents) and one test each in `tests/test_observability.py` (`/ready`), `tests/test_feedback.py`
  and `tests/test_sweep.py`; `query_rag_stream`, `list_documents` and `delete_document` are unpinned. Add
  every new handler to that parametrize list.
- **Validated owner**: `user_id: str = USER_ID_QUERY` in the query string, `USER_ID_FIELD` in a body (1-64
  of `[A-Za-z0-9_-]`); a document handle is `file_hash: str = FILE_HASH_PATH` (16 lowercase hex, the start
  of the upload's sha256). user_id is used raw as a directory name, as the chunk-id prefix
  `<user_id>-<file_hash>-<i>` and as a Chroma filter value.
- **Filenames** go through `safe_filename` first: basename (backslashes count as separators), characters
  outside `[\w. -]` become `_` (Unicode letters stay), stem cut to 150 UTF-8 bytes and extension (with its
  dot) to 10 characters; empty names, dotfiles and names of only `.`, space and `_` are 400
  `Invalid filename`.
- **Fixed error text**: `logger.exception(...)`, then `HTTPException` with a constant detail. Never put
  exception text, provider wording, paths or keys in a detail: details reach end users verbatim. The one
  exception is the operator-only sweep report (`failed[].error`, cut to 200 characters).
- **Caller headers first**: `_caller_model` and `_caller_reranker` (400 on `byok.BringYourOwnKeyError`)
  run before `byok.client_for`, and a caller client is closed with `byok.close_quietly` on every way out.
  `tracer.start_trace` runs between building that client and the `try`; that is safe only because every
  tracer method swallows its own errors, so keep it that way.
- **Config from state**: read `request.app.state.settings`, never `get_settings()` or `os.getenv` in a
  handler; tests build apps with their own settings.

## Owner locks

- `state.upload_locks` is an `OwnerLocks`: 64 `threading.Lock` stripes picked by `sha256(user_id)[0] % 64`,
  so memory stays bounded while the web UI mints an owner per browser session, at the price of unrelated
  owners sometimes sharing a stripe.
- Taken by `upload_document` (around all of `_store_and_index`), `_clear_namespace` (`POST /clear`) and
  `sweep.execute` (per swept namespace and per orphan cleanup). Not taken by `delete_document`,
  `list_documents`, the query routes, `submit_feedback`, or the lazy BM25 sync in `HybridRetriever.retrieve`.
- Non-reentrant and shared: never acquire it twice in one thread, never hold two owners' locks (they may be
  one stripe), never call a lock-taking helper such as `_clear_namespace` while holding it. That is why
  `sweep.execute` calls `storage.wipe_namespace` (which expects the caller to hold the lock) directly;
  `app/sweep.py` also cannot import `app.main`, which imports it.
- Locks and `state.bm25_index` live in process memory: one backend process only (Compose runs a single
  uvicorn without `--workers`).

## Quotas in the upload path

Inside the lock `_store_and_index` runs: dedup, quota, write, parse and chunk, index (Chroma, then BM25),
retire older revisions, touch. After the write, the parse, no-text and index failures unlink the new file.
- Dedup comes first (`embeddings.has_file_hash(file_hash, user_id)`), so an identical re-upload answers
  `duplicate: true, chunks: 0` even at the limit. Earlier revisions of the same sanitized name
  (`embeddings.file_hashes_for_source`) are passed as `exclude_hashes`, so a new revision fits too.
- `_owner_usage(state, settings, user_id, exclude_hashes)` returns (documents, bytes): documents from
  `EmbeddingsManager.list_documents`, bytes only of stored files whose `<file_hash>_` prefix backs a listed
  document (orphan files are free; a file that vanished counts 0). `GET /documents` reports the same
  numbers through `_quota_usage`.
- `_quota_problem(settings, documents, total_bytes, adding_bytes, user_id)` returns the user text or None.
  MAX_DOCUMENTS_PER_USER (default 200) refuses when documents + 1 exceeds it, MAX_BYTES_PER_USER (default
  1073741824) when bytes + this upload exceeds it; 0 turns a limit off.
- 413 wording: the numbers (sizes via `human_size`) plus `Remove documents you no longer need, or ask the
  operator to raise the limit.`, never an env var name; the setting goes to the WARNING log. Pinned by
  `tests/test_quotas.py::test_the_message_says_what_to_do_and_names_no_setting` and
  `tests/test_quotas.py::test_the_log_names_the_setting`. 413 because both clients show its detail
  verbatim and hide 401/403 behind an operator message, and a 429 would mean "retry", which cannot help.
- MAX_FILE_SIZE (default 31457280) is a per-file 400 `File too large. Maximum allowed size is <size>.`,
  checked on `file.size` before reading and on the bytes after. A failed lookup is 503
  `Vector store unavailable`.

## Status codes

| Code | Used for | Detail |
| --- | --- | --- |
| 400 | caller-fixable input: upload name, type, size, empty or unreadable file, no text; BYOK or reranker header problems; the provider refused the caller's own key (its 429 included) | fixed text; BYOK text from `app/byok.py` |
| 401 | missing or wrong `X-API-Key` on a `router` route | `Invalid or missing API key` |
| 404 | `DELETE` of a hash this owner does not hold; `POST /feedback` with FEEDBACK_ENABLED false | `No such document`, `Feedback collection is disabled` |
| 413 | upload over a per-owner quota | numbers and remedy, no setting name |
| 422 | FastAPI validation: user_id, file_hash, body fields, sweep params, missing file part, malformed JSON | list of error objects |
| 429 | rate limit on the operator's key (`RateLimitError`, no caller key) | `Upstream model is rate limited. Please retry shortly.` plus `Retry-After: 20` |
| 500 | `POST /clear` failure; anything uncaught | `Failed to clear documents`; catch-all body with `request_id` |
| 503 | store, disk or model failure the caller cannot fix; `GET /ready` not ready | `Vector store unavailable`, `Storage unavailable`, `Query failed`, `Feedback storage unavailable` |
| 504 | `APITimeoutError` on either query route, whoever's key | `The model did not answer in time. Please retry.` |
| 507 | the rating would push the feedback file past FEEDBACK_MAX_BYTES | `Feedback storage is full. The operator needs to rotate it.` |

- Caller-key refusals are 400, never 401/403 (clients replace those with `OPERATOR_ERROR`) and never 429. In
  both query handlers a `RateLimitError` while a caller client exists, and any other exception that carries
  a `status_code` while one exists, become 400 worded by `byok.describe_upstream_refusal`. Exceptions
  without a status (connection errors) stay 503.
- The check looks only at "a caller key was sent" and "the exception has a status", not at which call
  failed: with `X-Model-Key` set, a 401 from the operator's own question embedding comes back as
  `The provider rejected the API key you sent.` (probe-verified). Mind this before widening the mapping.
- `clients/backend.py` `describe_error` shows a string detail verbatim, joins a 422 list into one line, and
  appends `(request <id>)` only for 401/403 and 5xx. So a new user-fixable error is a 4xx other than
  401/403 with a string detail, and a new operator-side failure is a logged 503.

## SSE: `POST /query/stream`

- Framing: `_sse(event)` writes `data: <json>` and a blank line (`ensure_ascii=False`); no `event:` or
  `id:` lines, the type lives in the JSON.
- Order: one `{"type": "sources", "sources": [...]}`, zero or more `{"type": "token", "text": "..."}`, then
  `{"type": "done"}`. Nothing retrieved: sources `[]`, one token `No relevant information found.`, done,
  no answering call (a follow-up with history has already made its condensing call). A failure after
  the first byte is logged (`Streaming failed after the response started`) and ends the stream with
  `{"type": "error", "detail": "The answer was cut short."}` and no done.
- Priming: the handler calls `next(stream)` inside the same except chain as `POST /query` before returning,
  and `RAGChain.ask_stream` runs `_prepare` and opens the completion (`stream=True`) before its first
  yield, so retrieval, reranking and opening the model stream still fail as 400/429/503/504. Work moved
  after the first yield turns those into a 200 plus an error event.
- Headers: `text/event-stream; charset=utf-8`, `Cache-Control: no-cache`, `X-Accel-Buffering: no` (nginx
  buffering would defeat streaming), `X-Request-ID`.
- Sources are the chain's raw dicts, so absent keys stay absent, while `POST /query`'s `response_model`
  fills `rerank_score` and `metadata` with null. `done` carries no model name.
- The caller's client is closed in the `finally` of the `events()` generator, after the last event or, on
  a client disconnect, when Python finalizes the abandoned generator (Starlette never closes a sync
  iterator); the priming error branches close it too, except the timeout one (known issue).
- Activity is touched once priming succeeds; the access line is written when the stream ends and says 200
  even after an error event. The web UI handles `token`, `sources` and `error` and ignores other types;
  the bot uses `POST /query`.

## Request ids, access log, activity

- The id is the caller's `X-Request-ID` if it matches `^[A-Za-z0-9._-]{1,64}$`, else 16 new hex
  characters; the middleware keeps it on the scope and in a ContextVar. In a handler use
  `request_id_of(request)` (scope first, then the ContextVar). It is the Langfuse trace id on both query
  routes, unchanged when the caller supplied it, and `FeedbackRequest.request_id` (`^[A-Za-z0-9._-]+$`, at
  most 64) scores that trace: keep the two charsets compatible.
- Access line: `METHOD /path -> STATUS in N ms (user=<id>)`. `user=` comes from the query string only, so
  routes with the owner in the body print `-` and an invalid value prints `invalid`. `GET /health` and
  `GET /ready` (`_QUIET_PATHS`) are logged only at 400 or above.
- Activity markers (`UPLOAD_DIR/.activity/<user_id>`) tell the sweep idle from active. Touched by upload
  (also a duplicate), both query routes (even for owners with no uploads), `GET /documents` only when the
  owner holds documents, delete and feedback; `POST /clear` and an applied sweep remove the marker.
  `touch` never raises; failures are counted and make the sweep refuse to apply.

## Schemas

- `QueryRequest`: `question` 1-4000 characters; `language` free string, default `Auto`, no length bound,
  unknown values fall back to the Auto rule; `history` at most 20 `ChatTurn`s (question 1-4000, answer
  1-8000; MAX_HISTORY_TURNS decides how many are used); `metadata_filter` an unvalidated
  `Optional[Dict[str, Any]]` (known issue).
- `QueryResponse`: `answer` (may be an empty string, which a client must not send back as a `ChatTurn`),
  `sources` (`Source`: `id`, `source`, `preview`, optional `rerank_score`, optional `metadata` with
  company, year, quarter, doc_type) and `model` (null when no model was asked).
- `FeedbackRequest` and the other bodies: field lists and bounds per route in
  [references/routes.md](references/routes.md).
- No model forbids extra keys: unknown JSON fields are dropped silently, so a misspelled field is ignored
  instead of a 422, and an older backend ignores a newer client's field.
- `response_model` filters handler output: a key the chain returns that `QueryResponse` or `Source` lacks is
  dropped. `POST /query/stream` has no response model. `ErrorResponse` is unused and its docstring is wrong
  (a 422 detail is a list; the catch-all adds `request_id`).
- Bound every new string and list with `max_length`, give optional fields defaults, and import every typing
  name you use (a missing import once broke module import; ruff F821 catches it).

## Recipes

### Add an endpoint
1. Models in `app/models/schemas.py` (bounded fields, `USER_ID_FIELD` for a body owner); add them to the
   `from app.models.schemas import (...)` block of `app/main.py`.
2. A module-level `@router.<get|post|delete>("/path", response_model=Model)` handler, path as a double-quoted
   literal right after the opening parenthesis (the docs test's regex sees only that form; single quotes, a
   line break before the path or `add_api_route` slip past it),
   declared `def`, taking `request: Request` and `USER_ID_QUERY` / `FILE_HASH_PATH` where they apply.
3. Tenancy: EmbeddingsManager lookups and deletes take `owner`; new chunks carry `user_id` in metadata;
   retrieval filters come only from `build_chroma_filter` (rag-retrieval-pipeline); BM25 calls pass the
   owner. A route that changes what the owner holds follows "Add or change a mutation path".
4. Errors from the status table; BYOK headers through `_caller_model` / `_caller_reranker`;
   `state.activity.touch(user_id)` on success if the route means the owner is active.
5. Clients: shared non-transport pieces (re-exported header constants, error wording, env parsing) go in
   `clients/backend.py` and its `__all__`; the call itself in `frontend/streamlit_app.py` (`requests`,
   `HEADERS`, `backend_error`) and `clients/telegram_bot.py` (`httpx.AsyncClient`, `BACKEND_HEADERS`,
   `backend_error`). Details and bot-command rules: rag-clients.
6. Tests in `tests/test_<feature>.py` with the `api` fixture: the 200 path; 401 with
   `headers={"X-API-Key": "wrong"}`; 422 for user_id `../x` and `"x" * 65`; two owners kept apart; error
   mapping via `monkeypatch.setattr(api.app_state.embeddings, "<method>", boom)`; the handler name in the
   not-coroutines parametrize list.
7. DOCUMENTATION.md: a heading ``### `METHOD /path` `` (path exactly as in the decorator) in the API
   reference with auth, parameters, headers, body fields and bounds, response JSON and every status code
   with its detail; `tests/test_docs_consistency.py` checks only the heading. README.md only if
   user-facing, and never with a `/query` or `/upload` token, URLs and `/query/stream` included (a README
   regex reads every `/word` as a bot command). Doc rules: rag-config-and-docs.

A public probe instead goes on `application` in `create_app` and joins `_QUIET_PATHS` if polled; a path that
`docker-compose.yml` probes needs **No authentication** within 400 characters after its DOCUMENTATION.md
heading.

### Add a request field or header
1. Give the field a default and bounds, so older clients still validate.
2. A `QueryRequest` field is passed from both `query_rag` and `query_rag_stream` into `RAGChain.ask` and
   `RAGChain.ask_stream`, which share `RAGChain._prepare` (rag-retrieval-pipeline). A `FeedbackRequest`
   field goes into the `store.record(...)` call in `submit_feedback`, which writes it verbatim to the file.
3. A header: define its constant next to its parser (as `app/byok.py` does), re-export it from
   `clients/backend.py`, validate it before any client is built and answer 400 on a bad value.
4. Test the round trip and one 422 per bound; update the endpoint's DOCUMENTATION.md section (no test
   notices a missing field).

### Change a response
1. `POST /query`: add the key to every return dict of `RAGChain.ask` (the no-context one too) and to
   `QueryResponse`. Per-source keys go in `RAGChain._collect_sources` and `Source` (the stream's sources
   event gets them too); a top-level key reaches the stream only if you add it to an event in
   `RAGChain.ask_stream` (`tests/test_streaming.py::test_both_paths_agree_on_sources`).
2. `POST /upload`: `UploadResponse` is built twice in `_store_and_index` (duplicate and success); set the
   field in both. `GET /documents` entries come from `EmbeddingsManager.list_documents`.
3. Stay additive. Readers: both clients, `scripts/sweep.py` (sweep report), `scripts/backup.py` and
   `scripts/restore.py` (the `GET /ready` probe), and `evaluation/run_eval.py`,
   `evaluation/langfuse_eval.py`, `evaluation/eval_sberquad_failures.py` (`sources[].source`). Clients read
   new keys with `.get()` (pattern: `app/humanize.py` `describe_quota`). Update the DOCUMENTATION.md JSON.

### Add or change a mutation path
1. Hold `state.upload_locks.for_owner(user_id)` across the whole check-then-write, with no nested
   lock-taking helper.
2. Change every store together: Chroma (`add_documents(chunks, ids=...)`, `delete_by_file_hash(file_hash,
   owner)`, `delete_documents(filter={"user_id": ...})`), the BM25 side index `state.bm25_index`
   (`add_documents(chunks, owner=...)`, `delete_by_file_hash`, `clear_owner`), disk
   (`storage.remove_stored_file`, `storage.wipe_namespace`) and the marker (`touch` or `forget`). The
   lifespan always builds `BM25Index` and every upload feeds it, even with HYBRID_SEARCH_ENABLED false.
   Prefer one shared helper: `POST /clear` and the sweep already drifted apart on BM25.
3. Index new data before retiring old data (`_retire_older_revisions` runs last); unlink a just-written
   file on any later failure.
4. Keep raw files at `UPLOAD_DIR/<user_id>/<file_hash>_<name>`: quotas, `ActivityTracker.last_seen`,
   backups and the sweep's orphan pass rely on it (an applied sweep deletes, inside its prefix, any file
   whose hash prefix backs no listed document). Backups cover only CHROMA_PERSIST_DIR, UPLOAD_DIR and
   FEEDBACK_DIR (`app/backup.py` `storage_locations`), and `sweep.plan` finds owners only in Chroma,
   upload directories and markers: per-owner state anywhere else needs backup, restore and sweep support
   (rag-operations).
5. Test that Chroma (`api.app_state.embeddings.list_documents(owner)`), disk (`api.upload_dir`) and BM25
   (`api.app_state.bm25_index.count_owner(owner)`) all changed and that another owner did not.

### Change the user_id rule
Change `USER_ID_PATTERN` / `USER_ID_QUERY` (`app/main.py`), `USER_ID_FIELD`, `OWNER_NAME`
(`app/activity.py`) and `_USER_ID_PATTERN` (`app/observability.py`) together, and keep it path-safe (no
`/`, backslash or leading `.`; `.activity` must stay impossible). The web UI mints `web-<uuid4 hex>`, bot ids
are Telegram user ids (digits), and the sweep defaults to prefix `web-`. Tests:
`tests/test_api.py::test_user_id_with_unsafe_characters_is_rejected`,
`tests/test_sweep.py::test_a_user_id_the_api_accepts_is_an_owner_name`.

## Pitfalls

- `app.main.app` is a lazy module attribute: every access builds a new app (settings from the cached
  `get_settings()`). Tools and tests call `create_app(settings)`.
- The module-level `router` is shared by every app built in the process: put test-only routes on a fresh
  `APIRouter` and `api.app.include_router(...)` it (pattern in `tests/test_observability.py`).
- Dedup ignores the filename: the same bytes under a new name answer `duplicate: true` echoing the new
  name, while the listing keeps the first. Re-chunking after a CHUNK_SIZE change needs delete + re-upload.
- Revision replacement matches the sanitized filename exactly and case-sensitively: `Report.txt` and
  `report.txt` are two documents.
- Any `add_documents` failure during upload, including an auth or quota error on the operator's embedding
  key, is 503 `Vector store unavailable`.
- `DELETE /documents/{file_hash}` calls `storage.remove_stored_file` outside its try: an OSError there is a
  catch-all 500 after the chunks and BM25 entries are gone (probe-verified). It also runs without the
  owner lock; a race with an identical re-upload is an unverified risk, so take the lock if you touch it.

## Known issues in this area

- An applied sweep removes vectors, files and markers but not the owners' BM25 entries, so swept
  documents stay retrievable until a restart or `POST /clear`. See KNOWN_ISSUES.md, section Operations:
  backup, restore and sweep.
- BM25 starts empty and re-syncs an owner from Chroma only while that owner has no BM25 entries: an
  upload before the first question after a restart keeps older documents out of keyword search. See
  KNOWN_ISSUES.md, section Retrieval and search.
- `metadata_filter` is unvalidated: a value ChromaDB rejects is 503 `Query failed` instead of 422, and a
  filter matching nothing silently falls back to the whole namespace. See KNOWN_ISSUES.md, section
  Retrieval and search.
- `query_rag_stream`'s `APITimeoutError` branch answers 504 without `byok.close_quietly(client)`; and
  `X-Reranker-Provider` needs no caller key, overrides RERANKER_ENABLED, and with `cohere` but no
  `X-Reranker-Key` spends RERANKER_API_KEY. See KNOWN_ISSUES.md, section Models, keys and reranking.
- With Langfuse on, `POST /feedback` without `request_id` scores whatever trace the process created last,
  and requests that reuse a caller `X-Request-ID` write into one trace; streamed answers record no token
  usage. See KNOWN_ISSUES.md, section Tracing and observability.
- A `.doc` upload gets the generic `Unsupported file type` 400 (the extension gate answers first), and a
  document that splits into more chunks than one ChromaDB batch fails with 503 after it was embedded. See
  KNOWN_ISSUES.md, section Document ingestion.
- `ChatTurn.answer` needs at least one character, but an answer can be empty; the web UI sends it back in
  `history` and every later question is a 422. See KNOWN_ISSUES.md, section Web UI (the bot has the same
  problem: section Telegram bot).

## Verify

```bash
python -m pytest tests/test_api.py tests/test_streaming.py tests/test_document_management.py \
  tests/test_upload_errors.py tests/test_quotas.py tests/test_observability.py \
  tests/test_resilience.py tests/test_docs_consistency.py -q
python -m pytest tests/test_byok.py tests/test_feedback.py tests/test_sweep.py \
  tests/test_query_e2e.py tests/test_clients_backend.py -q
ruff check app frontend clients evaluation scripts tests
```

The first command for any handler change, the second when BYOK, feedback, the sweep, query results or
client error wording are involved, the full `python -m pytest -q` before committing. Run from the repo root
with the project's Python 3.10 environment; the suite is offline and `tests/conftest.py` sets a dummy
OPENAI_API_KEY itself. WSL and Windows specifics: rag-testing.

## Related skills

rag-architecture (where the API sits), rag-retrieval-pipeline, rag-document-ingestion, rag-llm-providers,
rag-clients, rag-config-and-docs, rag-operations, rag-testing, rag-change-review (review checklist for these
rules).
