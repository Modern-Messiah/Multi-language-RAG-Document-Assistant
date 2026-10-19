# Route reference

One entry per route in `app/main.py`: handler, auth, inputs, response, owner lock, side effects,
activity touch, every status code with its exact detail, and the tests that cover it. Read the
entry before changing a route; update it when you change one.

Contents:
- Common to every route
- `GET /health`, `GET /ready` (public probes) and FastAPI's own docs routes
- `POST /upload`
- `POST /query`
- `POST /query/stream`
- `GET /documents`
- `DELETE /documents/{file_hash}`
- `POST /clear`
- `POST /feedback`
- `POST /maintenance/sweep`

## Common to every route

- Every response carries `X-Request-ID` (the caller's value if it matches `^[A-Za-z0-9._-]{1,64}$`,
  otherwise 16 new hex characters); one access line per request, except passing probes.
- `router` routes: 401 `Invalid or missing API key` without a valid `X-API-Key`, unless
  BACKEND_API_KEY is empty. Malformed JSON is 422 even before the key is checked; a valid body
  with bad fields and a wrong key is 401.
- 422 bodies are FastAPI's: `{"detail": [{"type", "loc", "msg", "input", ...}, ...]}`. Every other
  error body is `{"detail": "<fixed text>"}`, except the catch-all 500, which is
  `{"detail": "Internal server error", "request_id": "<id>"}`.
- A path that matches no route is FastAPI's 404 `{"detail": "Not Found"}`.

## `GET /health`, `GET /ready` (public probes)

`GET /health`
- Handler: `health`, the only `async def`, registered on `application` in `create_app`. No auth.
- 200 `{"status": "ok", "version": "<API_VERSION>"}`, also before startup has finished. Liveness
  only; the CI image smoke test greps it for `"status":"ok"`.
- Logged only when the status is 400 or above. Tests: `tests/test_api.py`,
  `tests/test_observability.py`.

`GET /ready`
- Handler: `ready` (`def`, because `readiness` reads ChromaDB), on `application`. No auth.
- 200 `{"status": "ready", "checks": {"startup": "ok", "vector_store": "ok"}}`; 503
  `{"status": "not ready", "checks": {...}}` with `failed` values and never a reason. Before the
  lifespan has run: 503 with only `{"startup": "failed"}`. OpenAI is deliberately not checked.
- The `docker-compose.yml` healthcheck probes it and the web UI and bot wait on it;
  `scripts/backup.py` and `scripts/restore.py` probe it to refuse running against a live backend.
  Its DOCUMENTATION.md section must say **No authentication** within 400 characters of the heading.
- Logged only when the status is 400 or above. Tests: `tests/test_observability.py`.

FastAPI's `/docs`, `/redoc` and `/openapi.json` are public too (FastAPI defaults left on). They have
no decorator in `app/main.py`, so the docs test asks no API-reference heading for them.

## `POST /upload`

- Handler: `upload_document` (`def`), then `_store_and_index` under the owner lock.
- Input: query `user_id` (`USER_ID_QUERY`); multipart field `file` (`File(...)`).
- Response: `UploadResponse` `{message, filename, chunks, duplicate, file_hash, replaced}`.
- Order: `safe_filename` -> extension in `SUPPORTED_EXTENSIONS` (case-insensitive) -> `file.size`
  against MAX_FILE_SIZE -> read -> empty check -> size backstop -> `file_hash` = sha256[:16] ->
  owner lock -> dedup -> quota -> write -> load and chunk -> metadata -> Chroma add -> BM25 add ->
  retire older revisions -> touch.
- Lock: `state.upload_locks.for_owner(user_id)` around everything after hashing.
- Side effects: writes `UPLOAD_DIR/<user_id>/<file_hash>_<safe name>`; Chroma `add_documents` with
  ids `<user_id>-<file_hash>-<i>` and metadata `source`, `file_hash`, `user_id` plus whichever of
  company, year, quarter and doc_type the extractor finds; `bm25_index.add_documents(chunks,
  owner=user_id)`; then `_retire_older_revisions` deletes other hashes stored under the same sanitized
  name from Chroma, BM25 and disk (its errors are only logged). A parse, no-text or index failure
  after the write unlinks the new file.
- Activity: touched on success and on a duplicate.
- Status:
  - 200 `Document processed successfully` (`duplicate: false`, `replaced` true when an earlier
    revision was retired) or `Document already indexed (identical content)` (`chunks: 0`,
    `duplicate: true`, the filename echoed is the new one).
  - 400 `Invalid filename`; `Unsupported file type. Supported: PDF, DOCX, TXT, MD, MARKDOWN`;
    `File too large. Maximum allowed size is 30 MB.` (the size from MAX_FILE_SIZE via `human_size`);
    `Uploaded file is empty`; `Could not read the document. It may be corrupted or empty.` (any
    loader or chunker exception); `Document contains no extractable text`.
  - 401; 422 for a missing or invalid `user_id` or no file part (an empty filename means no part).
  - 413 `Document limit reached: you hold <n> of <max>. Remove documents you no longer need, or ask
    the operator to raise the limit.` or `Storage limit reached: this upload would take you to
    <size> of <max size>. ...` with the same remedy.
  - 503 `Vector store unavailable` (dedup lookup, quota lookup, or indexing, including embedding
    failures on the operator's key); `Storage unavailable` (the file could not be written).
- Tests: `tests/test_api.py`, `tests/test_upload_errors.py`, `tests/test_quotas.py`,
  `tests/test_document_management.py`, `tests/test_docx_markdown.py`, `tests/test_sweep.py`
  (markers).

## `POST /query`

- Handler: `query_rag` (`def`). Optional headers: `X-Model-Key`, `X-Model`, `X-Model-Provider`,
  `X-Reranker-Provider`, `X-Reranker-Key`, `X-Reranker-Model` (rules: rag-llm-providers).
- Body `QueryRequest`: `question` 1-4000 characters; `language` (default `Auto`, no length bound,
  unknown values get the Auto rule); `user_id` (`USER_ID_FIELD`); `history` at most 20 `ChatTurn`s
  (`question` 1-4000, `answer` 1-8000; the chain uses the last MAX_HISTORY_TURNS); `metadata_filter`
  an unvalidated `Optional[Dict[str, Any]]` whose `user_id` key is dropped. Unknown fields are ignored.
- Response: `QueryResponse` `{answer, sources: [Source], model}`; with nothing retrieved the answer
  is `No relevant information found.`, `sources` is `[]`, `model` is null and no answer is generated
  (a follow-up with history has already made its condensing call).
- Lock: none. Retrieval reads Chroma and the in-memory BM25 index (the lazy BM25 sync takes no
  owner lock).
- Side effects: `tracer.start_trace` with the request id as trace id; question embedding on the
  operator's key; condensing (with history) and answering on the caller's client if a key was sent,
  else the server's; that client is closed in `finally`.
- Activity: touched on success, even for owners with no uploads.
- Status:
  - 200.
  - 400 header problems (`byok.BringYourOwnKeyError` text, raised before any client is built); a
    caller-key refusal: a `RateLimitError` with a caller client, or any exception with a
    `status_code` while a caller client exists (even a 401 from the operator's question
    embedding), worded by `byok.describe_upstream_refusal`.
  - 401; 422 (question bounds, history over 20 turns or with an empty turn, bad user_id).
  - 429 `Upstream model is rate limited. Please retry shortly.` with `Retry-After: 20` (operator's
    key only).
  - 503 `Query failed` (everything else, logged with its traceback; also a metadata_filter value
    ChromaDB rejects).
  - 504 `The model did not answer in time. Please retry.` (`APITimeoutError`, either key).
- Tests: `tests/test_query_e2e.py`, `tests/test_resilience.py`, `tests/test_byok.py`,
  `tests/test_api.py`, `tests/test_conversations.py` (history over the API), `tests/test_tracing.py`
  (trace started).

## `POST /query/stream`

- Handler: `query_rag_stream` (`def`, not pinned by the not-coroutines test). Body and headers as
  `POST /query`; no `response_model`.
- Response: `StreamingResponse` of `_sse` events, `text/event-stream; charset=utf-8`,
  `Cache-Control: no-cache`, `X-Accel-Buffering: no`. Events: `sources`, then `token`s, then `done`;
  or a final `{"type": "error", "detail": "The answer was cut short."}` after a mid-stream failure.
- Priming: `next(stream)` runs retrieval, reranking and opens the completion before the response
  starts, so those failures keep the `POST /query` status codes (400, 429 with `Retry-After`, 503,
  504).
- Lock: none. Side effects as `POST /query`; the caller's client is closed in the `events()`
  generator's `finally`, or in the priming error branches - except the `APITimeoutError` one
  (known issue). No token usage is logged for streamed answers (known issue).
- Activity: touched once priming has succeeded. The access line is written when the stream ends,
  with status 200 even if an error event was sent.
- Tests: `tests/test_streaming.py`, `tests/test_byok.py`, `tests/test_sweep.py` (markers).

## `GET /documents`

- Handler: `list_documents` (`def`, unpinned). Input: query `user_id`.
- Response: `DocumentListResponse` `{documents: [DocumentSummary], total_chunks, quota}`;
  `DocumentSummary` is `{file_hash, source, chunks, type, pages}` (pages for PDFs), sorted
  case-insensitively by source; `quota` is `QuotaUsage` `{documents, max_documents, bytes,
  max_bytes}` from `_quota_usage`, where a max of 0 means that limit is off.
- Lock: none. Side effects: none besides the marker.
- Activity: touched only when the owner holds documents (the web UI lists on every rerun under a
  fresh per-session owner).
- Status: 200; 401; 422; 503 `Vector store unavailable`.
- Tests: `tests/test_document_management.py`, `tests/test_quotas.py`, `tests/test_sweep.py`.

## `DELETE /documents/{file_hash}`

- Handler: `delete_document` (`def`, unpinned). Input: path `file_hash` (`FILE_HASH_PATH`, 16
  lowercase hex), query `user_id`.
- Response: `DeleteResponse` `{message: "Document deleted", file_hash, chunks_removed}`.
- Lock: none (unlike upload, clear and the sweep; a race with an identical re-upload is an
  unverified risk).
- Side effects: `embeddings.delete_by_file_hash(file_hash, user_id)` and
  `bm25_index.delete_by_file_hash(file_hash, user_id)` inside a try; then, outside it,
  `storage.remove_stored_file`. Owner-scoped: another owner's copy of the same bytes is untouched.
- Activity: touched on success.
- Status: 200; 401; 404 `No such document` (unknown hash, or another owner's); 422 for a malformed
  hash (wrong length, non-hex, uppercase); 503 `Vector store unavailable`; a catch-all 500 if
  removing the file raises, after the chunks are already gone.
- Tests: `tests/test_document_management.py`, `tests/test_quotas.py`, `tests/test_sweep.py`.

## `POST /clear`

- Handler: `clear_documents` (`def`) -> `_clear_namespace`, which takes the owner lock, calls
  `storage.wipe_namespace` (Chroma delete by `{"user_id": ...}`, rmtree of the owner's upload
  directory, marker forgotten) and then `bm25_index.clear_owner`. Input: query `user_id`.
- Response: `ClearResponse` `{message: "Documents cleared successfully"}`, also when there was
  nothing to clear.
- Activity: the marker is removed, not touched.
- Status: 200; 401; 422; 500 `Failed to clear documents` (any exception, logged).
- Tests: `tests/test_api.py`, `tests/test_sweep.py`.

## `POST /feedback`

- Handler: `submit_feedback` (`def`).
- Body `FeedbackRequest`: `rating` `up` or `down`; `user_id` (`USER_ID_FIELD`); `question` 1-4000;
  `answer` up to 8000 (default empty); `sources` at most 20 strings; `request_id` optional,
  `^[A-Za-z0-9._-]+$`, at most 64 (the rated answer's `X-Request-ID`); `comment` up to 1000;
  `language` up to 32; `client` `^[a-z]*$`, up to 16. Unknown fields are ignored.
- Response: `FeedbackResponse` `{message: "Thanks - recorded."}`.
- Lock: none (`FeedbackStore` has its own lock for the size check).
- Side effects: appends one JSON line to `FEEDBACK_DIR/feedback.jsonl` (`at` server timestamp plus
  rating, user_id, question, answer, sources, request_id, comment, language, client); then
  `tracer.record_feedback` (a Langfuse `user_feedback` score, 1.0 or 0.0), called even when
  `request_id` is missing (known issue).
- Activity: touched on success.
- Status: 200; 401; 404 `Feedback collection is disabled` when FEEDBACK_ENABLED is false (only for a
  valid body: validation runs first, so a bad body is still 422); 422; 503 `Feedback storage
  unavailable` (OSError while writing); 507 `Feedback storage is full. The operator needs to rotate
  it.` when this record would push the file past FEEDBACK_MAX_BYTES (default 10485760).
- Tests: `tests/test_feedback.py` (route and store); `tests/test_tracing.py` (the score forwarded to
  the tracer).

## `POST /maintenance/sweep`

- Handler: `sweep_idle_namespaces` (`def`; decisions in `app/sweep.py`). Input, all query:
  `idle_days` (required, 1-3650), `prefix` (required; empty means every owner;
  `^[A-Za-z0-9_-]*$`, at most 64), `apply` (default false), `force` (default false).
- Response: `SweepResponse` (idle_days, prefix, dry_run, cutoff, newest_seen, candidates, empty,
  unknown, foreign, orphans, swept, became_active, failed, orphans_removed, refused).
- Refusals (apply without force): no prefix; idle_days below 7 (`MIN_IDLE_DAYS_TO_APPLY`); activity
  touches failed since startup; nobody active within idle_days while there is something to sweep.
  A refusal is still a 200 report with `refused` set and nothing deleted.
- Lock: `sweep.execute` takes each owner's lock per swept namespace (re-checking activity inside
  it) and per orphan cleanup; planning takes none.
- Side effects with apply: `storage.wipe_namespace` per candidate or empty owner (Chroma, files,
  marker - not BM25, known issue); stray files whose hash prefix backs no listed document are
  deleted inside the prefix. A dry run changes nothing.
- Activity: swept owners' markers are removed; nothing is touched.
- Status: 200; 401; 422 (missing or out-of-range parameters); 503 `Vector store unavailable`
  (`SweepError` while planning). Per-owner failures do not fail the request: they are listed in
  `failed` with `error` as `<ExceptionType>: <message>` cut to 200 characters.
- Client: `python -m scripts.sweep` (dry run by default; operating it: rag-operations).
  Tests: `tests/test_sweep.py`.
