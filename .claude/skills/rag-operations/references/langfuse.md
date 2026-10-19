# Langfuse tracing reference

Read this when switching tracing on, running the local Langfuse stack, connecting a containerised
backend to it, reading a trace during triage, or wondering why no traces arrive. Code:
`app/tracing.py` (tracer and handles), `app/rag/chain.py` `RAGChain._prepare`, `ask` and
`ask_stream` (spans), `app/main.py` `query_rag`, `query_rag_stream` and `submit_feedback`.

Contents:
- Settings and what reads them
- What gets traced
- Trace id = request id, and its two caveats
- The local stack (docker-compose.langfuse.yml)
- Connecting the backend
- Checking that it works
- Quiet failure modes and diagnostics
- Data that leaves the deployment

## Settings and what reads them

- `LANGFUSE_ENABLED` (default False): off means `NoOpTracer`, no SDK import, no network.
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (default empty): both required. With either missing
  the startup log says `LANGFUSE_ENABLED=true but keys are missing; tracing disabled`.
- `LANGFUSE_HOST` (default https://cloud.langfuse.com; an empty value also means Cloud).
- `app/tracing.py` `get_tracer` builds the tracer once in the lifespan from the cached Settings, so
  restart the backend after any change (Compose: `docker compose up -d backend` re-reads `.env`).
- The SDK reads its own tuning variables straight from the process environment: `LANGFUSE_DEBUG`
  (only the exact string `True`), `LANGFUSE_SAMPLE_RATE`, `LANGFUSE_FLUSH_AT`,
  `LANGFUSE_FLUSH_INTERVAL`, `LANGFUSE_TIMEOUT` (20), `LANGFUSE_MAX_RETRIES` (3). Compose's
  `env_file` exports `.env`, so they work there; a host uvicorn reads `.env` through Settings
  without exporting it, so export them in the shell. They are not Settings fields: keep them out
  of `.env.template` (rag-config-and-docs explains the doc tests that reject unknown variables).
- Versions: `requirements.txt` pins `langfuse==2.60.10` and the compose file runs
  `langfuse/langfuse:2`; `app/tracing.py` uses the v2 API (`client.trace`, `client.score`). Do not
  upgrade either alone: a mismatched SDK turns every trace into a NoOp with a
  `Failed to start Langfuse trace` warning per request, while the tests still pass because they
  mock `langfuse.Langfuse`.

## What gets traced

Only `POST /query`, `POST /query/stream` and `POST /feedback`; uploads, listings, deletes, `/clear`
and the sweep are not traced.

- Trace `rag-query`: id = the request id, `user_id` = the tenant, input `{"question": ...}`, tags =
  the provider name from `X-Model-Provider` (or `default`) plus the language, output
  `{"answer", "sources"}`, or `metadata.error` on failure.
- Span `query_condense`: only when a follow-up was rewritten; created after the call, so it shows
  no duration, and the condensing model call is not recorded as a generation.
- Span `retrieval`: input `query`, `k`, `filter`; output `count`, `sources`, `filter_used` (always
  the first filter, even when the tenant-wide fallback produced the documents).
- Span `rerank`: whenever something was retrieved, also with the reranker off (the no-op reranker
  still counts); its output `scores` stays empty unless a real reranker ran.
- Generation `generation`: model, the full messages (system rules, retrieved chunks, history),
  `temperature` and `max_tokens`, the output; token usage only for `/query`, none for streamed
  answers. On failure: level `ERROR` with `status_message` = the exception text, which for a
  provider error is its raw error body.
- Nothing retrieved: the trace ends with the `No relevant information found.` answer and no
  generation. Retrieval or reranking raising: the trace keeps only its input and the open span is
  never ended.
- `POST /feedback`: score `user_feedback` on `trace_id` = the rating's `request_id`, value 1.0 for
  up and 0.0 for down, with the comment. Sent only after the rating was stored (not on 404, 503 or
  507).

## Trace id = request id, and its two caveats

`RequestContextMiddleware` keeps a caller's `X-Request-ID` matching `^[A-Za-z0-9._-]{1,64}$` or
mints 16 hex characters, and that id becomes the trace id. A request id from a log line, an error
message or a rating therefore opens its trace directly: the SDK links traces as
`<LANGFUSE_HOST>/project/<project id>/traces/<request id>`, or search the project's Traces list.

- Caller-chosen ids are kept, and Langfuse upserts traces on id: a client or proxy that reuses an
  id puts several requests, possibly from different tenants, on one trace and makes log greps
  ambiguous. `evaluation/langfuse_eval.py` relies on choosing ids (rag-evaluation).
- A `POST /feedback` without `request_id` (only third-party API callers; both shipped clients always
  send it) is scored on the last trace this process created, possibly another user's, and the log
  says `Recorded feedback to Langfuse trace None`. Tell API callers to send the rated answer's
  `X-Request-ID` as `request_id`.

## The local stack (docker-compose.langfuse.yml)

Langfuse v2 (`langfuse/langfuse:2`) plus `postgres:16-alpine`, on its own `langfuse_net` network,
traces kept in the `langfuse_postgres_data` volume. For local use only.

```bash
docker compose -f docker-compose.langfuse.yml up -d   # UI on http://localhost:3000
docker compose -f docker-compose.langfuse.yml down    # stop, keep traces and accounts
docker compose -f docker-compose.langfuse.yml down -v # also delete them: ask first
```

- Both compose files sit in the checkout directory and share its Compose project name, so each
  stack reports the other's containers as orphans and `--remove-orphans` on one removes the
  other: never pass it. Adding `-p langfuse` to every command for this stack keeps the two apart,
  but only consistently: `-p` on a stack that was started without it creates a new, empty
  instance (and the reverse). Every other document shows the commands without `-p`.
- First start: open http://localhost:3000, sign up, create a project and its API keys
  (`pk-lf-...`, `sk-lf-...`). Use the UI at the address in `NEXTAUTH_URL` (http://localhost:3000);
  sign-in redirects go there.
- The file commits `NEXTAUTH_SECRET`, `SALT` and `ENCRYPTION_KEY` (the last two equal) and the
  Postgres password `langfuse`, and publishes port 3000 on all interfaces. If anyone else can reach
  the machine, set your own values (for example `openssl rand -hex 32`) before the first start and
  publish the port on 127.0.0.1 only, in a private copy of the file rather than the tracked one
  (a plain override file adds to `ports` rather than replacing them).

## Connecting the backend

- Backend on the host (uvicorn): `LANGFUSE_HOST=http://localhost:3000`.
- Backend in Compose: localhost is the backend container itself, and `langfuse-server` is on
  another network. Point `LANGFUSE_HOST` at an address the container can reach, such as
  http://host.docker.internal:3000 on Docker Desktop (Linux engines need
  `extra_hosts: ["host.docker.internal:host-gateway"]` on the backend service first), or connect
  the backend container to the Langfuse network and use http://langfuse-server:3000. Docker
  networking, not exercised in this repo.
- Langfuse Cloud: keep the default host and use the project's keys.

## Checking that it works

1. Startup log: `Langfuse tracing initialized (host=...)` means the client was built, nothing more
   (it does not contact the host); the two other outcomes are the missing-keys warning and
   `Failed to initialize Langfuse tracer; falling back to NoOp`.
2. Ask one question with your own id on a scratch tenant, then rate it:

   ```bash
   curl -s -H "X-API-Key: $BACKEND_API_KEY" -H "X-Request-ID: lf-check-1" -H "Content-Type: application/json" \
     -d '{"question": "test", "user_id": "lf-check"}' http://127.0.0.1:8000/query
   curl -s -H "X-API-Key: $BACKEND_API_KEY" -H "Content-Type: application/json" \
     -d '{"rating": "up", "user_id": "lf-check", "question": "test", "request_id": "lf-check-1"}' \
     http://127.0.0.1:8000/feedback
   ```

3. Trace `lf-check-1` should appear with a `user_feedback` score of 1.0. With an empty tenant it
   has only the no-context answer and no spans past `retrieval`. The rating lands in
   `feedback.jsonl` too; `POST /clear?user_id=lf-check` cleans up the tenant.

## Quiet failure modes and diagnostics

- Wrong keys: the server answers 4xx, and the SDK drops the batch without a retry and without any
  log above DEBUG. Check the keys directly (it raises on bad ones):

  ```bash
  python -c "from langfuse import Langfuse; print(Langfuse(public_key='pk-lf-...', secret_key='sk-lf-...', host='http://localhost:3000').auth_check())"
  ```

  or set `LANGFUSE_DEBUG=True` in the backend's environment and read the `langfuse` logger.
- Unreachable host: after its retries the SDK logs ERROR lines from the `langfuse` logger, with
  `[-]` as request id because they come from its background thread. Requests are not slowed:
  events are queued and a full queue drops them.
- Shutdown: the lifespan calls `flush()`, which waits for the queue; with the host unreachable
  every batch goes through its retries first, so stopping the backend can take noticeably longer.
- No trace for a request that failed early: a 400 from header validation or a 401 never reaches
  the tracer; a failure in retrieval leaves a trace with only its input.

## Data that leaves the deployment

With tracing on, every traced question, the full prompt (system rules, retrieved document text,
conversation history), every answer, its sources and provider error text go to `LANGFUSE_HOST`
unmasked; no mask is configured. The default host is Langfuse Cloud. This is far more than the
ratings log, which stores only exchanges a user chose to rate, so get the operator's agreement
before enabling it against a host outside their control.

Langfuse datasets, benchmark uploads and experiment runs (`python -m evaluation.langfuse_eval`,
`python -m scripts.upload_to_langfuse`): see the rag-evaluation skill.
