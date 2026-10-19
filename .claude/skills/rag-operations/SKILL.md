---
name: rag-operations
description: Runbook for operating a deployment - backup and restore (python -m scripts.backup / scripts.restore with the backend stopped), the idle-namespace sweep (python -m scripts.sweep, dry run first), quotas, the feedback log, tracing a request id through the logs, /health vs /ready, Compose volumes, and the local Langfuse stack. Use for backups, restores, cleaning idle web sessions, disk or quota problems, investigating a failed request, or setting up tracing. Destructive steps need the user's confirmation.
---

# Operating the RAG assistant

How to protect, clean up and debug a running deployment, Compose or host: where the data lives,
backup and restore, the idle-namespace sweep, quotas and the ratings log, following one request
through the logs and Langfuse, and a triage recipe. Several procedures hit verified defects; the
workaround sits next to the step it breaks.

## Ground rules

1. Get the user's explicit go-ahead before anything destructive, after showing what it will touch
   (dry run, `--inspect` output, file list): sweep `--apply` (and `--force`); any restore that is
   not `--inspect`, and its `--overwrite` or `--force`; `--live` on backup or restore;
   `POST /clear` or a document delete on a real tenant; emptying or deleting volumes
   (`docker compose down -v`, `docker volume rm`, the emptying step of the Compose restore).
2. Exactly one backend process per data set: no `uvicorn --workers`, no second replica. ChromaDB
   takes one writer, and the owner locks, the BM25 keyword index and the activity failure counter
   live in that process's memory. Hence two opposite rules: backup and restore run with the
   backend stopped (they copy and replace its files); the sweep runs inside the running backend
   (`scripts/sweep.py` only calls `POST /maintenance/sweep`).
3. Identify the deployment first. Compose: `docker compose ls` shows the project and the files it
   was started from; pass the same `-f` flags to every command (`-f docker-compose.yml` for a
   production-shaped stack) or one-off containers get the dev override merged in. Host: work from
   the repo root, where the backend reads `.env` and the scripts resolve `data/...`.
4. Rehearse on scratch data, never on a checkout's own `data/`: point `CHROMA_PERSIST_DIR`,
   `UPLOAD_DIR` and `FEEDBACK_DIR` at a temp dir (the scripts read them from the environment).

## Use this skill for / not for

For: backups and restores, the idle sweep, quota and disk problems, the ratings log, finding a
request's log lines and trace, `/health` vs `/ready`, Compose volumes, the local Langfuse stack.

Not for: route, status-code or handler changes (rag-api-endpoints); adding a setting or changing a
default (rag-config-and-docs); installing, starting services, Compose dev vs prod, CI
(rag-development-guide); retrieval and BM25 internals (rag-retrieval-pipeline); BYOK headers and
provider errors (rag-llm-providers); ratings into golden cases, Langfuse datasets and experiments
(rag-evaluation); what the web UI and bot display (rag-clients); writing tests (rag-testing).

## Map

| File | Responsibility | Key symbols |
| --- | --- | --- |
| `app/backup.py` | archive and restore the three data dirs | `storage_locations`, `create_backup`, `restore_backup`, `verify_members`, `incompatibilities`, `_safe_members` |
| `scripts/backup.py`, `scripts/restore.py` | CLIs, liveness probe | `backend_is_up`, `main`, `_describe` |
| `app/sweep.py`, `scripts/sweep.py` | sweep decisions (never HTTP); the CLI, an HTTP client | `plan`, `_refusal`, `execute`, `report`, `MIN_IDLE_DAYS_TO_APPLY`; `call_sweep`, `print_report` |
| `app/activity.py`, `app/storage.py` | last-seen markers; owner files and namespace wipe | `ActivityTracker` (`touch`, `last_seen`, `reset_all`, `touch_failures`), `wipe_namespace` |
| `app/main.py` | quotas, locks, feedback, sweep and probe endpoints | `OwnerLocks`, `_quota_problem`, `_clear_namespace`, `submit_feedback`, `sweep_idle_namespaces`, `create_app` (`health`, `ready`, `unhandled_error`) |
| `app/feedback.py` | ratings log | `FeedbackStore.record`, `FeedbackStorageFull`, `store_from_settings` |
| `app/observability.py` | request ids, log format, access line, readiness | `RequestContextMiddleware`, `RequestIdFilter`, `LOG_FORMAT`, `readiness` |
| `app/tracing.py` | optional Langfuse tracer | `get_tracer`, `LangfuseTracer`, `NoOpTracer` |
| `docker-compose.yml`, `docker-compose.override.yml`, `docker-compose.langfuse.yml`, `Dockerfile` | services, volumes, healthcheck; dev bind mounts; local Langfuse; image (user appuser, WORKDIR /app) | |

## Where the data lives

| Setting | Default | Compose (backend only) | Holds |
| --- | --- | --- | --- |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | volume `data_chroma` at /app/data/chroma_db | chroma.sqlite3 plus HNSW segment dirs |
| `UPLOAD_DIR` | `data/uploads` | `data_uploads` at /app/data/uploads | raw files `<user_id>/<hash16>_<name>`, markers `.activity/<user_id>` |
| `FEEDBACK_DIR` | `data/feedback` | `data_feedback` at /app/data/feedback | `feedback.jsonl` |

- Compose prefixes volume names with the project name; `down` keeps volumes, `down -v` deletes
  every tenant's data. Keep the three paths at their defaults under Compose: the volumes are
  mounted exactly there, and host-run scripts never see them.
- In memory only: the BM25 index (refilled per owner from Chroma after a restart), owner locks,
  the activity failure counter. In no archive: `.env`, logs, Langfuse's Postgres volume.

## Backup

`chroma.sqlite3` is copied through SQLite's backup API, but the HNSW segment files are plain
copies, so a write during the copy leaves the two out of step. The scripts probe `GET <url>/ready`
(3 s; `--url`, else `BACKEND_URL` from the process environment, else http://127.0.0.1:8000): any
HTTP answer, 503 included, exits 2 unless `--live`; refused, unresolvable or timed out reads as
stopped, so a hung backend passes.

```bash
# Compose: the volumes are only reachable from a container
docker compose stop backend
mkdir -p backups
docker compose run --rm -v "$PWD/backups:/backups" backend python -m scripts.backup --output /backups
docker compose start backend

# Host, backend stopped: scripts.backup and scripts.restore never read .env themselves
set -a; . ./.env; set +a
python -m scripts.backup                     # archive in data/backups (--output to change)
```

- Compose: stop the backend yourself. The backend service sets no `BACKEND_URL`, so a one-off
  container takes it from `.env` (the template's http://localhost:8000 is that container itself)
  and the probe cannot see the service (a risk read from the config, not run in Docker);
  `--url http://backend:8000` should fix that, also unverified. A `PermissionError` on `/backups`
  means the container user `appuser` cannot write the host directory: make it writable for that uid.
- Host: in any shell with the venv active, `python -m dotenv run -- python -m scripts.backup`
  loads `.env` too. Without either, custom paths, `EMBEDDING_MODEL` and `COLLECTION_NAME` in `.env`
  are ignored and the defaults are archived with exit 0. From WSL with a Windows venv, exported
  variables reach `.venv/Scripts/python.exe` only through `WSLENV`.
- Check the summary (chunks, uploads, ratings, files, model, collection) against the deployment.
  Nothing but `uploads: 1` and `files: 1` means only a `.gitkeep` was archived (default, empty
  paths): the guard refuses only when none of the three directories exists.
- Then `python -m scripts.restore <archive> --inspect` (Compose: same `docker compose run` and
  mount) must print `Contents: intact` and `Fits this deployment: yes`. It never probes or writes.
- Archives are currently 2-3 times the data size (files stored more than once; restores are still
  correct), staging needs a full copy's space in the temp dir, and a 1-second UTC stamp names them,
  so two runs in one second into one `--output` collide. Keep `.env` elsewhere and copy archives
  off the host: they hold every tenant's documents.

Flags: `--output`, `--url`, `--live`, `--json`. Exit 0 written; 1 `Backup failed: ...` or a
traceback; 2 the backend answered. Archive layout, manifest, scheduling, exit details:
[references/backup-restore.md](references/backup-restore.md).

## Restore

Before any target changes, restore checks the manifest, the probe, every listed member's SHA-256,
that `EMBEDDING_MODEL` and `COLLECTION_NAME` match (else refused unless `--force`), that each
target is missing, empty or holds only `.activity` (else refused unless `--overwrite`), and that no
member is a link or escapes the extraction dir. Treat archives from elsewhere as untrusted.

Host procedure:
1. Stop the backend; export `.env` as for a backup.
2. `python -m scripts.restore <archive> --inspect`. On `Fits this deployment: NO`, set
   `EMBEDDING_MODEL` and `COLLECTION_NAME` to the values printed under `Embedding:` and
   `Collection:` instead of using `--force`: a forced model mismatch leaves a backend that refuses
   to start (`app/rag/embeddings.py` `EmbeddingsManager.get_vectorstore`), a forced collection
   mismatch opens an empty collection.
3. If the targets hold anything worth keeping, back them up first: `--overwrite` deletes each
   target directory and moves the archive's copy in, one directory at a time, not atomically.
4. `python -m scripts.restore <archive>`. On `These directories already hold data`, ask the user,
   then add `--overwrite`. A fresh clone's tracked `data/uploads/.gitkeep` counts as data.
5. Read the `Restored:` lines and any `WARNING:` lines (stderr, only with `--force`).
6. Start the backend; check `GET /ready`, then `GET /documents?user_id=<known owner>` against the
   archive's counts. Every activity marker is dated now, so nothing is sweepable for idle_days.

Compose: never point `scripts.restore` at the mounted volumes. It deletes each destination
directory, here a volume mount point: the contents go, then the removal fails with EBUSY ("Device
or resource busy") as an uncaught traceback, so `--overwrite` over existing data empties the
ChromaDB volume and restores nothing. Restore into a scratch directory inside the one-off
container and copy into the volumes: commands in [references/backup-restore.md](references/backup-restore.md).

Exit 0 restored (or `--inspect` clean); 1 unreadable archive, refusal, `--inspect` problems or a
traceback; 2 the backend answered.

## Idle-namespace sweep

The web UI mints a `web-<uuid hex>` owner per browser session, abandoned when the tab closes;
Telegram ids are digits, so the default prefix `web-` never matches a bot user.

1. The backend must be up. The script calls `load_dotenv()`, so `BACKEND_API_KEY` and
   `BACKEND_URL` come from `.env`; `--api-key` and `--url` override them.
2. Dry run. Host: `python -m scripts.sweep` (defaults `--idle-days 30 --prefix web-`). Compose:
   `docker compose run --rm bot python -m scripts.sweep` (the bot service carries
   `BACKEND_URL=http://backend:8000` and the key; a one-off backend container would ask itself).
3. Read the report with the user. Candidates: idle, with documents or files. Empty: a marker or an
   empty directory only. Orphan files: files with no vectors behind them, reported for every owner,
   removed on apply for every owner inside the prefix, active ones included. Unknown: nothing dates
   them, never swept even with `--force` (`POST /clear?user_id=<id>` removes one on purpose). Not
   owners: names that cannot be a user_id, never touched.
4. After the user confirms, repeat the same command with `--apply` (`--json` for raw output).
5. Exit codes: 0 ok; 1 a non-200 answer (raw detail on stderr, such as 401 or 422) or owners under
   FAILED; 2 backend unreachable; 3 REFUSED, nothing deleted. The client waits up to 600 s and a
   timeout also exits 2 while an apply may still be running: before retrying, grep the backend log
   for `Sweeping idle namespace` (a WARNING written before each removal).
6. After an applied sweep, restart the backend or `POST /clear?user_id=<id>` for each swept id: the
   sweep does not purge the in-memory BM25 index, so with hybrid search on (the default) a swept id
   that comes back still gets answers citing its deleted documents. A restart has its own cost: an
   owner whose first action afterwards is an upload loses keyword matching on older documents
   until the next restart (KNOWN_ISSUES.md, section "Retrieval and search"); per-id `/clear` avoids it.

Apply refuses, saying why, on: an empty prefix; `--idle-days` below 7; any failed marker write
since startup (usually a full volume: free space, restart the backend to reset the per-process
counter, sweep again); nobody seen inside the window while something would be swept (stale markers,
for example after a long stop). `--force` overrides all four; use it only on the user's word.

- last_seen is the newer of the marker and the newest file under `UPLOAD_DIR/<user_id>`. Successful
  uploads (duplicates too), questions, a non-empty `GET /documents`, deletes and ratings touch it;
  `/clear` and the sweep remove it.
- The endpoint has no defaults: `idle_days` (1-3650) and `prefix` (may be empty, meaning every
  owner) are required. With `BACKEND_API_KEY` empty this destructive endpoint is unauthenticated.
  There is no scheduler; cron the same command only if the user wants unattended applies.

## Quotas and disk

- `MAX_DOCUMENTS_PER_USER` (200) and `MAX_BYTES_PER_USER` (1073741824, 1 GiB) apply per user_id; 0
  turns one off, a negative value fails startup. `_quota_problem` runs after dedup (an identical
  re-upload passes) inside the owner lock; a superseded revision is discounted, and only stored
  files backing a listed document count as bytes.
- A refusal is 413 with numbers and a remedy but no setting name; the WARNING
  `Quota refused upload for user_id=...` in the log names it. `GET /documents?user_id=...` returns
  `quota` (`documents`, `max_documents`, `bytes`, `max_bytes`).
- To change a limit, edit `.env` and restart (Settings are read once); under Compose recreate with
  `docker compose up -d backend`, since `docker compose restart` keeps the old environment.
- Room is freed by deleting documents or `POST /clear` (destructive). user_id is unauthenticated,
  so quotas guard against accidents and cost, not abuse. Disk grows in the data directories and
  wherever archives go, plus a full temp copy during a backup or restore (`docker system df -v`
  shows volume sizes); the sweep reclaims abandoned web sessions. A full volume makes marker
  writes fail, which blocks sweep apply until a restart.

## Feedback log

- Every rating, up or down, is appended to `FEEDBACK_DIR/feedback.jsonl`, created on the first
  rating; one JSON object per line: `at` (server UTC), `rating`, `user_id`, `question`, `answer`,
  `sources`, `request_id`, `comment`, `language`, `client`. Treat it as user data.
- `FEEDBACK_MAX_BYTES` (10485760, 10 MB) counts the incoming record. At the cap every rating gets
  507 `Feedback storage is full. The operator needs to rotate it.` and the log says
  `rotate the file`. Move the file aside; the next rating starts a fresh one, no restart:

  ```bash
  docker compose exec backend mv /app/data/feedback/feedback.jsonl /app/data/feedback/feedback-<date>.jsonl
  ```

  Files left in `FEEDBACK_DIR` still use the volume and stay in every backup (manifest `ratings`).
- A write error is 503 `Feedback storage unavailable`, logged as `Could not write feedback`.
- `FEEDBACK_ENABLED=false`: 404, nothing written, no Langfuse score. The clients draw rating
  buttons from their own `FEEDBACK_ENABLED` (Compose services share `.env`): restart all three
  (Compose: `docker compose up -d`, which recreates them with the new environment).
- Fresh-Compose risk (not verified in Docker): the image does not create /app/data/feedback, so a
  new `data_feedback` volume can come up owned by root and every rating answers 503. Check with
  `docker compose exec backend ls -ld /app/data/feedback`; with the user's agreement fix it once
  with `docker compose run --rm -u root backend chown appuser:appuser /app/data/feedback`, or set
  `FEEDBACK_ENABLED=false`.
- To analyse it, copy it out (`docker compose cp backend:/app/data/feedback/feedback.jsonl .`) and
  run `python -m evaluation.from_feedback --input feedback.jsonl` on a checkout (`evaluation/` is
  not in the image; the default `--input` is relative to the working directory); see rag-evaluation.

## Request ids, logs and probes

- `X-Request-ID`: a caller's value is kept if it matches `^[A-Za-z0-9._-]{1,64}$`, else the backend
  mints 16 lowercase hex characters. The id is on every response, on every log line written during
  the request, is the Langfuse trace id, goes into both clients' ratings, and into a crash 500 body.
  Clients append `(request <id>)` to the error text only for 401, 403 and 5xx.
- Log format: `%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s`; lines outside
  a request (startup, the Langfuse SDK thread) show `[-]`.
- One access line per request from `app.observability`: `POST /upload -> 200 in 16 ms (user=u1)`.
  `user=` reads only the `user_id` query parameter, so `/query`, `/query/stream` and `/feedback`
  show `user=-` (an unsafe value prints `invalid`); the tenant is in the `chat completion
  user_id=...` (`/query`) or `streamed completion user_id=...` line, written once the model ran.
- uvicorn adds its own access line without the id (`--no-access-log` drops it; the Compose command
  does not pass it). uvicorn lines carry no timestamp: `docker compose logs -t` adds Docker's.
- Unhandled crash: 500 `{"detail": "Internal server error", "request_id": "<id>"}`; only the access
  line `-> 500` carries the id, and uvicorn's `Exception in ASGI application` traceback follows it
  without one. Other errors put the id in the header only; their own log lines (`Query failed` with
  a traceback, for example) carry it.
- `/query/stream` logs its access line when the stream ends, as 200 even after a mid-stream error
  event; look for `Streaming failed after the response started`.
- `GET /health` is liveness only (`{"status": "ok", "version": ...}`). `GET /ready` is 200 once
  startup finished and `EmbeddingsManager.ping()` reads the store, else 503 `not ready` naming the
  failed check (`startup`, `vector_store`), never why. Neither calls a model or needs `X-API-Key`.
- Compose probes `/ready` every 15 s (5 s timeout, 5 retries, 15 s start period); frontend and bot
  start only once it passes, and Docker does not restart an unhealthy container (`docker compose ps`
  shows the state). Probes are logged only when they fail; a store failure also logs
  `Readiness check failed: vector store is not usable`.

## Langfuse tracing

Off by default: `LANGFUSE_ENABLED` (False), `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
`LANGFUSE_HOST` (https://cloud.langfuse.com; empty also means Cloud), read once at startup. Each
`/query` and `/query/stream` becomes a `rag-query` trace whose id is the request id; `/feedback`
adds a `user_feedback` score. Everything traced (questions, full prompts with retrieved text,
answers) goes to `LANGFUSE_HOST` unmasked. Failures are quiet: wrong keys drop events without a
warning. A rating without `request_id` scores the last trace the process created, possibly another
user's, and a reused caller `X-Request-ID` merges requests into one trace. Local stack and its
secrets, reaching it from containers, spans, diagnostics: [references/langfuse.md](references/langfuse.md).

## Incident triage

1. Collect the report: what the user saw (exact text), when (with time zone), which client, which
   account (a bot user's user_id is their Telegram id; web sessions are `web-<hex>`).
2. Find the request id: `(request <id>)` in the error text (401, 403, 5xx), the `X-Request-ID`
   response header for API callers, or the `request_id` of their rating in `feedback.jsonl`.
   Otherwise list the access lines around that time:
   `docker compose logs -t --since <start> --until <end> backend | grep app.observability`.
3. Pull the request with `docker compose logs -t backend | grep -F '[<id>]'` (host: wherever
   uvicorn's output goes) and read the access line's status. 400: a BYOK or reranker header problem
   or the provider refusing the caller's own key (only the access line is logged;
   rag-llm-providers), or a rejected upload. 401 `X-API-Key`; 404 not this owner's document, or
   feedback disabled; 413 quota; 422 validation; 429 operator key rate limited
   (`OpenAI rate limited the request`); 504 `OpenAI timed out`; 503 store, disk or model failure
   (traceback with the id); 507 feedback full; 500 a crash or `Error clearing documents`.
4. If Langfuse is on, open trace `<id>`: retrieval span (query, k, filter, sources), rerank span,
   generation (model, prompt, output, `ERROR` level with the provider's message), the trace's
   `metadata.error`, any `user_feedback` score. Only an input means retrieval or reranking raised.
5. Reproduce on a scratch tenant, never in the user's namespace, after the user agrees (it spends
   embedding and model calls): upload the same file as `user_id=triage-<n>`, ask the same question
   with the same language, history and filters and your own `X-Request-ID`, compare, then
   `POST /clear?user_id=triage-<n>`. Never reuse the user's request id: it merges into their trace.
6. Hand over: retrieval (rag-retrieval-pipeline), parsing (rag-document-ingestion), provider
   (rag-llm-providers), client display (rag-clients), a regression case (rag-evaluation).

## Known issues in this area

Each is described above where it bites; details and repro in KNOWN_ISSUES.md, sections
"Operations: backup, restore and sweep", "Tracing and observability" and "Retrieval and search".
Verified: the sweep leaves swept owners in the BM25 index; host-run backup and restore ignore
`.env`; restoring into Compose volumes fails with EBUSY (with `--overwrite`, after emptying the
ChromaDB volume); archives store files 2-3 times; a lone `.gitkeep` passes the empty-archive guard
and blocks restore without `--overwrite`; feedback without `request_id` scores another trace, and
a reused caller id merges traces; streamed answers record no token usage. Risks read from the
config, not verified in Docker: the probe cannot see the backend from a one-off Compose
container; a fresh `data_feedback` volume may be root-owned.

## Verify

```bash
# read-only checks against a deployment
curl -s -w ' %{http_code}\n' http://127.0.0.1:8000/ready
curl -s http://127.0.0.1:8000/health
docker compose ps
python -m scripts.sweep                      # dry run only
python -m scripts.restore <archive> --inspect

# after changing ops code (which interpreter to use from WSL: rag-testing)
python -m pytest tests/test_backup.py tests/test_sweep.py tests/test_feedback.py tests/test_quotas.py -q
python -m pytest tests/test_observability.py tests/test_tracing.py tests/test_docs_consistency.py -q
ruff check app frontend clients evaluation scripts tests
```

The tests do not cover duplicate archive members, restores onto mount points, the scripts' `.env`
handling, BM25 state after a sweep, the real Langfuse SDK (it is mocked) or Compose itself.

## Related skills

rag-architecture (how the pieces fit), rag-development-guide (running services, Compose dev vs
prod), rag-api-endpoints (the handlers behind these endpoints), rag-config-and-docs,
rag-evaluation, rag-testing, rag-change-review.
