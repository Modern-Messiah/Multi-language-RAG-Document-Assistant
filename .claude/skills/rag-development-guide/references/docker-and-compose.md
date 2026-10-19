# Docker and Compose

Read this before touching `Dockerfile`, a compose file or `.dockerignore`, or when a Compose
stack misbehaves at startup. Operating a running deployment (backups, the sweep, volume
maintenance, Langfuse tracing) is the rag-operations skill.

Contents: the image; what `.dockerignore` keeps out; services; the healthcheck; development vs
production; the Langfuse stack; reproducing the CI docker job; checklist for a new persistent
data directory; pitfalls.

## The image

`Dockerfile` has two stages, both `python:3.10-slim`:

1. builder: installs build-essential, creates a venv at /opt/venv and runs
   `pip install --require-hashes -r requirements.lock`. Only the venv leaves this stage.
2. runtime: installs only libgomp1 (onnxruntime, pulled in by chromadb, links against it),
   copies /opt/venv, creates `appuser`, copies the build context with
   `COPY --chown=appuser:appuser . .` into /app, runs
   `mkdir -p data/uploads data/chroma_db && chown -R appuser:appuser data`, switches to
   `USER appuser` and exposes 8000 and 8501. There is no CMD: every service and the CI smoke test
   pass their own command.

Rules:

- Build tools go in the builder stage only: the CI docker job fails if gcc is in the runtime
  image. A native library needed at runtime goes in the runtime stage's apt-get line, as
  libgomp1 does.
- Install from `requirements.lock` with `--require-hashes`, never from `requirements.txt`.
- The base image tag and the apt packages are not pinned; only the Python package set is
  reproducible.

## What `.dockerignore` keeps out

`venv/`, `.venv/`, `__pycache__/`, `*.pyc`, `.git/`, `.github/`, `.env`, `.pytest_cache/`,
`.ruff_cache/`, `tests/`, `evaluation/`, `*.md` (root-level files only: README.md,
DOCUMENTATION.md, CLAUDE.md and the like), `.claude/`, `app/docs/assets/` (the README GIFs), the
contents of data/chroma_db, data/uploads and data/feedback, and `data/backups/`.

Consequences:

- No tests, no evaluation package, and (from the runtime lock) no pytest or ruff: run the suite,
  lint and the evaluation modules (`python -m evaluation.run_eval` and the rest) on the host. The
  dev override's bind mount puts the files back into the containers, but not pytest.
- scripts/ ships, so the operator scripts run inside a container (rag-operations).
- Everything not listed is sent to the build and copied into the image. data/benchmarks/,
  written by `python -m scripts.download_benchmarks` (it downloads on any invocation, even
  `--help`), is git-ignored but not docker-ignored: a local download ships in the image unless
  you delete it or add it to `.dockerignore` first.
- `*.md` matches only the build-context root. Nested Markdown needs `**/*.md` or a directory
  entry such as `.claude/`.

## Services

From `docker-compose.yml`:

| Service | Command | Published | Volumes | Starts after |
| --- | --- | --- | --- | --- |
| backend | `uvicorn app.main:app --host 0.0.0.0 --port 8000` | 127.0.0.1:8000 only | data_uploads, data_chroma, data_feedback | - |
| frontend | `streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0` | 8501 on all interfaces | - | backend healthy |
| bot | `python -m clients.telegram_bot` | none | - | backend healthy |

- All three: `env_file: .env` (required: Compose refuses to start without the file),
  `restart: unless-stopped`, network rag_net. frontend and bot get
  `BACKEND_URL=http://backend:8000` through `environment:`, which beats the .env value.
- The backend is bound to loopback because its only credential is one shared secret and user_id
  is unauthenticated input: keep it that way. The frontend holds that secret, has no login of
  its own and is published on all interfaces: on any untrusted network, put it behind an
  authenticating reverse proxy.
- Volumes: data_uploads at /app/data/uploads, data_chroma at /app/data/chroma_db, data_feedback
  at /app/data/feedback, backend only. Compose prefixes their names with the project name.
  `docker compose down` keeps them; `down -v` deletes every tenant's data (ask the user first).
- The bot exits at once when TELEGRAM_BOT_TOKEN is missing or still the placeholder, and
  `restart: unless-stopped` keeps starting it again. Without a token, leave it out:
  `docker compose up backend frontend`.

## The healthcheck

```yaml
test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready', timeout=3)"]
interval: 15s
timeout: 5s
retries: 5
start_period: 15s
```

- /ready, not /health: /ready pings the vector store on every call (503 when it fails), /health
  only proves the process answers, and frontend and bot start on `condition: service_healthy`.
- Keep the python urllib form with single quotes (the slim image has no curl).
  `tests/test_docs_consistency.py` extracts `urlopen('http://localhost:8000/<path>'` from this
  file and requires DOCUMENTATION.md to have that route's API-reference heading with
  **No authentication** in the 400 characters after it.
- Docker does not restart an unhealthy container; `docker compose ps` shows the state.
- CI never runs Compose or /ready: test a healthcheck change by hand.

## Development vs production

- `docker compose up --build` merges `docker-compose.override.yml`: the checkout is
  bind-mounted at /app in all three services (bypassing `.dockerignore`), and the backend runs
  `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app`. The named volumes
  stay mounted on the backend on top of the bind mount, so the host's ./data is not what the
  containers use.
- Only the backend runs with `--reload`. After editing client code, restart that service
  (`docker compose restart frontend` or `docker compose restart bot`); if a backend edit does not
  reload, `docker compose restart backend`.
- `docker compose -f docker-compose.yml up -d --build` skips the override: the code is baked
  into the image, so a code change needs a rebuild (`--build`).
- After editing .env, run the same `up -d` command again: Compose recreates the containers whose
  environment changed. `docker compose restart` keeps the old environment.

## The Langfuse stack

- `docker compose -f docker-compose.langfuse.yml up -d` starts langfuse/langfuse:2 and
  postgres:16-alpine (volume langfuse_postgres_data) on their own network, langfuse_net. The UI is
  http://localhost:3000, published on all interfaces.
- The file commits NEXTAUTH_SECRET, SALT and ENCRYPTION_KEY: for local use only.
- LANGFUSE_HOST=http://localhost:3000 works for a backend running on the host. A backend inside
  Compose resolves localhost to its own container and is not on langfuse_net, so that URL does
  not reach Langfuse from there.
- Both compose files sit in the repository root, so by default they share one Compose project
  name: the main stack lists the Langfuse containers as orphans, and `--remove-orphans` on it
  would remove them. The `-p langfuse` alternative, and why it must then be used on every
  command for that stack, is in the rag-operations skill's Langfuse reference.
- Keys, LANGFUSE_ENABLED and what gets traced: the rag-operations skill.

## Reproducing the CI docker job

Needs a running Docker daemon and a free port 8000 (stop a host-run backend first):

```bash
docker build -t rag-assistant:ci .
docker image inspect rag-assistant:ci --format 'image bytes {{.Size}}'
docker run --rm --entrypoint sh rag-assistant:ci -c 'command -v gcc'   # must print nothing and fail
docker run -d --name rag-ci -e OPENAI_API_KEY=test-key-not-real -p 8000:8000 rag-assistant:ci uvicorn app.main:app --host 0.0.0.0 --port 8000
curl -fsS http://localhost:8000/health     # retry for up to a minute; CI expects "status":"ok"
docker logs rag-ci                         # when it does not come up
docker rm -f rag-ci
```

CI stops there. After a compose or `Dockerfile` change, also run the production shape (it needs
.env):

```bash
docker compose -f docker-compose.yml up -d --build
docker compose -f docker-compose.yml ps        # backend becomes healthy, then frontend and bot start
curl -fsS http://127.0.0.1:8000/ready
docker compose -f docker-compose.yml exec backend ls -ld /app/data/uploads /app/data/chroma_db /app/data/feedback
```

Every data directory should be owned by appuser; then rate one answer in the web UI to prove
data_feedback is writable.

## Checklist: a new persistent data directory

1. A Path setting in `app/config.py` with a default under data/, plus its .env.template line
   and DOCUMENTATION.md row (the rag-config-and-docs skill).
2. Create it where the code first writes (`app/feedback.py` `FeedbackStore.record` creates
   FEEDBACK_DIR on the first rating) or in the `app/main.py` lifespan (UPLOAD_DIR is created
   there).
3. `Dockerfile`: add it to the `mkdir -p data/uploads data/chroma_db` line, before
   `USER appuser`. Per the Dockerfile's own comment, named volumes inherit the ownership of this
   directory on first use. data/feedback is missing from that line, so a fresh deployment built
   from a checkout without data/feedback (a fresh clone has none) may get a root-owned
   data_feedback volume on which every rating fails (a risk, not verified without Docker; the
   one-off ownership fix is in KNOWN_ISSUES.md, section Operations: backup, restore and sweep).
4. `docker-compose.yml`: mount a named volume on the backend (`data_<name>:/app/data/<name>`)
   and declare it under the top-level `volumes:`. Repeat the mount in the backend `volumes:` of
   `docker-compose.override.yml`, which lists every backend mount.
5. `.dockerignore`: `data/<name>/*`, so local data never enters the build context. `.gitignore`:
   the directory.
6. If it holds user data: `app/backup.py` `storage_locations` (used by backup and restore) knows
   only the chroma, uploads and feedback directories, and the per-owner cleanup of `/clear` and
   the sweep must learn about per-owner state; see the rag-operations and rag-architecture
   skills.
7. Tests: route the path into tmp_path in `tests/conftest.py` `make_settings` (rag-testing).
8. Verify on a fresh stack with the production-shape commands above: the new directory is owned
   by appuser and a write succeeds.

## Pitfalls

- Host-run data (./data) and Compose data (named volumes) are separate stores, even under the
  dev bind mount.
- The CI smoke test and a Compose backend both want port 8000 on the host; so does
  `uvicorn app.main:app` on the host.
- `docker compose up` without `-f` is the development stack, with the checkout mounted over the
  image's code.
