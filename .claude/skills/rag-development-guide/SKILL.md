---
name: rag-development-guide
description: Sets up and runs the project day to day - Python 3.10 venv on Linux, macOS and Windows (including WSL), .env setup, Docker Compose development vs production, starting the backend, web UI and bot, ruff, dependency and lock-file regeneration with uv, the CI jobs, and commit, branch and PR conventions. Use when installing, starting services, fixing CI or lint, adding or upgrading a dependency, or preparing a commit or pull request.
---

# Development guide

Environment, services, lint, locks, CI and commit mechanics for this repository. Run every command
from the repository root: the backend reads .env and creates its data directories relative to the
working directory, and some tests open repo-relative paths.

## Use this skill for / not for

- For: installing or repairing an environment, .env questions, starting services, Docker Compose,
  a lint failure, adding or upgrading a dependency, a red CI job, preparing a commit or PR.
- Not for: writing or choosing tests (rag-testing); adding a setting or editing README.md,
  DOCUMENTATION.md, CLAUDE.md or skills under the doc-drift tests (rag-config-and-docs); backup,
  restore, sweep, quotas, request-id tracing, Langfuse tracing and running a deployment
  (rag-operations); live evaluation and benchmarks (rag-evaluation); where code lives and how
  requests flow (rag-architecture); reviewing a diff (rag-change-review).

## Map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | ruff (rules F, E9, I; line-length 100; target py310) and pytest (testpaths tests, `--strict-markers --strict-config`) |
| `requirements.txt`, `requirements-dev.txt` | direct runtime dependencies; pytest and ruff only (it does not include requirements.txt) |
| `requirements.lock`, `requirements-dev.lock` | generated, hashed linux / CPython 3.10 resolutions: the image installs the first, CI the second |
| `tests/test_lockfile.py` | offline checks of the lock headers, pins, hashes and coverage of the requirements files |
| `.github/workflows/ci.yml` | the lock, test and docker jobs |
| `Dockerfile` | two-stage python:3.10-slim image shared by backend, frontend and bot |
| `docker-compose.yml`, `docker-compose.override.yml`, `docker-compose.langfuse.yml` | production-shaped stack (services, named volumes, /ready healthcheck); dev overlay merged by a plain `docker compose up`; optional local Langfuse |
| `.dockerignore`, `.gitignore` | what stays out of the image and out of git |
| `.env.template` | every Settings field with its default (placeholders for OPENAI_API_KEY and BACKEND_API_KEY), plus TELEGRAM_BOT_TOKEN and BACKEND_URL |
| `app/config.py` | `Settings` (reads .env) and the cached `get_settings()` |
| `clients/backend.py` | `backend_url()` and `api_headers()`: how both clients read BACKEND_URL and BACKEND_API_KEY |

## Set up an environment

Python 3.10 only: both `Dockerfile` stages use python:3.10-slim, CI installs 3.10, ruff targets
py310 and both locks are resolved with `--python-version 3.10`. Another interpreter resolves a
different, untested set of packages.

Linux or macOS:

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
# Linux only, instead of the line above: exactly the set CI tests
pip install --require-hashes -r requirements-dev.lock
```

Windows (PowerShell or cmd), with uv, which also fetches Python 3.10 when it is missing:

```bat
uv venv --python 3.10 .venv
uv pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\activate
```

- `requirements-dev.txt` alone installs only pytest and ruff: always pass both files.
- The locks are a linux resolution without environment markers (they pin uvloop, which does not
  install on Windows), so Windows and macOS install from the requirements files and their
  transitive versions float. Adding `-c requirements-dev.lock` to the `uv pip install` line pins
  them to CI's versions.
- A uv-created venv has no pip (`python -m pip` fails with "No module named pip"); use `uv pip`.

WSL with the checkout on a Windows drive has two options:

1. A Linux venv inside WSL. This is the robust choice and the only one that works where Windows
   application control (Device Guard / WDAC) blocks unsigned interpreters, so calling
   `.venv/Scripts/python.exe` fails with "Invalid argument". The lock is Linux-native, so this is
   exactly CI's environment. Keep the venv outside the checkout (the Windows venv owns `.venv`)
   and run it from the repository root:

   ```bash
   uv venv --python 3.10 ~/.venvs/<name>
   uv pip install --python ~/.venvs/<name>/bin/python --require-hashes -r requirements-dev.lock
   ~/.venvs/<name>/bin/python -m pytest tests/test_<area>.py -q -p no:cacheprovider
   ~/.venvs/<name>/bin/python -m ruff check app frontend clients evaluation scripts tests
   ```

2. The Windows venv, called directly (there is no `.venv/bin` to source):

   ```bash
   .venv/Scripts/python.exe -m pytest tests/test_<area>.py -q -p no:cacheprovider
   WSLENV=MY_VAR MY_VAR=value .venv/Scripts/python.exe -m <module>   # env vars cross only via WSLENV
   WSLENV=MY_DIR/p MY_DIR="$PWD/data" .venv/Scripts/python.exe -m <module>   # /p converts a path
   ```

   Pass repo-relative paths: an absolute Linux path means nothing to a Windows process unless
   converted with `wslpath -w`.

Either way git may exist only as Windows git (`git.exe status` works from the checkout, as does
`git.exe -C "$(wslpath -w .)" log --oneline`), and `uv.exe` regenerates the locks byte-for-byte
too. The tests need no environment variables: `tests/conftest.py` sets a dummy OPENAI_API_KEY.

## Configure .env

`cp .env.template .env` (Windows: `copy .env.template .env`), then edit it. The file is
git-ignored, kept out of the image by `.dockerignore`, and given to every Compose service through
`env_file`, so Compose refuses to start without it.

| Variable | Read by | What to know |
| --- | --- | --- |
| OPENAI_API_KEY | backend | Required. Missing: startup aborts with a pydantic ValidationError. Empty: startup fails with "OPENAI_API_KEY not found". The template placeholder starts fine and fails on the first upload or question. Embeddings always use it; chat too unless LLM_API_KEY is set (rag-llm-providers). |
| BACKEND_API_KEY | backend and both clients | The X-API-Key shared secret. Empty disables auth on every route and logs "API authentication is DISABLED". A template placeholder keeps auth on with a publicly known key and only logs a warning: set a long random ASCII string (non-ASCII aborts startup). GET /health and GET /ready never need it. |
| TELEGRAM_BOT_TOKEN | bot | Missing or still the placeholder: the bot logs an error and exits. |
| BACKEND_URL | both clients, and the backup, restore and sweep scripts | Unset means http://127.0.0.1:8000 (the template has http://localhost:8000). Compose sets http://backend:8000 for frontend and bot; its `environment:` beats `env_file`. |

How the file is read:

- Backend: `app/config.py` `Settings` reads `.env` from the current working directory.
  Precedence: constructor arguments, then the process environment, then .env, then defaults. An
  exported shell variable silently beats .env, and a misspelled key is ignored (`extra="ignore"`).
- `get_settings()` is cached for the life of the process, and `--reload` watches only .py files:
  restart the backend after editing .env. In Compose run `docker compose up -d` again (it
  recreates containers whose environment changed); `docker compose restart` keeps the old one.
- Clients: `frontend/streamlit_app.py` and `clients/telegram_bot.py` call `load_dotenv()`, which
  searches upward from their own directory and never overrides variables already set. Both
  mirror FEEDBACK_ENABLED from it, and the web UI also mirrors MAX_FILE_SIZE (the bot never reads it).
- pydantic-settings never exports .env into os.environ, so code that reads os.environ without
  calling `load_dotenv()` does not see it. On the host, `python -m scripts.backup` and
  `python -m scripts.restore` ignore paths and settings that exist only in .env; export them first
  (`set -a; . ./.env; set +a`). See KNOWN_ISSUES.md, section Operations: backup, restore and sweep.
- A .env in the repository root (or a parent folder) leaks into pytest through those
  `load_dotenv()` calls and can change results by test order: move it aside while testing. See
  KNOWN_ISSUES.md, section Development and tests.

## Run the services on the host

```bash
uvicorn app.main:app --reload --port 8000    # backend: http://127.0.0.1:8000, API docs at /docs
streamlit run frontend/streamlit_app.py      # web UI: http://localhost:8501
python -m clients.telegram_bot               # bot: long polling, no port
```

- `app.main:app` is a lazy module attribute (`app/main.py` `__getattr__`): each access builds a
  new app with `create_app()` from `get_settings()`. Importing app.main needs no settings; tools
  and tests call `create_app(settings)` instead.
- Startup creates UPLOAD_DIR (default data/uploads) and CHROMA_PERSIST_DIR (default
  ./data/chroma_db); UPLOAD_DIR/.activity comes with the first owner request (or at startup for
  unmarked owner directories), FEEDBACK_DIR (default data/feedback) with the first rating.
- One backend process per data directory: no `--workers`, no second uvicorn on the same data.
  The owner locks and the BM25 index live in process memory.
- GET /ready pings the vector store on every call and answers 200 or 503; GET /health only proves
  the process answers.
- `--reload-dir app` keeps the reloader out of .venv and data (the Compose dev override uses it).

## Docker Compose

```bash
docker compose up --build                            # development: the override is merged
docker compose -f docker-compose.yml up -d --build   # production shape: code baked into the image
docker compose up backend frontend                   # everything except the Telegram bot
docker compose -f docker-compose.langfuse.yml up -d  # optional local Langfuse on port 3000
```

- A plain `docker compose` merges `docker-compose.override.yml`: `.:/app` bind mounts for all
  three services and `--reload --reload-dir app` for the backend.
- backend: published on 127.0.0.1:8000 only. frontend: Streamlit on 8501, published on all
  interfaces and without authentication of its own. bot: no port. frontend and bot start once
  the backend's healthcheck (a python urllib probe of /ready every 15 s) passes.
- Data lives in the named volumes data_uploads, data_chroma and data_feedback, mounted on the
  backend only (in dev mode too, so the host's ./data is a separate store). They survive `down`;
  `down -v` deletes every tenant's data, so ask first.
- The image leaves out tests/, evaluation/, root-level `*.md`, .claude/ and .env (`.dockerignore`),
  and the runtime lock has no pytest or ruff: run tests, lint and the evaluation modules
  (`python -m evaluation.run_eval` and the rest) on the host, never in the production image.

Services, healthcheck, dev vs prod, the Langfuse stack, reproducing the CI docker job and the
checklist for a new persistent data directory:
[references/docker-and-compose.md](references/docker-and-compose.md). Read it before touching
`Dockerfile`, a compose file or `.dockerignore`.

## Lint

- ruff 0.6.9 (pinned in `requirements-dev.txt`, configured in `pyproject.toml`) selects only F,
  E9 and I. Long code lines pass (E501 is off), but an unwrapped import line over 100 characters
  is I001. The usual CI failures: I001, F401 and F841 (unused), F821 (undefined name).
- The exact CI command; keep every copy of it identical:
  `ruff check app frontend clients evaluation scripts tests`
- To fix: `ruff check --diff <paths>` prints the changes without writing; read every hunk, then
  run `ruff check --fix <paths>`. Never run `ruff format`: the code is not formatted with it, and
  it would rewrite most files. `--no-cache` keeps ruff from writing .ruff_cache.
- A new top-level package (a root directory with `__init__.py`) must join the ruff command in
  `.github/workflows/ci.yml`, or `tests/test_docs_consistency.py` fails (frontend has no
  `__init__.py` and is listed by hand). Update the copies in README.md, DOCUMENTATION.md,
  CLAUDE.md and the skills in the same change (rag-config-and-docs).

## Dependencies

- The two locks are generated by uv 0.12.5 for linux / CPython 3.10, every transitive package
  pinned and sha256-hashed, extras kept. The image and CI install them with `--require-hashes`.
  Never edit a lock by hand.
- After any change to a requirements file, regenerate both locks with exactly these commands. uv
  writes the command line into line 2 of each lock and the CI lock job diffs the whole file, so
  flag order and path spelling matter (CI's extra `--quiet` is not recorded):

```bash
uv pip compile requirements.txt --python-platform linux --python-version 3.10 --generate-hashes --no-strip-extras --output-file requirements.lock
uv pip compile requirements.txt requirements-dev.txt --python-platform linux --python-version 3.10 --generate-hashes --no-strip-extras --output-file requirements-dev.lock
```

- Run `pytest tests/test_lockfile.py -q`, then commit the requirements files and both locks
  together. The test misses a removed requirement and a hand-edited transitive pin; only the CI
  lock job catches those.
- A package in neither requirements file is in neither the image nor CI, so an optional one must
  be import-guarded, like the `try`/`except ImportError` around flashrank in
  `app/rag/reranker.py`. flashrank is such a package: RERANKER_PROVIDER=flashrank reranks nothing
  in the image or in CI, with only a logged warning. See KNOWN_ISSUES.md, section Models, keys and
  reranking.

What each check catches, installing per OS, the add/bump/remove recipe, native libraries and
upgrade notes: [references/dependencies.md](references/dependencies.md). Read it before adding,
bumping or removing a package.

## CI

`.github/workflows/ci.yml` runs on pushes to main and on every pull request. Three jobs run in
parallel on ubuntu-latest, and a newer run on the same ref cancels the older one.

| Job | What it does | Reproduce locally |
| --- | --- | --- |
| lock | installs uv 0.12.5 (`UV_VERSION`), recompiles both locks into themselves, fails on `git diff --exit-code -- requirements.lock requirements-dev.lock` | the two uv commands above, then the same `git diff --exit-code`; no output means green |
| test | Python 3.10, `pip install --require-hashes -r requirements-dev.lock`, the ruff command, then `pytest -q` with OPENAI_API_KEY=test-key-not-real | Linux or WSL option 1: the same three steps in a fresh 3.10 venv. Elsewhere: ruff and `pytest -q` in your venv (transitive versions may differ) |
| docker | builds the image, prints its size, fails if gcc is in the runtime image, boots uvicorn in a container and polls GET /health | the commands in [references/docker-and-compose.md](references/docker-and-compose.md); needs a Docker daemon |

- lock red: a requirements edit without regeneration, reordered flags or a hand edit. Rerun both
  commands verbatim and commit the locks.
- test red at install ("hash" or "must be pinned with =="): a lock not produced by the documented
  commands, for example with extras stripped. Red on an ImportError: a package missing from the
  lock; declare it or guard the import.
- docker red at the smoke test: read the "Container logs on failure" step; usually an import
  error or a Settings ValidationError.
- CI never runs Compose, GET /ready, the volumes, the frontend or the bot, and has no Windows or
  macOS job. Check those by hand when you change them.

## Commits, branches and pull requests

- Subject: Conventional Commits, `type(scope): summary`, imperative, lowercase, no trailing
  period. A real one:
  `fix(reranker): provide fallback RerankRequest when flashrank is not installed`.
  Types in use: feat, fix, test, docs, style, refactor, perf. The scope names the area (rag,
  eval, metadata, reranker, config, scripts, hybrid, bm25, byok, docker, schemas, ...); join two
  with a comma, as in `feat(ui,api): ...`.
- Body, wrapped at about 80 columns: why the change was needed (the problem, the design, the
  alternatives rejected), what you verified and how, what you could not verify here (no Docker,
  no real key), and a closing paragraph `Tests: <after> (up from <before>).` with the collected
  counts (`pytest --collect-only -q | tail -1`) and how many mutations of the new code a named
  test caught, as the Stage-era commits record it.
- Plain hyphens only, never em or en dashes, in code, docs and messages. The history carries no
  trailers; add whatever attribution trailer your session requires.
- Commit related files together: requirements files with both locks; a setting with
  `.env.template` and its DOCUMENTATION.md row (rag-config-and-docs); code with its tests.
- Branches: `feature/<kebab-topic>` off main (older work used `stageN-<topic>`). Never commit to
  main directly.
- One pull request per topic, merged with a merge commit (no squash, no rebase), so every commit
  on the branch lands in main: keep each one green and meaningful. Avoid stacked branches; if
  one is unavoidable, say in the PR which PR it builds on.

## Before committing or opening a pull request

1. `git status`: only intended files; no .env, data, caches or venv. `.gitignore` silently drops
   names that match anywhere (`test_*.pdf`, `test_*.txt`, `sample_*.pdf`, `sample_*.txt`, `*.bin`,
   `*.sqlite3`, `*.log`, `*.key`, `lib/`), so check every new file with
   `git check-ignore -v <path>` (no output means git does not ignore it).
2. `ruff check app frontend clients evaluation scripts tests`
3. The tests for what you touched (the rag-testing skill maps modules to tests), then the doc,
   language, lock, offline and agent-doc contracts:

```bash
pytest tests/test_docs_consistency.py tests/test_language_consistency.py tests/test_lockfile.py tests/test_offline_guarantee.py tests/test_agent_docs.py -q
```

4. Before a PR, the full suite with no .env in the repository root: `pytest -q` (well under a
   minute on Linux, a few minutes on Windows).
5. Requirements changed: both locks regenerated with the exact commands, a second run leaves no
   diff, `pytest tests/test_lockfile.py -q` passes, and the image still builds if Docker is
   available.
6. `Dockerfile`, a compose file or `.dockerignore` changed: the CI docker steps and a Compose run
   from [references/docker-and-compose.md](references/docker-and-compose.md).
7. A setting, route, bot command, `python -m` entry point or document changed: the
   rag-config-and-docs checklist.
8. No em or en dashes in what you added: `git diff --cached | grep -nP '^\+.*[\x{2013}\x{2014}]'`
   prints nothing (GNU grep).
9. Self-review with the rag-change-review skill, then write the message as above.

## Pitfalls

- Lint misses an `async def` route doing blocking work (it stalls the event loop) and an unguarded
  optional import (ImportError in CI and the image); see rag-api-endpoints and Dependencies above.
- A Windows-only failure (file handles, path separators) never shows in CI, which is Linux-only.
- The `Dockerfile` pre-creates only data/uploads and data/chroma_db, so a fresh Compose
  deployment built from a checkout without data/feedback (a fresh clone has none) may get a
  root-owned data_feedback volume and failing ratings (an unverified risk; KNOWN_ISSUES.md,
  section Operations: backup, restore and sweep). data/benchmarks/ is git-ignored but not
  docker-ignored, so a local benchmark download ships in the image.

## Related skills

- rag-testing (fixtures, fakes, what to run), rag-config-and-docs (settings, doc-drift rules).
- rag-operations (backups, restores, the sweep, live volumes, Langfuse tracing), rag-evaluation
  (the live harness and benchmarks, host only).
- rag-clients and rag-llm-providers (client settings, BYOK, the server LLM endpoint),
  rag-architecture (components and request flows), rag-change-review (the pre-merge review).
