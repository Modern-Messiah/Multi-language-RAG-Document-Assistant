# Multi-language RAG Document Assistant - guide for coding agents

What every task here needs; task-specific detail is in the skills below, verified defects in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md). If this file and the code disagree, trust the code and fix this file.

## The project

A FastAPI backend (`app/`) indexes PDF, DOCX, TXT and Markdown uploads into one ChromaDB collection
shared by every tenant, separated only by an unauthenticated `user_id` behind one shared `X-API-Key`
secret, and answers in eight languages (or the question's own) using dense plus BM25 hybrid retrieval,
optional reranking and any OpenAI-compatible chat model, the operator's or the caller's own (BYOK). A
Streamlit UI (`frontend/streamlit_app.py`) and a Telegram bot (`clients/telegram_bot.py`) are its HTTP
clients; `evaluation/` measures retrieval, `scripts/` holds operator tools, Langfuse tracing is optional.

## Environment

Python 3.10 everywhere (Docker, CI, the ruff target, the locks). Work from the repository root: the
backend reads `.env` and creates `data/` relative to the working directory, and tests open repo-relative paths.

- Linux/macOS: `python3.10 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt`.
  On Linux, `pip install --require-hashes -r requirements-dev.lock` instead installs exactly CI's set.
- Windows: `uv venv --python 3.10 .venv`, `uv pip install -r requirements.txt -r requirements-dev.txt`,
  `.venv\Scripts\activate`; the locks are a linux/CPython 3.10 resolution, so Windows and macOS use the requirements files.
- WSL: call a Windows venv directly (`.venv/Scripts/python.exe -m pytest ...`; Linux variables reach it
  only via `WSLENV`, and `git` may exist only as `git.exe`), or, more robust and the only option where
  Windows application control (Device Guard, WDAC) blocks the interpreter with "Invalid argument", a
  Linux venv outside the checkout: `uv venv --python 3.10 ~/.venvs/<name>`, then
  `uv pip install --python ~/.venvs/<name>/bin/python --require-hashes -r requirements-dev.lock` (exactly CI's set).
- `cp .env.template .env` before running services: `OPENAI_API_KEY` is the only required setting and
  `BACKEND_API_KEY` is the API secret (empty disables auth). Tests need neither.

## Commands

```bash
pytest -q                                    # the whole offline suite, as CI runs it
pytest tests/test_<area>.py -q               # one module; or <file>::<test>, or -k "<expr>"
pytest tests/test_docs_consistency.py tests/test_agent_docs.py -q   # doc contracts, after doc or config edits
ruff check app frontend clients evaluation scripts tests            # lint exactly as CI; keep every copy identical
uvicorn app.main:app --reload --port 8000    # backend on 127.0.0.1:8000; one process only, no --workers
streamlit run frontend/streamlit_app.py      # web UI on :8501; BACKEND_URL defaults to http://127.0.0.1:8000
python -m clients.telegram_bot               # Telegram bot; needs TELEGRAM_BOT_TOKEN
docker compose up --build                    # dev stack: docker-compose.override.yml merged (bind mounts, --reload)
docker compose -f docker-compose.yml up -d --build    # production shape, no override
docker compose -f docker-compose.langfuse.yml up -d   # optional local Langfuse on :3000
```

- The full suite takes well under a minute on Linux, a few minutes on Windows. Move a root `.env` aside
  first: the UI and bot tests load it into the test process (KNOWN_ISSUES.md, "Development and tests").
- Operations (rag-operations skill; destructive steps need the user's go-ahead): `python -m scripts.backup`
  and `python -m scripts.restore <archive>` want the backend stopped (`--inspect` only reads the archive)
  and, run on the host, ignore `.env`: export its variables first. `python -m scripts.sweep` wants the
  backend running and is a dry run until `--apply`.
- Evaluation (rag-evaluation skill): `python -m evaluation.run_eval --url http://127.0.0.1:8000 --api-key <key>`
  costs real API calls; `python -m scripts.download_benchmarks` has no CLI, so even `--help` starts
  downloading every benchmark not yet in `data/benchmarks/` from the network.

## Invariants

1. **Blocking handlers are plain `def`.** Every route in `app/main.py` except `/health` is `def`, so
   FastAPI runs it in a threadpool; an `async def` doing Chroma, disk or model I/O stalls every request.
2. **Every Chroma read or delete is tenant-scoped**; the sweep's `EmbeddingsManager.list_owners` is the
   one deliberate exception. `EmbeddingsManager._owned(owner, ...)` serves only that class's own calls;
   `RAGChain` and `HybridRetriever` need a filter whose top level is `{"user_id": owner}` or the `$and`
   from `build_chroma_filter(user_id, metadata)` (`app/rag/metadata_extractor.py`) and crash on an
   `_owned`-shaped one. Always give BM25 an owner: `BM25Index.search` without one searches every tenant.
3. **One backend process, the only Chroma writer.** Owner locks and the BM25 index live in its memory.
   Backup and restore need it stopped (from a one-off Compose container their `/ready` probe may not
   reach it, so stop it yourself); the sweep runs inside it through `POST /maintenance/sweep`.
4. **Upload, `/clear` and the sweep hold the owner's lock** (`state.upload_locks.for_owner(user_id)`;
   `DELETE /documents/{file_hash}` takes none). Striped and non-reentrant: never take it twice or for two owners.
5. **A BYOK key lives for one request.** `X-Model-Key` (plus optional `X-Model`, `X-Model-Provider`) pays
   only for chat calls, never embeddings. Never store, log, cache or echo it; the endpoint comes from `PROVIDERS`
   in `app/byok.py`, not the request; a model or provider without a key is a 400; every exit path,
   streaming included, ends in `byok.close_quietly` (gaps: KNOWN_ISSUES.md, "Models, keys and reranking").
6. **Tests never touch the network.** `tests/conftest.py` blocks non-loopback sockets, fakes embeddings,
   forces `OPENAI_API_KEY=test-key-not-real` and strips Settings variables; use the `api` fixture,
   `make_settings(tmp_path, ...)` and `FakeChatClient`, and write only under `tmp_path`.
7. **`Settings` is the single source of configuration.** Backend components get values from
   `app/config.py` through the `app/main.py` lifespan; add no `os.environ` reads (pydantic-settings does
   not export `.env`). Every new field needs a default, a `.env.template` line and a DOCUMENTATION.md row.
8. **Side indexes follow every mutation path.** The in-memory BM25 index (`app.state.bm25_index`) changes
   with Chroma on upload, revision retirement, delete and `/clear`; new per-owner state must also reach
   backup, restore and the sweep, which skips BM25 (KNOWN_ISSUES.md, "Operations: backup, restore and sweep").
9. **One query path.** `ask` and `ask_stream` share `RAGChain._prepare` (`app/rag/chain.py`); change
   retrieval and prompts there. `/query/stream` primes its generator so early failures keep status codes.
10. **One table per fact.** Languages live only in `app/rag/languages.py`, upload formats only in
    `SUPPORTED_EXTENSIONS` (`app/rag/document_loader.py`), providers only in `PROVIDERS`; clients import them.
11. **Knob defaults.** A setting whose wrong value fails silently ships off until measured
    (`RELEVANCE_THRESHOLD=0.0`, `MMR_LAMBDA=1.0`); one that fails loudly may ship on (quotas answer 413).
    `HYBRID_SEARCH_ENABLED`, `METADATA_FILTERING_ENABLED` and `TABLE_AWARE_CHUNKING` ship on against this
    policy; measure with the rag-evaluation skill before adding or changing a default.

## Doc-drift contract

`tests/test_docs_consistency.py` pins README.md, DOCUMENTATION.md and `.env.template` to the code;
`tests/test_agent_docs.py` pins this file, KNOWN_ISSUES.md and the skills. The rules that surprise people
(all of them, with checklists for settings, routes and bot commands, are in the rag-config-and-docs skill):

- A `Settings` field needs an uncommented `NAME=` line in `.env.template` and a Configuration-table row
  `` | `NAME` | description | `default` | `` whose first backticked value is exactly `str(default)`
  (`True`/`False`, `45.0`, `""`), with no `|` in the description and no trailing whitespace.
- README.md may not contain a `/query` or `/upload` token anywhere (URLs included) and must name every
  bot command (`/start`, `/help`, `/clear`, `/documents`, `/model`). README.md and DOCUMENTATION.md both
  name `BACKEND_API_KEY`, `clients.telegram_bot` and `-f docker-compose.yml`, and neither may contain
  `telegram/bot.py`, `streamlit_user` or `user_id (optional)`.
- Each route in `app/main.py` needs a ``### `METHOD /path` `` heading in DOCUMENTATION.md, and the
  `/ready` section must say `**No authentication**` within 400 characters of its heading.
- A DOCUMENTATION.md line containing both "hunking" and "haracter" must include `1000` and `200` and
  neither `500` (it matches inside 1500 too) nor "50 char".
- In every doc, each `python -m` module of ours must exist and each full `ruff check` command must list
  exactly CI's packages; in agent docs, backticked repo paths must exist (write `<placeholders>`) and
  machine-specific absolute paths fail.

## Skills

| Skill | Use it when |
| --- | --- |
| [rag-architecture](.claude/skills/rag-architecture/SKILL.md) | orienting or planning a cross-module change: components on `app.state`, request lifecycles, data layout, tenant model, which skill applies, shipping a feature end to end, moving code out of `app/main.py` |
| [rag-development-guide](.claude/skills/rag-development-guide/SKILL.md) | setting up a venv on any OS (WSL included), `.env`, running services or Docker Compose, lint, dependencies and lock files, CI, commits and PRs |
| [rag-testing](.claude/skills/rag-testing/SKILL.md) | writing or fixing tests, choosing what to run after a change, debugging a failing, flaky or network-blocked test |
| [rag-config-and-docs](.claude/skills/rag-config-and-docs/SKILL.md) | adding or changing a setting, editing README.md, DOCUMENTATION.md, KNOWN_ISSUES.md, CLAUDE.md or a skill, or a red doc-drift test |
| [rag-api-endpoints](.claude/skills/rag-api-endpoints/SKILL.md) | adding or changing a route, request or response field, status code, auth, quota check, SSE or error mapping |
| [rag-retrieval-pipeline](.claude/skills/rag-retrieval-pipeline/SKILL.md) | tuning or debugging answers: condensation, threshold and MMR, hybrid BM25 plus RRF, how metadata filters are applied, reranking and reranker providers, prompt, languages, citations |
| [rag-document-ingestion](.claude/skills/rag-document-ingestion/SKILL.md) | file formats, parsing and encodings, chunking and tables, the metadata extractor's rules for uploads and questions (aliases, years, doc types), upload errors, very large documents |
| [rag-llm-providers](.claude/skills/rag-llm-providers/SKILL.md) | the server LLM endpoint (`LLM_BASE_URL`, DeepSeek), BYOK headers and LLM providers, model aliases, who may send reranker headers, refused keys |
| [rag-clients](.claude/skills/rag-clients/SKILL.md) | changing or testing the Streamlit UI, the Telegram bot or `clients/backend.py` |
| [rag-operations](.claude/skills/rag-operations/SKILL.md) | backup and restore, the idle-namespace sweep, quotas, the feedback log, tracing a request id, `/health` vs `/ready`, volumes, Langfuse |
| [rag-evaluation](.claude/skills/rag-evaluation/SKILL.md) | measuring retrieval before and after a change: offline metrics, `run_eval`, golden and feedback cases, Langfuse benchmarks |
| [rag-change-review](.claude/skills/rag-change-review/SKILL.md) | reviewing a diff, branch or pull request, or self-checking before a commit |

## Conventions

- Commits: a Conventional Commits subject, `type(scope): summary` (imperative, lowercase, no trailing
  period), then a body: why, what you verified and how (and what you could not), and a closing
  paragraph `Tests: <after> (up from <before>).` with the collected counts and the mutation-check result.
- Branches: `feature/<topic>` off main, one topic per pull request.
- Plain hyphens only, never em or en dashes, in code, docs and commit messages.
- Tests go in the module that already covers the area (rag-testing's test index); a sizeable new
  feature gets its own `tests/test_<feature>.py` whose docstring names what it guards. Check that
  a new test fails when the line it protects is reverted.
- Before committing: the ruff command above, the tests for what you touched (rag-testing skill) and a
  self-review with the rag-change-review skill.

## Known issues

[KNOWN_ISSUES.md](KNOWN_ISSUES.md) lists defects reproduced against the code, grouped by area. Check it
before treating odd behaviour as your own bug or relying on a default; a fix removes its entry and adds
a regression test.

Optional: CocoIndex's `ccc` code search is not part of this repository (`.cocoindex_code/` is
git-ignored); use it only if you have it installed and indexed locally, otherwise use Grep and Read.
