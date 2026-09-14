# 🤖 Claude Agent Guide: Multi-language RAG Document Assistant

Welcome! This repository implements a production-grade, multi-language **Retrieval-Augmented Generation (RAG)** assistant with a FastAPI backend, Streamlit Web UI, Telegram bot, ChromaDB vector store, and BYOK (Bring Your Own Key) model support.

---

## ⚡ Quick Orientation & Invariants

When working in this codebase, **always observe these foundational invariants**:

1. **Synchronous `def` Handlers for Blocking I/O**:
   In `app/main.py`, heavy endpoints (`/upload`, `/query`, `/query/stream`, `/clear`, `/documents`, `/feedback`, `/ready`) use **`def`**, NOT `async def`. This offloads PDF parsing, disk I/O, and ChromaDB queries to threadpools, preventing asyncio event loop starvation.

2. **ChromaDB Single-Writer Rule**:
   Only the live FastAPI backend process may write to ChromaDB. Offline scripts (`scripts/backup.py`, `scripts/restore.py`) verify `/ready` and refuse to run if the backend is live (unless passed `--live`).

3. **Multi-Tenancy & Path Isolation**:
   All vector queries must be filtered by tenant using `EmbeddingsManager._owned(owner, ...)`. Disk files reside strictly under `data/uploads/<user_id>/<hash>_<name>`.

4. **Ephemeral BYOK (Bring Your Own Key)**:
   Users can supply their own keys for answering via `X-Model-Key`, `X-Model`, and `X-Model-Provider`. Keys are validated, used for a single request, and closed in a `finally` block. Keys are **never** stored on disk, DB, or logs.

5. **Strict Offline Testing (Zero External Network Calls)**:
   The test suite enforces an offline socket guard in `tests/conftest.py`. Any test attempting an outbound socket connection outside loopback fails immediately. Use fake embeddings and mock responders.

---

## 📚 Dedicated Skills

For in-depth guidance, refer to the specialized skills in `.claude/skills/`:

- **[RAG Architecture & System Blueprint](.claude/skills/rag-architecture/SKILL.md)**:
  Deep dive into loaders, chunkers, embeddings, MMR re-ranking, query condensation, citation stripping, quotas, and tenant isolation.
- **[RAG Development & Operations Guide](.claude/skills/rag-development-guide/SKILL.md)**:
  Cheat-sheet for running services, test execution, dependency lock updates with `uv`, extension recipes (languages, formats, providers), and the evaluation harness.

---

## 💻 Essential Commands

```bash
# Environment setup
source .venv/bin/activate

# Testing (900+ tests, executes in ~30 seconds, 100% offline)
pytest -q
pytest tests/test_api.py -q

# Linter
ruff check app frontend clients evaluation scripts tests

# Running the backend
uvicorn app.main:app --reload --port 8000

# Running the Web UI
streamlit run frontend/streamlit_app.py

# Running the Telegram Bot
python -m clients.telegram_bot

# Running RAG Evaluation (Live backend)
python -m evaluation.run_eval --url http://127.0.0.1:8000 --api-key <key>

# Inspecting feedback
python -m evaluation.from_feedback
```

---

## 🗂️ Project Structure Summary

- **`app/`**: FastAPI backend, configuration (`config.py`), lifespan, routes, BYOK (`byok.py`), storage & quotas (`storage.py`, `sweep.py`), observability (`observability.py`), and RAG pipeline (`app/rag/`).
- **`clients/`**: Shared client utilities (`backend.py`) and Telegram Bot (`telegram_bot.py`).
- **`frontend/`**: Streamlit application (`streamlit_app.py`).
- **`evaluation/`**: Golden corpus (`golden.py`), deterministic metrics (`metrics.py`), and feedback extraction (`from_feedback.py`).
- **`scripts/`**: Backup, restore, and idle-namespace maintenance scripts.
- **`tests/`**: Comprehensive offline test suite.
- **`DOCUMENTATION.md`**: Complete system documentation.
