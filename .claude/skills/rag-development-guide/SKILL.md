---
name: rag-development-guide
description: Operational and developer guide for Multi-language-RAG-Document-Assistant. Use when running tests, executing linter checks, managing lockfiles, adding new languages, adding document formats, debugging, or running the evaluation harness.
---

# 🛠️ RAG Development, Testing & Operations Guide

This guide details developer workflows, test harness architecture, command cheat-sheets, extension recipes, and maintenance procedures for the **Multi-language RAG Document Assistant**.

---

## 1. Quick Command Reference

```bash
# Activate virtual environment
source .venv/bin/activate

# Run tests (runs completely offline, 900+ tests in ~30s)
pytest -q

# Run specific test file
pytest tests/test_api.py -q

# Run linter
ruff check app frontend clients evaluation scripts tests

# Run services locally
uvicorn app.main:app --reload --port 8000       # FastAPI backend
streamlit run frontend/streamlit_app.py         # Web interface (port 8501)
python -m clients.telegram_bot                  # Telegram bot

# Docker runs
docker compose up --build                       # Development (bind mounts + reload)
docker compose -f docker-compose.yml up -d      # Production (baked image)
```

---

## 2. Testing Philosophy & Test Harness Architecture

The test suite in `tests/` contains over 900 unit and integration tests designed for high speed and strict isolation.

### 2.1. Offline Socket Enforcement (`tests/conftest.py`)
- Any test attempting an outbound socket connection outside loopback (`127.0.0.1`, `::1`, `localhost`) raises an immediate `RuntimeError`.
- **Never make real external API calls in tests**. Embeddings are generated using deterministic local vectors (`_fake_vector` in `tests/conftest.py`), and chat completions are mocked using canned responders.

### 2.2. File Descriptor Leak Protection
- ChromaDB caches `SharedSystemClient` process-wide, which holds open SQLite connection pools.
- The fixture `_release_chroma` in `tests/conftest.py` monitors system clients and explicitly executes `system.stop()` after each test. Without this, test runs exceed OS open file limits (~800 tests).

### 2.3. Clean App State Fixture
- Use `test_app` or `client` fixtures from `tests/conftest.py`. Each test gets an isolated temporary upload directory (`data/uploads`) and a temporary ChromaDB directory (`data/chroma_db`) that are wiped on teardown.

---

## 3. Dependency Management & Lockfiles

The project maintains **two tiers** of dependency definitions:
1. `requirements.txt` & `requirements-dev.txt`: High-level direct dependencies.
2. `requirements.lock` & `requirements-dev.lock`: Fully resolved, SHA-256 hashed dependency trees targeting Linux / Python 3.10 (used by Docker and CI).

### Regenerating Lockfiles
Whenever dependencies change in `requirements*.txt`, regenerate the lock files using `uv`:
```bash
# Pin uv version matching CI
pip install "uv==0.12.5"

# Recompile production lock
uv pip compile requirements.txt \
  --python-platform linux --python-version 3.10 \
  --generate-hashes --no-strip-extras \
  --output-file requirements.lock

# Recompile development lock
uv pip compile requirements.txt requirements-dev.txt \
  --python-platform linux --python-version 3.10 \
  --generate-hashes --no-strip-extras \
  --output-file requirements-dev.lock
```
*Note: CI checks that lockfiles are fresh. A mismatch between `requirements*.txt` and `requirements*.lock` will fail the build.*

---

## 4. Extension Recipes

### 4.1. Adding a New Supported Language
All language rules are centralized in `app/rag/languages.py`. **Do not duplicate language lists across clients.**
1. In `app/rag/languages.py`:
   - Add the language and its system prompt instruction to `LANG_RULES`:
     ```python
     LANG_RULES["Italiano"] = "Rispondi rigorosamente in italiano."
     ```
2. (Optional) In `frontend/streamlit_app.py`, add the national flag to `LANG_FLAGS`:
   ```python
   "Italiano": "🇮🇹"
   ```
Both the Telegram bot and Streamlit UI read `SUPPORTED_LANGUAGES` directly from `app/rag/languages.py`.

### 4.2. Adding a New Document Format
Document loader dispatch is unified through `app/rag/document_loader.py`.
1. Add the file extension to `SUPPORTED_EXTENSIONS` (or `TEXT_EXTENSIONS` if plain text):
   ```python
   SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".rtf", *TEXT_EXTENSIONS)
   ```
2. Implement the loader method in `DocumentLoader` (e.g. `load_rtf(file_path)`).
3. Connect the extension in `DocumentLoader.load_document(cls, file_path)`.
4. Ensure returned `Document` objects include metadata: `source`, `type`, and `file_path`.

### 4.3. Adding a New BYOK Model Provider
The provider table is owned by `app/byok.py`.
1. In `app/byok.py`:
   - Add the provider name and its OpenAI-compatible base URL to `PROVIDERS`:
     ```python
     PROVIDERS["together"] = "https://api.together.xyz/v1"
     ```
2. (Optional) In `app/config.py`, update `allowed_model_providers` if restricting outbound connections.

### 4.4. Adding an API Endpoint
1. Define request and response schemas in `app/models/schemas.py`.
2. Implement the route in `app/main.py`.
3. **Important**:
   - If the endpoint touches ChromaDB, files on disk, or external LLMs, use **`def`** (not `async def`) to run in FastAPI's threadpool.
   - Always validate and filter by `user_id`.

---

## 5. RAG Evaluation Harness (`evaluation/`)

The evaluation suite calculates objective retrieval metrics against a known golden dataset without incurring LLM-as-a-judge costs.

### 5.1. Running Live Evaluation
Runs against a running backend server:
```bash
python -m evaluation.run_eval \
  --url http://127.0.0.1:8000 \
  --api-key <your-backend-api-key>
```
What it does:
1. Uploads the golden corpus (`evaluation/golden.py`) under a scratch tenant.
2. Queries answerable questions and verifies if expected sources are retrieved.
3. Queries unanswerable questions to observe distance scores.
4. Computes `Recall@k`, `Precision@k`, `MRR` (Mean Reciprocal Rank), and `Hit@k`.
5. Wipes the scratch tenant after completion.

### 5.2. Converting Feedback into Golden Cases
When users downvote answers via 👍/👎 buttons, entries are appended to `data/feedback/feedback.jsonl`.
To view problematic cases and generate golden test stubs:
```bash
# View negative ratings
python -m evaluation.from_feedback

# View positive ratings
python -m evaluation.from_feedback --up
```

---

## 6. Maintenance Procedures

### 6.1. Backup & Restore (`scripts/`)
```bash
# 1. Stop backend before snapshot (mandatory for ChromaDB SQLite consistency)
docker compose stop backend

# 2. Run backup
python -m scripts.backup --output data/backups

# 3. Restore from archive
python -m scripts.restore data/backups/rag-backup-*.tar.gz

# 4. Restart backend
docker compose start backend
```

### 6.2. Sweeping Abandoned Namespaces (`scripts/sweep.py`)
Streamlit creates temporary sessions (`web-<uuid>`). Over time, abandoned sessions consume disk and vector storage.
```bash
# Dry-run: view candidate namespaces idle for > 14 days with prefix "web-"
python -m scripts.sweep --idle-days 14 --prefix web-

# Apply deletion:
python -m scripts.sweep --idle-days 14 --prefix web- --apply
```
