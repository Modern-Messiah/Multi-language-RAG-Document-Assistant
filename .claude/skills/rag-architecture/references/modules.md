# Module reference

Every Python module outside `tests/` (package `__init__.py` files hold no code), with its
purpose and the symbols worth knowing. Use it to find where something lives or where new code
belongs. Tests are mapped in the rag-testing skill.

Contents:
- Backend shell (`app/`)
- Retrieval and ingestion pipeline (`app/rag/`)
- Clients (`clients/`, `frontend/`)
- Evaluation (`evaluation/`)
- Operator and benchmark scripts (`scripts/`)
- Import rules

## Backend shell (`app/`)

- `app/main.py` - the FastAPI application: factory, lifespan, every route, the upload pipeline, quotas and owner locks.
  - App and auth: `API_VERSION`, `create_app`, `lifespan`, module `__getattr__` (the lazy `app` for uvicorn), `router`, `require_api_key`, `PLACEHOLDER_API_KEYS`.
  - Validation: `USER_ID_PATTERN`, `USER_ID_QUERY`, `FILE_HASH_PATH`, `safe_filename` (`MAX_STEM_BYTES`, `MAX_EXT_LEN`).
  - Upload: `upload_document`, `_store_and_index`, `_retire_older_revisions`.
  - Query: `query_rag`, `query_rag_stream`, `_sse`, `_caller_model`, `_caller_reranker`.
  - Documents and namespaces: `list_documents`, `delete_document`, `clear_documents`, `_clear_namespace`.
  - Quotas and locks: `OwnerLocks` (`STRIPES = 64`, `for_owner`), `_owner_usage`, `_quota_problem`, `_quota_usage`.
  - Feedback and maintenance: `submit_feedback`, `sweep_idle_namespaces`.
- `app/config.py` - `Settings` (pydantic-settings, `env_file=".env"`, `extra="ignore"`), validators `_api_key_is_header_safe` (ASCII BACKEND_API_KEY) and `_chunk_overlap_fits_in_chunk` (CHUNK_OVERLAP below CHUNK_SIZE), and the cached `get_settings`.
- `app/models/schemas.py` - pydantic contracts: `USER_ID_FIELD`, `ChatTurn`, `QueryRequest` (question, language, user_id, history, metadata_filter), `Source`, `QueryResponse`, `UploadResponse`, `DocumentSummary`, `QuotaUsage`, `DocumentListResponse`, `DeleteResponse`, `ClearResponse`, `FeedbackRequest`, `FeedbackResponse`, `SweepEntry`, `OrphanEntry`, `SweepFailure`, `SweepResponse`; `ErrorResponse` is unused.
- `app/storage.py` - the disk half of an owner's namespace, always called through the module: `stored_files`, `size_of` (0 for a vanished file), `remove_stored_file`, `wipe_namespace` (Chroma delete, rmtree of the owner directory, `activity.forget`; the caller holds the owner lock; BM25 is not touched).
- `app/activity.py` - "last seen" markers: `ACTIVITY_DIRNAME` (`.activity`), `OWNER_NAME`, `is_owner_name`, `ActivityTracker` (`touch`, `forget`, `last_seen`, `known_owners`, `owner_dirs`, `seed_missing`, `reset_all`, `touch_failures`).
- `app/sweep.py` - idle-namespace sweep decisions, used by `POST /maintenance/sweep`: `MIN_IDLE_DAYS_TO_APPLY` (7), `SweepError`, `Plan`, `Outcome`, `plan`, `_refusal`, `execute`, `report`. Raises no HTTPException.
- `app/feedback.py` - append-only ratings log: `FEEDBACK_FILENAME`, `FeedbackStore` (`record`, `read_all`), `FeedbackStorageFull`, `store_from_settings`, and the offline helpers `read_records`, `summarise`, `unanswered_questions`, `golden_case_stub`.
- `app/backup.py` - snapshot and restore of the three data directories with a manifest of settings, counts and sha256 per member: `storage_locations`, `create_backup`, `read_manifest`, `verify_members`, `incompatibilities`, `restore_backup`, `chunk_count`, `archive_name`, `BackupError`. Works without a Settings instance: it reads environment variables (not `.env`), falling back to the Settings defaults.
- `app/byok.py` - bring-your-own-key: header names (`KEY_HEADER`, `MODEL_HEADER`, `PROVIDER_HEADER`, `RERANKER_PROVIDER_HEADER`, `RERANKER_KEY_HEADER`, `RERANKER_MODEL_HEADER`), the `PROVIDERS` URL table, `allowed_providers`, `normalize_model_name`, `wanted`, `client_for`, `describe_upstream_refusal`, `close_quietly`, `wanted_reranker`, `BringYourOwnKeyError`.
- `app/observability.py` - `REQUEST_ID_HEADER`, `RequestContextMiddleware` (raw ASGI: request id, access line, quiet probes), `RequestIdFilter`, `LOG_FORMAT`, `configure_logging`, `request_id_of`, `current_request_id`, `new_request_id`, `readiness`.
- `app/tracing.py` - optional Langfuse tracing: `get_tracer`, `LangfuseTracer` (`start_trace`, `record_feedback`, `flush`, `shutdown`), the `Langfuse*Handle` wrappers, and `NoOpTracer` with its `NoOp*Handle` classes used whenever tracing is off or fails.
- `app/humanize.py` - `human_size` and `describe_quota`; kept free of heavy imports because both clients use it.
- `app/docs/assets/` - README demo GIFs, not code (excluded from the image).

## Retrieval and ingestion pipeline (`app/rag/`)

- `app/rag/chain.py` - `RAGChain`: `ask`, `ask_stream`, the shared `_prepare`, `_retrieve`, `_retrieve_dense`, `_retrieve_diverse` (MMR), `_search_with_scores`, `_condense`, `_history_block`, `_collect_sources`, `_log_usage`; plus `CitationStripper`, `SYSTEM_PROMPT`, `NO_CONTEXT_ANSWER`, `CONDENSE_MAX_TOKENS`, `MMR_FETCH_MULTIPLIER`, and the re-exported `LANG_RULES`.
- `app/rag/embeddings.py` - `OpenAIEmbeddingFunction` (batches of 100), `distance_to_similarity`, `cosine_similarity`, `select_mmr`, `DEFAULT_SPACE`, and `EmbeddingsManager`: `get_vectorstore` with the `MODEL_METADATA_KEY` guard, `add_documents`, `has_file_hash`, `file_hashes_for_source`, `_owned`, `search_candidates`, `index_space`, `list_documents`, `list_owners` (deliberately unscoped), `delete_by_file_hash`, `delete_documents`, `ping`, `count`.
- `app/rag/bm25.py` - `tokenize_multilingual` and `BM25Index` (in memory, one RLock): `add_documents`, `delete_by_file_hash`, `clear_owner`, `count_owner`, `sync_from_chroma`, `search`.
- `app/rag/hybrid.py` - `HybridRetriever.retrieve` (owner from the filter's top-level user_id, lazy BM25 sync, fusion) and `reciprocal_rank_fusion`; `_document_key` identifies a chunk across the two lists.
- `app/rag/reranker.py` - `get_reranker`, `BaseReranker`, `NoOpReranker`, `FlashRankReranker`, `CohereReranker`, `normalize_reranker_model`, and a fallback `RerankRequest` so the module imports without flashrank.
- `app/rag/metadata_extractor.py` - `COMPANY_ALIASES`, `normalize_company`, `extract_document_metadata` (upload side), `extract_query_metadata` (question side) and `build_chroma_filter` (user_id ANDed with metadata clauses; a `user_id` key in the metadata is dropped).
- `app/rag/document_loader.py` - `SUPPORTED_EXTENSIONS` (the one list the API gate and both clients read), `TEXT_EXTENSIONS`, `DocumentLoader` (`load_document`, `load_pdf`, `load_docx`, `load_txt`).
- `app/rag/text_splitter.py` - `TextChunker` (`split_documents`, `split_text`): `RecursiveCharacterTextSplitter`, delegating to `TableAwareSplitter` when table-aware chunking is on.
- `app/rag/table_splitter.py` - `TableAwareSplitter`, `extract_segments`, `TableBlock`, `ContentSegment`, `is_markdown_table_row`, `is_markdown_separator_row`.
- `app/rag/languages.py` - `AUTO_LANGUAGE`, `AUTO_RULE`, `LANG_RULES`, `SUPPORTED_LANGUAGES`, `rule_for`; free of heavy imports so the clients can read it.

## Clients (`clients/`, `frontend/`)

- `clients/backend.py` - configuration and pure helpers shared by both clients, no transport: `backend_url` (BACKEND_URL, default `http://127.0.0.1:8000`), `api_headers` (BACKEND_API_KEY), `max_file_bytes` and `max_file_mb` (mirror MAX_FILE_SIZE), `feedback_enabled` (mirrors FEEDBACK_ENABLED), `describe_error`, `error_from_response`, `OPERATOR_ERROR`; re-exports the header names, `PROVIDERS`, `REQUEST_ID_HEADER`, `SUPPORTED_LANGUAGES`, `human_size`, `describe_quota`.
- `clients/telegram_bot.py` - python-telegram-bot polling bot, run with `python -m clients.telegram_bot`: command handlers `start`, `help_command`, `clear_command`, `documents_command`, `model_command` (/start, /help, /clear, /documents, /model); `handle_document` (POST /upload), `handle_message` (POST /query with history), `on_rating` (POST /feedback), `split_for_telegram`, `format_document_list`, `asking_headers`, `on_error`, `main`. user_id is `str(update.effective_user.id)`; per-user state (language, history, own key) lives only in memory, in python-telegram-bot's `user_data`.
- `frontend/streamlit_app.py` - Streamlit web UI, run with `streamlit run frontend/streamlit_app.py`: per-session `USER_ID` (`web-<uuid4 hex>`), sidebar upload, document list with per-document delete, clear, own-key, reranker and metadata-filter panels, language picker (`LANG_FLAGS`); answers streamed from POST /query/stream; `asking_headers`, `send_feedback`, `render_turn`, `render_sources`, `render_feedback`. It is a script, not a package module, and puts the repository root on `sys.path` itself.

## Evaluation (`evaluation/`)

Not part of the running service: `.dockerignore` keeps it out of the image.

- `evaluation/metrics.py` - pure functions `recall_at_k`, `precision_at_k`, `reciprocal_rank`, `hit_at_k`, `aggregate`; unit-tested in `tests/test_metrics.py` and used offline by `tests/test_retrieval_quality.py`.
- `evaluation/golden.py` - the small labelled set: `CORPUS`, `GOLDEN_CASES`, `UNANSWERABLE`.
- `evaluation/run_eval.py` - live harness: `python -m evaluation.run_eval --url URL --api-key KEY` uploads the golden corpus to a scratch tenant, asks every case and prints the metrics (`Backend`, `run`, `main`). Real embedding and model calls.
- `evaluation/from_feedback.py` - `python -m evaluation.from_feedback --input PATH` summarises `feedback.jsonl` and prints golden-case stubs (`main`).
- `evaluation/langfuse_eval.py` - runs a Langfuse dataset against a live backend and records an experiment run (`BackendClient`, `sync_dataset_items`, `get_corpus_and_expectations`, `run_experiment`, `main`).
- `evaluation/eval_sberquad_failures.py` - re-asks previously failed SberQuAD items against a tenant indexed earlier (`extract_failed_indices`, `run_hard_cases_benchmark`, `main`).

## Operator and benchmark scripts (`scripts/`)

Ships in the image, so a one-off container can run them against the data volumes.

- `scripts/backup.py` - `python -m scripts.backup --output DIR` archives the three data directories; refuses while something answers GET /ready unless `--live` (`backend_is_up`, `main`).
- `scripts/restore.py` - `python -m scripts.restore ARCHIVE` with `--inspect`, `--overwrite`, `--force`, `--url`, `--live`; its checks (manifest hashes, embedding model and collection name, occupied targets) all run before it writes anything.
- `scripts/sweep.py` - HTTP client of POST /maintenance/sweep: `python -m scripts.sweep` is a dry run of `web-` namespaces idle 30 days; `--idle-days`, `--prefix`, `--apply`, `--force`, `--url`, `--api-key`, `--json` (`call_sweep`, `print_report`, `main`).
- `scripts/download_benchmarks.py` - downloads six public benchmarks into `data/benchmarks` (network; no CLI, so any invocation, even with `--help`, starts downloading).
- `scripts/upload_to_langfuse.py` - `python -m scripts.upload_to_langfuse --dataset NAME` pushes downloaded benchmarks into Langfuse datasets (`--count`, `--all`).

Details for all of these are in the rag-operations and rag-evaluation skills.

## Import rules

- Only uvicorn and the tests import `app.main`. Modules it imports must not import it back at module level (`app/sweep.py` and `app/storage.py` would form a cycle). That, and the non-reentrant owner lock `sweep.execute` already holds, is why the sweep calls `storage.wipe_namespace` instead of `_clear_namespace` - and why it misses the BM25 purge.
- Some imports are deliberately inside functions: `app/main.py` imports `BM25Index`, `get_reranker`, `normalize_model_name` and `extract_document_metadata` lazily, `app/rag/chain.py` imports `HybridRetriever` and the metadata helpers lazily, and `app/byok.py` imports `get_reranker` inside `wanted_reranker`.
- Optional packages (flashrank) must never be imported unguarded at module level in `app/rag/reranker.py`: startup and `byok.wanted_reranker` import that module.
- The clients import only `app.byok`, `app.humanize`, `app.observability`, `app.rag.languages` and `app.rag.document_loader`. Keep `app/humanize.py` and `app/rag/languages.py` free of heavy imports.
- `scripts/backup.py` and `scripts/restore.py` need no OpenAI key. `evaluation/run_eval.py` uses only the standard library and `evaluation/`; `evaluation/from_feedback.py` adds `app/feedback.py`, itself standard-library only, so both run without the project requirements.
