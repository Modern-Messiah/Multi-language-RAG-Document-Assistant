# Test index

One line per module in `tests/`: what it covers, and the local fixtures or doubles worth copying.
Grouped by area; within an area, start with the first file listed. Every module opens with a
docstring saying what it covers and usually which failure prompted it: read that first.

## Harness and contracts

- `tests/conftest.py` - not a test module: the environment strip, socket guard, fake embeddings, ChromaDB release, `api`, `make_settings` and `FakeChatClient` (see SKILL.md).
- `tests/test_offline_guarantee.py` - guards the guard: outbound `create_connection` and `connect` raise ("must stay offline"), loopback works, embeddings are the 32-dim fake, `api` carries a FakeChatClient, OPENAI_API_KEY is the dummy, `RAGChain.ask` refuses a missing user_id before searching.
- `tests/test_docs_consistency.py` - README.md, DOCUMENTATION.md, `.env.template`, compose and CI against the code: config rows and defaults, retired claims, the chunking line, bot commands and /help, the bot entry point, the production compose command, the compose probe, `python -m` modules, the CI lint list, an API-reference heading per route.
- `tests/test_agent_docs.py` - CLAUDE.md, KNOWN_ISSUES.md and the skills: frontmatter and size, the CLAUDE.md skills table, backticked repo paths, `python -m` modules, machine-specific paths; and, in README.md and DOCUMENTATION.md too, relative links and their anchors, KNOWN_ISSUES.md sections cited by name and the full lint command.
- `tests/test_language_consistency.py` - one language table (`app/rag/languages.py`) and its derived copies: the chain, `clients/backend.py`, the bot keyboard (three per row), the Streamlit LANG_FLAGS literal read with ast.
- `tests/test_lockfile.py` - both locks pinned, hashed and headed for linux / CPython 3.10, covering requirements*.txt; the dev lock a superset with equal versions; extras kept; posthog below 4; no pytest or ruff in the runtime lock.
- `tests/test_config.py` - Settings validation (overlap below chunk size, ranges, valid defaults), the OpenAI key reaching components from Settings rather than os.environ, settings wired into the chain and chunker.

## HTTP API

- `tests/test_api.py` - X-API-Key (401s, empty BACKEND_API_KEY disables auth, non-ASCII keys), user_id validation (422), upload filename traversal, dotfiles and long names, per-owner dedup, /health, 500 and 503 mapping, /clear removing vectors and files for the caller only, cross-tenant isolation.
- `tests/test_upload_errors.py` - upload failures: corrupt, empty or whitespace files give 400 and leave no file, an indexing failure cleans up, MAX_FILE_SIZE with a readable message, extension checks; `_pdf()` builds a minimal PDF in memory.
- `tests/test_document_management.py` - GET /documents (one entry per document, sorted, PDF pages, tenant-scoped) and DELETE /documents/{file_hash} (404 for another owner, raw file removed), same-name uploads replacing the old revision.
- `tests/test_query_e2e.py` - /query end to end with FakeChatClient: answer and sources, retrieved context in the prompt, no model call on an empty corpus, tenant scoping, the language rule, model, temperature and top_k from settings.
- `tests/test_streaming.py` - CitationStripper across chunk boundaries, ask_stream event order (sources first, done last), /query/stream SSE and no-buffering headers, errors before the first byte as status codes, a mid-stream error event; `StreamingChat` and `_sse_events` helpers.
- `tests/test_conversations.py` - follow-up condensing from history, MAX_HISTORY_TURNS bounds, fallback to the original question, the request model's history cap.
- `tests/test_resilience.py` - the max_tokens cap, usage and finish_reason logging, explicit timeout, retries and base_url on both OpenAI clients, 429 with Retry-After, 504 and 503 mapping, request handlers that are plain def.
- `tests/test_observability.py` - X-Request-ID generation, propagation and sanitising, the id on log lines (`stamped`), one access line per request, probe logging, /ready semantics (503 before startup, no auth, no OpenAI call), crash JSON naming the request, client error descriptions.
- `tests/test_quotas.py` - MAX_DOCUMENTS_PER_USER and MAX_BYTES_PER_USER (413, dedup and revisions at the limit, per owner, 0 means off), the owner lock under a forced race, OwnerLocks striping, quota wording (the 413 detail, describe_quota, the bot's listing and advice), files vanishing mid-scan; `limited` fixture.
- `tests/test_feedback.py` - FeedbackStore (append-only JSONL, concurrent writes, the 507 cap), POST /feedback (404 when disabled, 503 on a write failure), evaluation.from_feedback summaries and golden stubs, the clients' FEEDBACK_ENABLED mirror, Telegram rating buttons and 64-byte callback data; `api_without_feedback`, `FakeMessage`.
- `tests/test_sweep.py` - activity markers, POST /maintenance/sweep planning, refusals, apply and orphan files, the scripts.sweep CLI routed through the TestClient (`via_api`, load_dotenv stubbed), startup seeding.
- `tests/test_backup.py` - app.backup snapshot, manifest, hashing, verify and restore (path escapes and symlinks refused), an API round trip after a restore, the scripts.backup and scripts.restore CLIs (refusing while the backend answers, --live, --inspect, --json), activity markers in the archive; `offline_backend`, `live_backend`.

## Retrieval

- `tests/test_chain.py` - RAGChain prompt assembly, language rules, citation stripping, sources deduplicated and numbered with a 200-character preview, model and temperature passed through; `FakeVectorStore`, `FakeOpenAI`.
- `tests/test_relevance.py` - distance to similarity per index space, the inclusive RELEVANCE_THRESHOLD (no model call when everything is dropped), score logging, tenant-scoped scored search, the 0.0 default; `ScoredStore`, `UnscoredStore`.
- `tests/test_mmr.py` - the cosine helper and MMR selection, chain MMR on a crowded bag-of-words corpus (`crowded`), threshold before diversity, the 1.0 default and its wiring.
- `tests/test_retrieval_quality.py` - golden-corpus floors (recall, MRR and hit rate) through the real retrieval path with a bag-of-words embedder (`indexed`), tenant scoping, threshold effects.
- `tests/test_bm25.py` - BM25Index multilingual tokenising, Russian ranking, tenant isolation, delete and clear, metadata filter.
- `tests/test_hybrid.py` - reciprocal_rank_fusion and HybridRetriever, including the fallback when dense retrieval fails.
- `tests/test_metadata_filtering.py` - metadata from file names, text and questions, build_chroma_filter, the chain's explicit and automatic filters with the tenant-wide fallback.
- `tests/test_reranker.py` - NoOp, FlashRank (also with the package missing) and Cohere (httpx mocked) rerankers, get_reranker and wanted_reranker, chain integration, reranker model aliases.
- `tests/test_embeddings.py` - EmbeddingsManager deletes by collection and by filter, the embedding model recorded in collection metadata, a mismatch refused, a legacy collection adopted.
- `tests/test_embedding_function.py` - the real OpenAIEmbeddingFunction body (batching, order, model, query unwrap, no proxy env vars) after `monkeypatch.undo()`; `StubEmbeddings`.

## Ingestion

- `tests/test_document_loader.py` - load_txt decoding cp1251 and stripping a UTF-8 BOM.
- `tests/test_docx_markdown.py` - SUPPORTED_EXTENSIONS as the one format list (named in `app/main.py`, the web UI and the bot), DOCX paragraphs, tables and merged cells built with python-docx (`_docx`), Markdown kept verbatim, .doc advice, uploads through the API.
- `tests/test_text_splitter.py` - TextChunker boundaries, overlap, metadata, upload-wide numbering, statistics.
- `tests/test_table_splitter.py` - Markdown and HTML table detection, tables kept whole, headers repeated on split, the TextChunker table-aware switch.

## Providers and tracing

- `tests/test_byok.py` - X-Model-Key, X-Model and X-Model-Provider validation, the caller's key never stored and its per-request client closed (streaming too), the pinned PROVIDERS set and per-provider endpoints, ALLOWED_MODEL_PROVIDERS, upstream refusal mapping, the bot's /model command; `Refusing`, `_model_command`.
- `tests/test_tracing.py` - NoOpTracer and get_tracer selection, LangfuseTracer trace and feedback score with `langfuse.Langfuse` mocked, /feedback and /query calling the tracer.

## Clients

- `tests/test_clients_backend.py` - `clients/backend.py`: backend_url, api_headers, the MAX_FILE_SIZE mirror, describe_error and error_from_response (auth failures hidden, pydantic lists flattened, non-JSON bodies).
- `tests/test_frontend_feedback.py` - the Streamlit app run headless with `requests` replaced (`FakeRequests`, `fake_backend`): rating buttons, history and request id sent, the `web-` session id, the quota box, the own-key, model and provider sidebar.
- `tests/test_telegram_splitting.py` - split_for_telegram under the message limit: entities never cut, paragraph boundaries preferred, nothing lost.

## Evaluation

- `tests/test_metrics.py` - `evaluation/metrics.py` recall, precision, reciprocal rank, hit and aggregate, all hand-checkable.
- No tests: `evaluation/run_eval.py`, `evaluation/langfuse_eval.py`, `evaluation/eval_sberquad_failures.py`, `scripts/download_benchmarks.py`, `scripts/upload_to_langfuse.py`.
