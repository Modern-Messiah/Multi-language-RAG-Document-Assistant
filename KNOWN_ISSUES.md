# Known issues

Verified defects in the current code, grouped by area, for operators, users and developers (coding
agents included). Every entry was reproduced against the code, either with an offline probe of the
real app or by following the exact code path. The only exceptions are the two entries under
[Unverified risks](#unverified-risks), which need Docker to confirm. Each entry gives its severity, who
is affected (and whether the default settings are), what happens, a workaround and the direction of a fix.

Severity: **high** means deleted data is still served, data or credentials are exposed or lost, or a
default path is broken; **medium** means wrong, misleading or failing results under realistic
conditions; **low** means latent, narrow or cosmetic.

Whoever fixes an issue removes its entry in the same change and adds a regression test that fails on
the old behaviour. Setting names are the environment variables of `app/config.py` `Settings` (see the
Configuration table in [DOCUMENTATION.md](DOCUMENTATION.md)); code is cited by path and symbol.

Sections: [Retrieval and search](#retrieval-and-search), [Document ingestion](#document-ingestion),
[Models, keys and reranking](#models-keys-and-reranking), [Tracing and observability](#tracing-and-observability),
[Operations: backup, restore and sweep](#operations-backup-restore-and-sweep), [Web UI](#web-ui),
[Telegram bot](#telegram-bot), [Evaluation and benchmarks](#evaluation-and-benchmarks),
[Development and tests](#development-and-tests).

## Retrieval and search

### Automatic metadata filtering hides the right document in general collections

- **Severity:** high
- **Affects:** default settings (`METADATA_FILTERING_ENABLED=true`), every upload and question on every client; worst for collections that are not financial filings.
- **What happens:** Uploads are tagged with a company taken from the file name (a known alias anywhere in it, otherwise its first word if that word is all letters and at least two long, so `leave_policy.txt` becomes `LEAVE`) and a year from the file name or the opening text. Questions are filtered on any number from 1900 to 2039 ("2000 dollars", "office 1905"), on alias words such as "block", "square", "meta" or "apple", and on Q1-Q4. When such a false filter matches some other document, retrieval is restricted to it, the zero-hit fallback never runs, and the document that answers the question is never retrieved. The web UI's source list also shows misleading tags such as `company: REPORT`.
- **Workaround:** set `METADATA_FILTERING_ENABLED=false` for deployments whose documents are not financial filings (an explicit `metadata_filter` still applies, see below).
- **Fix direction:** in `app/rag/metadata_extractor.py`, take the company only from the alias table and the year only from the file name or FY patterns, and require year context ("in 2024", "FY2024") in `extract_query_metadata`; or ship the flag off.

### Keyword search is skipped for questions that trigger a metadata filter

- **Severity:** medium
- **Affects:** default settings (`HYBRID_SEARCH_ENABLED=true`, `METADATA_FILTERING_ENABLED=true`) on `POST /query` and `POST /query/stream`, for questions naming a year from 1900 to 2039 (or FY plus two digits), a known company alias, a quarter (Q1-Q4, 1Q-4Q) or a filing type (10-K, 10-Q, 8-K, earnings, annual report), and for every request that sends a non-empty `metadata_filter`.
- **What happens:** `build_chroma_filter` returns ChromaDB's `$and` form. `HybridRetriever.retrieve` looks for the owner only at the top level, finds none and skips the lazy sync; `BM25Index.search` then scans every tenant's matching chunks and rejects all of them on the `$and` key. Retrieval is dense-only for exactly these questions, logged only at DEBUG. If the filtered dense search finds nothing, the tenant-wide fallback runs hybrid search normally.
- **Workaround:** none needed for correctness; expect weaker exact-term matching on such questions. `METADATA_FILTERING_ENABLED=false` restores keyword search for questions without an explicit filter.
- **Fix direction:** in `HybridRetriever.retrieve` (`app/rag/hybrid.py`), take the owner from the `user_id` clause inside `$and` (fail closed if there is none) and pass the remaining clauses to BM25 as flat constraints; fix together with the fail-open entry at the end of this section.

### Older documents drop out of keyword search after a restart if the user uploads first

- **Severity:** medium
- **Affects:** default settings (`HYBRID_SEARCH_ENABLED=true`), after a backend restart, for an owner whose first action is a new upload rather than a question without a metadata filter; typical for Telegram users and web sessions that span a restart.
- **What happens:** The BM25 index starts empty and is filled from ChromaDB for an owner only while that owner has no BM25 entries at all. The upload adds its own chunks first, so the owner's older documents stay out of keyword matching until the next restart, or until the owner deletes every document uploaded since the restart. Dense search still finds them, but exact names, numbers and rare terms in older documents can be missed, and any shared word boosts the newest upload.
- **Workaround:** none for operators. A user avoids it by asking any question that names no year, company, quarter or filing type, with no metadata filter set, before the first upload after a restart.
- **Fix direction:** remember which owners were synced from ChromaDB in this process instead of testing `count_owner(owner) == 0`, and sync before an owner's first `add_documents` or search, or rebuild the index at startup (`app/rag/bm25.py` `BM25Index.sync_from_chroma`, `app/rag/hybrid.py`).

### RELEVANCE_THRESHOLD does not stop off-topic questions while hybrid search is on

- **Severity:** medium
- **Affects:** deployments that set `RELEVANCE_THRESHOLD` above `0.0` (the default `0.0` is not affected) with `HYBRID_SEARCH_ENABLED=true` (the default).
- **What happens:** The threshold filters only the dense candidates. BM25 hits are added afterwards without one, and when the threshold removed every dense hit, the BM25 hits are returned alone. BM25 has no stopword list, so a question sharing any word with a chunk ("the", "what", "и") still reaches the model and lists unrelated sources; "No relevant information found." appears only when the question shares no word with any chunk. No reranker applies a score floor, so reranking does not restore the cut-off.
- **Workaround:** set `HYBRID_SEARCH_ENABLED=false` if you rely on the threshold.
- **Fix direction:** gate fused candidates on relevance in `RAGChain._retrieve` and `HybridRetriever.retrieve`: drop BM25-only hits without a passing dense score, and never return BM25 hits alone after the threshold removed every dense hit.

### MMR_LAMBDA below 1.0 makes filtered questions fail with 503 "Query failed"

- **Severity:** medium
- **Affects:** only deployments with `MMR_LAMBDA` below `1.0` (any value; the default `1.0` is not affected), on `POST /query` and `POST /query/stream` from every client, for questions that trigger metadata filtering and for every request with a non-empty `metadata_filter`, including the web UI's Metadata Filter, even with `METADATA_FILTERING_ENABLED=false`.
- **What happens:** `RAGChain._retrieve_diverse` reads `filter_dict["user_id"]`, which the `$and` filter does not have, so the request fails with HTTP 503 "Query failed" and the zero-hit fallback never runs. Questions without metadata still work. Fixing only that lookup would leave MMR ignoring the company, year and doc_type constraints, because `EmbeddingsManager.search_candidates` filters by owner only.
- **Workaround:** keep `MMR_LAMBDA=1.0`.
- **Fix direction:** give `EmbeddingsManager.search_candidates` a `where` argument, pass it the full filter from `_retrieve_diverse` (`app/rag/chain.py`), and add a test that combines MMR with a metadata filter.

### "annual report" in a question discards the company and year filters too

- **Severity:** medium
- **Affects:** default settings (`METADATA_FILTERING_ENABLED=true`), any question containing "annual report" in any case or spacing (the Russian phrase is not matched).
- **What happens:** The question gets doc_type `ANNUAL REPORT`, which no upload ever carries (documents are tagged only `10-K`, `10-Q`, `8-K` or `EARNINGS`), so the filtered search finds nothing and the fallback in `RAGChain._prepare` drops every constraint at once: the company and year detected in the same question are discarded and the answer draws on all of the user's documents. The same all-or-nothing fallback applies whenever any single constraint matches nothing, for example "earnings" against a collection of 10-K filings.
- **Workaround:** leave the phrase out ("Walmart 2020 revenue") or say "10-K".
- **Fix direction:** map "annual report" to 10-K, or drop it from `DOC_TYPE_PATTERN` in `app/rag/metadata_extractor.py`, and make the fallback relax one constraint at a time.

### A span of more than five years filters on the two end years only

- **Severity:** medium
- **Affects:** default settings (`METADATA_FILTERING_ENABLED=true`), questions naming two to five distinct years that lie more than five years apart, on collections tagged by year.
- **What happens:** "What happened between 1990 and 2010?" becomes `{"year": {"$in": [1990, 2010]}}`, so documents tagged 1991 to 2009 are excluded; spans of five years or less are expanded to the full range. Because the end years usually match something, the fallback does not run and the answer silently misses the years in between.
- **Workaround:** keep the earliest and latest year named in one question at most five years apart, or leave the years out of the question.
- **Fix direction:** expand every multi-year span to the full range, or treat "between X and Y" as a range, in `extract_query_metadata` (`app/rag/metadata_extractor.py`).

### Explicit metadata filters fail as 503 when malformed and are dropped silently when they match nothing

- **Severity:** medium
- **Affects:** any settings: API clients that send `metadata_filter` on `POST /query` or `POST /query/stream`, and the web UI's Metadata Filter (see also the [Web UI](#web-ui) entry about its boxes).
- **What happens:** The field is an untyped dictionary handed to ChromaDB as is. A value ChromaDB rejects (an unknown operator such as `{"year": {"$foo": 1}}`, or a list where a single value or `{"$in": [...]}` belongs) comes back as HTTP 503 "Query failed" with an ERROR traceback in the log instead of a 422. A filter that matches none of the user's chunks is dropped without notice: the answer comes from all of their documents and the response does not say so (one INFO log line). A year sent as the string `"2020"` never matches the stored number. Tenant isolation holds: a `user_id` key in the filter is ignored and every clause is combined with the owner.
- **Workaround:** send canonical values (company as stored, such as `WALMART`; year as a number; doc_type `10-K`, `10-Q`, `8-K` or `EARNINGS`) and do not take a filtered answer as proof that the filter matched.
- **Fix direction:** replace the `Dict[str, Any]` of `QueryRequest.metadata_filter` (`app/models/schemas.py`) with a typed model so bad input is a 422, normalise the company with `normalize_company`, and say in the response when a filter matched nothing.

### Turning METADATA_FILTERING_ENABLED off does not stop explicit filters

- **Severity:** low
- **Affects:** deployments that set `METADATA_FILTERING_ENABLED=false`.
- **What happens:** The flag only stops filters being extracted from questions. A `metadata_filter` in the request, including the web UI's Metadata Filter, is still applied, while `app/config.py` and `.env.template` describe the flag as enabling "automatic/explicit" filtering. With `MMR_LAMBDA` below 1.0 such requests still fail (see above).
- **Workaround:** leave the web UI's Metadata Filter empty and send no `metadata_filter` when filtering should be off.
- **Fix direction:** decide whether the flag covers explicit filters, then make `RAGChain._prepare` and the setting's description agree.

### One tenant's uploads reorder another tenant's keyword results

- **Severity:** low
- **Affects:** default settings (`HYBRID_SEARCH_ENABLED=true`) on any deployment with more than one user.
- **What happens:** `BM25Index` computes document frequencies and the average chunk length over every tenant's chunks. When one tenant uploads many chunks containing a term, that term weighs less for everybody and the order of another user's own keyword results changes (a probe flipped a user's top two results). No content crosses tenants, but the ranking does reflect how common terms are in other tenants' documents.
- **Workaround:** none.
- **Fix direction:** keep the statistics (chunk count, document frequencies, average length) per owner in `BM25Index` (`app/rag/bm25.py`).

### The keyword index is filled and kept in memory even with hybrid search off

- **Severity:** low
- **Affects:** deployments with `HYBRID_SEARCH_ENABLED=false`.
- **What happens:** The lifespan always creates the BM25 index and every upload adds its chunks to it, but retrieval never reads it, so chunk text is held in process memory for nothing. Only `/clear`, deleting or replacing the document, or a restart frees it; an applied sweep does not (see the sweep entry under Operations).
- **Workaround:** none besides restarting the backend.
- **Fix direction:** skip creating and filling the index when `hybrid_search_enabled` is false (`app/main.py` `lifespan` and `_store_and_index`).

### ChromaDB collections created outside the app score wrongly and skip the embedding-model check

- **Severity:** low
- **Affects:** only deployments whose `CHROMA_PERSIST_DIR` and `COLLECTION_NAME` point at a collection created outside the app with `hnsw:space` in its metadata; collections the app creates carry no `hnsw:space` and are not affected.
- **What happens:** For the inner-product space (`ip`), `distance_to_similarity` (`app/rag/embeddings.py`) returns `-distance` instead of `1 - distance`, so even an exact match scores 0 or below and any `RELEVANCE_THRESHOLD` above 0 drops every dense hit: with `HYBRID_SEARCH_ENABLED=false` every answer becomes "No relevant information found.", and with hybrid search on (the default) only keyword matches are left (see the RELEVANCE_THRESHOLD entry above). For any `hnsw:space` (ip, cosine or an explicit l2), ChromaDB refuses the metadata update that records the embedding model, so "Could not record the embedding model" is logged at every startup and a later `EMBEDDING_MODEL` change is accepted instead of refused.
- **Workaround:** use only collections the assistant created itself (choose a new `COLLECTION_NAME` and re-index).
- **Fix direction:** return `1.0 - distance` for `ip` in `distance_to_similarity`, and make `EmbeddingsManager._assert_embedding_model_matches` record the model in a way ChromaDB's `modify` accepts on such collections.

### Retrieval searches every tenant when no owner is passed (latent)

- **Severity:** low
- **Affects:** no current production path; any future caller of `BM25Index.search` or `HybridRetriever.retrieve` that passes no owner, an empty one, or a filter without a top-level `user_id`.
- **What happens:** A missing owner means "every tenant" in `BM25Index.search`, and `HybridRetriever.retrieve` called without a filter and without `dense_candidates` runs an unfiltered dense search. Today every filter is `{"user_id": user_id}` or comes from `build_chroma_filter(user_id, ...)`, and `RAGChain._prepare` rejects an empty `user_id`; the `$and` case above is safe only because BM25 happens to reject every chunk on that key.
- **Workaround:** always pass the owner explicitly. A fix for the keyword-search entry above must take the owner from the `$and` clauses rather than flatten them without one.
- **Fix direction:** make the owner a required non-empty string in `BM25Index.search` and `HybridRetriever.retrieve`, and raise when it cannot be derived.

### Passing an `_owned`-shaped filter to the chain or the hybrid retriever crashes

- **Severity:** low
- **Affects:** developers adding retrieval calls; no current production path.
- **What happens:** `EmbeddingsManager._owned(owner)` builds `{"user_id": {"$eq": owner}}`. Given to `RAGChain._retrieve` or `HybridRetriever.retrieve`, it raises TypeError in `BM25Index.count_owner` with hybrid search on, and a ChromaDB error with `MMR_LAMBDA` below 1.0, so every query would answer 503. `_owned(owner, **metadata)` builds an `$and` filter that switches BM25 off (see above) and, with `MMR_LAMBDA` below 1.0, fails as described in the MMR entry.
- **Workaround:** use `EmbeddingsManager._owned` only for EmbeddingsManager's own collection calls; give `RAGChain` and `HybridRetriever` the filter from `build_chroma_filter(user_id, metadata)` (`app/rag/metadata_extractor.py`), and pass `owner=` to `BM25Index` explicitly.
- **Fix direction:** accept both shapes in `HybridRetriever.retrieve`, or reject an unknown filter shape with a clear error.

## Document ingestion

### Word table rows drop repeated and empty cells, shifting values into the wrong columns

- **Severity:** high
- **Affects:** every `.docx` table with equal values in neighbouring cells or with empty cells, any settings.
- **What happens:** To undo merged cells, `DocumentLoader.load_docx` drops any cell whose text equals the cell before it, and drops empty cells. Under the header "Role | Annual leave 2024 | Annual leave 2025 | Sick leave", the row "Engineer | 28 | 28 | 10" becomes "Engineer | 28 | 10" and "Intern | (empty) | 14 | 5" becomes "Intern | 14 | 5", so the model is given values under the wrong columns and answers from them.
- **Workaround:** none in the app; for such tables upload a Markdown or text version instead, with a compact separator row (see the next entry).
- **Fix direction:** recognise merged cells by the underlying cell element (python-docx `cell._tc`) rather than by text, and keep empty cells as empty strings (`app/rag/document_loader.py`).

### Markdown tables with spaced separator rows are chunked as plain text

- **Severity:** medium
- **Affects:** default settings (`TABLE_AWARE_CHUNKING=true`): `.txt`, `.md` and `.markdown` pipe tables with three or more columns whose separator row has a space after the pipes (`| --- | --- | --- |`, `| :--- | ---: | :--- |`, the style Prettier writes).
- **What happens:** `MD_TABLE_SEPARATOR_PATTERN` (`app/rag/table_splitter.py`) does not allow a space before a cell's dashes, so the table is not recognised and goes through the ordinary splitter: in a table longer than a chunk only the first chunk carries the header row, and rows can share a chunk with prose. Compact separators (`|---|---|---|`) and two-column tables work.
- **Workaround:** write separator rows without spaces (`|---|---|---|`) before uploading.
- **Fix direction:** allow whitespace before each cell in `MD_TABLE_SEPARATOR_PATTERN` and add three- and five-column spaced cases to `tests/test_table_splitter.py`.

### Word and PDF tables never get table-aware chunking

- **Severity:** medium
- **Affects:** every `.docx` with a real Word table and every `.pdf`, whatever `TABLE_AWARE_CHUNKING` says.
- **What happens:** `DocumentLoader.load_docx` writes each table row as its own `a | b | c` block, separated by blank lines and with no separator row, and PDF text carries no table structure, so the table splitter recognises neither and chunks are identical with the flag on or off. Rows of a long table lose their column headers once they fall into later chunks.
- **Workaround:** upload tables that matter as Markdown (`.md`) with a compact separator row.
- **Fix direction:** have `load_docx` (`app/rag/document_loader.py`) emit each table as one contiguous Markdown table in place: header, separator row, rows joined by single newlines, pipes in cells escaped.

### Word tables are moved to the end of the document text

- **Severity:** medium
- **Affects:** every `.docx` with a table, any settings.
- **What happens:** `DocumentLoader.load_docx` reads all paragraphs first and appends every table's rows after them, so a table is separated from the heading and text that introduce it and lands after the last section; chunks then pair its rows with unrelated text.
- **Workaround:** none in the app; if the position matters, convert the document to Markdown and upload the `.md` file.
- **Fix direction:** walk the document body in order in `load_docx` (python-docx `document.element.body`), emitting paragraphs and tables as they occur.

### File names with a hyphenated filing type are not tagged with it

- **Severity:** medium
- **Affects:** default settings (`METADATA_FILTERING_ENABLED=true`), uploads whose name spells the filing type with a hyphen (`Coca-Cola_2022_10-K.pdf`, `Walmart 2020 8-K.txt`) and whose first 500 characters do not name it.
- **What happens:** `extract_document_metadata` splits the name on hyphens, so "10-K" becomes "10" and "K" and no doc_type is stored. A question that mentions 10-K, or the web UI's Document Type filter, then leaves that file out; if no other file matches, the zero-hit fallback also drops the company and year scoping.
- **Workaround:** name files without the hyphen (`COCACOLA_2022_10K.pdf`).
- **Fix direction:** match filing types against the whole file name, before splitting, in `extract_document_metadata` (`app/rag/metadata_extractor.py`).

### Very large documents fail with 503 after their embeddings are paid for

- **Severity:** medium
- **Affects:** default settings: uploads that split into more chunks than ChromaDB accepts in one call (`max_batch_size`, which depends on the SQLite build: 5,461 chunks, about 4.5 MB of extracted text at the default `CHUNK_SIZE` and `CHUNK_OVERLAP`, in the development environments checked), far below `MAX_FILE_SIZE` (30 MB).
- **What happens:** `EmbeddingsManager.add_documents` writes all chunks in one call. Every chunk is embedded, and billed, first; then ChromaDB rejects the batch, the upload answers 503 "Vector store unavailable" and the file is deleted. A retry fails the same way and pays again.
- **Workaround:** split such documents into smaller files.
- **Fix direction:** upsert in slices of `client.max_batch_size` in `EmbeddingsManager.add_documents` (`app/rag/embeddings.py`), deleting what was written for that file on failure; optionally reject over-large documents up front with a clear error.

### Legacy `.doc` uploads get a generic error instead of re-save advice

- **Severity:** low
- **Affects:** any `.doc` file sent through the API, the web UI or the bot.
- **What happens:** The extension check answers first with 400 "Unsupported file type. Supported: PDF, DOCX, TXT, MD, MARKDOWN" (the web UI's file picker and the bot refuse the file with a list of formats), so the loader's advice to re-save the file is never shown.
- **Workaround:** open the file in Word or LibreOffice, save it as `.docx` or export it to PDF, and upload that.
- **Fix direction:** special-case names ending in `.doc` in `upload_document` (`app/main.py`) and in the bot's local check, returning the loader's advice as the 400 detail.

## Models, keys and reranking

### Reranker headers bypass RERANKER_ENABLED and can spend the operator's Cohere key

- **Severity:** medium
- **Affects:** `POST /query` and `POST /query/stream` at default settings, for anyone holding the API key (anyone at all when `BACKEND_API_KEY` is empty); spending the operator's key needs `RERANKER_API_KEY` to be set.
- **What happens:** `X-Reranker-Provider` turns reranking on for that request even with `RERANKER_ENABLED=false` (the default), and `ALLOWED_MODEL_PROVIDERS` does not limit it, so a caller can make the backend send retrieved chunks to the Cohere API. With `cohere` and no `X-Reranker-Key`, the operator's `RERANKER_API_KEY` pays, and `X-Reranker-Model` chooses the Cohere model it pays for.
- **Workaround:** leave `RERANKER_API_KEY` empty unless every API-key holder may spend it; if chunks must not leave the deployment, block outbound access to the Cohere API at the network level.
- **Fix direction:** in `wanted_reranker` (`app/byok.py`), refuse a per-request provider unless reranking is enabled or allowed by a setting, and require `X-Reranker-Key` for cohere instead of falling back to `RERANKER_API_KEY`.

### Choosing the flashrank reranker does nothing because the package is not installed

- **Severity:** medium
- **Affects:** `RERANKER_ENABLED=true` with `RERANKER_PROVIDER=flashrank`, or any request with `X-Reranker-Provider: flashrank`, including the web UI's Cross-Encoder Reranker picker; `flashrank` is not in `requirements.txt`, the lock files or the Docker image.
- **What happens:** The reranker cannot import flashrank, logs "Failed to initialize FlashRank (...); falling back to NoOp" and returns the candidates in vector or hybrid order without `rerank_score`. `/ready` still reports ready and the user gets no sign that nothing was reranked.
- **Workaround:** install `flashrank` into the backend's environment, or use `cohere` or no reranker.
- **Fix direction:** add `flashrank` to `requirements.txt` and both lock files (or remove it from the picker and the accepted values), and fail at startup or answer 400 when it is selected but cannot be imported (`app/rag/reranker.py`).

### A streamed answer that times out leaves the caller's model client to garbage collection

- **Severity:** low
- **Affects:** `POST /query/stream` with a caller key (`X-Model-Key`), any provider, default settings, when the model call or the retrieval embedding times out before the first event.
- **What happens:** The `APITimeoutError` branch of `query_rag_stream` (`app/main.py`) answers 504 without `byok.close_quietly(client)`, so the per-request client holding the caller's key is released by garbage collection rather than closed when the request ends. `POST /query` and the other error paths close it; no leaked sockets were observed.
- **Workaround:** none needed.
- **Fix direction:** call `byok.close_quietly(client)` in that branch, or wrap the priming step in a handler that closes the client and re-raises.

## Tracing and observability

### Feedback without a request_id is scored on the wrong Langfuse trace

- **Severity:** medium
- **Affects:** `LANGFUSE_ENABLED=true` with `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` set (tracing is off by default), `FEEDBACK_ENABLED=true` (the default), and `POST /feedback` bodies without `request_id`, which only third-party callers send (the web UI and the bot always include it).
- **What happens:** `LangfuseTracer.record_feedback` (`app/tracing.py`) calls `client.score(trace_id=None)`, and the Langfuse SDK then uses the last trace this process created, from any user or endpoint, or a random id when there is none. The rating and its comment land on someone else's trace, per-trace and per-user quality figures in Langfuse are wrong, and the app log reports "trace None". The local feedback file is unaffected.
- **Workaround:** always send the rated answer's `X-Request-ID` as `request_id`.
- **Fix direction:** skip the Langfuse score when `request_id` is missing, in `LangfuseTracer.record_feedback` or in `submit_feedback` (`app/main.py`).

### Streamed answers record no token usage

- **Severity:** medium
- **Affects:** every `POST /query/stream` at default settings, with or without Langfuse; it is the web UI's only answering path.
- **What happens:** `RAGChain.ask_stream` does not ask the provider for usage, skips chunks without choices and logs "streamed completion ..." without token counts; the Langfuse generation gets no usage either. `POST /query`, which the bot uses, logs prompt and completion tokens and passes them to Langfuse, so per-tenant spend tracking misses every web UI answer.
- **Workaround:** take web UI spend from the provider's billing; per-request token counts exist only for `POST /query`.
- **Fix direction:** send `stream_options={"include_usage": True}` in `ask_stream` (`app/rag/chain.py`), read usage from the final chunk, log it like `_log_usage` and pass it to `gen_handle.end`.

### A client-chosen X-Request-ID becomes the Langfuse trace id unchanged

- **Severity:** low
- **Affects:** `LANGFUSE_ENABLED=true`, with clients or proxies that send their own `X-Request-ID` on `POST /query` or `POST /query/stream`.
- **What happens:** Any incoming id matching `^[A-Za-z0-9._-]{1,64}$` is kept by `RequestContextMiddleware` (`app/observability.py`) and passed to `client.trace(id=...)`. Requests that reuse an id, from different users or from a proxy that repeats ids, therefore send their traces under the same id, and a rating sent for that id is scored against it. The Langfuse SDK documents that traces are upserted on id, so the server is expected to merge them into one trace; that last step was not observed, because it needs a Langfuse server.
- **Workaround:** let the backend generate request ids, or make sure proxies send unique ones.
- **Fix direction:** keep client-supplied ids out of the Langfuse trace id, for example by deriving the trace id from a server-generated value that is returned to the client for feedback.

## Operations: backup, restore and sweep

### Swept documents stay searchable until the backend restarts

- **Severity:** high
- **Affects:** default settings, after any applied sweep (`python -m scripts.sweep --apply`, or `POST /maintenance/sweep` with `apply=true`); answering from swept text needs `HYBRID_SEARCH_ENABLED=true` (the default), but the memory is held either way.
- **What happens:** `sweep.execute` (`app/sweep.py`) calls `storage.wipe_namespace`, which deletes the vectors, the files and the activity marker but not the owner's entries in the in-memory BM25 index; only `/clear` or a restart removes those. If the same `user_id` asks again before a restart (a Telegram id or API tenant swept with a non-default prefix, or a web tab that is still open), answers are built from the deleted documents and list them as sources, while `GET /documents` shows nothing. The swept text stays in memory until a restart.
- **Workaround:** restart the backend after an applied sweep, or call `POST /clear` for each swept `user_id`.
- **Fix direction:** call `state.bm25_index.clear_owner(user_id)` in `wipe_namespace` (`app/storage.py`) so `/clear` and the sweep share one purge, and assert in `tests/test_sweep.py` that swept owners have no BM25 entries.

### Restoring into the Compose volumes fails after emptying the index volume

- **Severity:** high
- **Affects:** `python -m scripts.restore` run in a one-off backend container of the shipped Compose setup, the usual way to reach the named volumes: `CHROMA_PERSIST_DIR`, `UPLOAD_DIR` and `FEEDBACK_DIR` resolve to the mount points `/app/data/chroma_db`, `/app/data/uploads` and `/app/data/feedback`. Any restore whose target directory is a mount point fails; host restores into plain directories work.
- **What happens:** `restore_backup` (`app/backup.py`) deletes each target directory with `shutil.rmtree` and then moves the restored copy into its place. Removing a mount point fails ("Device or resource busy") after its contents are already gone, and the error escapes as a traceback rather than a `BackupError`. With `--overwrite` on a live deployment this leaves the ChromaDB volume empty while uploads and ratings remain; on fresh, empty volumes the restore fails the same way.
- **Workaround:** keep the archive and do not restore with the volumes as targets. With the backend stopped, restore into plain directories instead (point `CHROMA_PERSIST_DIR`, `UPLOAD_DIR` and `FEEDBACK_DIR` at empty, non-mounted paths inside the same one-off container), then empty the volume directories and copy the restored contents into them.
- **Fix direction:** replace the contents of each target directory instead of the directory itself, and turn an `OSError` during the swap into `BackupError`.

### Host-run backup and restore ignore .env

- **Severity:** medium
- **Affects:** `python -m scripts.backup` and `python -m scripts.restore` run on the host when `CHROMA_PERSIST_DIR`, `UPLOAD_DIR`, `FEEDBACK_DIR`, `EMBEDDING_MODEL`, `COLLECTION_NAME` or `BACKEND_URL` are set only in `.env`. Compose one-off containers are not affected (`env_file` exports the values), nor is a `.env` that only repeats the defaults.
- **What happens:** The scripts read only the process environment; they never load `.env` or build `Settings`. A backup exits 0 with an archive of the default `./data` paths and a manifest naming the default model and collection; `--inspect` reports the wrong fit; a restore refuses a correct archive, or with `--force --overwrite` restores into `./data`, which the backend never reads; and the running-backend check ignores `BACKEND_URL`.
- **Workaround:** export the variables first (`set -a; . ./.env; set +a` in bash) and check the counts, embedding model and collection the backup prints (`--json` also lists every archived file).
- **Fix direction:** call python-dotenv's `load_dotenv()` at the start of `main()` in `scripts/backup.py` and `scripts/restore.py`, as `scripts/sweep.py` does.

### Backup archives store most files two or three times

- **Severity:** medium
- **Affects:** every backup (`python -m scripts.backup`, on the host or in a container).
- **What happens:** `create_backup` adds every path from `staging.rglob("*")` with `tar.add`, which is recursive by default, so each file is stored once for itself and once more for every directory above it: the HNSW segments and uploads three times, `chroma.sqlite3` and the feedback file twice. Archives are about two to three times the data size, and writing, copying and extracting them takes correspondingly longer; restored data is correct.
- **Workaround:** plan disk space and transfer time for the larger archives.
- **Fix direction:** pass `recursive=False` to `tar.add` in the `create_backup` loop (`app/backup.py`) and add a test that archive member names are unique.

### The tracked uploads placeholder counts as data in backup and restore

- **Severity:** low
- **Affects:** backups and restores on a checkout whose data directories hold nothing but the tracked `data/uploads/.gitkeep`, such as a fresh clone or a host run with the wrong paths (see the `.env` entry above).
- **What happens:** `create_backup` only checks that a data directory exists, so such a backup exits 0 with an archive holding just `uploads/.gitkeep` and reports "uploads: 1". In the other direction, `restore_backup` treats the `.gitkeep` file as existing data and refuses to restore into a fresh clone without `--overwrite` ("These directories already hold data").
- **Workaround:** check the counts a backup prints before relying on it; on a fresh clone, confirm the target directories hold only `.gitkeep`, then restore with `--overwrite`.
- **Fix direction:** ignore placeholder files such as `.gitkeep` in the emptiness checks and the upload count in `app/backup.py`.

### Unverified risks

These two follow from `Dockerfile` and `docker-compose.yml` but could not be reproduced, because Docker was not available during verification. Confirm or rule them out before relying on either.

#### Ratings may fail on a fresh Docker deployment

- **Severity:** high
- **Affects:** a fresh Compose deployment built from a checkout without a `data/feedback` directory (a fresh clone has none), with `FEEDBACK_ENABLED=true` (the default).
- **What happens (expected):** The Dockerfile creates, and hands to `appuser`, only `data/uploads` and `data/chroma_db`. The `data_feedback` volume's mount point does not exist in the image, so Docker creates it owned by root and the backend, running as `appuser`, cannot write: `POST /feedback` answers 503 "Feedback storage unavailable", the web UI shows that warning and the bot answers "Could not send that, sorry.".
- **Workaround:** if ratings fail this way, fix the ownership once with `docker compose run --rm -u root backend chown appuser:appuser /app/data/feedback`, or set `FEEDBACK_ENABLED=false`.
- **Fix direction:** add `data/feedback` to the Dockerfile's `mkdir -p data/uploads data/chroma_db` step, before the `chown`.

#### One-off Compose backups and restores may not see the running backend

- **Severity:** medium
- **Affects:** the Compose one-off form `docker compose run --rm ... backend python -m scripts.backup` (and the same for restore) without `--url`.
- **What happens (expected):** The one-off container gets `BACKEND_URL` from `.env` (`http://localhost:8000` in `.env.template`) or the default `http://127.0.0.1:8000`, where nothing listens inside that container, so the "backend is still running" refusal never fires. If the backend was not stopped, a backup copies a ChromaDB that is being written and a restore replaces files under a running backend.
- **Workaround:** always run `docker compose stop backend` first, or add `--url http://backend:8000` to the command.
- **Fix direction:** set `BACKEND_URL=http://backend:8000` in the backend service's `environment` in `docker-compose.yml` (the backend itself never reads it), or put `--url http://backend:8000` into the documented commands.

## Web UI

### One empty or cut-short answer breaks every later question in the session

- **Severity:** high
- **Affects:** the web UI at default settings, whenever a streamed answer returns 200 but carries no text: the model fails before its first token (a "The answer was cut short." warning appears), or the reply is empty (a content filter, a reasoning model using up `MAX_ANSWER_TOKENS`, or a reply made only of citation markers).
- **What happens:** The empty answer is stored in the session transcript and sent back as history with the next question, and the API rejects history turns with an empty answer, so every later question in that browser tab fails with "String should have at least 1 character", even after the backend recovers.
- **Workaround:** click "Clear all documents" (which deletes the documents) or reload the page (which starts a new, empty document space), then upload again.
- **Fix direction:** in `frontend/streamlit_app.py`, do not append a turn with an empty answer to the transcript, or leave such turns out of the history it sends.

### After "Clear all documents", new answers can show as already rated

- **Severity:** medium
- **Affects:** the web UI with `FEEDBACK_ENABLED=true` (the default), after a user rated answers, clicked "Clear all documents" and asked again in the same tab.
- **What happens:** Ratings are remembered by transcript position and the Clear button does not reset them, so the first answers of the new conversation show "Thanks - recorded." instead of rating buttons and cannot be rated.
- **Workaround:** none besides reloading the page, which starts a new document space.
- **Fix direction:** also clear `feedback_given` in the Clear handler, or key ratings by a per-turn id such as the request id (`frontend/streamlit_app.py`).

### A failed upload is not retried for the rest of the browser session

- **Severity:** medium
- **Affects:** the web UI, any upload failure, including temporary ones: backend unreachable, a read timeout after 120 seconds, or 503 "Vector store unavailable" after an embedding or rate-limit failure.
- **What happens:** The failure is remembered by file name and content for the whole session, so selecting the same file again sends no request and the old error stays in the sidebar. Only deleting a document, "Clear all documents" or a page reload resets it.
- **Workaround:** rename the file and select it again; a page reload also works but starts a new document space.
- **Fix direction:** remember only deterministic rejections (client errors such as 400 and 413) in `failed_files` (`frontend/streamlit_app.py`), and show transient failures for that run only or offer a retry.

### Switching the provider clears the model typed in the key panel

- **Severity:** medium
- **Affects:** the web UI's "Answer on my own API key" panel, any settings, whenever the user types a model and then picks a different provider.
- **What happens:** The Model box's placeholder depends on the chosen provider, and with the pinned Streamlit 1.34 a widget whose parameters change between reruns is created anew, even with a fixed `key`, so `own_model` is reset to empty. The next question then sends no `X-Model`, the deployment's `MODEL_NAME` goes to the chosen provider, and most providers refuse it with a 400.
- **Workaround:** pick the provider first and then type the model, or type the model again after switching.
- **Fix direction:** keep the Model widget's parameters stable (show the per-provider suggestion in a caption instead of the placeholder) or restore the typed value from session state (`frontend/streamlit_app.py`), with an AppTest that types a model and then switches the provider.

### The Metadata Filter's company and year boxes match only exact stored values

- **Severity:** medium
- **Affects:** the web UI's Metadata Filter expander, any settings.
- **What happens:** The Company / Ticker value is only upper-cased, never normalised, so tickers (`KO`), multi-word or hyphenated names ("Coca Cola") and suffixes ("Inc") match nothing and the filter is dropped silently (see the explicit-filter entry under Retrieval and search); because it replaces the company detected in the question, typing one can make filtering worse than leaving the box empty. The Year box takes a single year; a list such as the placeholder's "2020, 2022" is ignored.
- **Workaround:** type the company as it appears in the source tags (for example `WALMART` or `COCACOLA`) and a single year.
- **Fix direction:** send `normalize_company()` output, or offer the known companies as a list, and accept year lists (`frontend/streamlit_app.py`).

## Telegram bot

### `/model <key>` without a model leaves the key in the chat

- **Severity:** high
- **Affects:** the Telegram bot, `/model` with exactly one argument, including `/model@BotName <key>` in groups.
- **What happens:** With a single argument the command replies with its usage text and returns before deleting the message, so the API key stays in the chat history (visible to every member of a group) and the bot neither deletes it nor warns; the usage text ("A model needs your own API key with it") even suggests no key was sent.
- **Workaround:** delete that message yourself (in a group, also rotate the key), then send `/model <model> <key>` or `/model <provider> <model> <key>`.
- **Fix direction:** in `model_command` (`clients/telegram_bot.py`), try to delete the message, and warn if that fails, whenever it has arguments other than `reset`, before validating them.

### `/model <key> <model>` with swapped arguments repeats the key in the bot's reply

- **Severity:** high
- **Affects:** the Telegram bot, `/model` with the key and the model in the wrong order.
- **What happens:** The two words are stored the other way round, so the key is taken as the model name. The user's message is deleted, but the confirmation "Answers now come from ... on your key." prints the key in the bot's own message, which is never deleted.
- **Workaround:** delete the bot's confirmation (in a group, also rotate the key), send `/model reset`, then use `/model <model> <key>`.
- **Fix direction:** in `model_command`, do not repeat user-supplied arguments verbatim in the confirmation.

### `/model <provider> <key>` sends the key to the deployment's default endpoint

- **Severity:** medium
- **Affects:** the Telegram bot, `/model` with a provider name and a key but no model.
- **What happens:** A provider is only recognised when three arguments are given, so the provider name is stored as the model and no provider is kept. The bot confirms, for example, "Answers now come from anthropic on your key.", and questions then send that provider's key to the deployment's default endpoint (`OPENAI_BASE_URL`, or OpenAI when it is empty).
- **Workaround:** always give all three: `/model <provider> <model> <key>`; send `/model reset` to undo.
- **Fix direction:** in `model_command`, treat a first argument that names a provider as the provider whatever the argument count, and ask for the model when it is missing.

### An empty answer breaks every later question until `/start`

- **Severity:** medium
- **Affects:** the Telegram bot at default settings, when `POST /query` returns an empty answer (empty model output, or a reply made only of citation markers).
- **What happens:** The bot stores the empty answer in its conversation history and sends it with the next question; the API rejects history turns with an empty answer, so every later question gets "String should have at least 1 character".
- **Workaround:** send `/start` (keeps your documents) or `/clear` (deletes them); both reset the history.
- **Fix direction:** in `handle_message` (`clients/telegram_bot.py`), record a turn only when its answer is not empty.

### `/documents` fails once the list no longer fits in one message

- **Severity:** medium
- **Affects:** the Telegram bot's `/documents` at default settings (`MAX_DOCUMENTS_PER_USER=200`), once the listing exceeds Telegram's 4096-character message limit: roughly 55 to 110 documents depending on file-name length, fewer with very long names.
- **What happens:** `format_document_list` builds a single HTML message; Telegram rejects it and the user sees "An error occurred.". The documents are still indexed and still answer questions, but the bot's only inventory view, which its upload-limit message points to, is gone.
- **Workaround:** none in the bot; an operator can list the documents with `GET /documents?user_id=<telegram user id>`.
- **Fix direction:** split the listing across several messages within `MessageLimit.MAX_TEXT_LENGTH`, as `split_for_telegram` does for answers, or paginate it, in `documents_command`.

## Evaluation and benchmarks

### `langfuse_eval` crashes on a dataset that does not exist yet

- **Severity:** medium
- **Affects:** `python -m evaluation.langfuse_eval`, including `--sync-only`, with the pinned `langfuse==2.60.10`, whenever looking up the dataset fails: the first run against a project without `multilingual-rag-golden`, a `--dataset` name that does not exist yet (a typo, or `eval-sberquad-ru` before `scripts/upload_to_langfuse.py` ran), or any other lookup error (401, 5xx, timeout), all of which are treated as "missing".
- **What happens:** `sync_dataset_items` creates the dataset and then reads `.items` from what `create_dataset` returns, which has none: `AttributeError: 'Dataset' object has no attribute 'items'`. The dataset now exists, so a second run with the default name works; a second run with a misspelled name uploads the golden corpus, evaluates nothing, prints zero metrics and the success banner, and leaves an empty stray dataset in Langfuse.
- **Workaround:** run the command again after the first crash; check the `--dataset` spelling and delete stray datasets in the Langfuse UI.
- **Fix direction:** in `sync_dataset_items` (`evaluation/langfuse_eval.py`), catch only the not-found error, fetch the dataset again after creating it, and auto-create only the built-in golden dataset.

### `eval_sberquad_failures` only works on the author's setup

- **Severity:** medium
- **Affects:** `python -m evaluation.eval_sberquad_failures` with its defaults.
- **What happens:** Without `--tenant` every query goes to a fixed tenant (`eval-528d05007556`) that is empty on any other backend, so every case prints FAIL without a warning. Without a usable `--log-file`, 20 hard-coded indices up to 134 are used, so a smaller dataset (the default upload has 100 items) crashes with IndexError part way; a log file is parsed only when its `[i/N]` counter equals the current dataset size, otherwise the hard-coded list is used silently. `--top-k` has no effect (hits are always counted at 3 and 5), `--limit 0` raises ZeroDivisionError, and the banner's claims about earlier failures, the pipeline and a 0.0% baseline are fixed text, not measurements.
- **Workaround:** always pass `--tenant` (a tenant that already holds the SberQuAD corpus) and a `--log-file` from a full run over the current dataset, and upload at least 134 SberQuAD items (`python -m scripts.upload_to_langfuse --dataset sberquad --count 134`, or `--all`).
- **Fix direction:** require `--tenant` and a source of failures, drop the hard-coded indices, bounds-check indices against the dataset, honour `--top-k` and print only measured values (`evaluation/eval_sberquad_failures.py`).

### The RAGTruth benchmark cannot be uploaded from the downloaded file

- **Severity:** medium
- **Affects:** `python -m scripts.download_benchmarks` followed by `python -m scripts.upload_to_langfuse --dataset ragtruth`, or `--dataset all` (the default).
- **What happens:** The download saves RAGTruth's `response.jsonl` (model responses) as `ragtruth/ragtruth_qa.jsonl`, but the upload reads `ragtruth/source_info.jsonl` (the prompts it expects), so it logs an ERROR, skips RAGTruth, prints "Upload to Langfuse completed!" and exits 0; renaming the file does not help because it is the wrong file. More generally the upload prints that banner whenever a dataset is skipped and logs per-item upload errors only at DEBUG, so each dataset's "N / M items uploaded successfully." log line is the only sign of trouble.
- **Workaround:** place RAGTruth's `source_info.jsonl` (in the `dataset/` folder of the RAGTruth repository) at `data/benchmarks/ragtruth/source_info.jsonl`, and check each dataset's "items uploaded successfully" line rather than the final banner.
- **Fix direction:** have `download_ragtruth` (`scripts/download_benchmarks.py`) fetch `source_info.jsonl` to the path the uploader reads, from one shared constant, and make `main()` in `scripts/upload_to_langfuse.py` exit non-zero when a dataset is skipped.

### `langfuse_eval` aggregate metrics count unanswerable questions as misses

- **Severity:** medium
- **Affects:** every `python -m evaluation.langfuse_eval` run whose items include questions without expected sources (the golden set's two unanswerable questions, SberQuAD items without context, FinanceBench items without evidence text), and every run in which some queries fail.
- **What happens:** The `--- Aggregate Metrics ---` block averages over every item whose query returned: unanswerable items count as misses for hit rate, precision and MRR but as perfect recall, and items whose query raised are left out. On the built-in golden set the hit rate cannot exceed 0.778 even when all nine items PASS, and the block disagrees with the per-item PASS/FAIL lines and the Langfuse scores of the same run.
- **Workaround:** compare runs on the per-item PASS/FAIL lines or the Langfuse scores; `python -m evaluation.run_eval` aggregates answerable questions only.
- **Fix direction:** aggregate answerable items only, as `evaluation/run_eval.py` does, and report the unanswerable "retrieved nothing" rate and the number of failed queries separately.

### Benchmark corpora over 200 documents run into the upload quota silently

- **Severity:** medium
- **Affects:** `python -m evaluation.langfuse_eval` without `--tenant` on a corpus of more than 200 documents (for example SberQuAD uploaded with `--all`), against a backend with the default `MAX_DOCUMENTS_PER_USER=200`.
- **What happens:** Every document goes into one scratch tenant, so uploads past 200 get HTTP 413. The script prints "Warning: failed to upload ...: HTTP Error 413: Request Entity Too Large" without the server's explanation and carries on; the questions whose documents were refused are scored as retrieval failures in Langfuse, and the run ends with the success banner. The "Uploaded i/N documents" progress counts attempts, not successes, and its last line is missing when the final upload fails.
- **Workaround:** set `MAX_DOCUMENTS_PER_USER=0` (no limit) or a higher limit on the backend before large benchmark runs, or use `--limit`.
- **Fix direction:** in `run_experiment` (`evaluation/langfuse_eval.py`), stop on a 413 and print the server's detail, or compare the corpus size with `quota.max_documents` from `GET /documents` first.

### `scripts/download_benchmarks.py` starts downloading on any invocation, even `--help`

- **Severity:** low
- **Affects:** `python -m scripts.download_benchmarks` with any arguments.
- **What happens:** The script parses no arguments: every run immediately starts downloading, from the network into `data/benchmarks/` (git-ignored), each of the six benchmarks whose file is not already there (an existing file over 1000 bytes is skipped).
- **Workaround:** run it only when you mean to download; read the module docstring instead of `--help`.
- **Fix direction:** add an `argparse` parser with `--help` (and a choice of datasets) to `main()` in `scripts/download_benchmarks.py`.

## Development and tests

### A `.env` file can change test results

- **Severity:** low
- **Affects:** developers running pytest with a `.env` in the repository root or any parent folder (for example after `cp .env.template .env`); users are not affected.
- **What happens:** `frontend/streamlit_app.py` (re-run by every web UI test) and `clients/telegram_bot.py` call `load_dotenv()` at import, which loads that file into the test process after `tests/conftest.py` stripped the settings variables once per session. Later tests then see the `.env` values, so results depend on the file and on test order: a test can fail (for example `test_quotas_ship_on` in `tests/test_quotas.py`) or pass while exercising non-default settings.
- **Workaround:** move `.env` aside while running the suite.
- **Fix direction:** neutralise `dotenv.load_dotenv` in `tests/conftest.py` before collection, or make the environment strip run per test.
