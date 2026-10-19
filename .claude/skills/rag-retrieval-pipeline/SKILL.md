---
name: rag-retrieval-pipeline
description: Explains and changes how a question becomes an answer in app/rag/chain.py - follow-up condensation, dense retrieval with RELEVANCE_THRESHOLD or MMR, hybrid BM25 plus RRF fusion, how metadata filters are applied (the extractor's own rules are rag-document-ingestion), reranking, prompt and language rules, citation stripping and streaming, including the known defects on this path. Use when tuning retrieval settings, debugging irrelevant, missing or wrong-language answers, editing chain, embeddings, bm25, hybrid, reranker or languages, or adding a pipeline stage or answer language.
---

# Retrieval pipeline: from question to answer

`POST /query` and `POST /query/stream` both end in one `RAGChain` (`app/rag/chain.py`), built once in
the lifespan. `RAGChain._prepare` does everything up to the chat request: filters, condensing, dense
retrieval, BM25 fusion, the tenant-wide fallback, reranking, the prompt and the sources. `ask` and
`ask_stream` only send that request and clean up the reply. Several verified defects are live on this
path; they are listed near the end with their conditions.

## Use this skill for / not for

Use it for: tuning the retrieval settings below; "the right chunk was not retrieved", "No relevant
information found.", answers in the wrong language; editing `app/rag/chain.py`, `app/rag/embeddings.py`
(distances, MMR), `app/rag/bm25.py`, `app/rag/hybrid.py`, `app/rag/reranker.py`, `app/rag/languages.py`;
adding a pipeline stage, an answer language or a reranker provider.

Not for: request fields, routes, SSE framing and status codes (rag-api-endpoints); which model answers,
BYOK keys, the X-Reranker-* header policy (rag-llm-providers); how chunks get their
company/year/quarter/doc_type tags, the extractor's aliases and patterns (rag-document-ingestion); live
measurement, golden cases, benchmarks (rag-evaluation); fixtures and fakes (rag-testing); a Settings
field end to end, the doc-drift tests (rag-config-and-docs); running the sweep, backups, the Langfuse
stack (rag-operations); the big picture (rag-architecture); review checklists (rag-change-review).

## Map

| File | Responsibility | Key symbols |
| --- | --- | --- |
| `app/rag/chain.py` | the whole query path | `RAGChain.__init__`, `_prepare`, `ask`, `ask_stream`, `_condense`, `_retrieve`, `_retrieve_dense`, `_retrieve_diverse`, `_collect_sources`, `CitationStripper`, `SYSTEM_PROMPT`, `NO_CONTEXT_ANSWER` |
| `app/rag/hybrid.py` | BM25 plus dense fusion | `HybridRetriever.retrieve`, `reciprocal_rank_fusion`, `_document_key` |
| `app/rag/bm25.py` | in-memory keyword index | `BM25Index`, `tokenize_multilingual` |
| `app/rag/embeddings.py` | distance to cosine, MMR, Chroma access | `distance_to_similarity`, `select_mmr`, `EmbeddingsManager.search_candidates`, `index_space`, `_owned` |
| `app/rag/metadata_extractor.py` | query-side filters | `extract_query_metadata`, `build_chroma_filter` |
| `app/rag/reranker.py` | rerankers | `get_reranker`, `NoOpReranker`, `FlashRankReranker`, `CohereReranker` |
| `app/rag/languages.py` | answer languages | `LANG_RULES`, `SUPPORTED_LANGUAGES`, `AUTO_RULE`, `rule_for` |
| `app/main.py` | wiring, handlers, BM25 upkeep | `lifespan`, `query_rag`, `query_rag_stream`, `_caller_reranker`, `_store_and_index`, `_retire_older_revisions`, `delete_document`, `_clear_namespace` |
| `app/tracing.py` | Langfuse or no-op spans | `get_tracer`, `NoOpTraceHandle` |

Tests: `tests/test_chain.py` (prompt, sources, citations), `tests/test_streaming.py` (events, both
endpoints build the same request), `tests/test_conversations.py`, `tests/test_relevance.py`,
`tests/test_mmr.py`, `tests/test_hybrid.py`, `tests/test_bm25.py`, `tests/test_reranker.py`,
`tests/test_metadata_filtering.py`, `tests/test_language_consistency.py`, `tests/test_query_e2e.py`
(the real app, hybrid on), `tests/test_retrieval_quality.py` (offline recall/MRR floors, dense path only).

## How it works

**Construction.** The lifespan creates `app.state.bm25_index = BM25Index()` (empty, per process) and
`app.state.reranker = get_reranker(settings)`, then `RAGChain(vectorstore, ...)` with the Settings
values model, api_key, base_url, timeout, max_retries (rag-llm-providers), top_k, temperature,
max_answer_tokens, relevance_threshold, max_history_turns, mmr_lambda, retrieval_candidates,
metadata_filtering_enabled and the four hybrid values, plus `embeddings_manager`, `bm25_index`,
`reranker` and `tracer`. `get_reranker` returns a `NoOpReranker` when RERANKER_ENABLED=false and
rerankers define no `__bool__`, so in the app a reranker is always active: candidate expansion is always
on and a `rerank` span is recorded whenever chunks were retrieved. A RAGChain built directly (tests) has
no BM25 index (dense only), no reranker and max_history_turns=0, and falls back to os.environ for model,
top_k, temperature, retrieval_candidates and the metadata-filtering and hybrid flags (on when unset).

**`RAGChain._prepare`**, shared by `ask` and `ask_stream`:
1. A falsy user_id raises ValueError.
2. Filters: `extract_query_metadata(question)` when METADATA_FILTERING_ENABLED, then the request's
   `metadata_filter` merged over it (even when the flag is off). `build_chroma_filter` returns
   `{"user_id": u}` or `{"$and": [{"user_id": u}, {key: value}, ...]}` and ignores user_id and None values.
3. Condensing: `_condense` makes one extra chat call (temperature 0, max_tokens 200, the caller's client
   and model when given) only when history is non-empty and MAX_HISTORY_TURNS > 0. Any exception or an
   empty reply falls back to the question. A changed rewrite records a `query_condense` span and, with
   METADATA_FILTERING_ENABLED and no filter found yet, re-extracts metadata from the rewrite. Retrieval
   and reranking use the rewrite; the answer prompt keeps the original question.
4. Size: fetch_k = max(TOP_K_RESULTS, RETRIEVAL_CANDIDATES) when a reranker object is active or hybrid
   is on with a BM25 index (always in the app: 20), else TOP_K_RESULTS.
5. Dense (`_retrieve_dense`). Scored path: `similarity_search_with_score(k=fetch_k, filter)`, then
   `distance_to_similarity` for the collection's `hnsw:space` (l2: 1 - d/2, cosine: 1 - d, ip: -d,
   unknown: filtering disabled with a WARNING), an INFO `retrieval space=...` line (Debug recipe), then
   keep similarity >= RELEVANCE_THRESHOLD when it is above 0. MMR path (MMR_LAMBDA < 1; the app wires the
   embeddings manager it needs): `_retrieve_diverse` reads `filter_dict["user_id"]`, fetches 4 x k
   candidates with vectors via `search_candidates` (owner-only where clause), thresholds, `select_mmr` keeps k.
6. Hybrid (`HybridRetriever.retrieve`, when HYBRID_SEARCH_ENABLED and a BM25 index exist): owner =
   top-level user_id; lazy BM25 sync from Chroma for an owner with no BM25 entries; BM25 top
   max(4 x k, 25) = 80; RRF score = sum of weight / (HYBRID_RRF_K + rank) over the dense and BM25 lists;
   returns at most 80. If either list is empty the other is returned unfused.
7. Fallback: nothing back and a metadata filter in play -> retrieve once more with `{"user_id": user_id}`
   and log INFO `Metadata filter ... yielded 0 results; falling back to tenant-wide search`. The
   `retrieval` span reports the pre-fallback filter.
8. Rerank or cut: `active_reranker.rerank(search_query, docs, top_n=TOP_K_RESULTS)` in a `rerank` span
   (a per-request reranker from the X-Reranker-* headers beats the server's), else `docs[:TOP_K_RESULTS]`.
   The reranker sees the whole fused pool, up to 80 chunks.
9. Nothing left -> `(None, [])`; the answer model is never called.
10. Prompt: system = `SYSTEM_PROMPT` + `Language rule:` + `rule_for(language)`; user = optional
    `Conversation so far:` (last MAX_HISTORY_TURNS turns) + `Context:` with one `Source: <filename>`
    block per chunk + `Question:` (the original) + `Answer:`. Request = model (caller's or server's),
    messages [system, user], temperature and max_tokens (MAX_ANSWER_TOKENS, always set in the app).
11. Sources (`_collect_sources`): one entry per distinct filename in ranked order, ids from 1, preview =
    first 200 characters of that file's highest-ranked chunk, `rerank_score` and `metadata`
    (company/year/quarter/doc_type) when present. Later chunks of a listed file are in the prompt only.

**`ask`** calls `(client or self.client).chat.completions.create(**request)`, logs INFO
`chat completion user_id=... model=... prompt_tokens=... completion_tokens=... total_tokens=... finish_reason=...`
(WARNING on `length`), strips the text, removes `[digits]` with a regex (the spaces around stay), and
returns `{answer, sources, model}`; nothing retrieved -> `No relevant information found.`, `[]`, `None`.
**`ask_stream`** opens the model stream (`stream=True`) before its first yield, so the handler's
priming `next()` turns condensing, retrieval, reranking and stream-opening failures into ordinary status
codes. It then yields `sources`, `token` events filtered by `CitationStripper` (holds a partial `[12`
until the next chunk; `flush` releases an unterminated one) and `done`; a later failure becomes a final
`error` event in `app/main.py`. It logs only `streamed completion ... finish_reason=...`, no tokens.
Only `[digits]` markers go: `[1, 2]` survives and a bracketed year such as `[2024]` is deleted.

**Tracing.** With LANGFUSE_ENABLED and both keys, each request is a `rag-query` trace keyed by the
request id, with spans `query_condense`, `retrieval` (query, k, filter; count, sources), `rerank`
(candidates, top_n; scores) and a `generation` holding the full messages (usage on /query only).
Otherwise every handle is a no-op. Prompts with tenant chunks leave the host when tracing is on.

## Knobs

| Setting | Default | Effect | Why it ships so |
| --- | --- | --- | --- |
| TOP_K_RESULTS | `5` | chunks in the prompt after fusion or reranking; reranker top_n | each chunk is prompt tokens |
| RETRIEVAL_CANDIDATES | `20` | dense k (always, see step 4); BM25 list and fused pool max(4x, 25); MMR fetches 4x | a wider pool than the prompt for fusion and reranking |
| RELEVANCE_THRESHOLD | `0.0` | inclusive cosine floor on dense candidates; 0 = off | corpus-specific; a guessed value silently drops relevant context, so measure first |
| MMR_LAMBDA | `1.0` | below 1 diversifies dense candidates with `select_mmr` | measured (before hybrid existed): recall rose only after precision fell from 0.60 to 0.35; also D7 |
| MAX_HISTORY_TURNS | `6` | turns used for condensing and the prompt; 0 = no condense call | each turn costs tokens, condensing a call; clients send at most 6 |
| METADATA_FILTERING_ENABLED | `True` | auto filters from the question; explicit `metadata_filter` applies regardless | built for financial filings; consider false for general corpora (D10) |
| HYBRID_SEARCH_ENABLED | `True` | BM25 plus RRF after dense retrieval | catches exact names, numbers and rare terms dense search misses |
| HYBRID_DENSE_WEIGHT, HYBRID_BM25_WEIGHT | `1.0`, `1.0` | RRF weight per list (>= 0); 0 demotes but never removes | plain RRF |
| HYBRID_RRF_K | `60` | k in weight / (k + rank) | the standard RRF constant |
| RERANKER_ENABLED, RERANKER_PROVIDER, RERANKER_MODEL, RERANKER_API_KEY | `False`, `none`, `""`, `""` | server-wide reranker ([references/reranking.md](references/reranking.md)) | flashrank is not installed; Cohere needs a key and sends chunks off-host |
| TEMPERATURE | `0.0` | answer temperature (condensing is always 0) | deterministic answers |
| MAX_ANSWER_TOKENS | `1000` | max_tokens on the answer call | caps spend; long answers broke Telegram's 4096-character limit |

Hard-coded: `CONDENSE_MAX_TOKENS` 200, `MMR_FETCH_MULTIPLIER` 4, hybrid multiplier 4 (floor 25), BM25 k1 1.5 and b 0.75, Cohere timeout 10 s. MODEL_NAME: rag-llm-providers.

## The BM25 index

Details, the RRF arithmetic and an offline harness: [references/hybrid-and-bm25.md](references/hybrid-and-bm25.md).
- One `BM25Index` per backend process (`app.state.bm25_index`), in memory only: not persisted, not in
  backups, empty after every restart and every dev `--reload`.
- Filled lazily per owner: `HybridRetriever.retrieve` runs `sync_from_chroma` when the owner has no BM25
  entries and the filter has a top-level user_id.
- Kept in step with Chroma in `app/main.py` on upload (`_store_and_index`, even with hybrid off),
  revision retirement, `delete_document` and `/clear` (`_clear_namespace`). Not by the idle sweep:
  `app/sweep.py` `execute` calls `storage.wipe_namespace`, which has no BM25 call (D1).
- Tokens: lower-cased `\w+` runs; no stemming, no stop words, ё and е differ, hyphens split (`10-K` -> `10`, `k`).
- IDF and average length are computed over every tenant's chunks (O2).
- Filters: equality and `{"$in": [...]}` only, user_id skipped; an `$and` key matches nothing (D3).

## Invariants

- Retrieval, prompt and request changes go in `_prepare` (or `_retrieve`), never in `ask` or `ask_stream`
  alone: `test_both_paths_build_the_same_request` pins that both endpoints send the same request.
- Tenancy helpers: `EmbeddingsManager._owned(owner, ...)` is only for EmbeddingsManager's own
  get/query/delete calls. `RAGChain` and `HybridRetriever` take a filter whose top level is
  `{"user_id": owner}` or the `$and` built by `build_chroma_filter(user_id, metadata)`
  (`app/rag/metadata_extractor.py`); an `_owned`-shaped filter there crashes (TypeError, D6).
- Give BM25 an owner every time: `BM25Index.search` without one, and `HybridRetriever.retrieve` without a
  top-level user_id, search every tenant (D5).
- Chroma and BM25 change together on every mutation path, under `state.upload_locks.for_owner(user_id)`
  (`delete_document` does not take the lock today).
- Rerankers and optional stages never raise: they log a WARNING and pass the input through, because an
  exception inside `_prepare` is a 503 "Query failed" for the whole question.
- Distances become cosines only through `distance_to_similarity` with the collection's real space; an
  unknown space disables the threshold instead of comparing on the wrong scale. The threshold runs
  before MMR, so diversity never buys an irrelevant chunk.
- The caller's client and model are passed per call and never stored on the chain (rag-llm-providers).
- Optional SDKs are imported lazily or behind try/except: `app/rag/reranker.py` is imported at startup.
- `app/rag/languages.py` is the only language table and stays free of heavy imports; clients derive from it.

## Recipes

### Add an answer language
1. `app/rag/languages.py`: add `"<Display name>": "<rule, written in that language>"` inside the
   `LANG_RULES` literal (`SUPPORTED_LANGUAGES` is computed at import, so a later assignment is missed).
   The display name is the exact wire value of `QueryRequest.language`; never add `Auto`.
2. `frontend/streamlit_app.py`: add the name and a flag to the `LANG_FLAGS` dict literal. Required:
   `tests/test_language_consistency.py` reads it with `ast.literal_eval` and wants exactly the supported set.
3. Nothing in `clients/telegram_bot.py` or `clients/backend.py`: both derive from `SUPPORTED_LANGUAGES`.
   The bot treats a message equal to a language name as a selection, so avoid names that read as questions.
4. Update by hand (no test checks them): the language list in README.md Key Features and in
   DOCUMENTATION.md (Key Features and "Supported languages").
5. Renaming or removing: older clients' stored choice silently falls back to Auto; `tests/test_query_e2e.py`
   expects `русском`, `қазақ` and `简体中文` in the system prompt.
6. Run `python -m pytest tests/test_language_consistency.py tests/test_chain.py tests/test_query_e2e.py tests/test_streaming.py -q`.

### Change the prompt or the context layout
1. Edit `SYSTEM_PROMPT`, `_build_context`, `_history_block` or the prompt f-strings in `_prepare`.
2. Keep messages as [system, user] (tests index them), one `Source: <filename>` header per chunk, the
   original question and the history in the user message, the language rule in the system message.
3. Citation changes touch both `_strip_citations` (ask) and `CitationStripper` (stream);
   `tests/test_chain.py` pins the exact double-space result.
4. The condensing prompt is in `_condense`; `tests/test_conversations.py` pins its content and limits.
5. Run `python -m pytest tests/test_chain.py tests/test_streaming.py tests/test_conversations.py tests/test_query_e2e.py tests/test_relevance.py tests/test_resilience.py -q`.

### Add a pipeline stage
1. Put it in `_prepare` (or `_retrieve` for retrieval-time work) so both endpoints get it.
2. Scope it: take the owner from the filter's top-level user_id or the `{"user_id": u}` clause inside
   `$and`, and raise when there is none.
3. If it keeps per-owner state, update it with Chroma in `_store_and_index`, `_retire_older_revisions`,
   `delete_document`, and inside `app/storage.py` `wipe_namespace` (reached by both `/clear` and the
   sweep, unlike BM25 today), under the owner lock; rebuild it after a restart before an owner's first
   upload, and decide how backup and restore treat it.
4. Configure it: a bounded Field in `app/config.py`, `.env.template` and a DOCUMENTATION.md row
   (rag-config-and-docs), passed in the lifespan's `RAGChain(...)` and taken by `RAGChain.__init__`
   without a new os.getenv fallback, plus a test like `tests/test_mmr.py` `test_the_setting_reaches_the_chain`.
   A per-request option also goes through `ask`, `ask_stream`, `_prepare` and both handlers
   (rag-api-endpoints); test doubles of `ask`/`ask_stream` take `**kwargs`.
5. Fail soft (log a WARNING, pass the input through); copy a Document before changing its metadata -
   with an empty dense list, hybrid returns the BM25 index's own objects.
6. Trace it: `span = trace_handle.span("<stage>", input={...})`, then `span.end(output={...})`; no keys
   or headers in payloads.
7. Measure offline: `tests/test_retrieval_quality.py` builds the chain without BM25 or a reranker, so add
   a hybrid variant for hybrid-side work (harness in references/hybrid-and-bm25.md); assert floors.
8. Measure live before and after (rag-evaluation): `python -m evaluation.run_eval --url http://127.0.0.1:8000 --api-key <key>`.
9. Document the behaviour in DOCUMENTATION.md and update this skill's Map and How it works.

### Tune knobs safely
1. Change `.env` and restart the backend: Settings are read once, in the lifespan.
2. Remember the real sizes: 20 dense, 80 BM25, up to 80 fused and reranked, TOP_K_RESULTS in the prompt.
3. Measure before and after with `python -m evaluation.run_eval --url http://127.0.0.1:8000 --api-key <key>`
   and the same `--top-k` both times (default 3; the golden corpus has five files, so 5 or more makes
   recall and hit rate trivially 1.0), and read the backend's `retrieval space=...` lines. The full
   before/after procedure is the rag-evaluation skill's.
4. RELEVANCE_THRESHOLD gates only while HYBRID_SEARCH_ENABLED=false (D4); pick a value with the
   rag-evaluation skill's "Calibrate RELEVANCE_THRESHOLD" recipe. Keep MMR_LAMBDA at 1.0 (D7).
   Weight 0 does not remove a list; HYBRID_SEARCH_ENABLED=false removes BM25.
5. A new code default also changes `.env.template`, the DOCUMENTATION.md row and pinning tests such as
   `test_the_threshold_defaults_to_disabled` and `test_mmr_ships_disabled` (rag-config-and-docs).

### Debug: the relevant chunk was not retrieved
1. Right tenant? The web UI uses a `web-<hex>` id per browser session, the bot the Telegram user id;
   `GET /documents?user_id=<id>` lists what that id holds.
2. Filtered? Run `extract_query_metadata("<question>")` in a REPL. A result means dense search saw only
   chunks tagged exactly so (tags show as `metadata` in sources), BM25 sat out (D3), and the fallback ran
   only if nothing matched. Retry with METADATA_FILTERING_ENABLED=false or rephrase.
3. Read the backend log lines for that request id (similarities are cosines for unit-normed embeddings):
   `app.rag.chain - INFO - [<request id>] retrieval space=l2 candidates=20 similarity_best=... similarity_worst=... threshold=0.0`,
   then `dropped X/Y chunk(s) below RELEVANCE_THRESHOLD=...` and the fallback line when they apply.
4. Exact-term misses: BM25 may lack the owner's older documents (D2; restart, then ask a question
   without a metadata filter before uploading), or tokens differ (`tokenize_multilingual` has no stemming).
5. Only TOP_K_RESULTS chunks reach the prompt and sources list one entry per file: check the prompt's
   `Source:` blocks (Langfuse generation input, or `FakeChatClient.calls` offline).
6. Reproduce offline with the harness in references/hybrid-and-bm25.md; pin the fix in the matching
   test module (retrieval_quality, hybrid, bm25 or metadata_filtering).

### Debug: "No relevant information found."
Nothing survived retrieval, the fallback and the cut, and no model was called (`model` is null on
/query). Causes, most common first: a tenant with no documents (wrong user_id); RELEVANCE_THRESHOLD
dropped every dense candidate (`dropped N/N` in the log) while hybrid is off or BM25 shared no word with
the question; a collection whose space mis-converts distances (D11). With hybrid on, an off-topic
question rarely gets this answer (D4).

### Debug: wrong-language answers
1. Check the `language` sent: only an exact `LANG_RULES` key works (`Русский`, not `ru` or `russian`);
   anything else silently gets `AUTO_RULE`. The bot keeps the choice in memory, so its restart means Auto.
2. Auto mirrors the original question, so a question that is only a name or a number gives the model
   little to mirror; context in another language has to be translated, which AUTO_RULE asks for.
3. Read the system prompt actually sent (Langfuse generation input, or `FakeChatClient.calls` offline);
   it ends with the rule. A caller's own model (`X-Model`) may follow it less reliably than the server's.
4. To reword a rule, edit `LANG_RULES` and keep the e2e substrings or update the tests.

More: adding a reranker provider, aliases and fallbacks in [references/reranking.md](references/reranking.md).

## Pitfalls and known issues

All verified (D and O labels are this skill's own; KNOWN_ISSUES.md uses titles); details and workarounds in KNOWN_ISSUES.md, section "Retrieval and search", unless another section is named.
- D1: after an applied idle sweep, the swept owner's chunks stay in BM25 and are still retrieved and cited for that user_id until restart; only `/clear` purges BM25 - section "Operations: backup, restore and sweep".
- D2: after a restart (or a dev `--reload`), an owner who uploads before asking a question without a metadata filter never gets the lazy sync; their older documents stay out of BM25 until the next restart.
- D3: any metadata filter (an auto-detected year, company, quarter or report type, or a `metadata_filter`) makes retrieval dense-only; nothing is logged above DEBUG.
- D4: with hybrid on, RELEVANCE_THRESHOLD filters dense candidates only; BM25 hits sharing any word still reach the model, so "No relevant information found." is not guaranteed.
- D5 (latent): the sparse path fails open without an owner; today only the `$and` mismatch behind D3 stops a cross-tenant read.
- D6: an `_owned`-shaped filter passed to the chain or hybrid path raises TypeError, a 503.
- D7: with MMR_LAMBDA < 1, any question that yields a metadata filter ("block", "meta", a 1900-2039 number, Q1-Q4, 10-K) or carries `metadata_filter` gets 503 (KeyError in `_retrieve_diverse`); MMR's candidate search ignores metadata filters anyway.
- D8: "annual report" becomes doc_type `ANNUAL REPORT`, which no document carries, so the search falls back to the whole tenant and loses the company and year filters too.
- D9: `metadata_filter` is unvalidated: a list value gives 503, values are not normalised (a ticker such as WMT matches nothing), and a filter matching nothing silently falls back to tenant-wide search.
- D10: on general corpora, any number from 1900 to 2039 or an alias word (block, square, apple) narrows retrieval to documents with that tag, right or wrong; the fallback runs only on zero hits.
- D11, O8: on a collection created outside the app with `hnsw:space` "ip", similarity is -distance (identical vectors score 0), so any threshold above 0 drops everything; any `hnsw:space` key also defeats the embedding-model guard.
- O2: BM25 statistics are global, so one tenant's uploads can reorder another tenant's keyword ranking (no content crosses).
- O3: uploads are added to BM25 even with HYBRID_SEARCH_ENABLED=false.
- O5: METADATA_FILTERING_ENABLED=false stops only auto-extraction; an explicit `metadata_filter` still applies.
- O7: two years more than five apart filter on just those two ("between 1990 and 2010"), and six or more distinct years give no year filter (extractor rules: rag-document-ingestion).
- D14: X-Reranker-Provider works with RERANKER_ENABLED=false, ignores ALLOWED_MODEL_PROVIDERS, and cohere without X-Reranker-Key spends RERANKER_API_KEY - section "Models, keys and reranking".
- D15: flashrank is in neither the requirements nor the image; choosing it logs a warning and keeps the input order - section "Models, keys and reranking".
- D16: /query/stream, the web UI's only path, logs no token usage and sends none to Langfuse - section "Tracing and observability".

Other traps: the `api` fixture's fake vectors are random and not unit-normed, so dense ranking there is
noise (use the bag-of-words embedder); a RERANKER_PROVIDER typo silently means no reranking; the hybrid
`dense=N, bm25=M` line is DEBUG (`app.rag.hybrid`), hidden at INFO; `_condense` swallows timeouts and
auth errors, so a follow-up to a failing provider waits out that failure first; MAX_HISTORY_TURNS above
6 changes nothing for the bundled clients; page numbers never reach the prompt or the sources.

## Verify

```bash
python -m pytest tests/test_chain.py tests/test_streaming.py tests/test_conversations.py tests/test_query_e2e.py tests/test_relevance.py tests/test_mmr.py tests/test_hybrid.py tests/test_bm25.py tests/test_reranker.py tests/test_metadata_filtering.py tests/test_language_consistency.py tests/test_retrieval_quality.py tests/test_resilience.py -q
python -m pytest tests/test_docs_consistency.py tests/test_config.py -q   # after touching settings or docs
ruff check app frontend clients evaluation scripts tests
```

These run offline in seconds; running them from WSL or a Windows venv is covered in rag-testing.

## Related skills

rag-api-endpoints (the two query handlers), rag-llm-providers (models, BYOK, reranker headers),
rag-document-ingestion (chunks and tags), rag-evaluation (measuring a change), rag-testing,
rag-config-and-docs, rag-operations (sweep, Langfuse stack), rag-change-review, rag-architecture.
