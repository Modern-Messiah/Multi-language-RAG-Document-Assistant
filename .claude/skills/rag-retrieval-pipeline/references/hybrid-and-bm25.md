# Hybrid search and the BM25 index

Read this when you change `app/rag/hybrid.py` or `app/rag/bm25.py`, tune the HYBRID_* settings,
add a write path that changes an owner's chunks, or debug why a keyword match did not come back.
D and O labels refer to the known-issues list in SKILL.md.

Contents:
1. Where hybrid runs
2. Sizes and RRF arithmetic
3. Index lifecycle: who keeps BM25 in step with Chroma
4. Tokens, scores, filters and tenancy
5. Debugging keyword retrieval
6. Offline reproduction
7. Changing hybrid or adding another signal

## 1. Where hybrid runs

`RAGChain._retrieve` runs dense retrieval first, then, when `hybrid_search_enabled` is true and a
`bm25_index` was passed (always in the app), builds a `HybridRetriever` per call and calls
`retrieve(query, filter_dict, k=fetch_k, dense_candidates=dense_docs)`. Inside:
1. `owner = filter_dict.get("user_id")` - top level only. Under an `$and` filter the owner is None.
2. `fetch = max(k * 4, 25)` (`candidate_multiplier` is a constructor argument, not a setting).
3. Lazy sync: with an owner, an embeddings manager holding a collection, and `count_owner(owner) == 0`,
   `bm25_index.sync_from_chroma(collection, owner)` loads that owner's chunks with
   `collection.get(where={"user_id": owner})`. Errors are logged as a WARNING and return 0.
4. `bm25_index.search(query, owner=owner, k=fetch, filter_dict=filter_dict)`.
5. Short circuits: both lists empty -> `[]`; dense empty -> the BM25 list as is (the index's own
   Document objects, so copy before mutating); BM25 empty -> the dense list; otherwise RRF.
6. `reciprocal_rank_fusion([(dense, HYBRID_DENSE_WEIGHT), (bm25, HYBRID_BM25_WEIGHT)], k_rrf=HYBRID_RRF_K)`,
   cut to `fetch`.

HybridRetriever's own dense search (used when `dense_candidates` is None) and its "dense failed,
continue with BM25" branch never run from the chain: a dense-store error propagates and the query is a
503. The `Hybrid retrieval for query '...': dense=N, bm25=M (fetch_k=80)` line is DEBUG; the app logs
at INFO and has no log-level setting, so raise `logging.getLogger("app.rag.hybrid")` to DEBUG in a test
or REPL to see it.

## 2. Sizes and RRF arithmetic

With the shipped settings: dense k = max(TOP_K_RESULTS, RETRIEVAL_CANDIDATES) = 20 (a reranker object
always exists, see SKILL.md step 4); BM25 k = max(4 x 20, 25) = 80; the fused list is cut to 80; the
reranker (or the plain cut) then keeps TOP_K_RESULTS = 5. With MMR on, the dense list is 20 chosen from
80 candidates.

RRF: score(d) = sum over lists of weight / (k_rrf + rank), ranks from 1. Identity is
`_document_key` = `source:file_hash:` plus the first 100 characters of the stripped text; the first
occurrence is copied and gets `metadata["rrf_score"]` (rounded to 6 places), which never reaches the
sources. Consequences worth knowing:
- At the defaults, every chunk found by both retrievers outranks every chunk found by only one: the
  weakest two-list score is 1/80 + 1/140 (about 0.0196), the best one-list score 1/61 (about 0.0164).
  A dense-only first place overtakes the weakest two-list chunk only when the dense/BM25 weight ratio
  exceeds about 1.83. A larger RETRIEVAL_CANDIDATES lengthens both lists and weakens this effect; a
  lower HYBRID_RRF_K favours top ranks.
- A weight of 0 keeps the list's documents in the fused output with `rrf_score` 0.0, and the short
  circuits return a lone list whatever its weight. To drop BM25, set HYBRID_SEARCH_ENABLED=false.
- Chunks that share source, file_hash and their first 100 characters merge into one. The table-aware
  splitter repeats a table's header on every slice, so slices whose header fills those 100 characters
  collapse during fusion into whichever slice fusion meets first (dense list first), and the other
  slices' rows cannot reach the prompt.
- A chunk listed twice in one list is counted twice; `BM25Index.add_documents` never deduplicates.

## 3. Index lifecycle: who keeps BM25 in step with Chroma

`app.state.bm25_index = BM25Index()` is created empty in the lifespan, lives only in that process's
memory and is not persisted or backed up. A second worker or replica would hold its own diverging copy;
the Compose files run one uvicorn process, and the dev override's `--reload` restarts it (empty) on
every change under `app/`.

| Event | Chroma | BM25 | Owner lock |
| --- | --- | --- | --- |
| upload, `_store_and_index` | `add_documents` | `add_documents(chunks, owner=user_id)`, even when hybrid is off | yes |
| new revision, `_retire_older_revisions` | `delete_by_file_hash` | `delete_by_file_hash(old_hash, user_id)` | yes (inside the upload) |
| `DELETE /documents/{file_hash}`, `delete_document` | `delete_by_file_hash` | `delete_by_file_hash(file_hash, user_id)` | no |
| `/clear`, `_clear_namespace` | `storage.wipe_namespace` | `clear_owner(user_id)` | yes |
| idle sweep, `app/sweep.py` `execute` | `storage.wipe_namespace` | nothing (D1) | yes |
| restart | unchanged | empty, then lazy per-owner sync | - |
| `scripts/restore.py` with `--live` | files replaced | stale until restart | - |

The lazy sync runs only while the owner has zero BM25 entries and the filter has a top-level user_id
(no metadata filter, or the tenant-wide fallback). An owner whose first action after a restart is an
upload therefore keeps a partial index (only the new chunks) until the next restart (D2), and a filtered
first question that finds dense hits syncs nothing (D3). Duplicate
uploads (same bytes) touch neither store. `sync_from_chroma` holds the global BM25 lock while it reads
Chroma, so a slow read stalls every tenant's keyword search.

## 4. Tokens, scores, filters and tenancy

- `tokenize_multilingual`: lower-cased `(?u)\b\w+\b` matches. No stemming or lemmatisation (`гитару`
  and `гитара` differ), no stop words, `ё` and `е` differ, hyphens and punctuation split (`10-K` ->
  `10`, `k`; `1,234.56` -> `1`, `234`, `56`), underscores stay inside a token.
- Okapi BM25 with k1 1.5 and b 0.75. IDF = ln((N - df + 0.5) / (df + 0.5) + 1), always positive, so the
  epsilon floor for negative IDF never applies and a shared stop word still scores. N, df and the average
  length cover every tenant's chunks and are recomputed after each add or delete: one tenant's uploads
  change another tenant's scores and can reorder its ranking (O2). Tests asserting exact scores need a
  fresh `BM25Index()`.
- `search(query, owner, k, filter_dict)`: with an owner, only that owner's chunks are candidates (none ->
  `[]`); with a falsy owner, every tenant's chunks are (D5). The filter loop skips `user_id`, supports
  equality and `{"$in": [...]}`, and compares any other key, `$and` included, as a metadata field, so an
  `$and` filter rejects everything (D3). Only scores above 0 are returned.
- A dict-valued owner, such as the `{"$eq": owner}` inside an `EmbeddingsManager._owned` filter, raises
  TypeError (D6). Always pass the plain user_id string.
- Whoever fixes D3 must take the owner from the `{"user_id": u}` clause inside `$and` and fail closed when
  it is missing; flattening `$and` while leaving owner None would search every tenant (D5).

## 5. Debugging keyword retrieval

1. Is the owner's corpus in BM25 at all? There is no endpoint for this. Reproduce in-process (the
   `api` fixture, or `create_app(make_settings(tmp_path))` under a `TestClient`) and compare
   `app.state.bm25_index.count_owner(u)` with
   `len(app.state.embeddings.collection.get(where={"user_id": u}, include=[])["ids"])`. Lower means the
   upload-first gap (D2), higher means sweep leftovers (D1). On a live deployment the only remedies are
   `/clear` for that user (deletes their documents) or a restart, followed for D2 by a question without a
   metadata filter before any upload.
2. To resync one owner in-process: `clear_owner(u)`, then `sync_from_chroma(collection, u)`; the sync is
   a no-op while the owner has any entries.
3. Did the question carry a metadata filter? Then BM25 contributed nothing (D3); check
   `extract_query_metadata(question)`.
4. Do the tokens match? Compare `tokenize_multilingual(question)` with `tokenize_multilingual(chunk)`.
5. Did fusion merge two chunks? Compare their `_document_key` values.

## 6. Offline reproduction

The `api` fixture's fake embeddings are random, so dense ranking there means nothing. For ranking
questions use the bag-of-words embedder and the `indexed` fixture from `tests/test_retrieval_quality.py`
(golden corpus in a real Chroma under `tmp_path`, owner `u1`). Its own chain has no BM25 index; a hybrid
variant looks like this (checked to run, the lazy sync fills BM25 from the collection on first use):

```python
from app.rag.bm25 import BM25Index
from app.rag.chain import RAGChain

def test_<hybrid_case>(indexed):
    manager, store = indexed
    chain = RAGChain(store, client=object(), top_k=TOP_K, embeddings_manager=manager,
                     bm25_index=BM25Index(), hybrid_search_enabled=True,
                     metadata_filtering_enabled=False)
    docs = chain._retrieve("<question>", {"user_id": "u1"},
                           k=max(chain.top_k, chain.retrieval_candidates))[:TOP_K]
    sources = [doc.metadata["source"] for doc in docs]
```

`_retrieve` stops before the cut to TOP_K_RESULTS that `_prepare` makes, hence the slice. Assert floors
(`MIN_RECALL`, `MIN_MRR`, `MIN_HIT_RATE`) or membership, not exact scores. `tests/test_hybrid.py` and
`tests/test_bm25.py` hold unit-level patterns: `reciprocal_rank_fusion` on hand-made lists, tenant
isolation, delete and clear, the metadata filter.

## 7. Changing hybrid or adding another signal

1. Return Documents whose `source`, `file_hash` and page text equal the Chroma chunk's, so
   `_document_key` merges them with the dense and BM25 hits.
2. Scope by tenant first: read the owner from either filter shape and raise when there is none.
3. Implement the filter semantics the extractor produces (`$and` of equality clauses, `{"$in": [...]}`
   for year spans) or document that the signal ignores them.
4. Add it as another `(docs, weight)` pair in `ranked_lists`; `reciprocal_rank_fusion` takes any number
   of lists, but the short circuits in `HybridRetriever.retrieve` assume two.
5. Keep any state in step on every row of the table in section 3, and put its per-owner clear inside
   `app/storage.py` `wipe_namespace` so `/clear` and the sweep both reach it. Rebuild after a restart
   before the owner's first upload, not only when their count is zero.
6. Decide whether RELEVANCE_THRESHOLD applies to it; today it gates dense candidates only (D4).
7. New settings go through `app/config.py`, `.env.template`, the DOCUMENTATION.md Configuration table,
   the lifespan, `RAGChain.__init__` and `RAGChain._retrieve` (rag-config-and-docs).
8. A new dependency means `requirements.txt` plus both lock files (rag-development-guide), and the
   signal must work offline because the test suite blocks sockets.
9. Tests: tenant isolation (pattern: `tests/test_bm25.py` `test_bm25_tenant_isolation`), multi-list
   fusion in `tests/test_hybrid.py`, a sweep test in `tests/test_sweep.py` asserting the owner is
   forgotten, and an end-to-end question that carries a year or company through the `api` fixture.
