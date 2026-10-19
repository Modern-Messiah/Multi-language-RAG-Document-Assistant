---
name: rag-evaluation
description: Measures retrieval quality - offline metrics in the test suite, the live harness python -m evaluation.run_eval, the golden set, turning ratings into cases with evaluation.from_feedback, and the Langfuse benchmark flow (scripts.download_benchmarks, scripts.upload_to_langfuse, evaluation.langfuse_eval, evaluation.eval_sberquad_failures). Use before and after any retrieval or chunking change, when adding golden cases, or when running SberQuAD, FinanceBench, TyDi QA or RAGTruth benchmarks.
---

# Retrieval evaluation

Retrieval knobs in this repo change on measured grounds: RELEVANCE_THRESHOLD and MMR_LAMBDA ship
off until a measurement justifies more. This skill produces that measurement: three tiers that
answer different questions, the golden set they share, and a Langfuse benchmark layer. Every tool
scores which files came back for a question; nothing judges the answer (there is no LLM judge).

Ground rules:
- Run every module from the repository root as `python -m <package>.<module>`;
  `python evaluation/run_eval.py` fails with `No module named 'evaluation'`.
- None of it is production code: `.dockerignore` keeps `evaluation/` out of the image (the benchmark
  scripts in `scripts/` ship but need network and Langfuse keys), so run everything from a checkout
  on the host. CI only lints these packages and runs the offline tests.
- Live runs cost money: uploads embed documents; every question embeds the query and, when anything
  is retrieved, runs a chat completion. Ask the user first, against a backend they chose.

## Use this skill for / not for

For: before/after measurements of retrieval, chunking, reranker or filter changes; picking
RELEVANCE_THRESHOLD from data; golden and feedback cases; Langfuse experiments and benchmarks.

Not for: changing the pipeline itself (rag-retrieval-pipeline); loaders, chunking and the metadata
extractor (rag-document-ingestion); fixtures, fakes and which test covers what (rag-testing); the
local Langfuse stack, backend tracing, the feedback log on a server and request-id tracing
(rag-operations); BYOK and reranker headers (rag-llm-providers); changing a default and the
documents that list it (rag-config-and-docs); reviewing a change that claims a quality gain
(rag-change-review).

## Map

| Path | Role | Key symbols |
| --- | --- | --- |
| `evaluation/metrics.py` | metric arithmetic, stdlib only | `recall_at_k`, `precision_at_k`, `reciprocal_rank`, `hit_at_k`, `aggregate` |
| `evaluation/golden.py` | small labelled corpus, answerable and unanswerable questions | `CORPUS`, `GOLDEN_CASES`, `UNANSWERABLE` |
| `evaluation/run_eval.py` | live golden run, stdlib only | `Backend`, `run`, `main` |
| `evaluation/from_feedback.py` | ratings report and case stubs, stdlib only; pinned by `tests/test_feedback.py` | `main`; `app/feedback.py` `read_records`, `summarise`, `unanswered_questions`, `golden_case_stub` |
| `evaluation/langfuse_eval.py` | runs a Langfuse dataset against a backend, records a dataset run | `sync_dataset_items`, `get_corpus_and_expectations`, `run_experiment` |
| `evaluation/eval_sberquad_failures.py` | re-asks failed SberQuAD items on an indexed tenant | `extract_failed_indices`, `run_hard_cases_benchmark` |
| `scripts/download_benchmarks.py`, `scripts/upload_to_langfuse.py` | six public datasets into `data/benchmarks/`, then into `eval-*` Langfuse datasets | `main` |
| `docker-compose.langfuse.yml` | local Langfuse v2 plus Postgres, UI on port 3000 | |
| `tests/test_retrieval_quality.py`, `tests/test_mmr.py`, `tests/test_metrics.py` | the offline tier | `BagOfWordsEmbedding`, fixtures `indexed`, `crowded` |

## The three tiers

### 1. Offline, in CI

```bash
python -m pytest tests/test_metrics.py tests/test_retrieval_quality.py tests/test_mmr.py -q
```

- `tests/test_retrieval_quality.py` indexes each `CORPUS` document whole into a temporary ChromaDB
  with a bag-of-words embedder (fixture `indexed`) and calls `RAGChain(store, client=object(), ...)._retrieve(question, {"user_id": "u1"})`.
  At k=3 it asserts recall@3, MRR and hit_rate@3 >= 0.85; each case's document in the top 3, at most
  one not first; `battery.txt` first for the storage question; threshold 0.25 raises precision@3
  and recall never rises with the threshold; `top_k` bounds the result; tenant isolation; the
  shipped threshold 0.0; corpus integrity (expected names exist, a Cyrillic document, UNANSWERABLE
  disjoint from the golden questions).
- `tests/test_mmr.py` checks on a crowded bag-of-words corpus (`crowded`) that MMR brings in other
  documents, runs after the threshold, keeps tenants apart, fetches 4x candidates and ships at 1.0.
  `tests/test_metrics.py` pins the arithmetic below.

It can tell you the dense step still ranks the right document, threshold, MMR and tenant scoping
behave, and the metrics compute what they claim. It says nothing about real embeddings or a real
threshold value (bag of words); hybrid BM25 and RRF, reranking, metadata filters, condensation or
source dedup (`_prepare` never runs); chunking (documents go in whole); or unanswerable questions.

### 2. Live golden run: python -m evaluation.run_eval

```bash
python -m evaluation.run_eval --url http://127.0.0.1:8000 --api-key <BACKEND_API_KEY value>
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--url` | `http://127.0.0.1:8000` | backend base URL |
| `--api-key` | empty | sent as X-API-Key; run_eval reads neither `.env` nor BACKEND_API_KEY, so pass it when auth is on (else `Backend returned HTTP 401`) |
| `--top-k` | `3` | metric cutoff only; the backend still returns up to TOP_K_RESULTS sources |
| `--user-id` | generated `eval-<12 hex>` | tenant to use; naming one implies `--keep` |
| `--keep` | off | do not clear the tenant afterwards |

Lifecycle and output (stdout; errors on stderr):
1. `Uploading N documents as tenant <id>`, then `<file>: <n> chunk(s)` per `CORPUS` file, sent as
   text/plain (0 chunks when identical bytes are already in that tenant).
2. `Answerable questions`: each `GOLDEN_CASES` question (`language: "Auto"`, no history) as `[ok ]`
   or `[MISS]` plus the returned files; the label checks the whole list, not the top `--top-k`.
   `Unanswerable questions`: each `UNANSWERABLE` question with its files, no label.
3. `Metrics`: `cases`, `recall@k`, `precision@k`, `mrr`, `hit_rate@k` over the answerable questions
   only, then advice on choosing RELEVANCE_THRESHOLD.
4. `POST /clear` and `Cleared tenant <id>.`, or `Leaving tenant <id> in place as requested.`
5. `Backend returned HTTP <code>: <body>` or `Could not reach <url>: <reason>` exits 1 without
   clearing. Remove the leftover: `curl -X POST "<url>/clear?user_id=eval-<id>" -H "X-API-Key: <key>"`.

Reading it: the run exercises the live pipeline as configured, with real models; the golden questions
trigger no metadata filter. Every `CORPUS` document is one short chunk, and while the corpus has no
more files than TOP_K_RESULTS (default 5), every question at threshold 0 returns every file: every
label reads `[ok ]`, recall@k and hit_rate@k only say whether the right file made the top k (1.0 once
`--top-k` reaches the file count), and for single-document cases precision@k is hit_rate@k / k. MRR
and each question's source order carry the signal. Similarity scores are only in the backend log.

### 3. Langfuse experiments and public benchmarks

```bash
docker compose -f docker-compose.langfuse.yml up -d     # or Langfuse Cloud
python -m scripts.download_benchmarks                    # no CLI: any call downloads every missing one into data/benchmarks/
python -m scripts.upload_to_langfuse --dataset sberquad --count 200
python -m evaluation.langfuse_eval --dataset eval-sberquad-ru --run-name <name> --keep
python -m evaluation.eval_sberquad_failures --tenant eval-<id> --log-file <saved langfuse_eval output>
```

`langfuse_eval` records a dataset run with per-item scores; its default dataset
`multilingual-rag-golden` is the golden set synced into Langfuse. Only that dataset,
`eval-sberquad-ru` and `eval-financebench` have a corpus mapping; HotpotQA, SciFact, TyDi QA and
RAGTruth items are all scored as unanswerable. Read
[references/langfuse-benchmarks.md](references/langfuse-benchmarks.md) before running any of these
(prerequisites, flags, output, recipes, defects). First traps: the Langfuse keys and host come from
the shell or the root `.env`; set MAX_DOCUMENTS_PER_USER=0 on the backend for corpora over 200
documents; a run against a dataset that does not exist yet crashes after creating it (run it
again); the printed `Aggregate Metrics` mix in unanswerable items, so never compare them with run_eval.

## Metrics and how sources are compared

`evaluation/metrics.py` scores `expected` (file names that should come back) against `retrieved`
(what came back, best first):
- recall@k: share of the distinct expected names in the first k; 1.0 when nothing is expected.
- precision@k: expected names in the first k divided by the size of that slice, not by k (2
  results with 1 hit at k=5 is 0.5); 0.0 for an empty result.
- reciprocal rank: 1 / position of the first expected name in the whole list, no k cutoff; 0.0 if
  none. MRR is its mean.
- hit@k: any expected name in the first k; false when nothing is expected. hit_rate@k: share of
  cases with a hit.
- `aggregate(cases, k)` returns `cases`, `recall@<k>`, `precision@<k>`, `mrr`, `hit_rate@<k>`, and
  zeros for no cases. A k of 0 or less makes the top-k slice empty.

The live tools compare `[s["source"] for s in response["sources"]]` from `POST /query`:
- `source` is the uploaded name after `app/main.py` `safe_filename`, which turns every character
  other than (Unicode) letters, digits, `_`, `.`, space and `-` into `_`. Expected names must survive it.
- `app/rag/chain.py` `RAGChain._prepare` keeps TOP_K_RESULTS chunks (reranker `top_n` or a slice),
  then `RAGChain._collect_sources` keeps each file's first chunk in rank order: at most
  TOP_K_RESULTS distinct files, fewer when one file holds several top chunks.
- So scores rank files, not chunks; a chunk-level change shows only in which files make the cut.

## Recipes

### Measure a change before and after

1. On the unchanged tree, run the offline tier plus the changed area's tests (rag-testing).
2. Write down the retrieval settings and keep all but the one under test identical in both runs:
   TOP_K_RESULTS, RETRIEVAL_CANDIDATES, RELEVANCE_THRESHOLD, MMR_LAMBDA, HYBRID_SEARCH_ENABLED,
   HYBRID_DENSE_WEIGHT, HYBRID_BM25_WEIGHT, HYBRID_RRF_K, RERANKER_ENABLED, RERANKER_PROVIDER,
   RERANKER_MODEL, METADATA_FILTERING_ENABLED, CHUNK_SIZE, CHUNK_OVERLAP, TABLE_AWARE_CHUNKING,
   EMBEDDING_MODEL, MODEL_NAME.
3. Before: baseline backend, real OPENAI_API_KEY, no other traffic (log lines are matched by
   order). Run `python -m evaluation.run_eval --url <url> --api-key <key> 2>&1 | tee before.txt` and
   keep the backend's `retrieval space=...` lines from that run.
4. Apply the change and restart the backend (settings are read once at startup; under Compose
   recreate the container, see rag-development-guide). Run the identical command, same `--top-k`,
   into `after.txt`.
5. Give each run a new tenant (no `--user-id`). Identical bytes uploaded into an existing tenant
   return `duplicate: true, chunks: 0` and keep the old chunks, so a reused tenant silently measures
   the old index after any ingestion change (chunking, loaders, metadata extraction); reuse is safe
   only for retrieval-only changes. A new tenant also gets a complete BM25 index after a restart.
   An EMBEDDING_MODEL change needs its own CHROMA_PERSIST_DIR or COLLECTION_NAME: the backend
   refuses to open a collection built with another model.
6. Chunking changes: the golden documents are far shorter than CHUNK_SIZE and hold no tables, so
   CHUNK_SIZE, CHUNK_OVERLAP or TABLE_AWARE_CHUNKING changes leave them one chunk each and run_eval
   shows nothing. Measure on documents longer than a chunk: long documents with their own cases in
   `CORPUS` (the offline test still indexes them whole), or a benchmark corpus after checking its
   document lengths (SberQuAD contexts are cut to 1000 characters on upload).
7. Compare the `Metrics` blocks and each question's source order. When the golden questions are too
   few, record both sides as Langfuse runs on SberQuAD or FinanceBench (reference file).
8. Record both runs (settings, command, `Metrics` block) in the commit message or PR. If a default
   changes, put the before/after table next to the setting in DOCUMENTATION.md, naming the corpus,
   as the MMR table does; rag-config-and-docs covers the config, template and table rows.
9. If the bag-of-words corpus can express the property, lock it in with a golden case or an
   assertion in `tests/test_retrieval_quality.py`.

### Calibrate RELEVANCE_THRESHOLD

Hybrid search defeats the threshold: `app/rag/chain.py` `RAGChain._retrieve` applies it in
`_retrieve_dense`, then `app/rag/hybrid.py` `HybridRetriever.retrieve` fuses in BM25 hits that never
saw it, and returns them alone when every dense chunk was dropped. With HYBRID_SEARCH_ENABLED=true
(the default) any question sharing a word ("the", "is") with a chunk still reaches the model, so a
threshold measured with hybrid on proves nothing (KNOWN_ISSUES.md, section "Retrieval and search").

1. Start the backend with HYBRID_SEARCH_ENABLED=false, RELEVANCE_THRESHOLD=0, MMR_LAMBDA=1.0 and
   RERANKER_ENABLED=false, otherwise idle.
2. Run run_eval. Logger `app.rag.chain` writes one INFO line per question to the backend console
   (`docker compose logs backend` under Compose), with cosines converted from the index distance:
   `<time> - app.rag.chain - INFO - [<request id>] retrieval space=l2 candidates=5 similarity_best=<x> similarity_worst=<y> threshold=0.0`.
   run_eval sends no X-Request-ID and the `/query` access line shows `user=-`, so match lines to
   questions by order: `GOLDEN_CASES`, then `UNANSWERABLE`.
3. Pick a value between the answerable questions' `similarity_best` and the unanswerable ones',
   nearer the unanswerable side: a threshold set too high drops relevant context silently.
4. Set it in `.env`, restart, run again. recall@3 should hold, precision@3 rise, and the
   unanswerable questions list `[]` (the API answers "No relevant information found." without a
   model call); `dropped n/m chunk(s) below RELEVANCE_THRESHOLD=...` lines show the cut. The golden
   questions are a weak sample: on a real deployment also ask a few answerable and unanswerable ones.
5. Keep the value in the deployment's `.env`; it only gates while HYBRID_SEARCH_ENABLED=false. The
   shipped 0.0 is pinned by `tests/test_relevance.py` `test_the_threshold_defaults_to_disabled` and
   `tests/test_retrieval_quality.py` `test_the_shipped_threshold_is_still_disabled`; changing it is
   a rag-config-and-docs task.

### Add a golden case

1. New answering document: add `"<name>.txt": "<text>"` to `CORPUS` in `evaluation/golden.py`. Use
   a `.txt` name that `safe_filename` leaves unchanged (otherwise the live run can never match),
   text unlike any other document (identical bytes are deduplicated per tenant), and, unless
   chunking is the point, keep it under CHUNK_SIZE (1000) so it is one chunk live, as offline.
2. Append to `GOLDEN_CASES` a dict with exactly `question`, `expected` (non-empty list of `CORPUS`
   keys) and `note` (the property it protects; also the pytest id, so it is required).
3. Reuse the document's words in the question: offline ranking is bag-of-words, so a pure
   paraphrase passes live and fails in CI.
4. Keep years 1900-2039, FY##, Q1-Q4, 10-K, 10-Q, 8-K, "earnings", "annual report" and the company
   aliases in `app/rag/metadata_extractor.py` `COMPANY_ALIASES` (everyday words such as block,
   square, meta, apple) out of questions and file names unless filtering is the point: live,
   METADATA_FILTERING_ENABLED (on by default) turns them into hard filters; offline never filters.
5. A question the corpus must not answer goes into `UNANSWERABLE`, never into `GOLDEN_CASES`.
6. Run `python -m pytest tests/test_retrieval_quality.py tests/test_metrics.py -q`; a new document
   can reorder other cases, and every floor must still hold. Then run the live harness.
7. The next `langfuse_eval` run (or `--sync-only`) adds new questions to `multilingual-rag-golden`
   but never updates existing items: after changing `expected` or wording, fix stale ones in its UI.
8. If the set covers a new property, say so in DOCUMENTATION.md, section "Measuring retrieval".

### Turn ratings into golden cases

1. Ratings are in FEEDBACK_DIR/feedback.jsonl (default `data/feedback/feedback.jsonl`); under
   Compose in the `data_feedback` volume. Dev stack (bind-mounted tree):
   `docker compose exec backend python -m evaluation.from_feedback`. The production image has no
   `evaluation/`: `docker compose cp backend:/app/data/feedback/feedback.jsonl ./feedback.jsonl`.
2. `python -m evaluation.from_feedback --input ./feedback.jsonl --limit 50`. Flags: `--input`
   (default `data/feedback/feedback.jsonl` under the working directory; FEEDBACK_DIR is not read),
   `--up` (liked answers instead of disliked), `--limit` (default 20, newest first). It prints
   `N ratings: U up, D down (P% negative)`, `Documents behind the negative ratings:`,
   `S of T cases rated down, newest first:` (or `Nothing rated down.`), then per case
   `# <at>  <client>  user=<id>`, `# comment: ...` if any, and a stub
   `GoldenCase(question=..., expected_sources=[], ...)` with `# retrieved instead: [...]` and
   `# request_id=...`; `(N more not shown; raise --limit)` when cut. Exit 1 with
   `No ratings file at <path>.` (stderr) or `<path> holds no readable records.`.
3. Decide by hand which document should have answered; the blank `expected_sources=[]` is
   deliberate, since a guess turns one bad answer into a permanently wrong benchmark. Grep the
   backend log for `[<request_id>]` to see that request's retrieval line (rag-operations).
4. There is no GoldenCase type. Write
   `{"question": ..., "expected": ["<doc>.txt"], "note": "from feedback <request_id>: <property>"}`
   (key `expected`, not `expected_sources`), or add the question to `UNANSWERABLE` if the corpus
   truly cannot answer it.
5. Ratings are real users' questions about private documents: add to `CORPUS` only a minimal,
   de-identified text that answers the question, never a user's file. Then continue with "Add a
   golden case" from step 3.

## Invariants

- Golden cases are dicts with `question`, `expected` (non-empty, all `CORPUS` keys) and `note`; the
  integrity tests pin this, since an empty expectation makes recall trivially 1.0 and a typo makes a
  case unpassable.
- The floors in `tests/test_retrieval_quality.py` sit below what the pipeline achieves; lowering
  one to get a change through hides the regression it exists to catch.
- The feedback tooling never fills in the expectation (`test_a_stub_leaves_the_expectation_blank`),
  and RELEVANCE_THRESHOLD (0.0) and MMR_LAMBDA (1.0) change default only with a recorded measurement.
- `metrics.py`, `golden.py`, `run_eval.py` and `from_feedback.py` (with `app/feedback.py`) stay
  stdlib only, so they run from a bare Python 3.10 checkout; the Langfuse scripts need requirements.
- Tests stay offline: `tests/conftest.py` fails any off-machine connection. The live and Langfuse
  scripts have no tests; a new one must fake urllib, requests and the Langfuse client.

## Pitfalls and known issues

- Pass only `eval-*` or your own scratch ids to run_eval's `--user-id`: it uploads the golden files
  into that tenant and never clears it, and a same-named file with other content replaces the user's copy.
- MMR_LAMBDA below 1.0 turns any question with a year, quarter, filing type or company alias, and
  any request with `metadata_filter`, into a 503; on SberQuAD or FinanceBench those show as `ERROR`
  lines and drop out of the aggregate (KNOWN_ISSUES.md, section "Retrieval and search").
- A question with a year or company alias is hard-filtered to documents carrying the same tag and
  loses BM25 matching, so benchmark numbers depend on METADATA_FILTERING_ENABLED and
  HYBRID_SEARCH_ENABLED; record both (same section). Hybrid also defeats the threshold (Calibrate).
- After a backend restart, a reused tenant that gets a new upload before its first question keeps
  its older documents out of BM25 until the next restart (same section); fresh tenants avoid it.
- `flashrank` is installed nowhere, so `--reranker flashrank` or RERANKER_PROVIDER=flashrank
  measures plain retrieval order (section "Models, keys and reranking").
- Langfuse layer defects (the reference file; section "Evaluation and benchmarks"): first-run
  crash, biased aggregate, quota misses, broken RAGTruth upload, author-specific defaults in
  `eval_sberquad_failures`, and `download_benchmarks` downloading even on `--help`.

## Verify

```bash
python -m pytest tests/test_metrics.py tests/test_retrieval_quality.py tests/test_mmr.py -q
python -m pytest tests/test_feedback.py -q -k "stub or report"
ruff check app frontend clients evaluation scripts tests
python -m evaluation.run_eval --help
```

`--help` is safe for every module here except `scripts.download_benchmarks`, which starts
downloading. Which Python to call on Windows or WSL: rag-development-guide and rag-testing.

## Related skills

rag-retrieval-pipeline, rag-document-ingestion, rag-testing, rag-operations, rag-llm-providers,
rag-config-and-docs, rag-change-review, rag-development-guide (roles under "Use this skill for / not for").
