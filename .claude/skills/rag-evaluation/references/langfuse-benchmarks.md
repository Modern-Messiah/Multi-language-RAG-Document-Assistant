# Langfuse experiments and public benchmarks

Read this before running `scripts.download_benchmarks`, `scripts.upload_to_langfuse`,
`evaluation.langfuse_eval` or `evaluation.eval_sberquad_failures`, or when changing them. Code:
`scripts/download_benchmarks.py`, `scripts/upload_to_langfuse.py`, `evaluation/langfuse_eval.py`
(`sync_dataset_items`, `get_corpus_and_expectations`, `run_experiment`),
`evaluation/eval_sberquad_failures.py` (`extract_failed_indices`, `run_hard_cases_benchmark`).
None of it has tests; CI only lints it.

Contents:
- Prerequisites
- 1. Download: python -m scripts.download_benchmarks
- 2. Upload: python -m scripts.upload_to_langfuse
- 3. Run: python -m evaluation.langfuse_eval
- 4. Hard cases: python -m evaluation.eval_sberquad_failures
- Recipe: a SberQuAD baseline and variants
- Recipe: make another benchmark evaluable
- Known issues

## Prerequisites

1. Network. The downloader fetches from huggingface.co (FinanceBench file, SberQuAD, HotpotQA and
   TyDi QA parquet through `huggingface_hub`), public.ukp.informatik.tu-darmstadt.de (SciFact) and
   raw.githubusercontent.com (RAGTruth); the uploader fetches HotpotQA and TyDi QA again.
2. A Langfuse v2 server: `docker compose -f docker-compose.langfuse.yml up -d` runs
   `langfuse/langfuse:2` plus Postgres with the UI on http://localhost:3000 (create a project and
   its API keys there), or use Langfuse Cloud. The compose file hard-codes its secrets and publishes
   port 3000 on all interfaces: local machines only. Stack details, and reaching it from a
   containerised backend: the rag-operations skill (its Langfuse reference).
3. Keys for the scripts: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST`
   (`http://localhost:3000` for the local stack; the SDK defaults to https://cloud.langfuse.com),
   in the shell or in the repository-root `.env`. The three Langfuse scripts call `load_dotenv()`,
   which finds that file by walking up from the script, and variables already set in the shell win.
   `LANGFUSE_ENABLED` is not read by the scripts; it only switches the backend's tracer.
4. The project requirements (`langfuse==2.60.10`; the scripts use v2 calls such as
   `get_dataset(...).items`, `item.link(...)` and `Langfuse.score(...)`, so re-check them before any
   SDK upgrade). `pyarrow` and `huggingface_hub` are imported but are only transitive dependencies.
5. A running backend with a real `OPENAI_API_KEY`. For corpora over 200 documents set
   `MAX_DOCUMENTS_PER_USER=0` (no limit) and restart it; every benchmark document goes into one
   tenant. Optionally `LANGFUSE_ENABLED=true` with the same project keys: the backend then records
   each question as a `rag-query` trace whose id is the `X-Request-ID` the script sends, so every
   run item opens the real trace with its retrieval span, prompt and answer.
6. Cost: each upload embeds a document; each question embeds the query and, when anything is
   retrieved, runs a chat completion. A full SberQuAD run is thousands of questions. Ask first.

## 1. Download: python -m scripts.download_benchmarks

No argument parser: every invocation, `--help` included, downloads all six datasets. Files land
in `data/benchmarks/` of the checkout (derived from the script's location, not the working
directory); a file already larger than 1000 bytes is skipped, so delete it to refresh.

| Directory | File | Content |
| --- | --- | --- |
| `financebench/` | `financebench.jsonl` | FinanceBench questions, answers and evidence pages |
| `sberquad/` | `sberquad_val.jsonl` | SberQuAD validation: id, title, context, question, answer |
| `hotpotqa/` | `hotpotqa_sample.jsonl` | first 200 HotpotQA distractor validation items |
| `beir_scifact/` | `corpus.jsonl`, `queries.jsonl` and the rest of the archive | BEIR SciFact |
| `tydiqa/` | `tydiqa_sample.jsonl` | up to 200 TyDi QA items (Russian or English), text cut to 2000 characters |
| `ragtruth/` | `ragtruth_qa.jsonl` | RAGTruth `response.jsonl` (model responses; the uploader cannot use it) |

`data/benchmarks/` is git-ignored but not listed in `.dockerignore`, so the Dockerfile's copy of the
build context (`COPY --chown=appuser:appuser . .`) bakes a local download into the next image build: delete or move it before `docker compose build`.

## 2. Upload: python -m scripts.upload_to_langfuse

```bash
python -m scripts.upload_to_langfuse --dataset sberquad --count 200   # or --all
```

Flags: `--dataset {financebench,sberquad,hotpotqa,scifact,tydiqa,ragtruth,all}` (default `all`),
`--count N` (items per dataset, default 100), `--all` (no limit; overrides `--count`).

| `--dataset` | Langfuse dataset | Read from | Item id | Evaluable by langfuse_eval |
| --- | --- | --- | --- | --- |
| `financebench` | `eval-financebench` | `financebench/financebench.jsonl` | `fb-<financebench_id>` | yes: one document per evidence page |
| `sberquad` | `eval-sberquad-ru` | `sberquad/sberquad_val.jsonl` | `sber-<id>` | yes: one document per distinct context |
| `hotpotqa` | `eval-hotpotqa-multihop` | Hugging Face again (local sample on error) | `hotpot-<id>` | no |
| `scifact` | `eval-beir-scifact` | `beir_scifact/queries.jsonl` | `scifact-<_id>` | no |
| `tydiqa` | `eval-tydiqa-multilingual` | Hugging Face again, all languages (0 items on error) | `tydi-<language>-<index>` | no |
| `ragtruth` | `eval-ragtruth-hallucinations` | `ragtruth/source_info.jsonl` (not downloaded) | `ragtruth-<source_id>` | no |

- A dataset is created when looking it up raises any error. Items go up on 24 threads; item ids are
  deterministic, so a re-upload updates items in place, except an item without a source id (id
  None), which is duplicated on every run.
- SberQuAD items keep the context in `metadata.context`, cut to 1000 characters; an answer span
  beyond that is missing from the indexed text. FinanceBench items keep `company`, `doc_name` and
  `question_type` in metadata and `answer` plus `evidence` in `expected_output`.
- Output is logging: `Uploading N items to '<dataset>' ...`, then per dataset
  `Finished '<dataset>': X / Y items uploaded successfully.` Per-item errors are logged at DEBUG
  only, and the closing "Upload to Langfuse completed!" prints whatever happened, so check each
  `X / Y` line.

## 3. Run: python -m evaluation.langfuse_eval

```bash
python -m evaluation.langfuse_eval --dataset eval-sberquad-ru --run-name sberquad-baseline --keep 2>&1 | tee sberquad-baseline.log
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--url` | `http://127.0.0.1:8000` | backend base URL |
| `--api-key` | `BACKEND_API_KEY` from the shell or `.env` | sent as X-API-Key |
| `--dataset` | `multilingual-rag-golden` | Langfuse dataset to run |
| `--run-name` | `eval-<model or default>[-<reranker>]-<YYYYmmdd-HHMMSS>` | dataset run name; pick unique, descriptive names |
| `--top-k` | `3` | cutoff for PASS/FAIL and the precision and recall scores |
| `--limit` | all | evaluate only the first N items |
| `--provider`, `--model`, `--model-key` | none | BYOK headers, sent only when `--model-key` is given; without it the server's model answers while the run name and metadata still name `--model` |
| `--reranker` | `none` | `none`, `flashrank` or `cohere`; `none` sends no reranker header, so the backend's RERANKER_* settings decide |
| `--reranker-key`, `--reranker-model` | empty | X-Reranker-Key and X-Reranker-Model; cohere without a key spends the backend's RERANKER_API_KEY, or every query gets 400 when that is empty |
| `--keep` | off | do not clear the scratch tenant |
| `--tenant` | none | reuse an indexed tenant: nothing is uploaded and the tenant is never cleared |
| `--sync-only` | off | only create or sync the dataset, then exit |

What a run does:
1. `sync_dataset_items`: gets the dataset, creating it if the lookup fails. For
   `multilingual-rag-golden` only, it adds every `GOLDEN_CASES` question and every `UNANSWERABLE`
   question not yet present (matched by question text, with `expected_output.expected_sources`);
   it never updates or removes an existing item.
2. `get_corpus_and_expectations` maps items to files and expected file names:
   `eval-financebench` makes `<doc_name>_p<page>.txt` per evidence page (an item without evidence
   gets a stub `<doc_name>.txt`); `eval-sberquad-ru` makes `doc_<n>_<title>_<hash>.txt` per distinct
   context, numbered by first appearance in the evaluated item list, and an item with an empty
   context expects nothing; every other dataset name gets the golden `CORPUS` and the items'
   `expected_sources`, so HotpotQA, SciFact, TyDi QA and RAGTruth items all expect nothing.
3. Upload into a scratch `eval-<12 hex>` tenant: `Uploading N documents to tenant: eval-<id>...`,
   one line per file, or a progress line every 50 files for corpora over 200. A failed upload prints
   `Warning: failed to upload <file>: <error>` and the run carries on. With `--tenant` nothing is
   uploaded; it prints `Reusing existing tenant <id> with pre-indexed corpus.`
4. Each item: `POST /query` with `language: "Auto"`, `X-Request-ID` = a fresh 16-hex trace id, and
   `metadata_filter: {"company": ...}` when the item's metadata has a company (FinanceBench). It
   prints `[i/N] [PASS|FAIL] <question> -> [<files>] (<ms>ms)`: PASS means hit@top-k, or for an item
   expecting nothing, an empty result. The item is linked to the run and scored on that trace id:
   `passed` (boolean), `retrieval_hit`, `precision@<k>` and `recall@<k>` (answerable items only) and
   `latency_ms`. A query that raises prints `[i/N] ERROR: ...` and the item is skipped entirely.
5. `--- Aggregate Metrics ---` (biased, see Known issues), `Total Duration`, then
   `Cleared temporary tenant <id>.` or `Kept tenant <id> as requested.` An uncaught exception
   leaves the scratch tenant behind.

Compare runs in the Langfuse UI under Datasets, the dataset, Runs; the per-run score averages there
come from the per-item scores above, not from the printed aggregate.

## 4. Hard cases: python -m evaluation.eval_sberquad_failures

Re-asks SberQuAD items that FAILed in an earlier `langfuse_eval` run against a tenant that already
holds the SberQuAD corpus. It uploads nothing and writes nothing to Langfuse.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--url` | `http://127.0.0.1:8000` | backend base URL |
| `--tenant` | `eval-528d05007556` (the author's tenant) | always pass the tenant from your baseline run |
| `--log-file` | none | saved `langfuse_eval` output to read `[i/N] [FAIL]` lines from |
| `--limit` | `50` | how many failed items to re-ask |
| `--top-k` | `5` | parsed but unused |
| `--reranker` | none | `none`, `flashrank` or `cohere`; sends X-Reranker-Provider but never a key, so cohere needs the backend's RERANKER_API_KEY |
| `--reranker-model` | none | X-Reranker-Model |

- Reads `eval-sberquad-ru` from Langfuse (keys needed) and rebuilds the same file names as
  `langfuse_eval`. X-API-Key comes from `BACKEND_API_KEY` in the shell or `.env`; there is no flag.
- `--log-file` lines count only when their `N` equals the dataset's current size; otherwise, or
  without a log, it silently uses 20 hard-coded indices up to 134.
- Queries use `requests` with a 30 s timeout and `language: "ru"`. An error response has no
  `sources` and counts as FAIL; an exception prints `ERROR` and still counts in the denominator.
- Output: a banner, `[i/N] [RECOVERED|FAIL] <question> -> Rank <r>` (or `N/A`) per case (RECOVERED
  means a hit in the top 5), then `New Hit Rate@3`, `New Hit Rate@5` and `Average Latency`. The banner's
  pipeline description and "Baseline Hit Rate: 0.0%" are fixed text. The baseline FAILs were judged
  at `langfuse_eval`'s `--top-k` (3 by default), so compare `New Hit Rate@3`.

## Recipe: a SberQuAD baseline and variants

1. Prerequisites above; record the backend's retrieval settings (the list in SKILL.md).
2. `python -m scripts.download_benchmarks`, then
   `python -m scripts.upload_to_langfuse --dataset sberquad --count 200` (at least 134 items if
   `eval_sberquad_failures` will fall back to its built-in indices; `--all` for the full set).
3. Backend with `MAX_DOCUMENTS_PER_USER=0`, restarted.
4. Baseline:
   `python -m evaluation.langfuse_eval --dataset eval-sberquad-ru --run-name sberquad-baseline --keep 2>&1 | tee sberquad-baseline.log`.
   No `--limit` if the log will feed `eval_sberquad_failures`. Note the tenant from the
   `Uploading N documents to tenant: eval-<id>...` line, and check that no upload warning appeared.
5. Retrieval-only variant (threshold, hybrid weights, reranker, MMR): change the setting, restart,
   then `python -m evaluation.langfuse_eval --dataset eval-sberquad-ru --tenant eval-<id> --run-name sberquad-<variant>`.
   Reuse a tenant only with the same dataset contents and an equal or smaller `--limit`, since the
   file names depend on item order. Any ingestion change (chunking, loaders, metadata extraction)
   needs a fresh tenant: omit `--tenant`. An EMBEDDING_MODEL change also needs its own
   CHROMA_PERSIST_DIR or COLLECTION_NAME (SKILL.md, before/after recipe).
6. Hard cases:
   `python -m evaluation.eval_sberquad_failures --tenant eval-<id> --log-file sberquad-baseline.log --limit 50`.
7. Compare the runs in the Langfuse UI and the PASS/FAIL lines, not the printed aggregates.
8. Clean up every kept tenant:
   `curl -X POST "http://127.0.0.1:8000/clear?user_id=eval-<id>" -H "X-API-Key: <key>"`.

## Recipe: make another benchmark evaluable

1. `scripts/download_benchmarks.py`: a `download_<name>()` writing under `BASE_DIR / "<name>"`
   with the same skip-if-present check, called from `main()`.
2. `scripts/upload_to_langfuse.py`: an `upload_<name>(client, max_count)` creating `eval-<name>`
   with deterministic, never-None ids `<prefix>-<source id>` (unique across datasets), input
   `{"question": ...}`, and the passage text the corpus needs in metadata or `expected_output`; add
   the name to `--dataset` choices and to `main()`.
3. `evaluation/langfuse_eval.py` `get_corpus_and_expectations`: an `elif dataset_name == "eval-<name>"`
   branch returning `({file name: text}, expected file names per item)`. Use `.txt` names that
   `app/main.py` `safe_filename` leaves unchanged and unique texts (identical bytes are deduplicated
   per tenant). Without a branch every item silently expects nothing.
4. Check the corpus size against MAX_DOCUMENTS_PER_USER.
5. Add an offline test (`tests/test_<name>.py`) with `SimpleNamespace` items and fake Langfuse and
   backend objects; the conftest socket guard fails any real connection (rag-testing).
6. A new third-party import goes into `requirements.txt` with both locks regenerated
   (rag-development-guide). Document the dataset and command in DOCUMENTATION.md; script-only
   variables stay out of `.env.template` and the Configuration table (rag-config-and-docs).

## Known issues

All verified; details and workarounds in KNOWN_ISSUES.md, section "Evaluation and benchmarks",
unless another section is named.

- The first `langfuse_eval` run on a project without the dataset (also `--sync-only` for the
  default dataset) crashes with `AttributeError: 'Dataset' object has no attribute 'items'` after
  creating it; run it again. A misspelled `--dataset` leaves an empty stray dataset, and the rerun
  evaluates nothing and reports success. Any lookup error (401, 5xx, timeout) is treated as "missing".
- The printed aggregate averages over every item whose query returned: items expecting nothing
  count as misses for hit rate, precision and MRR and as perfect recall, and failed queries drop
  out. On the golden set hit rate is capped at answerable/total. Use PASS/FAIL lines or the
  Langfuse scores; never compare it with `run_eval`.
- Corpora over 200 documents hit the default upload quota: `HTTP Error 413` warnings, refused
  documents scored as retrieval failures, success banner. The progress line counts attempts.
- The RAGTruth upload reads a file the downloader never writes, logs an error, and the uploader
  still reports success.
- `eval_sberquad_failures` defaults to the author's tenant and hard-coded indices (IndexError on
  datasets under 134 items), ignores `--top-k`, and `--limit 0` raises ZeroDivisionError.
- `download_benchmarks` downloads on any invocation, even `--help`.
- `flashrank` is installed nowhere, so `--reranker flashrank` measures plain retrieval order
  (section "Models, keys and reranking").
- FinanceBench queries carry a company `metadata_filter`, which makes retrieval dense-only, and
  with MMR_LAMBDA below 1.0 every such query fails with 503 and shows as `ERROR` (section
  "Retrieval and search").
