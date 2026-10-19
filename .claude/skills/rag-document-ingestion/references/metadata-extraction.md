# Metadata extraction and question filters

Reference for `app/rag/metadata_extractor.py`. Every example below is the function's real output.

Contents
- Where the extractor runs
- Document side: `extract_document_metadata`
- Company matching: `normalize_company` and the alias table
- Question side: `extract_query_metadata`
- `build_chroma_filter` and value types
- The patterns, verbatim
- Filename examples
- Question examples (English and Russian)
- Changing the rules

## Where the extractor runs

- Upload: `app/main.py` `_store_and_index` calls `extract_document_metadata(safe_name, chunks[0].page_content)` once per upload and applies the result to every chunk after `source`, `file_hash` and `user_id`. It runs whatever `METADATA_FILTERING_ENABLED` says; the tags are stored in ChromaDB and the BM25 index.
- Questions: `app/rag/chain.py` `RAGChain._prepare` calls `extract_query_metadata(question)` only when `METADATA_FILTERING_ENABLED` is true, lays the request's `metadata_filter` over the result (explicit keys win; an explicit `null` cancels a detected key because `build_chroma_filter` drops `None`), re-runs detection on the condensed follow-up when the question and the explicit filter together gave nothing, and builds the where-clause with `build_chroma_filter`. A non-empty filter that returns no chunks is retried as `{"user_id": user_id}`, dropping every constraint at once. The rag-retrieval-pipeline skill owns that code.
- Answers: `RAGChain._collect_sources` copies `company`, `year`, `quarter` and `doc_type` into each source's `metadata`; the web UI shows them as `key: value` tags after the file name.
- Evaluation: `evaluation/langfuse_eval.py` builds explicit FinanceBench filters with `normalize_company`, so alias changes move benchmark numbers.

## Document side: extract_document_metadata(filename, text)

`filename` is the clean upload name; `text` is the first chunk only: the start of the first PDF page that has text, the opening paragraphs of a DOCX (its tables come last).

1. Drop everything up to the last `/` and a leading 16-hex-digit `<hash>_` prefix (for callers that pass a stored name).
2. Split the name into tokens on `_`, `.`, `-` and whitespace.
3. company: `normalize_company` on the whole name; else on each token; else the first token uppercased, if it is alphabetic (`str.isalpha`, so Cyrillic counts) and at least 2 characters long. A first token with a digit (`x1.pdf`, `Q3_report.txt`) gives no company.
4. year (int): the first token holding a standalone 1900-2039; else `FY_YEAR_PATTERN` (FY plus 2 or 4 digits, 2 digits read as 20yy) in the first 600 characters of `text`; else the first standalone 1900-2039 in its first 400 characters.
5. quarter: the first token with Q1-Q4 (also glued to a year, as in 2023Q2) or 1Q-4Q, returned as `Qn`. No text fallback.
6. doc_type: the first token equal to 10K, 10Q, 8K or EARNINGS (any case); else the first 10-K, 10K, 10-Q, 10Q, 8-K, 8K or EARNINGS in the first 500 characters of `text`. Stored as `10-K`, `10-Q`, `8-K` or `EARNINGS`. A hyphenated form in the filename is split into two tokens and missed. `ANNUAL REPORT` is never produced here.
7. Only the keys found are returned.

The naming convention it was tuned for is FinanceBench's `COMPANY_YEAR[Qn]_DOCTYPE[_pN].ext` (`evaluation/langfuse_eval.py` uploads pages as `<doc_name>_p<page>.txt`).

## Company matching

`normalize_company(raw)`:
- turns every character outside `[A-Za-z0-9&]` into a space and uppercases, so `Coca-Cola` matches the `COCA COLA` alias, an alias written with a hyphen (the existing `COCA-COLA`) can never match, and Cyrillic text never matches anything;
- tries companies in descending order of their longest alias and, inside a company, its aliases longest first; an alias matches only as a whole word (no letter or digit on either side);
- returns the first canonical key that matches, else `""`. One company per text.

`COMPANY_ALIASES` maps a few dozen canonical keys (FinanceBench's S&P 500 companies: WALMART, COCACOLA, JOHNSON_JOHNSON, 3M, ...) to uppercase names and tickers. `ALL_COMPANY_ALIASES` adds each key itself plus its `_`-to-space and `_`-removed forms. The key is what ChromaDB stores and what an explicit `metadata_filter` must send.

Aliases that fire on ordinary words (each checked): BLOCK and SQUARE (both -> BLOCK), META, APPLE, AES, ALPHABET (-> GOOGLE), GOOGLE, AMAZON, BEST BUY, AMERICAN WATER, CVS, KRAFT, NIKE, TESLA, CORNING, FOOT LOCKER, MGM, 3M, MICROSOFT ("the Microsoft Word file").

## Question side: extract_query_metadata(query)

- company: `normalize_company(query)`, so at most one company and no Cyrillic names. "Compare Walmart and Costco" filters to WALMART alone.
- year: every standalone 1900-2039 number (amounts and room numbers included) plus `FY` with 2 digits (20yy). Then, by the number of distinct years:
  - 1: the int itself; if the question contains a word starting with forecast, guidance, expect, outlook, project or accelerat (English only), `{"$in": [y - 1, y]}`;
  - 2 to 5: `{"$in": every year from min to max}` when max - min is at most 5, else `{"$in": just the years named}`;
  - 6 or more: no year filter.
- quarter: the first Q1-Q4 or 1Q-4Q (any case) -> `Qn`. Russian questions match too ("цели на Q1").
- doc_type: the first 10-K, 10K, 10-Q, 10Q, 8-K, 8K, EARNINGS or ANNUAL REPORT (any case) -> `10-K`, `10-Q`, `8-K`, `EARNINGS`, or `ANNUAL REPORT` with the question's inner whitespace kept. No document is ever tagged `ANNUAL REPORT`, so that filter always misses and falls back to the whole tenant.
- The `available_companies` parameter is unused.

## build_chroma_filter and value types

- `build_chroma_filter(u)` -> `{"user_id": u}`; with constraints -> `{"$and": [{"user_id": u}, {key: value}, ...]}` in dict order.
- Any `user_id` key and every `None` value are skipped; an empty `u` raises `ValueError`.
- Values pass through unvalidated. Chroma equality is type-strict (`2020` and `"2020"` differ), a malformed operator reaches Chroma and the query fails with 503, and the BM25 index evaluates only plain equality and `{"$in": [...]}`.
- Stored tag values are str or int. Any new tag must be str, int, float or bool: ChromaDB rejects `None`, lists and dicts at upsert and the upload becomes a 503.
- Do not confuse it with `EmbeddingsManager._owned` in `app/rag/embeddings.py`, which builds `{"user_id": {"$eq": owner}}` for EmbeddingsManager's own lookups and deletes. `RAGChain` and `HybridRetriever` need the `build_chroma_filter` shape and crash on the `_owned` one.

## The patterns, verbatim

"(ci)" marks a case-insensitive match.
- `FY_YEAR_PATTERN` `\bFY\s*(?:20)?(\d{2,4})\b` (ci): document text fallback.
- `QUARTER_PATTERN` `\b(Q[1-4]|1Q|2Q|3Q|4Q)\b` (ci): questions.
- `DOC_TYPE_PATTERN` `\b(10-?K|10-?Q|8-?K|EARNINGS|ANNUAL\s+REPORT)\b` (ci): questions only.
- `YEAR_PATTERN` `\b(19\d\d|20[0-3]\d)\b`: defined, never used.
- Inline in `extract_document_metadata`: hash prefix `^[0-9a-f]{16}_` (ci); token split `[_.\-\s]+`; year `(?<!\d)(19\d\d|20[0-3]\d)(?!\d)`; quarter `(?:(?<=\d)|(?<=\b))Q[1-4]\b|(?:\b[1-4]Q\b)` (ci); doc type in a token `\b(10K|10Q|8K|EARNINGS)\b` (ci), in the text `\b(10-?K|10-?Q|8-?K|EARNINGS)\b` (ci).
- Inline in `extract_query_metadata`: years `(?<!\d)(?:19\d\d|20[0-3]\d)(?!\d)` and `\bFY\s*(\d{2})\b` (ci); widening `\b(forecast|guidance|expect|outlook|project|accelerat)\w*` (ci).
- Inline in `normalize_company`: cleaning `[^A-Za-z0-9&]` -> space, then per alias `(?<![A-Z0-9])` + the escaped alias + `(?![A-Z0-9])`.

## Filename examples

| Clean filename | First-chunk text | Tags |
| --- | --- | --- |
| `WALMART_2020_10K_p50.txt` | - | company WALMART, year 2020, doc_type 10-K |
| `Pfizer_2023Q2_10Q_p40.txt` | - | PFIZER, 2023, quarter Q2, 10-Q |
| `ULTABEAUTY_2023Q4_EARNINGS_p1.txt` | - | ULTABEAUTY, 2023, Q4, EARNINGS |
| `AMD_3Q2022_8K.pdf` | - | AMD, 2022, 8-K (no quarter: `3Q2022` is not a standalone 3Q) |
| `WALMART_2020_10-K.pdf` | - | WALMART, 2020 (no doc_type) |
| `Coca-Cola_2022_10-K.pdf` | Form 10-K annual report | COCACOLA, 2022, 10-K (from the text) |
| `policy.docx` | - | company POLICY |
| `annual_report.txt` | Approved 12.03.2021. Annual report. | company ANNUAL, year 2021 |
| `report.txt` | FY21 results | REPORT, 2021 |
| `salary_survey.txt` | Average earnings across roles | SALARY, doc_type EARNINGS |
| `apple_pie_recipe.txt` | - | company APPLE |
| `Meta-analysis.pdf` | - | company META |
| `Отчет_2023.docx` | - | company ОТЧЕТ, year 2023 |
| `Договор аренды _15.docx` | офис 1905 | company ДОГОВОР, year 1905 |
| `2023-annual-report.pdf` | - | year 2023 |
| `x1.pdf` | - | nothing |

## Question examples

| Question | Filter |
| --- | --- |
| What was Walmart's net revenue in FY2020? | company WALMART, year 2020 |
| What was revenue in FY20? | year 2020 |
| What is the FY2018 - FY2020 3 year average EBITDA margin for Walmart? | WALMART, year `$in` [2018, 2019, 2020] |
| What happened between 1990 and 2010? | year `$in` [1990, 2010] (the years between are excluded) |
| Revenue in 2015, 2016, 2017, 2018, 2019 and 2020 | nothing (six years) |
| What is the 2024 guidance? | year `$in` [2023, 2024] |
| project budget for 2024 | year `$in` [2023, 2024] |
| budget of 2000 dollars | year 2000 |
| As of Q2 2023, is Pfizer spinning off a business? | PFIZER, 2023, quarter Q2 |
| What does the annual report say? | doc_type ANNUAL REPORT |
| Any 8K filings? | doc_type 8-K |
| average earnings of employees | doc_type EARNINGS |
| How many square meters is the office? | company BLOCK |
| Is meta data stored? | company META |
| Coca-Cola sales / KO dividend | company COCACOLA |
| How to bake sourdough bread? | nothing |
| Сколько дней отпуска положено сотрудникам в 2023 году? | year 2023 |
| Какая выручка Walmart в 2020 году? | WALMART, 2020 |
| Какая выручка Уолмарт в 2020 году? | year 2020 (no company: Cyrillic) |
| Прогноз на 2025 год | year 2025 (no widening: the forecast words are English) |
| Что сказано в годовом отчете? | nothing |
| Какие цели на Q1? | quarter Q1 |
| Сравни AES и RSA | company AES |

To see what a question triggers, run from the repository root:

```bash
python -c "from app.rag.metadata_extractor import extract_query_metadata as q, build_chroma_filter as b; f = q('What was Walmart revenue in 2020?'); print(f, b('u1', f))"
```

## Changing the rules

- The document side and the question side must produce the same string or int for the same thing (`10-K`, `2020`). Doc types live in three regexes and three copies of the normalisation: `DOC_TYPE_PATTERN` for questions, plus the filename-token and first-chunk regexes inside `extract_document_metadata`.
- Keep values scalar, and never return `user_id`, `source` or `file_hash` from `extract_document_metadata`: `main.py` applies the tags after setting those keys, so they would be overwritten.
- Add a positive and a negative case to `tests/test_metadata_filtering.py` for every rule (a generic question must still give `{}`).
- Documents keep the tags they were uploaded with; only new uploads see a change.
- Before turning a rule on by default, measure it on a non-financial corpus too (rag-evaluation skill): a false positive that matches some document hides the right one, because the fallback only runs on zero hits.
