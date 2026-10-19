---
name: rag-document-ingestion
description: Covers how uploads become chunks - DocumentLoader formats (PDF, DOCX with tables, TXT and Markdown with encoding detection), TextChunker and the table-aware splitter, metadata extraction (company, year, quarter, doc_type) and the filters the same extractor builds for questions, upload errors and very large documents. Use when adding a file format, fixing parsing, encoding or chunking problems, changing CHUNK_SIZE, CHUNK_OVERLAP or TABLE_AWARE_CHUNKING, or editing metadata_extractor aliases and patterns.
---

# Document ingestion: loaders, chunking and metadata tags

After `POST /upload` has accepted a file, `app/main.py` `_store_and_index` passes the stored copy to `DocumentLoader.load_document`, cuts the result with `TextChunker` (by default through `TableAwareSplitter`), tags every chunk with company, year, quarter and doc_type from `extract_document_metadata`, and writes the chunks to ChromaDB and the in-memory BM25 index. The same extractor module turns questions into Chroma filters. This skill gives the real behaviour of each step, the recipes for changing it, and the defects on this path.

## Use this skill for / not for

Use it for: a new file format; parsing, encoding, table or chunking problems; `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TABLE_AWARE_CHUNKING`; company aliases, doc types and year rules in `app/rag/metadata_extractor.py`; an upload refused with one of the 400 or 503 details below; very large documents.

Not for (sibling skill):
- how `RAGChain._prepare` applies the filter (merging the explicit `metadata_filter`, re-detection on condensed follow-ups, the zero-hit fallback, MMR and BM25 under filters): rag-retrieval-pipeline.
- the upload handler's auth, owner lock, dedup, quotas (413) and status-code conventions: rag-api-endpoints.
- a new Settings field and the doc-drift tests: rag-config-and-docs. Fixtures and fakes: rag-testing. Dependency pins and lock files: rag-development-guide.
- the web UI file picker and Metadata Filter panel, the bot's upload replies: rag-clients. Measuring a chunking change: rag-evaluation.

## Map

| File | Responsibility | Key symbols |
| --- | --- | --- |
| `app/rag/document_loader.py` | the format list, one loader per format | `SUPPORTED_EXTENSIONS`, `TEXT_EXTENSIONS`, `DocumentLoader.load_document`, `load_pdf`, `load_docx`, `load_txt` |
| `app/rag/text_splitter.py` | chunking entry point and the plain path | `TextChunker`, `TextChunker.split_documents` |
| `app/rag/table_splitter.py` | table detection and table-aware packing | `MD_TABLE_SEPARATOR_PATTERN`, `is_markdown_table_row`, `is_markdown_separator_row`, `extract_segments`, `TableAwareSplitter.split_text`, `TableAwareSplitter.split_documents`, `_split_large_table`, `_split_large_html_table` |
| `app/rag/metadata_extractor.py` | document tags and question filters | `COMPANY_ALIASES`, `normalize_company`, `extract_document_metadata`, `extract_query_metadata`, `build_chroma_filter`, `DOC_TYPE_PATTERN`, `QUARTER_PATTERN`, `FY_YEAR_PATTERN` |
| `app/main.py` | wiring and the upload pipeline | `lifespan` (`app.state.loader`, `app.state.chunker`), `safe_filename`, `MAX_EXT_LEN`, `upload_document`, `_store_and_index` |
| `app/config.py` | defaults | `chunk_size`, `chunk_overlap`, `table_aware_chunking`, `metadata_filtering_enabled`, `max_file_size`, `Settings._chunk_overlap_fits_in_chunk` |
| `app/rag/embeddings.py` | the Chroma write and the document listing | `EmbeddingsManager.add_documents`, `EmbeddingsManager.list_documents` |

## How an upload becomes chunks

1. `upload_document`: `safe_filename`, the extension gate, size and empty checks, `file_hash = sha256(bytes)[:16]`, then under the owner lock `_store_and_index` (dedup: identical bytes from the same owner return `duplicate=true`, `chunks=0`; then quotas).
2. The bytes are written to `UPLOAD_DIR/<user_id>/<file_hash>_<safe_name>`.
3. `state.loader.load_document(path)` picks a loader by `Path.suffix.lower()`.
4. `state.chunker.split_documents(docs)`.
5. `extract_document_metadata(safe_name, chunks[0].page_content)`; every chunk gets `source=safe_name` (the loader saw the hashed name), `file_hash` and `user_id`, then the tags, applied last.
6. Ids `f"{user_id}-{file_hash}-{i}"` in chunk order; `state.embeddings.add_documents(chunks, ids=ids)`, then `state.bm25_index.add_documents(chunks, owner=user_id)`.
7. Older revisions stored under the same filename are retired.

The format list: `SUPPORTED_EXTENSIONS = (".pdf", ".docx", *TEXT_EXTENSIONS)` with `TEXT_EXTENSIONS = (".txt", ".md", ".markdown")` is the only one. Readers: the gate and its 400 text in `upload_document`; the routing and error message of `load_document`; `frontend/streamlit_app.py` (`UPLOAD_LABEL`, the `st.file_uploader` types); `clients/telegram_bot.py` (`SUPPORTED_LIST` in its replies, the check in `handle_document`). `tests/test_docx_markdown.py` pins it with `test_the_extension_list_is_the_single_source`, `test_text_extensions_are_a_subset` and `test_nothing_hardcodes_a_format_list_any_more`; the last only checks that those three files mention the name, so an extra hardcoded list would pass. README.md and DOCUMENTATION.md list the formats in prose by hand.

## Loaders

Every loader returns `List[Document]` with `source` (file name), `type` and `file_path` (the stored path) in the metadata and raises `FileNotFoundError` for a missing file. `load_docx` and `load_txt` raise `ValueError` when no text comes out; `load_pdf` only when the PDF has no pages.

**PDF** (`load_pdf`, `PyPDFLoader`)
- One Document per page, empty pages included; adds `page` (1-based), `total_pages`, `type="pdf"`. `GET /documents` shows `total_pages` as `pages`, so `pages` is null for every other format. Answers cite the file, not the page: `RAGChain._collect_sources` does not pass `page` on.
- No OCR: an image-only PDF loads as empty pages that yield no chunks, so the upload answers 400 `Document contains no extractable text`. A corrupt file raises pypdf's `PdfStreamError`.
- Tables arrive as space-separated words, never as a table.

**DOCX** (`load_docx`, python-docx)
- Every non-empty paragraph first, then every row of every top-level table as its cells joined with " | ", all blocks joined with blank lines. Tables therefore land after all the prose, away from their headings.
- In each row, empty cells are dropped and a cell equal to the previous kept cell is dropped. That collapses merged cells (python-docx repeats them once per spanned column) but also real repeated values, so later values shift columns: cells Engineer, 28, 28, 10 become `Engineer | 28 | 10`. A cell with several paragraphs puts a newline inside its row. Table placement and cell collapsing are known defects (KNOWN_ISSUES.md, section Document ingestion).
- Not read: headers, footers, footnotes, nested tables.
- Adds `type="docx"`, `char_count`, `paragraphs` (all of `document.paragraphs`, empty ones included) and `tables` (top-level tables).
- A corrupt file raises `PackageNotFoundError("Package not found at ...")` although the file exists.

**TXT and Markdown** (`load_txt`)
- Decodes the bytes as `utf-8-sig` (a BOM is stripped, `encoding="utf-8"` recorded); otherwise takes `charset_normalizer.from_bytes(raw).best()` and records its name (`cp1251`, `koi8_r`, `utf_16`, ...); no result raises `ValueError`.
- Detection is statistical: cp1251 or KOI8-R sentences and UTF-16 with a BOM decode correctly, but a very short legacy-encoded file can be mis-decoded without an error (cp1251 "Привет" comes out as `big5` mojibake and indexes with 200). Check `encoding`; the fix is to save the file as UTF-8.
- Markdown is kept verbatim: headings, lists and pipe tables stay as text, which is what lets the table-aware splitter see its tables.
- Adds `type` = the real lowercase extension (`txt`, `md`, `markdown`), `encoding`, `char_count`.

`.doc` and anything else: `load_document` raises re-save advice for `.doc` and `Unsupported file format` otherwise, but through the API the extension gate answers first with its generic list (see KNOWN_ISSUES.md, section Document ingestion).

## Chunking

**TextChunker** is built once in `lifespan` from `CHUNK_SIZE`, `CHUNK_OVERLAP` and `TABLE_AWARE_CHUNKING` (defaults 1000, 200, `True`); its constructor defaults only matter for direct construction.
- `RecursiveCharacterTextSplitter` with separators `"\n\n"`, `"\n"`, `". "`, `" "`, `""`, `keep_separator=True` and `length_function=len`: sizes are characters, not tokens. Text without spaces falls through to cuts between characters.
- Plain path (`table_aware=False`): langchain's `split_documents`, then every chunk gets `chunk_id`, `chunk_size` (its length), `total_chunks`, `contains_table=False`, `is_table=False`.
- Table-aware path (the default): `TableAwareSplitter.split_documents` splits each Document, copies its metadata and stamps the same five keys. On both paths `chunk_id` and `total_chunks` count the whole upload (all pages of a PDF), not each page.
- `CHUNK_OVERLAP` applies to prose only; `TableAwareSplitter` stores `chunk_overlap` and never reads it.
- `Settings._chunk_overlap_fits_in_chunk` refuses `CHUNK_OVERLAP >= CHUNK_SIZE` at startup. langchain alone accepts equal values (`TextChunker(chunk_size=100, chunk_overlap=100)` constructs), so keep the validator.

**TableAwareSplitter.split_text**
- Detects HTML `<table ...>...</table>` blocks and Markdown pipe tables: a row line (three or more `|`-separated parts, or a line starting or ending with `|`) directly followed by a separator line (`is_markdown_separator_row`), then consecutive row lines. A blank line ends the table.
- No table found: the whole text goes to the plain splitter, so the chunks equal the plain path's.
- Tables found: prose is split by the plain splitter and packed greedily (joined with blank lines) up to `chunk_size`. A table that fits in `chunk_size` is never split and shares a chunk with its neighbours when there is room. A larger table flushes the current chunk and is split by rows, with the header and separator lines repeated on every piece (row budget `max(chunk_size - header length, 100)`).
- Flags: `contains_table` marks every chunk holding a table; `is_table` only a chunk that is nothing but table. Both are stored in Chroma; no retrieval code reads them.
- Sizes can overshoot: one row longer than the budget becomes its own oversize chunk; a large HTML table is split by `<tr>` with its `<thead>` (or first `<tr>`) repeated, but with a `<thead>` the header row appears twice in the first piece; an HTML table without `<tr>` is sliced every `chunk_size` characters, mid-tag.
- Separator limitation: `MD_TABLE_SEPARATOR_PATTERN` rejects a separator with a space after an inner pipe once there are three or more columns (`| --- | --- | --- |`, the GitHub and Prettier style). `|---|---|---|` and the two-column `| --- | --- |` pass. Such a table is chunked as prose and only its first piece carries the header (see KNOWN_ISSUES.md, section Document ingestion).
- DOCX and PDF tables never qualify: DOCX rows have no separator line and are separated by blank lines (a two-cell row is not even a row line), and PDF text arrives as space-separated words, so `TABLE_AWARE_CHUNKING` changes nothing for them (see KNOWN_ISSUES.md, section Document ingestion).

## Metadata tags and question filters

Rules, regexes, the alias table and worked examples (English and Russian) are in [references/metadata-extraction.md](references/metadata-extraction.md); read it before changing `metadata_extractor.py` or explaining why a question was filtered.

- Document side, once per upload: `extract_document_metadata(safe_name, first chunk text)` returns `company`, `year` (int), `quarter` and `doc_type` when found. Company comes from the alias table (FinanceBench's S&P 500 companies) anywhere in the filename, else from the first filename token if it is alphabetic and 2+ letters long, Cyrillic included, so generic uploads get invented companies (`policy.docx` -> POLICY). Year comes from the filename, else from an FY mention in the first 600 characters of the first chunk, else from any 1900-2039 in its first 400. Doc type is 10-K, 10-Q, 8-K or EARNINGS. The convention it was tuned for: `COMPANY_YEAR[Qn]_DOCTYPE[_pN].ext`.
- Question side: `extract_query_metadata(question)` returns at most one company, a year (exact; `{"$in": [y-1, y]}` after forecast words; `$in` for 2-5 years; nothing for 6+), a quarter and a doc type (including `ANNUAL REPORT`, which no document carries). `build_chroma_filter(user_id, filters)` returns `{"user_id": u}` or `{"$and": [{"user_id": u}, {key: value}, ...]}`, drops any `user_id` key and `None` values, and raises `ValueError` on an empty user_id.
- `METADATA_FILTERING_ENABLED` (default `True`) only switches question-time detection in `RAGChain._prepare`. Upload tagging always runs, and an explicit `metadata_filter` in the query body is applied either way.
- On general corpora the defaults surprise: any number from 1900 to 2039 in a question ("2000 dollars") becomes a year filter, words like block, square, meta, apple or alphabet become company filters, and uploads take years from dates or phone numbers in their opening lines. When the filter matches a falsely tagged document, the tenant-wide fallback (zero hits only) never runs and the document that answers is not retrieved. For non-financial deployments recommend `METADATA_FILTERING_ENABLED=false` (see KNOWN_ISSUES.md, section Retrieval and search).
- Tags are fixed at upload: changed rules reach only new uploads. Delete the document and upload it again; re-sending identical bytes is a dedup no-op.

## Upload errors on this path

| Status and `detail` | Cause |
| --- | --- |
| 400 `Invalid filename` | `safe_filename`: empty, a dotfile, or only `.`, spaces and `_` |
| 400 `Unsupported file type. Supported: PDF, DOCX, TXT, MD, MARKDOWN` | extension not in `SUPPORTED_EXTENSIONS`: `.doc`, a compound name like `x.txt.zip`, or an extension longer than `MAX_EXT_LEN` (10, dot included) that `safe_filename` cut short |
| 400 `File too large. Maximum allowed size is <size>.` | over `MAX_FILE_SIZE` (default 30 MB) |
| 400 `Uploaded file is empty` | zero bytes |
| 400 `Could not read the document. It may be corrupted or empty.` | anything `load_document` or `split_documents` raised: a corrupt PDF or DOCX, a blank TXT, MD or DOCX, undecodable text, an extension in the list without a loader branch. Logged as `Failed to parse uploaded document` with the traceback |
| 400 `Document contains no extractable text` | the loader worked but there are no chunks (image-only PDF) |
| 503 `Vector store unavailable` | the Chroma or BM25 write raised (logged as `Failed to index document`): a metadata value that is not str, int, float or bool, more chunks than one Chroma batch, an embedding error. A failed dedup or quota lookup gives the same detail |
| 503 `Storage unavailable` | the file could not be written |

Every handled failure after the write deletes the stored file, so a refused upload leaves nothing on disk. Pinned by `tests/test_upload_errors.py` and `tests/test_docx_markdown.py`. `extract_document_metadata` runs outside those try blocks: an exception in it becomes a 500 that leaves the file behind, so keep it free of exceptions.

## Very large documents

`MAX_FILE_SIZE` limits the upload's bytes, not its text. `langchain_chroma` embeds every chunk (100 texts per embedding call) and then sends a single upsert, and ChromaDB refuses a batch larger than its client's `max_batch_size` (the SQLite build's variable limit divided by 6: 5461 with SQLite's default limit; some distribution builds allow more). A document that splits into more chunks (roughly 4.5 MB of extracted text at 1000/200) fails with 503 `Vector store unavailable` after all its embeddings were paid for; nothing is stored, the file is deleted, and a retry fails the same way. Workaround: split the document; the fix direction is in KNOWN_ISSUES.md, section Document ingestion.

## Recipes

### Add a file format
1. Plain text whose syntax should stay: add the extension to `TEXT_EXTENSIONS` (it is part of `SUPPORTED_EXTENSIONS`) and `load_txt` handles it with charset detection and `type=<extension>`; then only the length rule in step 5 and steps 7-8 apply. Anything else needs a loader, steps 2-8.
2. A new library: an exact `name==version` pin in `requirements.txt`, then regenerate both lock files (rag-development-guide); `tests/test_lockfile.py` checks them.
3. Write `@staticmethod load_<fmt>(file_path: str) -> List[Document]` in `DocumentLoader`, modelled on `load_docx`: wrap the path in `Path`; `FileNotFoundError` when missing; `ValueError` when the text is blank; metadata `source=file_path.name`, `type="<ext without dot>"`, `file_path=str(file_path)`, `char_count`, plus scalar extras only. Paged formats: one Document per page with `page` (1-based) and `total_pages`.
4. Add its branch to `DocumentLoader.load_document` before the final `raise`.
5. Add the extension to `SUPPORTED_EXTENSIONS`; that alone opens the API gate, the web UI picker and label, and the bot. Keep it at most 10 characters including the dot, word characters only (`safe_filename`).
6. Tables that should be table-aware: emit each one where it occurs as a contiguous Markdown pipe table with a compact separator (`|---|---|---|`) and rows on consecutive lines, or as `<table>` HTML.
7. Tests with fixtures built inside the test (offline, no binary files), modelled on `tests/test_docx_markdown.py`: it loads; metadata; blank -> `ValueError`; corrupt -> raises; missing -> `FileNotFoundError`; upload through the `api` fixture -> 200 with `chunks >= 1` and the `type` in `GET /documents`; corrupt upload -> 400 with nothing left in `api.upload_dir`. If the new extension is one the tests use as the unsupported example (`tests/test_upload_errors.py` `test_unsupported_extensions_are_rejected`, `.xlsx` in `tests/test_docx_markdown.py`), pick another example there.
8. Update the format lists in README.md and DOCUMENTATION.md (rag-config-and-docs).

### Add a company alias or a doc type
1. Alias: add uppercase names under the canonical key in `COMPANY_ALIASES`; the key is the value stored in Chroma and the one explicit filters must send. Matching sees the text with everything outside `[A-Za-z0-9&]` turned into spaces, so write `COCA COLA`, not `COCA-COLA`, and Cyrillic aliases can never match.
2. Avoid short tickers and common words; they turn ordinary questions into filters. Companies are tried in order of their longest alias, the first hit wins, and only one company is returned.
3. Doc type: change `DOC_TYPE_PATTERN` (questions) and both regexes inside `extract_document_metadata` (filename tokens, first chunk), each with its own copy of the normalisation, so stored and detected values are the same string (`10-K`). The web UI's Document Type options are a separate list (rag-clients).
4. A new tag: return it from `extract_document_metadata` as str, int, float or bool, never under the keys `user_id`, `source` or `file_hash` (it would overwrite them), and without raising (an exception there is a 500). A question-side filter must emit exactly the stored type (Chroma equality is type-strict: 2020 is not "2020") and stick to equality or `{"$in": [...]}`, the only operators the BM25 index evaluates. To show the tag in answer sources, add it to the key tuple in `RAGChain._collect_sources` (rag-retrieval-pipeline).
5. Tests in `tests/test_metadata_filtering.py`: a filename in `test_extract_document_metadata_from_filenames`, a question in `test_extract_query_metadata_from_questions`, and a negative case (a generic question must still give `{}`).
6. Stored chunks keep their old tags until re-uploaded. FinanceBench runs in `evaluation/langfuse_eval.py` filter with `normalize_company`, so alias changes move benchmark numbers (rag-evaluation).

### Change chunking defaults
1. The real defaults are `chunk_size`, `chunk_overlap` and `table_aware_chunking` in `app/config.py` `Settings`; update `.env.template` and the DOCUMENTATION.md Configuration rows with them (rag-config-and-docs).
2. Mind the DOCUMENTATION.md chunking line: `tests/test_docs_consistency.py` `test_docs_state_the_real_chunking_defaults` joins every line containing both "hunking" and "haracter", requires the current size and overlap in it, and forbids "500" and "50 char". A line like "chunking keeps tables up to 1500 characters" fails, and a default containing 500 (500, 1500, 2500) cannot pass without changing the test.
3. Keep the overlap validator, the same metadata keys on both paths, upload-wide numbering and deterministic output: chunk ids are positional and Chroma overwrites an existing id.
4. Existing documents keep their chunks; only new uploads use new sizes. Re-index by deleting and re-uploading (identical bytes are skipped). Backup manifests record `chunk_size` and `chunk_overlap`; restore prints them but never refuses an archive over a chunking difference.
5. Measure before and after (rag-evaluation): `tests/test_retrieval_quality.py` indexes whole documents and cannot see a chunking regression.
6. Run the chunking tests listed under Verify, plus `tests/test_api.py::test_chunk_size_setting_is_wired`. `tests/test_config.py` `test_settings_are_wired_into_components` checks size and overlap but not `table_aware`.

### Debug a document that indexes badly
1. Map the status and `detail` with the table above; the traceback is in the backend log (`Failed to parse uploaded document` or `Failed to index document`) next to the request id.
2. Reproduce offline on a copy of the file (the stored copy is `UPLOAD_DIR/<user_id>/<file_hash>_<name>`), keeping its file name because the tags come from it (a 16-hex `<file_hash>_` prefix is ignored). Run from the repository root with the project's Python and the deployment's chunk settings; no API key, network or ChromaDB is needed:

```bash
FILE=<copy-of-the-file> python - <<'EOF'
import os
from app.rag.document_loader import DocumentLoader
from app.rag.metadata_extractor import extract_document_metadata
from app.rag.text_splitter import TextChunker

path = os.environ["FILE"]
docs = DocumentLoader.load_document(path)
print(len(docs), "document(s);", docs[0].metadata)
# Keyword arguments: the third positional parameter is length_function.
chunks = TextChunker(chunk_size=1000, chunk_overlap=200, table_aware=True).split_documents(docs)
for c in chunks:
    m = c.metadata
    print(m["chunk_id"], m.get("page"), m["chunk_size"], m["contains_table"], m["is_table"],
          repr(c.page_content[:150]))
print("tags:", extract_document_metadata(os.path.basename(path), chunks[0].page_content if chunks else None))
EOF
```

3. Read the output: empty text on every PDF page is a scanned PDF; DOCX tables at the end with shifted cells are the loader's behaviour above; strange characters with an `encoding` other than `utf-8` are a mis-detected legacy encoding; `contains_table=False` on a Markdown table means the splitter did not see it (a spaced separator, no separator row, or a blank line inside the table); a chunk far over `chunk_size` is an oversize table row; a wrong `company` or `year` tag explains why filtered questions miss or wrongly hit this file.
4. Question side: run the one-liner in [references/metadata-extraction.md](references/metadata-extraction.md) with the user's question, and look for `Metadata filter ... yielded 0 results; falling back to tenant-wide search` in the backend log.
5. After fixing the file or the code, `DELETE /documents/{file_hash}` and upload again. `GET /documents?user_id=<id>` shows `type`, `pages` (PDF only) and the chunk count. Never open the live ChromaDB directory from a second process: the backend is its only writer.

## Invariants

- `SUPPORTED_EXTENSIONS` is the only format list, imported by the gate and both clients; copies are how a format gets half-added.
- Every extension in it has a branch in `load_document`; otherwise its uploads fail with the misleading "corrupted or empty" 400.
- Loaders raise `FileNotFoundError` for a missing file and `ValueError` for no text (`load_pdf` excepted: blank pages end as the no-extractable-text 400), and set `source`, `type` and `file_path`. Metadata values are str, int, float or bool only: ChromaDB rejects `None`, lists and dicts, and the upload becomes a 503.
- `_store_and_index` sets `source`, `file_hash` and `user_id` and then applies the tags; the extractor never returns those keys, which carry tenant scoping, dedup, revision replacement and deletes.
- Both chunking paths emit `chunk_id`, `chunk_size`, `total_chunks`, `contains_table` and `is_table`, numbered across the whole upload, and are deterministic for the same bytes.
- A refused upload leaves no file behind.
- Retrieval filters come from `build_chroma_filter` (top-level `user_id`, or `$and` starting with it). `EmbeddingsManager._owned` serves only EmbeddingsManager's own lookups and deletes; `RAGChain` and `HybridRetriever` crash on its `{"user_id": {"$eq": owner}}` shape.
- `CHUNK_OVERLAP` stays below `CHUNK_SIZE`, enforced at startup.

## Pitfalls and known issues

- Markdown tables with a spaced separator and 3+ columns are chunked as prose (Chunking above). See KNOWN_ISSUES.md, section Document ingestion.
- Word and PDF tables never get table-aware treatment, and DOCX moves tables to the end and collapses equal neighbouring cells (Loaders above; step 6 of "Add a file format" says what a table-aware loader emits). See KNOWN_ISSUES.md, section Document ingestion, which also gives the fix direction for `load_docx`.
- The `.doc` advice never reaches users: the gate refuses first, and the handler's catch-all would replace a loader message anyway. KNOWN_ISSUES.md, section Document ingestion.
- Documents over one Chroma batch cannot be indexed (above). KNOWN_ISSUES.md, section Document ingestion.
- Questions containing "annual report" get the doc_type `ANNUAL REPORT`, which matches nothing, so the fallback also throws away the company and year detected in the same question. KNOWN_ISSUES.md, section Retrieval and search.
- While a metadata filter is active (detected or explicit), MMR (`MMR_LAMBDA` below 1.0) fails with 503 and hybrid search runs without its BM25 half. Both live in the query path (rag-retrieval-pipeline; KNOWN_ISSUES.md, section Retrieval and search).
- A hyphenated `10-K` in a filename is not tagged (it is split into "10" and "K"); only the first chunk's text can supply the doc type. KNOWN_ISSUES.md, section Document ingestion.
- A span of more than five years keeps only the years named: "between 1990 and 2010" filters to exactly 1990 and 2010. KNOWN_ISSUES.md, section Retrieval and search.
- Explicit `metadata_filter` values are neither normalised nor validated: send canonical keys (`COCACOLA`) and int years. KNOWN_ISSUES.md, section Retrieval and search.
- Dead code: `YEAR_PATTERN`, `KNOWN_COMPANIES` and the `available_companies` parameter; `DocumentLoader.get_document_info` and `TextChunker.get_chunk_statistics` serve only tests and the `__main__` demos.

## Verify

Run from the repository root with the project's Python (on WSL, see rag-development-guide for which one); the suite is offline:

```bash
python -m pytest tests/test_document_loader.py tests/test_text_splitter.py tests/test_table_splitter.py tests/test_docx_markdown.py tests/test_metadata_filtering.py tests/test_upload_errors.py -q
python -m pytest tests/test_config.py tests/test_api.py tests/test_docs_consistency.py tests/test_lockfile.py -q
ruff check app frontend clients evaluation scripts tests
```

## Related skills

- rag-retrieval-pipeline: what happens to the filter and the chunks at question time.
- rag-api-endpoints: the `/upload` handler around this pipeline, `GET /documents`, `DELETE /documents/{file_hash}`.
- rag-config-and-docs: adding or changing a setting, the DOCUMENTATION.md tables and the drift tests.
- rag-testing: the `api` fixture, `make_settings` and fake embeddings used by the ingestion tests.
- rag-evaluation: measuring a chunking or extractor change before and after.
- rag-clients: the upload widgets and the Metadata Filter panel.
