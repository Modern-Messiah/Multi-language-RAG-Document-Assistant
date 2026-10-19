# Streamlit web UI reference

Everything here is in `frontend/streamlit_app.py` unless another file is named.

Contents
1. Run model and page order
2. session_state keys
3. Uploads and failed_files
4. Documents, delete, clear and the quota line
5. Sidebar panels: own key, reranker, metadata filter
6. Asking: payload, headers and the SSE stream
7. Sources and ratings
8. Languages
9. CSS caveats
10. Testing with AppTest

## 1. Run model and page order

- `streamlit run frontend/streamlit_app.py` puts `frontend/` on sys.path, so the script
  inserts the repo root itself before importing `app` and `clients` (hence the
  `# noqa: E402` imports).
- Streamlit re-executes the whole script on every interaction. Module-level code runs every
  time: `load_dotenv()`, `API_URL`, `MAX_FILE_MB`, `FEEDBACK_ENABLED`, `HEADERS`, and the
  `GET /documents` call. Only `st.session_state` survives between runs; a page reload starts
  a new session and therefore a new tenant.
- `st.rerun()` redraws immediately. It follows a successful rating, delete and clear, because
  what was drawn earlier in the same run (rating buttons, the document list, the uploader) is
  stale and, for ratings, still clickable. `st.stop()` ends the run after a failed question,
  so nothing is appended.
- Sidebar, top to bottom: the uploader and stored upload errors; the "Answer on my own API
  key", "Cross-Encoder Reranker" and "Metadata Filter" expanders; "Your documents" with a
  delete button per document; "Settings" with the "Answer language" radio; "Clear all
  documents"; the Limits box. Main pane: title, a status line with the indexed count, a caption
  saying who answers, the transcript, the chat input.

## 2. session_state keys

| Key | Holds | Set | Cleared |
| --- | --- | --- | --- |
| `user_id` | the tenant, `web-<uuid4 hex>` | first run of a session | never (a reload is a new session) |
| `own_key`, `own_model`, `own_provider` | own-key panel widgets | seeded with setdefault ("", "", "openai") before the sidebar | by the user |
| `own_reranker`, `own_reranker_key`, `own_reranker_model` | reranker panel widgets (first option "none") | widgets | by the user |
| `filter_company`, `filter_year`, `filter_doc_type` | metadata-filter widgets | widgets | by the user |
| `uploader_key` | suffix of the uploader's widget key `uploader_<n>` | setdefault 0 | bumped by delete and clear, which empties the uploader |
| `indexed_files` | set of `(name, sha256[:16])` uploaded in this session | successful upload | popped by delete and clear |
| `failed_files` | dict `(name, sha256[:16])` to the reason shown | local refusal, request error, non-200 | popped by delete and clear |
| `transcript` | list of turns `{question, answer, sources, language, request_id}` | after each streamed answer | popped by clear |
| `feedback_given` | `feedback_<index>` to "up" or "down" | successful rating | never (known issue) |

Other widget keys: `del_<file_hash>` (delete buttons) and `feedback_<index>_up` /
`feedback_<index>_down` (rating buttons). The language radio has no key.

The own-key widgets are seeded before the sidebar draws because the main-pane caption and
`asking_headers()` read them too. Seed any new widget key that other code reads the same way.

## 3. Uploads and failed_files

1. The uploader label is `<formats> files (max <MAX_FILE_MB> MB per file)`; the accepted types
   come from `SUPPORTED_EXTENSIONS`; several files at once.
2. Each file held by the uploader is keyed `(file.name, sha256(content)[:16])` - content, not
   size, so two different files with the same name and length do not collide - and skipped if
   the key is in `indexed_files` or `failed_files`. Without that, every rerun would upload the
   held files again, and a refused file would be retried forever.
3. Refused locally into `failed_files`: larger than `MAX_FILE_MB * 1024 * 1024` ("too large
   (x MB); the limit is N MB"), or empty ("the file is empty").
4. `POST /upload` with the multipart field `file`, `params={"user_id": USER_ID}`,
   `headers=HEADERS`, timeout 120 s, inside a spinner. A `requests.RequestException` is
   recorded as "backend unreachable (...)"; a 200 adds the key to `indexed_files` and shows
   "already indexed" when the body says `duplicate`, else "indexed" (the `replaced` flag, set
   when an earlier revision of the file was replaced, is not shown); any other status records
   `backend_error(response)`.
5. Every entry of `failed_files` is shown as an error on every rerun until a delete or a clear
   pops the dict.

Known issue: transient failures (backend down, 503) are remembered too, so the same file is
not retried until a delete, a clear or a reload - see KNOWN_ISSUES.md, section Web UI. AppTest cannot drive the
uploader, so this flow is tested only through seeded state
(`test_deleting_a_document_lets_a_refused_file_be_retried`); `test_uploading_stays_on_the_operators_key`
passes without any upload being posted.

## 4. Documents, delete, clear and the quota line

- `GET /documents` with `params={"user_id": USER_ID}` (30 s) on every run. A 200 gives the
  `documents` list and `describe_quota(body.get("quota"))`; `.get` because an older backend
  sends no quota block. Another status shows a warning; a request error shows "Could not reach
  the backend".
- Each document shows its source in bold, "N chunks" (plus ", M pages" when the listing has a
  page count), and a delete button keyed
  `del_<file_hash>`: `DELETE /documents/{file_hash}` with the user_id param (30 s). On 200 it
  pops `indexed_files` and `failed_files`, bumps `uploader_key` and reruns; otherwise
  `st.error`.
- "Clear all documents": `POST /clear` with the user_id param (60 s). On 200 it pops
  `indexed_files`, `failed_files` and `transcript` (not `feedback_given`), bumps
  `uploader_key` and reruns.
- The quota line, from `app/humanize.py` `describe_quota`: "3 of 200 documents, 12 KB of 1 GB";
  a limit of 0 shows usage only; anything but a dict gives "". The Limits box shows it as "You
  hold ..." above "Up to N MB per file" and the supported formats.
- The same listing drives the main pane: "N documents indexed" or "No documents indexed yet",
  and the chat input, which is disabled ("Upload a document first") while the list is empty.

## 5. Sidebar panels

"Answer on my own API key" (expanded when a key is set; must stay the first expander):
- `own_key` password box; `own_provider` selectbox over `PROVIDER_NAMES` (`list(PROVIDERS)`);
  `own_model` text box whose placeholder comes from `provider_placeholders` (fallback gpt-4o).
  Provider and model are disabled until a key is typed.
- `asking_headers()`: with a non-empty key, X-Model-Key; plus X-Model when the model box is not
  empty; plus X-Model-Provider when the provider is set and is not "openai".
- Captions: "never stored" in the panel, and in the main pane either
  `Answers come from <model> at <provider>, on your key` or where to open the panel. Tests
  check both.
- The placeholders are hints, not defaults: with a key and an empty Model box no X-Model is
  sent, so the server's MODEL_NAME goes to the chosen provider, which usually refuses it.
- Because the placeholder changes with the provider, Streamlit 1.34 re-creates the Model widget
  on a provider switch and resets `own_model` to "" (a known issue, KNOWN_ISSUES.md, section Web
  UI). Any widget whose label, placeholder, help or options vary between reruns behaves the same
  way despite its fixed `key=`; test such flows by typing first and switching second.
- The picker offers every `PROVIDERS` entry even when ALLOWED_MODEL_PROVIDERS narrows them;
  the backend answers 400 for the others. A new provider needs a placeholder here
  (rag-llm-providers has the full recipe).

"Cross-Encoder Reranker":
- `own_reranker` from the literal list none, flashrank, cohere (what `app/byok.py`
  `wanted_reranker` accepts); `own_reranker_key` and `own_reranker_model` boxes appear only
  for cohere.
- Anything but "none" sends X-Reranker-Provider, plus X-Reranker-Key and X-Reranker-Model when
  filled, with or without an own key. "none" sends nothing, which means the server default.
- Known issues (KNOWN_ISSUES.md, section Models, keys and reranking): flashrank is not
  installed, so it changes nothing; cohere without a key spends the operator's
  RERANKER_API_KEY; the header turns reranking on even with RERANKER_ENABLED=false.

"Metadata Filter":
- `filter_company` becomes `company`, upper-cased, as one exact value; `filter_year` becomes
  `year` as an int, only when the text is all digits; `filter_doc_type` (All, 10-K, 10-Q, 8-K,
  EARNINGS) becomes `doc_type` unless All. The dict joins the body as `metadata_filter` only
  when it is not empty.
- The backend lays it over the filter it extracts from the question (when
  METADATA_FILTERING_ENABLED) and ANDs it with the tenant; a `user_id` key is ignored. Type the
  company the way the source tags show it (WALMART, COCACOLA) and a single year - see
  KNOWN_ISSUES.md, section Web UI. A filter with no hits silently falls back to all of the
  user's documents - section Retrieval and search; the semantics are in rag-retrieval-pipeline.

Neither the reranker nor the metadata-filter panel has a test; add AppTest cases when you
touch them (section 10).

## 6. Asking: payload, headers and the SSE stream

1. `st.chat_input` returns the question on the run after Enter; empty or blank text is ignored.
2. Body: `question`, `language` (the radio's value), `user_id`, `history` (the last
   `HISTORY_TURNS_SENT` = 6 transcript turns reduced to `{question, answer}`) and, when set,
   `metadata_filter`.
3. `requests.post` to `/query/stream` with `json=payload`, `headers=asking_headers()`,
   `stream=True` and `timeout=120`. A request error shows "Backend unavailable: ..." and
   stops; a non-200 shows `backend_error(response)` and stops. The backend primes the stream
   before answering, so a BYOK refusal, 429, 504 or 503 arrives as an ordinary status.
4. `tokens()` reads `response.iter_lines(decode_unicode=True)` inside `with response:`, keeps
   lines starting with "data: " and decodes the JSON: `token` yields its `text`; `sources` is
   collected; `error` collects its `detail` (default "The answer was cut short."); `done` is
   ignored.
5. `st.write_stream(tokens())` renders the tokens and returns their concatenation - "" when no
   token arrived.
6. Afterwards a collected error is shown as a warning, the sources render under the answer, and
   the turn `{question, answer, sources, language, request_id}` is appended, `request_id`
   taken from the X-Request-ID response header. The new turn's rating buttons are drawn in the
   same run so they belong to this answer.

Known issue: an empty answer is appended too, and every later question then fails with "String
should have at least 1 character" - see KNOWN_ISSUES.md, section Web UI.

## 7. Sources and ratings

- `render_sources(sources)`: an expander "N source(s)"; per source the filename in bold,
  " (relevance: x.xxx)" when `rerank_score` is not None, `key: value` tags when `metadata` is
  not empty, and the preview as a caption.
- `render_turn` replays each stored turn with `st.chat_message`, which follows the viewer's
  theme.
- `render_feedback(index, turn)`: nothing when FEEDBACK_ENABLED is off; a "Thanks - recorded."
  caption when `feedback_given` holds `feedback_<index>`; otherwise thumbs-up and thumbs-down
  buttons keyed `feedback_<index>_up` and `feedback_<index>_down`.
- `send_feedback(turn, rating, key)` posts to `/feedback` the rating, user_id, question,
  answer, sources (filenames only), request_id and language, with `client` set to "web",
  `HEADERS` and a 15 s timeout. A request error or a non-200 shows a warning and keeps the
  buttons; a 200 stores the rating and calls `st.rerun()`.
- Ratings are keyed by transcript position, which is stable across reruns but not across a
  clear (known issue in section 2).

## 8. Languages

- `LANG_FLAGS` maps each language to a flag; its keys must be exactly `SUPPORTED_LANGUAGES`,
  Auto included. `LANG_OPTIONS` maps labels `<name> <flag>` to names in `SUPPORTED_LANGUAGES`
  order; the "Answer language" radio starts at index 0 (Auto); the plain name is sent.
- A new language gets its rule in `app/rag/languages.py` (rag-retrieval-pipeline) and then a
  flag here, which is required: `test_frontend_has_a_flag_for_every_supported_language` fails
  without it, `test_frontend_flags_name_no_unknown_language` fails on a flag for a language that
  is not supported, and the source must mention `SUPPORTED_LANGUAGES`. Keep `LANG_FLAGS` a pure literal: `_module_literal` in
  `tests/test_language_consistency.py` evaluates it with `ast.literal_eval`. The bot needs no
  change.

## 9. CSS caveats

- One `st.markdown("<style>...</style>", unsafe_allow_html=True)` block: a 20 px base font for
  the page, inputs and sidebar, and below 768 px a 16 px font with full-width buttons and
  sidebar.
- The selectors target Streamlit's internal `data-testid` attributes, which change between
  versions; after a Streamlit upgrade (pinned as streamlit==1.34.0 in requirements.txt), check
  the page in a browser.
- The rule meant to hide the uploader's own "Limit 200MB per file" text uses
  `section[data-testid="stFileUploader"] small`. In the 1.34 bundle `stFileUploader` is a div
  and the dropzone is `section[data-testid="stFileUploaderDropzone"]`, so the rule most likely
  matches nothing (read from the bundled JS, not checked in a browser).
- Every `st.caption` is coloured #cbd5f5, about 1.5:1 contrast on the light theme; that covers
  the "never stored" note, source previews and rating acknowledgements.
- Answers use `st.chat_message` because hand-rolled HTML with fixed colours broke the light
  theme; prefer Streamlit elements to raw HTML.

## 10. Testing with AppTest

- Run pytest from the repo root: `APP = "frontend/streamlit_app.py"` in
  `tests/test_frontend_feedback.py` is resolved against the working directory first, then
  against the test file's folder, where it does not exist.
- Fixtures there: `fake_backend` replaces `requests.get`, `requests.post` and `requests.delete`
  on the requests module itself (AppTest re-executes the script, so patching an attribute of an
  imported copy would not reach it). `FakeRequests` records `posted` as (url, json) and
  `sent_headers`; `headers_for(suffix)` returns the last headers sent to a path; `/documents`
  answers with one document and a quota block; `/query/stream` answers with SSE lines (sources,
  token) and an X-Request-ID header. `app` is `AppTest.from_file(APP, default_timeout=30)`
  after one `run()`, with `.fake` attached.
- Helpers: `_ask(at, question)` sets the chat input and runs; `_press(at, label)` clicks a
  rating button and runs.
- Patterns: seed state with `app.session_state["<key>"] = value`, then `app.run()`; click
  other buttons by key, as in `app.button(key="del_" + "a" * 16).click().run()`; change an env
  var the script reads with `monkeypatch.setenv` before `AppTest.from_file`; give the backend
  another answer by monkeypatching `requests.post` inside the test
  (`test_a_rejected_rating_says_so_and_keeps_the_buttons`); always assert `not app.exception`,
  because AppTest reports an uncaught exception as an element instead of raising.
- Limits: Streamlit 1.34's AppTest has no file-uploader driver (seed `indexed_files` or
  `failed_files` instead); the script is never imported (that would render the page), so read
  literals with ast; AppTest reruns after every interaction, so a missing `st.rerun()` goes
  unnoticed.
- Not covered by tests: the upload request itself, the reranker and metadata-filter panels,
  and the SSE `error` event with an empty answer.
