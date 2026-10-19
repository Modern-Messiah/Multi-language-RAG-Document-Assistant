---
name: rag-clients
description: "Changes the two front ends - the Streamlit web UI (frontend/streamlit_app.py), the Telegram bot (clients/telegram_bot.py) and their shared helpers in clients/backend.py: session namespaces, uploads, streamed answers, sources, ratings, the key, reranker and filter panels, bot commands and /help, Telegram limits, and how to test both. Use when adding a bot command or UI control, changing what the clients send or show, or debugging client-side behaviour."
---

# Web UI and Telegram bot

The two front ends hold no data of their own. `frontend/streamlit_app.py` is one Streamlit
script that re-runs top to bottom on every interaction; each browser session is its own
tenant (`web-<uuid4 hex>`) and answers stream from `POST /query/stream`.
`clients/telegram_bot.py` is a python-telegram-bot polling bot; each Telegram user is a
tenant (`str(update.effective_user.id)`) and answers come from `POST /query`.
`clients/backend.py` holds what both need - environment, the `X-API-Key` header, error
wording, backend-owned constants - while the transport stays in each client (`requests` in
Streamlit, `httpx.AsyncClient` in the bot).

## Use this skill for / not for

Use it for: a new bot command or UI control; what the clients send, show or remember
(uploads, answers, sources, documents, ratings, errors, env vars); client bugs and tests.

Not for (see the sibling skill):
- routes, schemas and status codes behind the clients: rag-api-endpoints
- what the BYOK and reranker headers mean on the server, `PROVIDERS`, adding a provider: rag-llm-providers
- what metadata filters, reranking and answer languages do in retrieval: rag-retrieval-pipeline
- running the UI and bot locally or in Compose: rag-development-guide
- sweeping abandoned `web-` namespaces, the feedback log, tracing a request id: rag-operations
- shared fixtures and the socket guard: rag-testing; Settings fields and doc-drift rules: rag-config-and-docs

## Map

| File | Responsibility | Key symbols |
| --- | --- | --- |
| `clients/backend.py` | client env, auth header, error wording, re-exports | `backend_url`, `api_headers`, `max_file_bytes`, `max_file_mb`, `feedback_enabled`, `describe_error`, `error_from_response`, `OPERATOR_ERROR` |
| `frontend/streamlit_app.py` | the web UI | `asking_headers`, `HEADERS`, `USER_ID`, `LANG_FLAGS`, `LANG_OPTIONS`, `send_feedback`, `render_feedback`, `render_sources`, `tokens` |
| `clients/telegram_bot.py` | the bot | `main`, `help_command`, `model_command`, `documents_command`, `handle_document`, `handle_message`, `on_rating`, `on_error`, `split_for_telegram`, `format_document_list`, `asking_headers`, `backend_error` |
| `app/rag/languages.py` | answer languages | `SUPPORTED_LANGUAGES`, `AUTO_LANGUAGE` |
| `app/byok.py` | header names, provider table | `KEY_HEADER`, `MODEL_HEADER`, `PROVIDER_HEADER`, `RERANKER_PROVIDER_HEADER`, `RERANKER_KEY_HEADER`, `RERANKER_MODEL_HEADER`, `PROVIDERS` |
| `app/humanize.py` | the quota line | `describe_quota`, `human_size` |
| `app/rag/document_loader.py` | upload formats | `SUPPORTED_EXTENSIONS` |
| `app/models/schemas.py` | what the backend accepts | `QueryRequest`, `ChatTurn`, `FeedbackRequest` |

Read [references/streamlit.md](references/streamlit.md) before touching the web UI (state
keys, uploads, panels, streaming, ratings, CSS, AppTest) and
[references/telegram-bot.md](references/telegram-bot.md) before touching the bot (handlers,
/help, limits, splitting, ratings, /model, user_data, test fakes).

| Test file | Pins |
| --- | --- |
| `tests/test_clients_backend.py`, `tests/test_observability.py` | env helpers, every `describe_error` and `error_from_response` branch, when a request id is appended, the client MAX_FILE_SIZE default equal to the Settings default |
| `tests/test_frontend_feedback.py` | the real web script under Streamlit AppTest: ratings, history sent, `web-` identity, limits box, BYOK headers, provider picker, key panel first |
| `tests/test_telegram_splitting.py`, `tests/test_language_consistency.py` | `split_for_telegram`; the language re-export, the bot keyboard, `LANG_FLAGS` read with ast |
| `tests/test_docs_consistency.py` | `BOT_COMMANDS`, the /help block, README mentions, `CLIENT_ONLY_VARS` |
| `tests/test_byok.py`, `tests/test_feedback.py` | bot /model and the bot's `asking_headers`; `feedback_enabled`, the rating keyboard (64 bytes), `on_rating`, keyboard placement |
| `tests/test_quotas.py`, `tests/test_sweep.py`, `tests/test_docx_markdown.py` | the quota line, `format_document_list`, the 413 remedy in `handle_document`; both clients use `SUPPORTED_EXTENSIONS` |

## How it works

### clients/backend.py

The helpers read the client's own process environment at call time (`os.getenv`), not
Settings. Both clients call `load_dotenv()` first, which finds the repo-root .env and never
overrides a variable that is already set.

| Variable | Helper | Value and fallback | Read by |
| --- | --- | --- | --- |
| BACKEND_URL | `backend_url()` | unset or empty: `http://127.0.0.1:8000` (`DEFAULT_BACKEND_URL`); .env.template ships `http://localhost:8000`, Compose sets `http://backend:8000`; returned raw, a trailing slash is kept | both clients, `scripts/backup.py`, `scripts/restore.py`, `scripts/sweep.py` |
| BACKEND_API_KEY | `api_headers()` | `{"X-API-Key": value or ""}`; empty works only while backend auth is off | every request of both clients |
| MAX_FILE_SIZE | `max_file_bytes()`, `max_file_mb()` | non-integer, 0 or negative: 31457280 (`DEFAULT_MAX_FILE_SIZE`, the Settings default); `max_file_mb` floors to whole MiB | web UI only |
| FEEDBACK_ENABLED | `feedback_enabled()` | empty: True; 0, false, f, no, n, off (any case, stripped): False; anything else True | both clients (rating buttons) |
| TELEGRAM_BOT_TOKEN | bot module constant | missing or a .env.template placeholder: `main()` logs an error and returns | bot only |

The web script re-reads all of this on every rerun. The bot reads it once at import into
module constants (`BACKEND_URL`, `BACKEND_HEADERS`, `FEEDBACK_ENABLED`, `TELEGRAM_BOT_TOKEN`),
so it needs a restart, and its tests patch the module attribute
(`monkeypatch.setattr(bot, "FEEDBACK_ENABLED", False)`) rather than the environment.

Every user-facing backend failure goes through
`describe_error(status_code, detail, request_id)`, usually via `error_from_response(response)`:
1. 401 and 403 become `OPERATOR_ERROR` ("The assistant is not configured correctly. Please
   contact the operator."), never the backend's detail: a user cannot fix a rejected
   X-API-Key. The bot's `backend_error` also logs "check BACKEND_API_KEY" for the operator.
2. A list detail (pydantic 422) becomes its `msg` values joined with "; "; a non-empty string
   passes verbatim; a non-empty dict becomes `str(dict)`; anything else
   `Backend error (HTTP <status>)`.
3. `(request <id>)` is appended only for 401/403 and 5xx, the failures a user reports to
   someone; a 400 BYOK refusal, a 413 quota or a 422 already says what to change.
4. `error_from_response` is duck-typed for requests and httpx: a body whose `.json()` raises (a
   proxy's HTML 502, an empty 500) is tolerated, the X-Request-ID header wins over a body
   `request_id`, and headers are read with getattr.

Re-exported in `__all__`: the `X-Model-*` and `X-Reranker-*` header constants and `PROVIDERS`
from `app/byok.py`, `REQUEST_ID_HEADER` from `app/observability.py`, `AUTO_LANGUAGE` and
`SUPPORTED_LANGUAGES` from `app/rag/languages.py`, `describe_quota` and `human_size` from
`app/humanize.py`. Both clients import `SUPPORTED_EXTENSIONS` from `app/rag/document_loader.py`.

The clients never import `app.main`: it loads FastAPI, chromadb and the whole server into the
UI or bot process, and its lazy `app` attribute builds a complete application (Settings
validation, OPENAI_API_KEY required) on every access. Shared pieces live in light modules next
to their owner (`app/humanize.py` was split out of `app.main` for this) and are re-exported
here. Even so the clients need the full requirements set: `app.byok` imports openai and httpx,
`app.rag.document_loader` imports langchain and python-docx.

### The two clients side by side

| | Web UI | Bot |
| --- | --- | --- |
| Tenant | `web-<uuid4 hex>` in `st.session_state["user_id"]`, new on every reload | `str(update.effective_user.id)`, digits only |
| Question | `POST /query/stream` (SSE), 120 s | `POST /query`, 120 s |
| Body | question, language, user_id, last 6 history turns, optional metadata_filter | question, language (Auto until chosen), user_id, last 6 history turns |
| Own key | sidebar panel, kept in session_state | /model, kept in `user_data` |
| Reranker, metadata filter | sidebar expanders | not offered |
| Upload pre-checks | MAX_FILE_SIZE in whole MiB, empty file, already seen | extension, Telegram's 20,000,000-byte download cap |
| Documents | `GET /documents` on every rerun, delete per document, clear all | /documents as one HTML message, /clear |
| Answer | `st.write_stream` of the token events | split by `split_for_telegram`, escaped, "Answer (i/n)" parts |
| Sources | expander: name, rerank score, metadata tags, preview | filenames only, in their own message |
| Ratings | buttons keyed `feedback_<index>` (transcript position) | inline buttons with callback data `fb:<up or down>:<request_id>` |
| State lives | `st.session_state`, lost on reload | `user_data` per Telegram user, in memory, no persistence |

## Recipes

### Add a bot command

1. In `clients/telegram_bot.py`, write `async def <name>_command(update, context)` beside the
   others; the tenant is `str(update.effective_user.id)`. Call the backend inside `try:` with
   `async with httpx.AsyncClient(timeout=<CONSTANT>) as client:`, using a named module
   constant, and an `except Exception` that logs and replies with a generic error. Send
   `headers=BACKEND_HEADERS` (only questions use `asking_headers(context)`) and show
   `backend_error(response)` on a non-200.
2. Put formatting in a pure function, as `format_document_list` does, so it can be tested
   without Telegram. With `reply_html` or `parse_mode="HTML"`, `html.escape` every user- or
   document-derived string; keep output that can pass 4096 characters within the limit
   (`split_for_telegram` or several messages - /documents does not, see known issues).
3. Register it in `main()` beside the others with
   `app.add_handler(CommandHandler("<name>", <name>_command))`. The name must be a
   double-quoted literal, because `tests/test_docs_consistency.py` finds commands with the
   regex `CommandHandler\("(\w+)"`; lower-case, since PTB lower-cases it and rejects anything
   but 1-32 ASCII letters, digits or underscores; never name it upload or query (README test).
4. Add a numbered line naming `/<name>` to `help_text` in `help_command`. Keep the shape the
   doc test parses with `help_text = \(\n(.*?)\n    \)`: `help_text = (` ends its line and
   the closing `)` stands alone, indented exactly 4 spaces.
5. Add the name to `BOT_COMMANDS` in `tests/test_docs_consistency.py` (it must equal the
   registered set), and mention `/<name>` in README.md (the Telegram Bot list in Quick Start,
   checked by `test_every_bot_command_is_documented`) and in DOCUMENTATION.md, Usage.
6. Test it with `SimpleNamespace` fakes and `asyncio.run(...)` (pytest-asyncio is not
   installed), replacing `bot.httpx.AsyncClient` with an async-context-manager fake: copy
   `test_the_bot_tells_a_telegram_user_how_to_make_room` in `tests/test_quotas.py` or `_ask`
   in `tests/test_feedback.py`. Test the formatter directly; if the command handles a secret,
   assert it is never echoed.
7. Optional: register it with @BotFather (/setcommands); the code never calls
   `set_my_commands`, so the "/" menu does not update itself.

### Add a web UI control that maps to an API parameter

1. Check that the backend takes it. A body field must exist on `QueryRequest` in
   `app/models/schemas.py`: pydantic drops unknown keys silently, so a misspelt field does
   nothing and returns no 422. A header must be parsed in `app/byok.py` or `app/main.py`.
   Otherwise change the backend first (rag-api-endpoints, rag-llm-providers).
2. A header constant lives next to its parser. Import it in `clients/backend.py`, add it to
   `__all__`, and import it in `frontend/streamlit_app.py` from `clients.backend`; run ruff,
   because the reranker constants once shipped unimported (F821) and failed only when a user
   picked a reranker.
3. Add the widget inside `with st.sidebar:` with a unique `key=` (its value lands in
   `st.session_state`), in an expander after "Answer on my own API key", which a test requires
   to be the first. Fill options from backend constants. If code above the sidebar reads the
   key, seed it first with `st.session_state.setdefault`, as `own_key` is.
4. Send it with questions only, and only when set: a body field beside `metadata_filter` in the
   question block, a header in `asking_headers()`. Never add it to the `HEADERS` calls
   (upload, documents, delete, clear, feedback). Keep client validation cheap: a backend 400
   or 422 detail already reaches the user.
5. Decide on bot parity: its body is built in `handle_message`, its headers in its own
   `asking_headers(context)`; a setting there usually means a command (recipe above).
6. Test in `tests/test_frontend_feedback.py`: set `app.session_state["<key>"]`, call
   `app.run()` and `_ask(app)`, then check the bodies in `app.fake.posted` or
   `app.fake.headers_for("/query/stream")`; also check that the default sends nothing and
   that `app.exception` is empty.
7. Docs: DOCUMENTATION.md, Usage, plus the API reference if the API changed; README Key
   Features if the control is user-visible.

### Change client error wording

1. Edit only `clients/backend.py` (`OPERATOR_ERROR`, `_describe`, `describe_error`,
   `error_from_response`) and keep its rules. Both clients use it; per-client copies are how
   the wording drifted before (one relayed "Invalid or missing API key", the other raw
   pydantic lists).
2. If a message is wrong because of its status code, change the backend (rag-api-endpoints): a
   user-fixable failure must stay a 4xx other than 401/403 with a string detail, which is why
   caller-key refusals are 400.
3. Framing stays at the call sites. The bot prefixes a cross mark (plus "Failed to process
   document:" for uploads, "Failed to clear documents:" for /clear), adds the /documents and
   /clear hint to a 413 in `handle_document`, and for ratings only answers "Could not send
   that, sorry." while logging the reason. The web UI uses `st.error` for uploads (through `failed_files`), questions,
   delete and clear, and `st.warning` for the document list and ratings.
4. Run `tests/test_clients_backend.py`, `tests/test_observability.py`, `tests/test_quotas.py`
   (`test_a_413_reaches_the_user_verbatim`) and `tests/test_byok.py`
   (`test_a_caller_never_sees_the_operator_error_for_their_own_key`).

### Add an environment variable the clients read

1. Read it in `clients/backend.py` through a function that calls `os.getenv` at call time,
   parses defensively and falls back to the backend's default (like `max_file_bytes`); export it.
2. A mirror of a Settings field keeps the Settings default as its fallback, pinned by a test
   like `test_max_file_size_default_matches_the_backend`; the Settings side is
   rag-config-and-docs.
3. A client-only variable goes into `CLIENT_ONLY_VARS` in `tests/test_docs_consistency.py`
   before it gets a `NAME=` line in .env.template or a Configuration-table row; its documented
   default is not checked automatically, so compare it with the code by hand.

## Invariants

- Only question requests carry `X-Model-*` (and, from the web UI, `X-Reranker-*`) headers; the
  rest send only `X-API-Key` (`HEADERS`, `BACKEND_HEADERS`) - the collection is bound to the
  operator's embedding model, and a user's key should pay for the answer and nothing else.
- No `X-Model` or `X-Model-Provider` without `X-Model-Key` (the web UI also omits `openai`) -
  a model chosen without a key would be billed to the operator, so the backend answers 400.
- History turns are exactly `{question, answer}`, at most 6, stored only after a 200, dropped
  by /start, /clear and "Clear all documents" - the API is stateless and caps history at 20
  turns (question 4000 characters, answer 8000), and an error must not be replayed as an answer.
- Languages, providers and formats come from backend modules, never a client copy (existing
  literals: the web reranker and doc-type choices, `provider_placeholders`); `LANG_FLAGS` stays a
  pure literal keyed by exactly `SUPPORTED_LANGUAGES`, since tests read it with `ast.literal_eval`.
- Telegram text is split raw and escaped per part, sources go separately, the keyboard sits on
  the last message - escaping first could cut an entity; buttons on every part invite duplicates.
- Tenants: `web-` plus a uuid per browser session, digits for Telegram - the sweep script's
  default prefix `web-` can never reach a bot user, and a shared web tenant once let visitors clear
  each other's documents.
- The web UI asks `GET /documents` on every rerun instead of trusting session state, which
  knows nothing after a reload while the backend still holds the documents.
- /model never echoes the key and deletes its message or says it could not; the confirmation
  goes through `chat.send_message` because that message is gone.
- The bot keeps `on_error` and `concurrent_updates(True)` - otherwise an escaped exception gets
  no reply and one slow upload blocks every user.
- Rating buttons only when FEEDBACK_ENABLED and, in the bot, a request id; callback data within
  64 bytes; a rejected rating keeps its buttons. Keys stay in memory (no PTB persistence).
- The clients act for anonymous users with the operator's shared `X-API-Key`, so operator
  endpoints such as `POST /maintenance/sweep` are never wired into them.

## Pitfalls and known issues

Current behaviour that surprises (verified in the code; more in the references):
- A trailing slash in BACKEND_URL breaks every request: `backend_url()` keeps it, paths become
  `//upload`, and FastAPI answers 404, shown as "Not Found".
- The bot never reads MAX_FILE_SIZE, only Telegram's 20,000,000-byte download cap, so a file
  between the two is downloaded, uploaded and refused with 400 "File too large...". The web UI
  floors MAX_FILE_SIZE to whole MiB, so below 1 MiB it refuses every file; above 200 MB
  Streamlit's own `server.maxUploadSize` (default 200, flag `--server.maxUploadSize`) caps first.
- A Telegram message may hold 4096 characters, a question 4000 ("String should have at most
  4000 characters"). Client timeouts (120 s) are shorter than the backend's worst case
  (OPENAI_TIMEOUT 45.0 with OPENAI_MAX_RETRIES 2), so a client can give up on a late answer.
- The panels do less than they look: an empty Model box sends MODEL_NAME to the chosen
  provider; reranker "none" sends no header; Year must be all digits and Company is one exact
  upper-cased value - see KNOWN_ISSUES.md, section Web UI, and references/streamlit.md.
- Bot handlers read `update.message`, None for an edited message, while PTB's filters match
  edits, so an edit ends in `on_error`'s generic reply; `user_data` follows the user into group
  chats - see references/telegram-bot.md.

Known defects (KNOWN_ISSUES.md holds repro and workaround):
- An answer that ends before any token is stored as "", and every later question in that
  session fails with "String should have at least 1 character" until "Clear all documents" or
  a reload - see KNOWN_ISSUES.md, section Web UI.
- "Clear all documents" keeps `feedback_given`, so answers at positions rated before cannot be
  rated - see KNOWN_ISSUES.md, section Web UI.
- Every upload failure, transient ones included, stays in `failed_files`, so that file is not
  retried until a delete, a clear or a reload - see KNOWN_ISSUES.md, section Web UI.
- Switching the provider clears a model already typed in the key panel: the Model box's
  placeholder follows the provider, and in Streamlit 1.34 a widget whose parameters change is
  re-created even with a fixed `key=`, resetting its value. Keep a keyed widget's label,
  placeholder, help and options stable across reruns - see KNOWN_ISSUES.md, section Web UI.
- Bot: an empty /query answer poisons the history the same way until /start or /clear - see
  KNOWN_ISSUES.md, section Telegram bot.
- Bot: /documents is one unsplit message and fails past roughly 55-110 documents, depending on
  file-name length - see KNOWN_ISSUES.md, section Telegram bot.
- Bot: `/model <key>` gets the usage text and leaves the key in the chat; `/model <key> <model>`
  echoes the key in the confirmation; `/model <provider> <key>` stores the provider as the model
  and sends the key to the default endpoint - see KNOWN_ISSUES.md, section Telegram bot.
- A .env in the repo root or a parent folder leaks into pytest through the clients'
  `load_dotenv()`; move it aside while testing - see KNOWN_ISSUES.md, section Development and tests.
- Elsewhere in KNOWN_ISSUES.md: flashrank is not installed and cohere without a key spends the
  operator's RERANKER_API_KEY (Models, keys and reranking); a filter with no hits silently
  searches everything (Retrieval and search); streamed answers record no token usage (Tracing
  and observability).

## Verify

From the repo root (the AppTest script path is resolved against the working directory):

```
python -m pytest tests/test_clients_backend.py tests/test_telegram_splitting.py tests/test_frontend_feedback.py tests/test_language_consistency.py tests/test_docs_consistency.py -q
python -m pytest tests/test_byok.py tests/test_feedback.py tests/test_quotas.py tests/test_sweep.py tests/test_observability.py tests/test_docx_markdown.py -q
ruff check app frontend clients evaluation scripts tests
```

Both sets are offline and take seconds; run the whole suite before committing (rag-testing;
from WSL with a Linux venv, `~/.venvs/<name>/bin/python -m pytest ...`). Uploads and Telegram
itself need a manual check against a running backend: `streamlit run frontend/streamlit_app.py`,
or `python -m clients.telegram_bot` with a real token, which contacts Telegram - ask first.

## Related skills

rag-api-endpoints (routes and status codes the clients call), rag-llm-providers (BYOK and
reranker headers, `PROVIDERS`), rag-retrieval-pipeline (filters, reranking, languages),
rag-testing (fixtures, running the suite), rag-config-and-docs (Settings and the doc-drift
contract), rag-operations (`web-` namespaces, feedback log, request-id tracing),
rag-development-guide (running the clients), rag-change-review (checklist before merging).
