# Telegram bot reference

Everything here is in `clients/telegram_bot.py` unless another file is named.

Contents
1. Startup and module constants
2. Handlers and the command contract
3. The /help block
4. Languages
5. Uploads and limits
6. Questions and answer rendering
7. Ratings
8. /model
9. user_data and its lifetime
10. Errors and concurrency
11. Testing the bot

## 1. Startup and module constants

- Run it as a module from the repo root, `python -m clients.telegram_bot`, so `app` and
  `clients` import; the Compose service `bot` runs the same command. The bot only polls
  Telegram and opens no port.
- At import: `load_dotenv()`, logging at INFO, then these constants. They are computed once,
  so an .env change needs a restart and tests patch them on the module.

| Constant | Value |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | from the environment |
| `BACKEND_URL`, `BACKEND_HEADERS` | `backend_url()`, `api_headers()` |
| `LANGUAGES` | `list(SUPPORTED_LANGUAGES)` |
| `QUERY_TIMEOUT`, `UPLOAD_TIMEOUT`, `CLEAR_TIMEOUT` | 120.0, 120.0 and 30.0 s; CLEAR_TIMEOUT also bounds /documents |
| `HISTORY_TURNS` | 6 |
| `SUPPORTED_LIST` | the formats from `SUPPORTED_EXTENSIONS`, upper-case, comma-separated |
| `FEEDBACK_ENABLED`, `FEEDBACK_TIMEOUT` | `feedback_enabled()`, 15.0 s |
| `FEEDBACK_PREFIX`, `RATEABLE_ANSWERS` | "fb", 20 |
| `PLACEHOLDER_TOKENS` | the .env.template placeholder tokens that `main()` refuses |
| `_CHUNK_BUDGET` | `MessageLimit.MAX_TEXT_LENGTH - 200`, i.e. 3896 |

- `main()` refuses a missing or placeholder token (logs an error and returns; under Compose the
  restart policy then loops the container), builds
  `ApplicationBuilder().token(...).concurrent_updates(True).build()`, registers the handlers
  and `on_error`, and calls `run_polling()`.

## 2. Handlers and the command contract

| Trigger | Handler | Backend call | Behaviour |
| --- | --- | --- | --- |
| /start | `start` | none | drops `user_data["history"]`, greets, shows the language keyboard |
| /help | `help_command` | none | sends `help_text` as plain text |
| /clear | `clear_command` | `POST /clear` (30 s) | on 200 also drops the history |
| /documents | `documents_command` | `GET /documents` (30 s) | `format_document_list(documents, quota)` as one HTML message |
| /model | `model_command` | none | own key and model, section 8 |
| a document | `handle_document` | `POST /upload` (120 s) | section 5 |
| text that is not a command | `handle_message` | `POST /query` (120 s) | a language name sets the language, anything else is a question |
| callback data starting `fb:` | `on_rating` | `POST /feedback` (15 s) | section 7 |

Every backend call passes `params={"user_id": str(update.effective_user.id)}` or puts the same
id in the JSON body. Registration in `main()`: one line per command,
`app.add_handler(CommandHandler("<name>", <handler>))`, then
`MessageHandler(filters.Document.ALL, handle_document)`,
`MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)` and a `CallbackQueryHandler`
for `on_rating` whose pattern is built from `FEEDBACK_PREFIX`. The callback handler is
registered even when FEEDBACK_ENABLED is off, so buttons sent before the switch are still
answered.

The docs contract in `tests/test_docs_consistency.py`:
- `BOT_COMMANDS` = {start, help, clear, documents, model} must equal the set the regex
  `CommandHandler\("(\w+)"` finds in the bot source, so register with a double-quoted literal
  name (single quotes or a variable are invisible to it).
- Every registered command must appear as `/<name>` somewhere in README.md, and every command
  except help inside the help_text block (section 3).
- `test_docs_do_not_advertise_nonexistent_bot_commands`, which also holds the `BOT_COMMANDS`
  check, scans README.md for any "/word" token, URLs included, and fails on upload or query;
  never name a command after those.
- README.md, DOCUMENTATION.md and docker-compose.yml must name `clients.telegram_bot` (the bot
  once lived in a top-level telegram directory that shadowed the python-telegram-bot package).

## 3. The /help block

```python
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = (
        "How to use me:\n"
        f"1. Attach a file ({SUPPORTED_LIST}) and I will index it.\n"
        ...
        "7. Use /start to reset conversation."
    )
    await update.message.reply_text(help_text)
```

`test_every_bot_command_is_documented` captures the block with the DOTALL regex
`help_text = \(\n(.*?)\n    \)`: `help_text = (` must end its line, and the block ends at the
first line that is exactly four spaces and `)`. Parentheses inside the strings are fine (an
earlier parser broke on them); renumber the lines when inserting one. It is sent with
`reply_text`, so it is plain text, not HTML.

## 4. Languages

- `get_language_keyboard()`: a `ReplyKeyboardMarkup` (`resize_keyboard=True`) of `LANGUAGES`
  in `SUPPORTED_LANGUAGES` order, three buttons per row. /start shows it.
- A message whose whole text equals a language name sets `user_data["language"]` and replies
  with a short HTML how-to; nothing goes to the backend. Questions use
  `user_data.get("language", AUTO_LANGUAGE)`. A user who types a bare language name as a
  question therefore switches language instead.
- New languages need no bot change; `test_bot_keyboard_offers_exactly_the_supported_languages`
  and `test_bot_keyboard_is_laid_out_three_per_row` in `tests/test_language_consistency.py`
  pin the derivation.

## 5. Uploads and limits

1. Extension: `file_name.lower().endswith(SUPPORTED_EXTENSIONS)`, where `file_name` falls back
   to "" because the Bot API may omit it; otherwise "Sorry, I only support these formats: ...".
2. Size: refused when `doc.file_size` exceeds `FileSizeLimit.FILESIZE_DOWNLOAD`, 20,000,000
   bytes (decimal), with "Telegram limits bots to 20 MB"; a bot cannot download more through
   getFile. MAX_FILE_SIZE is never read, so the effective limit is the smaller of 20 MB and the
   backend's MAX_FILE_SIZE, and a file between the two is downloaded, uploaded and refused with
   400 "File too large...".
3. A `Processing <name>...` status message and a typing action, then
   `context.bot.get_file(doc.file_id)`, `download_as_bytearray()` and `POST /upload` with the
   multipart field `file`, the user_id param and `BACKEND_HEADERS`.
4. The status message is edited: 200 gives "This document is already indexed." (duplicate) or
   "Document processed successfully!" (the `replaced` flag is not shown); 413 gives the
   backend's quota message plus "Use /documents to see what is indexed and /clear to remove it
   all." (the bot has no per-document delete); other statuses
   `Failed to process document: <error>`; an exception a generic error.
5. Only document messages reach this handler (`filters.Document.ALL`); photos are ignored.

## 6. Questions and answer rendering

1. A typing action, then `POST /query` with question, language, user_id and the last 6
   entries of `user_data["history"]`, sent with `asking_headers(context)`: `BACKEND_HEADERS`
   plus, when `own_key` is set, X-Model-Key, X-Model (if `own_model`) and X-Model-Provider (if
   `own_provider`).
2. On 200: `parts = split_for_telegram(answer)`; each part goes out with `reply_html` as a bold
   "Answer:" or "Answer (i/n):" heading plus `html.escape(part)`. The answer is escaped plain
   text, so markdown from the model shows literally.
3. The sources follow in their own message - a bold "Sources:" and one escaped filename per
   line, no previews - so a long answer cannot push them over the limit.
4. With FEEDBACK_ENABLED and an X-Request-ID response header, the exchange is remembered and the
   rating keyboard is attached to the last message only.
5. `{question, answer}` is appended to the history, trimmed to 6, only on 200. A non-200 shows
   `backend_error(response)`; an exception a generic error.

`split_for_telegram(text, budget=_CHUNK_BUDGET)` cuts the raw text so that every part, once
escaped and given its heading, fits one message: the 200 characters held back cover the
heading, and the budget is measured on `len(html.escape(part))` because escaping can quadruple
a character (`<` becomes `&lt;`). Text that fits is one part. Otherwise a binary search finds
the longest raw prefix whose escaped length fits, and the cut moves back to the last `"\n\n"`,
`"\n"`, `". "` or `" "` if that boundary lies past half the prefix; every step makes progress,
and nothing is lost or duplicated (`tests/test_telegram_splitting.py`). Escaping before
splitting could cut an entity in half, which is why the raw text is what gets divided.

## 7. Ratings

- `rating_keyboard(request_id)`: two inline buttons with callback data `fb:up:<id>` and
  `fb:down:<id>`. Telegram caps callback data at 64 bytes, which is why the button carries only
  the id: an id the backend mints is 16 hex characters (24 bytes in all). An X-Request-ID longer
  than 56 characters, which the backend keeps when a caller or proxy sends one, would not fit.
- `remember_for_rating(context, request_id, exchange)` stores the question, answer, source
  filenames and language in `user_data["rateable"]` and drops the oldest beyond
  `RATEABLE_ANSWERS` (20).
- `on_rating` splits the data into prefix, rating and id; unparseable data is just answered. An
  unknown id (a restart, an old answer, or another user's button) gets "That answer is too old
  to rate now." and the keyboard is removed. Otherwise it posts to `/feedback` the rating,
  user_id, question, answer, sources, request_id and language with `client` set to "telegram",
  using `BACKEND_HEADERS`. A failure or non-200 answers "Could not send that, sorry." and keeps
  the keyboard (the reason is logged); a 200 answers "Thanks!", removes the keyboard and
  forgets the entry.

## 8. /model

| Words after /model | Result |
| --- | --- |
| none | status: whether a key is held, provider and model (never the key), the usage lines, `PROVIDERS`, and a note that the key is not stored on the server |
| `reset` | forgets `own_key`, `own_model` and `own_provider` |
| `<model> <key>` | stores both and forgets any provider |
| `<provider> <model> <key>` | stores all three; a provider is recognised only as the first of three or more words, and only if it is a `PROVIDERS` name (any case) |
| one word | usage text, nothing stored, and the message is not deleted (known issue) |
| more words | words after the key are ignored |

After storing, the bot deletes the user's message (`update.message.delete()`); when that fails
(for example no rights in a group) the confirmation says in bold to delete it by hand. The
confirmation goes through `update.message.chat.send_message(..., parse_mode="HTML")`, because
the original message is gone, and names the model and provider, never the key. The key is not
checked here; the backend checks it on the next question and a refusal comes back as a 400
detail. Any `PROVIDERS` name is accepted, even one ALLOWED_MODEL_PROVIDERS excludes (the
backend then answers 400). /start and /clear do not forget the key.

Known issues (KNOWN_ISSUES.md, section Telegram bot): `/model <key>` leaves the key in the
chat; `/model <key> <model>` stores the key as the model and repeats it in the confirmation;
`/model <provider> <key>` stores the provider as the model and sends the key to the default
endpoint. The server side of these headers is in rag-llm-providers.

## 9. user_data and its lifetime

| Key | Holds | Set by | Cleared by |
| --- | --- | --- | --- |
| `language` | chosen answer language | a language-keyboard message | a restart only |
| `history` | last 6 `{question, answer}` | an answered question | /start, /clear (on 200) |
| `own_key`, `own_model`, `own_provider` | own key, model, provider | /model | /model reset |
| `rateable` | request id to exchange, newest 20 | an answered question with rating buttons | a sent rating (that entry), or eviction past 20 |

`context.user_data` belongs to the Telegram user, not the chat: it follows them into group
chats (the /model text says "this chat's memory", which is not quite right), and in a group a
rating button pressed by anyone but the asker finds nothing. No PTB persistence is configured,
so a restart forgets everything; adding persistence would write keys to disk. Under the dev
Compose override the code is bind-mounted but the bot does not reload, so restart the `bot`
service after a code change.

## 10. Errors and concurrency

- `backend_error(response)` logs "Backend rejected the bot's API key (HTTP 401) - check
  BACKEND_API_KEY" (or 403) at ERROR, then returns `error_from_response(response)` from
  `clients/backend.py`.
- Every handler that calls the backend catches `Exception`, logs it and replies with a generic
  error. `on_error` catches what escapes: it logs the traceback and replies "Something went
  wrong on my side. Please try again." through `update.effective_message`.
- `concurrent_updates(True)` handles updates concurrently, so one slow upload does not block
  other users; handlers must not assume an order between updates.
- Handlers read `update.message`, which is None for an edited message, yet `CommandHandler`'s
  default filter and `filters.TEXT` both match edits, so an edited command or question ends in
  `on_error`. Use `update.effective_message` in new handlers when edits should work.

## 11. Testing the bot

- pytest-asyncio is not installed: call handlers with
  `asyncio.run(bot.<handler>(update, context))`.
- Fakes: `update = SimpleNamespace(message=..., effective_user=SimpleNamespace(id=7))` and
  `context = SimpleNamespace(user_data={})`, plus `bot=SimpleNamespace(get_file=...)` for
  uploads. Messages need the async methods the handler calls (`reply_text`, `reply_html`,
  `delete`, `chat.send_message`, `chat.send_action`).
- Network: `monkeypatch.setattr(bot.httpx, "AsyncClient", FakeClient)`, where `FakeClient`
  accepts any constructor arguments, implements `__aenter__` and `__aexit__`, and has async
  `post` or `get` returning an object with `status_code`, `headers` and `json()`.
- Module constants: `monkeypatch.setattr(bot, "FEEDBACK_ENABLED", False)`; `setenv` has no
  effect after import.
- Pure functions (`split_for_telegram`, `format_document_list`, `rating_keyboard`,
  `asking_headers`) are tested directly.
- Copy from: `tests/test_byok.py` `_model_command` (/model) and
  `test_the_bot_actually_sends_the_headers_when_asked_a_question` (question wiring);
  `tests/test_feedback.py` `_press` (`on_rating`) and `_ask` (`handle_message` with keyboard
  checks); `tests/test_quotas.py` `test_the_bot_tells_a_telegram_user_how_to_make_room`
  (`handle_document`); `tests/test_sweep.py` `test_the_bot_shows_the_quota_when_listing`
  (`documents_command`).
- Importing `clients.telegram_bot` runs its `load_dotenv()`, which matters for the .env known
  issue (KNOWN_ISSUES.md, section Development and tests); `tests/test_telegram_splitting.py`
  imports it at collection time.
