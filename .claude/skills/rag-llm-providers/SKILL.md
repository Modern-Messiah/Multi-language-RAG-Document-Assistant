---
name: rag-llm-providers
description: Handles which model answers - the server LLM endpoint (OPENAI_BASE_URL, LLM_BASE_URL, LLM_API_KEY and the DeepSeek switch), the bring-your-own-key headers X-Model-Key, X-Model and X-Model-Provider, the server-owned PROVIDERS table in app/byok.py, model-name normalisation, the X-Reranker-* headers, key lifecycle rules and how refused keys are reported. Use when adding or changing an LLM provider or model alias (a reranker provider is rag-retrieval-pipeline), debugging BYOK errors, or pointing the server at another OpenAI-compatible endpoint.
---

# LLM providers and bring-your-own-key

Every generated answer is one OpenAI SDK chat-completions call (plus one condensing call for a follow-up),
made on the deployment's own client, built once at startup, or on a per-request client built from
the caller's `X-Model-Key`, whose base URL comes from `PROVIDERS` in `app/byok.py` or from
OPENAI_BASE_URL, never from the request. Embeddings always use the operator's key, OPENAI_BASE_URL
and EMBEDDING_MODEL, because the ChromaDB collection is bound to one embedding model. This skill
covers how the choice is made, validated, closed and reported, and how to extend it without
opening an SSRF hole or leaking a key.

## Use this skill for / not for

Use it for: adding or changing a provider or a model alias; a 400 caused by the `X-Model-*` or
`X-Reranker-*` headers, or "my own key does not work"; pointing the deployment's answers at
DeepSeek, a gateway (LiteLLM, OpenRouter, vLLM, Azure) or another OpenAI-compatible endpoint;
changing how a provider refusal becomes a status code.

Not for (see the sibling skill):
- routes, schemas, SSE framing, status conventions in general: rag-api-endpoints
- reranking inside retrieval, reranker models and aliases, a new reranker provider, the condensing
  prompt: rag-retrieval-pipeline
- the rest of the web UI and the bot (uploads, ratings, other commands): rag-clients
- adding a Settings field end to end, the doc-drift tests: rag-config-and-docs
- Langfuse setup, following a request id through the logs: rag-operations
- fixtures and fakes (`api`, `make_settings`, `FakeChatClient`): rag-testing
- `python -m evaluation.langfuse_eval` runs with `--provider`, `--model`, `--model-key`: rag-evaluation
- where this sits in the system: rag-architecture; the BYOK checks in a review: rag-change-review

## Map

| File | Responsibility | Key symbols |
| --- | --- | --- |
| `app/byok.py` | header names, provider table, header validation, per-request client, refusal wording, closing, reranker headers | `KEY_HEADER`, `MODEL_HEADER`, `PROVIDER_HEADER`, `PROVIDERS`, `_KEY`, `_MODEL`, `BringYourOwnKeyError`, `allowed_providers`, `normalize_model_name`, `wanted`, `client_for`, `describe_upstream_refusal`, `close_quietly`, `wanted_reranker` |
| `app/main.py` | the server's client and model at startup; BYOK parsing, error mapping and closing per request | `lifespan`, `_caller_model`, `_caller_reranker`, `query_rag`, `query_rag_stream` and its `events()` |
| `app/rag/chain.py` | `client or self.client` and `model or self.model` for condensing and answering; keeps neither | `RAGChain.__init__`, `_condense`, `_prepare`, `ask`, `ask_stream` |
| `app/rag/embeddings.py` | the operator-only embedding client | `OpenAIEmbeddingFunction`, `EmbeddingsManager` |
| `app/rag/reranker.py` | reranker factory with per-request overrides | `get_reranker`, `NoOpReranker`, `FlashRankReranker`, `CohereReranker` |
| `clients/backend.py` | re-exports the header names and `PROVIDERS` to both clients; one-line error text | `describe_error`, `error_from_response`, `OPERATOR_ERROR` |
| `frontend/streamlit_app.py` | sidebar key panel and reranker expander | `asking_headers`, `PROVIDER_NAMES`, the `provider_placeholders` dict |
| `clients/telegram_bot.py` | the `/model` command | `model_command`, `asking_headers` |

Tests: `tests/test_byok.py` (almost everything here, the bot's `/model` included),
`tests/test_resilience.py` (transport settings, operator-key 429/504/503), `tests/test_reranker.py`
(reranker headers), `tests/test_streaming.py` (priming), `tests/test_frontend_feedback.py` (web UI
headers). [references/providers.md](references/providers.md) holds every caller-facing message,
the web panels, the bot's `/model` parse table and a test index - read it when matching a user's
error text, writing a test, or touching a client.

## How it works

### Who answers

| Request | Chat endpoint | Chat key | Model sent |
| --- | --- | --- | --- |
| no `X-Model-Key` | LLM_BASE_URL, else OPENAI_BASE_URL, else api.openai.com | LLM_API_KEY, else OPENAI_API_KEY | the server model: MODEL_NAME after the DeepSeek switch |
| `X-Model-Key`, no provider or `X-Model-Provider: openai` | OPENAI_BASE_URL, else api.openai.com (LLM_BASE_URL ignored) | the caller's | `X-Model`, else the server model |
| `X-Model-Key` with another provider | `PROVIDERS[provider]`, even when OPENAI_BASE_URL is set | the caller's | `X-Model` (DeepSeek aliases normalised), else the server model |
| embeddings (every upload and question) | OPENAI_BASE_URL, else api.openai.com | OPENAI_API_KEY | EMBEDDING_MODEL |

- A follow-up (history sent, MAX_HISTORY_TURNS above 0) costs one condensing call on the same client
  and model, before retrieval; `RAGChain._condense` swallows every error and retrieves with the
  question as asked.
- There is no default model per provider: a provider named without `X-Model` gets the server model
  (gpt-4o-mini by default), which normally ends as the caller's 400.
- `POST /query` returns `model`, the one actually sent; null when nothing was retrieved and no
  answer was generated.

### Server endpoint and transport

`lifespan` in `app/main.py` builds the one chat client (`RAGChain(...)`) from
`LLM_API_KEY or OPENAI_API_KEY` and `LLM_BASE_URL or OPENAI_BASE_URL`, with OPENAI_TIMEOUT,
OPENAI_MAX_RETRIES and `httpx.Client(trust_env=False)`; `EmbeddingsManager(...)` gets OPENAI_API_KEY,
OPENAI_BASE_URL and the same timeout and retries, never the LLM_* values. DeepSeek switch: when
"deepseek" appears anywhere in that chat URL (case-insensitive), MODEL_NAME exactly `gpt-4o-mini`
becomes `deepseek-flash`, and any other value goes through `normalize_model_name("deepseek", ...)`.

| Setting | Default | Notes |
| --- | --- | --- |
| OPENAI_API_KEY | required | embeddings always; chat unless LLM_API_KEY is set |
| OPENAI_BASE_URL | `""` | embeddings; chat unless LLM_BASE_URL is set; BYOK with no provider or `openai` |
| LLM_BASE_URL | `""` | server chat only; with OPENAI_BASE_URL as fallback, the URL the DeepSeek switch tests |
| LLM_API_KEY | `""` | server chat only; never used for BYOK |
| MODEL_NAME | `gpt-4o-mini` | `min_length=1`; after the DeepSeek switch, also what a BYOK caller gets without `X-Model` |
| EMBEDDING_MODEL | `text-embedding-3-small` | startup refuses a collection built with another model |
| OPENAI_TIMEOUT | `45.0` | seconds per attempt, for the chat, embedding and every BYOK client |
| OPENAI_MAX_RETRIES | `2` | the SDK retries timeouts, connection errors, 408, 409, 429 and 5xx this many times |
| ALLOWED_MODEL_PROVIDERS | `""` | which `X-Model-Provider` values are accepted (Header rules) |
| RERANKER_API_KEY | `""` | the operator's Cohere key, also spent by `X-Reranker-Provider: cohere` without `X-Reranker-Key` |

One hanging call can hold a request for about (OPENAI_MAX_RETRIES + 1) x OPENAI_TIMEOUT, 135 s at
the defaults, longer than the 120 s both clients wait. Settings are cached (`get_settings`) and the
clients are built at startup, so restart the backend after changing any of these. `/ready` never
calls the model or the embedding API, so a wrong endpoint or key shows only on the first question.

### BYOK per request (`query_rag`, `query_rag_stream`)

1. `_caller_model` calls `byok.wanted(request.headers, request.app.state.settings)`, then
   `_caller_reranker` calls `byok.wanted_reranker(...)`; a `BringYourOwnKeyError` becomes 400 with
   its text.
2. Only after both pass: `client = byok.client_for(key, settings, provider) if key else None`, with
   the deployment's timeout and retries and `httpx.Client(trust_env=False)`.
3. The chain gets `client=`, `model=` and `reranker=` per call and stores none of them.
4. `/query` closes the client in `finally: byok.close_quietly(client)`.
5. `/query/stream` primes the generator with `next(stream)` - condensing, retrieval and opening the
   completion all run before the first yield - and closes the client in the caller-key
   `RateLimitError` branch and the catch-all `except Exception` branch; once streaming, `events()`
   closes it in its `finally`, after the last token or a dropped connection. The
   `APITimeoutError` priming branch does not close it (known issue below).

### Header rules (`byok.wanted`, in check order)

Values are stripped and the provider is lower-cased. No message contains the key.

1. `X-Model-Key` must match `^[A-Za-z0-9._\-]{8,256}$` (8, so a short gateway token passes).
2. `X-Model` must match `^[A-Za-z0-9._:/\-]{1,80}$` (allows `vendor/model` and `model:tag`).
3. `X-Model` or `X-Model-Provider` without a key is refused, so nobody chooses a model on the
   operator's bill.
4. `X-Model-Provider` must be in `allowed_providers(settings)`: an empty ALLOWED_MODEL_PROVIDERS
   means every `PROVIDERS` key; otherwise the list is comma-separated, trimmed and case-insensitive,
   unknown names are dropped and table order is kept. The 400 lists what is on offer. A key sent
   without a provider is never checked against the list.
5. Only when both a provider and a model are sent: `normalize_model_name(provider, model)`. It acts
   only for `deepseek`: compared lower-cased with "_" read as "-", `deepseek-v4.1-flash`,
   `deepseek-v4-flash`, `v4.1-flash` and `deepseek-chat` become `deepseek-flash`; anything else
   passes unchanged.

### PROVIDERS

| Name | Base URL | A bad key answers (per `app/byok.py`) |
| --- | --- | --- |
| openai | None: OPENAI_BASE_URL, else api.openai.com | 401 |
| anthropic | https://api.anthropic.com/v1/ | 401 |
| gemini | https://generativelanguage.googleapis.com/v1beta/openai/ | 400 |
| deepseek | https://api.deepseek.com/v1 | 401 |
| kimi | https://api.moonshot.ai/v1 | 401 |
| kimi-cn | https://api.moonshot.cn/v1 | 401 |

All speak OpenAI chat completions; the SDK appends `chat/completions`. `openai` is None on purpose:
it means the deployment's own endpoint, which may be a gateway, so its users reach whatever that
gateway offers. Dict order is the order of the "This deployment offers" message, the web picker and
the bot's `/model` replies.

### Refusals and status codes

| Failure | With the caller's key | On the operator's key |
| --- | --- | --- |
| a header rule (model or reranker headers) | 400 | 400 |
| `RateLimitError` (after the SDK's retries) | 400, "out of quota or rate limited" | 429 with `Retry-After: 20` |
| `APITimeoutError` | 504 | 504 |
| status 401/403, 404 or 402 | 400, one fixed message per status | 503 `Query failed` |
| any other status (a provider's 400, a 5xx after retries) | 400, "The provider refused the request (HTTP n)..." | 503 `Query failed` |
| no status (connection error, anything else) | 503 `Query failed` | 503 `Query failed` |

`describe_upstream_refusal` maps the status and never passes on the provider's own words. Why: the
caller can fix their own key, both clients show 401/403 as `OPERATOR_ERROR` ("contact the
operator"), and 429 with Retry-After means "retry shortly", wrong advice for an exhausted caller
key. The operator's 429 and every 504 and 503 are logged; a caller-key 400 leaves only the access
line, plus the condensing call's traceback when a follow-up was refused.

### Reranker headers (`byok.wanted_reranker`)

- `X-Reranker-Provider`: `none`, `flashrank` or `cohere` (stripped, lower-cased); anything else is
  a 400 that echoes the value; absent or empty means the operator's reranker. `none` switches the
  operator's reranker off for this request.
- `cohere` needs `X-Reranker-Key` or the operator's RERANKER_API_KEY, else 400; without
  `X-Reranker-Model` it uses RERANKER_MODEL, then `rerank-v3.5`. Key and model are stripped but not
  shape-checked.
- No `X-Model-Key` is needed, and neither RERANKER_ENABLED nor ALLOWED_MODEL_PROVIDERS is consulted
  (known issue below). A refused Cohere key is not reported: `CohereReranker.rerank` logs a
  warning, keeps the retrieval order, and the answer is still 200.

### Clients

Both send the BYOK headers with questions only and show a caller-key 400 as its detail text,
without a request id. Web UI: the sidebar panel "Answer on my own API key" (session state only)
feeds `asking_headers` for `/query/stream`, omitting the provider when it is `openai`. Bot:
`/model <model> <key>` or `/model <provider> <model> <key>`, `/model` shows the state without the
key, `/model reset` forgets; the key sits in `context.user_data` (per Telegram user, in memory) and
its message is deleted, or the user is told to delete it. Panel details and the `/model` parse
table: [references/providers.md](references/providers.md).

## Recipes

### Add a provider

1. Confirm it serves OpenAI-compatible chat completions; take the base URL without
   `chat/completions`.
2. Learn what a bad key answers, for the docs table. This is a real network call: ask the user
   first, never use a real key, and never put it in a test.
   ```
   curl -s -o /dev/null -w '%{http_code}\n' <base-url>/chat/completions \
     -H 'Authorization: Bearer sk-invalid-0000000000' -H 'Content-Type: application/json' \
     -d '{"model": "<a model it offers>", "messages": [{"role": "user", "content": "hi"}]}'
   ```
3. `app/byok.py`: add `"<name>": "<base url>"` to `PROVIDERS`, in lower-case letters, digits and
   hyphens (the header and the comma-separated allow-list are lower-cased). Its position is the
   order users see.
4. Only if needed: a status branch in `describe_upstream_refusal` (the catch-all already makes any
   status a 400) and aliases in `normalize_model_name` (next recipe).
5. `tests/test_byok.py`: add the name to the exact set in
   `test_the_offered_providers_are_the_ones_documented` and a `(provider, host)` row to
   `test_each_provider_gets_its_own_endpoint`.
6. `frontend/streamlit_app.py`: add a model hint to `provider_placeholders` (otherwise it shows
   `gpt-4o`). The picker reads `PROVIDERS`, and so do the bot's `/model` replies; the bot's `/help`
   names no provider, so no bot text needs editing.
7. Docs: a row (name, endpoint, bad-key status) in the provider table of DOCUMENTATION.md, section
   "Your own key and model"; the provider list in the `.env.template` comment above
   ALLOWED_MODEL_PROVIDERS (a comment only: a `NAME=` line that is not a Settings field fails
   `tests/test_docs_consistency.py`); the names in the README "Your Own Key and Model" feature and
   its Telegram `/model` line; the PROVIDERS table in this skill.
   `grep -rnI "kimi-cn" --exclude-dir=.venv --exclude-dir=.git .` finds every list.
8. Leave the ALLOWED_MODEL_PROVIDERS default alone: empty already admits the new provider, and a
   narrowed list is each operator's decision.
9. Run Verify. Tell callers to send `X-Model`: nothing supplies a per-provider default.

### Add a model alias or change normalisation

1. Add a branch to `normalize_model_name` keyed on the provider; compare
   `model.lower().replace("_", "-")` and return the canonical id.
2. It runs in two places: `byok.wanted` (both `X-Model-Provider` and `X-Model` sent) and the
   lifespan (a DeepSeek URL only, always as provider `deepseek`, after the `gpt-4o-mini` special
   case). A model sent without a provider is never normalised; server-side normalisation for
   another vendor's URL needs a new condition in `lifespan`.
3. Tests: a `byok.wanted` case like `test_deepseek_v4_flash_model_is_accepted`. The startup path
   has no test yet: add one shaped like `test_transport_settings_reach_both_clients` in
   `tests/test_resilience.py` - `create_app(make_settings(tmp_path, llm_base_url=...,
   model_name=...))`, then assert `app.state.rag_chain.model` inside `with TestClient(app):`.
4. If the canonical name changes, update the deepseek hint and the model box help text in
   `frontend/streamlit_app.py`, the README "Any OpenAI-compatible Chat Endpoint" feature and
   DOCUMENTATION.md.
5. Aliases rewrite silently: confirm the target id exists at the provider (not checkable offline).

### Point the server at another OpenAI-compatible endpoint

- Chat and embeddings through one gateway: OPENAI_BASE_URL and OPENAI_API_KEY. The gateway must
  serve EMBEDDING_MODEL, and BYOK callers with no provider (or `openai`) go there too.
- Answers elsewhere, embeddings on OpenAI: LLM_BASE_URL, LLM_API_KEY and a MODEL_NAME that endpoint
  serves; leave OPENAI_BASE_URL empty.
- DeepSeek: `LLM_BASE_URL=https://api.deepseek.com` and LLM_API_KEY; the default MODEL_NAME
  becomes `deepseek-flash`.
- Keep EMBEDDING_MODEL unless you re-index: startup refuses a collection built with another model.

Then restart and send one question. A wrong model or key on the operator's side shows as 503
`Query failed`, with the provider's `Error code: 4xx - ...` in the logged traceback; an unreachable
endpoint as 504 (timeout) or 503 (connection error) after the retries. The OpenAI clients ignore
HTTP(S)_PROXY (`trust_env=False`), so an egress proxy needs a code change; the Cohere reranker and
Langfuse do honour it. A new transport knob goes through the rag-config-and-docs skill, with a test
like `test_transport_settings_reach_both_clients`.

### Change how a refusal is reported

Edit `describe_upstream_refusal` (fixed wording, never the key or the provider's text), then the
`except` chain of `query_rag` and the priming block of `query_rag_stream` together, closing the
client in every stream branch and keeping caller-key outcomes out of 401/403 and 429. Extend
`test_the_caller_hears_what_their_key_did` (helpers `_openai_error` and `Refusing` in
`tests/test_byok.py`), add a `/query/stream` case asserting `refusing.closed` like
`test_a_streamed_refusal_is_a_status_code_not_a_broken_stream`, and update DOCUMENTATION.md "Your
own key and model" plus the error lists of `POST /query` and `POST /query/stream` (rag-api-endpoints).

### Debug "my own key does not work"

- Match the status and detail text in [references/providers.md](references/providers.md): a
  header-rule message names the broken rule; a refusal message reflects the upstream status.
- Never ask for the key or put one in a command, log or test; reproduce with a dummy key-shaped
  value, or in a test with `Refusing`.
- A caller-key 400 leaves only the access line (and, for a follow-up, the condensing call's
  traceback with the provider's text). With Langfuse on, the trace (id = request id) is
  tagged with the provider or `default`, and a failed generation's `status_message` holds the
  provider's error text (tracing: rag-operations).
- A refusal message does not prove the caller's key is bad: with a key present, any status error is
  reported that way, the operator's question-embedding call included. If the same question without
  `X-Model-Key` also fails (503), the deployment is at fault. "Does not offer that model" (404) or
  "HTTP 400" after naming a provider without `X-Model` usually means the server model went there.

## Invariants

- The endpoint never comes from the request - base URLs come only from `PROVIDERS`,
  OPENAI_BASE_URL or (the server's own client only) LLM_BASE_URL, because a caller-supplied URL
  makes the backend fetch anything (SSRF).
- A model or provider only together with a key - otherwise a caller picks a costlier model on the
  operator's bill.
- One client per request from `client_for`, released by `close_quietly` on every exit, nothing
  cached or pooled per key - the key must not outlive its request, and an unclosed client leaks a
  connection pool.
- Validate every header before `client_for`, and keep the code between `client_for` and the `try`
  that closes it non-raising (today `tracer.start_trace`, which swallows its own errors, and
  creating the stream generator) - otherwise a 400 or an exception leaks the client.
- Pass `request.app.state.settings` to `wanted` and `wanted_reranker` - without it `wanted` reads
  the allow-list as empty, which admits every provider, and `wanted_reranker` ignores
  RERANKER_API_KEY and RERANKER_MODEL.
- The keys (`X-Model-Key`, `X-Reranker-Key`) never reach an HTTP detail, a log line, a trace or
  the disk, and the provider's own words never reach an HTTP detail - error text is the easiest
  way for a secret into a log. The model and provider names are not secret: `X-Model` is logged
  with each completion and traced, the provider is a trace tag. Add no new paths for provider text: it already reaches Langfuse and exception tracebacks.
- A refusal of the caller's own key is 400, never 401/403 or 429; operator failures keep
  429/504/503.
- Embeddings stay on the operator's key and model - the collection is bound to one embedding model.
- `ask_stream` opens the completion before yielding sources and the handler primes it, so a refusal
  or timeout is a status code rather than a broken 200 stream.

## Pitfalls and known issues

Known issues (one line each; KNOWN_ISSUES.md holds repro and impact):
- `/query/stream` does not close the caller's client when the model or the operator's question
  embedding times out while priming (504). See KNOWN_ISSUES.md, section Models, keys and reranking.
- `X-Reranker-Provider` works without `X-Model-Key`, ignores RERANKER_ENABLED and
  ALLOWED_MODEL_PROVIDERS, and `cohere` without `X-Reranker-Key` spends the operator's
  RERANKER_API_KEY. See KNOWN_ISSUES.md, section Models, keys and reranking.
- flashrank is in no requirements file or image, so choosing it (setting or header) keeps the
  retrieval order and only logs a warning. See KNOWN_ISSUES.md, section Models, keys and reranking.
- Bot `/model`: `/model <key>` alone is refused and the key stays in the chat; `/model <key> <model>`
  stores the two swapped, repeats the key in the bot's confirmation and later sends it as
  `X-Model`; `/model <provider> <key>` stores the provider name as the model and sends the key to
  the default endpoint. See KNOWN_ISSUES.md, section Telegram bot.

Current behaviour that surprises:
- With a caller key, any status error is reported as a refusal of that key: a broken operator
  OPENAI_API_KEY failing the question embedding shows the caller "The provider rejected the API key
  you sent.", and a provider 5xx after retries becomes a 400 telling the caller to check their key.
- ALLOWED_MODEL_PROVIDERS is checked only when `X-Model-Provider` is sent; a key alone always goes
  to OPENAI_BASE_URL or api.openai.com, even when `openai` is not listed. No setting turns BYOK off.
- BYOK ignores LLM_BASE_URL and LLM_API_KEY: on a deployment answering through LLM_BASE_URL, a
  key-only caller's key goes to OPENAI_BASE_URL or api.openai.com with the server model (possibly
  `deepseek-flash`).
- Both clients offer every `PROVIDERS` entry, not the deployment's allow-list; a list made only of
  unknown names produces "This deployment offers: ."
- The DeepSeek switch is a substring test: an OPENAI_BASE_URL gateway whose URL contains "deepseek"
  also rewrites `gpt-4o-mini` (and serves the embeddings); `deepseek-chat` is rewritten both as the
  server model on a DeepSeek URL and as an `X-Model` sent with the `deepseek` provider (a key-only
  caller's `X-Model` is never rewritten); the `gpt-4o-mini` match is case-sensitive. Whether
  DeepSeek serves `deepseek-flash` cannot be checked offline.
- Provider error text (`Error code: n - body`) reaches Langfuse (the generation's `status_message`,
  the trace's `metadata.error`) and the app log via `logger.exception` for a failed condensing call,
  an operator-side 503 or a stream that fails after it started.
- `evaluation/langfuse_eval.py` sends `--provider` and `--model` only together with `--model-key`;
  without it the run silently uses the server model (see rag-evaluation).

## Verify

```
python -m pytest tests/test_byok.py tests/test_reranker.py tests/test_resilience.py tests/test_streaming.py -q
python -m pytest tests/test_frontend_feedback.py tests/test_docs_consistency.py -q
ruff check app frontend clients evaluation scripts tests
```

Run from the repo root (the web UI tests resolve the script path from it); from WSL, use the venv's
own interpreter, `.venv/Scripts/python.exe -m pytest ... -p no:cacheprovider` or a Linux venv's
`~/.venvs/<name>/bin/python -m pytest ...` (see rag-testing). No real key or network is needed:
`tests/conftest.py` sets a dummy OPENAI_API_KEY and blocks non-loopback sockets, API tests replace
`byok.client_for` via monkeypatch (`FakeChatClient`, or `Refusing` for refusals), and tests of the
real `client_for` only inspect `base_url`, `timeout` and `api_key`, then close it. After editing
docs, also run the doc-drift tests listed in the rag-config-and-docs skill.
