# Providers reference: messages, bot parsing, tests

Exact caller-facing text, the Telegram `/model` parse outcomes and the tests that pin each
rule. The SKILL.md explains why; this file is for matching a user's error text, writing a
test, or touching a client.

Contents
- Messages a caller can see
- Where the text surfaces
- The web UI's key and reranker panels
- The bot's `/model` parsing
- Test index and helpers

## Messages a caller can see

Header rules, from `app/byok.py` `wanted` then `wanted_reranker`, in check order. All are
400 and none contains the key.

| Rule broken | `detail` |
| --- | --- |
| `X-Model-Key` shape | The API key you sent is not shaped like one. Check for a stray space or a truncated paste. |
| `X-Model` shape | That model name has characters a model name cannot have. |
| `X-Model` or `X-Model-Provider` without a key | Choosing a model or a provider needs your own API key: send it too, or send neither and the assistant answers with its own. |
| provider not allowed (or unknown) | Unknown or disallowed provider. This deployment offers: (the allowed names, comma-separated). |
| `X-Reranker-Provider` not none, flashrank or cohere | Unknown reranker provider '(the value, lower-cased)'. Supported: none, flashrank, cohere. |
| `cohere` with no `X-Reranker-Key` and no RERANKER_API_KEY | Cohere reranker requires an API key in X-Reranker-Key header. |

The no-key rule runs before the allow-list, so an unknown provider sent without a key gets
the "needs your own API key" message.

Refusals of the caller's own key, from `describe_upstream_refusal` (400):

| Upstream status | `detail` |
| --- | --- |
| 401 or 403 | The provider rejected the API key you sent. |
| 404 | The provider does not offer that model to your key. Check the model name. |
| 429 (`RateLimitError`, after the SDK's retries) | Your API key is out of quota or rate limited by the provider. This is on your account, not on the assistant. |
| 402 | Your account with that provider is out of balance. |
| any other status, 5xx included | The provider refused the request (HTTP n). Check the key and that the model name is one that provider offers. |

Outcomes that are the same with or without a caller key, from `query_rag` and
`query_rag_stream`:

| Status | `detail` | Logged |
| --- | --- | --- |
| 504 (`APITimeoutError`) | The model did not answer in time. Please retry. | WARNING "OpenAI timed out after (OPENAI_TIMEOUT)s" |
| 503 (no status: connection error, anything else) | Query failed | `logger.exception("Query failed")` with the traceback |

Operator key only: 429 "Upstream model is rate limited. Please retry shortly." with
`Retry-After: 20`, logged as a WARNING "OpenAI rate limited the request: ..."; a status
error (401, 404, 400, 5xx) is 503 `Query failed`, and the traceback holds the provider's
`Error code: n - body`.

## Where the text surfaces

- Web UI: `st.error(...)` of `clients/backend.py` `error_from_response`, so a 400 shows its
  detail as written. 401/403 become `OPERATOR_ERROR` ("The assistant is not configured
  correctly. Please contact the operator."). A request id is appended only for 401/403
  and 5xx.
- Bot: the same text, after an error mark; `backend_error` also logs 401/403 as a
  BACKEND_API_KEY problem.
- Backend log: a caller-key 400 leaves only the access line, for example
  `POST /query -> 400 in 9 ms (user=-)`; a refused follow-up also leaves the condensing
  call's traceback ("Could not condense the follow-up; using it as written").
- Langfuse, when enabled: each `/query` and `/query/stream` trace is tagged with the
  provider name or `default`; a failed generation stores `str(exc)` as `status_message` and
  in the trace's `metadata.error`, which for an SDK status error is the provider's own text.
  The key itself is never traced.

## The web UI's key and reranker panels

`frontend/streamlit_app.py`:
- "Answer on my own API key" must stay the first sidebar expander (a test checks it). It
  holds a password field, a provider picker (`PROVIDER_NAMES`, every `PROVIDERS` key) and a
  model box; the last two are disabled until a key is typed. All three live in Streamlit
  session state only and go with the tab.
- `asking_headers` sends the key, the model when set and the provider unless it is `openai`
  (the default, so sending it would be noise), on `/query/stream` only; uploads, listing,
  deletes, clear and ratings send only `X-API-Key`.
- The model placeholder comes from the `provider_placeholders` dict (fallback `gpt-4o`) and is
  only a hint: an empty box sends no `X-Model`, so the deployment's model goes to the chosen
  provider.
- The picker shows every `PROVIDERS` entry, not the deployment's allow-list.
- "Cross-Encoder Reranker" offers none, flashrank and cohere (a hard-coded copy of the tuple
  in `wanted_reranker`, so a new reranker provider needs both) and sends `X-Reranker-*` with
  or without an own key. Its `none` sends no header, which means the server default, so it
  cannot switch off a server-enabled reranker. The Cohere key and model boxes appear only
  for cohere.
- The main pane says which model answers ("Answers come from (model) at (provider), on your
  key.") or where to set one.

## The bot's `/model` parsing

`clients/telegram_bot.py` `model_command` splits the text on whitespace. A provider is
taken only when there are three or more words and the first is a `PROVIDERS` key (any
case). Then the first remaining word is the model and the second the key; later words are
ignored. Nothing is validated in the bot: the backend checks the values on the next
question. State lives in `context.user_data` under `own_key`, `own_model` and
`own_provider`.

| Message | Stored | Key message deleted | What follows |
| --- | --- | --- | --- |
| `/model` | nothing | no key sent | reply: key held yes/no, provider, model, usage, the `PROVIDERS` list |
| `/model reset` | forgets all three | no key sent | answers from the server model again; `Reset` in another case counts as a model without a key |
| `/model <model> <key>` | model and key; forgets any provider | yes, or a bold "delete it yourself" | questions go to OPENAI_BASE_URL or api.openai.com |
| `/model <provider> <model> <key>` | all three | yes, or the warning | questions go to that provider's URL |
| `/model <key>` | nothing | no, and no warning | usage reply; the key stays in the chat (known issue) |
| `/model <key> <model>` | model = the key, key = the model | yes, but the confirmation repeats the key | questions send the key as `X-Model`; if it fits the model pattern the backend forwards it upstream as the model name (known issue) |
| `/model <provider> <key>` | model = the provider name, the key, no provider | yes | the key goes to the default endpoint, and the bot confirms "Answers now come from (provider)" (known issue) |

The three known issues are described in KNOWN_ISSUES.md, section Telegram bot.

Questions only, through `/query`: `asking_headers(context)` adds `X-Model-Key`, then
`X-Model` and `X-Model-Provider` when stored, and nothing at all without a key. `/start` and
`/clear` drop the history but keep the key. The data is per Telegram user, so a key set in a
private chat also answers that user's questions in a group, until `/model reset` or a bot
restart.

## Test index and helpers

`tests/test_byok.py` unless another file is named.

| Area | Tests |
| --- | --- |
| header rules | `test_a_model_without_a_key_is_refused`, `test_a_key_that_is_not_shaped_like_one_is_refused`, `test_model_names_people_actually_use_are_accepted`, `test_a_model_name_that_is_not_one_is_refused`, `test_the_refusal_never_repeats_the_key_back`, `test_a_provider_without_a_key_is_refused`, `test_provider_names_are_case_insensitive` |
| provider table | `test_the_offered_providers_are_the_ones_documented` (exact set), `test_openai_means_the_deployments_own_endpoint`, `test_each_provider_gets_its_own_endpoint`, `test_a_providers_url_beats_the_deployments`, `test_the_caller_still_cannot_name_a_url` |
| allow-list | `test_an_operator_can_narrow_the_list`, `test_an_empty_list_means_every_provider`, `test_the_refusal_names_what_is_on_offer`, `test_the_deployments_allowlist_is_enforced_by_the_api` |
| normalisation | `test_deepseek_v4_flash_model_is_accepted` (the startup DeepSeek switch and LLM_* have no test) |
| transport | `test_the_client_carries_their_key`, `test_the_transport_is_the_deployments_own`; in `tests/test_resilience.py`: `test_openai_client_gets_an_explicit_timeout`, `test_embedding_client_gets_an_explicit_timeout`, `test_base_url_override_reaches_the_client`, `test_transport_settings_reach_both_clients` (checks the chat client only) |
| API wiring | `test_a_caller_with_a_key_gets_their_own_model`, `test_a_key_without_a_model_answers_with_the_configured_one`, `test_a_model_without_a_key_is_a_400`, `test_a_malformed_key_is_a_400_and_answers_nothing`, `test_the_follow_up_rewrite_goes_on_their_key_too`, `test_a_provider_reaches_the_client_through_the_api`, `test_an_unknown_provider_is_a_400` |
| refusals | `test_the_caller_hears_what_their_key_did` (401, 404, 429), `test_a_gemini_style_400_is_still_the_callers_problem`, `test_an_empty_balance_says_so`, `test_a_refusal_does_not_leak_the_providers_own_words`, `test_a_failure_with_no_status_is_not_blamed_on_the_caller`, `test_a_caller_never_sees_the_operator_error_for_their_own_key`, `test_the_deployments_own_failure_is_still_a_503`; in `tests/test_resilience.py`: `test_rate_limit_becomes_429_with_retry_after`, `test_timeout_becomes_504`, `test_other_failures_still_return_503`, `test_upstream_errors_do_not_leak_provider_detail` |
| key lifecycle | `test_the_clients_key_does_not_outlive_the_request`, `test_the_key_is_let_go_even_when_the_provider_refuses`, `test_nothing_writes_the_key_anywhere` (logs and disk), `test_closing_is_quiet_about_a_client_that_will_not`, `test_indexing_stays_on_the_deployments_key` |
| streaming | `test_a_streamed_answer_uses_their_key_and_closes_it`, `test_a_streamed_refusal_is_a_status_code_not_a_broken_stream`, `test_the_sources_event_still_comes_first`; in `tests/test_streaming.py`: `test_a_rate_limit_before_the_stream_is_a_normal_429`, `test_a_timeout_before_the_stream_is_a_normal_504` (operator key only; nothing covers a caller-key timeout) |
| reranker headers | in `tests/test_reranker.py`: the `test_wanted_reranker_*` tests and `test_get_reranker_respects_provider_override` (the operator-key fallback for cohere is untested) |
| bot | `test_the_bot_remembers_the_key_and_model`, `test_the_bot_deletes_the_message_carrying_the_key`, `test_when_it_cannot_delete_it_says_so`, `test_the_bot_refuses_a_model_without_a_key`, `test_the_bot_forgets_on_request`, `test_asking_without_arguments_never_shows_the_key_back`, `test_the_bot_takes_a_provider_as_a_third_word`, `test_two_words_still_mean_model_and_key`, `test_naming_a_model_without_a_provider_forgets_the_last_one`, `test_reset_forgets_the_provider_too`, `test_the_bot_lists_the_providers_it_can_reach`, the `asking_headers` tests, `test_the_bot_actually_sends_the_headers_when_asked_a_question` |
| web UI | in `tests/test_frontend_feedback.py`: `test_no_key_typed_means_no_header_sent`, `test_the_key_and_model_go_with_the_question`, `test_a_model_is_never_sent_without_a_key`, `test_the_web_ui_sends_the_provider_it_was_given`, `test_the_default_provider_is_not_sent`, `test_the_picker_offers_every_provider_the_backend_knows`, `test_the_key_box_is_the_second_thing_in_the_sidebar` (the key expander must be the first expander), `test_uploading_stays_on_the_operators_key` (also passes when no upload was posted; the backend test is the one that holds) |

Helpers in `tests/test_byok.py`:
- `GOOD_KEY` - a key-shaped dummy; use it, never a real key.
- `Refusing(exc)` - a client whose every completion raises `exc`; `.closed` records the close.
- `_openai_error(cls, status)` - builds an SDK status error (`AuthenticationError`,
  `NotFoundError`, `RateLimitError`, `BadRequestError`, `APIStatusError`) without a network.
- `_model_command(text, user_data=None, delete_raises=False)` - drives the bot handler with
  `SimpleNamespace` fakes and returns (replies, sent, user_data).
- Replace the per-request client with
  `monkeypatch.setattr(byok, "client_for", lambda key, settings, provider=None: FakeChatClient())`;
  `FakeChatClient` comes from `tests/conftest.py`.
