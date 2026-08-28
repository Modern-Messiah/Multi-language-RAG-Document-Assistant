"""Answering on the caller's own key, with the model they chose.

The deployment's key still pays for indexing: the vector store is one
collection bound to one embedding model, so the choice on offer is the answer
model - which is the one a person can tell apart anyway.

Most of this file is about the ways it must not go wrong: a key that is never
written anywhere, a model that cannot be chosen on someone else's account, and
a rejection that reaches the person who can act on it instead of being hidden
as an operator problem.
"""
import json

import pytest
from openai import AuthenticationError, NotFoundError, RateLimitError

from app import byok
from app.byok import KEY_HEADER, MODEL_HEADER
from tests.conftest import FakeChatClient

TEXT = b"Annual leave for an engineer is twenty eight calendar days."
GOOD_KEY = "sk-test-0123456789abcdefghij"


def _index(api, user_id="u1"):
    response = api.post(
        "/upload", params={"user_id": user_id},
        files={"file": ("policy.txt", TEXT, "text/plain")},
    )
    assert response.status_code == 200, response.text


def _ask(api, headers=None, user_id="u1"):
    return api.post(
        "/query",
        json={"question": "How many leave days?", "language": "Auto", "user_id": user_id},
        headers=headers or {},
    )


class Refusing:
    """An OpenAI client whose every completion is refused by the provider."""

    def __init__(self, exc):
        outer = self

        class _Completions:
            def create(self, **kwargs):
                raise outer.exc

        self.exc = exc
        self.closed = False
        self.chat = type("Chat", (), {"completions": _Completions()})()

    def close(self):
        self.closed = True


def _openai_error(cls, status):
    """One of the SDK's status errors, built without a live response."""
    import httpx

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": {"message": "no"}})
    return cls("refused", response=response, body=None)


# =========================
# What a caller may ask for
# =========================

def test_nothing_asked_for_is_nothing_returned():
    assert byok.wanted({}) == (None, None, None)


def test_a_key_and_a_model_come_back():
    key, model, provider = byok.wanted({KEY_HEADER: GOOD_KEY, MODEL_HEADER: "gpt-4o"})

    assert (key, model, provider) == (GOOD_KEY, "gpt-4o", None)


def test_a_key_alone_is_enough():
    """Their key, the deployment's model: a reasonable thing to want."""
    assert byok.wanted({KEY_HEADER: GOOD_KEY}) == (GOOD_KEY, None, None)


def test_a_model_without_a_key_is_refused():
    """Otherwise choosing a model is a way to spend the operator's money on a
    costlier one than they configured."""
    with pytest.raises(byok.BringYourOwnKeyError, match="needs your own API key"):
        byok.wanted({MODEL_HEADER: "gpt-4o"})


@pytest.mark.parametrize("bad", ["tiny", "has space", "sk-\nnewline", "ключ" * 6, "x" * 300])
def test_a_key_that_is_not_shaped_like_one_is_refused(bad):
    with pytest.raises(byok.BringYourOwnKeyError):
        byok.wanted({KEY_HEADER: bad})


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1", "o3-mini", "openai/gpt-4o", "a" * 80])
def test_model_names_people_actually_use_are_accepted(model):
    assert byok.wanted({KEY_HEADER: GOOD_KEY, MODEL_HEADER: model})[1] == model


@pytest.mark.parametrize("bad", ["gpt 4o", "gpt-4o;rm -rf", "a" * 81, "model\nname"])
def test_a_model_name_that_is_not_one_is_refused(bad):
    with pytest.raises(byok.BringYourOwnKeyError):
        byok.wanted({KEY_HEADER: GOOD_KEY, MODEL_HEADER: bad})


def test_the_refusal_never_repeats_the_key_back():
    """Not even truncated: whatever they sent, the fix is the same, and an
    error message is the easiest place for a secret to end up in a log."""
    with pytest.raises(byok.BringYourOwnKeyError) as exc:
        byok.wanted({KEY_HEADER: "sk-secret-but-has a space"})

    assert "sk-secret" not in str(exc.value)


# =========================
# The client built for them
# =========================

def test_the_client_carries_their_key(tmp_path):
    from tests.conftest import make_settings

    client = byok.client_for(GOOD_KEY, make_settings(tmp_path))
    try:
        assert client.api_key == GOOD_KEY
    finally:
        client.close()


def test_the_transport_is_the_deployments_own(tmp_path):
    """A caller's key must not make the backend talk somewhere else or wait
    longer than the operator allows - a caller-supplied base URL would make
    this an errand boy for any address it is handed."""
    from tests.conftest import make_settings

    settings = make_settings(tmp_path, openai_timeout=7.0, openai_base_url="http://gw:9/v1")
    client = byok.client_for(GOOD_KEY, settings)
    try:
        assert str(client.base_url).startswith("http://gw:9/v1")
        assert client.timeout == 7.0
    finally:
        client.close()


def test_closing_is_quiet_about_a_client_that_will_not():
    class Stubborn:
        def close(self):
            raise RuntimeError("no")

    byok.close_quietly(Stubborn())  # must not raise
    byok.close_quietly(None)


# =========================
# Through the API
# =========================

def test_a_question_without_a_key_uses_the_deployments_model(api):
    _index(api)

    body = _ask(api).json()

    assert body["model"] == api.app_state.settings.model_name
    assert api.chat.calls, "the deployment's client was not the one used"


def test_a_caller_with_a_key_gets_their_own_model(api, monkeypatch):
    _index(api)
    theirs = FakeChatClient(answer="Answered on your key.")
    monkeypatch.setattr(byok, "client_for", lambda key, settings, provider=None: theirs)

    body = _ask(api, {KEY_HEADER: GOOD_KEY, MODEL_HEADER: "gpt-4o"}).json()

    assert body["answer"] == "Answered on your key."
    assert body["model"] == "gpt-4o"
    assert theirs.calls[0]["model"] == "gpt-4o"
    assert not api.chat.calls, "the deployment's key answered anyway"


def test_their_key_is_the_one_handed_to_the_client(api, monkeypatch):
    _index(api)
    seen = {}

    def watching(key, settings, provider=None):
        seen["key"] = key
        return FakeChatClient()

    monkeypatch.setattr(byok, "client_for", watching)

    _ask(api, {KEY_HEADER: GOOD_KEY})

    assert seen["key"] == GOOD_KEY


def test_a_key_without_a_model_answers_with_the_configured_one(api, monkeypatch):
    _index(api)
    theirs = FakeChatClient()
    monkeypatch.setattr(byok, "client_for", lambda key, settings, provider=None: theirs)

    body = _ask(api, {KEY_HEADER: GOOD_KEY}).json()

    assert body["model"] == api.app_state.settings.model_name
    assert theirs.calls[0]["model"] == api.app_state.settings.model_name


def test_a_model_without_a_key_is_a_400(api):
    _index(api)

    response = _ask(api, {MODEL_HEADER: "gpt-4o"})

    assert response.status_code == 400
    assert "needs your own API key" in response.json()["detail"]


def test_a_malformed_key_is_a_400_and_answers_nothing(api):
    _index(api)

    response = _ask(api, {KEY_HEADER: "nope"})

    assert response.status_code == 400
    assert not api.chat.calls, "it fell back to the operator's key"


def test_the_follow_up_rewrite_goes_on_their_key_too(api, monkeypatch):
    """Billing half an exchange to each side would be the strangest split."""
    _index(api)
    theirs = FakeChatClient()
    monkeypatch.setattr(byok, "client_for", lambda key, settings, provider=None: theirs)

    api.post("/query", json={
        "question": "and the second one?",
        "language": "Auto", "user_id": "u1",
        "history": [{"question": "How many leave days?", "answer": "28."}],
    }, headers={KEY_HEADER: GOOD_KEY})

    assert len(theirs.calls) == 2, "the condensing call went elsewhere"
    assert not api.chat.calls


# =========================
# When the provider says no
# =========================

@pytest.mark.parametrize("cls,status,expected", [
    (AuthenticationError, 401, "rejected the API key"),
    (NotFoundError, 404, "does not offer that model"),
    (RateLimitError, 429, "out of quota"),
])
def test_the_caller_hears_what_their_key_did(api, monkeypatch, cls, status, expected):
    """Every other 401 in this system is the operator's problem and both
    clients hide it as one. This one the caller can fix."""
    _index(api)
    monkeypatch.setattr(
        byok, "client_for",
        lambda key, settings, provider=None: Refusing(_openai_error(cls, status)),
    )

    response = _ask(api, {KEY_HEADER: GOOD_KEY})

    assert response.status_code == 400, response.text
    assert expected in response.json()["detail"]


def test_a_refusal_does_not_leak_the_providers_own_words(api, monkeypatch):
    """A provider message can carry account details, and says nothing the
    caller cannot work out from "your key was rejected"."""
    _index(api)
    monkeypatch.setattr(
        byok, "client_for",
        lambda key, settings, provider=None: Refusing(_openai_error(AuthenticationError, 401)),
    )

    detail = _ask(api, {KEY_HEADER: GOOD_KEY}).json()["detail"]

    assert "refused" not in detail.lower()


def test_a_caller_never_sees_the_operator_error_for_their_own_key():
    """clients.backend hides 401 and 403 behind "contact the operator", which
    would be nonsense advice for a key the caller owns."""
    from clients.backend import OPERATOR_ERROR, describe_error

    shown = describe_error(400, "The provider rejected the API key you sent.")

    assert shown != OPERATOR_ERROR
    assert "rejected" in shown


def test_the_deployments_own_failure_is_still_a_503(api):
    """Without a caller's key, an upstream failure is the operator's problem
    and keeps its old status."""
    _index(api)

    def boom(**kwargs):
        raise RuntimeError("chroma exploded")

    api.chat.chat.completions.create = boom

    assert _ask(api).status_code == 503


# =========================
# Letting go of the key
# =========================

def test_the_clients_key_does_not_outlive_the_request(api, monkeypatch):
    """No cache keyed on the secret: a pool per key would be faster and would
    mean the key outlives the request that carried it."""
    _index(api)
    built = []

    def watching(key, settings, provider=None):
        client = FakeChatClient()
        client.closed = False
        client.close = lambda: setattr(client, "closed", True)
        built.append(client)
        return client

    monkeypatch.setattr(byok, "client_for", watching)

    _ask(api, {KEY_HEADER: GOOD_KEY})
    _ask(api, {KEY_HEADER: GOOD_KEY})

    assert len(built) == 2, "a client was reused, so the key was kept"
    assert all(client.closed for client in built)


def test_the_key_is_let_go_even_when_the_provider_refuses(api, monkeypatch):
    _index(api)
    refusing = Refusing(_openai_error(AuthenticationError, 401))
    monkeypatch.setattr(byok, "client_for", lambda key, settings, provider=None: refusing)

    _ask(api, {KEY_HEADER: GOOD_KEY})

    assert refusing.closed


def test_nothing_writes_the_key_anywhere(api, monkeypatch, caplog):
    """The one property that makes "not stored" true: not in the log, not in
    the access line, not in a rating."""
    _index(api)
    monkeypatch.setattr(byok, "client_for", lambda key, settings, provider=None: FakeChatClient())

    with caplog.at_level("DEBUG"):
        _ask(api, {KEY_HEADER: GOOD_KEY, MODEL_HEADER: "gpt-4o"})
        api.post("/feedback", json={
            "rating": "up", "user_id": "u1", "question": "How many leave days?",
        }, headers={KEY_HEADER: GOOD_KEY})

    assert not any(GOOD_KEY in r.getMessage() for r in caplog.records)
    on_disk = [
        path for path in api.upload_dir.parent.rglob("*")
        if path.is_file() and GOOD_KEY.encode() in path.read_bytes()
    ]
    assert on_disk == [], on_disk


# =========================
# Streaming
# =========================

def test_a_streamed_answer_uses_their_key_and_closes_it(api, monkeypatch):
    _index(api)
    theirs = FakeChatClient(answer="Streamed on your key.")
    theirs.closed = False
    theirs.close = lambda: setattr(theirs, "closed", True)
    monkeypatch.setattr(byok, "client_for", lambda key, settings, provider=None: theirs)

    with api.stream("POST", "/query/stream", json={
        "question": "How many leave days?", "language": "Auto", "user_id": "u1",
    }, headers={KEY_HEADER: GOOD_KEY, MODEL_HEADER: "gpt-4o"}) as response:
        events = [
            json.loads(line[len("data: "):])
            for line in response.iter_lines() if line.startswith("data: ")
        ]

    assert response.status_code == 200
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer == "Streamed on your key."
    assert theirs.calls[0]["model"] == "gpt-4o"
    assert theirs.closed, "the caller's key outlived the stream"


def test_a_streamed_refusal_is_a_status_code_not_a_broken_stream(api, monkeypatch):
    """Retrieval and the first token happen before the response starts, which
    is what lets this stay an ordinary 400."""
    _index(api)
    refusing = Refusing(_openai_error(AuthenticationError, 401))
    monkeypatch.setattr(byok, "client_for", lambda key, settings, provider=None: refusing)

    response = api.post("/query/stream", json={
        "question": "How many leave days?", "language": "Auto", "user_id": "u1",
    }, headers={KEY_HEADER: GOOD_KEY})

    assert response.status_code == 400
    assert "rejected the API key" in response.json()["detail"]
    assert refusing.closed


def test_the_sources_event_still_comes_first(api, monkeypatch):
    _index(api)
    monkeypatch.setattr(byok, "client_for", lambda key, settings, provider=None: FakeChatClient())

    with api.stream("POST", "/query/stream", json={
        "question": "How many leave days?", "language": "Auto", "user_id": "u1",
    }, headers={KEY_HEADER: GOOD_KEY}) as response:
        first = next(response.iter_lines())

    assert json.loads(first[len("data: "):])["type"] == "sources"


# =========================
# What the deployment keeps paying for
# =========================

def test_indexing_stays_on_the_deployments_key(api, monkeypatch):
    """One collection, one embedding model: Chroma fixes the dimension per
    collection and embeddings.py refuses to open one built by another model."""
    calls = []
    monkeypatch.setattr(
        byok, "client_for",
        lambda key, settings, provider=None: calls.append(key) or FakeChatClient(),
    )

    api.post(
        "/upload", params={"user_id": "u1"},
        files={"file": ("policy.txt", TEXT, "text/plain")},
        headers={KEY_HEADER: GOOD_KEY},
    )

    assert calls == [], "an upload tried to embed on the caller's key"


# =========================
# The bot's /model command
# =========================

def _model_command(text, user_data=None, delete_raises=False):
    """Drive /model with no Telegram server in the loop."""
    import asyncio
    from types import SimpleNamespace

    import clients.telegram_bot as bot

    replies, sent = [], []

    async def reply_text(message, **kwargs):
        replies.append(message)

    async def send_message(message, **kwargs):
        sent.append(message)

    async def delete():
        if delete_raises:
            raise RuntimeError("no permission")

    message = SimpleNamespace(
        text=text, reply_text=reply_text, delete=delete,
        chat=SimpleNamespace(send_message=send_message),
    )
    context = SimpleNamespace(user_data=user_data if user_data is not None else {})
    asyncio.run(bot.model_command(
        SimpleNamespace(message=message, effective_user=SimpleNamespace(id=7)), context
    ))
    return replies, sent, context.user_data


def test_the_bot_remembers_the_key_and_model():
    _, sent, data = _model_command(f"/model gpt-4o {GOOD_KEY}")

    assert data["own_key"] == GOOD_KEY
    assert data["own_model"] == "gpt-4o"
    assert "gpt-4o" in sent[0]


def test_the_bot_deletes_the_message_carrying_the_key():
    """It lands in the chat history on the device and on Telegram's servers,
    which is the one bad thing about doing this over a chat."""
    _, sent, _ = _model_command(f"/model gpt-4o {GOOD_KEY}")

    assert "has been deleted" in sent[0]
    assert GOOD_KEY not in sent[0], "the confirmation repeated the key back"


def test_when_it_cannot_delete_it_says_so():
    """In a group the bot may lack the right. Silence would leave the user
    thinking it was handled."""
    _, sent, _ = _model_command(f"/model gpt-4o {GOOD_KEY}", delete_raises=True)

    assert "delete it yourself" in sent[0]


def test_the_bot_refuses_a_model_without_a_key():
    replies, sent, data = _model_command("/model gpt-4o")

    assert "needs your own API key" in replies[0]
    assert data == {}
    assert sent == []


def test_the_bot_forgets_on_request():
    replies, _, data = _model_command(
        "/model reset", user_data={"own_key": GOOD_KEY, "own_model": "gpt-4o"}
    )

    assert data == {}
    assert "Forgotten" in replies[0]


def test_asking_without_arguments_never_shows_the_key_back():
    replies, _, _ = _model_command(
        "/model", user_data={"own_key": GOOD_KEY, "own_model": "gpt-4o"}
    )

    assert GOOD_KEY not in replies[0]
    assert "gpt-4o" in replies[0]
    assert "never stored" in replies[0]


def test_the_bot_sends_the_key_with_a_question():
    from types import SimpleNamespace

    from clients.telegram_bot import asking_headers

    headers = asking_headers(
        SimpleNamespace(user_data={"own_key": GOOD_KEY, "own_model": "gpt-4o"})
    )

    assert headers[KEY_HEADER] == GOOD_KEY
    assert headers[MODEL_HEADER] == "gpt-4o"


def test_a_chat_that_set_nothing_sends_nothing():
    from types import SimpleNamespace

    from clients.telegram_bot import asking_headers

    headers = asking_headers(SimpleNamespace(user_data={}))

    assert KEY_HEADER not in headers
    assert MODEL_HEADER not in headers


def test_the_bot_never_sends_a_model_without_a_key():
    from types import SimpleNamespace

    from clients.telegram_bot import asking_headers

    headers = asking_headers(SimpleNamespace(user_data={"own_model": "gpt-4o"}))

    assert MODEL_HEADER not in headers


def test_the_bot_actually_sends_the_headers_when_asked_a_question():
    """asking_headers is tested as a pure function; this is the wiring. Passing
    BACKEND_HEADERS here instead would silently answer on the operator's key."""
    import asyncio
    from types import SimpleNamespace

    import clients.telegram_bot as bot

    seen = {}

    class FakeResponse:
        status_code = 200
        headers = {}

        def json(self):
            return {"answer": "28 days.", "sources": []}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            seen["headers"] = headers or {}
            return FakeResponse()

    class FakeMessage:
        text = "How many leave days?"

        def __init__(self):
            self.chat = SimpleNamespace(send_action=self._noop)

        async def _noop(self, *args, **kwargs):
            return None

        async def reply_html(self, text, **kwargs):
            return SimpleNamespace(text=text)

        async def reply_text(self, text, **kwargs):
            return SimpleNamespace(text=text)

    import pytest as _pytest

    monkeypatch = _pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(bot.httpx, "AsyncClient", FakeClient)
        update = SimpleNamespace(
            message=FakeMessage(), effective_user=SimpleNamespace(id=7)
        )
        context = SimpleNamespace(
            user_data={"own_key": GOOD_KEY, "own_model": "gpt-4o"}
        )
        asyncio.run(bot.handle_message(update, context))
    finally:
        monkeypatch.undo()

    assert seen["headers"][KEY_HEADER] == GOOD_KEY
    assert seen["headers"][MODEL_HEADER] == "gpt-4o"


# =========================
# Choosing a provider
# =========================
# Every provider here answers OpenAI's chat completions protocol, which is why
# one client library reaches all of them. The URLs were checked by asking each
# one with a deliberately invalid key.

def test_the_offered_providers_are_the_ones_documented():
    from app.byok import PROVIDERS

    assert set(PROVIDERS) == {
        "openai", "anthropic", "gemini", "deepseek", "kimi", "kimi-cn"
    }


def test_openai_means_the_deployments_own_endpoint(tmp_path):
    """Named explicitly so a caller can say "the usual one" after another
    provider was set, and so an operator's gateway is not overridden."""
    from app.byok import PROVIDERS
    from tests.conftest import make_settings

    assert PROVIDERS["openai"] is None

    settings = make_settings(tmp_path, openai_base_url="http://gw:9/v1")
    client = byok.client_for(GOOD_KEY, settings, "openai")
    try:
        assert str(client.base_url).startswith("http://gw:9/v1")
    finally:
        client.close()


@pytest.mark.parametrize("provider,host", [
    ("anthropic", "api.anthropic.com"),
    ("gemini", "generativelanguage.googleapis.com"),
    ("deepseek", "api.deepseek.com"),
    ("kimi", "api.moonshot.ai"),
    ("kimi-cn", "api.moonshot.cn"),
])
def test_each_provider_gets_its_own_endpoint(tmp_path, provider, host):
    from tests.conftest import make_settings

    client = byok.client_for(GOOD_KEY, make_settings(tmp_path), provider)
    try:
        assert host in str(client.base_url)
    finally:
        client.close()


def test_a_providers_url_beats_the_deployments(tmp_path):
    """Otherwise naming Anthropic on a deployment configured for a gateway
    would quietly send an Anthropic key to the gateway."""
    from tests.conftest import make_settings

    settings = make_settings(tmp_path, openai_base_url="http://gw:9/v1")
    client = byok.client_for(GOOD_KEY, settings, "deepseek")
    try:
        assert "api.deepseek.com" in str(client.base_url)
    finally:
        client.close()


def test_the_caller_still_cannot_name_a_url():
    """The whole reason a provider is a name and not an address: a URL from the
    request would make this backend fetch whatever it is handed."""
    with pytest.raises(byok.BringYourOwnKeyError, match="Unknown or disallowed"):
        byok.wanted({
            KEY_HEADER: GOOD_KEY,
            byok.PROVIDER_HEADER: "http://169.254.169.254/latest/meta-data",
        })


def test_a_provider_without_a_key_is_refused():
    with pytest.raises(byok.BringYourOwnKeyError, match="needs your own API key"):
        byok.wanted({byok.PROVIDER_HEADER: "anthropic"})


def test_provider_names_are_case_insensitive():
    assert byok.wanted({KEY_HEADER: GOOD_KEY, byok.PROVIDER_HEADER: "Anthropic"})[2] == "anthropic"


def test_an_operator_can_narrow_the_list(tmp_path):
    """Outbound connections are the operator's business: a deployment behind
    egress rules decides where its backend may talk."""
    from tests.conftest import make_settings

    settings = make_settings(tmp_path, allowed_model_providers="openai, deepseek")

    assert byok.allowed_providers(settings) == ["openai", "deepseek"]
    with pytest.raises(byok.BringYourOwnKeyError, match="This deployment offers"):
        byok.wanted({KEY_HEADER: GOOD_KEY, byok.PROVIDER_HEADER: "kimi"}, settings)


def test_an_empty_list_means_every_provider(tmp_path):
    from app.byok import PROVIDERS
    from tests.conftest import make_settings

    assert byok.allowed_providers(make_settings(tmp_path)) == list(PROVIDERS)


def test_the_refusal_names_what_is_on_offer(tmp_path):
    from tests.conftest import make_settings

    settings = make_settings(tmp_path, allowed_model_providers="gemini")

    with pytest.raises(byok.BringYourOwnKeyError) as exc:
        byok.wanted({KEY_HEADER: GOOD_KEY, byok.PROVIDER_HEADER: "anthropic"}, settings)

    assert "gemini" in str(exc.value)


def test_a_provider_reaches_the_client_through_the_api(api, monkeypatch):
    _index(api)
    seen = {}

    def watching(key, settings, provider=None):
        seen["provider"] = provider
        return FakeChatClient()

    monkeypatch.setattr(byok, "client_for", watching)

    _ask(api, {KEY_HEADER: GOOD_KEY, byok.PROVIDER_HEADER: "gemini",
               MODEL_HEADER: "gemini-2.0-flash"})

    assert seen["provider"] == "gemini"


def test_an_unknown_provider_is_a_400(api):
    _index(api)

    response = _ask(api, {KEY_HEADER: GOOD_KEY, byok.PROVIDER_HEADER: "nonesuch"})

    assert response.status_code == 400
    assert "anthropic" in response.json()["detail"]


# =========================
# Providers disagree about refusals
# =========================

def test_a_gemini_style_400_is_still_the_callers_problem(api, monkeypatch):
    """OpenAI, Anthropic, DeepSeek and Moonshot answer a bad key with 401;
    Gemini answers 400. Checked against all five with an invalid key. Without
    this branch a Gemini typo fell through to "Query failed", which blames the
    operator for the caller's mistake."""
    from openai import BadRequestError

    _index(api)
    monkeypatch.setattr(
        byok, "client_for",
        lambda key, settings, provider=None: Refusing(_openai_error(BadRequestError, 400)),
    )

    response = _ask(api, {KEY_HEADER: GOOD_KEY, byok.PROVIDER_HEADER: "gemini"})

    assert response.status_code == 400
    assert "HTTP 400" in response.json()["detail"]
    assert "model name" in response.json()["detail"]


def test_an_empty_balance_says_so(api, monkeypatch):
    """DeepSeek answers 402 when an account runs dry."""
    from openai import APIStatusError

    _index(api)
    monkeypatch.setattr(
        byok, "client_for",
        lambda key, settings, provider=None: Refusing(_openai_error(APIStatusError, 402)),
    )

    assert "out of balance" in _ask(api, {KEY_HEADER: GOOD_KEY}).json()["detail"]


def test_a_failure_with_no_status_is_not_blamed_on_the_caller():
    """A connection reset or a timeout says nothing about their key."""
    assert byok.describe_upstream_refusal(RuntimeError("connection reset")) is None


def test_the_bot_takes_a_provider_as_a_third_word():
    """The provider list is a closed set, so "gemini gemini-2.0-flash <key>"
    cannot be mistaken for a model called "gemini"."""
    _, sent, data = _model_command(f"/model gemini gemini-2.0-flash {GOOD_KEY}")

    assert data["own_provider"] == "gemini"
    assert data["own_model"] == "gemini-2.0-flash"
    assert data["own_key"] == GOOD_KEY
    assert "gemini" in sent[0]


def test_two_words_still_mean_model_and_key():
    _, _, data = _model_command(f"/model gpt-4o {GOOD_KEY}")

    assert data["own_model"] == "gpt-4o"
    assert "own_provider" not in data


def test_naming_a_model_without_a_provider_forgets_the_last_one():
    """Otherwise a switch back to OpenAI would keep sending an OpenAI key to
    Anthropic."""
    _, _, data = _model_command(
        f"/model gpt-4o {GOOD_KEY}",
        user_data={"own_provider": "anthropic", "own_model": "claude", "own_key": "old"},
    )

    assert "own_provider" not in data


def test_reset_forgets_the_provider_too():
    _, _, data = _model_command(
        "/model reset",
        user_data={"own_provider": "kimi", "own_model": "k", "own_key": GOOD_KEY},
    )

    assert data == {}


def test_the_bot_lists_the_providers_it_can_reach():
    replies, _, _ = _model_command("/model")

    for provider in ("anthropic", "gemini", "deepseek", "kimi"):
        assert provider in replies[0]


def test_the_bot_sends_the_provider_with_a_question():
    from types import SimpleNamespace

    from clients.telegram_bot import asking_headers

    headers = asking_headers(SimpleNamespace(user_data={
        "own_key": GOOD_KEY, "own_model": "claude-sonnet-4", "own_provider": "anthropic",
    }))

    assert headers[byok.PROVIDER_HEADER] == "anthropic"


def test_the_bot_sends_no_provider_when_none_was_chosen():
    from types import SimpleNamespace

    from clients.telegram_bot import asking_headers

    headers = asking_headers(SimpleNamespace(user_data={"own_key": GOOD_KEY}))

    assert byok.PROVIDER_HEADER not in headers


def test_the_deployments_allowlist_is_enforced_by_the_api(tmp_path, fake_openai_embeddings):
    """Not just by byok.wanted when a test calls it directly: the handler has
    to hand the settings over, or the operator's list is decoration."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import TEST_API_KEY, make_settings

    settings = make_settings(tmp_path, allowed_model_providers="openai")
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        app.state.rag_chain.client = FakeChatClient()

        response = client.post("/query", json={
            "question": "q", "language": "Auto", "user_id": "u1",
        }, headers={KEY_HEADER: GOOD_KEY, byok.PROVIDER_HEADER: "gemini"})

    assert response.status_code == 400
    assert "This deployment offers: openai" in response.json()["detail"]
