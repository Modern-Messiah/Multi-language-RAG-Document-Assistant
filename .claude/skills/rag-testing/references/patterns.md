# Test patterns

Copyable skeletons for each kind of test in this suite. Every block runs as written inside a
module in `tests/` (it was run that way). Read the section you need, then open the named test
file for the full set of cases.

Contents:

1. An endpoint through `api`
2. A fixture with non-default settings
3. RAGChain with a fake store
4. Rankings that mean something
5. The streaming endpoint
6. The web UI through AppTest
7. A Telegram handler
8. DOCX and PDF fixtures built in the test
9. Forcing a race with a barrier
10. Logs with caplog

## 1. An endpoint through `api`

`api` is a TestClient whose lifespan has run, with X-API-Key set and FakeChatClient installed.
Copy from `tests/test_query_e2e.py`.

```python
from tests.conftest import FAKE_ANSWER


def test_upload_then_query(api):
    response = api.post(
        "/upload", params={"user_id": "u1"},
        files={"file": ("policy.txt", b"Annual leave is 28 days.", "text/plain")},
    )
    assert response.status_code == 200, response.text

    response = api.post("/query", json={"question": "How long is leave?", "user_id": "u1"})

    assert response.status_code == 200, response.text
    assert response.json()["answer"] == FAKE_ANSWER
    assert "Annual leave" in api.chat.calls[0]["messages"][1]["content"]
```

- Failure paths: replace the component on `api.app_state` with a double that takes
  `*args, **kwargs` and raises, e.g. `monkeypatch.setattr(api.app_state.rag_chain, "ask", boom)`
  (`tests/test_resilience.py`, `tests/test_api.py`).
- A caller's own key: `monkeypatch.setattr(byok, "client_for", lambda key, settings,
  provider=None: FakeChatClient())` with `from app import byok` (`tests/test_byok.py`).
- A blocking handler must stay a plain `def`: find it in `create_app(...).routes` and assert
  `not inspect.iscoroutinefunction(route.endpoint)` (`tests/test_feedback.py`).

## 2. A fixture with non-default settings

The lifespan builds `app.state.rag_chain`, so install the fake chat client inside the `with`
block. Copy from `limited` in `tests/test_quotas.py`.

```python
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import TEST_API_KEY, FakeChatClient, make_settings


@pytest.fixture()
def tight(tmp_path, fake_openai_embeddings):
    settings = make_settings(tmp_path, max_documents_per_user=1)
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:  # runs the lifespan
        client.headers["X-API-Key"] = TEST_API_KEY
        app.state.rag_chain.client = FakeChatClient()  # after startup built the chain
        client.app_state = app.state
        yield client


def test_second_document_is_refused(tight):
    for name in ("a.txt", "b.txt"):
        last = tight.post("/upload", params={"user_id": "u1"},
                          files={"file": (name, name.encode() * 20, "text/plain")})
    assert last.status_code == 413, last.text
```

Without the `with`, startup never runs: there is no `rag_chain` and GET /ready answers 503
(`test_ready_is_503_before_startup_finishes` in `tests/test_observability.py` does that on
purpose).

## 3. RAGChain with a fake store

The store needs `similarity_search(query, k, filter)`; add `similarity_search_with_score` and a
`_collection.metadata` dict only to exercise the scored path (`ScoredStore` in
`tests/test_relevance.py`). Copy from `tests/test_chain.py`.

```python
from langchain.schema import Document

from app.rag.chain import RAGChain
from tests.conftest import FakeChatClient


class Store:
    def __init__(self, docs):
        self.docs, self.calls = docs, []

    def similarity_search(self, query, k=4, filter=None, **kwargs):
        self.calls.append({"query": query, "k": k, "filter": filter})
        return self.docs


def test_chain_scopes_and_cites():
    store = Store([Document(page_content="Leave is 28 days.", metadata={"source": "p.txt"})])
    chat = FakeChatClient(answer="28 days [1].")
    chain = RAGChain(store, client=chat, top_k=3, metadata_filtering_enabled=False)

    result = chain.ask("How long is leave in 2020?", user_id="u1")

    assert store.calls[0]["filter"] == {"user_id": "u1"}
    assert result["answer"] == "28 days ."
    assert [s["source"] for s in result["sources"]] == ["p.txt"]
```

Arguments left as None fall back to os.getenv and then to the code defaults, so automatic
metadata filtering is on: without `metadata_filtering_enabled=False` the question above (it
names a year) is searched with `{"$and": [{"user_id": "u1"}, {"year": 2020}]}`. Hybrid search
stays off unless you pass `bm25_index`. `client=object()` is enough when the test never reaches
the model.

## 4. Rankings that mean something

The autouse fake vectors are random per text, so an order or a similarity computed from them
means nothing. For ranking assertions build a real ChromaDB collection with a deterministic
bag-of-words embedder:

- `EmbeddingsManager(persist_directory=str(tmp_path / "chroma"), embedding_fn=BagOfWords(vocab))`,
  then `store = manager.get_vectorstore("documents")` and `manager.add_documents(docs, ids=...)`
  with `user_id` in every document's metadata.
- Query through `RAGChain(store, client=object(), top_k=..., relevance_threshold=...)._retrieve(
  question, {"user_id": "u1"})`; for MMR also pass `embeddings_manager=manager` and an
  `mmr_lambda` below 1.0.
- Copy the embedder and fixture from `indexed` in `tests/test_retrieval_quality.py` or `crowded`
  in `tests/test_mmr.py`. Always pass `persist_directory`: its default is the repository's
  `./data/chroma_db`.

## 5. The streaming endpoint

Copy from `tests/test_streaming.py` (`_sse_events`).

```python
import json


def test_stream_events(api):
    api.post("/upload", params={"user_id": "u1"},
             files={"file": ("a.txt", b"RAG grounds answers.", "text/plain")})
    with api.stream("POST", "/query/stream",
                    json={"question": "what is rag", "user_id": "u1"}) as response:
        assert response.status_code == 200
        events = [json.loads(line[6:]) for line in response.iter_lines()
                  if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
```

FakeChatClient streams its answer in two pieces, hence two token events.

## 6. The web UI through AppTest

AppTest executes `frontend/streamlit_app.py` as a fresh module on every run, so patch the
`requests` module itself; patching an imported `frontend.streamlit_app` changes nothing. The path
is relative to the working directory: run pytest from the repository root. Copy `FakeRequests`
and `fake_backend` from `tests/test_frontend_feedback.py`.

```python
class FakeResponse:
    def __init__(self, payload=None, status_code=200, lines=(), headers=None):
        self.status_code, self._payload = status_code, payload or {}
        self._lines, self.headers = list(lines), headers or {}

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_web_ui_sends_the_question(monkeypatch):
    import requests
    from streamlit.testing.v1 import AppTest

    posted = []

    def fake_post(url, **kwargs):
        posted.append((url, kwargs.get("json")))
        return FakeResponse(lines=['data: {"type": "token", "text": "Hi"}'])

    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResponse({"documents": []}))
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "delete", lambda url, **kw: FakeResponse())

    at = AppTest.from_file("frontend/streamlit_app.py", default_timeout=30).run()
    at.chat_input[0].set_value("How many leave days?").run()

    assert not at.exception, [str(e) for e in at.exception]
    assert any(url.endswith("/query/stream") for url, _ in posted)
```

- AppTest records a script exception as an element instead of raising, so always assert
  `not at.exception`.
- Drive it with `button.click().run()` and `at.session_state["own_key"] = ...` followed by
  `at.run()`.
- The script re-reads the environment on every run, so `monkeypatch.setenv("FEEDBACK_ENABLED",
  "false")` before `AppTest.from_file(...).run()` works.

## 7. A Telegram handler

No pytest-asyncio here: build the update and context from SimpleNamespace with async methods,
replace the transport, and drive the coroutine with `asyncio.run` inside a plain test. Copy from
`tests/test_feedback.py` (`_ask`, `_press`) or `_model_command` in `tests/test_byok.py`.

```python
import asyncio
from types import SimpleNamespace


def test_bot_answers(monkeypatch):
    import clients.telegram_bot as bot

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kwargs):
            return SimpleNamespace(status_code=200, headers={},
                                   json=lambda: {"answer": "28 days.", "sources": []})

    replies = []

    async def reply_html(text, **kwargs):
        replies.append(text)
        return SimpleNamespace(text=text)

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(bot.httpx, "AsyncClient", FakeClient)
    message = SimpleNamespace(text="How many leave days?", reply_html=reply_html,
                              reply_text=reply_html, chat=SimpleNamespace(send_action=noop))
    update = SimpleNamespace(message=message, effective_user=SimpleNamespace(id=7))

    asyncio.run(bot.handle_message(update, SimpleNamespace(user_data={})))

    assert any("28 days." in r for r in replies), replies
```

The bot reads BACKEND_URL, BACKEND_HEADERS, FEEDBACK_ENABLED and TELEGRAM_BOT_TOKEN once, at
import: patch them on the module (`monkeypatch.setattr(bot, "FEEDBACK_ENABLED", False)`), not
with setenv. Pure helpers (`split_for_telegram`, `format_document_list`, `asking_headers`) are
called directly.

## 8. DOCX and PDF fixtures built in the test

No binary files in the repository: build them in the test. Copy `_docx` from
`tests/test_docx_markdown.py`; for a PDF, `from tests.test_upload_errors import _pdf` returns the
bytes of a minimal PDF that pypdf can read.

```python
from docx import Document as Docx


def test_docx_upload(api, tmp_path):
    document = Docx()
    document.add_paragraph("Leave policy.")
    table = document.add_table(rows=2, cols=2)
    for (row, col), text in {(0, 0): "Role", (0, 1): "Days",
                             (1, 0): "Engineer", (1, 1): "28"}.items():
        table.cell(row, col).text = text
    path = tmp_path / "policy.docx"
    document.save(path)

    response = api.post("/upload", params={"user_id": "u1"},
                        files={"file": ("policy.docx", path.read_bytes(),
                                        "application/octet-stream")})

    assert response.status_code == 200, response.text
```

## 9. Forcing a race with a barrier

Two threads on one TestClient interleave only by luck. Wrap the call that sits between the check
and the write so each request waits for the other; with the lock in place the second never
arrives, the barrier times out and the first goes on. Copy from
`test_two_simultaneous_uploads_cannot_both_squeeze_in` in `tests/test_quotas.py`.

```python
import threading

from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import TEST_API_KEY, make_settings


def test_barrier(tmp_path, fake_openai_embeddings, monkeypatch):
    import app.main as main

    app = create_app(make_settings(tmp_path, max_documents_per_user=1))
    barrier = threading.Barrier(2)
    real = main._owner_usage

    def synchronized(*args, **kwargs):
        usage = real(*args, **kwargs)
        try:
            barrier.wait(timeout=1.5)
        except threading.BrokenBarrierError:
            pass  # the lock kept the other request out
        return usage

    monkeypatch.setattr(main, "_owner_usage", synchronized)
    codes = []
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["X-API-Key"] = TEST_API_KEY

        def upload(name):
            codes.append(client.post("/upload", params={"user_id": "u1"},
                                     files={"file": (name, name.encode() * 9,
                                                     "text/plain")}).status_code)

        threads = [threading.Thread(target=upload, args=(n,)) for n in ("a.txt", "b.txt")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert sorted(codes) == [200, 413], codes
```

- When forcing the interleaving would mean instrumenting the guarded code, assert that the lock
  is taken instead: wrap `api.app_state.upload_locks.for_owner` and record the owners
  (`test_clearing_waits_for_an_upload_in_flight` in `tests/test_sweep.py`).
- Check a lock is released with `lock.acquire(blocking=False)`, so a regression fails instead of
  hanging the suite (`test_a_failed_upload_releases_the_lock` in `tests/test_quotas.py`).

## 10. Logs with caplog

Modules under `app/` and `clients/` log through `logging.getLogger(__name__)`, so the logger
name is the module path: `app.main`, `app.rag.chain`, `app.observability`, `app.sweep`,
`clients.telegram_bot`. Name it in `caplog.at_level` so the level applies to that logger.

```python
import logging


def test_the_access_line(api, caplog):
    with caplog.at_level(logging.INFO, logger="app.observability"):
        api.get("/documents", params={"user_id": "u1"})

    assert "GET /documents -> 200" in caplog.records[0].getMessage()
```

To see the request id on records, add `RequestIdFilter` to `caplog.handler` (the `stamped`
fixture in `tests/test_observability.py`); caplog's handler does not get the app's filters.
