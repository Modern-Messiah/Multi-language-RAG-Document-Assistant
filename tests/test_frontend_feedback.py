"""The web UI's rating buttons, exercised by running the app.

The frontend had no tests at all: the only way to know whether a button worked
was to click it. Streamlit ships a headless harness, so at least the path that
sends a rating can be driven here - script run, button press, rerun, and what
went over the wire.

`requests` is replaced wholesale, so nothing reaches a backend and the assertions
are about what the UI *does*, not about what a server would answer.
"""
import json

import pytest

APP = "frontend/streamlit_app.py"

ANSWER = "Twenty eight calendar days."
SOURCES = [{"id": 1, "source": "policy.docx", "preview": "Leave policy ..."}]
REQUEST_ID = "req0123456789ab"


class FakeResponse:
    def __init__(self, payload=None, status_code=200, lines=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self._lines = lines or []
        self.headers = headers or {}

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _sse(event):
    return f"data: {json.dumps(event)}"


class FakeRequests:
    """Stands in for the `requests` module, recording every call."""

    RequestException = Exception

    def __init__(self):
        self.posted = []

    def get(self, url, **kwargs):
        if url.endswith("/documents"):
            return FakeResponse({
                "documents": [{
                    "file_hash": "a" * 16, "source": "policy.docx",
                    "chunks": 3, "type": "docx",
                }],
                "total_chunks": 3,
                "quota": {"documents": 1, "max_documents": 200,
                          "bytes": 2048, "max_bytes": 1024 ** 3},
            })
        return FakeResponse({"status": "ok"})

    def post(self, url, **kwargs):
        self.posted.append((url, kwargs.get("json")))
        if url.endswith("/query/stream"):
            return FakeResponse(
                lines=[_sse({"type": "sources", "sources": SOURCES}),
                       _sse({"type": "token", "text": ANSWER})],
                headers={"X-Request-ID": REQUEST_ID},
            )
        return FakeResponse({"message": "Thanks - recorded."})

    def delete(self, url, **kwargs):
        return FakeResponse({"message": "Document deleted"})


@pytest.fixture()
def fake_backend(monkeypatch):
    """Replace the transport on the `requests` module itself.

    AppTest executes the script as its own module each run, so patching an
    already-imported `frontend.streamlit_app` has no effect on what runs - the
    script's `import requests` resolves to the one module object, which is the
    thing worth patching.
    """
    import requests

    fake = FakeRequests()
    monkeypatch.setattr(requests, "get", fake.get)
    monkeypatch.setattr(requests, "post", fake.post)
    monkeypatch.setattr(requests, "delete", fake.delete)
    return fake


@pytest.fixture()
def app(fake_backend):
    """A run of the real script with the network replaced."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.fake = fake_backend
    return at


def _ask(at, question="How many leave days?"):
    at.chat_input[0].set_value(question).run()
    return at


# =========================
# The buttons appear
# =========================

def test_the_app_runs_at_all(app):
    """AppTest reports an uncaught exception as an error element rather than
    raising, so this is the check that the script did not blow up."""
    assert not app.exception, [str(e) for e in app.exception]


def test_an_answer_arrives_with_two_buttons(app):
    _ask(app)

    labels = [b.label for b in app.button]
    assert "👍" in labels
    assert "👎" in labels


def test_the_buttons_come_with_the_answer_not_a_rerun_later(app):
    """Rendered inline after streaming: buttons that only appeared on the next
    interaction would be asking about the previous answer."""
    _ask(app)

    assert any(ANSWER in md.value for md in app.markdown)
    assert [b for b in app.button if b.label == "👍"]


def test_there_are_no_buttons_before_anything_is_asked(app):
    assert not [b for b in app.button if b.label in ("👍", "👎")]


# =========================
# Pressing one
# =========================

def _press(at, label):
    button = next(b for b in at.button if b.label == label)
    button.click().run()
    return at


def test_pressing_a_button_posts_the_rating(app):
    _ask(app)

    _press(app, "👎")

    ratings = [body for url, body in app.fake.posted if url.endswith("/feedback")]
    assert len(ratings) == 1, ratings
    assert ratings[0]["rating"] == "down"
    assert ratings[0]["client"] == "web"


def test_the_rating_carries_the_exchange_it_refers_to(app):
    _ask(app, "How many leave days?")

    _press(app, "👍")

    body = [b for url, b in app.fake.posted if url.endswith("/feedback")][0]
    assert body["question"] == "How many leave days?"
    assert body["answer"] == ANSWER
    assert body["sources"] == ["policy.docx"]


def test_the_rating_carries_the_request_id_from_the_header(app):
    """The thread back to the log lines that produced the answer."""
    _ask(app)

    _press(app, "👍")

    body = [b for url, b in app.fake.posted if url.endswith("/feedback")][0]
    assert body["request_id"] == REQUEST_ID


def test_the_buttons_give_way_to_an_acknowledgement(app):
    """Without the rerun the user sees no change at all until they do something
    else, and can rate the same answer repeatedly."""
    _ask(app)

    _press(app, "👍")

    assert not [b for b in app.button if b.label in ("👍", "👎")]
    assert any("recorded" in c.value.lower() for c in app.caption)


def test_only_the_answer_that_was_rated_loses_its_buttons(app):
    _ask(app, "First question?")
    _press(app, "👍")
    _ask(app, "Second question?")

    assert [b for b in app.button if b.label == "👍"], (
        "the new answer should still be rateable"
    )
    ratings = [b for url, b in app.fake.posted if url.endswith("/feedback")]
    assert len(ratings) == 1


# =========================
# Switched off
# =========================

def test_no_buttons_when_collection_is_disabled(monkeypatch, fake_backend):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FEEDBACK_ENABLED", "false")

    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.chat_input[0].set_value("How many leave days?").run()

    assert not at.exception, [str(e) for e in at.exception]
    assert not [b for b in at.button if b.label in ("👍", "👎")]


# =========================
# When the rating cannot be sent
# =========================

def test_a_rejected_rating_says_so_and_keeps_the_buttons(app, monkeypatch):
    _ask(app)

    def refuse(url, **kwargs):
        return FakeResponse({"detail": "Feedback storage is full."}, status_code=507)

    import requests

    monkeypatch.setattr(requests, "post", refuse)

    _press(app, "👍")

    assert [b for b in app.button if b.label == "👍"], "the user cannot try again"
    assert any("full" in w.value.lower() for w in app.warning), [w.value for w in app.warning]


def test_a_backend_that_is_gone_does_not_break_the_page(app, monkeypatch):
    _ask(app)

    import requests

    def explode(url, **kwargs):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(requests, "post", explode)

    _press(app, "👎")

    assert not app.exception, [str(e) for e in app.exception]
    assert app.warning, "the user was told nothing"


# =========================
# What the page sends with a question
# =========================

def test_the_transcript_rides_along_with_the_next_question(app):
    """Not a feedback concern, but the same session state now carries the
    request id, and a typo there would silently drop the history."""
    _ask(app, "First question?")
    _ask(app, "Second question?")

    queries = [b for url, b in app.fake.posted if url.endswith("/query/stream")]
    assert queries[1]["history"] == [
        {"question": "First question?", "answer": ANSWER}
    ]


def test_the_history_sent_holds_only_question_and_answer(app):
    """The backend rejects unknown keys in a history turn, and request_id now
    lives in the same dictionary."""
    _ask(app, "First question?")
    _ask(app, "Second question?")

    queries = [b for url, b in app.fake.posted if url.endswith("/query/stream")]
    assert set(queries[1]["history"][0]) == {"question", "answer"}


def test_the_session_identity_is_the_web_namespace(app):
    _ask(app)

    body = [b for url, b in app.fake.posted if url.endswith("/query/stream")][0]
    assert body["user_id"].startswith("web-")


def test_the_transport_patch_does_not_outlive_the_test():
    """monkeypatch undoes it, and the rest of the suite relies on the socket
    guard rather than on a patched `requests`."""
    import requests

    assert requests.post.__module__.startswith("requests")


# =========================
# Quotas in the sidebar
# =========================

def test_the_limits_box_shows_the_real_usage(app):
    """It used to promise "Any number of documents" and "TXT, PDF", both false."""
    boxes = [i.value for i in app.info]

    assert any("1 of 200 documents, 2 KB of 1 GB" in text for text in boxes), boxes
    assert not any("Any number of documents" in text for text in boxes)
    assert any("DOCX" in text for text in boxes), boxes


def test_an_older_backend_without_a_quota_block_still_renders(monkeypatch):
    import requests
    from streamlit.testing.v1 import AppTest

    fake = FakeRequests()

    def old_listing(url, **kwargs):
        if url.endswith("/documents"):
            return FakeResponse({"documents": [], "total_chunks": 0})
        return fake.get(url, **kwargs)

    monkeypatch.setattr(requests, "get", old_listing)
    monkeypatch.setattr(requests, "post", fake.post)
    monkeypatch.setattr(requests, "delete", fake.delete)

    at = AppTest.from_file(APP, default_timeout=30)
    at.run()

    assert not at.exception, [str(e) for e in at.exception]


def test_deleting_a_document_lets_a_refused_file_be_retried(app):
    """The 413 message says to remove a document. Before this, the app remembered
    the refusal and silently skipped the same file forever - the remedy worked on
    the backend and failed in the UI."""
    app.session_state["failed_files"] = {
        ("big.pdf", "0123456789abcdef"): "Document limit reached: you hold 200 of 200."
    }
    app.run()
    assert any("big.pdf" in e.value for e in app.error), "the refusal was not shown"

    app.button(key="del_" + "a" * 16).click().run()

    state = app.session_state
    assert "failed_files" not in state or not state["failed_files"], "the refusal was kept"
    assert not any("big.pdf" in e.value for e in app.error)
