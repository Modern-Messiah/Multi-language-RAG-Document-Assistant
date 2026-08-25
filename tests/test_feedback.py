"""Rating an answer, and what the rating is for.

The golden set in evaluation/golden.py is questions I invented, so it measures
my guesses about what people ask. A thumbs-down is a real question that got a
real bad answer, carrying the documents behind it and the request id that finds
it in the log. These tests cover the path from the button to a golden-case stub.
"""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from app.feedback import (
    FEEDBACK_FILENAME,
    FeedbackStorageFull,
    FeedbackStore,
    golden_case_stub,
    read_records,
    store_from_settings,
    summarise,
    unanswered_questions,
)
from tests.conftest import make_settings

RATING = {
    "rating": "down",
    "user_id": "u1",
    "question": "How many leave days does an engineer get?",
    "answer": "Thirty five.",
    "sources": ["policy.docx"],
    "request_id": "abc123def456ab78",
    "client": "web",
}


def _store(tmp_path, max_bytes=10_000):
    return FeedbackStore(tmp_path / "feedback", max_bytes=max_bytes)


def _lines(store):
    return store.path.read_text(encoding="utf-8").splitlines()


# =========================
# The store
# =========================

def test_a_rating_becomes_one_line(tmp_path):
    store = _store(tmp_path)

    store.record(**RATING)

    assert len(_lines(store)) == 1
    assert json.loads(_lines(store)[0])["question"] == RATING["question"]


def test_ratings_accumulate(tmp_path):
    store = _store(tmp_path)

    for _ in range(3):
        store.record(**RATING)

    assert len(_lines(store)) == 3


def test_the_server_stamps_the_time(tmp_path):
    """A client's clock is only useful for lining up against a client's log."""
    store = _store(tmp_path)

    record = store.record(**RATING)

    assert record["at"].endswith("+00:00"), record["at"]
    assert "at" == next(iter(record)), "the timestamp should lead the record"


def test_a_question_with_a_newline_stays_one_line(tmp_path):
    """Otherwise one rating becomes two half-records and the file stops being
    greppable."""
    store = _store(tmp_path)

    store.record(**{**RATING, "question": "First line\nSecond line"})

    assert len(_lines(store)) == 1
    assert json.loads(_lines(store)[0])["question"] == "First line\nSecond line"


def test_cyrillic_is_written_readably(tmp_path):
    """The file is read by a person, and \\u0421 for every letter is not
    readable."""
    store = _store(tmp_path)

    store.record(**{**RATING, "question": "Сколько дней отпуска?"})

    assert "Сколько дней отпуска?" in store.path.read_text(encoding="utf-8")


def test_nothing_is_created_until_something_is_rated(tmp_path):
    store = _store(tmp_path)

    assert not store.path.parent.exists()


def test_concurrent_ratings_do_not_interleave(tmp_path):
    """Handlers run in a threadpool, so two ratings really can land at once."""
    store = _store(tmp_path)

    def rate(index):
        store.record(**{**RATING, "question": f"question {index}"})

    threads = [threading.Thread(target=rate, args=(i,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = _lines(store)
    assert len(lines) == 20
    questions = {json.loads(line)["question"] for line in lines}
    assert len(questions) == 20, "a line was lost or two were spliced together"


def test_the_cap_refuses_instead_of_filling_the_volume(tmp_path):
    """The vector store lives on the same volume.

    The cap is derived from one real record rather than guessed, so the test
    does not depend on how wide the sample rating happens to be.
    """
    store = _store(tmp_path)
    store.record(**RATING)
    store.max_bytes = store.path.stat().st_size

    with pytest.raises(FeedbackStorageFull):
        store.record(**RATING)

    assert len(_lines(store)) == 1


def test_the_incoming_record_counts_towards_the_cap(tmp_path):
    """Checking only the current size would let the file overshoot by one
    record, which for a cap protecting a shared volume is the wrong way to be
    wrong."""
    store = _store(tmp_path, max_bytes=1)

    with pytest.raises(FeedbackStorageFull):
        store.record(**RATING)

    assert not store.path.exists()


# =========================
# Reading it back
# =========================

def test_records_come_back(tmp_path):
    store = _store(tmp_path)
    store.record(**RATING)
    store.record(**{**RATING, "rating": "up"})

    assert [r["rating"] for r in store.read_all()] == ["down", "up"]


def test_a_missing_file_reads_as_empty(tmp_path):
    assert read_records(tmp_path / "nothing.jsonl") == []


def test_a_truncated_last_line_does_not_hide_the_rest(tmp_path, caplog):
    """Which is what a crash mid-append leaves behind."""
    store = _store(tmp_path)
    store.record(**RATING)
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write('{"rating": "do')

    assert len(store.read_all()) == 1
    assert any("unreadable" in r.getMessage() for r in caplog.records)


def test_blank_lines_are_skipped_quietly(tmp_path, caplog):
    """A trailing newline is normal, not damage. json.loads would reject it
    either way, so the point of the check is that it produces no warning - the
    warning is reserved for a line that really was cut in half.
    """
    store = _store(tmp_path)
    store.record(**RATING)
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")

    assert len(store.read_all()) == 1
    assert not [r for r in caplog.records if "unreadable" in r.getMessage()]


# =========================
# Switched off
# =========================

def test_no_store_when_collection_is_disabled(tmp_path):
    settings = make_settings(tmp_path, feedback_enabled=False)

    assert store_from_settings(settings) is None


def test_a_store_when_it_is_enabled(tmp_path):
    settings = make_settings(tmp_path, feedback_enabled=True)

    store = store_from_settings(settings)

    assert store.path.name == FEEDBACK_FILENAME


def test_feedback_ships_enabled(tmp_path):
    """It records only on an explicit button press, so the default is on; the
    knob exists for an operator who may not keep questions at all."""
    from app.config import Settings

    assert Settings(_env_file=None, openai_api_key="k").feedback_enabled is True


# =========================
# Through the API
# =========================

def test_a_rating_is_accepted(api):
    response = api.post("/feedback", json=RATING)

    assert response.status_code == 200, response.text
    assert "recorded" in response.json()["message"].lower()


def test_the_rating_reaches_the_file(api):
    api.post("/feedback", json=RATING)

    records = read_records(api.app_state.feedback.path)

    assert len(records) == 1
    assert records[0]["rating"] == "down"
    assert records[0]["sources"] == ["policy.docx"]
    assert records[0]["request_id"] == RATING["request_id"]
    assert records[0]["client"] == "web"


def test_feedback_needs_the_api_key(api):
    response = api.post("/feedback", json=RATING, headers={"X-API-Key": "wrong"})

    assert response.status_code == 401


def test_a_client_cannot_backdate_a_rating(api):
    """`at` is not a field of the request, so a supplied one is dropped."""
    api.post("/feedback", json={**RATING, "at": "1999-01-01T00:00:00+00:00"})

    assert not read_records(api.app_state.feedback.path)[0]["at"].startswith("1999")


@pytest.mark.parametrize("bad", [
    {"rating": "sideways"},
    {"rating": ""},
    {"question": ""},
    {"user_id": "../etc"},
    {"request_id": "not a valid id"},
    {"question": "x" * 4001},
    {"answer": "x" * 8001},
    {"comment": "x" * 1001},
    {"client": "WEB"},
    {"sources": ["a"] * 21},
])
def test_a_malformed_rating_is_rejected(api, bad):
    response = api.post("/feedback", json={**RATING, **bad})

    assert response.status_code == 422, f"{bad} was accepted"
    assert not api.app_state.feedback.path.exists()


def test_a_rating_without_a_request_id_is_still_kept(api):
    """A client that did not keep the header still knows what was asked, and a
    rating recorded without the log thread beats one thrown away."""
    payload = {k: v for k, v in RATING.items() if k != "request_id"}

    assert api.post("/feedback", json=payload).status_code == 200
    assert read_records(api.app_state.feedback.path)[0]["request_id"] is None


def test_the_answer_is_optional(api):
    payload = {k: v for k, v in RATING.items() if k != "answer"}

    assert api.post("/feedback", json=payload).status_code == 200


def test_a_full_store_says_so_instead_of_pretending(api):
    api.app_state.feedback.max_bytes = 1

    response = api.post("/feedback", json=RATING)

    assert response.status_code == 507
    assert "rotate" in response.json()["detail"].lower()


def test_a_broken_store_is_a_503(api):
    def boom(**fields):
        raise OSError("read-only file system")

    api.app_state.feedback.record = boom

    assert api.post("/feedback", json=RATING).status_code == 503


def test_the_endpoint_is_dispatched_off_the_event_loop(tmp_path):
    """It appends to a file, which blocks."""
    import inspect

    from app.main import create_app

    app = create_app(make_settings(tmp_path))
    endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, "path", None) == "/feedback"
    )

    assert not inspect.iscoroutinefunction(endpoint)


# =========================
# Switched off, through the API
# =========================

@pytest.fixture()
def api_without_feedback(tmp_path, fake_openai_embeddings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import TEST_API_KEY

    settings = make_settings(tmp_path, feedback_enabled=False)
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        client.settings = settings
        yield client


def test_a_disabled_endpoint_says_it_does_not_exist(api_without_feedback):
    response = api_without_feedback.post("/feedback", json=RATING)

    assert response.status_code == 404


def test_nothing_is_written_when_collection_is_off(api_without_feedback):
    api_without_feedback.post("/feedback", json=RATING)

    assert not (api_without_feedback.settings.feedback_dir / FEEDBACK_FILENAME).exists()


def test_the_rest_of_the_api_still_works_with_feedback_off(api_without_feedback):
    assert api_without_feedback.get("/documents", params={"user_id": "u1"}).status_code == 200


# =========================
# What the ratings are for
# =========================

def _records(*specs):
    return [
        {"at": at, "rating": rating, "question": question, "sources": sources}
        for at, rating, question, sources in specs
    ]


# "policy.docx" sorts after "handbook.pdf" but is behind more bad answers, so a
# sort by name and a sort by count give different orders. With the two agreeing,
# sorting by name would have looked correct.
SAMPLE = _records(
    ("2026-08-01T10:00:00+00:00", "down", "VAT rate?", ["handbook.pdf", "policy.docx"]),
    ("2026-08-02T10:00:00+00:00", "up", "Who signs?", ["policy.docx"]),
    ("2026-08-03T10:00:00+00:00", "down", "Leave days?", ["policy.docx"]),
)


def test_the_summary_counts_both_ways():
    counts = summarise(SAMPLE)

    assert (counts["total"], counts["up"], counts["down"]) == (3, 1, 2)
    assert counts["down_rate"] == pytest.approx(0.667, abs=0.001)


def test_nothing_rated_is_not_a_zero_rate():
    """A negative rate of 0.0 reads as "nothing is wrong", which is not what no
    data means."""
    assert summarise([])["down_rate"] is None


def test_the_summary_names_the_documents_behind_bad_answers():
    by_source = summarise(SAMPLE)["down_by_source"]

    assert by_source == {"policy.docx": 2, "handbook.pdf": 1}
    assert list(by_source) == ["policy.docx", "handbook.pdf"], "worst should come first"


def test_an_upvote_does_not_blame_its_sources():
    assert "policy.docx" not in summarise(SAMPLE[1:2])["down_by_source"]


def test_a_rating_that_is_neither_is_ignored():
    counts = summarise(SAMPLE + [{"rating": None, "sources": ["x.pdf"]}])

    assert counts["total"] == 3


def test_the_worst_cases_come_first():
    picked = unanswered_questions(SAMPLE)

    assert [r["question"] for r in picked] == ["Leave days?", "VAT rate?"]


def test_upvotes_can_be_listed_too():
    assert [r["question"] for r in unanswered_questions(SAMPLE, "up")] == ["Who signs?"]


def test_a_stub_leaves_the_expectation_blank():
    """Guessing which document should have answered would turn one bad answer
    into a permanently wrong benchmark."""
    stub = golden_case_stub(SAMPLE[0])

    assert "expected_sources=[]" in stub
    assert "TODO" in stub


def test_a_stub_carries_the_question_and_what_was_retrieved():
    stub = golden_case_stub(SAMPLE[0])

    assert '"VAT rate?"' in stub
    assert '"handbook.pdf"' in stub


def test_a_stub_quotes_a_question_that_contains_a_quote():
    """It is pasted into Python, so a raw quote would not parse."""
    stub = golden_case_stub({"question": 'What is a "chunk"?'})

    assert '\\"chunk\\"' in stub


def test_a_stub_names_the_request_to_grep_for():
    stub = golden_case_stub({**SAMPLE[0], "request_id": "abc123def456ab78"})

    assert "abc123def456ab78" in stub


def test_a_stub_without_a_request_id_says_so():
    assert "request_id=-" in golden_case_stub(SAMPLE[0])


# =========================
# The report script
# =========================

def test_the_report_runs_over_a_real_file(tmp_path, capsys):
    from evaluation.from_feedback import main

    store = _store(tmp_path)
    store.record(**RATING)

    exit_code = main(["--input", str(store.path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "1 ratings: 0 up, 1 down" in output
    assert "policy.docx" in output
    assert "expected_sources=[]" in output


def test_the_report_says_when_there_is_nothing_to_read(tmp_path, capsys):
    from evaluation.from_feedback import main

    assert main(["--input", str(tmp_path / "absent.jsonl")]) == 1
    assert "No ratings file" in capsys.readouterr().err


def test_the_report_can_list_the_good_answers(tmp_path, capsys):
    from evaluation.from_feedback import main

    store = _store(tmp_path)
    store.record(**{**RATING, "rating": "up", "question": "Who signs?"})

    main(["--input", str(store.path), "--up"])

    assert "Who signs?" in capsys.readouterr().out


def test_the_report_admits_what_it_did_not_print(tmp_path, capsys):
    """A list that stops at the limit reads as "that is all of them"."""
    from evaluation.from_feedback import main

    store = _store(tmp_path)
    for index in range(4):
        store.record(**{**RATING, "question": f"question {index}"})

    main(["--input", str(store.path), "--limit", "2"])

    assert "2 more not shown" in capsys.readouterr().out


# =========================
# The clients
# =========================

def test_the_buttons_are_offered_by_default(monkeypatch):
    from clients.backend import feedback_enabled

    monkeypatch.delenv("FEEDBACK_ENABLED", raising=False)

    assert feedback_enabled() is True


@pytest.mark.parametrize("raw", ["false", "False", "0", "no", "off", "  FALSE  "])
def test_the_clients_read_the_same_knob_as_the_backend(monkeypatch, raw):
    from clients.backend import feedback_enabled

    monkeypatch.setenv("FEEDBACK_ENABLED", raw)

    assert feedback_enabled() is False


@pytest.mark.parametrize("raw", ["true", "1", "yes", "on"])
def test_a_truthy_value_keeps_them(monkeypatch, raw):
    from clients.backend import feedback_enabled

    monkeypatch.setenv("FEEDBACK_ENABLED", raw)

    assert feedback_enabled() is True


def test_an_empty_value_falls_back_to_the_backends_default(monkeypatch):
    monkeypatch.setenv("FEEDBACK_ENABLED", "")
    from clients.backend import feedback_enabled

    assert feedback_enabled() is True


# =========================
# The bot's buttons
# =========================

def test_the_callback_data_fits_telegrams_limit():
    """64 bytes, which is why the button carries an id and not the exchange."""
    from clients.telegram_bot import rating_keyboard

    for button in rating_keyboard("a" * 16).inline_keyboard[0]:
        assert len(button.callback_data.encode("utf-8")) <= 64


def test_the_buttons_carry_both_ratings():
    from clients.telegram_bot import rating_keyboard

    data = [b.callback_data for b in rating_keyboard("req1").inline_keyboard[0]]

    assert data == ["fb:up:req1", "fb:down:req1"]


def test_the_bot_remembers_the_exchange_a_button_refers_to():
    from clients.telegram_bot import remember_for_rating

    context = SimpleNamespace(user_data={})

    remember_for_rating(context, "r1", {"question": "q", "answer": "a", "sources": []})

    assert context.user_data["rateable"]["r1"]["question"] == "q"


def test_a_long_chat_does_not_grow_without_bound():
    from clients.telegram_bot import RATEABLE_ANSWERS, remember_for_rating

    context = SimpleNamespace(user_data={})
    for index in range(RATEABLE_ANSWERS + 5):
        remember_for_rating(context, f"r{index}", {"question": str(index)})

    rateable = context.user_data["rateable"]
    assert len(rateable) == RATEABLE_ANSWERS
    assert "r0" not in rateable, "the oldest should have fallen off"
    assert f"r{RATEABLE_ANSWERS + 4}" in rateable, "the newest must be kept"


# =========================
# The bot's callback handler
# =========================

class FakeQuery:
    def __init__(self, data):
        self.data = data
        self.answers = []
        self.markup_edits = []

    async def answer(self, text=None, show_alert=False):
        self.answers.append(text)

    async def edit_message_reply_markup(self, reply_markup=None):
        self.markup_edits.append(reply_markup)


def _press(monkeypatch, data, rateable, status=200):
    """Drive on_rating without a network or an event loop of its own."""
    import clients.telegram_bot as bot

    posted = {}

    class FakeResponse:
        status_code = status

        def json(self):
            return {"message": "Thanks - recorded."}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            posted["url"] = url
            posted["json"] = json
            return FakeResponse()

    monkeypatch.setattr(bot.httpx, "AsyncClient", FakeClient)

    query = FakeQuery(data)
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=4242),
    )
    context = SimpleNamespace(user_data={"rateable": dict(rateable)})

    asyncio.run(bot.on_rating(update, context))
    return query, context, posted


EXCHANGE = {
    "question": "How many leave days?",
    "answer": "28",
    "sources": ["policy.docx"],
    "language": "Auto",
}


def test_pressing_a_button_sends_the_remembered_exchange(monkeypatch):
    query, _, posted = _press(monkeypatch, "fb:down:r1", {"r1": EXCHANGE})

    assert posted["url"].endswith("/feedback")
    assert posted["json"]["rating"] == "down"
    assert posted["json"]["question"] == EXCHANGE["question"]
    assert posted["json"]["sources"] == ["policy.docx"]
    assert posted["json"]["request_id"] == "r1"
    assert posted["json"]["client"] == "telegram"
    assert posted["json"]["user_id"] == "4242"
    assert query.answers == ["Thanks!"]


def test_the_buttons_are_taken_away_after_a_rating(monkeypatch):
    """A second press would record a second rating of the same answer."""
    query, context, _ = _press(monkeypatch, "fb:up:r1", {"r1": EXCHANGE})

    assert query.markup_edits == [None]
    assert "r1" not in context.user_data["rateable"]


def test_an_answer_the_bot_no_longer_remembers_is_declined(monkeypatch):
    """A restart clears the map; sending a rating with an empty question would
    poison the file."""
    query, _, posted = _press(monkeypatch, "fb:down:gone", {})

    assert posted == {}
    assert "too old" in query.answers[0]
    assert query.markup_edits == [None], "a live keyboard would keep failing"


def test_a_rejected_rating_keeps_the_buttons(monkeypatch):
    """So the user can try again rather than be told nothing happened."""
    query, context, _ = _press(monkeypatch, "fb:down:r1", {"r1": EXCHANGE}, status=507)

    assert query.markup_edits == []
    assert "r1" in context.user_data["rateable"]


def test_unparseable_callback_data_is_answered_and_dropped(monkeypatch):
    """An unanswered callback query spins in the client forever."""
    query, _, posted = _press(monkeypatch, "fb:nonsense", {"r1": EXCHANGE})

    assert query.answers == [None]
    assert posted == {}


# =========================
# Where the keyboard lands
# =========================

class FakeMessage:
    """Enough of telegram.Message for handle_message to run."""

    def __init__(self, text):
        self.text = text
        self.sent = []          # (text, reply_markup)
        self.plain = []
        self.chat = SimpleNamespace(send_action=self._noop)

    async def _noop(self, *args, **kwargs):
        return None

    async def reply_html(self, text, reply_markup=None, **kwargs):
        self.sent.append((text, reply_markup))
        return SimpleNamespace(text=text)

    async def reply_text(self, text, **kwargs):
        self.plain.append(text)
        return SimpleNamespace(text=text)


def _ask(monkeypatch, answer="Twenty eight days.", sources=None,
         request_id="req0123456789ab", enabled=True, status=200):
    """Drive handle_message with no network and no event loop of its own."""
    import clients.telegram_bot as bot

    monkeypatch.setattr(bot, "FEEDBACK_ENABLED", enabled)

    class FakeResponse:
        status_code = status
        headers = {bot.REQUEST_ID_HEADER: request_id} if request_id else {}

        def json(self):
            return {"answer": answer, "sources": sources or []}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            return FakeResponse()

    monkeypatch.setattr(bot.httpx, "AsyncClient", FakeClient)

    message = FakeMessage("How many leave days?")
    update = SimpleNamespace(message=message, effective_user=SimpleNamespace(id=7))
    context = SimpleNamespace(user_data={})

    asyncio.run(bot.handle_message(update, context))
    return message, context


def test_an_answer_comes_with_rating_buttons(monkeypatch):
    message, _ = _ask(monkeypatch)

    text, markup = message.sent[-1]
    assert markup is not None, "no keyboard was attached"
    assert [b.callback_data for b in markup.inline_keyboard[0]] == [
        "fb:up:req0123456789ab", "fb:down:req0123456789ab"
    ]


def test_the_keyboard_goes_under_everything_it_refers_to(monkeypatch):
    """With sources sent as their own message, buttons on the answer would sit
    above the sources they were rating."""
    message, _ = _ask(monkeypatch, sources=[{"source": "policy.docx"}])

    markups = [markup for _, markup in message.sent]
    assert len(message.sent) == 2
    assert markups[0] is None
    assert markups[1] is not None
    assert "Sources" in message.sent[1][0]


def test_only_one_keyboard_per_answer(monkeypatch):
    """A long answer is split into several messages; buttons on each would
    invite several ratings of one answer."""
    long_answer = "Ответ. " * 2000

    message, _ = _ask(monkeypatch, answer=long_answer)

    assert len(message.sent) > 1, "the answer was not split, so this proves nothing"
    assert sum(markup is not None for _, markup in message.sent) == 1


def test_the_exchange_is_remembered_for_the_button(monkeypatch):
    message, context = _ask(monkeypatch, sources=[{"source": "policy.docx"}])

    remembered = context.user_data["rateable"]["req0123456789ab"]
    assert remembered["question"] == "How many leave days?"
    assert remembered["answer"] == "Twenty eight days."
    assert remembered["sources"] == ["policy.docx"]


def test_no_buttons_when_collection_is_off(monkeypatch):
    message, context = _ask(monkeypatch, enabled=False)

    assert message.sent[-1][1] is None
    assert "rateable" not in context.user_data


def test_no_buttons_without_a_request_id(monkeypatch):
    """A button whose rating could not be tied back to anything is worth less
    than the confusion of a dead keyboard."""
    message, context = _ask(monkeypatch, request_id=None)

    assert message.sent[-1][1] is None
    assert "rateable" not in context.user_data


def test_a_failed_question_offers_no_rating(monkeypatch):
    message, context = _ask(monkeypatch, status=503)

    assert message.sent == []
    assert message.plain, "the user was told nothing"
    assert "rateable" not in context.user_data
