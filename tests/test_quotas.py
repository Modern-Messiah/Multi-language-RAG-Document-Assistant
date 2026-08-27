"""Per-owner limits on documents and bytes.

Until these existed the only bound was MAX_FILE_SIZE, on one file: one user_id
could fill the volume the vector store lives on, and every upload was paid
embedding calls with no ceiling. The limits are loud when hit - a 413 with the
numbers in it - which is why they ship on, unlike the retrieval knobs that
fail silently and ship off.
"""
import threading

import pytest
from fastapi.testclient import TestClient

from app.humanize import describe_quota, human_size
from app.main import OwnerLocks, create_app
from tests.conftest import TEST_API_KEY, FakeChatClient, make_settings

ONE = b"Annual leave for an engineer is twenty eight calendar days."
TWO = b"Sick leave is paid from the first day of absence, on a certificate."
THREE = b"Remote work is agreed with the line manager one week ahead."


def _upload(client, filename, content, user_id="u1"):
    return client.post(
        "/upload", params={"user_id": user_id},
        files={"file": (filename, content, "text/plain")},
    )


def _quota(client, user_id="u1"):
    response = client.get("/documents", params={"user_id": user_id})
    assert response.status_code == 200, response.text
    return response.json()["quota"]


@pytest.fixture()
def limited(tmp_path, fake_openai_embeddings):
    """An API whose owner may hold two documents or 150 bytes, whichever first."""
    settings = make_settings(tmp_path, max_documents_per_user=2, max_bytes_per_user=150)
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        app.state.rag_chain.client = FakeChatClient()
        client.settings = settings
        client.app_state = app.state
        yield client


# =========================
# Settings
# =========================

def test_quotas_ship_on(tmp_path):
    """Loud when hit, so a blind default is acceptable - unlike the retrieval
    knobs, which fail silently and ship off."""
    from app.config import Settings

    settings = Settings(_env_file=None, openai_api_key="k")

    assert settings.max_documents_per_user == 200
    assert settings.max_bytes_per_user == 1024 ** 3


@pytest.mark.parametrize("field", ["max_documents_per_user", "max_bytes_per_user"])
def test_zero_means_off_and_negative_is_rejected(field):
    """0 is a legitimate value here, unlike MAX_FILE_SIZE where 0 is refused:
    a file of at most zero bytes is nonsense, a namespace with no ceiling is a
    choice."""
    from pydantic import ValidationError

    from app.config import Settings

    assert getattr(Settings(_env_file=None, openai_api_key="k", **{field: 0}), field) == 0
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key="k", **{field: -1})


# =========================
# The document limit
# =========================

def test_the_third_document_is_refused(limited):
    assert _upload(limited, "one.txt", ONE).status_code == 200
    assert _upload(limited, "two.txt", TWO).status_code == 200

    response = _upload(limited, "three.txt", THREE)

    assert response.status_code == 413, response.text
    assert "2 of 2" in response.json()["detail"]


def test_a_refused_upload_leaves_nothing_behind(limited):
    _upload(limited, "one.txt", ONE)
    _upload(limited, "two.txt", TWO)
    before = limited.app_state.embeddings.count()

    _upload(limited, "three.txt", THREE)

    assert limited.app_state.embeddings.count() == before
    stored = sorted(p.name for p in (limited.settings.upload_dir / "u1").iterdir())
    assert not any(name.endswith("three.txt") for name in stored)


def test_the_message_says_what_to_do_and_names_no_setting(limited):
    """User-facing wording follows the other limits: the numbers and the
    remedy, no environment variable. That goes to the log."""
    _upload(limited, "one.txt", ONE)
    _upload(limited, "two.txt", TWO)

    detail = _upload(limited, "three.txt", THREE).json()["detail"]

    assert "Remove documents" in detail
    assert "operator" in detail
    assert "MAX_" not in detail


def test_the_log_names_the_setting(limited, caplog):
    _upload(limited, "one.txt", ONE)
    _upload(limited, "two.txt", TWO)

    with caplog.at_level("WARNING", logger="app.main"):
        _upload(limited, "three.txt", THREE)

    assert any("MAX_DOCUMENTS_PER_USER" in r.getMessage() for r in caplog.records)


def test_the_byte_limit_says_the_same_kind_of_thing(limited, caplog):
    """Both branches have to follow the same rule: numbers and a remedy for the
    user, the setting name for the log. Only the document branch was pinned."""
    assert _upload(limited, "big.txt", b"x" * 140, user_id="u3").status_code == 200

    with caplog.at_level("WARNING", logger="app.main"):
        detail = _upload(limited, "two.txt", TWO, user_id="u3").json()["detail"]

    assert "Storage limit" in detail
    assert "MAX_" not in detail
    assert "Remove documents" in detail
    assert any("MAX_BYTES_PER_USER" in r.getMessage() for r in caplog.records)


def test_deleting_one_makes_room(limited):
    """The remedy the message gives has to work."""
    first = _upload(limited, "one.txt", ONE).json()["file_hash"]
    _upload(limited, "two.txt", TWO)
    assert _upload(limited, "three.txt", THREE).status_code == 413

    assert limited.delete(f"/documents/{first}", params={"user_id": "u1"}).status_code == 200

    assert _upload(limited, "three.txt", THREE).status_code == 200


def test_an_identical_reupload_succeeds_at_the_limit(limited):
    """It adds nothing, so it cannot be over anything."""
    _upload(limited, "one.txt", ONE)
    _upload(limited, "two.txt", TWO)

    response = _upload(limited, "one.txt", ONE)

    assert response.status_code == 200
    assert response.json()["duplicate"] is True


def test_a_new_revision_of_an_existing_file_succeeds_at_the_limit(limited):
    """It replaces the old revision, so the count does not move."""
    _upload(limited, "policy.txt", ONE)
    _upload(limited, "two.txt", TWO)

    response = _upload(limited, "policy.txt", THREE)  # same name, new content

    assert response.status_code == 200, response.text
    assert response.json()["replaced"] is True
    assert _quota(limited)["documents"] == 2


def test_limits_are_per_owner(limited):
    _upload(limited, "one.txt", ONE)
    _upload(limited, "two.txt", TWO)

    assert _upload(limited, "three.txt", THREE, user_id="u2").status_code == 200


def test_zero_is_unlimited(tmp_path, fake_openai_embeddings):
    settings = make_settings(tmp_path, max_documents_per_user=0, max_bytes_per_user=0)
    app = create_app(settings)
    with TestClient(app) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        for index, content in enumerate((ONE, TWO, THREE)):
            assert _upload(client, f"doc{index}.txt", content).status_code == 200

        quota = _quota(client)
        assert quota["max_documents"] == 0
        assert quota["max_bytes"] == 0


# =========================
# The byte limit
# =========================

def test_the_byte_limit_counts_what_is_on_disk(limited):
    """150 bytes allowed; ONE is 59, TWO is 66, together 125; THREE would cross."""
    assert _upload(limited, "one.txt", ONE).status_code == 200
    assert _upload(limited, "two.txt", TWO).status_code == 200
    # The document limit is also 2 here, so use a fresh owner with room in
    # documents but not in bytes.
    big = b"x" * 140
    assert _upload(limited, "big.txt", big, user_id="u3").status_code == 200

    response = _upload(limited, "two.txt", TWO, user_id="u3")

    assert response.status_code == 413
    assert "Storage limit" in response.json()["detail"]
    assert "150 bytes" in response.json()["detail"]


def test_bytes_reported_equal_the_stored_files(limited):
    _upload(limited, "one.txt", ONE)
    _upload(limited, "two.txt", TWO)

    quota = _quota(limited)

    assert quota["bytes"] == len(ONE) + len(TWO)
    assert quota["documents"] == 2
    assert quota["max_documents"] == 2
    assert quota["max_bytes"] == 150


def test_an_orphan_file_is_not_held_against_the_user(limited):
    """A file with no vectors behind it is invisible in /documents and cannot
    be deleted through the API, so counting it would present a limit there is
    no way to get under. The sweep is where it is reconciled."""
    _upload(limited, "one.txt", ONE)
    owner_dir = limited.settings.upload_dir / "u1"
    (owner_dir / ("f" * 16 + "_ghost.txt")).write_bytes(b"g" * 140)

    assert _quota(limited)["bytes"] == len(ONE)
    assert _upload(limited, "two.txt", TWO).status_code == 200


# =========================
# Two uploads at once
# =========================

def test_two_simultaneous_uploads_cannot_both_squeeze_in(tmp_path, fake_openai_embeddings, monkeypatch):
    """The check is a read followed by a write, and without the owner lock two
    uploads arriving together both read "room for one more".

    Forced rather than hoped for: a barrier holds the first request inside the
    usage read until the second arrives there too. With the lock the second
    never reaches it, the barrier times out, and the first goes on; without the
    lock both read "0 of 1" and both index. Without the barrier this test passed
    on a fast machine whether or not the lock existed.
    """
    import app.main as main

    settings = make_settings(tmp_path, max_documents_per_user=1)
    app = create_app(settings)
    barrier = threading.Barrier(2)
    real_usage = main._owner_usage

    def synchronized(state, settings_, user_id, exclude_hashes=()):
        usage = real_usage(state, settings_, user_id, exclude_hashes)
        try:
            barrier.wait(timeout=1.5)
        except threading.BrokenBarrierError:
            pass  # the lock kept the other request out, which is the point
        return usage

    monkeypatch.setattr(main, "_owner_usage", synchronized)

    results = []
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["X-API-Key"] = TEST_API_KEY

        def go(name, content):
            results.append(_upload(client, name, content).status_code)

        threads = [
            threading.Thread(target=go, args=("one.txt", ONE)),
            threading.Thread(target=go, args=("two.txt", TWO)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(results) == [200, 413], results
        assert _quota(client)["documents"] == 1


def test_a_failed_upload_releases_the_lock(limited):
    """Otherwise one corrupt file would freeze an owner's uploads forever.

    Checked without blocking first: if a refactor ever loses the release on the
    exception path, a second upload would wait forever inside the threadpool and
    the suite would hang instead of failing this test.
    """
    assert _upload(limited, "broken.pdf", b"not a pdf").status_code == 400

    lock = limited.app_state.upload_locks.for_owner("u1")
    assert lock.acquire(blocking=False), "the lock was not released"
    lock.release()

    assert _upload(limited, "one.txt", ONE).status_code == 200


def test_the_same_owner_always_gets_the_same_lock():
    locks = OwnerLocks()

    assert locks.for_owner("u1") is locks.for_owner("u1")


def test_different_owners_do_not_wait_on_each_other():
    """Not a lock per owner - the web UI would grow one per browser session -
    but not a single lock either, which would serialize every upload in the
    process."""
    locks = OwnerLocks()

    distinct = {id(locks.for_owner(f"u{i}")) for i in range(200)}

    assert len(distinct) > OwnerLocks.STRIPES // 2, (
        f"200 owners landed on only {len(distinct)} lock(s)"
    )


def test_locks_are_bounded_however_many_owners_appear():
    """A lock per user_id would grow for as long as the process lived: the web
    UI mints a new owner per browser session."""
    locks = OwnerLocks()

    # The locks themselves, not their id(): nothing would retain a discarded
    # object, and CPython reuses freed addresses - so an implementation handing
    # out a fresh Lock on every call (no mutual exclusion at all) produced a
    # tiny set of ids and passed.
    distinct = {locks.for_owner(f"web-{i:032x}") for i in range(5000)}

    assert len(distinct) <= OwnerLocks.STRIPES


# =========================
# What the clients show
# =========================

def test_a_413_reaches_the_user_verbatim():
    """Not hidden as an operator problem (401/403 are), not told to retry
    (429), no request id appended (5xx)."""
    from clients.backend import describe_error

    detail = "Document limit reached: you hold 2 of 2. Remove documents you no longer need."

    assert describe_error(413, detail, request_id="abc") == detail


def test_the_quota_line_reads_as_a_sentence():
    line = describe_quota({"documents": 3, "max_documents": 200,
                           "bytes": 12 * 1024, "max_bytes": 1024 ** 3})

    assert line == "3 of 200 documents, 12 KB of 1 GB"


def test_an_unlimited_quota_never_reads_as_of_zero():
    """"3 of 0 documents" is what a naive format produces when a limit is off."""
    line = describe_quota({"documents": 3, "max_documents": 0, "bytes": 12 * 1024, "max_bytes": 0})

    assert line == "3 documents, 12 KB"
    assert " of 0" not in line


def test_one_document_is_singular():
    assert describe_quota({"documents": 1, "max_documents": 0, "bytes": 5, "max_bytes": 0}) == (
        "1 document, 5 bytes"
    )


@pytest.mark.parametrize("junk", [None, "", 42, []])
def test_anything_but_a_quota_block_draws_nothing(junk):
    """An older backend has no quota block, and a caption must not crash over
    it - nor invent "0 documents", which would be a false statement about the
    user's namespace."""
    assert describe_quota(junk) == ""


def test_a_dict_missing_every_key_reads_as_empty_usage():
    """Not the same case: something quota-shaped arrived, so the numbers it
    does not carry are zero."""
    assert describe_quota({"unexpected": 1}) == "0 documents, 0 bytes"


def test_the_bot_appends_usage_to_the_listing():
    from clients.telegram_bot import format_document_list

    text = format_document_list(
        [{"source": "policy.docx", "chunks": 3}],
        {"documents": 1, "max_documents": 200, "bytes": 2048, "max_bytes": 1024 ** 3},
    )

    assert "1 of 200 documents, 2 KB of 1 GB" in text


def test_the_bot_shows_usage_even_with_nothing_indexed():
    """Which is when the user most wants to know how much room there is."""
    from clients.telegram_bot import format_document_list

    text = format_document_list([], {"documents": 0, "max_documents": 200, "bytes": 0, "max_bytes": 0})

    assert "0 of 200 documents" in text


def test_the_bot_works_against_an_older_backend():
    from clients.telegram_bot import format_document_list

    assert "policy.docx" in format_document_list([{"source": "policy.docx", "chunks": 3}])


# =========================
# human_size grows a unit
# =========================

@pytest.mark.parametrize("num_bytes,expected", [
    (1024 ** 3, "1 GB"),
    (int(1.5 * 1024 ** 3), "1.5 GB"),
    (1023 * 1024 ** 2, "1023 MB"),
])
def test_gigabytes_read_as_gigabytes(num_bytes, expected):
    """A 1 GiB quota is "1 GB", not "1024 MB"."""
    assert human_size(num_bytes) == expected


def test_human_size_is_still_importable_from_main():
    """It moved to a light module so the web UI can use it; the old name has
    callers."""
    from app.main import human_size as from_main

    assert from_main is human_size


def test_the_bot_tells_a_telegram_user_how_to_make_room(monkeypatch):
    """The backend's wording is client-neutral ("remove documents you no longer
    need"), and this bot has no per-document delete - so it has to say where the
    remedy is."""
    import asyncio
    from types import SimpleNamespace

    import clients.telegram_bot as bot

    class FakeResponse:
        status_code = 413
        headers = {}

        def json(self):
            return {"detail": "Document limit reached: you hold 200 of 200."}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(bot.httpx, "AsyncClient", FakeClient)

    edits = []

    class FakeStatus:
        async def edit_text(self, text, **kwargs):
            edits.append(text)

    async def status(text, **kwargs):
        return FakeStatus()

    message = SimpleNamespace(
        document=SimpleNamespace(file_name="policy.docx", file_size=1000, file_id="f1"),
        reply_text=status,
        chat=SimpleNamespace(send_action=lambda *a, **k: asyncio.sleep(0)),
    )
    update = SimpleNamespace(message=message, effective_user=SimpleNamespace(id=7))

    async def get_file(file_id):
        return SimpleNamespace(download_as_bytearray=lambda: _bytes())

    async def _bytes():
        return bytearray(b"content")

    context = SimpleNamespace(bot=SimpleNamespace(get_file=get_file), user_data={})

    asyncio.run(bot.handle_document(update, context))

    assert edits, "the user was told nothing"
    assert "200 of 200" in edits[-1]
    assert "/documents" in edits[-1] and "/clear" in edits[-1]


def test_a_file_that_vanishes_mid_scan_is_worth_nothing_not_a_500(limited, monkeypatch):
    """Every caller lists a directory and then stats what it found, and a
    delete or a retired revision can remove a file in between - the listing is
    not always under the owner's lock. A vanished file counts as zero bytes."""
    import app.main as main

    uploaded = _upload(limited, "one.txt", ONE).json()
    real = main._stored_files

    def with_a_ghost(settings, user_id):
        ghost = settings.upload_dir / user_id / f"{uploaded['file_hash']}_vanished.txt"
        return [*real(settings, user_id), ghost]

    monkeypatch.setattr(main, "_stored_files", with_a_ghost)

    quota = _quota(limited)

    assert quota["bytes"] == len(ONE)
    assert _upload(limited, "two.txt", TWO).status_code == 200


def test_a_vanished_orphan_does_not_break_the_sweep(limited, monkeypatch):
    import app.main as main

    _upload(limited, "one.txt", ONE)
    real = main._stored_files

    def with_a_ghost(settings, user_id):
        ghost = settings.upload_dir / user_id / ("f" * 16 + "_vanished.txt")
        return [*real(settings, user_id), ghost]

    monkeypatch.setattr(main, "_stored_files", with_a_ghost)

    response = limited.post("/maintenance/sweep", params={"idle_days": 30, "prefix": ""})

    assert response.status_code == 200, response.text
    assert response.json()["orphans"][0]["bytes"] == 0
