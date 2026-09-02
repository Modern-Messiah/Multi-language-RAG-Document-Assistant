"""Who was here when, and removing the namespaces nobody comes back to.

The web UI mints a fresh owner per browser session, so its documents are
orphaned the moment the tab closes. Telegram ids are stable, so a quiet user is
just quiet. The sweep has to tell those apart, and every mistake it can make is
in one direction - deleting something someone still wanted - so most of this
file is about when it refuses.
"""
import os
import time
import urllib.error
from datetime import datetime, timedelta, timezone

import pytest

from app.activity import ACTIVITY_DIRNAME, ActivityTracker, is_owner_name
from app.main import MIN_IDLE_DAYS_TO_APPLY, USER_ID_PATTERN

TEXT = b"Annual leave for an engineer is twenty eight calendar days."
OTHER = b"Sick leave is paid from the first day of absence, on a certificate."


def _upload(api, user_id, filename="doc.txt", content=TEXT):
    response = api.post(
        "/upload", params={"user_id": user_id},
        files={"file": (filename, content, "text/plain")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _age(path, days, extra_seconds=5):
    """Make a file look `days` (and a few seconds) old."""
    stamp = time.time() - days * 86400 - extra_seconds
    os.utime(path, (stamp, stamp))


def _marker(api, user_id):
    return api.upload_dir / ACTIVITY_DIRNAME / user_id


def _make_idle(api, user_id, days=40):
    """Age everything that dates this owner: the marker and the stored files.

    Aging only the marker proves nothing - last_seen is the newer of the two,
    so the file an upload just wrote keeps the owner looking active and an
    assertion about a touch holds whether or not the touch happened. Two
    mutations passed against that mistake before it was found.
    """
    _age(_marker(api, user_id), days=days)
    owner_dir = api.upload_dir / user_id
    if owner_dir.is_dir():
        for path in owner_dir.rglob("*"):
            if path.is_file():
                _age(path, days=days)


def _sweep(api, **params):
    params.setdefault("idle_days", 30)
    params.setdefault("prefix", "web-")
    response = api.post("/maintenance/sweep", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _owners(api):
    return set(api.app_state.embeddings.list_owners())


# =========================
# The tracker
# =========================

def test_a_touch_leaves_a_marker(tmp_path):
    tracker = ActivityTracker(tmp_path)

    tracker.touch("u1")

    assert (tmp_path / ACTIVITY_DIRNAME / "u1").exists()


def test_the_marker_is_not_in_the_owners_own_directory(tmp_path):
    """The upload tests assert an owner directory is empty after a rejected
    upload, and the byte quota counts what is in it."""
    tracker = ActivityTracker(tmp_path)

    tracker.touch("u1")

    assert not (tmp_path / "u1").exists()


def test_nothing_dates_an_owner_nobody_has_seen(tmp_path):
    assert ActivityTracker(tmp_path).last_seen("ghost") is None


def test_a_second_touch_moves_the_marker_forward(tmp_path):
    tracker = ActivityTracker(tmp_path)
    tracker.touch("u1")
    _age(tracker.marker_for("u1"), days=10)
    before = tracker.last_seen("u1")

    tracker.touch("u1")

    assert tracker.last_seen("u1") > before


def test_last_seen_is_the_newer_of_marker_and_files(tmp_path):
    """An upload in flight has a fresh file and a stale marker, and the file is
    the truth."""
    tracker = ActivityTracker(tmp_path)
    tracker.touch("u1")
    _age(tracker.marker_for("u1"), days=40)
    owner_dir = tmp_path / "u1"
    owner_dir.mkdir()
    (owner_dir / ("a" * 16 + "_fresh.txt")).write_bytes(b"fresh")

    seen = tracker.last_seen("u1")

    assert seen > datetime.now(timezone.utc) - timedelta(minutes=1)


def test_last_seen_is_aware_utc(tmp_path):
    """So it compares with datetime.now(timezone.utc) and prints with an offset
    that means the same thing in every zone."""
    tracker = ActivityTracker(tmp_path)
    tracker.touch("u1")

    seen = tracker.last_seen("u1")

    assert seen.tzinfo is not None
    assert seen.utcoffset() == timedelta(0)


def test_forget_removes_the_marker_and_tolerates_its_absence(tmp_path):
    tracker = ActivityTracker(tmp_path)
    tracker.touch("u1")

    tracker.forget("u1")
    tracker.forget("u1")

    assert tracker.last_seen("u1") is None


def test_a_touch_that_cannot_be_written_is_counted_not_raised(tmp_path):
    """A full volume is when someone reaches for the sweep, and also when the
    markers stop being true - so the sweep is told, and the request that
    carried the touch is not failed."""
    (tmp_path / ACTIVITY_DIRNAME).write_bytes(b"a file where the directory should be")
    tracker = ActivityTracker(tmp_path)

    tracker.touch("u1")

    assert tracker.touch_failures == 1


def test_seeding_dates_owners_that_have_no_marker(tmp_path):
    """Owners from before markers existed would otherwise be dated by their
    newest upload - a lower bound on activity, which can only make a live
    owner look more idle than they are."""
    (tmp_path / "12345").mkdir()
    (tmp_path / "web-old").mkdir()
    tracker = ActivityTracker(tmp_path)

    seeded = tracker.seed_missing()

    assert seeded == 2
    assert tracker.last_seen("12345") > datetime.now(timezone.utc) - timedelta(minutes=1)


def test_seeding_never_moves_an_existing_marker(tmp_path):
    """Restarts must not reset idleness, or a nightly restart would make the
    sweep useless."""
    tracker = ActivityTracker(tmp_path)
    (tmp_path / "u1").mkdir()
    tracker.touch("u1")
    _age(tracker.marker_for("u1"), days=40)
    before = tracker.last_seen("u1")

    tracker.seed_missing()

    assert tracker.last_seen("u1") == before


def test_seeding_skips_names_that_are_not_owners(tmp_path):
    (tmp_path / "tmp restore").mkdir()
    (tmp_path / ACTIVITY_DIRNAME).mkdir()
    tracker = ActivityTracker(tmp_path)

    assert tracker.seed_missing() == 0
    assert not (tmp_path / ACTIVITY_DIRNAME / "tmp restore").exists()


def test_reset_dates_every_owner_now(tmp_path):
    tracker = ActivityTracker(tmp_path)
    (tmp_path / "u1").mkdir()
    tracker.touch("u1")
    _age(tracker.marker_for("u1"), days=40)

    assert tracker.reset_all() == 1
    assert tracker.last_seen("u1") > datetime.now(timezone.utc) - timedelta(minutes=1)


# =========================
# What counts as an owner name
# =========================

@pytest.mark.parametrize("name", ["u1", "12345", "web-" + "a" * 32, "A_b-9", "x" * 64])
def test_a_user_id_the_api_accepts_is_an_owner_name(name):
    import re

    assert is_owner_name(name)
    assert re.fullmatch(USER_ID_PATTERN, name), "the two rules have drifted apart"


@pytest.mark.parametrize("name", [ACTIVITY_DIRNAME, "lost+found", "tmp restore", "", "x" * 65, "../u1"])
def test_anything_else_is_not(name):
    assert not is_owner_name(name)


# =========================
# Which requests count as being here
# =========================

def test_a_successful_upload_leaves_a_marker(api):
    _upload(api, "u1")

    assert _marker(api, "u1").exists()


def test_a_rejected_upload_leaves_none(api):
    """The upload tests also assert the owner directory is empty afterwards;
    this asserts the same about the marker."""
    response = api.post("/upload", params={"user_id": "u1"},
                        files={"file": ("broken.pdf", b"not a pdf", "application/pdf")})

    assert response.status_code == 400
    assert not _marker(api, "u1").exists()


def test_a_question_counts(api):
    _upload(api, "u1")
    _make_idle(api, "u1")

    api.post("/query", json={"question": "leave?", "language": "Auto", "user_id": "u1"})

    assert api.app_state.activity.last_seen("u1") > datetime.now(timezone.utc) - timedelta(minutes=1)


def test_a_streamed_question_counts(api):
    _upload(api, "u1")
    _make_idle(api, "u1")

    with api.stream("POST", "/query/stream",
                    json={"question": "leave?", "language": "Auto", "user_id": "u1"}) as response:
        assert response.status_code == 200

    assert api.app_state.activity.last_seen("u1") > datetime.now(timezone.utc) - timedelta(minutes=1)


def test_listing_a_namespace_with_documents_counts(api):
    _upload(api, "u1")
    _make_idle(api, "u1")

    api.get("/documents", params={"user_id": "u1"})

    assert api.app_state.activity.last_seen("u1") > datetime.now(timezone.utc) - timedelta(minutes=1)


def test_listing_an_empty_namespace_leaves_no_marker(api):
    """The web UI lists on every rerun under a fresh per-session owner; a marker
    per page view would fill the activity directory with nothing."""
    api.get("/documents", params={"user_id": "web-visitor"})

    assert not _marker(api, "web-visitor").exists()


def test_deleting_a_document_counts(api):
    uploaded = _upload(api, "u1")
    _make_idle(api, "u1")

    api.delete(f"/documents/{uploaded['file_hash']}", params={"user_id": "u1"})

    assert api.app_state.activity.last_seen("u1") > datetime.now(timezone.utc) - timedelta(minutes=1)


def test_a_rating_counts(api):
    _upload(api, "u1")
    _make_idle(api, "u1")

    api.post("/feedback", json={"rating": "up", "user_id": "u1", "question": "q"})

    assert api.app_state.activity.last_seen("u1") > datetime.now(timezone.utc) - timedelta(minutes=1)


def test_clearing_waits_for_an_upload_in_flight(api):
    """Without the lock a sweep or a /clear can land between an upload's
    write_bytes and its add_documents, and the upload then answers 200 with
    nothing stored. Asserted on the lock being taken, because provoking that
    interleaving reliably would mean instrumenting the code it guards."""
    taken = []
    real = api.app_state.upload_locks.for_owner

    def recording(user_id):
        taken.append(user_id)
        return real(user_id)

    api.app_state.upload_locks.for_owner = recording
    _upload(api, "u1")

    assert api.post("/clear", params={"user_id": "u1"}).status_code == 200
    assert taken.count("u1") >= 2, "the clear did not take the owner's lock"


def test_clearing_forgets_the_owner(api):
    _upload(api, "u1")

    api.post("/clear", params={"user_id": "u1"})

    assert not _marker(api, "u1").exists()
    assert api.app_state.activity.last_seen("u1") is None


# =========================
# The sweep: looking
# =========================

def test_the_sweep_needs_the_key(api):
    response = api.post("/maintenance/sweep", params={"idle_days": 30, "prefix": "web-"},
                        headers={"X-API-Key": "wrong"})

    assert response.status_code == 401


def test_the_prefix_is_required(api):
    """The choice to sweep every tenant has to be written into the request."""
    response = api.post("/maintenance/sweep", params={"idle_days": 30})

    assert response.status_code == 422


@pytest.mark.parametrize("bad", [{"idle_days": 0}, {"idle_days": -1}, {"idle_days": 4000},
                                 {"idle_days": 30, "prefix": "web-/../"}])
def test_bad_parameters_are_refused(api, bad):
    params = {"prefix": "web-", **bad}

    assert api.post("/maintenance/sweep", params=params).status_code == 422


def test_an_idle_owner_is_a_candidate(api):
    _upload(api, "web-idle")
    _age(_marker(api, "web-idle"), days=31)
    for path in (api.upload_dir / "web-idle").iterdir():
        _age(path, days=31)

    body = _sweep(api)

    assert [c["user_id"] for c in body["candidates"]] == ["web-idle"]
    candidate = body["candidates"][0]
    assert candidate["documents"] == 1
    assert candidate["bytes"] == len(TEXT)
    assert candidate["last_seen"].endswith("+00:00")


def test_an_active_owner_is_not_listed_anywhere(api):
    _upload(api, "web-busy")

    body = _sweep(api)

    for bucket in ("candidates", "empty", "unknown", "foreign"):
        assert "web-busy" not in [c if isinstance(c, str) else c["user_id"] for c in body[bucket]]


def test_idle_is_strictly_older_than_the_cutoff(api):
    """A few seconds inside the window is not idle; a few seconds past it is."""
    _upload(api, "web-inside")
    _upload(api, "web-outside")
    for owner, offset in (("web-inside", -300), ("web-outside", 300)):
        stamp = time.time() - 30 * 86400 + (-offset)
        os.utime(_marker(api, owner), (stamp, stamp))
        for path in (api.upload_dir / owner).iterdir():
            os.utime(path, (stamp, stamp))

    names = [c["user_id"] for c in _sweep(api)["candidates"]]

    assert names == ["web-outside"], names


def test_the_prefix_narrows_the_sweep(api):
    for owner in ("web-a", "12345"):
        _upload(api, owner)
        _age(_marker(api, owner), days=40)
        for path in (api.upload_dir / owner).iterdir():
            _age(path, days=40)

    names = [c["user_id"] for c in _sweep(api, prefix="web-")["candidates"]]

    assert names == ["web-a"]


def test_an_owner_nothing_dates_is_unknown_and_never_swept(api):
    """Vectors but no directory and no marker: only a pre-marker owner whose
    files were removed by hand looks like this, and unknown is not idle."""
    import shutil

    _upload(api, "web-mystery")
    shutil.rmtree(api.upload_dir / "web-mystery")
    _marker(api, "web-mystery").unlink()

    body = _sweep(api, apply="true", force="true")

    assert body["unknown"] == ["web-mystery"]
    assert body["swept"] == []
    assert "web-mystery" in _owners(api)


def test_a_name_that_cannot_be_an_owner_is_reported_and_left_alone(api):
    """`.activity` itself is the first such name; a directory an operator made
    for a manual restore is another."""
    (api.upload_dir / "tmp restore").mkdir(parents=True)
    (api.upload_dir / "tmp restore" / "note.txt").write_bytes(b"keep")
    _upload(api, "web-x")

    body = _sweep(api, prefix="", apply="true", force="true")

    assert body["foreign"] == ["tmp restore"]
    assert ACTIVITY_DIRNAME not in body["foreign"]
    assert (api.upload_dir / "tmp restore" / "note.txt").exists()
    assert (api.upload_dir / ACTIVITY_DIRNAME).is_dir()


def test_an_owner_who_deleted_everything_is_empty(api):
    uploaded = _upload(api, "web-gone")
    api.delete(f"/documents/{uploaded['file_hash']}", params={"user_id": "web-gone"})
    _age(_marker(api, "web-gone"), days=40)

    body = _sweep(api)

    assert [e["user_id"] for e in body["empty"]] == ["web-gone"]
    assert body["candidates"] == []


def test_orphan_files_are_reported(api):
    """A file with no vectors behind it, left by a crash between write and
    index or by an older /clear that deleted vectors only."""
    _upload(api, "12345")
    (api.upload_dir / "12345" / ("f" * 16 + "_ghost.txt")).write_bytes(b"g" * 100)

    body = _sweep(api, prefix="")

    assert body["orphans"] == [
        {"user_id": "12345", "files": 1, "bytes": 100, "in_scope": True}
    ]


def test_every_dated_field_carries_an_offset(api):
    _upload(api, "web-a")

    body = _sweep(api)

    assert body["cutoff"].endswith("+00:00")
    assert body["newest_seen"].endswith("+00:00")


def test_a_dry_run_changes_nothing(api):
    _upload(api, "web-idle")
    _age(_marker(api, "web-idle"), days=40)
    for path in (api.upload_dir / "web-idle").iterdir():
        _age(path, days=40)

    body = _sweep(api)

    assert body["dry_run"] is True
    assert body["swept"] == []
    assert "web-idle" in _owners(api)
    assert _marker(api, "web-idle").exists()


# =========================
# The sweep: refusing
# =========================

def _idle_web_owner(api, name="web-idle", days=40):
    _upload(api, name)
    _age(_marker(api, name), days=days)
    for path in (api.upload_dir / name).iterdir():
        _age(path, days=days)


def test_apply_without_a_prefix_is_refused(api):
    """It would sweep every tenant, stable Telegram ids included."""
    _idle_web_owner(api)
    _upload(api, "12345")

    body = _sweep(api, prefix="", apply="true")

    assert body["refused"] and "prefix" in body["refused"]
    assert body["dry_run"] is True
    assert body["swept"] == []
    assert "web-idle" in _owners(api)


def test_apply_with_too_few_idle_days_is_refused(api):
    """A typo in a cron line must not sweep yesterday's users."""
    _idle_web_owner(api)
    _upload(api, "12345")

    body = _sweep(api, idle_days=MIN_IDLE_DAYS_TO_APPLY - 1, apply="true")

    assert body["refused"] and str(MIN_IDLE_DAYS_TO_APPLY) in body["refused"]
    assert "web-idle" in _owners(api)


def test_apply_is_refused_when_activity_writes_have_failed(api):
    """Then some live owners may look idle, and the data is wrong in the one
    direction that deletes things."""
    _idle_web_owner(api)
    _upload(api, "12345")
    api.app_state.activity.touch_failures = 3

    body = _sweep(api, apply="true")

    assert body["refused"] and "3 activity update" in body["refused"]
    assert "web-idle" in _owners(api)


def test_apply_is_refused_when_nobody_at_all_has_been_seen(api):
    """After a restore or a long stop everyone looks idle at once. That is
    stale data, not a mass departure."""
    _idle_web_owner(api)

    body = _sweep(api, apply="true")

    assert body["refused"] and "no owner at all" in body["refused"]
    assert body["newest_seen"] < body["cutoff"]
    assert "web-idle" in _owners(api)


def test_someone_active_under_another_prefix_lifts_that_refusal(api):
    """A prefix sweep of web sessions while Telegram users are busy is fine."""
    _idle_web_owner(api)
    _upload(api, "12345")

    body = _sweep(api, apply="true")

    assert body["refused"] is None
    assert body["swept"] == ["web-idle"]


def test_force_overrides_each_refusal(api):
    _idle_web_owner(api)

    body = _sweep(api, prefix="", idle_days=1, apply="true", force="true")

    assert body["refused"] is None
    assert body["swept"] == ["web-idle"]


def test_a_dry_run_never_refuses(api):
    """Looking is always allowed; the refusals are about deleting."""
    _idle_web_owner(api)

    body = _sweep(api, prefix="", idle_days=1)

    assert body["refused"] is None
    assert [c["user_id"] for c in body["candidates"]] == ["web-idle"]


# =========================
# The sweep: applying
# =========================

def test_apply_removes_vectors_files_and_marker(api):
    _idle_web_owner(api)
    _upload(api, "12345")

    body = _sweep(api, apply="true")

    assert body["dry_run"] is False
    assert body["swept"] == ["web-idle"]
    assert "web-idle" not in _owners(api)
    assert not (api.upload_dir / "web-idle").exists()
    assert not _marker(api, "web-idle").exists()


def test_apply_leaves_other_owners_alone(api):
    _idle_web_owner(api)
    _upload(api, "12345")
    _idle_web_owner(api, name="web-kept", days=10)  # inside the window

    _sweep(api, apply="true")

    assert {"12345", "web-kept"} <= _owners(api)
    assert _marker(api, "12345").exists()


def test_apply_cleans_empty_namespaces_too(api):
    uploaded = _upload(api, "web-gone")
    api.delete(f"/documents/{uploaded['file_hash']}", params={"user_id": "web-gone"})
    _age(_marker(api, "web-gone"), days=40)
    _upload(api, "12345")

    body = _sweep(api, apply="true")

    assert body["swept"] == ["web-gone"]
    assert not _marker(api, "web-gone").exists()


def test_apply_removes_orphan_files_and_keeps_the_real_ones(api):
    _upload(api, "12345")
    owner_dir = api.upload_dir / "12345"
    (owner_dir / ("f" * 16 + "_ghost.txt")).write_bytes(b"g" * 100)
    real = [p for p in owner_dir.iterdir() if not p.name.startswith("f" * 16)]

    body = _sweep(api, prefix="1", apply="true")

    assert body["orphans_removed"] == 1
    assert not (owner_dir / ("f" * 16 + "_ghost.txt")).exists()
    assert all(p.exists() for p in real)
    assert "12345" in _owners(api)


def test_deletion_is_logged_before_it_happens(api, caplog):
    """A crash mid-loop must still leave a record of what was about to go; the
    response only exists if the loop ends."""
    _idle_web_owner(api)
    _upload(api, "12345")

    with caplog.at_level("WARNING", logger="app.main"):
        _sweep(api, apply="true")

    lines = [r.getMessage() for r in caplog.records if "Sweeping" in r.getMessage()]
    assert lines and "web-idle" in lines[0] and "documents=1" in lines[0]


def test_an_owner_who_comes_back_mid_sweep_is_spared(api, monkeypatch):
    """The candidate list is built first and cleared second; a returning user
    can upload in between. Re-read under the lock before deleting."""
    _idle_web_owner(api)
    _upload(api, "12345")
    activity = api.app_state.activity
    real = activity.last_seen
    calls = {"web-idle": 0}

    def flaky(user_id):
        if user_id == "web-idle":
            calls[user_id] += 1
            if calls[user_id] > 1:
                # The re-check must happen with the owner's lock held: that is
                # what closes the window where a sweep deletes a file between an
                # upload's write and its indexing. threading.Lock is not
                # reentrant, so failing to take it here proves it is held.
                lock = api.app_state.upload_locks.for_owner(user_id)
                assert not lock.acquire(blocking=False), (
                    "the sweep re-read last_seen without holding the owner's lock"
                )
                return datetime.now(timezone.utc)  # came back
        return real(user_id)

    monkeypatch.setattr(activity, "last_seen", flaky)

    body = _sweep(api, apply="true")

    assert body["became_active"] == ["web-idle"]
    assert body["swept"] == []
    assert "web-idle" in _owners(api)


def test_one_failure_does_not_stop_the_others_or_lose_the_record(api, monkeypatch, caplog):
    import app.storage as storage

    _idle_web_owner(api, name="web-a")
    _idle_web_owner(api, name="web-b")
    _upload(api, "12345")
    real = storage.wipe_namespace

    def wipe(state, settings, user_id):
        if user_id == "web-a":
            raise RuntimeError("chroma refused")
        return real(state, settings, user_id)

    monkeypatch.setattr(storage, "wipe_namespace", wipe)

    with caplog.at_level("WARNING", logger="app.main"):
        body = _sweep(api, apply="true")

    assert body["swept"] == ["web-b"]
    assert [f["user_id"] for f in body["failed"]] == ["web-a"]
    assert "chroma refused" in body["failed"][0]["error"]
    assert "web-a" in _owners(api)
    # web-a was named before the wipe was attempted, so the wipe failing does not
    # erase the record: this is what pins log-then-delete rather than the reverse.
    lines = [r.getMessage() for r in caplog.records if "Sweeping" in r.getMessage()]
    assert any("user_id=web-a" in line for line in lines)


def test_the_sweep_is_dispatched_off_the_event_loop(tmp_path):
    """A full metadata scan plus a stat per owner, none of which yields."""
    import inspect

    from app.main import create_app
    from tests.conftest import make_settings

    app = create_app(make_settings(tmp_path))
    endpoint = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/maintenance/sweep")

    assert not inspect.iscoroutinefunction(endpoint)


# =========================
# The script
# =========================

@pytest.fixture()
def via_api(api, monkeypatch):
    """Route the script's one HTTP call through the test client.

    load_dotenv is stubbed out: the script calls it so a cron job finds the
    repository's .env, and on a machine that has one it would put every setting
    back into os.environ for the rest of the session - which is exactly what the
    session-scoped isolation fixture strips.
    """
    import scripts.sweep as sweep_cli

    monkeypatch.setattr(sweep_cli, "load_dotenv", lambda *args, **kwargs: None)
    captured = {}

    def call(url, api_key, idle_days, prefix, apply, force):
        captured.update(url=url, api_key=api_key, idle_days=idle_days,
                        prefix=prefix, apply=apply, force=force)
        response = api.post("/maintenance/sweep", params={
            "idle_days": idle_days, "prefix": prefix,
            "apply": "true" if apply else "false", "force": "true" if force else "false",
        })
        return response.status_code, response.json()

    monkeypatch.setattr(sweep_cli, "call_sweep", call)
    return captured


def test_the_script_dry_runs_web_sessions_by_default(api, via_api, capsys):
    import scripts.sweep as sweep_cli

    _idle_web_owner(api)

    code = sweep_cli.main([])

    output = capsys.readouterr().out
    assert code == 0
    assert via_api["prefix"] == "web-"
    assert via_api["idle_days"] == 30
    assert via_api["apply"] is False
    assert "DRY RUN" in output
    assert "web-idle" in output


def test_the_script_says_where_it_is_going(api, via_api, capsys):
    import scripts.sweep as sweep_cli

    sweep_cli.main(["--url", "http://backend:8000"])

    assert "Asking http://backend:8000" in capsys.readouterr().err
    assert via_api["url"] == "http://backend:8000"


def test_the_script_reports_a_refusal_and_exits_nonzero(api, via_api, capsys):
    import scripts.sweep as sweep_cli

    _idle_web_owner(api)

    code = sweep_cli.main(["--prefix", "", "--apply"])

    assert code == 3
    assert "REFUSED" in capsys.readouterr().out


def test_the_script_applies_when_told(api, via_api, capsys):
    import scripts.sweep as sweep_cli

    _idle_web_owner(api)
    _upload(api, "12345")

    code = sweep_cli.main(["--apply"])

    output = capsys.readouterr().out
    assert code == 0
    assert "APPLIED" in output
    assert "Swept: 1" in output
    assert "web-idle" not in _owners(api)


def test_the_script_can_print_json(api, via_api, capsys):
    import json

    import scripts.sweep as sweep_cli

    sweep_cli.main(["--json"])

    printed = json.loads(capsys.readouterr().out)
    assert printed["dry_run"] is True


def test_the_script_exits_two_when_nobody_answers(monkeypatch, capsys):
    import scripts.sweep as sweep_cli

    def refused(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(sweep_cli, "call_sweep", refused)

    assert sweep_cli.main(["--url", "http://127.0.0.1:9"]) == 2
    assert "has to be up" in capsys.readouterr().err


def test_the_script_shows_the_raw_answer_on_a_rejection(monkeypatch, capsys):
    """The operator is the audience, so a 401 must not be softened into
    'contact the operator'."""
    import scripts.sweep as sweep_cli

    monkeypatch.setattr(sweep_cli, "call_sweep",
                        lambda *a, **k: (401, {"detail": "Invalid or missing API key"}))

    assert sweep_cli.main([]) == 1
    err = capsys.readouterr().err
    assert "HTTP 401" in err and "Invalid or missing API key" in err


def test_the_request_the_script_builds(monkeypatch):
    """The real transport, with the network replaced at urlopen."""
    import io

    import scripts.sweep as sweep_cli

    seen = {}

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["key"] = request.get_header("X-api-key")
        return Response(b'{"dry_run": true}')

    monkeypatch.setattr(sweep_cli.urllib.request, "urlopen", urlopen)

    status, body = sweep_cli.call_sweep("http://backend:8000/", "secret", 45, "web-", True, False)

    assert status == 200 and body == {"dry_run": True}
    assert seen["method"] == "POST"
    assert seen["key"] == "secret"
    assert seen["url"].startswith("http://backend:8000/maintenance/sweep?")
    assert "idle_days=45" in seen["url"] and "prefix=web-" in seen["url"]
    assert "apply=true" in seen["url"] and "force=false" in seen["url"]


def test_orphans_are_reported_outside_the_prefix_but_not_removed(api):
    """The default cron line sweeps 'web-' sessions, and orphan files are most
    likely under the stable ids that prefix excludes. Reporting them costs one
    lookup; removing them would delete files the operator did not ask about."""
    _upload(api, "12345")
    ghost = api.upload_dir / "12345" / ("f" * 16 + "_ghost.txt")
    ghost.write_bytes(b"g" * 100)
    _idle_web_owner(api)

    body = _sweep(api, prefix="web-", apply="true", force="true")

    reported = {o["user_id"]: o for o in body["orphans"]}
    assert "12345" in reported
    assert reported["12345"]["in_scope"] is False
    assert body["orphans_removed"] == 0
    assert ghost.exists(), "a file outside the prefix was removed"


def test_orphans_inside_the_prefix_are_marked_in_scope(api):
    _upload(api, "web-live")
    (api.upload_dir / "web-live" / ("f" * 16 + "_ghost.txt")).write_bytes(b"g")

    body = _sweep(api, prefix="web-")

    assert [o["in_scope"] for o in body["orphans"]] == [True]


def test_the_script_says_which_orphans_it_will_not_touch(api, via_api, capsys):
    """A number with no explanation reads as work that was done."""
    import scripts.sweep as sweep_cli

    _upload(api, "12345")
    (api.upload_dir / "12345" / ("f" * 16 + "_ghost.txt")).write_bytes(b"g" * 100)

    sweep_cli.main([])

    output = capsys.readouterr().out
    assert "outside the prefix" in output
    assert "will not be cleaned by this run" in output


def test_startup_seeds_owners_that_predate_the_markers(tmp_path, fake_openai_embeddings):
    """Not just ActivityTracker.seed_missing in isolation - the lifespan has to
    call it. Without that, an owner who uploaded once before the upgrade and has
    asked questions ever since is dated by that upload and swept on the first
    run. Found when a stray edit removed the call and every test still passed.
    """
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import make_settings

    settings = make_settings(tmp_path)
    (settings.upload_dir / "12345").mkdir(parents=True)
    (settings.upload_dir / "12345" / ("a" * 16 + "_old.txt")).write_bytes(b"from before")
    _age(settings.upload_dir / "12345" / ("a" * 16 + "_old.txt"), days=90)

    app = create_app(settings)
    with TestClient(app):
        seen = app.state.activity.last_seen("12345")

    assert seen is not None
    assert seen > datetime.now(timezone.utc) - timedelta(minutes=1), (
        "the owner is still dated by a 90-day-old upload, so the next sweep takes them"
    )


def test_the_script_exits_one_when_an_owner_could_not_be_swept(api, via_api, monkeypatch, capsys):
    """A cron job that only reads the exit code has to notice a partial sweep."""
    import app.storage as storage
    import scripts.sweep as sweep_cli

    _idle_web_owner(api, name="web-a")
    _upload(api, "12345")
    real = storage.wipe_namespace

    def wipe(state, settings, user_id):
        if user_id == "web-a":
            raise RuntimeError("chroma refused")
        return real(state, settings, user_id)

    monkeypatch.setattr(storage, "wipe_namespace", wipe)

    code = sweep_cli.main(["--apply"])

    assert code == 1
    assert "FAILED" in capsys.readouterr().err


def test_the_bot_shows_the_quota_when_listing(monkeypatch):
    """format_document_list is tested as a pure function; this is the wiring.
    Dropping the second argument would silently stop showing the usage line
    that the docs promise in both clients."""
    import asyncio
    from types import SimpleNamespace

    import clients.telegram_bot as bot

    class FakeResponse:
        status_code = 200
        headers = {}

        def json(self):
            return {
                "documents": [{"source": "policy.docx", "chunks": 3}],
                "total_chunks": 3,
                "quota": {"documents": 1, "max_documents": 200,
                          "bytes": 2048, "max_bytes": 1024 ** 3},
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(bot.httpx, "AsyncClient", FakeClient)

    sent = []

    async def reply_html(text, **kwargs):
        sent.append(text)

    update = SimpleNamespace(
        message=SimpleNamespace(reply_html=reply_html, reply_text=reply_html),
        effective_user=SimpleNamespace(id=7),
    )

    asyncio.run(bot.documents_command(update, SimpleNamespace(user_data={})))

    assert sent, "the bot replied with nothing"
    assert "1 of 200 documents, 2 KB of 1 GB" in sent[0]
