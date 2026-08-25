"""Snapshot and restore.

A backup nobody has restored is a rumour, so the centre of this file is a full
round trip: index a document through the API, take a snapshot, delete
everything, restore, and ask a question that the document answers.

The rest is about refusing. A restore that half-works, or that quietly installs
vectors built by a different embedding model, is worse than one that stops and
says why.
"""
import io
import json
import tarfile

import pytest
from fastapi.testclient import TestClient

from app.backup import (
    ARCHIVE_PREFIX,
    CHROMA_DB_NAME,
    MANIFEST_NAME,
    BackupError,
    chunk_count,
    create_backup,
    incompatibilities,
    read_manifest,
    restore_backup,
    storage_locations,
    verify_members,
)
from app.main import create_app
from tests.conftest import TEST_API_KEY, make_settings

TEXT = b"Annual leave for an engineer is twenty eight calendar days."


def _locations(tmp_path):
    return {
        "chroma": tmp_path / "chroma",
        "uploads": tmp_path / "uploads",
        "feedback": tmp_path / "feedback",
    }


def _populate(tmp_path, with_feedback=True):
    """A data directory shaped like a real one, without running the app."""
    locations = _locations(tmp_path)
    (locations["chroma"]).mkdir(parents=True)
    _make_sqlite(locations["chroma"] / CHROMA_DB_NAME)
    (locations["chroma"] / "index-segment").mkdir()
    (locations["chroma"] / "index-segment" / "data_level0.bin").write_bytes(b"\x00" * 64)

    user_dir = locations["uploads"] / "u1"
    user_dir.mkdir(parents=True)
    (user_dir / "abcdef0123456789_policy.txt").write_bytes(TEXT)

    if with_feedback:
        locations["feedback"].mkdir(parents=True)
        (locations["feedback"] / "feedback.jsonl").write_text(
            '{"rating": "down", "question": "q"}\n{"rating": "up", "question": "q2"}\n',
            encoding="utf-8",
        )
    return locations


def _make_sqlite(path):
    import sqlite3

    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO embeddings (id) VALUES (?)", [(i,) for i in range(7)]
        )


# =========================
# Where the data is
# =========================

def test_the_three_directories_are_the_ones_settings_names(tmp_path):
    settings = make_settings(tmp_path)

    locations = storage_locations(settings)

    assert locations["chroma"] == settings.chroma_persist_dir
    assert locations["uploads"] == settings.upload_dir
    assert locations["feedback"] == settings.feedback_dir


def test_the_locations_are_found_without_an_api_key(monkeypatch):
    """Copying files should not require an OpenAI key, and an operator
    restoring onto a fresh machine may not have one yet."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/srv/chroma")

    assert str(storage_locations()["chroma"]) in ("/srv/chroma", "\\srv\\chroma")


def test_the_fallbacks_are_settings_own_defaults(monkeypatch):
    """Otherwise the tool would back up a different directory from the one the
    app writes to, and nobody would notice until a restore came up empty."""
    from app.config import Settings

    for name in ("CHROMA_PERSIST_DIR", "UPLOAD_DIR", "FEEDBACK_DIR"):
        monkeypatch.delenv(name, raising=False)

    locations = storage_locations()
    fields = Settings.model_fields

    assert locations["chroma"] == fields["chroma_persist_dir"].default
    assert locations["uploads"] == fields["upload_dir"].default
    assert locations["feedback"] == fields["feedback_dir"].default


# =========================
# Taking a snapshot
# =========================

def test_an_archive_is_written(tmp_path):
    locations = _populate(tmp_path)

    manifest = create_backup(tmp_path / "out", locations=locations)

    archive = manifest["archive"]
    assert archive.exists()
    assert archive.name.startswith(ARCHIVE_PREFIX)
    assert archive.name.endswith(".tar.gz")


def test_the_archive_holds_all_three_directories(tmp_path):
    locations = _populate(tmp_path)

    archive = create_backup(tmp_path / "out", locations=locations)["archive"]

    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert MANIFEST_NAME in names
    assert f"chroma/{CHROMA_DB_NAME}" in names
    assert "chroma/index-segment/data_level0.bin" in names
    assert "uploads/u1/abcdef0123456789_policy.txt" in names
    assert "feedback/feedback.jsonl" in names


def test_the_manifest_records_what_a_restore_has_to_agree_with(tmp_path):
    """An archive alone says nothing about whether it fits the deployment it is
    being restored into."""
    locations = _populate(tmp_path)

    manifest = create_backup(tmp_path / "out", locations=locations)

    settings = manifest["settings"]
    assert settings["embedding_model"]
    assert settings["collection_name"]
    assert settings["chunk_size"]


def test_the_manifest_counts_what_it_took(tmp_path):
    locations = _populate(tmp_path)

    counts = create_backup(tmp_path / "out", locations=locations)["counts"]

    assert counts["chunks"] == 7
    assert counts["uploads"] == 1
    assert counts["ratings"] == 2


def test_every_file_is_hashed(tmp_path):
    locations = _populate(tmp_path)

    members = create_backup(tmp_path / "out", locations=locations)["members"]

    assert members
    assert all(len(entry["sha256"]) == 64 for entry in members.values())
    assert MANIFEST_NAME not in members, "the manifest cannot hash itself"


def test_a_missing_directory_is_skipped_not_fatal(tmp_path):
    """A deployment that has never collected a rating has no feedback dir."""
    locations = _populate(tmp_path, with_feedback=False)

    manifest = create_backup(tmp_path / "out", locations=locations)

    with tarfile.open(manifest["archive"]) as tar:
        assert not [n for n in tar.getnames() if n.startswith("feedback")]


def test_backing_up_nothing_is_an_error(tmp_path):
    """An empty archive would restore cleanly over real data, which is the
    worst way for this to fail."""
    with pytest.raises(BackupError, match="None of the data directories"):
        create_backup(tmp_path / "out", locations=_locations(tmp_path))


def test_the_copy_is_a_working_database(tmp_path):
    """Copied through SQLite's own backup API, so a torn read is not possible;
    this is the check that the copy is openable at all."""
    locations = _populate(tmp_path)
    archive = create_backup(tmp_path / "out", locations=locations)["archive"]

    extracted = tmp_path / "extracted"
    with tarfile.open(archive) as tar:
        tar.extractall(extracted)

    assert chunk_count(extracted / "chroma" / CHROMA_DB_NAME) == 7


def test_a_corrupt_database_stops_the_backup_with_advice(tmp_path):
    """Handing over an archive that looks fine is the failure worth avoiding.

    Asserted on BackupError rather than Exception: sqlite3 raises here on its
    own, so a test that accepted any exception would also have passed while the
    only thing an operator saw was a raw traceback.
    """
    locations = _populate(tmp_path)
    (locations["chroma"] / CHROMA_DB_NAME).write_bytes(b"this is not a database")

    with pytest.raises(BackupError) as exc:
        create_backup(tmp_path / "out", locations=locations)

    assert "stop it and try again" in str(exc.value)


def test_a_failed_copy_leaves_no_archive_behind(tmp_path):
    """Half an archive is worse than none: it would restore cleanly."""
    locations = _populate(tmp_path)
    (locations["chroma"] / CHROMA_DB_NAME).write_bytes(b"this is not a database")

    with pytest.raises(BackupError):
        create_backup(tmp_path / "out", locations=locations)

    assert not list((tmp_path / "out").glob("*.tar.gz"))


def test_an_unreadable_chunk_count_is_reported_as_unknown(tmp_path):
    """The embeddings table is ChromaDB's schema, not ours."""
    import sqlite3

    path = tmp_path / "other.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE something_else (id INTEGER)")

    assert chunk_count(path) is None


# =========================
# Checking an archive
# =========================

def test_an_intact_archive_verifies(tmp_path):
    locations = _populate(tmp_path)
    archive = create_backup(tmp_path / "out", locations=locations)["archive"]

    assert verify_members(archive, read_manifest(archive)) == []


def test_a_tampered_member_is_detected(tmp_path):
    locations = _populate(tmp_path)
    archive = create_backup(tmp_path / "out", locations=locations)["archive"]
    manifest = read_manifest(archive)
    name = "uploads/u1/abcdef0123456789_policy.txt"
    manifest["members"][name]["sha256"] = "0" * 64

    assert any(name in line for line in verify_members(archive, manifest))


def test_a_member_the_manifest_expects_but_the_archive_lacks_is_detected(tmp_path):
    locations = _populate(tmp_path)
    archive = create_backup(tmp_path / "out", locations=locations)["archive"]
    manifest = read_manifest(archive)
    manifest["members"]["uploads/u1/ghost.txt"] = {"bytes": 1, "sha256": "0" * 64}

    assert any("ghost" in line for line in verify_members(archive, manifest))


def test_an_archive_without_a_manifest_is_refused(tmp_path):
    stranger = tmp_path / "stranger.tar.gz"
    with tarfile.open(stranger, "w:gz") as tar:
        info = tarfile.TarInfo("hello.txt")
        info.size = 5
        tar.addfile(info, io.BytesIO(b"hello"))

    with pytest.raises(BackupError, match="not written by this tool"):
        read_manifest(stranger)


# =========================
# Does it fit this deployment
# =========================

def test_a_matching_archive_has_no_complaints(tmp_path):
    settings = make_settings(tmp_path)
    manifest = create_backup(
        tmp_path / "out", settings=settings, locations=_populate(tmp_path)
    )

    assert incompatibilities(manifest, settings) == []


def test_a_different_embedding_model_is_an_incompatibility(tmp_path):
    """The reason this check exists: vectors built by another model are not
    visibly wrong, they are quietly meaningless."""
    manifest = create_backup(
        tmp_path / "out",
        settings=make_settings(tmp_path, embedding_model="text-embedding-3-large"),
        locations=_populate(tmp_path),
    )

    problems = incompatibilities(manifest, make_settings(tmp_path))

    assert any("EMBEDDING_MODEL" in line for line in problems)


def test_a_different_collection_name_is_an_incompatibility(tmp_path):
    """Duller, but it would leave the app opening an empty collection."""
    manifest = create_backup(
        tmp_path / "out",
        settings=make_settings(tmp_path, collection_name="other"),
        locations=_populate(tmp_path),
    )

    problems = incompatibilities(manifest, make_settings(tmp_path))

    assert any("COLLECTION_NAME" in line for line in problems)


def test_chunking_differences_are_not_an_incompatibility(tmp_path):
    """Re-chunking would change future uploads, not break existing vectors, so
    refusing on it would be theatre."""
    manifest = create_backup(
        tmp_path / "out",
        settings=make_settings(tmp_path, chunk_size=500, chunk_overlap=50),
        locations=_populate(tmp_path),
    )

    assert incompatibilities(manifest, make_settings(tmp_path)) == []


# =========================
# Putting it back
# =========================

def _archive_of(tmp_path, settings=None):
    source = tmp_path / "source"
    locations = _populate(source)
    return create_backup(tmp_path / "out", settings=settings, locations=locations)["archive"]


def test_a_restore_puts_the_files_back(tmp_path):
    archive = _archive_of(tmp_path)
    target = _locations(tmp_path / "target")

    result = restore_backup(archive, locations=target)

    assert (target["chroma"] / CHROMA_DB_NAME).exists()
    assert (target["uploads"] / "u1" / "abcdef0123456789_policy.txt").read_bytes() == TEXT
    assert (target["feedback"] / "feedback.jsonl").exists()
    assert set(result["restored"]) == {"chroma", "uploads", "feedback"}


def test_the_index_files_come_back_too(tmp_path):
    """SQLite alone is not the index; the HNSW segments are separate files."""
    archive = _archive_of(tmp_path)
    target = _locations(tmp_path / "target")

    restore_backup(archive, locations=target)

    assert (target["chroma"] / "index-segment" / "data_level0.bin").exists()


def test_a_restore_over_existing_data_is_refused(tmp_path):
    archive = _archive_of(tmp_path)
    target = _locations(tmp_path / "target")
    target["uploads"].mkdir(parents=True)
    (target["uploads"] / "precious.txt").write_bytes(b"do not lose me")

    with pytest.raises(BackupError, match="already hold data"):
        restore_backup(archive, locations=target)

    assert (target["uploads"] / "precious.txt").exists()


def test_overwrite_replaces_it(tmp_path):
    archive = _archive_of(tmp_path)
    target = _locations(tmp_path / "target")
    target["uploads"].mkdir(parents=True)
    (target["uploads"] / "stale.txt").write_bytes(b"old")

    restore_backup(archive, locations=target, overwrite=True)

    assert not (target["uploads"] / "stale.txt").exists()
    assert (target["uploads"] / "u1" / "abcdef0123456789_policy.txt").exists()


def test_an_empty_directory_does_not_count_as_data(tmp_path):
    """Docker creates the mount points, so they exist before anything is in
    them; refusing on that would make the first restore impossible."""
    archive = _archive_of(tmp_path)
    target = _locations(tmp_path / "target")
    for path in target.values():
        path.mkdir(parents=True)

    restore_backup(archive, locations=target)

    assert (target["chroma"] / CHROMA_DB_NAME).exists()


def test_a_mismatched_archive_is_refused(tmp_path):
    archive = _archive_of(
        tmp_path, settings=make_settings(tmp_path, embedding_model="text-embedding-3-large")
    )
    target = _locations(tmp_path / "target")

    with pytest.raises(BackupError, match="does not fit this deployment"):
        restore_backup(archive, settings=make_settings(tmp_path), locations=target)

    assert not target["chroma"].exists(), "nothing should have been written"


def test_force_restores_a_mismatch_but_says_so(tmp_path):
    archive = _archive_of(
        tmp_path, settings=make_settings(tmp_path, embedding_model="text-embedding-3-large")
    )
    target = _locations(tmp_path / "target")

    result = restore_backup(
        archive, settings=make_settings(tmp_path), locations=target, force=True
    )

    assert (target["chroma"] / CHROMA_DB_NAME).exists()
    assert any("EMBEDDING_MODEL" in line for line in result["warnings"])


def test_a_tampered_archive_is_refused_before_anything_is_touched(tmp_path):
    archive = _archive_of(tmp_path)
    # Rewrite one member, leaving the manifest's hash of it in place.
    with tarfile.open(archive) as tar:
        members = [(m, tar.extractfile(m.name)) for m in tar.getmembers()]
        contents = {m.name: (m, handle.read() if handle else b"") for m, handle in members}
    with tarfile.open(archive, "w:gz") as tar:
        for name, (info, data) in contents.items():
            if name.endswith("policy.txt"):
                data = b"tampered"
                info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    target = _locations(tmp_path / "target")
    with pytest.raises(BackupError, match="does not match its manifest"):
        restore_backup(archive, locations=target)

    assert not target["uploads"].exists()


def test_a_missing_archive_is_an_error(tmp_path):
    with pytest.raises(BackupError, match="No such archive"):
        restore_backup(tmp_path / "nope.tar.gz", locations=_locations(tmp_path))


# =========================
# A hostile archive
# =========================

def _hostile(tmp_path, member_name, payload=b"pwned", symlink_to=None):
    """An archive with a member that tries to escape the target directory."""
    archive = tmp_path / "hostile.tar.gz"
    manifest = json.dumps({"settings": {}, "counts": {}, "members": {}}).encode("utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))
        if symlink_to:
            link = tarfile.TarInfo(member_name)
            link.type = tarfile.SYMTYPE
            link.linkname = symlink_to
            tar.addfile(link)
        else:
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return archive


def test_a_member_reaching_outside_the_target_is_refused(tmp_path):
    """An archive is operator input, but not necessarily one this tool wrote."""
    archive = _hostile(tmp_path, "../../escaped.txt")

    with pytest.raises(BackupError, match="outside the target"):
        restore_backup(archive, locations=_locations(tmp_path / "target"))

    assert not (tmp_path / "escaped.txt").exists()


def test_an_absolute_member_is_refused(tmp_path):
    archive = _hostile(tmp_path, "/tmp/escaped.txt")

    with pytest.raises(BackupError):
        restore_backup(archive, locations=_locations(tmp_path / "target"))


def test_a_symlink_member_is_refused(tmp_path):
    """A link is how a later member gets written through it to somewhere else."""
    archive = _hostile(tmp_path, "chroma/link", symlink_to="/etc/passwd")

    with pytest.raises(BackupError, match="links are not allowed"):
        restore_backup(archive, locations=_locations(tmp_path / "target"))


# =========================
# The round trip that matters
# =========================

def _index_a_document(settings, filename="policy.txt", text=TEXT):
    """Build a real index through the API and return what /documents said."""
    from tests.conftest import FakeChatClient

    app = create_app(settings)
    with TestClient(app) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        app.state.rag_chain.client = FakeChatClient()
        upload = client.post(
            "/upload", params={"user_id": "u1"},
            files={"file": (filename, text, "text/plain")},
        )
        assert upload.status_code == 200, upload.text
        return client.get("/documents", params={"user_id": "u1"}).json()


def test_a_restored_snapshot_still_answers_questions(tmp_path, fake_openai_embeddings):
    """The whole point, end to end: index a document, snapshot, restore onto a
    clean machine, and ask something only that document answers.

    A backup nobody has restored is a rumour.

    Restored into fresh directories rather than over the originals, because that
    is the case that matters - a volume is gone and a new host is coming up -
    and because the process that built the index still holds the ChromaDB files
    open. Replacing them in place is what stopping the backend is for, and both
    scripts refuse while it answers.
    """
    from tests.conftest import FakeChatClient

    original = make_settings(tmp_path / "before")
    before = _index_a_document(original)
    assert before["total_chunks"] >= 1

    manifest = create_backup(
        tmp_path / "out", settings=original, locations=storage_locations(original)
    )

    revived_settings = make_settings(tmp_path / "after")
    restore_backup(
        manifest["archive"],
        settings=revived_settings,
        locations=storage_locations(revived_settings),
    )

    revived = create_app(revived_settings)
    with TestClient(revived) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        revived.state.rag_chain.client = FakeChatClient()

        assert client.get("/ready").status_code == 200

        after = client.get("/documents", params={"user_id": "u1"}).json()
        assert after == before, "the restored index does not match what was taken"

        answer = client.post("/query", json={
            "question": "How many leave days does an engineer get?",
            "language": "Auto", "user_id": "u1",
        })
        assert answer.status_code == 200, answer.text
        assert [s["source"] for s in answer.json()["sources"]] == ["policy.txt"]


def test_the_raw_file_survives_the_round_trip(tmp_path, fake_openai_embeddings):
    """Chunks are searchable; the uploaded bytes are what a person asked to keep,
    and they are what a re-index would start from."""
    original = make_settings(tmp_path / "before")
    _index_a_document(original)

    manifest = create_backup(
        tmp_path / "out", settings=original, locations=storage_locations(original)
    )

    revived_settings = make_settings(tmp_path / "after")
    locations = storage_locations(revived_settings)
    restore_backup(manifest["archive"], settings=revived_settings, locations=locations)

    stored = list(locations["uploads"].rglob("*policy.txt"))
    assert stored, "the raw upload did not come back"
    assert stored[0].read_bytes() == TEXT


def test_a_restored_deployment_can_be_added_to(tmp_path, fake_openai_embeddings):
    """A restore that answers questions but cannot take a new document would be
    a museum piece."""
    original = make_settings(tmp_path / "before")
    _index_a_document(original)
    manifest = create_backup(
        tmp_path / "out", settings=original, locations=storage_locations(original)
    )

    revived_settings = make_settings(tmp_path / "after")
    restore_backup(
        manifest["archive"], settings=revived_settings,
        locations=storage_locations(revived_settings),
    )

    listing = _index_a_document(
        revived_settings, filename="second.txt",
        text=b"Sick leave is paid from the first day of absence.",
    )

    assert sorted(d["source"] for d in listing["documents"]) == [
        "policy.txt", "second.txt"
    ]


def test_the_ratings_come_back_too(tmp_path, fake_openai_embeddings):
    """They are the one thing users are asked to contribute, and until this
    stage they were not even on a volume."""
    from app.feedback import read_records

    original = make_settings(tmp_path / "before")
    app = create_app(original)
    with TestClient(app) as client:
        client.headers["X-API-Key"] = TEST_API_KEY
        assert client.post("/feedback", json={
            "rating": "down", "user_id": "u1", "question": "Leave days?",
            "answer": "Thirty five.", "sources": ["policy.txt"], "client": "web",
        }).status_code == 200

    manifest = create_backup(
        tmp_path / "out", settings=original, locations=storage_locations(original)
    )
    assert manifest["counts"]["ratings"] == 1

    revived_settings = make_settings(tmp_path / "after")
    locations = storage_locations(revived_settings)
    restore_backup(manifest["archive"], settings=revived_settings, locations=locations)

    records = read_records(locations["feedback"] / "feedback.jsonl")
    assert [r["question"] for r in records] == ["Leave days?"]


# =========================
# The two commands
# =========================

@pytest.fixture()
def offline_backend(monkeypatch):
    """Nothing is answering, which is the state both scripts insist on."""
    import scripts.backup as backup_cli

    monkeypatch.setattr(backup_cli, "backend_is_up", lambda url: False)
    return backup_cli


@pytest.fixture()
def live_backend(monkeypatch):
    import scripts.backup as backup_cli
    import scripts.restore as restore_cli

    monkeypatch.setattr(backup_cli, "backend_is_up", lambda url: True)
    monkeypatch.setattr(restore_cli, "backend_is_up", lambda url: True)
    return backup_cli


def test_the_backup_command_writes_an_archive(tmp_path, offline_backend, monkeypatch, capsys):
    locations = _populate(tmp_path)
    monkeypatch.setattr(offline_backend, "storage_locations", lambda: locations)

    code = offline_backend.main(["--output", str(tmp_path / "out")])

    output = capsys.readouterr().out
    assert code == 0
    assert list((tmp_path / "out").glob(f"{ARCHIVE_PREFIX}*.tar.gz"))
    assert "chunks: 7" in output
    assert "embedding model:" in output


def test_the_backup_command_refuses_while_the_backend_answers(
    tmp_path, live_backend, monkeypatch, capsys
):
    """ChromaDB's database and its index files are written separately, and a copy
    taken mid-write can catch them out of step. Nothing the script does can fix
    that, so it says so instead of handing over an archive that looks fine."""
    locations = _populate(tmp_path)
    monkeypatch.setattr(live_backend, "storage_locations", lambda: locations)

    code = live_backend.main(["--output", str(tmp_path / "out")])

    assert code == 2
    assert "docker compose stop backend" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_live_takes_the_snapshot_anyway(tmp_path, live_backend, monkeypatch):
    """For a deployment where a minute of downtime is worse than a small risk."""
    locations = _populate(tmp_path)
    monkeypatch.setattr(live_backend, "storage_locations", lambda: locations)

    code = live_backend.main(["--output", str(tmp_path / "out"), "--live"])

    assert code == 0
    assert list((tmp_path / "out").glob("*.tar.gz"))


def test_the_backup_command_reports_failure_rather_than_a_traceback(
    tmp_path, offline_backend, monkeypatch, capsys
):
    monkeypatch.setattr(
        offline_backend, "storage_locations", lambda: _locations(tmp_path / "empty")
    )

    code = offline_backend.main(["--output", str(tmp_path / "out")])

    assert code == 1
    assert "Backup failed" in capsys.readouterr().err


def test_the_backup_command_can_print_json(tmp_path, offline_backend, monkeypatch, capsys):
    """So a cron job can record what it took."""
    locations = _populate(tmp_path)
    monkeypatch.setattr(offline_backend, "storage_locations", lambda: locations)

    offline_backend.main(["--output", str(tmp_path / "out"), "--json"])

    printed = json.loads(capsys.readouterr().out)
    assert printed["counts"]["chunks"] == 7
    assert printed["archive"].endswith(".tar.gz")


def test_a_probe_that_answers_503_still_counts_as_running(monkeypatch):
    """A backend that is up but not ready is still writing files."""
    import urllib.error

    import scripts.backup as backup_cli

    def unready(url, timeout=None):
        raise urllib.error.HTTPError(url, 503, "not ready", {}, None)

    monkeypatch.setattr(backup_cli.urllib.request, "urlopen", unready)

    assert backup_cli.backend_is_up("http://127.0.0.1:8000") is True


def test_a_refused_connection_means_nobody_is_home(monkeypatch):
    import scripts.backup as backup_cli

    def refused(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(backup_cli.urllib.request, "urlopen", refused)

    assert backup_cli.backend_is_up("http://127.0.0.1:8000") is False


def test_the_restore_command_puts_it_back(tmp_path, monkeypatch, capsys):
    import scripts.restore as restore_cli

    archive = _archive_of(tmp_path)
    target = _locations(tmp_path / "target")
    monkeypatch.setattr(restore_cli, "backend_is_up", lambda url: False)
    monkeypatch.setattr(restore_cli, "storage_locations", lambda: target)

    code = restore_cli.main([str(archive)])

    assert code == 0
    assert (target["chroma"] / CHROMA_DB_NAME).exists()
    assert "Restored:" in capsys.readouterr().out


def test_the_restore_command_refuses_while_the_backend_answers(
    tmp_path, live_backend, monkeypatch, capsys
):
    """It holds the ChromaDB files open and would go on serving a state that no
    longer exists on disk."""
    import scripts.restore as restore_cli

    archive = _archive_of(tmp_path)
    target = _locations(tmp_path / "target")
    monkeypatch.setattr(restore_cli, "storage_locations", lambda: target)

    code = restore_cli.main([str(archive)])

    assert code == 2
    assert "docker compose stop backend" in capsys.readouterr().err
    assert not target["chroma"].exists()


def test_inspect_changes_nothing(tmp_path, monkeypatch, capsys):
    """The command an operator runs on an archive they are not sure about."""
    import scripts.restore as restore_cli

    archive = _archive_of(tmp_path)
    target = _locations(tmp_path / "target")
    monkeypatch.setattr(restore_cli, "storage_locations", lambda: target)
    # Left running on purpose: --inspect must not care.
    monkeypatch.setattr(restore_cli, "backend_is_up", lambda url: True)

    code = restore_cli.main([str(archive), "--inspect"])

    output = capsys.readouterr().out
    assert code == 0
    assert "Contents: intact" in output
    assert "Fits this deployment: yes" in output
    assert not target["chroma"].exists()


def test_inspect_names_a_mismatch_and_fails(tmp_path, monkeypatch, capsys):
    import scripts.restore as restore_cli

    archive = _archive_of(
        tmp_path,
        settings=make_settings(tmp_path, embedding_model="text-embedding-3-large"),
    )
    monkeypatch.setattr(restore_cli, "storage_locations", lambda: _locations(tmp_path / "t"))

    code = restore_cli.main([str(archive), "--inspect"])

    assert code == 1
    assert "EMBEDDING_MODEL" in capsys.readouterr().out


def test_the_restore_command_explains_a_refusal(tmp_path, monkeypatch, capsys):
    import scripts.restore as restore_cli

    archive = _archive_of(tmp_path)
    target = _locations(tmp_path / "target")
    target["uploads"].mkdir(parents=True)
    (target["uploads"] / "precious.txt").write_bytes(b"keep me")
    monkeypatch.setattr(restore_cli, "backend_is_up", lambda url: False)
    monkeypatch.setattr(restore_cli, "storage_locations", lambda: target)

    code = restore_cli.main([str(archive)])

    assert code == 1
    assert "--overwrite" in capsys.readouterr().err
    assert (target["uploads"] / "precious.txt").exists()


def test_a_file_that_is_not_an_archive_is_reported_plainly(tmp_path, monkeypatch, capsys):
    import scripts.restore as restore_cli

    junk = tmp_path / "notes.txt"
    junk.write_text("this is not a backup", encoding="utf-8")
    monkeypatch.setattr(restore_cli, "storage_locations", lambda: _locations(tmp_path / "t"))

    code = restore_cli.main([str(junk)])

    assert code == 1
    assert "Cannot read the archive" in capsys.readouterr().err
