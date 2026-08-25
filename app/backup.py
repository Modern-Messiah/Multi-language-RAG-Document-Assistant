"""Snapshot the data and put it back.

Everything this assistant knows lives in three directories: the ChromaDB index,
the raw uploads, and the collected ratings. Losing the volume lost all of it,
and there was no procedure - not even a documented list of what to copy. A
backup nobody has restored is a rumour, so restore is here too, and the test
suite does a full round trip: index a document, back up, wipe, restore, ask a
question and get the document back.

Two properties are worth more than the copying itself.

**The manifest.** An archive alone tells you nothing about whether it fits the
deployment you are restoring into. Vectors built by one embedding model are
meaningless to another - the startup guard in embeddings.py exists for exactly
that - so the manifest records the model, the collection name and the chunking,
and restore refuses on a mismatch instead of quietly producing a corpus that
answers nothing.

**Consistency.** ChromaDB is a SQLite database plus HNSW index files written
separately. Copying it under load can catch the two out of step, and no amount
of care here can fix that: the fix is to stop the backend first. So the backup
checks whether the backend is answering and says so, rather than handing over an
archive that looks fine.
"""
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tarfile
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
ARCHIVE_PREFIX = "rag-backup-"
ARCHIVE_SUFFIX = ".tar.gz"

# Names inside the archive. Fixed rather than mirroring the host's directory
# names, so an archive can be restored into a deployment configured differently.
CHROMA_MEMBER = "chroma"
UPLOADS_MEMBER = "uploads"
FEEDBACK_MEMBER = "feedback"

# ChromaDB's metadata database. Copied through SQLite's own backup API, which
# produces a file that is internally consistent even if something is writing.
CHROMA_DB_NAME = "chroma.sqlite3"

_READ_CHUNK = 1024 * 1024


class BackupError(Exception):
    """Anything that should stop a backup or a restore with a message."""


# =========================
# Where the data is
# =========================

def storage_locations(settings=None) -> dict:
    """The three directories a snapshot covers.

    Works without a Settings instance on purpose: copying files should not
    require an OpenAI key, and an operator restoring onto a fresh machine may
    not have one to hand yet. The env var names and the fallbacks are Settings'
    own, so there is still one source of truth for where things live.
    """
    if settings is not None:
        return {
            CHROMA_MEMBER: Path(settings.chroma_persist_dir),
            UPLOADS_MEMBER: Path(settings.upload_dir),
            FEEDBACK_MEMBER: Path(settings.feedback_dir),
        }

    fields = Settings.model_fields

    def located(field: str) -> Path:
        raw = os.getenv(field.upper(), "").strip()
        return Path(raw) if raw else Path(fields[field].default)

    return {
        CHROMA_MEMBER: located("chroma_persist_dir"),
        UPLOADS_MEMBER: located("upload_dir"),
        FEEDBACK_MEMBER: located("feedback_dir"),
    }


def _settings_summary(settings=None) -> dict:
    """The configuration a restore has to agree with."""
    if settings is not None:
        return {
            "embedding_model": settings.embedding_model,
            "collection_name": settings.collection_name,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }

    fields = Settings.model_fields
    return {
        name: os.getenv(name.upper(), "").strip() or fields[name].default
        for name in ("embedding_model", "collection_name")
    } | {
        name: int(os.getenv(name.upper(), "") or fields[name].default)
        for name in ("chunk_size", "chunk_overlap")
    }


# =========================
# Copying
# =========================

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_sqlite(source: Path, target: Path) -> None:
    """Copy a live SQLite file through its own backup API.

    shutil.copy of a database being written to can produce a file that no
    longer parses. This cannot make the HNSW index files consistent with it -
    only stopping the writer does that - but it removes one of the two ways a
    live snapshot goes wrong.

    A database SQLite cannot read stops the backup here, with advice. There was
    once an integrity_check on the copy as well; it could never fire, because
    the copy API refuses a source it cannot read, and an unreachable check is
    worse than none - it reads as protection that is not there.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # closing(), not `with sqlite3.connect(...)`: that context manager
        # commits a transaction and leaves the connection open. The handle then
        # outlives this function, and on Windows the staging directory cannot be
        # removed - which is how this was found, in the error path where a
        # traceback keeps the frames alive.
        with closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as origin:
            with closing(sqlite3.connect(target)) as copy:
                origin.backup(copy)
    except sqlite3.Error as exc:
        raise BackupError(
            f"{source} could not be copied: {exc}. "
            "If the backend is running, stop it and try again; otherwise the "
            "database is damaged and this snapshot would be worthless."
        ) from exc


def _copy_tree(source: Path, target: Path) -> int:
    """Copy a directory, returning how many files were copied."""
    copied = 0
    for item in sorted(source.rglob("*")):
        if item.is_dir():
            continue
        relative = item.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        copied += 1
    return copied


def chunk_count(database: Path):
    """Rows in ChromaDB's embeddings table, or None if it cannot be read.

    Informational: it is a number an operator can compare before and after a
    restore. Not a schema this code controls, so a missing table is reported as
    unknown rather than as an error.
    """
    try:
        with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
            return connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    except sqlite3.Error:
        return None


# =========================
# Backup
# =========================

def archive_name(stamp=None) -> str:
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ARCHIVE_PREFIX}{stamp}{ARCHIVE_SUFFIX}"


def create_backup(output_dir, settings=None, locations=None) -> dict:
    """Write one archive and return its manifest, with `archive` added.

    Raises BackupError if there is nothing to back up: an empty archive would
    restore cleanly over real data, which is the worst possible way for this to
    fail.
    """
    locations = locations or storage_locations(settings)
    output_dir = Path(output_dir)

    present = {name: path for name, path in locations.items() if path.exists()}
    if not present:
        raise BackupError(
            "None of the data directories exist: "
            + ", ".join(str(p) for p in locations.values())
        )

    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        counts = {}

        if CHROMA_MEMBER in present:
            source = present[CHROMA_MEMBER]
            target = staging / CHROMA_MEMBER
            database = source / CHROMA_DB_NAME
            if database.exists():
                _copy_sqlite(database, target / CHROMA_DB_NAME)
                # Everything else in the directory - the HNSW segments - copied
                # as files, because nothing else can be done about them.
                for item in sorted(source.rglob("*")):
                    if item.is_dir() or item == database:
                        continue
                    destination = target / item.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, destination)
            else:
                _copy_tree(source, target)

            copied_db = target / CHROMA_DB_NAME
            if copied_db.exists():
                counts["chunks"] = chunk_count(copied_db)

        if UPLOADS_MEMBER in present:
            counts["uploads"] = _copy_tree(
                present[UPLOADS_MEMBER], staging / UPLOADS_MEMBER
            )

        if FEEDBACK_MEMBER in present:
            counts["ratings"] = _sum_lines(present[FEEDBACK_MEMBER])
            _copy_tree(present[FEEDBACK_MEMBER], staging / FEEDBACK_MEMBER)

        members = {}
        for item in sorted(staging.rglob("*")):
            if item.is_file():
                members[item.relative_to(staging).as_posix()] = {
                    "bytes": item.stat().st_size,
                    "sha256": _sha256(item),
                }

        manifest = {
            "created_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "settings": _settings_summary(settings),
            "counts": counts,
            "members": members,
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        archive = output_dir / archive_name()
        with tarfile.open(archive, "w:gz") as tar:
            for item in sorted(staging.rglob("*")):
                tar.add(item, arcname=item.relative_to(staging).as_posix())

    return {**manifest, "archive": archive}


def _sum_lines(directory: Path) -> int:
    total = 0
    for item in directory.rglob("*.jsonl"):
        with item.open("r", encoding="utf-8", errors="replace") as handle:
            total += sum(1 for line in handle if line.strip())
    return total


# =========================
# Restore
# =========================

def _open_archive(archive) -> tarfile.TarFile:
    """tarfile.ReadError is not an OSError, so a caller that guards against
    unreadable files still got a traceback for "this is not an archive"."""
    try:
        return tarfile.open(archive, "r:gz")
    except tarfile.TarError as exc:
        raise BackupError(f"{archive} is not a readable .tar.gz: {exc}") from exc


def read_manifest(archive) -> dict:
    with _open_archive(archive) as tar:
        try:
            handle = tar.extractfile(MANIFEST_NAME)
        except KeyError:
            handle = None
        if handle is None:
            raise BackupError(
                f"{archive} has no {MANIFEST_NAME}: it was not written by this tool."
            )
        try:
            return json.loads(handle.read().decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise BackupError(f"{archive} has an unreadable {MANIFEST_NAME}: {exc}") from exc


def _safe_members(tar: tarfile.TarFile, destination: Path):
    """Members that stay inside the destination.

    An archive is operator input, but not necessarily an archive this tool
    wrote. A member named ../../etc/cron.d/x, or a symlink pointing out of the
    tree, would have tarfile write outside the directory. Python 3.12 added a
    filter for this; on 3.10 it is this function's job.
    """
    root = destination.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            raise BackupError(f"{member.name}: links are not allowed in a backup")
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise BackupError(f"{member.name}: would extract outside the target")
        yield member


def incompatibilities(manifest: dict, settings=None) -> list:
    """Reasons this archive does not fit this deployment.

    The embedding model is the one that matters: vectors built by another model
    are not wrong-looking, they are quietly meaningless, and search degrades
    without an error anywhere. The collection name matters for a duller reason -
    the app would open a different, empty collection and report no documents.
    """
    current = _settings_summary(settings)
    recorded = manifest.get("settings") or {}
    problems = []

    for key in ("embedding_model", "collection_name"):
        was, now = recorded.get(key), current.get(key)
        if was and now and was != now:
            problems.append(f"{key.upper()}: archive has {was!r}, this deployment {now!r}")

    return problems


def verify_members(archive, manifest: dict) -> list:
    """Members whose bytes no longer match the manifest."""
    recorded = manifest.get("members") or {}
    damaged = []

    with _open_archive(archive) as tar:
        for name, expected in recorded.items():
            try:
                handle = tar.extractfile(name)
            except KeyError:
                handle = None
            if handle is None:
                damaged.append(f"{name}: missing from the archive")
                continue
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(_READ_CHUNK), b""):
                digest.update(block)
            if digest.hexdigest() != expected.get("sha256"):
                damaged.append(f"{name}: contents do not match the manifest")

    return damaged


def restore_backup(archive, settings=None, locations=None,
                   overwrite=False, force=False) -> dict:
    """Put an archive's contents back, refusing anything questionable.

    Returns what was restored. Raises BackupError rather than half-restoring:
    every check runs before anything on disk is touched.
    """
    archive = Path(archive)
    if not archive.exists():
        raise BackupError(f"No such archive: {archive}")

    locations = locations or storage_locations(settings)
    manifest = read_manifest(archive)

    damaged = verify_members(archive, manifest)
    if damaged:
        raise BackupError("The archive does not match its manifest:\n  " + "\n  ".join(damaged))

    problems = incompatibilities(manifest, settings)
    if problems and not force:
        raise BackupError(
            "This archive does not fit this deployment:\n  "
            + "\n  ".join(problems)
            + "\nRestoring anyway would leave a corpus that answers nothing. "
            "Change the settings to match, or pass --force if you know better."
        )

    occupied = [
        str(path) for name, path in locations.items()
        if path.exists() and any(path.iterdir())
    ]
    if occupied and not overwrite:
        raise BackupError(
            "These directories already hold data: " + ", ".join(occupied)
            + "\nPass --overwrite to replace them."
        )

    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        with _open_archive(archive) as tar:
            tar.extractall(staging, members=_safe_members(tar, staging))

        restored = {}
        for name, destination in locations.items():
            source = staging / name
            if not source.exists():
                continue
            if destination.exists():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            restored[name] = str(destination)

    return {
        "restored": restored,
        "manifest": manifest,
        "warnings": problems if force else [],
    }
