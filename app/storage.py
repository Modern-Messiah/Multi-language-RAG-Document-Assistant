"""What one owner holds on this deployment, and how it is taken away.

Two things make up an owner's namespace: the raw uploads under their own
directory, named "<content hash>_<filename>", and the chunks those files were
turned into in the vector store. Quotas count both, the sweep reconciles them
against each other, /clear and a swept namespace remove both.

Extracted from app.main because three callers now need the same handful of
operations - the upload path, /clear, and the idle-namespace sweep - and the
sweep could not move into a module of its own while they lived next to the
endpoints.

Every function here is called through the module (`storage.stored_files(...)`)
rather than imported by name. That is deliberate: a test that replaces one of
these to simulate a file vanishing mid-scan patches `app.storage.stored_files`
once, and every caller sees it. Imported by name, each caller would hold its
own reference and a patch would reach whichever module the test happened to
name - which is how the previous arrangement worked, and it only ever patched
app.main.
"""
import logging
import shutil
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def stored_files(settings, user_id: str) -> List[Path]:
    """Every raw upload this owner has on disk."""
    owner_dir = settings.upload_dir / user_id
    if not owner_dir.is_dir():
        return []
    return [path for path in owner_dir.iterdir() if path.is_file()]


def size_of(path) -> int:
    """Bytes on disk, or 0 if the file is already gone.

    Every caller lists a directory and then stats what it found, and a delete
    or a retired revision can remove a file in between - the listing is not
    always under the owner's lock. A vanished file is worth nothing towards a
    quota, not a 500.
    """
    try:
        return path.stat().st_size
    except OSError:
        return 0


def remove_stored_file(settings, user_id: str, file_hash: str) -> None:
    """Delete the raw upload whose name starts with this content hash."""
    owner_dir = settings.upload_dir / user_id
    if not owner_dir.is_dir():
        return
    for path in owner_dir.glob(f"{file_hash}_*"):
        path.unlink(missing_ok=True)


def wipe_namespace(state, settings, user_id: str) -> None:
    """Remove everything one owner has: vectors, raw uploads, activity marker.

    Assumes the caller already holds that owner's upload lock. Without one, a
    sweep could run between an upload's write_bytes and its add_documents and
    remove the file from under it - the upload then answers 200 with nothing
    stored, or a spurious 400. The exact owner a sweep targets is one who has
    just come back.

    Raises if the vectors cannot be deleted; a leftover file is logged, since
    the data the user cares about is gone.
    """
    state.embeddings.delete_documents(filter={"user_id": user_id})

    # Drop the raw uploads too. Deleting only the vectors left every file the
    # user ever sent on disk forever - unbounded volume growth, and "cleared"
    # documents that are still sitting there.
    # user_id is validated ([A-Za-z0-9_-]{1,64}), so it cannot escape upload_dir.
    owner_dir = settings.upload_dir / user_id
    if owner_dir.is_dir():
        shutil.rmtree(owner_dir, ignore_errors=True)
        if owner_dir.exists():
            logger.warning("Could not fully remove upload dir for %s", user_id)

    state.activity.forget(user_id)
