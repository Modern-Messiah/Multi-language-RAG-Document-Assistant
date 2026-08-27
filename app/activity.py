"""When was each owner last here.

Needed for one thing: telling an abandoned namespace from a quiet one. The web
UI mints a fresh "web-<uuid>" owner per browser session, so its documents are
orphaned the moment the tab closes and can never be reached again; a Telegram
user, whose id is stable, may go silent for a month and come back. Both look
identical in the vector store. The difference is whether anyone has touched the
namespace lately, and nothing recorded that.

The record is a marker file per owner, under a directory of its own next to
the uploads. A file rather than a database row because the rest of this
system's per-owner state is already files on that volume: the marker rides
along in backups (tar and shutil.copy2 both keep mtime) and disappears with
the volume like everything else.

Not inside the owner's own upload directory, for two reasons that are both
tests: the upload tests assert an owner directory is empty after a rejected
upload, and the per-owner byte quota counts what is in that directory.

The failure direction matters more than the mechanism. A marker that is too
OLD makes a live owner look idle, and idle is what gets deleted. So: a missing
marker never means idle (the sweep reports it as unknown), owner directories
that predate markers are seeded at startup so "idle" means "idle since the
upgrade", and a touch that fails is counted so the sweep can refuse to trust
data it knows is wrong.
"""
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# A user_id matches [A-Za-z0-9_-], so a name starting with "." can never
# collide with one.
ACTIVITY_DIRNAME = ".activity"

# The same shape the API accepts for user_id (USER_ID_PATTERN plus the length
# cap); a test asserts the two agree. Anything else found on disk is not an
# owner and is never acted on - ".activity" itself is the first such name.
OWNER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_owner_name(name: str) -> bool:
    return OWNER_NAME.fullmatch(name) is not None


class ActivityTracker:
    def __init__(self, upload_dir):
        self.upload_dir = Path(upload_dir)
        self.directory = self.upload_dir / ACTIVITY_DIRNAME
        # Touches that could not be written since this process started. A full
        # volume is the exact condition under which someone reaches for the
        # sweep, and also the condition under which the markers stop being
        # true - so the sweep is told.
        self.touch_failures = 0

    def marker_for(self, user_id: str) -> Path:
        return self.directory / user_id

    def touch(self, user_id: str) -> None:
        """Record that this owner was just here.

        Never raises: a marker that cannot be written is a warning in the log
        and a count, not a failed request. The request it rides on has already
        done its real work.
        """
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            # Path.touch both creates a missing file and updates an existing
            # one's mtime. An exists()/utime pair was racy: forget() runs
            # without the owner's lock, from /clear and from the sweep, and
            # could unlink the marker between the two calls. The FileNotFoundError
            # was then counted as a write failure, and a single one of those
            # makes every later sweep refuse to apply until the process restarts.
            self.marker_for(user_id).touch()
        except OSError:
            self.touch_failures += 1
            logger.warning("Could not record activity for user_id=%s", user_id)

    def forget(self, user_id: str) -> None:
        """Drop the marker, when the namespace itself is being removed."""
        try:
            self.marker_for(user_id).unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Could not remove the activity marker for user_id=%s", user_id)

    def last_seen(self, user_id: str) -> Optional[datetime]:
        """When this owner was last active, or None if nothing says.

        The newer of the marker and the newest uploaded file. Owners from
        before markers existed have only their files; an owner whose upload is
        in flight has a fresh file and a stale marker, and the file is the
        truth. An owner with neither is unknown, and unknown is not idle: the
        sweep never removes what it cannot date.

        Aware UTC, always, so it can be compared with datetime.now(timezone.utc)
        and printed with an offset that means the same thing in every zone.
        """
        candidates = []

        marker = self.marker_for(user_id)
        if marker.exists():
            candidates.append(_mtime(marker))

        owner_dir = self.upload_dir / user_id
        if owner_dir.is_dir():
            candidates.extend(
                _mtime(item) for item in owner_dir.rglob("*") if item.is_file()
            )

        return max(candidates) if candidates else None

    def known_owners(self) -> set:
        """Owners that have a marker, whether or not they still have files."""
        if not self.directory.is_dir():
            return set()
        return {item.name for item in self.directory.iterdir() if item.is_file()}

    def owner_dirs(self) -> set:
        """Every directory under the uploads, by name, markers' own excluded.

        Unfiltered on purpose: the caller decides what to do with a name that
        is not an owner, and "report it" is a better answer than "hide it".
        """
        if not self.upload_dir.is_dir():
            return set()
        return {
            item.name for item in self.upload_dir.iterdir()
            if item.is_dir() and item.name != ACTIVITY_DIRNAME
        }

    def seed_missing(self) -> int:
        """Give every owner directory that has no marker one dated now.

        Run at startup. Without it, an owner from before markers existed is
        dated by their newest upload, which is a LOWER bound on activity - a
        Telegram user who uploaded once and has asked questions weekly ever
        since would look idle for months and be swept on the first run after
        the upgrade. Seeding makes "idle" mean "idle since the upgrade", which
        is the only thing the data can honestly say.
        """
        seeded = 0
        for name in self.owner_dirs():
            if is_owner_name(name) and not self.marker_for(name).exists():
                self.touch(name)
                seeded += 1
        if seeded:
            logger.info("Seeded activity markers for %d owner(s) that had none", seeded)
        return seeded

    def reset_all(self) -> int:
        """Date every owner directory now. For after a restore.

        The archive carries the markers with their original mtimes, so a
        restore of last month's snapshot would make everyone look a month idle
        by construction. Restarting the clock is the safe direction: at worst a
        genuinely abandoned namespace lives idle_days longer.
        """
        reset = 0
        # Marker names as well as directories: /query and /feedback record an
        # owner who has never uploaded, so a marker with no directory behind it
        # is a real state. Restoring one with its old mtime and not resetting it
        # would leave that owner looking idle by construction.
        for name in self.owner_dirs() | self.known_owners():
            if is_owner_name(name):
                self.touch(name)
                reset += 1
        return reset


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
