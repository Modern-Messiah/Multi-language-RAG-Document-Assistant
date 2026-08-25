"""Where a thumbs-down goes.

The evaluation harness (Stage 9) measures retrieval against a golden set that
*I wrote*, which measures my guesses about what people ask. The request id
(Stage 12) made a single bad answer findable in the log. This closes the loop:
a rating carries the question, the answer and the sources that produced it, so a
complaint from production becomes a case the harness can be run against.

Only rated exchanges are stored. Logging every question would be a larger
privacy decision made on the operator's behalf; pressing a button is the user
saying "look at this one". FEEDBACK_ENABLED turns even that off.

Append-only JSONL, one record per line. The lock is there for the size check,
which is a read followed by a write: two handlers - and they do run in a
threadpool - could otherwise both find room and both write. No test proves the
lock, because provoking that interleaving reliably would mean instrumenting the
code it guards.

Not fsynced: losing the last line to a machine crash costs one rating, and a
disk flush per click is a poor price for that. Not rotated either - the cap
refuses new records rather than growing without bound, and says so loudly enough
for an operator to move the file.
"""
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FEEDBACK_FILENAME = "feedback.jsonl"


class FeedbackStorageFull(Exception):
    """The file has reached FEEDBACK_MAX_BYTES and was not written to."""


class FeedbackStore:
    """Append-only rating log."""

    def __init__(self, directory, max_bytes: int):
        self.path = Path(directory) / FEEDBACK_FILENAME
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def record(self, **fields) -> dict:
        """Append one rating. Returns the record as stored.

        Raises FeedbackStorageFull if the file is at its cap, OSError if the
        write itself fails.
        """
        record = {"at": _now(), **fields}
        # ensure_ascii=False keeps Russian and Kazakh questions readable in the
        # file; json.dumps still escapes any newline, so one record stays one
        # line and the file stays greppable.
        line = json.dumps(record, ensure_ascii=False) + "\n"

        encoded = line.encode("utf-8")

        with self._lock:
            size = self.path.stat().st_size if self.path.exists() else 0
            # The incoming record counts towards the cap, so the file never
            # exceeds it. Checking only the current size would let it overshoot
            # by one record, which for a cap that exists to protect a shared
            # volume is the wrong way to be wrong.
            if size + len(encoded) > self.max_bytes:
                raise FeedbackStorageFull(
                    f"{self.path} is at {size} of {self.max_bytes} bytes"
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)

        return record

    def read_all(self) -> list:
        return read_records(self.path)


def read_records(path) -> list:
    """Every record in a ratings file, skipping any line that is not JSON.

    A truncated last line is what a crash mid-append leaves behind, and it must
    not stop the rest from being read. A free function because the offline
    tooling is handed a path, not a configured store.
    """
    path = Path(path)
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                logger.warning(
                    "Skipping unreadable feedback line %s in %s", number, path
                )
    return records


def _now() -> str:
    """UTC, to the second, ISO 8601.

    Server time, not the client's: a rating's timestamp is only useful for
    lining it up against the server's own log.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def summarise(records: list) -> dict:
    """Counts worth printing: how many ratings, and which documents are behind
    the bad answers."""
    down_by_source: dict = {}
    up = down = 0

    for record in records:
        if record.get("rating") == "up":
            up += 1
            continue
        if record.get("rating") != "down":
            continue
        down += 1
        for source in record.get("sources") or []:
            down_by_source[source] = down_by_source.get(source, 0) + 1

    return {
        "total": up + down,
        "up": up,
        "down": down,
        # None rather than 0.0 when nothing has been rated: a rate of zero
        # reads as "nothing is wrong", which is not what no data means.
        "down_rate": round(down / (up + down), 3) if (up + down) else None,
        "down_by_source": dict(
            sorted(down_by_source.items(), key=lambda item: -item[1])
        ),
    }


def unanswered_questions(records: list, rating: str = "down") -> list:
    """The questions worth turning into golden cases, newest first."""
    picked = [r for r in records if r.get("rating") == rating]
    return sorted(picked, key=lambda r: r.get("at") or "", reverse=True)


def golden_case_stub(record: dict) -> str:
    """A GoldenCase the harness could run, with the expectation left blank.

    Deliberately not filled in: what *should* have been retrieved is a judgement
    about the corpus, and a stub that guesses it would quietly turn one bad
    answer into a permanently wrong benchmark.
    """
    question = json.dumps(record.get("question", ""), ensure_ascii=False)
    retrieved = ", ".join(
        json.dumps(s, ensure_ascii=False) for s in (record.get("sources") or [])
    )
    return (
        "GoldenCase(\n"
        f"    question={question},\n"
        "    expected_sources=[],  # TODO: which document answers this?\n"
        f"    # retrieved instead: [{retrieved}]\n"
        f"    # request_id={record.get('request_id') or '-'}\n"
        ")"
    )


def store_from_settings(settings) -> Optional[FeedbackStore]:
    """None when collection is switched off, so nothing is created on disk."""
    if not settings.feedback_enabled:
        return None
    return FeedbackStore(settings.feedback_dir, settings.feedback_max_bytes)
