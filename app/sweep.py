"""Namespaces nobody comes back to, and what it takes to remove them safely.

The web UI mints a fresh owner per browser session, so its documents are
orphaned the moment the tab closes and can never be reached again - and since
Stage 14 they are faithfully copied into every backup. This finds namespaces
nobody has touched in `idle_days` and, when asked twice, removes them.

It runs in the backend's own process rather than as an offline script because
ChromaDB must have a single writer: the same reason backup and restore insist
the backend is stopped. app.main keeps the endpoint - the HTTP shape, the
validation and the status codes - and this module keeps the decisions.

The split here is the one the feature is built around: `plan` works out what
*would* go and never deletes anything, `execute` deletes what a plan named.
A dry run is simply a plan nobody executed. That is also why apply's own
refusals live in the plan rather than in execute: an operator has to be able
to see a refusal without anything being at risk.

Nothing here raises HTTPException. A vector store that cannot be read is a
SweepError, which the endpoint turns into a 503 - the same arrangement
app/backup.py has with BackupError, and what lets this module be used from
somewhere that is not a request.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from app import storage
from app.activity import is_owner_name
from app.models.schemas import (
    OrphanEntry,
    SweepEntry,
    SweepFailure,
    SweepResponse,
)

logger = logging.getLogger(__name__)

# Below this, apply needs force. A typo in a cron line must not sweep
# yesterday's users; a web session left open over a weekend is not abandoned.
MIN_IDLE_DAYS_TO_APPLY = 7


class SweepError(Exception):
    """The store could not be read, so nothing can be said about who is idle."""


@dataclass
class Plan:
    """What a sweep would do, and why it may refuse to do it.

    Every owner matching the prefix is in exactly one of candidates, empty,
    unknown, or none of them - which means they are active. `foreign` is names
    on disk that cannot be a user_id and are never acted on. `refused` is set
    when apply was asked for and declined; execute must not be called then.
    """

    idle_days: int
    prefix: str
    cutoff: datetime
    newest_seen: Optional[datetime]
    candidates: List[SweepEntry]
    empty: List[SweepEntry]
    unknown: List[str]
    foreign: List[str]
    orphans: List[OrphanEntry]
    refused: Optional[str] = None


@dataclass
class Outcome:
    """What executing a plan actually did."""

    swept: List[str] = field(default_factory=list)
    became_active: List[str] = field(default_factory=list)
    failed: List[SweepFailure] = field(default_factory=list)
    orphans_removed: int = 0


def plan(state, settings, idle_days: int, prefix: str,
         apply: bool = False, force: bool = False) -> Plan:
    """Work out who is idle. Changes nothing.

    `apply` and `force` are read here rather than in execute because their only
    effect at this stage is whether a refusal is recorded - and a refusal is
    something a dry run has to be able to show.
    """
    activity = state.activity

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=idle_days)

    # Every name the data knows: owners with vectors, owners with a directory,
    # owners with a marker. A name that could not be a user_id is reported and
    # never acted on - ".activity" itself is one, and so is whatever an
    # operator or a filesystem left there.
    try:
        with_vectors = set(state.embeddings.list_owners())
    except Exception as exc:
        logger.exception("Could not enumerate owners")
        raise SweepError("could not enumerate owners") from exc
    names = with_vectors | activity.owner_dirs() | activity.known_owners()

    foreign = sorted(name for name in names if not is_owner_name(name))
    valid = sorted(name for name in names if is_owner_name(name))

    # "Anyone active at all" is judged over every owner, not only the prefix:
    # after a restore or a long stop everyone looks idle at once, and that is
    # stale data, not a mass departure. A prefix sweep of web sessions while
    # Telegram users are busy is fine; a sweep where nobody has been seen is
    # the one that needs a second look.
    dated = {name: activity.last_seen(name) for name in valid}
    known_dates = [seen for seen in dated.values() if seen is not None]
    newest_seen = max(known_dates) if known_dates else None
    anyone_active = newest_seen is not None and newest_seen >= cutoff

    candidates, empty, unknown, orphans = [], [], [], []
    for owner in valid:
        # Orphan files are looked for under EVERY owner, not only the prefix.
        # They are most likely under the stable ids the default prefix excludes -
        # left by a crash between write and index, or by an older /clear that
        # deleted vectors only - and reporting them costs one lookup. Removing
        # them stays inside the prefix: an operator who asked to sweep web
        # sessions should not have files deleted elsewhere.
        in_scope = owner.startswith(prefix)
        try:
            held = {d["file_hash"] for d in state.embeddings.list_documents(owner)}
        except Exception as exc:
            logger.exception("Could not list documents for %s during sweep", owner)
            raise SweepError(f"could not list documents for {owner}") from exc

        files = storage.stored_files(settings, owner)
        backed = sum(storage.size_of(p) for p in files if p.name.split("_", 1)[0] in held)
        stray = [p for p in files if p.name.split("_", 1)[0] not in held]
        if stray:
            orphans.append(OrphanEntry(
                user_id=owner, files=len(stray),
                bytes=sum(storage.size_of(p) for p in stray),
                in_scope=in_scope,
            ))

        if not in_scope:
            continue

        seen = dated[owner]
        entry = SweepEntry(
            user_id=owner, documents=len(held), bytes=backed,
            last_seen=seen.isoformat() if seen else None,
        )
        if seen is None:
            unknown.append(owner)
        elif seen >= cutoff:
            # Inclusive: exactly idle_days is not yet idle. No test pins the
            # difference between >= and >, because cutoff is derived from this
            # request's own clock and no marker's mtime can be made equal to
            # it; the choice is here so it is at least deliberate.
            continue  # active: not this sweep's business
        elif held or files:
            candidates.append(entry)
        else:
            # A marker or an empty directory left after every document was
            # deleted. Nothing to lose, so cleaned without ceremony.
            empty.append(entry)

    return Plan(
        idle_days=idle_days,
        prefix=prefix,
        cutoff=cutoff,
        newest_seen=newest_seen,
        candidates=candidates,
        empty=empty,
        unknown=unknown,
        foreign=foreign,
        orphans=orphans,
        refused=_refusal(
            activity, idle_days, prefix, apply, force,
            anyone_active=anyone_active,
            anything_to_sweep=bool(candidates or empty),
        ),
    )


def _refusal(activity, idle_days: int, prefix: str, apply: bool, force: bool,
             anyone_active: bool, anything_to_sweep: bool) -> Optional[str]:
    """Why this sweep will not apply itself, or None if it may.

    Each case is one where the activity data is most likely to be wrong, or
    the request most likely to be wider than intended. force overrides all of
    them; the wording says what to pass.
    """
    if not apply or force:
        return None

    if not prefix:
        return (
            "apply without a prefix would sweep every tenant, stable Telegram "
            "ids included. Pass prefix=web- for the web UI's per-session "
            "namespaces, or force=true to mean it."
        )
    if idle_days < MIN_IDLE_DAYS_TO_APPLY:
        return (
            f"apply with idle_days below {MIN_IDLE_DAYS_TO_APPLY} would sweep "
            "namespaces a returning user still expects. Pass force=true to mean it."
        )
    if activity.touch_failures:
        return (
            f"{activity.touch_failures} activity update(s) failed since startup, "
            "so some live owners may look idle. Fix the volume (it is probably "
            "full), restart, then sweep - or pass force=true."
        )
    if not anyone_active and anything_to_sweep:
        return (
            f"no owner at all has been active in the last {idle_days} days, which "
            "looks like stale activity data (a restore, a long stop) rather than "
            "everyone leaving. Pass force=true if it really is everyone."
        )
    return None


def execute(state, settings, plan: Plan) -> Outcome:
    """Remove what the plan named. Only for a plan that was not refused.

    One owner's failure is recorded and the rest go on: a sweep that stopped at
    the first error would leave an operator to run it again and again.
    """
    activity = state.activity
    outcome = Outcome()

    for entry in plan.candidates + plan.empty:
        owner = entry.user_id
        try:
            with state.upload_locks.for_owner(owner):
                # Re-read under the lock: the owner may have come back while
                # the plan was being built.
                seen_now = activity.last_seen(owner)
                if seen_now is not None and seen_now >= plan.cutoff:
                    outcome.became_active.append(owner)
                    continue
                # Logged BEFORE deleting, so a crash or a failure mid-loop
                # still leaves a record of what was about to go; the response
                # only exists if the loop ends. The re-check above comes
                # first so an owner who came back is not logged as swept.
                logger.warning(
                    "Sweeping idle namespace user_id=%s documents=%d bytes=%d last_seen=%s",
                    owner, entry.documents, entry.bytes, entry.last_seen,
                )
                storage.wipe_namespace(state, settings, owner)
            outcome.swept.append(owner)
        except Exception as exc:
            logger.exception("Could not sweep %s", owner)
            outcome.failed.append(_failure(owner, exc))

    for orphan in plan.orphans:
        if not orphan.in_scope or orphan.user_id in outcome.swept:
            continue  # out of scope, or already gone with the namespace
        try:
            with state.upload_locks.for_owner(orphan.user_id):
                # Recomputed under the lock: an upload since the plan was built
                # may have given a file its vectors.
                held = {d["file_hash"] for d in state.embeddings.list_documents(orphan.user_id)}
                for path in storage.stored_files(settings, orphan.user_id):
                    if path.name.split("_", 1)[0] not in held:
                        path.unlink()
                        outcome.orphans_removed += 1
        except Exception as exc:
            logger.exception("Could not remove orphan files for %s", orphan.user_id)
            outcome.failed.append(_failure(orphan.user_id, exc))

    return outcome


def _failure(user_id: str, exc: Exception) -> SweepFailure:
    # Bounded: the message reaches an operator's terminal, and a driver's
    # exception can carry a page of SQL.
    return SweepFailure(user_id=user_id, error=f"{type(exc).__name__}: {exc}"[:200])


def report(plan: Plan, outcome: Optional[Outcome] = None) -> SweepResponse:
    """The plan, and what came of it. No outcome means it was a dry run."""
    dry_run = outcome is None
    if outcome is None:
        outcome = Outcome()
    return SweepResponse(
        idle_days=plan.idle_days,
        prefix=plan.prefix,
        dry_run=dry_run,
        cutoff=plan.cutoff.isoformat(),
        newest_seen=plan.newest_seen.isoformat() if plan.newest_seen else None,
        candidates=plan.candidates,
        empty=plan.empty,
        unknown=plan.unknown,
        foreign=plan.foreign,
        orphans=plan.orphans,
        swept=outcome.swept,
        became_active=outcome.became_active,
        failed=outcome.failed,
        orphans_removed=outcome.orphans_removed,
        refused=plan.refused,
    )
