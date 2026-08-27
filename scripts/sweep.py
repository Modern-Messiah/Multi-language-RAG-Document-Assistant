"""Find namespaces nobody has touched in a while, and remove them on request.

    python -m scripts.sweep                       # dry run: web sessions idle 30+ days
    python -m scripts.sweep --idle-days 60
    python -m scripts.sweep --apply               # actually remove them
    python -m scripts.sweep --prefix "" --apply --force   # every tenant; think first

Talks to the RUNNING backend's POST /maintenance/sweep, because ChromaDB must
have a single writer - the same reason backup and restore insist the backend is
stopped. So unlike those two this needs the backend up and an API key.

Everything is a dry run unless --apply is given, and the backend refuses apply
on its own where the activity data is most likely to be wrong: no prefix, fewer
than 7 idle days, failed activity writes since startup, or nobody active at
all (which after a restore or a long stop means stale markers, not a mass
departure). --force overrides each of those; the output says what was refused
and why.

--prefix defaults to "web-", the web UI's per-session namespaces, which are
orphaned the moment a browser tab closes. Telegram ids are digit-only, so that
prefix can never match a Telegram user.

In Compose the backend is only reachable from another service on the network:

    docker compose run --rm bot python -m scripts.sweep --apply

The bot service carries BACKEND_URL=http://backend:8000 and the API key; a
one-off `backend` container would try http://localhost:8000 and find nobody.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

from dotenv import load_dotenv

from app.humanize import human_size

DEFAULT_IDLE_DAYS = 30
DEFAULT_PREFIX = "web-"
TIMEOUT = 600.0  # a full metadata scan of a large collection is not quick


def call_sweep(url: str, api_key: str, idle_days: int, prefix: str,
               apply: bool, force: bool) -> tuple:
    """POST to the endpoint. Returns (status_code, parsed body or text)."""
    query = urllib.parse.urlencode({
        "idle_days": idle_days,
        "prefix": prefix,
        "apply": "true" if apply else "false",
        "force": "true" if force else "false",
    })
    request = urllib.request.Request(
        f"{url.rstrip('/')}/maintenance/sweep?{query}",
        method="POST",
        headers={"X-API-Key": api_key} if api_key else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw


def _row(entry: dict) -> str:
    return (
        f"  {entry['user_id']:<40} {entry['documents']:>5} docs "
        f"{human_size(entry['bytes']):>10}   last seen {entry.get('last_seen') or '-'}"
    )


def print_report(body: dict) -> None:
    mode = "DRY RUN" if body["dry_run"] else "APPLIED"
    print(f"{mode}: owners matching {body['prefix']!r} idle since {body['cutoff']} "
          f"({body['idle_days']} days)")

    if body.get("refused"):
        print(f"\nREFUSED to apply: {body['refused']}")

    print(f"\nCandidates ({len(body['candidates'])}):")
    for entry in body["candidates"]:
        print(_row(entry))
    if not body["candidates"]:
        print("  none")

    if body["empty"]:
        print(f"\nEmpty (marker or directory only, nothing to lose): {len(body['empty'])}")
        for entry in body["empty"]:
            print(f"  {entry['user_id']}")

    if body["orphans"]:
        total = sum(o["bytes"] for o in body["orphans"])
        print(f"\nOrphan files with no vectors behind them: {human_size(total)} across "
              f"{len(body['orphans'])} owner(s)")
        for orphan in body["orphans"]:
            scope = "" if orphan.get("in_scope", True) else "   (outside the prefix)"
            print(f"  {orphan['user_id']:<40} {orphan['files']:>5} files "
                  f"{human_size(orphan['bytes']):>10}{scope}")
        outside = [o for o in body["orphans"] if not o.get("in_scope", True)]
        if outside:
            # Said out loud: these are reported so they are not invisible, but
            # this run will not touch them.
            print(f"  {len(outside)} owner(s) above are outside {body['prefix']!r} and "
                  "will not be cleaned by this run.")

    if body["unknown"]:
        print(f"\nUnknown - nothing dates these, never swept ({len(body['unknown'])}):")
        for name in body["unknown"]:
            print(f"  {name}")

    if body["foreign"]:
        print(f"\nNot owners - names on disk that cannot be a user_id, left alone ({len(body['foreign'])}):")
        for name in body["foreign"]:
            print(f"  {name}")

    if not body["dry_run"]:
        print(f"\nSwept: {len(body['swept'])}")
        for name in body["swept"]:
            print(f"  {name}")
        if body["became_active"]:
            print(f"Skipped, came back while the sweep ran: {', '.join(body['became_active'])}")
        if body["orphans_removed"]:
            print(f"Orphan files removed: {body['orphans_removed']}")
        if body["failed"]:
            print(f"\nFAILED ({len(body['failed'])}):", file=sys.stderr)
            for failure in body["failed"]:
                print(f"  {failure['user_id']}: {failure['error']}", file=sys.stderr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--idle-days", type=int, default=DEFAULT_IDLE_DAYS,
                        help=f"how long untouched counts as idle (default: {DEFAULT_IDLE_DAYS})")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help=f"only owners whose id starts with this (default: {DEFAULT_PREFIX!r}; "
                             "pass '' for every tenant)")
    parser.add_argument("--apply", action="store_true",
                        help="remove what the dry run would list")
    parser.add_argument("--force", action="store_true",
                        help="override the backend's own refusals to apply")
    parser.add_argument("--url", default=None,
                        help="backend URL (default: BACKEND_URL, or http://127.0.0.1:8000)")
    parser.add_argument("--api-key", default=None,
                        help="X-API-Key (default: BACKEND_API_KEY)")
    parser.add_argument("--json", action="store_true",
                        help="print the backend's response as JSON")
    args = parser.parse_args(argv)

    # Like the bot and the web UI, read .env: this is the first script that
    # has to authenticate, and an empty key answered by 401 would otherwise
    # tell the operator to contact the operator.
    load_dotenv()

    from clients.backend import backend_url

    url = args.url or backend_url()
    api_key = args.api_key if args.api_key is not None else os.getenv("BACKEND_API_KEY", "")

    print(f"Asking {url} ...", file=sys.stderr)
    try:
        status, body = call_sweep(url, api_key, args.idle_days, args.prefix,
                                  args.apply, args.force)
    except (urllib.error.URLError, OSError) as exc:
        print(f"Could not reach the backend at {url}: {exc}", file=sys.stderr)
        print("The sweep runs inside the backend, so it has to be up.", file=sys.stderr)
        return 2

    if status != 200:
        # The operator is the audience here, so the raw answer is more use
        # than the clients' softened wording.
        detail = body.get("detail") if isinstance(body, dict) else body
        print(f"Backend answered HTTP {status}: {detail}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(body, indent=2, ensure_ascii=False))
    else:
        print_report(body)

    if body.get("refused"):
        return 3
    if body.get("failed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
