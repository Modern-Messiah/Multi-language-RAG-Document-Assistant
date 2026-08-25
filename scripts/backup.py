"""Take a snapshot of the ChromaDB index, the uploads and the ratings.

    python -m scripts.backup --output data/backups
    docker compose run --rm -v "$PWD/backups:/backups" backend \
        python -m scripts.backup --output /backups

Stop the backend first. ChromaDB is a SQLite database plus HNSW index files
written separately, and a snapshot taken mid-write can catch the two out of
step - nothing this script does can fix that, so by default it refuses to run
while the backend answers. --live overrides that for a deployment where a
minute of downtime is worse than a small risk.

No OpenAI key is needed: this only copies files.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

from app.backup import BackupError, create_backup, storage_locations

DEFAULT_OUTPUT = "data/backups"
PROBE_TIMEOUT = 3.0


def backend_is_up(url: str) -> bool:
    """Whether something answers /ready at `url`.

    A 503 counts as up: the process is there and may still be writing. Only a
    connection failure means nobody is home.
    """
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/ready", timeout=PROBE_TIMEOUT):
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"directory for the archive (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--url", default=None,
                        help="backend URL to check before starting "
                             "(default: BACKEND_URL, or http://127.0.0.1:8000)")
    parser.add_argument("--live", action="store_true",
                        help="take the snapshot even though the backend is running")
    parser.add_argument("--json", action="store_true",
                        help="print the manifest as JSON instead of prose")
    args = parser.parse_args(argv)

    from clients.backend import backend_url

    url = args.url or backend_url()
    if backend_is_up(url) and not args.live:
        print(f"The backend at {url} is answering.", file=sys.stderr)
        print(
            "Stop it before taking a snapshot - ChromaDB's database and its "
            "index files are written separately, and a copy taken mid-write can "
            "catch them out of step.\n"
            "  docker compose stop backend\n"
            "Or pass --live to accept that risk.",
            file=sys.stderr,
        )
        return 2

    locations = storage_locations()
    try:
        manifest = create_backup(args.output, locations=locations)
    except BackupError as exc:
        print(f"Backup failed: {exc}", file=sys.stderr)
        return 1

    archive = manifest.pop("archive")
    if args.json:
        print(json.dumps({**manifest, "archive": str(archive)}, indent=2, ensure_ascii=False))
        return 0

    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"Wrote {archive} ({size_mb:.1f} MB)")
    for name, count in (manifest.get("counts") or {}).items():
        print(f"  {name}: {'unknown' if count is None else count}")
    print(f"  files: {len(manifest.get('members') or {})}")
    settings = manifest.get("settings") or {}
    print(f"  embedding model: {settings.get('embedding_model')}")
    print(f"  collection: {settings.get('collection_name')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
