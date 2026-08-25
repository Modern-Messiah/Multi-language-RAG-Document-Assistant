"""Put a snapshot back.

    python -m scripts.restore data/backups/rag-backup-20260825-120000.tar.gz
    python -m scripts.restore <archive> --overwrite

Stop the backend first: the process holds the ChromaDB files open, and
replacing them underneath it leaves it serving from a state that no longer
exists on disk.

Every check runs before anything is touched, so a refusal leaves the current
data exactly as it was:

- the archive's contents must match the hashes in its manifest
- the archive's embedding model and collection name must match this deployment,
  because vectors built by another model are not visibly wrong, only quietly
  meaningless
- the target directories must be empty unless --overwrite is given

    python -m scripts.restore <archive> --inspect

prints what an archive holds and whether it fits, without writing anything.
"""
import argparse
import sys

from app.backup import (
    BackupError,
    incompatibilities,
    read_manifest,
    restore_backup,
    storage_locations,
    verify_members,
)
from scripts.backup import backend_is_up


def _describe(manifest: dict) -> None:
    settings = manifest.get("settings") or {}
    print(f"Taken:      {manifest.get('created_at')}")
    print(f"Embedding:  {settings.get('embedding_model')}")
    print(f"Collection: {settings.get('collection_name')}")
    print(f"Chunking:   {settings.get('chunk_size')} / {settings.get('chunk_overlap')}")
    for name, count in (manifest.get("counts") or {}).items():
        print(f"  {name}: {'unknown' if count is None else count}")
    print(f"  files: {len(manifest.get('members') or {})}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="the .tar.gz written by scripts.backup")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace data that is already there")
    parser.add_argument("--force", action="store_true",
                        help="restore even though the archive does not fit this "
                             "deployment's settings")
    parser.add_argument("--inspect", action="store_true",
                        help="report what the archive holds and change nothing")
    parser.add_argument("--url", default=None,
                        help="backend URL to check before starting")
    parser.add_argument("--live", action="store_true",
                        help="restore even though the backend is running")
    args = parser.parse_args(argv)

    try:
        manifest = read_manifest(args.archive)
    except (BackupError, OSError) as exc:
        print(f"Cannot read the archive: {exc}", file=sys.stderr)
        return 1

    if args.inspect:
        _describe(manifest)
        damaged = verify_members(args.archive, manifest)
        problems = incompatibilities(manifest)
        print("\nContents: " + ("intact" if not damaged else "DAMAGED"))
        for line in damaged:
            print(f"  {line}")
        print("Fits this deployment: " + ("yes" if not problems else "NO"))
        for line in problems:
            print(f"  {line}")
        return 0 if not (damaged or problems) else 1

    from clients.backend import backend_url

    url = args.url or backend_url()
    if backend_is_up(url) and not args.live:
        print(f"The backend at {url} is answering.", file=sys.stderr)
        print(
            "Stop it before restoring - it holds the ChromaDB files open and "
            "would go on serving a state that no longer exists on disk.\n"
            "  docker compose stop backend\n"
            "Or pass --live to accept that.",
            file=sys.stderr,
        )
        return 2

    try:
        result = restore_backup(
            args.archive,
            locations=storage_locations(),
            overwrite=args.overwrite,
            force=args.force,
        )
    except BackupError as exc:
        print(f"Restore refused: {exc}", file=sys.stderr)
        return 1

    print("Restored:")
    for name, path in result["restored"].items():
        print(f"  {name} -> {path}")
    for line in result["warnings"]:
        print(f"  WARNING: {line}", file=sys.stderr)
    _describe(result["manifest"])
    print("\nStart the backend and check GET /documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
