# Backup and restore reference

Read this before a real restore, before restoring into Compose volumes, when scheduling backups,
or when an archive or a restore message looks wrong. Code: `app/backup.py`, `scripts/backup.py`,
`scripts/restore.py`.

Contents:
- Archive layout and manifest
- What no archive covers
- Restore: checks, refusals and what replacement does
- Restoring into Compose volumes (workaround)
- After a restore
- Scheduling and retention
- Exit codes and messages

## Archive layout and manifest

`python -m scripts.backup` writes one `<output>/rag-backup-<YYYYMMDD-HHMMSS>.tar.gz` (UTC, 1-second
resolution; a second run in the same second into the same `--output` overwrites the first). The
top-level names are fixed, so an archive restores into differently configured paths:

| Member | Source | Notes |
| --- | --- | --- |
| `chroma/` | `CHROMA_PERSIST_DIR` | `chroma.sqlite3` through SQLite's backup API (source opened read-only); HNSW segment dirs as plain copies |
| `uploads/` | `UPLOAD_DIR` | raw uploads plus the `.activity/` markers, mtimes kept |
| `feedback/` | `FEEDBACK_DIR` | every file there, rotated-aside ratings files included |
| `manifest.json` | generated | below |

A missing directory is skipped silently. The backup fails with `Backup failed: None of the data
directories exist: ...` only when none of the three exists, and a fresh clone's tracked
`data/uploads/.gitkeep` is enough to pass that guard with an archive holding nothing else. An
unreadable `chroma.sqlite3` fails with advice (stop the backend, or the database is damaged) and
leaves no archive.

`manifest.json` (`--json` prints it with `archive` added):
- `created_at`: UTC, to the second.
- `settings`: `embedding_model`, `collection_name`, `chunk_size`, `chunk_overlap`, read from the
  process environment or the Settings defaults, never from `.env`. Restore compares only the first
  two; chunking is informational.
- `counts`: `chunks` (rows in ChromaDB's `embeddings` table across all owners; null, printed
  `unknown`, if unreadable), `uploads` (files under `UPLOAD_DIR` outside `.activity`), `ratings`
  (non-blank lines of every `*.jsonl` under `FEEDBACK_DIR`, unparseable ones included). A count is
  absent when its directory was.
- `members`: `{path: {bytes, sha256}}` per archived file. `manifest.json` itself is not hashed.

`create_backup` adds each staged directory recursively and then every file again, so files inside
a subdirectory (HNSW segments, uploads, markers: the bulk) are stored three times and top-level
files such as `chroma.sqlite3` twice; archives are about 2-3 times the data. Restored data is
still correct; plan disk, transfer and retention for the larger size.

## What no archive covers

- `.env`: `OPENAI_API_KEY`, `BACKEND_API_KEY` and every setting. Keep it in a secret store; a
  restore needs at least the archive's `EMBEDDING_MODEL` and `COLLECTION_NAME`.
- Logs, the image and the code (rebuild from the checkout).
- The in-memory BM25 index: rebuilt per owner from ChromaDB after a start.
- The Langfuse stack's Postgres volume (traces, scores, accounts).

## Restore: checks, refusals and what replacement does

`scripts/restore.py` `main`, then `app/backup.py` `restore_backup`, in this order:

1. Read `manifest.json`. A missing file, a non-gzip tar or no manifest prints
   `Cannot read the archive: ...` (exit 1).
2. `--inspect` stops here: it prints the description, verifies hashes and compares settings, then
   exits 0 only for `Contents: intact` plus `Fits this deployment: yes`. No probe, no writes.
3. Probe `GET <url>/ready`; any answer exits 2 unless `--live`.
4. Hashes of every member the manifest lists (`Restore refused: The archive does not match its
   manifest`). Members the manifest does not list are not checked, yet they are restored when they
   sit under `chroma/`, `uploads/` or `feedback/`.
5. Fit: the archive's `embedding_model` and `collection_name` against the environment or the
   defaults. Refused unless `--force`, which prints what it ignored as `WARNING:` lines on stderr.
   After a forced restore the backend refuses to start on a model mismatch and opens an empty
   collection on a collection mismatch, so set the two settings instead.
6. Occupied targets: a target directory holding anything but `.activity` is refused unless
   `--overwrite`. Missing or empty targets, or ones holding only markers, pass; `.gitkeep` counts
   as data.
7. Extraction into a temp dir (`TMPDIR`, which needs room for the whole archive): symlinks, hard
   links, absolute paths and paths escaping the dir abort with `Restore refused: <member>: ...`
   before any target changes.
8. Replacement, per directory in the order chroma, uploads, feedback: delete the target, move the
   staged copy in. Not atomic. An `OSError` here (disk full, a mount point) escapes as a traceback
   with exit 1 and can leave old and new directories mixed, so keep the archive until the restored
   backend checks out. A directory the archive lacks leaves its target untouched.
9. `ActivityTracker.reset_all` dates every owner directory and marker now.

On success it prints `Restored:` with `chroma -> <path>` style lines, the archive's description and
`Start the backend and check GET /documents.`

## Restoring into Compose volumes (workaround)

`restore_backup` deletes each destination directory before moving the staged copy in. Under Compose
the destinations are the volume mount points (/app/data/chroma_db, /app/data/uploads,
/app/data/feedback): deleting one empties it and then fails with EBUSY, uncaught, and chroma comes
first, so its volume is emptied and nothing is restored. Until the code empties and refills the
directories instead, restore into a scratch directory in the one-off container and copy from there.
The procedure follows from the code and the compose files but was not run against Docker here:
read every step's output before the next, and give every command the `-f` flags the stack was
started with.

Step 1. Ask the user: step 4 empties all three volumes. If they hold anything that may be needed,
take a Compose backup of them first (see SKILL.md, Backup).

Step 2. `docker compose stop backend`

Step 3. Inspect, read-only:

```bash
docker compose run --rm -v "$PWD/backups:/backups" backend \
  python -m scripts.restore /backups/rag-backup-<stamp>.tar.gz --inspect
```

Step 4. Restore into /tmp/restore inside the container, then replace the volume contents. `set -e`
stops before any volume is touched if `scripts.restore` refuses or fails. Root because
`data_feedback` may be root-owned; ownership is handed back to `appuser` at the end.

```bash
docker compose run --rm -u root -v "$PWD/backups:/backups" \
  -e CHROMA_PERSIST_DIR=/tmp/restore/chroma_db \
  -e UPLOAD_DIR=/tmp/restore/uploads \
  -e FEEDBACK_DIR=/tmp/restore/feedback \
  backend sh -c '
set -e
python -m scripts.restore /backups/rag-backup-<stamp>.tar.gz
for d in chroma_db uploads feedback; do
  if [ -d "/tmp/restore/$d" ]; then
    find "/app/data/$d" -mindepth 1 -delete
    cp -a "/tmp/restore/$d/." "/app/data/$d/"
  fi
done
chown -R appuser:appuser /app/data/chroma_db /app/data/uploads /app/data/feedback
'
```

Step 5. `docker compose start backend`, then check `GET /ready` and `GET /documents` for a known
owner against the archive's counts.

Notes:
- `-e` overrides the three paths from `.env` for this container only, so the fit check still uses
  the deployment's own `EMBEDDING_MODEL` and `COLLECTION_NAME`; never add `--force` here.
- The probe inside the one-off container most likely cannot see the backend (see SKILL.md,
  Backup); stopping it in step 2 is the real safeguard.
- The container's writable layer needs room for the extracted archive.
- A directory missing from the archive leaves its volume as it was, as `scripts.restore` does.

## After a restore

- Every activity marker is dated now, so the sweep treats every restored owner as active for
  idle_days.
- The BM25 keyword index starts empty and refills an owner from ChromaDB on that owner's first
  question without a metadata filter; an owner who uploads before that keeps older documents out of
  keyword matching until the next restart (KNOWN_ISSUES.md, section "Retrieval and search").
- `GET /documents?user_id=<owner>` (with `X-API-Key`) lists one owner's documents; the manifest's
  `chunks` and `uploads` are totals across owners.

## Scheduling and retention

- Nothing in the repo schedules backups. A cron job must stop and start the backend around the
  backup (downtime) or pass `--live` and accept a possibly torn ChromaDB snapshot: let the user
  choose. Start the backend again even when the backup fails, for example (production shape, run
  from the checkout directory):

  ```bash
  docker compose -f docker-compose.yml stop backend
  docker compose -f docker-compose.yml run --rm -v "$PWD/backups:/backups" backend \
    python -m scripts.backup --output /backups --json >> backups/backup.log
  docker compose -f docker-compose.yml start backend
  ```

- Archives hold every tenant's documents and ratings: copy them off the host, restrict access,
  prune old ones (each is 2-3 times the data), and keep `.env` with them in a secret store.
- An archive nobody has restored is unproven: run `--inspect` on every new one and, now and then, a
  full restore into scratch directories.

## Exit codes and messages

| Command | 0 | 1 | 2 |
| --- | --- | --- | --- |
| `python -m scripts.backup` | `Wrote <archive> (<n> MB)` plus counts | `Backup failed: ...`, or a traceback such as `PermissionError` on `--output` | `The backend at <url> is answering.` |
| `python -m scripts.restore` | `Restored:` lines, or `--inspect` intact and fitting | `Cannot read the archive`, `Restore refused: ...`, `--inspect` problems, or a traceback | `The backend at <url> is answering.` |

Both print their refusal advice on stderr, including `docker compose stop backend`.
