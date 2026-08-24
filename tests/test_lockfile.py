"""The lock files must stay faithful to requirements*.txt.

A lock exists to stop transitive dependencies from floating — that is how an
incompatible posthog release started logging errors on every startup. But a
lock is only worth having if it cannot silently rot: these tests fail when a
direct requirement is added, changed, or removed without regenerating it, and
they need neither uv nor network access.

CI additionally recompiles the locks and diffs them, which catches drift these
tests cannot see (a transitive pin edited by hand). This is the fast local half.
"""
import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]

RUNTIME_IN = ROOT / "requirements.txt"
DEV_IN = ROOT / "requirements-dev.txt"
RUNTIME_LOCK = ROOT / "requirements.lock"
DEV_LOCK = ROOT / "requirements-dev.lock"

# The locks are resolved for the image and CI, not for a developer laptop.
EXPECTED_TARGET = ("--python-platform linux", "--python-version 3.10", "--generate-hashes")


def _direct_requirements(path: Path) -> dict[str, Requirement]:
    """Parse a requirements*.txt into {canonical name: Requirement}."""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        requirement = Requirement(line)
        found[canonicalize_name(requirement.name)] = requirement
    return found


def _locked_versions(path: Path) -> dict[str, str]:
    """Parse a lock into {canonical name: pinned version}."""
    text = path.read_text(encoding="utf-8")
    pins = re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", text, re.M)
    return {canonicalize_name(name): version for name, version in pins}


def _lock_entries(path: Path) -> list[tuple[str, str]]:
    """[(name==version, trailing block)] for every entry in a lock."""
    text = path.read_text(encoding="utf-8")
    starts = [m for m in re.finditer(r"^[A-Za-z0-9_.-]+==[^\s;]+", text, re.M)]
    entries = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        entries.append((match.group(0), text[match.end():end]))
    return entries


LOCKS = [RUNTIME_LOCK, DEV_LOCK]


# =========================
# The files exist and target the right platform
# =========================

@pytest.mark.parametrize("lock", LOCKS, ids=lambda p: p.name)
def test_lock_exists_and_is_not_empty(lock):
    assert lock.is_file(), f"{lock.name} is missing"
    assert len(_locked_versions(lock)) > 50, "suspiciously few pins"


@pytest.mark.parametrize("lock", LOCKS, ids=lambda p: p.name)
def test_lock_header_records_the_resolution_target(lock):
    """Regenerating for a different platform would silently change the image."""
    header = lock.read_text(encoding="utf-8").split("\n", 2)[:2]
    header_text = "\n".join(header)

    assert "uv pip compile" in header_text, header_text
    for expected in EXPECTED_TARGET:
        assert expected in header_text, f"{lock.name} header lost {expected!r}"


# =========================
# Every entry is pinned and hashed
# =========================

@pytest.mark.parametrize("lock", LOCKS, ids=lambda p: p.name)
def test_every_entry_is_exactly_pinned(lock):
    text = lock.read_text(encoding="utf-8")

    loose = re.findall(r"^([A-Za-z0-9_.-]+)\s*(>=|<=|~=|!=|>|<)", text, re.M)
    assert not loose, f"{lock.name} has unpinned entries: {loose[:5]}"


@pytest.mark.parametrize("lock", LOCKS, ids=lambda p: p.name)
def test_every_entry_carries_a_hash(lock):
    """pip install --require-hashes refuses the whole file otherwise."""
    unhashed = [name for name, body in _lock_entries(lock) if "--hash=sha256:" not in body]

    assert not unhashed, f"{lock.name} entries without hashes: {unhashed[:5]}"


@pytest.mark.parametrize("lock", LOCKS, ids=lambda p: p.name)
def test_hashes_are_well_formed(lock):
    hashes = re.findall(r"--hash=sha256:([0-9a-f]*)", lock.read_text(encoding="utf-8"))

    assert hashes, "no hashes at all"
    malformed = [h for h in hashes if len(h) != 64]
    assert not malformed, f"{len(malformed)} malformed sha256 digests"


# =========================
# The locks agree with the input specs
# =========================

def test_runtime_lock_covers_every_direct_requirement():
    direct = _direct_requirements(RUNTIME_IN)
    locked = _locked_versions(RUNTIME_LOCK)

    missing = sorted(set(direct) - set(locked))
    assert not missing, f"requirements.txt entries absent from the lock: {missing}"


def test_dev_lock_covers_runtime_and_dev_requirements():
    direct = {**_direct_requirements(RUNTIME_IN), **_direct_requirements(DEV_IN)}
    locked = _locked_versions(DEV_LOCK)

    missing = sorted(set(direct) - set(locked))
    assert not missing, f"entries absent from the dev lock: {missing}"


@pytest.mark.parametrize(
    "input_file,lock_file",
    [(RUNTIME_IN, RUNTIME_LOCK), (DEV_IN, DEV_LOCK)],
    ids=["runtime", "dev"],
)
def test_locked_versions_satisfy_the_declared_specifiers(input_file, lock_file):
    """A changed pin in requirements.txt with a stale lock is the failure mode."""
    locked = _locked_versions(lock_file)

    for name, requirement in _direct_requirements(input_file).items():
        if name not in locked:
            continue  # covered by the coverage tests above
        version = locked[name]
        assert requirement.specifier.contains(version, prereleases=True), (
            f"{name}: requirements declares {requirement.specifier}, "
            f"lock pins {version} — regenerate the lock"
        )


def test_dev_lock_is_a_superset_of_the_runtime_lock():
    runtime = _locked_versions(RUNTIME_LOCK)
    dev = _locked_versions(DEV_LOCK)

    missing = sorted(set(runtime) - set(dev))
    assert not missing, f"packages in the runtime lock but not the dev lock: {missing}"


def test_runtime_and_dev_locks_agree_on_shared_versions():
    """Testing against different versions than the image ships is the bug."""
    runtime = _locked_versions(RUNTIME_LOCK)
    dev = _locked_versions(DEV_LOCK)

    disagreements = {
        name: (runtime[name], dev[name])
        for name in set(runtime) & set(dev)
        if runtime[name] != dev[name]
    }
    assert not disagreements, f"version skew between the locks: {disagreements}"


# =========================
# Specific regressions worth naming
# =========================

def test_posthog_is_constrained():
    """chromadb 0.4.24 requires only posthog>=2.4.0, and posthog 4 broke it."""
    assert "posthog" in _locked_versions(RUNTIME_LOCK)
    major = int(_locked_versions(RUNTIME_LOCK)["posthog"].split(".")[0])

    assert major < 4, "posthog 4+ changed capture() and floods the log at startup"


def test_dev_tools_are_absent_from_the_runtime_lock():
    """The image must not ship pytest and ruff."""
    runtime = _locked_versions(RUNTIME_LOCK)

    for tool in ("pytest", "ruff"):
        assert tool not in runtime, f"{tool} leaked into the runtime lock"
