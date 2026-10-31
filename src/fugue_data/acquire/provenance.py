"""Git provenance: clone a repository and pin it to an immutable commit SHA.

Implements Spec Section 4.1. We never rely on a moving ``main``/``master`` branch
after acquisition: the repository is checked out in detached-HEAD state at a full
40-char SHA, HEAD is verified, and the ``.git`` directory is removed so the raw
tree on disk is immutable.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class ProvenanceError(RuntimeError):
    """Raised when a repository cannot be pinned to the expected commit."""


@dataclass
class Checkout:
    source_name: str
    repository_url: str
    repository_commit: str
    retrieved_at_utc: str
    root: Path  # directory containing the immutable checked-out tree (no .git)


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ProvenanceError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pin_repository(
    source_name: str,
    repository_url: str,
    repository_commit: str,
    dest_root: Path,
    *,
    force: bool = False,
) -> Checkout:
    """Clone ``repository_url``, check out ``repository_commit`` detached, strip .git.

    The result is an immutable raw tree at ``dest_root/<source_name>``. Restartable:
    if a valid pinned tree already exists (marked complete), it is reused unless
    ``force`` is set.
    """
    if len(repository_commit) != 40 or not all(
        c in "0123456789abcdef" for c in repository_commit.lower()
    ):
        raise ProvenanceError(
            f"{source_name}: repository_commit must be a full 40-char SHA, got "
            f"{repository_commit!r}"
        )

    target = dest_root / source_name
    marker = target / ".pinned_commit"

    if target.exists() and marker.exists() and not force:
        existing = marker.read_text().strip()
        if existing == repository_commit:
            # Already acquired at the requested commit — reuse (idempotent).
            retrieved = (target / ".retrieved_at_utc")
            ts = retrieved.read_text().strip() if retrieved.exists() else utc_now_iso()
            return Checkout(
                source_name, repository_url, repository_commit, ts, target
            )
        raise ProvenanceError(
            f"{source_name}: existing checkout is pinned to {existing} but config "
            f"requests {repository_commit}; pass force=True to re-acquire."
        )

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest_root / f".{source_name}.clone.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    retrieved_at = utc_now_iso()
    # Full clone (repos are small) then detached checkout of the exact SHA.
    _run(["git", "clone", "--quiet", repository_url, str(tmp)])
    _run(["git", "-C", str(tmp), "checkout", "--quiet", "--detach", repository_commit])

    head = _run(["git", "-C", str(tmp), "rev-parse", "HEAD"])
    if head.lower() != repository_commit.lower():
        shutil.rmtree(tmp)
        raise ProvenanceError(
            f"{source_name}: HEAD {head} does not match requested {repository_commit}"
        )

    # Strip .git to make the raw tree immutable and self-contained.
    git_dir = tmp / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)

    tmp.rename(target)
    marker.write_text(repository_commit + "\n")
    (target / ".retrieved_at_utc").write_text(retrieved_at + "\n")

    return Checkout(source_name, repository_url, repository_commit, retrieved_at, target)
