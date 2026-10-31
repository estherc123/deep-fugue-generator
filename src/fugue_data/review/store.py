"""Concurrency-safe JSONL label store for multi-reviewer review (Spec §18.5).

The single-user apps read the whole file, change one record and write it back.
That is safe only because `http.server.HTTPServer` is single-threaded, so requests
inside one process serialise. The moment the app is hosted -- behind any threaded
server, or with more than one worker -- two reviewers submitting at the same time
each write a copy of the file built from a stale read, and one label vanishes
with no error anywhere.

An advisory lock around read-modify-write makes the update atomic across
processes, and the write is still staged through a temp file so an interrupted
save cannot truncate the store.
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def locked(path: Path):
    """Hold an exclusive advisory lock for the duration of the block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(
        json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows))
    tmp.replace(path)


def update(path: Path, mutate, *, sort_key=None):
    """Read, mutate and write back under an exclusive lock.

    `mutate(rows)` edits the list in place and returns whatever the caller needs.
    The read happens *inside* the lock, so no update is ever computed from a
    snapshot another writer has already replaced.
    """
    with locked(path):
        rows = read_jsonl(path)
        result = mutate(rows)
        if sort_key:
            rows.sort(key=sort_key)
        write_jsonl(path, rows)
    return result


def assigned_slice(pair_ids: list[str], reviewer: str, reviewers: int) -> list[str]:
    """Deterministic per-reviewer partition of the pending pool.

    Several reviewers browsing the same queue would all start at the top and
    label the same pairs, so the effort multiplies without covering more ground.
    Hashing the pair id against the reviewer's name splits the pool into disjoint
    slices with no claim protocol, no leases and nothing to time out.
    """
    if reviewers <= 1:
        return pair_ids
    import hashlib
    slot = int(hashlib.sha256(reviewer.encode()).hexdigest()[:8], 16) % reviewers
    out = []
    for pid in pair_ids:
        if int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16) % reviewers == slot:
            out.append(pid)
    return out
