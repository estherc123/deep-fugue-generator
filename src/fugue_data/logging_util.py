"""Structured stage logging (Spec §24.1).

One JSONL line per stage invocation with the fields §24.1 names, appended to
`reports/pipeline.log.jsonl` so a build leaves an audit trail independent of
whatever was printed to the terminal.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def log_event(stage: str, *, status: str, piece_id=None, source_file_id=None,
              operation=None, warning_code=None, error_code=None,
              duration_ms=None, details=None, root: Path | None = None) -> None:
    root = root or _repo_root()
    path = root / "reports" / "pipeline.log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": stage, "piece_id": piece_id, "source_file_id": source_file_id,
        "operation": operation, "status": status, "warning_code": warning_code,
        "error_code": error_code, "duration_ms": duration_ms,
        "details": details or {},
    }
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


@contextmanager
def stage(name: str, operation: str = "run", root: Path | None = None):
    """Log a stage's start and outcome, including on failure."""
    t0 = time.time()
    record: dict = {}
    try:
        yield record
    except Exception as e:
        log_event(name, status="error", operation=operation,
                  error_code=type(e).__name__,
                  duration_ms=int((time.time() - t0) * 1000),
                  details={"message": str(e)[:500]}, root=root)
        raise
    failures = record.get("failures") or []
    log_event(name, status="failed" if failures else "ok", operation=operation,
              duration_ms=int((time.time() - t0) * 1000),
              warning_code=("acceptance_failure" if failures else None),
              details={k: v for k, v in record.items()
                       if k in ("accepted", "pieces", "rows", "failures", "count")},
              root=root)
