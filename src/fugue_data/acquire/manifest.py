"""Write the source manifest to disk (Spec Section 6, 20.6).

We emit newline-delimited JSON as the machine format plus a CSV for humans.
(A Parquet mirror is produced by the export stage once pyarrow is installed;
JSONL keeps acquisition dependency-light and diff-friendly.)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

# Strict manifest columns per schemas/source_manifest.schema.json (Section 6.1).
MANIFEST_COLUMNS = [
    "source_file_id",
    "source_name",
    "repository_url",
    "repository_commit",
    "retrieved_at_utc",
    "relative_source_path",
    "sha256",
    "source_format",
    "composer_claimed",
    "work_id_claimed",
    "movement_claimed",
    "license_name",
    "license_url",
    "license_scope",
    "commercial_use_known",
    "attribution_required",
    "quarantine_reason",
    "ingest_status",
]


def _strict(row: dict) -> dict:
    return {k: row.get(k) for k in MANIFEST_COLUMNS}


def write_manifest(rows: list[dict], manifests_dir: Path) -> dict:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = manifests_dir / "sources.jsonl"
    csv_path = manifests_dir / "sources.csv"

    with jsonl_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(_strict(r), sort_keys=True) + "\n")

    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(_strict(r))

    return {"jsonl": str(jsonl_path), "csv": str(csv_path), "rows": len(rows)}
