"""Generate the ``progress/`` record for the dedup + split stage (Spec §24).

Every count is re-derived from the split manifest on disk, never hand-typed.
Run via ``python -m fugue_data.splitting.progress``.
"""

from __future__ import annotations

import collections
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

SPLITS = ("train", "validation", "test")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def generate(root: Path | None = None) -> dict:
    root = root or _repo_root()
    pipeline = yaml.safe_load((root / "config" / "pipeline.yaml").read_text())
    manifests = root / pipeline["paths"]["manifests"]
    progress = root / pipeline["paths"]["progress"]
    progress.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in
            (manifests / "splits.jsonl").read_text().splitlines() if l.strip()]
    summary = json.loads((manifests / "split_summary.json").read_text())
    dupes = json.loads((manifests / "duplicate_groups.json").read_text())

    with (progress / "splits.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "piece_id", "work_group_id", "corpus", "split", "split_unit_id",
            "book", "mode", "voice_count", "meter", "length_bucket"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["piece_id"]):
            sf = r.get("stratification_fields", {})
            w.writerow({
                "piece_id": r["piece_id"], "work_group_id": r["work_group_id"],
                "corpus": r["corpus"], "split": r["split"],
                "split_unit_id": r["split_unit_id"],
                "book": sf.get("book"), "mode": sf.get("mode"),
                "voice_count": sf.get("voice_count"), "meter": sf.get("meter"),
                "length_bucket": sf.get("length_bucket"),
            })

    balance: dict = {}
    for corpus, ccfg in summary["corpora"].items():
        sub = [r for r in rows if r["corpus"] == corpus]
        fields: dict = {}
        for field in ccfg["stratify_by"]:
            table: dict = collections.defaultdict(lambda: {s: 0 for s in SPLITS})
            for r in sub:
                v = r["stratification_fields"].get(field)
                if v is not None:
                    table[str(v)][r["split"]] += 1
            fields[field] = {k: dict(v) for k, v in sorted(table.items())}
        balance[corpus] = fields

    status = {
        "stage": "split",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "splits_version": summary["splits_version"],
        "split_seed": summary["split_seed"],
        "total_pieces": summary["total_pieces"],
        "total_split_units": summary["total_split_units"],
        "duplicate_groups": [
            {"duplicate_group_id": g["duplicate_group_id"], "members": g["members"],
             "preferred_piece_id": g["preferred_piece_id"],
             "matched_levels": g["matched_levels"],
             "work_group_ids": g["work_group_ids"]}
            for g in dupes["groups"]],
        "duplicate_level_census": summary["duplicate_level_census"],
        "corpora": summary["corpora"],
        "stratification_balance": balance,
        "leakage_gates_pass": not summary["leakage_errors"],
        "leakage_errors": summary["leakage_errors"],
    }
    (progress / "split_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, sort_keys=True))
