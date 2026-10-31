"""Generate the ``progress/`` dataset-collection record from acquired sources.

Reproducible: re-derives every row from the pinned raw checkouts (idempotent,
reuses existing pins) so the progress folder always reflects real files on disk,
never hand-typed counts. Run via ``python -m fugue_data.acquire.report``.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .acquire import load_sources, acquire_source


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in columns})


def generate(root: Path | None = None) -> dict:
    root = root or _repo_root()
    pipeline = yaml.safe_load((root / "config" / "pipeline.yaml").read_text())
    raw_root = root / pipeline["paths"]["raw"]
    progress = root / pipeline["paths"]["progress"]
    progress.mkdir(parents=True, exist_ok=True)
    overrides = pipeline["acquisition"].get("overrides") or {}

    specs = load_sources(root / "config" / "sources.yaml")
    results = {s.source_name: acquire_source(s, raw_root) for s in specs}
    summaries = {name: r.counts() for name, r in results.items()}
    checkouts = {name: r.checkout for name, r in results.items()}

    # ---- WTC fugues inventory -------------------------------------------- #
    wtc_rows = []
    for r in results["humdrum-wtc"].rows:
        if r["movement_claimed"] != "fugue":
            continue
        pid = r["_piece_id"]
        book = 1 if pid.startswith("bach_wtc1") else 2
        wtc_rows.append({
            "piece_id": pid,
            "book": book,
            "work_group_id": r["work_id_claimed"],
            "movement": "fugue",
            "source_format": r["source_format"],
            "relative_source_path": r["relative_source_path"],
            "sha256": r["sha256"],
        })
    wtc_rows.sort(key=lambda x: x["piece_id"])
    _write_csv(progress / "wtc_fugues.csv", wtc_rows,
               ["piece_id", "book", "work_group_id", "movement",
                "source_format", "relative_source_path", "sha256"])

    # ---- BCFB chorale inventory (139 unique / 143 variants) --------------- #
    bcfb_variants = [r for r in results["bcfb"].rows if r["_included_in_default"]]
    bcfb_variants.sort(key=lambda x: (x["work_id_claimed"], x["_variant_id"]))
    _write_csv(progress / "bcfb_chorales.csv", [
        {
            "work_group_id": r["work_id_claimed"],
            "piece_id": r["_piece_id"],
            "variant_id": r["_variant_id"],
            "source_format": r["source_format"],
            "relative_source_path": r["relative_source_path"],
            "sha256": r["sha256"],
        } for r in bcfb_variants
    ], ["work_group_id", "piece_id", "variant_id", "source_format",
        "relative_source_path", "sha256"])

    # ---- Algomus annotation coverage ------------------------------------- #
    algo_ref = {}
    algo_computed = set()
    for r in results["algomus"].rows:
        if not r["relative_source_path"].endswith(".dez"):
            continue
        pid = r["_piece_id"]
        if r["_variant_id"] == "algomus-ref":
            algo_ref[pid] = r
        elif r["_variant_id"] == "algomus-computed":
            algo_computed.add(pid)
    algo_rows = []
    for i in range(1, 25):
        pid = f"bach_wtc1_fugue_{i:02d}"
        ref = algo_ref.get(pid)
        algo_rows.append({
            "piece_id": pid,
            "fugue_index": i,
            "work_group_id": ref["work_id_claimed"] if ref else f"BWV{845 + i}",
            "has_reference_annotation": bool(ref),
            "has_computed_annotation": pid in algo_computed,
            "ref_source_path": ref["relative_source_path"] if ref else "",
        })
    _write_csv(progress / "algomus_annotations.csv", algo_rows,
               ["piece_id", "fugue_index", "work_group_id",
                "has_reference_annotation", "has_computed_annotation",
                "ref_source_path"])

    # ---- Machine-readable status ----------------------------------------- #
    status = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": "acquisition",
        "sources": {
            name: {
                "repository_url": co.repository_url,
                "repository_commit": co.repository_commit,
                "retrieved_at_utc": co.retrieved_at_utc,
            } for name, co in checkouts.items()
        },
        "summaries": summaries,
        "overrides": overrides,
        "totals": {
            "wtc_fugues": len(wtc_rows),
            "bcfb_variants": len(bcfb_variants),
            "bcfb_unique_chorales": summaries["bcfb"]["unique_chorales"],
            "algomus_ref_fugues": summaries["algomus"]["ref_annotation_fugues"],
            "unique_real_bach_movements": (
                len(wtc_rows) + summaries["bcfb"]["unique_chorales"]
            ),
        },
    }
    (progress / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status


def generate_normalization(root: Path | None = None) -> dict:
    """Record canonical-normalization results into progress/ (Spec §10.4)."""
    import glob
    root = root or _repo_root()
    pipeline = yaml.safe_load((root / "config" / "pipeline.yaml").read_text())
    canonical_dir = root / pipeline["paths"]["canonical"]
    progress = root / pipeline["paths"]["progress"]
    progress.mkdir(parents=True, exist_ok=True)

    fugue_rows, chorale_rows = [], []
    stats = {"fugue": {"dist": {}, "notes": 0, "bars": 0},
             "chorale": {"dist": {}, "notes": 0, "bars": 0, "work_groups": set()}}
    for f in sorted(glob.glob(str(canonical_dir / "*.json"))):
        if f.endswith(".invalid.json"):
            continue
        d = json.loads(Path(f).read_text())
        notes = sum(len(v["notes"]) for v in d["voices"])
        mt = d["movement_type"]
        meter = d["meter_map"][0] if d["meter_map"] else {}
        key = d["key_signature_map"][0] if d["key_signature_map"] else {}
        row = {
            "piece_id": d["piece_id"], "work_group_id": d["work_group_id"],
            "voice_count": d["voice_count"], "measure_count": d["measure_count"],
            "notes": notes,
            "meter": f'{meter.get("numerator","?")}/{meter.get("denominator","?")}',
            "key_fifths": key.get("fifths"),
            "quarter_length_total": d["quarter_length_total"],
            "warnings": len(d["warnings"]),
            "event_hash": d["normalization"]["event_hash"][:16],
        }
        bucket = stats.get(mt)
        if mt == "fugue":
            fugue_rows.append(row)
        elif mt == "chorale":
            row["excluded_parts"] = ";".join(d["normalization"].get("excluded_parts", []))
            chorale_rows.append(row)
            bucket["work_groups"].add(d["work_group_id"])
        if bucket is not None:
            bucket["dist"][d["voice_count"]] = bucket["dist"].get(d["voice_count"], 0) + 1
            bucket["notes"] += notes
            bucket["bars"] += d["measure_count"] or 0

    _write_csv(progress / "canonical_fugues.csv", fugue_rows,
               ["piece_id", "work_group_id", "voice_count", "measure_count", "notes",
                "meter", "key_fifths", "quarter_length_total", "warnings", "event_hash"])
    _write_csv(progress / "canonical_chorales.csv", chorale_rows,
               ["piece_id", "work_group_id", "voice_count", "measure_count", "notes",
                "meter", "key_fifths", "quarter_length_total", "excluded_parts",
                "warnings", "event_hash"])

    def _load(name):
        p = root / "manifests" / name
        return json.loads(p.read_text()) if p.exists() else {}
    fsum, csum = _load("normalization_summary.json"), _load("normalization_chorales_summary.json")

    out = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": "normalization",
        "fugues": {
            "canonical_files": len(fugue_rows), "expected": 48,
            "voice_count_distribution": {str(k): v for k, v in sorted(stats["fugue"]["dist"].items())},
            "total_notes": stats["fugue"]["notes"], "total_bars": stats["fugue"]["bars"],
            "validation_failures": fsum.get("validation_failure_count", 0),
            "failed": fsum.get("failed_count", 0),
        },
        "chorales": {
            "canonical_variant_files": len(chorale_rows), "expected_variants": 143,
            "unique_work_groups": len(stats["chorale"]["work_groups"]), "expected_unique": 139,
            "voice_count_distribution": {str(k): v for k, v in sorted(stats["chorale"]["dist"].items())},
            "total_notes": stats["chorale"]["notes"], "total_bars": stats["chorale"]["bars"],
            "validation_failures": csum.get("validation_failure_count", 0),
            "failed": csum.get("failed_count", 0),
            "cross_format_discrepancies": csum.get("cross_format_discrepancies", 0),
        },
        "corpus_total_pieces_normalized": len(fugue_rows) + len(chorale_rows),
        "corpus_total_bars": stats["fugue"]["bars"] + stats["chorale"]["bars"],
        "corpus_total_notes": stats["fugue"]["notes"] + stats["chorale"]["notes"],
    }
    (progress / "normalization_status.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


if __name__ == "__main__":
    s = generate()
    print(json.dumps(s["totals"], indent=2))
    n = generate_normalization()
    print(json.dumps(n, indent=2))
    print("progress/ written.")
