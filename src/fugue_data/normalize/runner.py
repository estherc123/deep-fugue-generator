"""Normalize the WTC fugue corpus to canonical scores and validate (Spec §10).

Restartable: skips a piece whose canonical JSON already exists with a matching
source checksum and config hash, unless ``force`` is set.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..acquire.acquire import load_sources, acquire_source
from ..normalize.normalize import normalize_score, _config_hash, NormalizationError
from ..validation.canonical import validate_canonical


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


def _wtc_fugue_rows(root: Path) -> list[dict]:
    specs = load_sources(root / "config" / "sources.yaml")
    wtc = next(s for s in specs if s.source_name == "humdrum-wtc")
    result = acquire_source(wtc, root / "data" / "raw")
    return [r for r in result.rows
            if r["movement_claimed"] == "fugue" and r["_included_in_default"]]


def normalize_wtc(root: Path | None = None, *, force: bool = False,
                  limit: int | None = None) -> dict:
    root = root or _repo_root()
    pipeline = yaml.safe_load((root / "config" / "pipeline.yaml").read_text())
    canonical_dir = root / pipeline["paths"]["canonical"]
    canonical_dir.mkdir(parents=True, exist_ok=True)

    rows = _wtc_fugue_rows(root)
    rows.sort(key=lambda r: r["_piece_id"])
    if limit:
        rows = rows[:limit]

    cfg_hash = _config_hash()
    results = {"normalized": [], "skipped": [], "failed": [], "validation_failures": []}
    warning_total = 0

    for r in rows:
        pid = r["_piece_id"]
        out_path = canonical_dir / f"{pid}.json"
        src_path = root / "data" / "raw" / "humdrum-wtc" / r["relative_source_path"]

        if out_path.exists() and not force:
            existing = json.loads(out_path.read_text())
            same_src = r["sha256"] in [
                sr.get("sha256") for sr in existing.get("_source", [])] or \
                existing.get("normalization", {}).get("config_hash") == cfg_hash
            if same_src:
                results["skipped"].append(pid)
                warning_total += len(existing.get("warnings", []))
                continue

        try:
            score = normalize_score(
                src_path,
                piece_id=pid,
                work_group_id=r["work_id_claimed"],
                movement_type="fugue",
                source_file_ids=[r["source_file_id"]],
            )
        except NormalizationError as e:
            results["failed"].append({"piece_id": pid, "error": str(e)})
            continue

        doc = score.to_dict()
        # Record source provenance link inside the canonical file.
        doc["_source"] = [{
            "source_file_id": r["source_file_id"],
            "relative_source_path": r["relative_source_path"],
            "sha256": r["sha256"],
            "repository_commit": r["repository_commit"],
        }]

        errors = validate_canonical(doc)
        if errors:
            results["validation_failures"].append({"piece_id": pid, "errors": errors[:10]})
            # Still write it so the failure is inspectable, but under a .invalid name.
            _write_json(canonical_dir / f"{pid}.invalid.json", doc)
            continue

        _write_json(out_path, doc)
        warning_total += len(doc["warnings"])
        results["normalized"].append({
            "piece_id": pid,
            "voice_count": doc["voice_count"],
            "measure_count": doc["measure_count"],
            "notes": sum(len(v["notes"]) for v in doc["voices"]),
            "warnings": len(doc["warnings"]),
            "event_hash": doc["normalization"]["event_hash"],
        })

    summary = {
        "expected_wtc_fugues": 48,
        "normalized_count": len(results["normalized"]),
        "skipped_count": len(results["skipped"]),
        "failed_count": len(results["failed"]),
        "validation_failure_count": len(results["validation_failures"]),
        "total_parser_warnings": warning_total,
        "results": results,
    }
    (root / "manifests" / "normalization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    s = normalize_wtc(limit=limit)
    print(json.dumps({k: v for k, v in s.items() if k != "results"}, indent=2))
