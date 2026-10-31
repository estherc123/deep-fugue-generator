"""Normalize BCFB chorales to canonical 4-voice SATB scores (Spec §10, §10.2).

The BCFB MusicXML files are full cantata movements: 4 vocal parts (all named
"Voice") plus, in most files, instrumental doublings and a continuo. The chorale
proper is the 4 vocal parts, so we select those as the canonical voices and
exclude the rest (recorded as ``excluded_parts``).

Cross-format validation (§10.2): the repo ships each score also as translated
kern and MEI. We compare per-part note-count vectors and totals across the three
encodings and record discrepancies, preferring MusicXML as the canonical source.
"""

from __future__ import annotations

import json
import warnings as _warnings
from pathlib import Path

import yaml

_warnings.filterwarnings("ignore")
from music21 import converter

from ..acquire.acquire import load_sources, acquire_source
from ..normalize.normalize import normalize_score, _config_hash, NormalizationError
from ..validation.canonical import validate_canonical


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_vocal(index, part) -> bool:
    return (part.partName or "").strip().lower() == "voice"


def _write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


def _part_note_counts(path: Path) -> list[int] | None:
    """Sorted per-part note counts for any format, or None if unparseable."""
    try:
        s = converter.parse(str(path))
    except Exception:
        return None
    return sorted(len(list(p.recurse().notes)) for p in s.parts)


def _cross_format_check(root: Path, stem: str) -> list[dict]:
    """Compare MusicXML vs translated kern vs MEI note structure (§10.2)."""
    fb = root / "data" / "raw" / "bcfb" / "FB_source"
    paths = {
        "musicxml": fb / "musicXML_master" / f"{stem}.musicxml",
        "kern": fb / "translated_kern" / f"{stem}.musicxml.krn",
        "mei": fb / "translated_mei" / f"{stem}.musicxml.krn.mei",
    }
    counts = {fmt: _part_note_counts(p) for fmt, p in paths.items() if p.exists()}
    ref = counts.get("musicxml")
    warns: list[dict] = []
    if ref is None:
        return warns
    ref_total = sum(ref)
    for fmt in ("kern", "mei"):
        c = counts.get(fmt)
        if c is None:
            warns.append({"code": "cross_format_unavailable",
                          "message": f"{fmt} encoding missing or unparseable for {stem}",
                          "voice_id": None, "onset_q": None})
            continue
        if sum(c) != ref_total or c != ref:
            warns.append({
                "code": "cross_format_discrepancy",
                "message": (f"{fmt} note structure differs from musicxml for {stem}: "
                            f"musicxml per-part={ref} (total {ref_total}) vs "
                            f"{fmt} per-part={c} (total {sum(c)})"),
                "voice_id": None, "onset_q": None,
            })
    return warns


def _chorale_variant_rows(root: Path) -> list[dict]:
    specs = load_sources(root / "config" / "sources.yaml")
    bcfb = next(s for s in specs if s.source_name == "bcfb")
    result = acquire_source(bcfb, root / "data" / "raw")
    # Primary canonical variants = the 143 MusicXML master files.
    return [r for r in result.rows if r["_included_in_default"]]


def normalize_chorales(root: Path | None = None, *, force: bool = False,
                       limit: int | None = None, cross_format: bool = True) -> dict:
    root = root or _repo_root()
    pipeline = yaml.safe_load((root / "config" / "pipeline.yaml").read_text())
    canonical_dir = root / pipeline["paths"]["canonical"]
    canonical_dir.mkdir(parents=True, exist_ok=True)

    rows = _chorale_variant_rows(root)
    rows.sort(key=lambda r: r["_piece_id"])
    if limit:
        rows = rows[:limit]

    cfg_hash = _config_hash()
    results = {"normalized": [], "skipped": [], "failed": [], "validation_failures": []}
    work_groups: set[str] = set()
    cross_format_discrepancies = 0

    for r in rows:
        pid = r["_piece_id"]                       # e.g. bach_chorale_bwv10.07a
        work_groups.add(r["work_id_claimed"])      # e.g. BWV10.07 (groups a/b)
        out_path = canonical_dir / f"{pid}.json"
        src_path = root / "data" / "raw" / "bcfb" / r["relative_source_path"]
        stem = Path(r["relative_source_path"]).stem  # BWV_10.07a_FB

        if out_path.exists() and not force:
            existing = json.loads(out_path.read_text())
            if existing.get("normalization", {}).get("config_hash") == cfg_hash:
                results["skipped"].append(pid)
                continue

        try:
            score = normalize_score(
                src_path, piece_id=pid, work_group_id=r["work_id_claimed"],
                movement_type="chorale", source_file_ids=[r["source_file_id"]],
                part_filter=_is_vocal,
            )
        except NormalizationError as e:
            results["failed"].append({"piece_id": pid, "error": str(e)})
            continue

        doc = score.to_dict()
        if cross_format:
            xwarns = _cross_format_check(root, stem)
            doc["warnings"].extend(xwarns)
            cross_format_discrepancies += sum(
                1 for w in xwarns if w["code"] == "cross_format_discrepancy")
        doc["_source"] = [{
            "source_file_id": r["source_file_id"],
            "relative_source_path": r["relative_source_path"],
            "sha256": r["sha256"],
            "repository_commit": r["repository_commit"],
        }]

        errors = validate_canonical(doc)
        if errors:
            results["validation_failures"].append({"piece_id": pid, "errors": errors[:10]})
            _write_json(canonical_dir / f"{pid}.invalid.json", doc)
            continue

        _write_json(out_path, doc)
        results["normalized"].append({
            "piece_id": pid, "work_group_id": r["work_id_claimed"],
            "voice_count": doc["voice_count"],
            "measure_count": doc["measure_count"],
            "notes": sum(len(v["notes"]) for v in doc["voices"]),
            "excluded_parts": doc["normalization"].get("excluded_parts", []),
        })

    summary = {
        "expected_variants": 143,
        "expected_unique_chorales": 139,
        "normalized_count": len(results["normalized"]),
        "skipped_count": len(results["skipped"]),
        "failed_count": len(results["failed"]),
        "validation_failure_count": len(results["validation_failures"]),
        "unique_work_groups_seen": len(work_groups),
        "cross_format_discrepancies": cross_format_discrepancies,
        "results": results,
    }
    (root / "manifests" / "normalization_chorales_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    s = normalize_chorales(limit=limit)
    print(json.dumps({k: v for k, v in s.items() if k != "results"}, indent=2))
