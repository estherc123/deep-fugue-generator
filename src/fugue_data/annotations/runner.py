"""Run annotation alignment over the corpus and enforce §11.4 (Spec §11).

Restartable: an annotation set whose source `.dez` checksums and annotation-map
version already match on disk is skipped unless ``force`` is set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ..normalize.rational import parse_frac
from ..validation.annotations import validate_annotations
from .align import (ANNOTATION_SCHEMA_VERSION, AlignmentError, align_dez,
                    assign_alternative_groups, piece_quarter_length)
from .dezrann import discover_dez_files, load_config
from .entries import classify_entries
from .report import write_alignment_report


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(annotations: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for a in annotations:
        k = str(a.get(key))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def annotate_algomus(root: Path | None = None, *, force: bool = False,
                     limit: int | None = None) -> dict:
    root = root or _repo_root()
    cfg = load_config(root)
    pipeline = yaml.safe_load((root / "config" / "pipeline.yaml").read_text())
    canonical_dir = root / pipeline["paths"]["canonical"]
    ann_dir = root / pipeline["paths"]["annotations"]
    ann_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = root / pipeline["paths"]["reports"] / "annotations"
    map_version = cfg["annotation_map_version"]

    # Group source analyses by piece: a piece may carry several annotators.
    by_piece: dict[str, list] = {}
    for f in discover_dez_files(root):
        by_piece.setdefault(f.piece_id, []).append(f)
    piece_ids = sorted(by_piece)
    if limit:
        piece_ids = piece_ids[:limit]

    results = {"aligned": [], "skipped": [], "failed": [], "validation_failures": []}
    warning_total = 0
    coverages: list[float] = []

    for pid in piece_ids:
        dez_files = by_piece[pid]
        score_path = canonical_dir / f"{pid}.json"
        if not score_path.exists():
            results["failed"].append({"piece_id": pid, "error": "no canonical score"})
            continue
        score = json.loads(score_path.read_text())
        out_path = ann_dir / f"{pid}.json"

        sources = [{
            "source_file_id": f"algomus:{_sha256(f.path)[:12]}",
            "relative_source_path": f.path.relative_to(root / "data" / "raw" / "algomus").as_posix(),
            "sha256": _sha256(f.path),
            "annotator_kind": f.annotator_kind,
        } for f in dez_files]

        if out_path.exists() and not force:
            existing = json.loads(out_path.read_text())
            same = (existing.get("annotation_map_version") == map_version
                    and [s["sha256"] for s in existing.get("_source", [])]
                    == [s["sha256"] for s in sources])
            if same:
                results["skipped"].append(pid)
                warning_total += len(existing.get("warnings", []))
                cov = existing.get("alignment", {}).get("mean_voice_note_coverage")
                if cov is not None:
                    coverages.append(cov)
                continue

        annotations: list[dict] = []
        warnings: list[dict] = []
        try:
            for f in dez_files:
                anns, warns = align_dez(f, score, cfg)
                annotations.extend(anns)
                warnings.extend(warns)
        except AlignmentError as e:
            results["failed"].append({"piece_id": pid, "error": str(e)})
            continue

        annotations.sort(key=lambda a: (parse_frac(a["onset_q"]), a["annotation_id"]))
        n_groups = assign_alternative_groups(annotations, pid)
        entry_summary = classify_entries(annotations, score, cfg)

        cov_values = [a["voice_note_coverage"] for a in annotations
                      if a.get("voice_note_coverage") is not None
                      and a["analytical_status"] == "accepted"]
        mean_cov = sum(cov_values) / len(cov_values) if cov_values else None
        accepted = [a for a in annotations if a["analytical_status"] == "accepted"]

        doc = {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "piece_id": pid,
            "work_group_id": score.get("work_group_id"),
            "source_name": cfg["source"]["name"],
            "annotation_map_version": map_version,
            "source_file_ids": [s["source_file_id"] for s in sources],
            "annotations": annotations,
            "alignment": {
                "voice_mapping_strategy": cfg["voice_mapping"]["strategy"],
                "piece_quarter_length": score.get("quarter_length_total"),
                "voice_count": score["voice_count"],
                "mean_voice_note_coverage": mean_cov,
                "min_voice_note_coverage": min(cov_values) if cov_values else None,
                "counts_by_type": _counts(accepted, "type"),
                "counts_by_annotator": _counts(annotations, "annotator"),
                "rejected_candidate_count": len(annotations) - len(accepted),
                "alternative_group_count": n_groups,
                "entry_function": entry_summary,
            },
            "warnings": warnings,
            "_source": sources,
        }

        errors = validate_annotations(doc, score)
        if errors:
            results["validation_failures"].append({"piece_id": pid, "errors": errors[:10]})
            _write_json(ann_dir / f"{pid}.invalid.json", doc)
            continue

        _write_json(out_path, doc)
        write_alignment_report(doc, score, reports_dir / f"{pid}.md")
        warning_total += len(warnings)
        if mean_cov is not None:
            coverages.append(mean_cov)
        results["aligned"].append({
            "piece_id": pid,
            "annotations": len(annotations),
            "accepted": len(accepted),
            "rejected_candidates": len(annotations) - len(accepted),
            "alternative_groups": n_groups,
            "mean_voice_note_coverage": mean_cov,
            "warnings": len(warnings),
        })

    acceptance = cfg["acceptance"]
    overrides = acceptance.get("overrides") or {}
    covered = len(results["aligned"]) + len(results["skipped"])
    expected = acceptance["expected_ref_fugues"]
    override = overrides.get("ref_fugue_coverage")
    if override:
        expected = override["accepted_value"]

    failures = []
    if not limit and covered != expected:
        failures.append(f"ref_fugue_coverage: expected {expected}, got {covered}")
    if results["failed"]:
        failures.append(f"{len(results['failed'])} piece(s) failed alignment")
    if results["validation_failures"]:
        failures.append(f"{len(results['validation_failures'])} piece(s) failed validation")

    summary = {
        "stage": "annotations",
        "source": "algomus",
        "annotation_map_version": map_version,
        "expected_ref_fugues": expected,
        "spec_expected_ref_fugues": acceptance["expected_ref_fugues"],
        "overrides": overrides,
        "aligned_count": len(results["aligned"]),
        "skipped_count": len(results["skipped"]),
        "failed_count": len(results["failed"]),
        "validation_failure_count": len(results["validation_failures"]),
        "total_annotations": sum(r["annotations"] for r in results["aligned"]),
        "total_accepted": sum(r["accepted"] for r in results["aligned"]),
        "total_rejected_candidates": sum(r["rejected_candidates"] for r in results["aligned"]),
        "total_alternative_groups": sum(r["alternative_groups"] for r in results["aligned"]),
        "corpus_mean_voice_note_coverage": (
            sum(coverages) / len(coverages) if coverages else None),
        "total_warnings": warning_total,
        "failures": failures,
        "results": results,
    }
    (root / "manifests" / "annotation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
