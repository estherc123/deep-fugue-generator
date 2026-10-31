"""Generate the ``progress/`` record for the annotation stage (Spec §24).

Every count is re-derived from the aligned annotation files on disk, never
hand-typed. Run via ``python -m fugue_data.annotations.progress``.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .dezrann import load_config


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
    cfg = load_config(root)
    ann_dir = root / pipeline["paths"]["annotations"]
    progress = root / pipeline["paths"]["progress"]
    progress.mkdir(parents=True, exist_ok=True)

    docs = [json.loads(p.read_text()) for p in sorted(ann_dir.glob("*.json"))
            if not p.name.endswith(".invalid.json")]

    rows: list[dict] = []
    by_type: dict[str, int] = {}
    by_transform: dict[str, int] = {}
    cadence_types: dict[str, int] = {}
    coverages: list[float] = []
    entries = {"subject": 0, "answer": 0, "undetermined": 0, "entries": 0}
    totals = {"annotations": 0, "accepted": 0, "rejected": 0, "groups": 0, "warnings": 0}

    for d in docs:
        al = d["alignment"]
        accepted = [a for a in d["annotations"] if a["analytical_status"] == "accepted"]
        for a in accepted:
            by_type[a["type"]] = by_type.get(a["type"], 0) + 1
            for t in a.get("transformations") or []:
                by_transform[t] = by_transform.get(t, 0) + 1
            if a.get("cadence_type"):
                cadence_types[a["cadence_type"]] = cadence_types.get(a["cadence_type"], 0) + 1
        ef = al.get("entry_function") or {}
        for k in ("subject", "answer", "undetermined", "entries"):
            entries[k] += ef.get(k, 0) or 0
        if al.get("mean_voice_note_coverage") is not None:
            coverages.append(al["mean_voice_note_coverage"])
        totals["annotations"] += len(d["annotations"])
        totals["accepted"] += len(accepted)
        totals["rejected"] += al.get("rejected_candidate_count", 0)
        totals["groups"] += al.get("alternative_group_count", 0)
        totals["warnings"] += len(d.get("warnings", []))

        counts = al.get("counts_by_type", {})
        rows.append({
            "piece_id": d["piece_id"],
            "work_group_id": d.get("work_group_id"),
            "voice_count": al["voice_count"],
            "annotations_total": len(d["annotations"]),
            "accepted": len(accepted),
            "rejected_candidates": al.get("rejected_candidate_count", 0),
            "subject": counts.get("subject", 0),
            "countersubject": counts.get("countersubject", 0),
            "cadence": counts.get("cadence", 0),
            "pedal": counts.get("pedal", 0),
            "episode": counts.get("episode", 0),
            "other": counts.get("other", 0),
            "derived_answers": ef.get("answer", 0),
            "mean_voice_note_coverage": (
                None if al.get("mean_voice_note_coverage") is None
                else round(al["mean_voice_note_coverage"], 4)),
            "alternative_groups": al.get("alternative_group_count", 0),
            "warnings": len(d.get("warnings", [])),
        })

    _write_csv(progress / "algomus_alignment.csv", rows, [
        "piece_id", "work_group_id", "voice_count", "annotations_total", "accepted",
        "rejected_candidates", "subject", "countersubject", "cadence", "pedal",
        "episode", "other", "derived_answers", "mean_voice_note_coverage",
        "alternative_groups", "warnings"])

    all_wtc1 = {f"bach_wtc1_fugue_{i:02d}" for i in range(1, 25)}
    aligned_ids = {d["piece_id"] for d in docs}
    override = (cfg["acceptance"].get("overrides") or {}).get("ref_fugue_coverage")

    status = {
        "stage": "annotations",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": cfg["source"]["name"],
        "annotation_map_version": cfg["annotation_map_version"],
        "annotation_schema_version": docs[0]["schema_version"] if docs else None,
        "include_computed": cfg["source"].get("include_computed", False),
        "fugues_aligned": len(docs),
        "spec_expected_fugues": cfg["acceptance"]["expected_ref_fugues"],
        "fugues_without_annotations": sorted(all_wtc1 - aligned_ids),
        "overrides": {"ref_fugue_coverage": override} if override else {},
        "totals": {
            "annotations": totals["annotations"],
            "accepted": totals["accepted"],
            "rejected_candidates": totals["rejected"],
            "alternative_groups": totals["groups"],
            "warnings": totals["warnings"],
        },
        "accepted_by_type": dict(sorted(by_type.items())),
        "transformations": dict(sorted(by_transform.items(), key=lambda kv: -kv[1])),
        "cadence_types": dict(sorted(cadence_types.items(), key=lambda kv: -kv[1])),
        "derived_entry_function": entries,
        "voice_mapping": {
            "strategy": cfg["voice_mapping"]["strategy"],
            "min_coverage_gate": cfg["voice_mapping"]["min_voice_note_coverage"],
            "corpus_mean_voice_note_coverage": (
                round(sum(coverages) / len(coverages), 4) if coverages else None),
        },
    }
    (progress / "annotation_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, sort_keys=True))
