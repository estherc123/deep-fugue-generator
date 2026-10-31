"""Run deduplication and complete-work splitting, and enforce §12/§13.5.

Deterministic and idempotent: the same corpus, config and seed reproduce the same
manifest byte for byte, so re-running is safe and a diff means something changed
upstream.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from ..validation.splits import validate_splits
from .dedup import build_duplicate_groups, load_signatures
from .features import length_bucket, length_thresholds, piece_features
from .report import write_split_report
from .split import assign_splits, build_split_units, split_manifest_rows


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows))


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["stratification_fields"] = json.dumps(
                r.get("stratification_fields", {}), sort_keys=True)
            w.writerow({c: out.get(c) for c in columns})


def run_split(root: Path | None = None, *, seed: int | None = None) -> dict:
    root = root or _repo_root()
    cfg = yaml.safe_load((root / "config" / "splits.yaml").read_text())
    pipeline = yaml.safe_load((root / "config" / "pipeline.yaml").read_text())
    canonical_dir = root / pipeline["paths"]["canonical"]
    manifests = root / pipeline["paths"]["manifests"]
    seed = cfg["split_seed"] if seed is None else seed

    # ---- §12 deduplication ------------------------------------------------ #
    signatures = load_signatures(canonical_dir)
    dedup = build_duplicate_groups(signatures, cfg["dedup"]["blocking_levels"])
    _write_json(manifests / "duplicate_groups.json", {
        "splits_version": cfg["splits_version"],
        "blocking_levels": cfg["dedup"]["blocking_levels"],
        "level_census": dedup["level_census"],
        "groups": dedup["groups"],
    })

    # ---- features + corpus-relative length buckets ------------------------ #
    features = []
    for p in sorted(canonical_dir.glob("*.json")):
        if p.name.endswith(".invalid.json"):
            continue
        features.append(piece_features(json.loads(p.read_text())))
    for corpus in ("wtc", "bcfb"):
        subset = [f for f in features if f["corpus"] == corpus]
        thresholds = length_thresholds([f["measure_count"] for f in subset])
        for f in subset:
            f["length_bucket"] = length_bucket(f["measure_count"], thresholds)
            f["length_thresholds"] = list(thresholds)

    # ---- §13 assignment, per corpus --------------------------------------- #
    all_rows: list[dict] = []
    per_corpus: dict[str, dict] = {}
    for corpus, ccfg in sorted(cfg["corpora"].items()):
        subset = [f for f in features if f["corpus"] == corpus]
        units = build_split_units(subset, dedup["group_of_piece"])
        units = assign_splits(units, ccfg["targets"], ccfg["stratify_by"], seed)
        rows = split_manifest_rows(units, seed, corpus)
        all_rows.extend(rows)

        works = {}
        pieces = {}
        for r in rows:
            works.setdefault(r["split"], set()).add(r["work_group_id"])
            pieces[r["split"]] = pieces.get(r["split"], 0) + 1
        per_corpus[corpus] = {
            "targets": ccfg["targets"],
            "stratify_by": ccfg["stratify_by"],
            "split_units": len(units),
            "pieces_by_split": dict(sorted(pieces.items())),
            "works_by_split": {k: len(v) for k, v in sorted(works.items())},
            "strata": len({u["stratum"] for u in units}),
        }

    all_rows.sort(key=lambda r: r["piece_id"])
    _write_jsonl(manifests / "splits.jsonl", all_rows)
    _write_csv(manifests / "splits.csv", all_rows, [
        "piece_id", "work_group_id", "corpus", "split", "split_seed",
        "split_unit_id", "split_unit_weight", "stratification_fields",
        "mode_source", "assignment_hash"])

    # ---- §13.5 leakage gates ---------------------------------------------- #
    targets = {c: v["targets"] for c, v in cfg["corpora"].items()}
    errors = validate_splits(all_rows, dedup["groups"], targets)

    summary = {
        "stage": "split",
        "splits_version": cfg["splits_version"],
        "split_seed": seed,
        "total_pieces": len(all_rows),
        "total_split_units": sum(v["split_units"] for v in per_corpus.values()),
        "duplicate_groups": len(dedup["groups"]),
        "duplicate_level_census": dedup["level_census"],
        "corpora": per_corpus,
        "leakage_errors": errors,
        "failures": errors[:20],
    }
    _write_json(manifests / "split_summary.json", summary)
    write_split_report(summary, dedup, all_rows,
                       root / pipeline["paths"]["reports"] / "splits.md")
    return summary
