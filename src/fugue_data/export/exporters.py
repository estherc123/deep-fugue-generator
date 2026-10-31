"""Produce the five training exports plus metadata (Spec §20).

Each export is a separate dataset with its own contract. Two rules are enforced
here rather than assumed downstream:

  §20.1 / §27  Only *approved* excerpts may appear in an export marked
               human-reviewed; a pending excerpt never does.
  §20.5 / §18.6  Only *reviewed* preference pairs are exported.

Both currently produce empty, clearly-labelled outputs, because those two
datasets depend on human review that has not happened yet. An empty file that
says why is the honest artefact; a populated one would be a false claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
import zstandard as zstd

from ..normalize.rational import parse_frac
from ..tokenization.views import event_records, grid_tokens
from ..tokenization.vocab import build_vocabulary, vocabulary_hash

SPLITS = ("train", "validation", "test")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_jsonl_zst(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
                      for r in rows).encode("utf-8")
    path.write_bytes(zstd.ZstdCompressor(level=10).compress(payload))
    return len(rows)


def _write_parquet(path: Path, rows: list[dict]) -> int:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    df.to_parquet(path, index=False)
    return len(rows)


def _splits_map(root: Path) -> dict[str, str]:
    out = {}
    p = root / "manifests" / "splits.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["piece_id"]] = r["split"]
    return out


def _iter_shards(d: Path):
    for shard in sorted(d.glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def _tokenize(doc: dict, tcfg: dict, annotations=None) -> dict:
    try:
        toks, gpolicy = grid_tokens(doc, tcfg, annotations)
    except Exception as e:
        toks, gpolicy = None, {"view": "grid", "exact": False,
                               "unsupported_reason": str(e)}
    events, epolicy = event_records(doc, tcfg, annotations)
    return {"grid_tokens": toks, "grid_policy": gpolicy,
            "event_records": events, "event_policy": epolicy}


def export_all(root: Path | None = None) -> dict:
    root = root or _repo_root()
    pipeline = yaml.safe_load((root / "config" / "pipeline.yaml").read_text())
    tcfg = yaml.safe_load((root / "config" / "tokenizer.yaml").read_text())
    gen_cfg = yaml.safe_load((root / "config" / "generation.yaml").read_text())
    out = root / "data" / "exports"
    canonical = root / "data" / "canonical"
    ann_dir = root / "data" / "annotations"
    splits = _splits_map(root)
    vocab = build_vocabulary()
    vhash = vocabulary_hash(vocab)
    tmeta = {"tokenizer_name": tcfg["tokenizer_name"],
             "tokenizer_version": tcfg["tokenizer_version"],
             "vocabulary_hash": vhash}

    summary: dict = {"stage": "export", "exports": {}, "failures": []}

    # ---- §20.1 style prior -------------------------------------------------- #
    style: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for p in sorted(canonical.glob("*.json")):
        if p.name.endswith(".invalid.json"):
            continue
        doc = json.loads(p.read_text())
        pid = doc["piece_id"]
        split = splits.get(pid)
        if split is None:
            continue
        ap = ann_dir / f"{pid}.json"
        anns = json.loads(ap.read_text())["annotations"] if ap.exists() else []
        style[split].append({
            "id": pid, "source_type": "real_bach",
            "work_group_id": doc.get("work_group_id"),
            "movement_type": doc["movement_type"],
            "voice_count": doc["voice_count"],
            "source_canonical_event_hash": doc["normalization"]["core_event_hash"],
            **tmeta, **_tokenize(doc, tcfg, anns),
        })

    # Approved excerpts only (§20.1 + §27). Pending excerpts are never included.
    ex_path = root / "data" / "proposed" / "excerpts.jsonl"
    approved = 0
    pending = 0
    if ex_path.exists():
        for line in ex_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("auto_rejected"):
                continue
            if r["review_status"] != "approved":
                pending += 1
                continue
            approved += 1
            doc = json.loads((canonical / f"{r['piece_id']}.json").read_text())
            start, end = parse_frac(r["start_onset_q"]), parse_frac(r["end_onset_q"])
            clipped = {**doc, "piece_id": r["excerpt_id"],
                       "quarter_length_total": r["end_onset_q"],
                       "voices": [{**v, "notes": [
                           n for n in v["notes"]
                           if start <= parse_frac(n["onset_q"]) < end]}
                           for v in doc["voices"]]}
            split = r.get("split") or splits.get(r["piece_id"]) or "train"
            style[split].append({
                "id": r["excerpt_id"], "source_type": "real_bach_excerpt_human_approved",
                "parent_piece_id": r["piece_id"], "reviewer": r.get("reviewer"),
                **tmeta, **_tokenize(clipped, tcfg, None)})

    for s in SPLITS:
        _write_jsonl_zst(out / "style" / f"{s}.jsonl.zst", style[s])
    summary["exports"]["style"] = {
        "rows": {s: len(style[s]) for s in SPLITS},
        "approved_excerpts_included": approved,
        "pending_excerpts_excluded": pending,
        "note": ("Real Bach movements plus human-approved excerpts only. "
                 "Pending excerpts are excluded: §27 forbids presenting "
                 "automatically selected material as human approved."),
    }

    # ---- §20.2 feasibility -------------------------------------------------- #
    feas: dict[str, list[dict]] = {"train": [], "validation": []}
    for rec in _iter_shards(root / "data" / "generated" / "valid"):
        if rec["split"] not in feas:
            continue
        feas[rec["split"]].append({
            "id": rec["sample_id"], "source_type": rec["source_type"],
            "rule_profile": rec["rule_profile"], "ruleset_hash": rec["ruleset_hash"],
            "hard_valid": rec["hard_valid"], "seed": rec["seed"],
            "parent_subject_id": rec["parent_subject_id"],
            "source_canonical_event_hash": rec["event_hash"],
            **tmeta, **_tokenize(rec["score"], tcfg, None)})
    for s in ("train", "validation"):
        for src in style[s]:
            feas[s].append({**src, "rule_profile": "real_bach_reference",
                            "hard_valid": True})
        _write_jsonl_zst(out / "feasibility" / f"valid_{s}.jsonl.zst", feas[s])
    summary["exports"]["feasibility"] = {
        "rows": {s: len(feas[s]) for s in feas},
        "note": "Hard-valid real and synthetic material with rule-profile metadata.",
    }

    # ---- §20.3 constraint critic -------------------------------------------- #
    positives = []
    for rec in _iter_shards(root / "data" / "generated" / "valid"):
        positives.append({
            "sample_id": rec["sample_id"], "label": "valid",
            "rule_profile": rec["rule_profile"], "ruleset_hash": rec["ruleset_hash"],
            "split": rec["split"], "parent_subject_id": rec["parent_subject_id"],
            "event_hash": rec["event_hash"],
            "soft_violation_total": sum((rec.get("soft_scores") or {}).values()),
            "violations_json": "[]"})
    negatives = []
    for rec in _iter_shards(root / "data" / "generated" / "invalid"):
        negatives.append({
            "sample_id": rec["sample_id"], "label": "invalid",
            "parent_valid_sample_id": rec["parent_valid_sample_id"],
            "mutation_id": rec["mutation_id"], "target_rule_id": rec["target_rule_id"],
            "rule_profile": rec["rule_profile"], "ruleset_hash": rec["ruleset_hash"],
            "split": rec["split"], "parent_subject_id": rec["parent_subject_id"],
            "pre_validation_hash": rec["pre_validation_hash"],
            "post_validation_hash": rec["post_validation_hash"],
            "notation_only": rec.get("notation_only", False),
            "violations_json": json.dumps(rec["violations"], sort_keys=True)})
    _write_parquet(out / "constraint_critic" / "positives.parquet", positives)
    _write_parquet(out / "constraint_critic" / "controlled_negatives.parquet", negatives)
    summary["exports"]["constraint_critic"] = {
        "positives": len(positives), "controlled_negatives": len(negatives),
        "note": "Matched parents: every negative names the valid sample it came from.",
    }

    # ---- §20.4 structural supervision --------------------------------------- #
    ann_rows = []
    for p in sorted(ann_dir.glob("*.json")):
        if p.name.endswith(".invalid.json"):
            continue
        d = json.loads(p.read_text())
        for a in d["annotations"]:
            ann_rows.append({
                "piece_id": a["piece_id"], "annotation_id": a["annotation_id"],
                "type": a["type"], "voice_id": a["voice_id"],
                "onset_q": a["onset_q"], "duration_q": a["duration_q"],
                "measure_number_source": a["measure_number_source"],
                "analytical_status": a["analytical_status"],
                "annotator": a["annotator"],
                "entry_function": a.get("entry_function"),
                "entry_function_derivation": a.get("entry_function_derivation"),
                "transformations": ",".join(a.get("transformations") or []),
                "cadence_key_area": a.get("cadence_key_area"),
                "cadence_type": a.get("cadence_type"),
                "analytical_alternative_group": a.get("analytical_alternative_group"),
                "split": splits.get(a["piece_id"])})
    _write_parquet(out / "structure" / "wtc1_annotations.parquet", ann_rows)

    section_rows = []
    for p in sorted(ann_dir.glob("*.json")):
        if p.name.endswith(".invalid.json"):
            continue
        d = json.loads(p.read_text())
        doc = json.loads((canonical / f"{d['piece_id']}.json").read_text())
        for a in d["annotations"]:
            if a["analytical_status"] != "accepted" or not a["duration_q"]:
                continue
            start = parse_frac(a["onset_q"])
            end = start + parse_frac(a["duration_q"])
            window = {**doc, "piece_id": a["annotation_id"],
                      "quarter_length_total": a["duration_q"],
                      "voices": [{**v, "notes": [
                          n for n in v["notes"]
                          if start <= parse_frac(n["onset_q"]) < end]}
                          for v in doc["voices"]]}
            section_rows.append({
                "id": a["annotation_id"], "piece_id": d["piece_id"],
                "type": a["type"], "voice_id": a["voice_id"],
                "onset_q": a["onset_q"], "duration_q": a["duration_q"],
                "split": splits.get(d["piece_id"]),
                **tmeta, **_tokenize(window, tcfg, None)})
    _write_jsonl_zst(out / "structure" / "section_windows.jsonl.zst", section_rows)
    summary["exports"]["structure"] = {
        "annotations": len(ann_rows), "section_windows": len(section_rows)}

    # ---- §20.5 preferences: reviewed pairs only ----------------------------- #
    pairs_path = root / "data" / "generated" / "preference_candidates" / "pairs.jsonl"
    reviewed = {"train": [], "validation": []}
    total_pairs = unreviewed = 0
    if pairs_path.exists():
        for line in pairs_path.read_text().splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            total_pairs += 1
            if p["review_status"] != "reviewed" or p["label"] is None:
                unreviewed += 1
                continue
            if p["split"] in reviewed:
                # `label_confidence` travels with the row. A pair the same
                # reviewer called two different ways under blind re-listening is
                # one a reward model should not be trained to discriminate
                # confidently, and the export is where that has to be visible --
                # a consumer reading only this file cannot otherwise tell.
                reviewed[p["split"]].append(
                    {**p, "label_confidence": p.get("label_confidence", "unrated")})
    for s in ("train", "validation"):
        path = out / "preferences" / f"{s}_pairs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(
            json.dumps(p, sort_keys=True, separators=(",", ":")) + "\n"
            for p in reviewed[s]))
    conf: dict[str, int] = {}
    for s in reviewed:
        for r in reviewed[s]:
            k = r.get("label_confidence", "unrated")
            conf[k] = conf.get(k, 0) + 1
    summary["exports"]["preferences"] = {
        "rows": {s: len(reviewed[s]) for s in reviewed},
        "label_confidence": dict(sorted(conf.items())),
        "candidate_pairs_available": total_pairs,
        "unreviewed_excluded": unreviewed,
        "note": ("Only reviewed pairs are exported (§18.6). Unreviewed candidates "
                 "are excluded, so an empty export means labelling has not happened "
                 "yet, not that no candidates exist."),
    }

    # ---- §20.6 metadata ------------------------------------------------------ #
    meta = out / "metadata"
    src_rows = [json.loads(l) for l in
                (root / "manifests" / "sources.jsonl").read_text().splitlines() if l.strip()]
    _write_parquet(meta / "sources.parquet",
                   [{k: v for k, v in r.items() if not k.startswith("_")} for r in src_rows])
    piece_rows = []
    for p in sorted(canonical.glob("*.json")):
        if p.name.endswith(".invalid.json"):
            continue
        d = json.loads(p.read_text())
        piece_rows.append({
            "piece_id": d["piece_id"], "work_group_id": d.get("work_group_id"),
            "composer_id": d["composer_id"], "movement_type": d["movement_type"],
            "voice_count": d["voice_count"], "measure_count": d.get("measure_count"),
            "quarter_length_total": d.get("quarter_length_total"),
            "event_hash": d["normalization"]["core_event_hash"],
            "split": splits.get(d["piece_id"])})
    _write_parquet(meta / "pieces.parquet", piece_rows)
    split_rows = [json.loads(l) for l in
                  (root / "manifests" / "splits.jsonl").read_text().splitlines() if l.strip()]
    _write_parquet(meta / "splits.parquet",
                   [{**r, "stratification_fields": json.dumps(r["stratification_fields"],
                                                              sort_keys=True)}
                    for r in split_rows])
    deriv = []
    for rec in _iter_shards(root / "data" / "generated" / "valid"):
        deriv.append({"id": rec["sample_id"], "kind": "synthetic_valid",
                      "parent": rec["parent_subject_id"], "split": rec["split"]})
    for rec in _iter_shards(root / "data" / "generated" / "invalid"):
        deriv.append({"id": rec["sample_id"], "kind": "synthetic_controlled_invalid",
                      "parent": rec["parent_valid_sample_id"], "split": rec["split"]})
    _write_parquet(meta / "derivatives.parquet", deriv)

    reg = yaml.safe_load((root / "config" / "rules" / "registry.yaml").read_text())
    profiles = {}
    for name in ("species_strict", "high_baroque_conservative", "bach_empirical"):
        profiles[name] = yaml.safe_load(
            (root / "config" / "rules" / f"{name}.yaml").read_text())
    (meta / "ruleset.json").write_text(json.dumps(
        {"registry": reg, "profiles": profiles}, indent=2, sort_keys=True) + "\n")
    (meta / "vocabulary.json").write_text(json.dumps(
        {**tmeta, "size": len(vocab), "tokens": vocab}, indent=2) + "\n")

    summary["exports"]["metadata"] = {
        "sources": len(src_rows), "pieces": len(piece_rows),
        "splits": len(split_rows), "derivatives": len(deriv)}
    summary["human_checkpoints"] = {
        "excerpts": {"approved": approved, "pending": pending,
                     "target": "50-100 approved (§14.6)"},
        "preferences": {"reviewed": sum(len(v) for v in reviewed.values()),
                        "candidates": total_pairs, "target": "500+ reviewed (§18.6)"},
    }
    (root / "manifests" / "export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
