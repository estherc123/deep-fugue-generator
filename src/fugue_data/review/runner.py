"""Propose excerpt candidates and render them for review (Spec §14).

Restartable and non-destructive: an existing review record keeps its human
decision. Re-running refreshes proposals and renders but never overwrites
`review_status`, `review_tags`, `review_comment`, `reviewer` or `reviewed_at_utc`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import jsonschema
import yaml

from ..normalize.rational import frac_str, parse_frac
from .propose import (clipped_boundary_notes, excerpt_id, measure_table,
                      notes_in, propose_windows, rejection_reasons,
                      snap_window, sounding_voices)
from .render import piano_roll_svg, write_midi


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


_HUMAN_FIELDS = ("review_status", "review_tags", "review_comment", "reviewer",
                 "reviewed_at_utc")


def propose_excerpts(root: Path | None = None, *, count: int | None = None,
                     render: bool = True) -> dict:
    root = root or _repo_root()
    cfg = yaml.safe_load((root / "config" / "excerpts.yaml").read_text())
    pipeline = yaml.safe_load((root / "config" / "pipeline.yaml").read_text())
    canonical_dir = root / pipeline["paths"]["canonical"]
    ann_dir = root / pipeline["paths"]["annotations"]
    out_dir = root / "data" / "proposed"
    render_dir = out_dir / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads((root / "schemas" / "review.schema.json").read_text())
    target = count or cfg["proposal"]["target_count"]

    splits = {}
    sp = root / "manifests" / "splits.jsonl"
    if sp.exists():
        for line in sp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                splits[r["piece_id"]] = r["split"]

    existing: dict[str, dict] = {}
    store = out_dir / "excerpts.jsonl"
    if store.exists():
        for line in store.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["excerpt_id"]] = r

    # Annotated fugues first: their triggers are the musically richest.
    pieces = sorted(p for p in canonical_dir.glob("*.json")
                    if not p.name.endswith(".invalid.json"))
    annotated = [p for p in pieces if (ann_dir / p.name).exists()]
    rest = [p for p in pieces if not (ann_dir / p.name).exists()]
    ordered = annotated + rest

    records: list[dict] = []
    seen_ids: dict[str, dict] = {}
    rejected_count = 0
    reason_census: dict[str, int] = {}
    rejection_census: dict[str, int] = {}

    for path in ordered:
        if len([r for r in records if not r["auto_rejected"]]) >= target:
            break
        score = json.loads(path.read_text())
        pid = score["piece_id"]
        ann_path = ann_dir / f"{pid}.json"
        annotations = (json.loads(ann_path.read_text())["annotations"]
                       if ann_path.exists() else [])
        bars = measure_table(score)
        bar_start = {frac_str(st): (i, num) for i, num, st in bars}

        for w in propose_windows(score, annotations, cfg):
            start, end = parse_frac(w["start_onset_q"]), parse_frac(w["end_onset_q"])
            # Preserve context rather than discard the candidate (§14.3).
            snapped = snap_window(score, annotations, start, end, bars, cfg)
            if snapped:
                start, end = snapped
            reasons = rejection_reasons(score, annotations, start, end, cfg)
            eid = excerpt_id(pid, start, end)
            n_notes = len(notes_in(score, start, end))
            si = bar_start.get(frac_str(start))
            ei = None
            for i, num, st in bars:
                # `end` is exclusive: a bar beginning exactly at `end` lies
                # outside the window. Accepting it reported one bar too many for
                # every window not reaching the end of the piece -- 183 of 200.
                if st < end:
                    ei = (i, num)
            n_bars = (ei[0] - si[0] + 1) if (si and ei) else None

            rec = {
                "excerpt_id": eid,
                "piece_id": pid,
                "work_group_id": score.get("work_group_id"),
                "split": splits.get(pid),
                "start_onset_q": frac_str(start),
                "end_onset_q": frac_str(end),
                "start_measure": si[1] if si else None,
                "end_measure": ei[1] if ei else None,
                "bars": n_bars,
                "voice_count": len(sounding_voices(score, start, end)),
                "note_count": n_notes,
                "clipped_boundary_notes": clipped_boundary_notes(score, start, end),
                "proposal_reasons": w["proposal_reasons"],
                "render_path": None,
                "midi_preview_path": None,
                # §14.6 / §27: proposal never implies approval.
                "review_status": "pending",
                "review_tags": [],
                "review_comment": None,
                "reviewer": None,
                "reviewed_at_utc": None,
                "auto_rejected": bool(reasons),
                "auto_rejection_reasons": reasons,
                "source_canonical_event_hash": score["normalization"]["core_event_hash"],
                "excerpts_version": cfg["excerpts_version"],
            }
            if reasons:
                rejected_count += 1
                for r in reasons:
                    rejection_census[r] = rejection_census.get(r, 0) + 1
            else:
                for r in w["proposal_reasons"]:
                    reason_census[r] = reason_census.get(r, 0) + 1

            # A human decision already recorded is never overwritten.
            prior = existing.get(eid)
            if prior:
                for f in _HUMAN_FIELDS:
                    rec[f] = prior.get(f, rec[f])

            if render and not reasons:
                rel_svg = f"renders/{eid.replace(':', '_')}.svg"
                rel_mid = f"renders/{eid.replace(':', '_')}.mid"
                render_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / rel_svg).write_text(piano_roll_svg(
                    score, start, end, annotations,
                    f"{pid} mm.{rec['start_measure']}-{rec['end_measure']}"))
                if write_midi(score, start, end, out_dir / rel_mid):
                    rec["midi_preview_path"] = rel_mid
                rec["render_path"] = rel_svg

            jsonschema.Draft7Validator(schema).validate(rec)
            # Identity is assigned AFTER snapping, but `propose_windows` dedupes
            # BEFORE it: two distinct proposals can snap to the same window and so
            # to the same excerpt_id. Appending both wrote 45 duplicate records
            # over 41 ids -- a reviewer would have been shown the same excerpt
            # twice, and each copy carried only the reasons of the proposal that
            # produced it, so "why was this window chosen" depended on which copy
            # you happened to read. The window is one window: merge the reasons.
            prior_rec = seen_ids.get(eid)
            if prior_rec is not None:
                prior_rec["proposal_reasons"] = sorted(
                    set(prior_rec["proposal_reasons"]) | set(rec["proposal_reasons"]))
                prior_rec["auto_rejection_reasons"] = sorted(
                    set(prior_rec["auto_rejection_reasons"])
                    | set(rec["auto_rejection_reasons"]))
                prior_rec["auto_rejected"] = bool(prior_rec["auto_rejection_reasons"])
                continue
            seen_ids[eid] = rec
            records.append(rec)
            if len([r for r in records if not r["auto_rejected"]]) >= target:
                break

    records.sort(key=lambda r: r["excerpt_id"])
    store.write_text("".join(
        json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records))

    proposed = [r for r in records if not r["auto_rejected"]]
    by_status: dict[str, int] = {}
    for r in proposed:
        by_status[r["review_status"]] = by_status.get(r["review_status"], 0) + 1

    summary = {
        "stage": "propose_excerpts",
        "excerpts_version": cfg["excerpts_version"],
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidates_proposed": len(proposed),
        "auto_rejected": rejected_count,
        "windows_examined": len(records),
        "target_count": target,
        "trigger_census": dict(sorted(reason_census.items(), key=lambda kv: -kv[1])),
        "rejection_census": dict(sorted(rejection_census.items(), key=lambda kv: -kv[1])),
        "review_status_census": dict(sorted(by_status.items())),
        "pieces_covered": len({r["piece_id"] for r in proposed}),
        "rendered": sum(1 for r in proposed if r["render_path"]),
        "midi_previews": sum(1 for r in proposed if r["midi_preview_path"]),
        "failures": ([f"only {len(proposed)} candidates, need at least 200 (§14.6)"]
                     if len(proposed) < 200 else []),
    }
    (root / "manifests" / "excerpt_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
