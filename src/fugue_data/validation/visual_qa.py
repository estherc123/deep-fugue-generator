"""Visual QA rendering and recording (Spec §21.1).

§21.1 names specific quotas -- at least 20 WTC fugues or excerpts, 20 BCFB
chorales, every anomalous normalization, 50 synthetic valid fragments across
coverage categories, and 50 controlled negatives with violation highlighting --
and insists the results be *recorded*, not merely performed informally. So this
renders to files, writes a manifest naming every artefact, and reports whether
each quota was met.

The renders are piano rolls with the violated notes highlighted; nothing here
claims a human has looked at them. The manifest records what was produced and
leaves an `inspected` field for a person to fill in.
"""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

from ..normalize.rational import parse_frac
from ..review.render import piano_roll_svg

QUOTAS = {
    "wtc_fugues_or_excerpts": 20,
    "bcfb_chorales": 20,
    "synthetic_valid": 50,
    "controlled_negatives": 50,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _highlight_svg(doc: dict, violations: list[dict], title: str) -> str:
    """Piano roll with the notes named by a violation ringed in red."""
    end = parse_frac(doc.get("quarter_length_total", "0"))
    svg = piano_roll_svg(doc, Fraction(0), end, None, title)
    ids = {nid for v in violations for nid in (v.get("note_ids") or [])}
    if not ids:
        return svg
    marks = []
    lo = min((n["midi"] for v in doc["voices"] for n in v["notes"]), default=60) - 2
    hi = max((n["midi"] for v in doc["voices"] for n in v["notes"]), default=72) + 2
    W, H = 900, max(160, (hi - lo) * 9)
    pad_l, pad_t = 44, 26
    span = float(end) or 1.0
    xw = (W - pad_l - 12) / span
    yh = (H - pad_t - 24) / max(1, hi - lo)
    for v in doc["voices"]:
        for n in v["notes"]:
            if n["note_id"] not in ids:
                continue
            x = pad_l + float(parse_frac(n["onset_q"])) * xw
            y = pad_t + (hi - n["midi"]) * yh
            w = max(3.0, float(parse_frac(n["duration_q"])) * xw)
            marks.append(f"<rect x='{x-1.5:.1f}' y='{y-1.5:.1f}' width='{w+3:.1f}' "
                         f"height='{max(3,yh-1)+3:.1f}' fill='none' stroke='#dc2626' "
                         f"stroke-width='2' rx='2'/>")
    return svg.replace("</svg>", "".join(marks) + "</svg>")


def _shards(d: Path, limit: int, seed: int):
    rows = []
    for shard in sorted(d.glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit]


def run_visual_qa(root: Path | None = None, *, seed: int = 20260728) -> dict:
    root = root or _repo_root()
    out = root / "reports" / "visual_qa"
    out.mkdir(parents=True, exist_ok=True)
    canonical = root / "data" / "canonical"
    ann_dir = root / "data" / "annotations"
    rng = random.Random(seed)
    artefacts: list[dict] = []

    def write(name: str, svg: str, kind: str, subject: str, note: str = "",
              violations: list | None = None):
        p = out / f"{name}.svg"
        p.write_text(svg)
        artefacts.append({
            "file": f"visual_qa/{name}.svg", "kind": kind, "subject": subject,
            "note": note, "violations": violations or [],
            # Left for a person; the pipeline never fills this in.
            "inspected": False, "inspector": None, "inspection_note": None})

    # ---- 20 WTC fugues (first 4 bars, annotations overlaid) ----------------- #
    fugues = sorted(canonical.glob("bach_wtc*_fugue_*.json"))
    rng.shuffle(fugues)
    for p in fugues[:QUOTAS["wtc_fugues_or_excerpts"]]:
        d = json.loads(p.read_text())
        ap = ann_dir / p.name
        anns = json.loads(ap.read_text())["annotations"] if ap.exists() else None
        mm = (d.get("meter_map") or [{"numerator": 4, "denominator": 4}])[0]
        bar = Fraction(mm["numerator"] * 4, mm["denominator"])
        write(f"fugue_{d['piece_id']}",
              piano_roll_svg(d, Fraction(0), bar * 4, anns, f"{d['piece_id']} mm.1-4"),
              "wtc_fugues_or_excerpts", d["piece_id"], "opening four bars")

    # ---- 20 BCFB chorales ---------------------------------------------------- #
    chorales = sorted(canonical.glob("bach_chorale_*.json"))
    rng.shuffle(chorales)
    for p in chorales[:QUOTAS["bcfb_chorales"]]:
        d = json.loads(p.read_text())
        mm = (d.get("meter_map") or [{"numerator": 4, "denominator": 4}])[0]
        bar = Fraction(mm["numerator"] * 4, mm["denominator"])
        write(f"chorale_{d['piece_id']}",
              piano_roll_svg(d, Fraction(0), bar * 4, None, f"{d['piece_id']} mm.1-4"),
              "bcfb_chorales", d["piece_id"], "opening four bars")

    # ---- every anomalous normalization --------------------------------------- #
    anomalies = 0
    for p in sorted(canonical.glob("*.json")):
        if p.name.endswith(".invalid.json"):
            continue
        d = json.loads(p.read_text())
        warns = d.get("warnings") or []
        if not warns:
            continue
        anomalies += 1
        onsets = sorted({parse_frac(w["onset_q"]) for w in warns
                         if w.get("onset_q")} or {Fraction(0)})
        start = max(Fraction(0), onsets[0] - 4)
        write(f"anomaly_{d['piece_id']}",
              piano_roll_svg(d, start, start + 12, None,
                             f"{d['piece_id']} — {warns[0]['code']}"),
              "anomalous_normalization", d["piece_id"],
              f"{len(warns)} parser warnings; first: {warns[0]['code']}")

    # ---- 50 synthetic valid, spread across coverage categories --------------- #
    valid_rows = _shards(root / "data" / "generated" / "valid", 800, seed)
    # Spread across coverage categories first, then top up. The narrow key
    # (voices | mode | meter) has only 18 combinations, so on its own it can never
    # fill a quota of 50 -- distinctness is the goal, not a cap.
    picked: dict[str, dict] = {}
    for key_fields in (("voice_count", "mode", "meter", "length_bars", "register"),
                       ("voice_count", "mode", "meter")):
        for rec in valid_rows:
            if len(picked) >= QUOTAS["synthetic_valid"]:
                break
            cov = rec.get("coverage") or {}
            key = "|".join(str(cov.get(f)) for f in key_fields)
            if key not in picked and rec["sample_id"] not in {
                    r["sample_id"] for r in picked.values()}:
                picked[key] = rec
    if len(picked) < QUOTAS["synthetic_valid"]:
        chosen_ids = {r["sample_id"] for r in picked.values()}
        for rec in valid_rows:
            if len(picked) >= QUOTAS["synthetic_valid"]:
                break
            if rec["sample_id"] not in chosen_ids:
                picked[rec["sample_id"]] = rec
                chosen_ids.add(rec["sample_id"])
    for rec in list(picked.values())[:QUOTAS["synthetic_valid"]]:
        cov = rec.get("coverage") or {}
        write(f"valid_{rec['sample_id'].replace(':', '_')}",
              piano_roll_svg(rec["score"], Fraction(0),
                             parse_frac(rec["score"]["quarter_length_total"]), None,
                             f"{rec['sample_id']} — {cov.get('voice_count')}v "
                             f"{cov.get('mode')} {cov.get('meter')}"),
              "synthetic_valid", rec["sample_id"],
              f"coverage {cov.get('voice_count')}v/{cov.get('mode')}/{cov.get('meter')}")

    # ---- 50 controlled negatives with violation highlighting ----------------- #
    neg_rows = _shards(root / "data" / "generated" / "invalid", 600, seed)
    by_rule: dict[str, list] = {}
    for rec in neg_rows:
        by_rule.setdefault(rec["target_rule_id"], []).append(rec)
    chosen: list[dict] = []
    while len(chosen) < QUOTAS["controlled_negatives"] and any(by_rule.values()):
        for rid in sorted(by_rule):
            if by_rule[rid] and len(chosen) < QUOTAS["controlled_negatives"]:
                chosen.append(by_rule[rid].pop())
    for rec in chosen:
        write(f"invalid_{rec['sample_id'].replace(':', '_')}",
              _highlight_svg(rec["score"], rec["violations"],
                             f"{rec['sample_id']} — {rec['target_rule_id']}"),
              "controlled_negatives", rec["sample_id"],
              f"mutation {rec['mutation_id']}; violated {rec['target_rule_id']}",
              rec["violations"])

    counts: dict[str, int] = {}
    for a in artefacts:
        counts[a["kind"]] = counts.get(a["kind"], 0) + 1
    shortfalls = [f"{k}: {counts.get(k, 0)} rendered, quota {q}"
                  for k, q in QUOTAS.items() if counts.get(k, 0) < q]

    summary = {
        "stage": "visual_qa",
        "seed": seed,
        "quotas": QUOTAS,
        "rendered_by_kind": dict(sorted(counts.items())),
        "anomalous_normalizations_rendered": anomalies,
        "total_rendered": len(artefacts),
        "artefacts": artefacts,
        "inspected_count": sum(1 for a in artefacts if a["inspected"]),
        "note": ("Renders are produced and recorded here. `inspected` is left false "
                 "for every artefact: the pipeline does not claim a human looked at "
                 "them (§1, §27). Fill it in as they are reviewed."),
        "failures": shortfalls,
    }
    (root / "manifests" / "visual_qa.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
