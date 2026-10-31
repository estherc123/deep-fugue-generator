"""Empirical rule validation against real Bach (Spec §15.6).

The premise: real Bach is, by definition, stylistically correct. So any rule that
is set to `hard` and still fires on the corpus is producing a false positive --
either the implementation is wrong, the rule is missing context, or the rule
simply is not a hard law of this style. Measuring that is what stops a rule from
becoming hard merely because a textbook phrases it categorically.

Two rates are reported, because one of them alone is misleading.

`pieces_firing_rate` is the share of real pieces on which a rule fires at least
once. It is the right test for a rule that should never fire at all -- a parallel
fifth, a note outside the range -- where one occurrence is one too many.

`detections_per_1000_notes` is the right test for event-level rules. A
1,500-note fugue containing a single unrecovered leap scores identically to one
containing two hundred under the per-piece rate, so judging melodic rules that
way demotes rules Bach observes almost all of the time. That error was made here
and then caught by comparing generated output against Bach: `leap_recovery`
showed a per-piece rate of 1.00, implying Bach ignores it, while the per-note
rate is 65.6 -- roughly one unrecovered leap every fifteen notes, which is a real
constraint the generator was measurably violating more often than Bach did.

A rule may only be `hard` in a profile if its `pieces_firing` count is 0 there.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .engine import PROFILES, evaluate, load_profile, ruleset_hash
from .fragment import Fragment


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _held_out_pieces(root: Path) -> list[str]:
    sp = root / "manifests" / "splits.jsonl"
    if not sp.exists():
        return []
    out = []
    for line in sp.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["split"] in ("validation", "test"):
                out.append(r["piece_id"])
    return sorted(out)


def validate_rules_empirically(root: Path | None = None, *, held_out_only: bool = True,
                               limit: int | None = None) -> dict:
    root = root or _repo_root()
    canonical = root / "data" / "canonical"
    wanted = set(_held_out_pieces(root)) if held_out_only else None

    files = sorted(p for p in canonical.glob("*.json")
                   if not p.name.endswith(".invalid.json"))
    if wanted:
        files = [p for p in files if json.loads(p.read_text())["piece_id"] in wanted]
    if limit:
        files = files[:limit]

    report: dict = {"stage": "rule_empirical_validation",
                    "pieces_evaluated": len(files),
                    "held_out_only": held_out_only,
                    "profiles": {}}

    for pname in PROFILES:
        profile = load_profile(pname)
        fires: dict[str, int] = {}
        detections: dict[str, int] = {}
        samples: dict[str, list] = {}
        hard_fires: dict[str, int] = {}
        total_notes = 0
        for path in files:
            score = json.loads(path.read_text())
            total_notes += sum(len(v["notes"]) for v in score["voices"])
            frag = Fragment.from_score(score)
            vs = evaluate(frag, profile)
            seen = set()
            for v in vs:
                rid = v["rule_id"]
                detections[rid] = detections.get(rid, 0) + 1
                if rid not in seen:
                    seen.add(rid)
                    fires[rid] = fires.get(rid, 0) + 1
                    if v["severity"] == "hard":
                        hard_fires[rid] = hard_fires.get(rid, 0) + 1
                if len(samples.get(rid, [])) < 3:
                    samples.setdefault(rid, []).append(
                        {"piece_id": v["piece_id"], "onset_q": v["onset_q"],
                         "message": v["message"]})

        n = max(1, len(files))
        rules = {}
        for rid in sorted(profile["rules"]):
            sev = profile["severities"].get(rid, "off")
            rules[rid] = {
                "severity": sev,
                "category": profile["rules"][rid]["category"],
                "pieces_firing": fires.get(rid, 0),
                "total_detections": detections.get(rid, 0),
                # Share of pieces firing at least once: the right test for a rule
                # that should never fire at all.
                "pieces_firing_rate": round(fires.get(rid, 0) / n, 4),
                "false_positive_rate": round(fires.get(rid, 0) / n, 4),
                # Event density: the right test for a rule about individual
                # melodic or vertical events, and the number to compare generated
                # output against.
                "detections_per_1000_notes": round(
                    1000 * detections.get(rid, 0) / max(1, total_notes), 2),
                "samples": samples.get(rid, []),
            }
        bad = sorted(rid for rid, r in rules.items()
                     if r["severity"] == "hard" and r["pieces_firing"] > 0)
        report["profiles"][pname] = {
            "ruleset_hash": ruleset_hash(profile),
            "notes_evaluated": total_notes,
            "rules": rules,
            "hard_rules_firing_on_real_bach": bad,
            "calibrated": not bad,
        }

    report["failures"] = [
        f"{p}: hard rules fire on real Bach: {v['hard_rules_firing_on_real_bach']}"
        for p, v in report["profiles"].items()
        if v["hard_rules_firing_on_real_bach"] and p == "bach_empirical"
    ]
    (root / "manifests" / "rule_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
