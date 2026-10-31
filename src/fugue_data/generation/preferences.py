"""Preference-pair candidate generation (Spec §18).

Candidates in a pair share every generation condition except the stochastic seed
(§18.2), which is what makes the comparison about musical quality rather than
about being asked different questions. `shared_conditions_hash` is computed from
those conditions and checked by the validation gate.

Pairing is not random: §18.3 asks for pairs where the diagnostics disagree, where
scores are close but character differs, or where uncertainty is high. Those are
the comparisons a reviewer's time is worth spending on.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from fractions import Fraction

from ..normalize.rational import parse_frac
from ..tokenization.views import merge_tied

# The conditions a pair must hold identical (§18.2). Seed is deliberately absent.
_SHARED_FIELDS = ("rule_profile", "voice_count", "meter", "key", "length_bars",
                  "subject", "entry_plan", "cadence_target",
                  "allowed_rhythmic_values", "range_profile", "texture_profile")


def prompt_conditions(req: dict) -> dict:
    return {k: req[k] for k in _SHARED_FIELDS if k in req}


def shared_conditions_hash(req: dict) -> str:
    payload = json.dumps(prompt_conditions(req), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def prompt_id(req: dict) -> str:
    return f"prompt:{shared_conditions_hash(req)[:16]}"


# --------------------------------------------------------------------------- #
# Diagnostics (§18.3 step 3)
# --------------------------------------------------------------------------- #
def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    out = 0.0
    for c in counts:
        if c:
            p = c / total
            out -= p * math.log(p, 2)
    return out


def diagnostics(rec: dict) -> dict:
    """Separate diagnostic scores, deliberately not combined into one number.

    §18.3 asks for a vector so that pairs can be chosen where the measures
    disagree; collapsing them to a single score would hide exactly the
    disagreements worth showing a reviewer.
    """
    doc = rec["score"]
    total = parse_frac(doc["quarter_length_total"])
    voices = doc["voices"]
    all_notes = [n for v in voices for n in v["notes"]]
    n = max(1, len(all_notes))

    durations: dict[str, int] = {}
    intervals: list[int] = []
    repeats = 0
    participation = []
    for v in voices:
        merged = merge_tied(v["notes"])
        sounding = sum(d for _, d, _ in merged)
        participation.append(float(sounding / total) if total else 0.0)
        for _, d, _ in merged:
            durations[str(d)] = durations.get(str(d), 0) + 1
        seq = [m for _, _, m in merged]
        for a, b in zip(seq, seq[1:]):
            intervals.append(b - a)
            if a == b:
                repeats += 1

    soft = rec.get("soft_scores", {}) or {}
    iv_counts: dict[int, int] = {}
    for i in intervals:
        iv_counts[i] = iv_counts.get(i, 0) + 1
    stepwise = sum(c for i, c in iv_counts.items() if 0 < abs(i) <= 2)
    leaps = sum(c for i, c in iv_counts.items() if abs(i) > 4)

    return {
        # style-prior proxy: how far the interval profile sits from stepwise-dominant
        # writing, which is what Baroque lines mostly are.
        "style_prior_proxy": round(stepwise / max(1, len(intervals)), 4),
        "structural_completion": 1.0 if rec.get("hard_valid") else 0.0,
        "voice_participation_min": round(min(participation) if participation else 0.0, 4),
        "voice_participation_mean": round(
            sum(participation) / len(participation) if participation else 0.0, 4),
        "rhythmic_diversity": round(_entropy(list(durations.values())), 4),
        "melodic_diversity": round(_entropy(list(iv_counts.values())), 4),
        "repetition": round(repeats / max(1, len(intervals)), 4),
        "leap_share": round(leaps / max(1, len(intervals)), 4),
        "note_density": round(n / float(total) if total else 0.0, 4),
        "soft_violation_total": sum(soft.values()),
        "soft_rule_vector": soft,
    }


_COMPARE_KEYS = ("style_prior_proxy", "rhythmic_diversity", "melodic_diversity",
                 "repetition", "voice_participation_mean", "note_density")


def _disagreement(a: dict, b: dict) -> tuple[float, str]:
    """How informative comparing these two would be, and why."""
    diffs = {k: a[k] - b[k] for k in _COMPARE_KEYS}
    signs = {k: (1 if v > 1e-6 else -1 if v < -1e-6 else 0) for k, v in diffs.items()}
    pos = sum(1 for s in signs.values() if s > 0)
    neg = sum(1 for s in signs.values() if s < 0)
    spread = sum(abs(v) for v in diffs.values())

    # Evaluators disagree: some measures favour A, others favour B.
    if pos and neg:
        return 1.0 + min(pos, neg) / len(_COMPARE_KEYS), "evaluators_disagree"
    # Close overall but different character.
    if spread < 0.35 and abs(a["soft_violation_total"] - b["soft_violation_total"]) <= 1:
        return 0.8, "close_scores_different_character"
    # One side has many more soft-rule hits: a suspected reward-model weakness.
    if abs(a["soft_violation_total"] - b["soft_violation_total"]) >= 4:
        return 0.6, "soft_rule_gap"
    return 0.2 + min(spread, 1.0) * 0.2, "uncertain"


def build_pairs(records: list[dict], cfg: dict, *, seed: int | None = None) -> list[dict]:
    """Group hard-valid fragments by prompt and pair the informative comparisons."""
    pcfg = cfg["preferences"]
    seed = pcfg["seed"] if seed is None else seed
    max_per_prompt = int(pcfg.get("max_pairs_per_prompt", 5))
    rng = random.Random(seed)

    by_prompt: dict[str, list[dict]] = {}
    for rec in records:
        req = rec["request"]
        by_prompt.setdefault(prompt_id(req), []).append(rec)

    pairs: list[dict] = []
    for pid, group in sorted(by_prompt.items()):
        if len(group) < 2:
            continue
        # Reject duplicates within a prompt (§18.3 step 2).
        seen: set[str] = set()
        uniq = []
        for rec in sorted(group, key=lambda r: r["sample_id"]):
            h = rec["event_hash"]
            if h in seen:
                continue
            seen.add(h)
            uniq.append(rec)
        uniq = uniq[:pcfg["candidates_per_prompt"]]
        if len(uniq) < 2:
            continue

        diags = {r["sample_id"]: diagnostics(r) for r in uniq}
        scored = []
        for i, a in enumerate(uniq):
            for b in uniq[i + 1:]:
                da, db = diags[a["sample_id"]], diags[b["sample_id"]]
                score, reason = _disagreement(da, db)
                scored.append((score, reason, a, b, da, db))
        scored.sort(key=lambda x: (-x[0], x[2]["sample_id"], x[3]["sample_id"]))

        # Take only the most informative comparisons from each prompt. Emitting
        # all 15 combinations of 6 candidates would fill the budget from a handful
        # of prompts and spend reviewer effort re-judging the same subject, when
        # §18.3 asks for the pairs where the diagnostics actually disagree.
        taken = 0
        for score, reason, a, b, da, db in scored:
            if taken >= max_per_prompt:
                break
            # A pair is only meaningful if it spans one split (§13.1).
            if a["split"] != b["split"] or a["split"] == "test":
                continue
            taken += 1
            req = a["request"]
            # Content-derived, not positional. A positional id is reused by the
            # next generation run for entirely different music, so a label
            # produced against an earlier run would ingest cleanly and silently
            # attach to the wrong fragments. Deriving the id from the two
            # candidates' event hashes makes a stale label fail to match instead.
            ident = hashlib.sha256(
                "|".join(sorted([a["event_hash"], b["event_hash"]])).encode()
            ).hexdigest()[:16]
            pairs.append({
                "pair_id": f"pair:{ident}",
                "prompt_id": pid,
                "candidate_a_id": a["sample_id"],
                "candidate_b_id": b["sample_id"],
                "shared_conditions_hash": shared_conditions_hash(req),
                "diagnostics_a": da,
                "diagnostics_b": db,
                "label": None,
                "review_status": "pending",
                "reviewer": None,
                "review_comment": None,
                "reviewed_at_utc": None,
                "split": a["split"],
                "selection_reason": reason,
                "is_consistency_repeat": False,
                # §18.5: left/right presentation is randomized per pair.
                "presentation_seed": rng.randrange(2 ** 31),
            })
    return pairs
