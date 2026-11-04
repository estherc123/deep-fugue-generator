"""Rule cost of one bar, for bar-level selection while generating.

`bar_score(grid, a, b, meta)` runs the rule engine over steps [a, b) of a
generated grid, with one bar of context before it, and returns a weighted
count of violations that begin inside [a, b).

Weights follow the rule audit (`audit.py`, per 100 bars of held-out Bach):
rules Bach essentially never breaks weigh most; rules Bach bends regularly
(leap recovery, dissonance treatment by the engine's broad definition) weigh
1, so selection pulls their rates toward Bach's rather than to zero.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import numpy as np

from ..constraints.engine import evaluate, load_profile
from .encode import STEPS_PER_Q, tonic_pc
from .report import _fragment

WEIGHTS = {
    "parallel_fifths": 20.0, "parallel_octaves": 20.0,     # Bach 0-0.2 / 100 bars
    "direct_fifths_outer": 10.0,                           # 2.0
    "melodic_leap_limit": 10.0,                            # 2.7
    "suspension_resolution": 5.0,                          # 0.7
    "voice_crossing": 3.0,
    "voice_spacing": 1.5,                                  # 9.7
    "augmented_interval_melodic": 1.0,                     # 29
    "leap_recovery": 1.0,                                  # 109
    "dissonance_treatment": 1.0,                           # 290
    "voice_overlap": 0.5,                                  # 53
}
_PROFILE = None


def _profile():
    global _PROFILE
    if _PROFILE is None:
        prof = dict(load_profile("bach_empirical"))
        prof["severities"] = {k: ("diagnostic" if k in WEIGHTS else "off") for k in prof["rules"]}
        _PROFILE = prof
    return _PROFILE


def piece_meta(p: dict) -> dict:
    """The bits of a canonical score the rule engine needs, from an encoded piece."""
    if p["meter_tok"] == "METER_other":
        num, den = 4, 4
    else:
        num, den = (int(x) for x in p["meter_tok"].split("_")[1:3])
    fifths = next((x for x in range(-7, 8) if tonic_pc(x, p["mode"]) == p["tonic"]), 0)
    return {"piece_id": p["piece_id"],
            "canon": {"meter_map": [{"numerator": num, "denominator": den}],
                      "key_signature_map": [{"fifths": fifths}]}}


def violations(grid: np.ndarray, a: int, b: int, meta: dict) -> Counter:
    w0 = max(0, a - (b - a))
    frag = _fragment(grid[w0:b], meta["piece_id"], meta["canon"])
    c = Counter()
    for v in evaluate(frag, _profile()):
        if Fraction(v["onset_q"]) * STEPS_PER_Q + w0 >= a:
            c[v["rule_id"]] += 1
    return c


def bar_score(grid: np.ndarray, a: int, b: int, meta: dict) -> float:
    if b <= a:
        return 0.0
    return float(sum(WEIGHTS[r] * k for r, k in violations(grid, a, b, meta).items()))


# --------------------------------------------------------------------------- #
# The research-based checker (bach_rules.py): forbidden rules cost 20, rules
# with a Bach rate cost 1 (strong-beat dissonance 1.5), so selection drives the
# forbidden ones to ~0 and pulls the others toward Bach's rate.
# --------------------------------------------------------------------------- #
def rule_weights():
    from .bach_rules import FORBIDDEN, RULES
    w = {r: (20.0 if r in FORBIDDEN else 1.0) for r in RULES}
    w["strong_diss"] = 3.0
    w["final_cadence"] = 0.0          # judged on whole pieces, not bars
    return w


_W = None


def bar_score_v2(grid: np.ndarray, a: int, b: int, meta: dict) -> float:
    global _W
    from .bach_rules import check
    if _W is None:
        _W = rule_weights()
    if b <= a:
        return 0.0
    w0 = max(0, a - (b - a))
    keys = meta.get("keys")
    hits = check(grid[w0:b], meter=meta["meter"], tonic=meta["tonic"], mode=meta["mode"],
                 keys=keys[w0:b] if keys is not None else None, unit=meta.get("unit"),
                 start=a - w0)
    # a dissonance at the bar's last beat may resolve in the next bar: do not
    # charge strong_diss on the final beat of the window
    last_beat = (b - w0) - meta["beat"]
    return float(sum(_W[r] for r, s, _ in hits if not (r == "strong_diss" and s >= last_beat)))


def piece_meta_v2(p: dict) -> dict:
    from .encode import _beat_q
    m = piece_meta(p)
    num = m["canon"]["meter_map"][0]["numerator"]
    den = m["canon"]["meter_map"][0]["denominator"]
    keys = None
    if "key" in p:
        keys = [((p["tonic"] + (int(k) - 1) // 2) % 12, bool((int(k) - 1) % 2)) if k > 0
                else (p["tonic"], p["mode"] == "minor") for k in p["key"]]
    return {**m, "meter": (num, den), "tonic": p["tonic"], "mode": p["mode"], "keys": keys,
            "unit": None, "beat": int(_beat_q((num, den)) * STEPS_PER_Q)}
