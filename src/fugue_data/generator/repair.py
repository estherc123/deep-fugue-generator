"""Proofreading pass: fix the forbidden-rule breaks the sampler let through.

The masks in `sample.generate` stop nearly every forbidden progression as it
is written, but a few slip past: a free note chosen before a written-in note
it could not see, a mask relaxed because every note was banned. A composer
would correct the wrong note; this does the same.

For every forbidden hit (bach_rules.FORBIDDEN), the notes involved that the
generator chose (never a written-in entry, pedal or final-chord constraint)
are each tried at nearby pitches (within a sixth either way). A change is kept
if it lowers the weighted rule cost of the surrounding bars (forbidden rules
weigh 20, the others 1, rules_select.rule_weights); ties go to the smaller
change. Repeated until no forbidden hit is left or nothing helps.
"""

from __future__ import annotations

import numpy as np

from .bach_rules import FORBIDDEN, check, notes
from .encode import MIDI_LO, NOTE0
from .rules_select import rule_weights

PAIR_RULES = {"par5", "par8", "hidden_outer", "crossing_bass", "lt_double"}


def _cost(g, a, b, kw, W):
    hits = check(g[a:b], start=0, **{**kw, "keys": kw["keys"][a:b] if kw.get("keys") else None})
    return sum(W.get(r, 1.0) for r, _, _ in hits), sum(1 for r, _, _ in hits if r in FORBIDDEN)


def repair(grid: np.ndarray, fixed: np.ndarray, kw: dict, bar: int, start: int = 0,
           max_passes: int = 4, final_from: int | None = None, final_pcs=None, bass_pcs=None) -> tuple[np.ndarray, int]:
    """(repaired grid, notes changed). kw: check() keyword args (meter, tonic,
    mode, keys) for the whole grid."""
    g = grid.copy()
    W = rule_weights()
    W["final_cadence"] = 20.0
    changed = 0
    n, nv = g.shape
    for _ in range(max_passes):
        hits = [h for h in check(g, start=start, final=True, fixed=fixed, **kw) if h[0] in FORBIDDEN]
        if not hits:
            break
        improved = False
        ev = notes(g)
        for rule, s, v in hits:
            voices = [v]
            if rule in PAIR_RULES:
                voices += [w for w in range(nv) if w != v]
            cands = []
            for u in voices:
                k = next((j for j, e in enumerate(ev[u]) if e[0] <= s < e[1]), None)
                if k is None:
                    continue
                cands.append((u, ev[u][k]))
                if u == v and k > 0 and rule not in PAIR_RULES:
                    cands.append((u, ev[u][k - 1]))    # the free note leading into it
            for u, note in cands:
                if fixed[note[0], u]:
                    continue
                n0, n1, m = note
                a, b = max(0, n0 - bar), min(n, n1 + bar)
                base, base_f = _cost(g, a, b, kw, W)
                best = (base, 0, m)
                for dm in sorted(range(-9, 10), key=abs):
                    m2 = m + dm
                    if dm == 0 or not (36 <= m2 <= 86):
                        continue
                    if final_from is not None and n0 >= final_from:
                        ok = (bass_pcs if u == nv - 1 else final_pcs)
                        if m2 % 12 not in ok:
                            continue
                    g[n0, u] = NOTE0 + m2 - MIDI_LO
                    c, f = _cost(g, a, b, kw, W)
                    if c < best[0] - 1e-9:
                        best = (c, dm, m2)
                    g[n0, u] = NOTE0 + m - MIDI_LO
                if best[1] != 0:
                    g[n0, u] = NOTE0 + best[2] - MIDI_LO
                    changed += 1
                    improved = True
                    ev = notes(g)
                    break
        if not improved:
            break
    return g, changed
