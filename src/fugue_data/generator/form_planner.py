"""Form planner: decides the large events of a fugue from its subject alone.

    PYTHONPATH=src python -m fugue_data.generator.form_planner --subject-from bach_wtc1_fugue_02

The decisions and their probabilities come from research/bach_fugue_techniques.md
(§3, "Decision rules for the form planner"), corrected by what the project's
own labels measure on 108 training fugues (labels.py): episodes median 4 bars
(10-90%: 1-14), 3.3 voices on average in episodes, stretto in 43% of fugues,
inversion 20%, augmentation 6%, pedals at a median 75% of the piece.

Plan, in order:
1. **Exposition**: every voice enters once, subject / answer alternating; the
   answer is *tonal* (scale degree 5 in the subject's head is answered by 1,
   not 2: research/bach_fugue_rules.md FG-2); last entry in an outer voice;
   sometimes a one-bar codetta before the third entry.
2. **Middle section**: 2-4 entry groups in closely related keys (minor: III
   first p 0.6, then v, iv, VI, VII; major: vi p 0.45, V 0.35, iii/ii 0.2), no
   key twice, the group before the return in IV/iv with p 0.45. A group is one
   entry, or a subject-answer pair in its key (p 0.4). Voices rotate: never the
   voice of the previous entry; the first middle entry in an outer voice p 0.75.
   Inversion of the first middle entry p 0.2. A stretto group (two entries
   overlapping at an interval and delay that pass the rule checker) at
   0.5-0.75 when the subject admits one, p 0.6.
3. **Episodes** between groups: 2-6 bars; the voice that enters next rests
   (p 0.7); a sequence (p 0.6): descending fifths (0.5), rising 5-6 (0.25),
   falling 6-3 steps (0.25), restated by the top and bottom active voices;
   modulating to the next group's key with a cadence (PAC 0.65, IAC 0.35)
   that the next entry overlaps.
4. **Final section** at ~0.75-0.85: dominant pedal before the last entry (p 0.3);
   the final entry in the tonic, in the bass (0.45) or soprano (0.45) or an inner
   voice, after that voice rests a bar; then a coda of 2-3 bars over a tonic
   pedal (p 0.55) ending on a perfect cadence, with a Picardy third in minor
   (p 0.85).

The output is a piece dict for `sample.generate` (entries written in, key,
chord, bass, texture, cadence, pedal and sequence plans), plus the plan itself
for inspection (`p["form"]`).
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import harmony
from .encode import (CAD_KEYS, STEPS_PER_Q, _beat_q, _bucket, encode_piece, key_lookahead,
                     tonic_pc)
from .labels import QUAL_NAMES, cad_key_name, harmonic_unit
from .plan import MAJOR, MINOR, _from_degree, _to_degree, realise

ROOT = Path(__file__).resolve().parents[3]
CENTRES = {2: [70, 52], 3: [71, 62, 51], 4: [72, 65, 58, 50], 5: [74, 67, 61, 55, 48]}


# --------------------------------------------------------------------------- #
# Subject transformations
# --------------------------------------------------------------------------- #
def _scale(mode):
    return MAJOR if mode == "major" else MINOR


def tonal_answer(subject, home, mode, near):
    """Answer at the fifth, with degree 5 in the head answered by degree 1."""
    dom_mode = "major" if mode == "major" else "minor"
    real = realise(subject, home, mode, 7, dom_mode, near)
    sc = _scale(mode)
    out = list(real)
    for k, (_, _, m) in enumerate(subject[:4]):
        d, alt = _to_degree(m, home, sc)
        if d % 7 == 4 and alt == 0:
            out[k] = real[k] - 2                 # 5 -> 1 instead of 5 -> 2
        elif d % 7 not in (0, 4):
            break                                # the head ends at the first other degree
    return out


def tonal_head(subject, home, mode) -> int:
    """How many leading notes the tonal answer changes (0: a real answer)."""
    sc = _scale(mode)
    k_last = 0
    for k, (_, _, m) in enumerate(subject[:4]):
        d, alt = _to_degree(m, home, sc)
        if d % 7 == 4 and alt == 0:
            k_last = k + 1
        elif d % 7 not in (0, 4):
            break
    return k_last


def inverted(subject, home, mode):
    sc = _scale(mode)
    degs = [_to_degree(m, home, sc) for _, _, m in subject]
    D0 = degs[0][0]
    return [_from_degree(2 * D0 - D, 0, home, sc) for D, _ in degs]


def _place(pitches, centre):
    """Octave-shift so the line's mean sits nearest the voice's centre."""
    mean = sum(pitches) / len(pitches)
    k = round((centre - mean) / 12)
    return [m + 12 * k for m in pitches]


# --------------------------------------------------------------------------- #
# Stretto admissibility
# --------------------------------------------------------------------------- #
def stretto_options(subject, meter, home, mode):
    """[(delay_steps, interval)] for which the subject against itself breaks no
    forbidden rule and has at most one unresolved strong-beat dissonance."""
    from .bach_rules import FORBIDDEN, check
    from .encode import MIDI_LO, NOTE0, STOI
    s_on = subject[0][0]
    rel = [(int((o - s_on) * STEPS_PER_Q), int(d * STEPS_PER_Q), m) for o, d, m in subject]
    L = rel[-1][0] + rel[-1][1]
    beat = int(_beat_q(meter) * STEPS_PER_Q)
    bar = int(Fraction(4 * meter[0], meter[1]) * STEPS_PER_Q)
    out = []
    for delay in range(beat, min(2 * bar, L - beat) + 1, beat):
        for iv in (12, -12, 7, -5, 5, -7):
            n = L + delay
            g = np.full((n, 2), STOI["REST"], dtype=np.int16)
            lead_hi = iv < 0                     # the leader is the upper voice
            for (s, d, m) in rel:
                for (vv, off, tr) in ((0 if lead_hi else 1, 0, 0), (1 if lead_hi else 0, delay, iv)):
                    mm = m + tr
                    g[s + off, vv] = NOTE0 + mm - MIDI_LO
                    g[s + off + 1:s + off + d, vv] = STOI["HOLD"]
            hits = check(g, meter=meter, tonic=home, mode=mode, start=delay)
            bad = [h for h in hits if h[0] in FORBIDDEN and h[0] not in ("final_cadence", "range")]
            diss = [h for h in hits if h[0] == "strong_diss"]
            cross = any(g[s, 0] < g[s, 1] for s in range(delay, L) if g[s, 0] >= NOTE0 and g[s, 1] >= NOTE0)
            if not bad and len(diss) <= 1 and not cross:
                out.append((delay, iv))
    return out


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #
def _choose(rng, opts):
    keys, ps = zip(*opts)
    ps = np.array(ps, dtype=float)
    return keys[int(rng.choice(len(keys), p=ps / ps.sum()))]


def plan_form(subject, meter, mode, tonic, nv, rng, groups=None) -> dict:
    bar = int(Fraction(4 * meter[0], meter[1]) * STEPS_PER_Q)
    beat = int(_beat_q(meter) * STEPS_PER_Q)
    s_on = subject[0][0]
    s0 = int(s_on * STEPS_PER_Q)
    off = s0 % bar
    S = int((subject[-1][0] + subject[-1][1] - s_on) * STEPS_PER_Q)
    centres = CENTRES[nv]

    def next_slot(t, extra_bars=0):
        """First step >= t at the subject's metric position, plus whole bars."""
        k = math.ceil((t - off) / bar)
        return k * bar + off + extra_bars * bar

    entries, sections, cadences, pedals, sequences, rests = [], [], [], [], [], []
    # ---- 1. exposition --------------------------------------------------- #
    orders = {2: [(0, 1), (1, 0)], 3: [(1, 0, 2), (0, 1, 2), (2, 1, 0), (1, 2, 0)],
              4: [(1, 0, 2, 3), (1, 0, 3, 2), (2, 1, 0, 3), (0, 1, 2, 3), (3, 2, 1, 0), (2, 3, 1, 0)],
              5: [(1, 0, 2, 3, 4), (2, 1, 0, 3, 4), (0, 1, 2, 3, 4)]}[nv]
    order = orders[int(rng.integers(len(orders)))]
    if order[-1] not in (0, nv - 1) and rng.random() < 0.88:     # last entry in an outer voice
        order = next((o for o in orders if o[-1] in (0, nv - 1)), order)
    t = s0
    for k, v in enumerate(order):
        if k == 2 and rng.random() < 0.25:
            t = next_slot(t, 1)                  # codetta before the third entry
        kind = "subject" if k % 2 == 0 else "answer"
        entries.append({"voice": v, "start": t, "kind": kind, "key": (0, mode == "minor"),
                        "section": "exposition"})
        t = t + S
    expo_end = t
    sections.append({"type": "exposition", "start": 0, "end": expo_end})

    # ---- 2. middle-section keys and groups ------------------------------- #
    if groups is None:
        groups = int(rng.choice([2, 3, 3, 4]))
    if mode == "minor":
        first = _choose(rng, [((3, False), 0.6), ((7, True), 0.2), ((5, True), 0.1), ((8, False), 0.1)])
        pool = [(3, False), (7, True), (5, True), (8, False), (10, False)]
        sub = (5, True)
    else:
        first = _choose(rng, [((9, True), 0.45), ((7, False), 0.35), ((4, True), 0.1), ((2, True), 0.1)])
        pool = [(9, True), (7, False), (4, True), (2, True), (5, False)]
        sub = (5, False)
    # a counter-exposition (I / V, answer first): Prout counts 13/48; here in
    # long fugues (5+ groups) only, decided without a random draw so default
    # plans are unchanged for a given seed
    counter = groups >= 5
    keys = [first]
    for g in range(1, groups):
        if g == groups - 1 and sub not in keys and rng.random() < 0.45:
            keys.append(sub)
            continue
        cands = [k for k in pool if k not in keys and k != sub]
        if not cands:                     # long fugues revisit keys, never twice running
            cands = [k for k in pool if k != keys[-1]]
        keys.append(cands[int(rng.integers(len(cands)))])
    if counter:
        keys = [(0, mode == "minor")] + keys
    st_opts = stretto_options(subject, meter, tonic, mode)
    stretto_groups = set()
    if st_opts and len(keys) >= 2 and rng.random() < 0.6:
        G = len(keys)
        stretto_groups.add(G - 1 if G >= 3 else 1)
        if G >= 5:                        # long fugues: a first stretto near 0.4-0.6 too
            stretto_groups.add(max(1 + int(counter), G // 2))
    invert_first = rng.random() < 0.2

    last_voice = order[-1]
    for g, key in enumerate(keys):
        # episode before the group
        ep_bars = int(rng.choice([2, 3, 3, 4, 4, 5, 6])) if (g > 0 or rng.random() < 0.6) else 0
        start = next_slot(t, ep_bars)
        # the group's voices
        cand = [v for v in range(nv) if v != last_voice]
        if g == int(counter) and rng.random() < 0.75:
            outer = [v for v in cand if v in (0, nv - 1)]
            cand = outer or cand
        v1 = cand[int(rng.integers(len(cand)))]
        if ep_bars:
            sections.append({"type": "episode", "start": t, "end": start, "to_key": key,
                             "rest_voice": v1 if rng.random() < 0.7 else None})
        grp = []
        if counter and g == 0:
            # counter-exposition: answer first, then the subject, in two voices
            grp.append({"voice": v1, "start": start, "kind": "answer", "key": key, "section": "counter"})
            v2c = [v for v in range(nv) if v != v1]
            v2 = v2c[int(rng.integers(len(v2c)))]
            s2 = next_slot(start + S)
            grp.append({"voice": v2, "start": s2, "kind": "subject", "key": key, "section": "counter"})
            entries += grp
            end = s2 + S
            sections.append({"type": "entries", "start": start, "end": end, "key": key, "counter": True})
            last_voice = v2
            t = end
            continue
        kind = "inverted" if (g == int(counter) and invert_first) else "subject"
        grp.append({"voice": v1, "start": start, "kind": kind, "key": key, "section": f"group{g}"})
        if g in stretto_groups:
            delay, iv = st_opts[int(rng.integers(len(st_opts)))]
            v2c = [v for v in range(nv) if v != v1]
            # the follower above the leader when the interval rises
            v2c = [v for v in v2c if (v < v1) == (iv > 0)] or v2c
            v2 = v2c[int(rng.integers(len(v2c)))]
            grp.append({"voice": v2, "start": start + delay, "kind": "stretto", "key": key,
                        "interval": iv, "section": f"group{g}", "leader": len(entries)})
            end = start + delay + S
        elif rng.random() < 0.4:
            v2c = [v for v in range(nv) if v != v1]
            v2 = v2c[int(rng.integers(len(v2c)))]
            s2 = next_slot(start + S)
            grp.append({"voice": v2, "start": s2, "kind": "answer", "key": key, "section": f"group{g}"})
            end = s2 + S
        else:
            end = start + S
        entries += grp
        sections.append({"type": "entries", "start": start, "end": end, "key": key})
        cadences.append({"step": start, "type": "PAC" if rng.random() < 0.65 else "IAC", "key": key})
        last_voice = grp[-1]["voice"]
        t = end

    # ---- 3. final section ------------------------------------------------ #
    fin_voice = _choose(rng, [(nv - 1, 0.45), (0, 0.45)] + [(v, 0.1 / max(1, nv - 2)) for v in range(1, nv - 1)])
    if fin_voice == last_voice:
        fin_voice = 0 if last_voice == nv - 1 else nv - 1
    ep_bars = int(rng.choice([2, 3, 4]))
    dom_pedal = rng.random() < 0.3 and fin_voice != nv - 1
    start = next_slot(t, ep_bars)
    sections.append({"type": "episode", "start": t, "end": start, "to_key": (0, mode == "minor"),
                     "rest_voice": fin_voice, "final": True})
    if dom_pedal:
        pedals.append({"start": max(t, start - 2 * bar), "end": start, "degree": 7})
        cadences.append({"step": start, "type": "HC_before", "key": (0, mode == "minor")})
    else:
        cadences.append({"step": start, "type": "PAC", "key": (0, mode == "minor")})
    entries.append({"voice": fin_voice, "start": start, "kind": "subject", "key": (0, mode == "minor"),
                    "section": "final"})
    end = start + S
    sections.append({"type": "entries", "start": start, "end": end, "key": (0, mode == "minor"),
                     "final": True})
    # intensification before the end (techniques report §3 E22, F25)
    if rng.random() < 0.35:
        cadences.append({"step": end, "type": "DC", "key": (0, mode == "minor")})
    dim7_into_final = rng.random() < 0.6
    neapolitan = mode == "minor" and rng.random() < 0.3
    coda_bars = int(rng.choice([2, 2, 3]))
    n = math.ceil((end + coda_bars * bar) / bar) * bar
    tonic_pedal = rng.random() < 0.55
    if tonic_pedal:
        pedals.append({"start": max(end, n - coda_bars * bar), "end": n, "degree": 0})
    sections.append({"type": "coda", "start": end, "end": n})
    picardy = mode == "minor" and rng.random() < 0.85
    countersubject = rng.random() < 0.65              # 31/48 WTC fugues have one (Prout)
    cadences.append({"step": n - beat, "type": "final", "key": (0, mode == "minor"), "picardy": picardy})

    # sequences in episodes
    for sec in sections:
        if sec["type"] != "episode" or sec["end"] - sec["start"] < 2 * bar or rng.random() >= 0.6:
            continue
        kind = _choose(rng, [("fifths", 0.5), ("rising56", 0.25), ("falling63", 0.25)])
        sec["sequence"] = kind
    return {"meter": list(meter), "mode": mode, "tonic": tonic, "nv": nv, "n_steps": n,
            "bar": bar, "beat": beat, "subject_steps": S, "order": list(order), "entries": entries,
            "sections": sections, "cadences": cadences, "pedals": pedals, "keys": keys,
            "stretto_options": st_opts[:8], "picardy": picardy, "neapolitan": neapolitan,
            "counter_exposition": counter,
            "countersubject": countersubject,
            "dim7_into_final": dim7_into_final}


# --------------------------------------------------------------------------- #
# Realisation into a piece dict
# --------------------------------------------------------------------------- #
def realise_plan(form: dict, subject, rng, piece_id="planned") -> dict:
    meter = tuple(form["meter"])
    mode, tonic, nv, n = form["mode"], form["tonic"], form["nv"], form["n_steps"]
    bar, beat = form["bar"], form["beat"]
    centres = CENTRES[nv]
    s_on = subject[0][0]
    voices = {v: [] for v in range(nv)}
    ann = []
    entry_notes = []                           # (start_step, midi) for harmony emissions
    for k, e in enumerate(form["entries"]):
        kd, kmin = e["key"]
        kmode = "minor" if kmin else "major"
        if k == 0:
            pitches = _place([m for _, _, m in subject], centres[e["voice"]])
            pitches = [m for _, _, m in subject] if abs(np.mean(pitches) - np.mean([m for _, _, m in subject])) < 1 \
                else pitches
        elif e["kind"] == "answer" and e["section"] in ("exposition", "counter"):
            pitches = _place(tonal_answer(subject, tonic, mode, centres[e["voice"]]), centres[e["voice"]])
        elif e["kind"] == "answer":
            # local answer: the subject at the fifth of the group's key
            pitches = _place(realise(subject, tonic, mode, (kd + 7) % 12,
                                     "major" if not kmin or kd == 7 else kmode, centres[e["voice"]]),
                             centres[e["voice"]])
        elif e["kind"] == "inverted":
            inv = inverted(subject, tonic, mode)
            pitches = _place(realise([(o, d, m) for (o, d, _), m in zip(subject, inv)], tonic, mode,
                                     kd, kmode, centres[e["voice"]]), centres[e["voice"]])
        elif e["kind"] == "stretto":
            lead = form["entries"][e["leader"]]
            lp = voices_pitch_cache[e["leader"]]
            pitches = [m + e["interval"] for m in lp]
        else:
            pitches = _place(realise(subject, tonic, mode, kd, kmode, centres[e["voice"]]),
                             centres[e["voice"]])
        if k == 0:
            voices_pitch_cache = {}
        voices_pitch_cache[k] = pitches
        shift = Fraction(e["start"], STEPS_PER_Q) - s_on
        for (o, d, _), m in zip(subject, pitches):
            voices[e["voice"]].append({"onset_q": str(o + shift), "duration_q": str(d), "midi": int(m),
                                       "tie": None, "grace": False})
            entry_notes.append((int((o + shift) * STEPS_PER_Q), int(m)))
        ann.append({"type": "subject", "analytical_status": "accepted",
                    "voice_id": f"voice_{e['voice'] + 1}",
                    "onset_q": str(Fraction(e["start"], STEPS_PER_Q)),
                    "duration_q": str(Fraction(form["subject_steps"], STEPS_PER_Q)),
                    "entry_function": "answer" if e["kind"] == "answer" else "subject",
                    "source_name": "planner"})
    fifths = next(x for x in range(-7, 8) if tonic_pc(x, mode) == tonic)
    score = {"piece_id": piece_id, "movement_type": "fugue",
             "meter_map": [{"numerator": meter[0], "denominator": meter[1]}],
             "key_signature_map": [{"fifths": fifths}],
             "quarter_length_total": str(Fraction(n, STEPS_PER_Q)),
             "voices": [{"voice_id": f"voice_{v + 1}", "mean_midi": 100 - v, "notes": nts}
                        for v, nts in voices.items()]}
    p = encode_piece(score, {"split": "generated", "stratification_fields": {"mode": mode}}, ann)

    # ---- keys per step ---------------------------------------------------- #
    key_rel = [(0, mode == "minor")] * n
    for e in form["entries"]:
        if e["kind"] == "answer" and e["section"] in ("exposition", "counter"):
            # a tonal answer's head stays in the tonic (that is what answering
            # 5 with 1 does): only the notes it changed; the rest, and all of a
            # real answer, is in the dominant
            k = tonal_head(subject, tonic, mode)
            head = 0 if k == 0 or k >= len(subject) else \
                int((subject[k][0] - subject[0][0]) * STEPS_PER_Q)
            for s in range(e["start"] + head, min(n, e["start"] + form["subject_steps"])):
                key_rel[s] = (7, mode == "minor")
    for sec in form["sections"]:
        if sec["type"] == "entries":
            for s in range(sec["start"], sec["end"]):
                key_rel[s] = sec["key"]
        elif sec["type"] == "episode":
            prev = key_rel[sec["start"] - 1] if sec["start"] > 0 else (0, mode == "minor")
            # a sequential episode pivots to the next key from its start (its
            # chain is written in that key); a free one modulates at its middle
            mid = sec["start"] if sec.get("sequence") else sec["start"] + (sec["end"] - sec["start"]) // 2
            for s in range(sec["start"], sec["end"]):
                key_rel[s] = prev if s < mid else sec["to_key"]
    p["key"] = np.array([1 + 2 * d + int(mn) for d, mn in key_rel], dtype=np.int16)
    p["next_key"], p["to_key"] = key_lookahead(p["key"])
    keys_abs = [((tonic + d) % 12, bool(mn)) for d, mn in key_rel]

    # ---- harmony ---------------------------------------------------------- #
    unit = harmonic_unit(meter)
    U = n // unit
    fixed = [set() for _ in range(U)]
    weak = [set() for _ in range(U)]           # entry notes struck inside a unit
    for s, m in entry_notes:
        if s // unit < U:
            (fixed if s % unit == 0 else weak)[s // unit].add(m % 12)
    for q in form["pedals"]:
        for u in range(q["start"] // unit, min(U, q["end"] // unit)):
            fixed[u].add((tonic + q["degree"]) % 12)
    allowed = [None] * U
    S_ = harmony.state_ids

    def put(u, sids):
        if 0 <= u < U:
            allowed[u] = sids if allowed[u] is None else (allowed[u] & sids) or sids

    for c in form["cadences"]:
        u = c["step"] // unit
        kd, kmin = c["key"]
        if c["type"] == "final":
            final_q = ("maj",) if (not kmin or c.get("picardy")) else ("min",)
            put(U - 1, S_(0, final_q, (0,)))
            put(U - 2, S_(0, final_q, (0,)) | S_(7, ("maj", "dom7"), (0,)))
            put(U - 3, S_(7, ("maj", "dom7"), (0,)))
            if form.get("neapolitan"):
                put(U - 4, S_(1, ("maj",), (1,)))                 # bII6
            else:
                put(U - 4, S_(2, ("min", "hdim7", "dim", "min7")) | S_(5, ("maj", "min")))
            continue
        if c["type"] == "DC":
            put(u, S_(8 if kmin else 9, ("maj",) if kmin else ("min",), (0,)))
            put(u - 1, S_(7, ("maj", "dom7"), (0,)))
            continue
        tq = ("min",) if kmin else ("maj",)
        # chords are relative to the local key: the cadence key is the key there
        if c["type"] == "HC_before":
            put(u - 1, S_(7, ("maj", "dom7"), (0,)))
            put(u - 2, S_(7, ("maj", "dom7"), (0,)) | S_(2, ("min", "hdim7", "dim"), (0, 1)) | S_(5, ("maj", "min"), (0, 1)))
        else:
            put(u, S_(0, tq, (0,) if c["type"] == "PAC" else (0, 1)))
            final_entry = kd == 0 and c["step"] == max(e["start"] for e in form["entries"])
            if final_entry and form.get("dim7_into_final"):
                put(u - 1, S_(11, ("dim7",), (0,)))               # vii°7 into the last entry
            else:
                put(u - 1, S_(7, ("maj", "dom7"), (0,)))
            put(u - 2, S_(2, ("min", "hdim7", "dim", "min7"), (0, 1)) | S_(5, ("maj", "min"), (0, 1))
                | S_(7, ("maj", "dom7"), (0,)))
    # sequences: fix root motion per chord span in the episode's target key
    seq_list = []
    per_bar = max(1, bar // unit)
    for sec in form["sections"]:
        kind = sec.get("sequence")
        if not kind:
            continue
        a, b = sec["start"], sec["end"] - bar          # leave the last bar for the cadence
        if b - a < 2 * bar:
            continue
        sc = MINOR if sec["to_key"][1] else MAJOR
        n_bars = (b - a) // bar
        if kind == "fifths":
            # two chords per bar, roots falling by diatonic fifths, landing on ii-V
            chain = []
            deg = 1                                         # ends on ii before the cadence bar
            for j in range(2 * n_bars):
                chain.append(deg)
                deg = (deg + 3) % 7
            chain = chain[::-1]
            # rebuild so the last chord is ii and each step down a fifth
            chain = [(1 + 4 * (2 * n_bars - 1 - j)) % 7 for j in range(2 * n_bars)]
            spans = [(a + j * (bar // 2), a + (j + 1) * (bar // 2), chain[j], 0) for j in range(2 * n_bars)]
            step = -2
        elif kind == "rising56":
            spans = []
            for j in range(n_bars):
                d0 = j % 7
                spans.append((a + j * bar, a + j * bar + bar // 2, d0, 0))
                spans.append((a + j * bar + bar // 2, a + (j + 1) * bar, (d0 + 5) % 7, 1))
            step = 2
        else:                                               # falling 6-3 chain, a chord per beat
            spans = []
            nb = (b - a) // beat
            for j in range(nb):
                spans.append((a + j * beat, a + (j + 1) * beat, (4 - j) % 7, 1))
            step = -2
        for s_a, s_b, deg, inv in spans:
            r = sc[deg] if deg < len(sc) else sc[deg % 7]
            third = (sc[(deg + 2) % 7] - r) % 12
            fifth = (sc[(deg + 4) % 7] - r) % 12
            q = "maj" if third == 4 and fifth == 7 else "min" if third == 3 and fifth == 7 else \
                "dim" if third == 3 and fifth == 6 else "aug"
            for u in range(s_a // unit, min(U, max(s_a // unit + 1, s_b // unit))):
                # the chain is in the target key; chord states are relative to
                # the key in force at that unit
                local = key_rel[min(n - 1, u * unit)][0]
                r_loc = (sec["to_key"][0] + r - local) % 12
                put(u, S_(r_loc, (q,), (inv,)))
        # the melodic restatement: top and bottom active voices
        active = [v for v in range(nv) if v != sec.get("rest_voice")]
        unit_steps = bar if kind != "falling63" else 2 * beat
        stm = (b - a) // unit_steps
        if stm >= 2:
            for v in (active[0], active[-1]):
                seq_list.append({"voice": v, "start": a, "unit": unit_steps, "statements": int(stm),
                                 "step": step})
    states = harmony.plan_chords(U, [keys_abs[min(n - 1, u * unit)] for u in range(U)], fixed, allowed, rng,
                                 weak_notes=weak)
    chord = np.ones(n, dtype=np.int16)
    bass = np.ones(n, dtype=np.int16)
    for u, st in enumerate(states):
        if st is None:
            continue
        kt = keys_abs[min(n - 1, u * unit)][0]
        r, q, inv = st
        root = (kt + r) % 12
        code = 2 + 12 * QUAL_NAMES.index(q) + (root - tonic) % 12
        b_pc = root if inv == 0 else (root + harmony.QUALS[q][1]) % 12
        chord[u * unit:(u + 1) * unit] = code
        bass[u * unit:(u + 1) * unit] = 2 + (b_pc - tonic) % 12
    nxt = np.ones(n, dtype=np.int16)
    cur = 1
    for s in range(n - 1, -1, -1):
        nxt[s] = cur
        if s and chord[s] != chord[s - 1]:
            cur = chord[s]
    p["chord"], p["next_chord"], p["bass"] = chord, nxt, bass

    # ---- texture ---------------------------------------------------------- #
    active = np.full((n, nv), 2, dtype=np.int16)
    first_in = {e["voice"]: e["start"] for e in form["entries"][::-1] if e["section"] == "exposition"}
    for v, st in first_in.items():
        active[:st, v] = 1
    for sec in form["sections"]:
        if sec["type"] == "episode" and sec.get("rest_voice") is not None:
            # the next entry's voice rests for the episode's last bar(s)
            a = max(sec["start"], sec["end"] - (2 * bar if sec.get("final") else bar))
            active[a:sec["end"], sec["rest_voice"]] = 1
    p["active"] = active

    # ---- cadences, pedals, sequences -------------------------------------- #
    to_cad = np.full(n, 16, dtype=np.int16)
    cad_key = np.zeros(n, dtype=np.int16)
    cs = sorted((c["step"], CAD_KEYS.index(cad_key_name(c["key"][0], c["key"][1])) + 1)
                for c in form["cadences"])
    ci = 0
    for k in range(n):
        while ci < len(cs) and cs[ci][0] < k:
            ci += 1
        if ci < len(cs):
            to_cad[k] = _bucket(cs[ci][0] - k)
            cad_key[k] = cs[ci][1]
    p["to_cad"], p["cad_key"] = to_cad, cad_key
    pedal = np.ones(n, dtype=np.int16)
    for q in form["pedals"]:
        pedal[q["start"]:q["end"]] = 2
    p["pedal"] = pedal
    # pedal tones are written into the bass like entries: one held note
    from .encode import MIDI_LO, NOTE0, STOI
    # the final chord: root-position tonic (Picardy third if planned). Not
    # written in: a fixed voicing ignored where the voices were and produced
    # octave leaps and parallels into it. The sampler keeps the last bar to
    # the chord's pitch classes, with the bass on the root, and voices it.
    third = 4 if (mode == "major" or form.get("picardy")) else 3
    p["final_chord"] = {"from": n - bar, "pcs": {tonic, (tonic + third) % 12, (tonic + 7) % 12},
                        "bass": {tonic}}
    for q in form["pedals"]:
        pc = (tonic + q["degree"]) % 12
        m = next(x for x in range(36, 48) if x % 12 == pc)
        v = nv - 1
        if (p["pnote"][q["start"]:q["end"], v] > 0).any():
            continue                                   # an entry is there: no pedal
        p["pnote"][q["start"], v] = 2 + m - MIDI_LO
        p["pnote"][q["start"] + 1:q["end"], v] = 1
        p["grid"][q["start"], v] = NOTE0 + m - MIDI_LO
        p["grid"][q["start"] + 1:q["end"], v] = STOI["HOLD"]
    seq = np.ones((n, nv), dtype=np.int16)
    for q in seq_list:
        beats = min(4, max(1, q["unit"] // beat))
        seq[q["start"]:q["start"] + q["unit"], q["voice"]] = 1 + beats
        seq[q["start"] + q["unit"]:q["start"] + q["statements"] * q["unit"], q["voice"]] = 5 + beats
    p["seq"] = seq
    p["sequences"] = seq_list
    # ---- countersubject: the line against the answer, restated ------------- #
    copies = []
    ents = form["entries"]
    if len(ents) >= 3 and form.get("countersubject"):
        a1 = ents[1]                                   # the exposition's answer
        src_voice = ents[0]["voice"]
        S = form["subject_steps"]
        side = 1
        for k in range(2, len(ents)):
            e = ents[k]
            if e["kind"] in ("stretto", "inverted") or (k + 1 < len(ents) and ents[k + 1]["kind"] == "stretto"):
                continue
            if e["section"] == "exposition":
                target = ents[k - 1]["voice"]           # the previous entry's voice goes on
            else:
                if rng.random() > 0.8:
                    continue
                cand = [v for v in range(nv) if v != e["voice"]
                        and (p["active"][e["start"]:e["start"] + S, v] == 2).all()
                        and not (p["pnote"][e["start"]:e["start"] + S, v] > 0).any()]
                if not cand:
                    continue
                side = -side                            # alternate above / below the subject
                pref = [v for v in cand if (v < e["voice"]) == (side > 0)] or cand
                target = min(pref, key=lambda v: abs(v - e["voice"]))
            if (p["pnote"][e["start"]:e["start"] + S, target] > 0).any():
                continue
            shift = voices_pitch_cache[k][-1] - voices_pitch_cache[1][-1]
            shift += 12 * round((centres[target] - (centres[src_voice] + shift)) / 12)
            copies.append({"voice": int(target), "start": int(e["start"]), "length": int(S),
                           "src_voice": int(src_voice), "src_start": int(a1["start"]), "shift": int(shift)})
    p["copies"] = copies
    # ---- climax: the soprano's register ceiling over the piece --------------- #
    # research §3 F23: highest soprano note at 0.80-0.92; before it the ceiling
    # rises gradually, after it the line relaxes.
    peak_at = int(n * float(rng.uniform(0.80, 0.90)))
    top_entries = [m for k, e in enumerate(form["entries"]) if e["voice"] == 0
                   for m in voices_pitch_cache[k]]
    base = max(top_entries) if top_entries else centres[0] + 5
    peak = min(86, max(base + 2, centres[0] + 9))
    ceil = np.zeros(n, dtype=np.int16)
    for s_ in range(n):
        if s_ <= peak_at:
            ceil[s_] = int(round(base - 2 + (peak - base + 2) * (s_ / peak_at) ** 1.5))
        else:
            ceil[s_] = int(round(peak - 4 * (s_ - peak_at) / max(1, n - peak_at)))
    # write the climax note in: the soft pull reached the planned peak in only
    # 5 of 40 compositions. A chord tone of the planned harmony at or just under
    # the peak, on a beat where the soprano is free, held for a beat.
    from .labels import QUALS
    from .encode import MIDI_LO as _LO, NOTE0 as _N0, STOI as _ST
    placed = None
    for off in sorted(range(-2 * bar, 2 * bar + 1, beat), key=abs):
        t = peak_at + off
        t -= t % beat
        if t < form["entries"][0]["start"] + form["subject_steps"] or t + beat > n - bar:
            continue
        if (p["pnote"][max(0, t - beat):t + 2 * beat, 0] > 0).any() or (p["active"][t:t + beat, 0] != 2).all():
            continue
        if any(q["voice"] == 0 and q["start"] <= t < q["start"] + q["length"] for q in copies):
            continue
        code = int(p["chord"][t])
        if code < 2:
            continue
        qi, r = divmod(code - 2, 12)
        pcs = {(tonic + r + x) % 12 for x in QUALS[QUAL_NAMES[qi]]}
        cands = [m for m in range(peak - 3, peak + 1) if m % 12 in pcs]
        if not cands:
            continue
        m = max(cands)
        p["pnote"][t, 0] = 2 + m - _LO
        p["pnote"][t + 1:t + beat, 0] = 1
        p["grid"][t, 0] = _N0 + m - _LO
        p["grid"][t + 1:t + beat, 0] = _ST["HOLD"]
        placed = (t, m)
        break
    p["climax"] = {"step": placed[0] if placed else peak_at, "peak": int(placed[1] if placed else peak),
                   "ceiling": ceil, "written": placed is not None}
    p["plan_source"] = "planner"
    p["form"] = form
    p["prompt_steps"] = form["entries"][0]["start"] + form["subject_steps"]
    p["pedal_notes"] = [((tonic + q["degree"]) % 12, q["start"], q["end"]) for q in form["pedals"]]
    return p


def plan_piece(subject, meter, mode, tonic, nv, seed=0, piece_id="planned", groups=None):
    """groups: number of middle-section entry groups (default 2-4; 6-8 gives a
    fugue at the scale of BWV 543)."""
    rng = np.random.default_rng(seed)
    form = plan_form(subject, meter, mode, tonic, nv, rng, groups)
    return realise_plan(form, subject, rng, piece_id)


def describe(form) -> str:
    bar = form["bar"]
    roman = lambda k: cad_key_name(k[0], k[1])
    out = [f"{form['n_steps'] / bar:.0f} bars, {form['nv']} voices, {form['mode']}"]
    for sec in form["sections"]:
        line = f"  bars {sec['start'] / bar + 1:5.1f}-{sec['end'] / bar + 1:5.1f} {sec['type']:10s}"
        if sec["type"] == "entries":
            line += f" in {roman(sec['key'])}"
        if sec["type"] == "episode":
            line += f" -> {roman(sec['to_key'])}" + (f", sequence {sec['sequence']}" if sec.get("sequence") else "")
            if sec.get("rest_voice") is not None:
                line += f", voice {sec['rest_voice'] + 1} rests before its entry"
        ents = [e for e in form["entries"] if sec["start"] <= e["start"] < sec["end"]]
        if ents:
            line += "; entries: " + ", ".join(f"v{e['voice'] + 1} {e['kind']}"
                                             + (f" ({e['interval']:+d}, +{(e['start'] - ents[0]['start']) / form['beat']:.0f} beats)"
                                                if e["kind"] == "stretto" else "") for e in ents)
        out.append(line)
    for q in form["pedals"]:
        out.append(f"  pedal on {'tonic' if q['degree'] == 0 else 'dominant'} bars "
                   f"{q['start'] / bar + 1:.1f}-{q['end'] / bar + 1:.1f}")
    if form["mode"] == "minor":
        out.append(f"  Picardy third: {form['picardy']}; Neapolitan before the last V: {form.get('neapolitan')}")
    out.append(f"  countersubject restated against later entries: {form.get('countersubject')}")
    out.append(f"  diminished 7th into the final entry: {form.get('dim7_into_final')}; "
               f"deceptive cadence before the coda: {any(c['type'] == 'DC' for c in form['cadences'])}")
    return "\n".join(out)


def subject_from_piece(pid: str):
    """(subject, meter, mode, tonic, voices) of a corpus fugue."""
    from .encode import load_corpus
    from .entries import detect
    p = next(x for x in load_corpus(ROOT, baroque=True, detect_entries=False) if x["piece_id"] == pid)
    f = ROOT / "data" / "canonical" / f"{pid}.json"
    if not f.exists():
        f = ROOT / "data" / "canonical_baroque" / f"{pid}.json"
    score = json.loads(f.read_text())
    ap = ROOT / "data" / "annotations" / f"{pid}.json"
    ann = json.loads(ap.read_text())["annotations"] if ap.exists() else None
    det = detect(score, ann)
    subject = [(Fraction(o), Fraction(d), int(m)) for o, d, m, _ in det["subject"]]
    meter = (4, 4) if p["meter_tok"] == "METER_other" else tuple(int(x) for x in p["meter_tok"].split("_")[1:3])
    return subject, meter, p["mode"], p["tonic"], p["nv"]


def main(argv=None) -> int:
    from .compose import parse_subject
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-from")
    ap.add_argument("--subject")
    ap.add_argument("--meter", default="4/4")
    ap.add_argument("--mode", default="minor")
    ap.add_argument("--tonic", type=int, default=0)
    ap.add_argument("--voices", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.subject_from:
        subject, meter, mode, tonic, nv = subject_from_piece(args.subject_from)
    else:
        subject = parse_subject(args.subject)
        meter = tuple(int(x) for x in args.meter.split("/"))
        mode, tonic, nv = args.mode, args.tonic, args.voices
    p = plan_piece(subject, meter, mode, tonic, nv, seed=args.seed)
    print(describe(p["form"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
