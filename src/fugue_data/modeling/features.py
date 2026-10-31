"""Musical features for a preference model.

Deliberately computed from the canonical score rather than taken from the stored
diagnostics vector: those diagnostics were used to *select* which pairs to show a
reviewer, so training on them risks learning the selection rule instead of the
preference. These are recomputed from the notes.
"""

from __future__ import annotations

import collections
import math

from ..normalize.rational import parse_frac

CONSONANT = {0, 3, 4, 7, 8, 9}

NAMES = [
    "note_density", "mean_dur", "dur_entropy", "distinct_durs",
    "step_share", "leap_share", "mean_abs_interval", "interval_entropy",
    "direction_changes", "repeat_share",
    "pitch_range", "mean_voice_range", "voice_spread",
    "simul_attack_rate", "attacks_per_onset",
    "triad_rate", "hollow_rate", "dissonant_rate", "semitone_clash_rate",
    "ends_triad", "peak_strikes", "min_voice_activity",
]


def _entropy(counts) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    out = 0.0
    for c in counts:
        if c:
            p = c / total
            out -= p * math.log(p + 1e-12, 2)
    return out


def features(score: dict) -> list[float]:
    voices = score["voices"]
    total = float(parse_frac(score.get("quarter_length_total", "1"))) or 1.0
    notes = [n for v in voices for n in v["notes"]]
    if not notes:
        return [0.0] * len(NAMES)

    durs = collections.Counter(n["duration_q"] for n in notes)
    ivs, dirs, repeats, nmoves = [], 0, 0, 0
    vranges, activity = [], []
    prev_dir = 0
    for v in voices:
        ns = sorted(v["notes"], key=lambda n: parse_frac(n["onset_q"]))
        if ns:
            vranges.append(max(n["midi"] for n in ns) - min(n["midi"] for n in ns))
            activity.append(sum(float(parse_frac(n["duration_q"])) for n in ns) / total)
        for a, b in zip(ns, ns[1:]):
            d = b["midi"] - a["midi"]
            ivs.append(d)
            nmoves += 1
            if d == 0:
                repeats += 1
            s = (d > 0) - (d < 0)
            if s and prev_dir and s != prev_dir:
                dirs += 1
            if s:
                prev_dir = s

    onsets = sorted({parse_frac(n["onset_q"]) for n in notes})
    simul = attacks = 0
    triad = hollow = diss = clash = slices_ = 0
    att = collections.Counter(parse_frac(n["onset_q"]) for n in notes)
    for t in onsets:
        sounding = []
        for v in voices:
            for n in v["notes"]:
                o = parse_frac(n["onset_q"])
                if o <= t < o + parse_frac(n["duration_q"]):
                    sounding.append(n)
                    break
        attacks += att[t]
        if att[t] == len(voices):
            simul += 1
        if len(sounding) >= 3:
            slices_ += 1
            pcs = {n["pitch_class"] for n in sounding}
            ps = sorted((n["midi"] for n in sounding), reverse=True)
            if any((b - a) % 12 in (1, 11) for i, a in enumerate(ps) for b in ps[i + 1:]):
                clash += 1
            if len(pcs) < 3:
                hollow += 1
            if any(((x + 4) % 12 in pcs and (x + 7) % 12 in pcs)
                   or ((x + 3) % 12 in pcs and (x + 7) % 12 in pcs) for x in pcs):
                triad += 1
            bass = min(n["midi"] for n in sounding)
            if any((n["midi"] - bass) % 12 not in CONSONANT for n in sounding):
                diss += 1

    last = onsets[-1]
    endl = []
    for v in voices:
        for n in v["notes"]:
            o = parse_frac(n["onset_q"])
            if o <= last < o + parse_frac(n["duration_q"]):
                endl.append(n)
                break
    epcs = {n["pitch_class"] for n in endl}
    ends_triad = float(any(((x + 4) % 12 in epcs and (x + 7) % 12 in epcs)
                           or ((x + 3) % 12 in epcs and (x + 7) % 12 in epcs)
                           for x in epcs))

    peaks = 0
    for v in voices:
        if v["notes"]:
            hi = max(n["midi"] for n in v["notes"])
            peaks += sum(1 for n in v["notes"] if n["midi"] == hi)

    ns_ = max(1, len(notes)); nm = max(1, nmoves); sl = max(1, slices_)
    no = max(1, len(onsets))
    return [
        len(notes) / total,
        sum(float(parse_frac(n["duration_q"])) for n in notes) / ns_,
        _entropy(list(durs.values())),
        float(len(durs)),
        sum(1 for d in ivs if 0 < abs(d) <= 2) / nm,
        sum(1 for d in ivs if abs(d) > 4) / nm,
        sum(abs(d) for d in ivs) / nm,
        _entropy(list(collections.Counter(ivs).values())),
        dirs / nm,
        repeats / nm,
        float(max(n["midi"] for n in notes) - min(n["midi"] for n in notes)),
        sum(vranges) / max(1, len(vranges)),
        float(len(voices)),
        simul / no,
        attacks / no,
        triad / sl, hollow / sl, diss / sl, clash / sl,
        ends_triad,
        peaks / max(1, len(voices)),
        min(activity) if activity else 0.0,
    ]
