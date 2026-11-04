"""Make a plan for a new subject (step 5 of progress/generator.md).

There is no Bach piece to borrow a plan from when the subject is new, so the
form is taken from a *training* fugue with the same voice count and a similar
subject length (the template) and rescaled to the new subject:

- **Entries.** The template's entries (voice, position, level), with positions
  scaled by new/template subject length and snapped to the subject's metric
  placement. Each entry is the new subject moved *degree by degree* into the
  entry's key, so a minor subject entering in the relative major comes out
  major, and chromatic alterations (a raised seventh) travel with it. Register
  follows the template entry. Answers are real, not tonal (a known gap).
- **Key path.** The template's per-bar local keys (they are relative to the
  tonic, so they transfer), time-scaled.
- **Sequences.** The template's sequence placements, scaled, outside entries.

Only the template's *form* is used: no template notes reach the output.

The result is a piece dict in the shape `encode.encode_piece` returns, with the
first entry as the only notes, ready for `sample.generate`.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .encode import (MIDI_LO, NOTE0, STEPS_PER_Q, _beat_q, encode_piece, key_lookahead,
                     tonic_pc)
from .sequences import find_sequences

MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]


def _to_degree(m: int, tonic: int, scale: list[int]) -> tuple[int, int]:
    octv, pc = divmod(m - tonic, 12)
    d = max(i for i, x in enumerate(scale) if x <= pc)
    return octv * 7 + d, pc - scale[d]


def _from_degree(D: int, alt: int, tonic: int, scale: list[int]) -> int:
    octv, d = divmod(D, 7)
    return tonic + 12 * octv + scale[d] + alt


def realise(subject: list[tuple[Fraction, Fraction, int]], home: int, home_mode: str,
            interval: int, mode: str, near: int) -> list[int]:
    """Subject pitches moved degree-for-degree into the key `interval` semitones
    above home, in `mode`, at the octave putting its first note nearest `near`."""
    hs = MAJOR if home_mode == "major" else MINOR
    ts = MAJOR if mode == "major" else MINOR
    degs = [_to_degree(m, home, hs) for _, _, m in subject]
    first_deg = degs[0][0]
    # keep the subject's first scale degree; the target tonic sits `interval` up
    out = [_from_degree(D - first_deg, a, home + interval, ts) for D, a in degs]
    base = _from_degree(first_deg, degs[0][1], home + interval, ts) - out[0]
    out = [m + base for m in out]
    k = round((near - out[0]) / 12)
    return [m + 12 * k for m in out]


def _entries_of(p: dict) -> list[dict]:
    """Template entries from role arrays: (voice, start, end, first pitch, answer)."""
    out = []
    for v in range(p["nv"]):
        r = p["role"][:, v]
        s = 0
        while s < len(r):
            if r[s] in (2, 3):
                e = s
                while e < len(r) and r[e] == r[s]:
                    e += 1
                col = p["grid"][s:e, v]
                notes = np.flatnonzero((col >= NOTE0) & (col < NOTE0 + 88))
                if len(notes):
                    out.append({"voice": v, "start": s, "end": e,
                                "first": MIDI_LO + int(col[notes[0]]) - NOTE0,
                                "answer": bool(r[s] == 3)})
                s = e
            else:
                s += 1
    return sorted(out, key=lambda x: (x["start"], x["voice"]))


def choose_template(pieces: list[dict], nv: int, subject_steps: int,
                    rng: np.random.Generator, k: int = 4, exclude: str | None = None) -> dict:
    cands = [p for p in pieces if p["split"] == "train" and p["corpus"] == "fugue"
             and p["has_plan"] and p["nv"] == nv and p["subject"] is not None
             and p["piece_id"] != exclude and len(_entries_of(p)) >= nv]
    cands.sort(key=lambda p: abs(math.log(subject_steps / len(p["subject"]))))
    return cands[int(rng.integers(0, min(k, len(cands))))]


def make_plan(subject: list[tuple[Fraction, Fraction, int]], meter: tuple[int, int],
              mode: str, tonic: int, template: dict, piece_id: str = "new") -> dict:
    """A piece dict for `sample.generate`: the subject's first entry as prompt,
    entries/keys/sequences planned from the template's form."""
    s_on = subject[0][0]
    s_steps = int((subject[-1][0] + subject[-1][1] - s_on) * STEPS_PER_Q)
    t_steps = len(template["subject"])
    f = s_steps / t_steps
    bar_steps = int(Fraction(4 * meter[0], meter[1]) * STEPS_PER_Q)
    beat_steps = int(_beat_q(meter) * STEPS_PER_Q)
    snap = bar_steps // 2 if (meter[0] % 2 == 0 and meter[1] <= 4) else bar_steps
    s0 = int(s_on * STEPS_PER_Q)
    t_entries = _entries_of(template)
    t0 = t_entries[0]["start"]
    n = int(math.ceil((s0 + (template["n_steps"] - t0) * f) / bar_steps) * bar_steps)

    def scale(s):  # template step -> new step
        return s0 + (s - t0) * f

    # Template home-key register per voice, and the voice order (slot 0 highest)
    t_first = MIDI_LO + int(template["subject"][np.flatnonzero(
        (template["subject"] >= NOTE0) & (template["subject"] < NOTE0 + 88))[0]]) - NOTE0
    voices = {v: [] for v in range(template["nv"])}
    ann = []
    busy = {v: 0 for v in voices}
    for k, e in enumerate(t_entries):
        start = s0 + round((scale(e["start"]) - s0) / snap) * snap
        if k == 0:
            start = s0
        end = start + s_steps
        if end > n or start < busy[e["voice"]]:
            continue
        interval = (e["first"] - t_first) % 12
        kk = int(template["key"][min(e["start"], len(template["key"]) - 1)])
        kdeg, kminor = ((kk - 1) // 2, (kk - 1) % 2) if kk else (0, mode == "minor")
        emode = (("minor" if kminor else "major") if kdeg == interval
                 else ("major" if interval in (3,) and mode == "minor"
                       else "minor" if interval == 9 and mode == "major" else mode))
        if k == 0:
            pitches = [m for _, _, m in subject]
        else:
            # the template entry's register, moved to the nearest equivalent
            # pitch in the new home key
            near = e["first"] + ((tonic - template["tonic"] + 6) % 12 - 6)
            pitches = realise(subject, tonic, mode, interval, emode, near)
        shift = Fraction(start, STEPS_PER_Q) - s_on
        for (o, d, _), m in zip(subject, pitches):
            voices[e["voice"]].append({"onset_q": str(o + shift), "duration_q": str(d),
                                       "midi": int(m), "tie": None, "grace": False})
        busy[e["voice"]] = end
        ann.append({"type": "subject", "analytical_status": "accepted",
                    "voice_id": f"voice_{e['voice'] + 1}",
                    "onset_q": str(Fraction(start, STEPS_PER_Q)),
                    "duration_q": str(Fraction(s_steps, STEPS_PER_Q)),
                    "entry_function": "answer" if e["answer"] else "subject",
                    "source_name": "plan"})

    fifths = next(x for x in range(-7, 8) if tonic_pc(x, mode) == tonic)
    score = {
        "piece_id": piece_id, "movement_type": "fugue",
        "meter_map": [{"numerator": meter[0], "denominator": meter[1]}],
        "key_signature_map": [{"fifths": fifths}],
        "quarter_length_total": str(Fraction(n, STEPS_PER_Q)),
        "voices": [{"voice_id": f"voice_{v + 1}", "mean_midi": 100 - v, "notes": notes}
                   for v, notes in voices.items()],
    }
    p = encode_piece(score, {"split": "generated", "stratification_fields": {"mode": mode}}, ann)
    # encode_piece sets the subject line from the earliest entry; key path and
    # sequences come from the template's form, time-scaled.
    idx = np.clip(np.round(t0 + (np.arange(n) - s0) / f).astype(int), 0, template["n_steps"] - 1)
    p["key"] = template["key"][idx].copy()
    p["key"][:s0 + s_steps] = 1                          # exposition opens at home
    p["next_key"], p["to_key"] = key_lookahead(p["key"])
    seqs = []
    from .report import beat_steps_of
    t_beat = beat_steps_of(template)
    for q in find_sequences(template["grid"], t_beat):
        start = int(round(scale(q["start"]) / beat_steps) * beat_steps)
        unit = max(1, round(q["unit"] / t_beat)) * beat_steps
        end = start + q["statements"] * unit
        v = q["voice"]
        if start < s0 + s_steps or end > n or (p["role"][start:end, v] >= 2).any():
            continue
        seqs.append({"voice": v, "start": start, "unit": unit,
                     "statements": q["statements"], "step": q["step"]})
    seq = np.ones((n, p["nv"]), dtype=np.int16)
    for q in seqs:
        beats = min(4, max(1, q["unit"] // beat_steps))
        seq[q["start"]:q["start"] + q["unit"], q["voice"]] = 1 + beats
        seq[q["start"] + q["unit"]:q["start"] + q["statements"] * q["unit"], q["voice"]] = 5 + beats
    p["seq"] = seq
    p["sequences"] = seqs
    p["template"] = template["piece_id"]
    p["plan_source"] = "template"
    p["prompt_steps"] = s0 + s_steps
    return p
