"""Label the musical events a form planner would decide (step 1 of the
hierarchical plan, progress/generator.md).

From a canonical score (or a generated grid) this finds:

- **Chords**, one per harmonic unit (the beat; half the beat in slow meters,
  a third of it in compound ones): root, quality and bass, by template
  matching on duration-weighted pitch classes, with the note sounding at the
  unit's start and the bass weighted up and a small preference for chords
  diatonic to the local key.
- **Cadences**: authentic (V(7) -> I with the bass falling 5 -> 1, arriving
  on a strong beat; perfect if the top voice lands on the tonic) and half
  (a strong-beat arrival on V, bass on 5, held at least two units).
- **Pedal points**: the lowest voice holding or restriking one pitch for at
  least a bar while the harmony above changes.
- **Texture**: per bar, which voices sound (at least a quarter of the bar).
- **Entries** (from `entries.detect`), with **stretto** (entries overlapping
  in time) and **countersubjects** (the line that accompanied the first answer,
  found again against later entries).
- **Sections**: exposition, then entry groups and episodes, then the coda,
  each with key, texture, sequences, pedals and the cadence that closes it.

Each detector is checked against expert labels where they exist (`validate`):
Algomus cadences, pedals and countersubjects on WTC I, DCML chord labels on
Corelli, Bach suites, Couperin, Handel and Frescobaldi.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..normalize.rational import parse_frac
from ..tokenization.views import merge_tied
from .encode import CAD_KEYS, STEPS_PER_Q, _beat_q, tonic_pc

QUALS = {"maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
         "dom7": (0, 4, 7, 10), "min7": (0, 3, 7, 10), "hdim7": (0, 3, 6, 10),
         "dim7": (0, 3, 6, 9), "maj7": (0, 4, 7, 11)}
QUAL_NAMES = list(QUALS)
FAMILY = {"maj": "maj", "dom7": "maj", "maj7": "maj", "min": "min", "min7": "min",
          "dim": "dim", "hdim7": "dim", "dim7": "dim", "aug": "aug"}
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10, 11)        # natural + raised 7th
KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


# --------------------------------------------------------------------------- #
# Notes
# --------------------------------------------------------------------------- #
def notes_from_score(score: dict) -> list[list[tuple[int, int, int]]]:
    """Per voice (highest first): (start step, end step, midi), ties merged,
    chord members kept."""
    voices = sorted(score["voices"], key=lambda v: -float(v.get("mean_midi") or 0))
    out = []
    for v in voices:
        ns = [n for n in v["notes"] if not n.get("grace") and parse_frac(n["duration_q"]) > 0]
        ev = []
        for o, d, m in merge_tied(ns):
            s = int(round(o * STEPS_PER_Q))
            ev.append((s, max(s + 1, int(round((o + d) * STEPS_PER_Q))), m))
        out.append(sorted(ev))
    return out


def notes_from_grid(grid: np.ndarray) -> list[list[tuple[int, int, int]]]:
    from .encode import MIDI_LO, NOTE0, STOI
    out = []
    for v in range(grid.shape[1]):
        ev, cur = [], None
        for s, t in enumerate(grid[:, v]):
            t = int(t)
            if NOTE0 <= t < NOTE0 + 88:
                if cur:
                    ev.append((cur[0], s, cur[1]))
                cur = (s, MIDI_LO + t - NOTE0)
            elif t != STOI["HOLD"] and cur:
                ev.append((cur[0], s, cur[1]))
                cur = None
        if cur:
            ev.append((cur[0], grid.shape[0], cur[1]))
        out.append(ev)
    return out


def harmonic_unit(meter: tuple[int, int]) -> int:
    """Steps per harmonic unit: the beat, or a finer pulse in slow/compound meters."""
    beat = _beat_q(meter)
    num, den = meter
    if den >= 8 and num % 3 == 0 and num > 3:        # compound: the beat's third
        u = beat / 3
    elif beat > 1:                                    # 2/2, 3/2: the quarter
        u = beat / 2
    else:
        u = beat
    return max(1, int(u * STEPS_PER_Q))


def _sounding(voices, n_steps):
    """(n_steps, nv) midi sounding (-1 rest) and attack flags, top note per voice."""
    nv = len(voices)
    snd = np.full((n_steps, nv), -1, dtype=np.int16)
    att = np.zeros((n_steps, nv), dtype=bool)
    for v, ev in enumerate(voices):
        for s, e, m in ev:
            if s >= n_steps:
                continue
            seg = snd[s:min(e, n_steps), v]
            snd[s:min(e, n_steps), v] = np.maximum(seg, m)
            att[s, v] = True
    return snd, att


def _pc_hist(voices, a, b):
    """Duration-weighted pitch-class histogram over [a, b), all notes."""
    h = np.zeros(12)
    for ev in voices:
        for s, e, m in ev:
            lo, hi = max(s, a), min(e, b)
            if hi > lo:
                h[m % 12] += hi - lo
    return h


def local_key_per_unit(voices, n_steps, unit, bar_steps):
    """(tonic pc, minor) per unit: Krumhansl-Schmuckler over a 3-bar window."""
    keys = []
    cache = {}
    for a in range(0, n_steps, unit):
        b0 = (a // bar_steps) * bar_steps
        if b0 not in cache:
            h = _pc_hist(voices, max(0, b0 - bar_steps), b0 + 2 * bar_steps)
            best = (-2.0, (0, 0))
            if h.sum() > 0:
                for t in range(12):
                    rot = np.roll(h, -t)
                    for mi, prof in enumerate((KK_MAJOR, KK_MINOR)):
                        r = np.corrcoef(rot, prof)[0, 1]
                        if r > best[0]:
                            best = (r, (t, mi))
            cache[b0] = best[1]
        keys.append(cache[b0])
    return keys


# --------------------------------------------------------------------------- #
# Chords
# --------------------------------------------------------------------------- #
def chords(voices, n_steps, unit, bar_steps):
    """One dict per unit: root, qual, bass, top, key (tonic, minor), weight."""
    snd, att = _sounding(voices, n_steps)
    keys = local_key_per_unit(voices, n_steps, unit, bar_steps)
    out = []
    for i, a in enumerate(range(0, n_steps, unit)):
        b = min(n_steps, a + unit)
        h = _pc_hist(voices, a, b)
        start = snd[a]
        live = start[start >= 0]
        if len(live) == 0:                       # nothing at the unit start: first attack
            cols = snd[a:b][(snd[a:b] >= 0).any(axis=1)]
            live = cols[0][cols[0] >= 0] if len(cols) else live
        if h.sum() == 0 or len(live) == 0:
            out.append({"root": None, "qual": None, "bass": None, "top": None,
                        "key": keys[i], "weight": 0.0})
            continue
        w = h / h.sum()
        bass, top = int(live.min()) % 12, int(live.max()) % 12
        for m in live:
            w[int(m) % 12] += 0.15                # what sounds on the beat counts more
        w[bass] += 0.25
        kt, km = keys[i]
        scale = {(kt + x) % 12 for x in (MINOR_SCALE if km else MAJOR_SCALE)}
        best = (-9.0, 0, "maj")
        total = w.sum()
        for r in range(12):
            for q, tones in QUALS.items():
                pcs = {(r + t) % 12 for t in tones}
                s = sum(w[p] for p in pcs) - 0.7 * sum(w[p] for p in range(12) if p not in pcs)
                if len(tones) == 4:
                    s -= 0.12 * total + (0.0 if w[(r + tones[3]) % 12] > 0.05 else 0.5)
                if q == "aug":
                    s -= 0.2 * total
                if bass == r:
                    s += 0.12 * total
                if pcs <= scale:
                    s += 0.08 * total
                if s > best[0]:
                    best = (s, r, q)
        out.append({"root": best[1], "qual": best[2], "bass": bass, "top": top,
                    "key": keys[i], "weight": float(h.sum())})
    return out


# --------------------------------------------------------------------------- #
# Cadences, pedals, texture
# --------------------------------------------------------------------------- #
def _strong(i, unit, bar_steps, meter):
    pos = (i * unit) % bar_steps
    if pos == 0:
        return True
    num = meter[0]
    return num in (4, 12) and pos == bar_steps // 2


def cad_key_name(pc_rel: int, minor: bool) -> str:
    roman = {0: "I", 1: "II", 2: "II", 3: "III", 4: "III", 5: "IV", 6: "IV", 7: "V",
             8: "VI", 9: "VI", 10: "VII", 11: "VII"}[pc_rel % 12]
    name = roman.lower() if minor else roman
    return name if name in CAD_KEYS else "other"


def cadences(ch, voices, unit, bar_steps, meter, tonic):
    """[{step, type: PAC/IAC/HC, key_pc, minor, key_area}]."""
    snd, _ = _sounding(voices, len(ch) * unit)
    lowest = [int(x[x >= 0].min()) if (x >= 0).any() else -1 for x in snd]
    out = []
    for i in range(1, len(ch)):
        c = ch[i]
        if c["root"] is None or not _strong(i, unit, bar_steps, meter):
            continue
        X = c["root"]
        # authentic: previous different chord is V(7) of X, bass 5 -> 1
        j = i - 1
        while j > 0 and ch[j]["root"] == X and ch[j]["qual"] == c["qual"]:
            j -= 1
        p = ch[j]
        if (FAMILY.get(c["qual"]) in ("maj", "min") and c["qual"] in ("maj", "min")
                and c["bass"] == X and p["root"] == (X + 7) % 12
                and p["qual"] in ("maj", "dom7") and p["bass"] == (X + 7) % 12
                and i - j <= 2):
            b1, b0 = lowest[i * unit], lowest[j * unit]
            if b1 >= 0 and b0 >= 0 and (b0 - b1) % 12 == 7 and abs(b0 - b1) in (5, 7, 17, 19):
                t = "PAC" if c["top"] == X else "IAC"
                out.append({"step": i * unit, "type": t, "key_pc": X, "minor": c["qual"] == "min"})
                continue
        # half: strong-beat arrival on V of the local key, bass on 5, held 2 units
        kt, km = c["key"]
        if (c["qual"] == "maj" and X == (kt + 7) % 12 and c["bass"] == X and i + 1 < len(ch)
                and ch[i + 1]["root"] == X and ch[i + 1]["bass"] == X
                and (i + 2 >= len(ch) or ch[i + 2]["root"] != X or ch[i + 2]["qual"] != "maj")
                and p["root"] != X):
            pb = p["bass"]
            if pb in ((X + 1) % 12, (X - 1) % 12, (X + 5) % 12, (X + 2) % 12, (X - 2) % 12):
                out.append({"step": i * unit, "type": "HC", "key_pc": kt, "minor": bool(km)})
    # one cadence per bar half: keep PAC > IAC > HC
    rank = {"PAC": 0, "IAC": 1, "HC": 2}
    keep = {}
    for c in out:
        k = c["step"] // max(unit, bar_steps // 2)
        if k not in keep or rank[c["type"]] < rank[keep[k]["type"]]:
            keep[k] = c
    res = sorted(keep.values(), key=lambda c: c["step"])
    for c in res:
        c["key_area"] = cad_key_name(c["key_pc"] - tonic, c["minor"])
    return res


def pedals(voices, ch, unit, bar_steps, tonic):
    """[{start, end, midi, degree}]: the lowest sounding voice holds or restrikes
    one pitch for at least a bar, on the tonic or dominant (of the home or the
    local key), while the chord above changes at least three times.

    Against Algomus (WTC I, 19 pedals): precision / recall in progress/generator.md.
    Top-voice (inverted) pedals are not looked for: in the sweep they added more
    false hits than true ones."""
    n = len(ch) * unit
    snd, _ = _sounding(voices, n)
    line = np.array([int(x[x >= 0].min()) if (x >= 0).any() else -1 for x in snd])
    out = []
    s = 0
    while s < n:
        m = line[s]
        if m < 0:
            s += 1
            continue
        e, gap = s, 0
        while e < n and (line[e] == m or (line[e] < 0 and gap < unit)):
            gap = gap + 1 if line[e] < 0 else 0
            e += 1
        e -= gap
        if e - s >= max(bar_steps, 4 * STEPS_PER_Q):
            k0, k1 = s // unit, min(len(ch), e // unit)
            changes = sum(1 for k in range(k0 + 1, k1) if ch[k]["root"] != ch[k - 1]["root"])
            lt = ch[k0]["key"][0]
            rel_home, rel_local = (m - tonic) % 12, (m - lt) % 12
            if changes >= 3 and (rel_home in (0, 7) or rel_local in (0, 7)):
                out.append({"start": int(s), "end": int(e), "midi": int(m),
                            "degree": {0: "I", 7: "V"}.get(rel_home, "local " + ("I" if rel_local == 0 else "V"))})
        s = max(e, s + 1)
    return out


def texture(voices, n_steps, bar_steps):
    """(n_bars, nv) bool: voice sounds at least a quarter of the bar."""
    snd, _ = _sounding(voices, n_steps)
    nb = (n_steps + bar_steps - 1) // bar_steps
    out = np.zeros((nb, len(voices)), dtype=bool)
    for b in range(nb):
        seg = snd[b * bar_steps:(b + 1) * bar_steps]
        out[b] = (seg >= 0).mean(axis=0) >= 0.25
    return out


# --------------------------------------------------------------------------- #
# Entries, stretto, countersubjects
# --------------------------------------------------------------------------- #
def entry_list(score: dict) -> tuple[list[dict], dict]:
    from .entries import COMPLETE, detect
    det = detect(score)
    if not det.get("subject"):
        return [], det
    L = len(det["subject"]) - 1
    slot = {v["voice_id"]: i for i, v in enumerate(
        sorted(score["voices"], key=lambda v: -float(v.get("mean_midi") or 0)))}
    out = []
    for e in det["entries"]:
        o = parse_frac(e["onset_q"])
        d = parse_frac(e["duration_q"])
        out.append({"voice": slot[e["voice_id"]], "voice_id": e["voice_id"],
                    "start": int(o * STEPS_PER_Q), "end": int((o + d) * STEPS_PER_Q),
                    "transpose": e["transpose"], "transformations": e["transformations"],
                    "complete": e["matched_intervals"] >= COMPLETE * L})
    out.sort(key=lambda x: (x["start"], x["voice"]))
    for i, a in enumerate(out):
        a["stretto"] = any(b is not a and b["complete"] and a["complete"] and b["voice"] != a["voice"]
                           and min(a["end"], b["end"]) - max(a["start"], b["start"]) >= STEPS_PER_Q
                           for b in out)
    return out, det


def _line(score_ev, a, b):
    """(rel onset, midi, diatonic step) of one voice's attacks in [a, b) steps."""
    return [(int(o * STEPS_PER_Q) - a, m, st) for o, d, m, st in score_ev
            if a <= int(o * STEPS_PER_Q) < b]


def _match_share(cand, line) -> float:
    """Share of the candidate's notes found in `line` at the same relative onset
    (within a sixteenth) and a constant diatonic offset (within one step)."""
    by_on = {}
    for r, m, st in line:
        by_on.setdefault(r, st)
    offs = []
    for r, m, st in cand:
        hit = by_on.get(r)
        if hit is None:
            hit = by_on.get(r + 1, by_on.get(r - 1))
        if hit is not None:
            offs.append(hit - st)
    if not offs:
        return 0.0
    mode = Counter(offs).most_common(1)[0][0]
    return sum(abs(x - mode) <= 1 for x in offs) / len(cand)


def countersubjects(score: dict, ents: list[dict], min_share: float = 0.6) -> list[dict]:
    """Countersubjects: lines that accompany entries and recur against later ones.

    CS1 is the first voice's line against the answer; CS2, the first voice's
    line against the third entry when that is not CS1 again. Each is then
    looked for in every other voice against every later plain entry."""
    from .entries import voice_events
    ev = voice_events(score)
    slot_vid = {i: v["voice_id"] for i, v in enumerate(
        sorted(score["voices"], key=lambda v: -float(v.get("mean_midi") or 0)))}
    plain = [e for e in ents if e["complete"] and not {"inverted", "augmented"} & set(e["transformations"])]
    if len(plain) < 3:
        return []
    first = plain[0]
    cands = []
    for k in (1, 2):
        if k < len(plain):
            c = _line(ev[slot_vid[first["voice"]]], plain[k]["start"], plain[k]["end"])
            if len(c) >= 4 and not any(_match_share(c0, c) >= min_share for _, c0 in cands):
                cands.append((k, c))
    out = []
    for idx, (k0, cand) in enumerate(cands, start=1):
        out.append({"index": idx, "voice": first["voice"], "start": plain[k0]["start"],
                    "end": plain[k0]["end"], "with_entry": k0, "share": 1.0})
        for k, e in enumerate(plain):
            if k <= k0:
                continue
            for u, vid in slot_vid.items():
                if u == e["voice"] or any(o["voice"] == u and o["with_entry"] == k for o in out):
                    continue
                line = _line(ev[vid], e["start"], e["end"])
                share = _match_share(cand, line) if len(line) >= 3 else 0.0
                if share >= min_share:
                    out.append({"index": idx, "voice": u, "start": e["start"], "end": e["end"],
                                "with_entry": k, "share": round(share, 2)})
    return out


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def sections(n_steps, nv, ents, cads, peds, seqs, tex, keys_unit, unit, bar_steps, tonic):
    covered = np.zeros(n_steps, dtype=bool)
    for e in ents:
        if e["complete"]:
            covered[e["start"]:e["end"]] = True
    # exposition: until every voice has had a complete entry
    seen, expo_end = set(), None
    for e in ents:
        if e["complete"]:
            seen.add(e["voice"])
            if len(seen) == nv:
                expo_end = e["end"]
                break
    spans = []
    s = 0
    while s < n_steps:
        e = s
        while e < n_steps and covered[e] == covered[s]:
            e += 1
        spans.append([bool(covered[s]), s, e])
        s = e
    # episodes shorter than a bar join the surrounding entry group
    merged = []
    for sp in spans:
        if not sp[0] and sp[2] - sp[1] < bar_steps and merged and sp[2] < n_steps:
            merged[-1][2] = sp[2]
            continue
        if merged and merged[-1][0] == sp[0]:
            merged[-1][2] = sp[2]
        else:
            merged.append(sp)
    last_entry_end = max((e["end"] for e in ents if e["complete"]), default=0)
    out = []
    for is_entry, a, b in merged:
        if expo_end and b <= expo_end:
            kind = "exposition"
        elif not is_entry and a >= last_entry_end:
            kind = "coda"
        else:
            kind = "entries" if is_entry else "episode"
        if out and out[-1]["type"] == kind == "exposition":
            out[-1]["end"] = b
            continue
        out.append({"type": kind, "start": int(a), "end": int(b)})
    if expo_end and out and out[0]["type"] != "exposition":
        out.insert(0, {"type": "exposition", "start": 0, "end": int(expo_end)})
    for sec in out:
        a, b = sec["start"], sec["end"]
        k0 = keys_unit[min(len(keys_unit) - 1, a // unit)]
        k1 = keys_unit[min(len(keys_unit) - 1, max(a, b - 1) // unit)]
        sec["bars"] = round((b - a) / bar_steps, 2)
        sec["key_start"] = cad_key_name(k0[0] - tonic, bool(k0[1]))
        sec["key_end"] = cad_key_name(k1[0] - tonic, bool(k1[1]))
        sec["entries"] = [{"voice": e["voice"], "level": e["transpose"] % 12,
                           "transformations": e["transformations"], "stretto": e["stretto"]}
                          for e in ents if e["complete"] and a <= e["start"] < b]
        sec["sequences"] = sum(1 for q in seqs if a <= q["start"] < b)
        sec["pedal"] = [p["degree"] for p in peds if a <= p["start"] < b]
        cs = [c for c in cads if a < c["step"] <= b + unit]
        sec["cadence"] = ({"type": cs[-1]["type"], "key_area": cs[-1]["key_area"]} if cs else None)
        tb = tex[a // bar_steps:max(a // bar_steps + 1, (b + bar_steps - 1) // bar_steps)]
        sec["voices_mean"] = round(float(tb.sum(axis=1).mean()), 2) if len(tb) else 0.0
    return out


# --------------------------------------------------------------------------- #
# One piece
# --------------------------------------------------------------------------- #
def label_piece(score: dict, mode: str, fugue: bool = True, grid: np.ndarray | None = None) -> dict:
    from .encode import encode_piece
    from .sequences import find_sequences
    mm = (score.get("meter_map") or [{"numerator": 4, "denominator": 4}])[0]
    meter = (mm["numerator"], mm["denominator"])
    km = (score.get("key_signature_map") or [{"fifths": 0}])[0]
    tonic = tonic_pc(km["fifths"], mode)
    voices = notes_from_score(score) if grid is None else notes_from_grid(grid)
    n_steps = int(parse_frac(score["quarter_length_total"]) * STEPS_PER_Q) if grid is None else len(grid)
    bar_steps = int(Fraction(4 * meter[0], meter[1]) * STEPS_PER_Q)
    beat_steps = int(_beat_q(meter) * STEPS_PER_Q)
    unit = harmonic_unit(meter)
    ch = chords(voices, n_steps, unit, bar_steps)
    cads = cadences(ch, voices, unit, bar_steps, meter, tonic)
    peds = pedals(voices, ch, unit, bar_steps, tonic)
    tex = texture(voices, n_steps, bar_steps)
    if grid is None:
        p = encode_piece(score, {"split": "x", "stratification_fields": {"mode": mode}}, None)
        grid = p["grid"] if p else None
    seqs = find_sequences(grid, beat_steps) if grid is not None else []
    ents, cs = [], []
    if fugue:
        ents, _ = entry_list(score)
        cs = countersubjects(score, ents)
    secs = sections(n_steps, len(voices), ents, cads, peds, seqs, tex,
                    [c["key"] for c in ch], unit, bar_steps, tonic) if fugue else []
    return {"piece_id": score["piece_id"], "meter": list(meter), "tonic": tonic, "mode": mode,
            "unit_steps": unit, "bar_steps": bar_steps, "n_steps": n_steps,
            "chords": [[c["root"], c["qual"], c["bass"]] for c in ch],
            "chord_keys": [list(c["key"]) for c in ch],
            "cadences": cads, "pedals": peds, "texture": tex.astype(int).tolist(),
            "entries": ents, "countersubjects": cs, "sequences": seqs, "sections": secs}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
_ROMAN = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5, "VII": 6}


def _numeral_pc(num: str, minor_ctx: bool) -> int:
    """Semitones above the tonic for a key numeral such as 'V', 'bVI', '#iv'."""
    acc = 0
    while num and num[0] in "b#":
        acc += -1 if num[0] == "b" else 1
        num = num[1:]
    deg = _ROMAN[num.upper()]
    scale = (0, 2, 3, 5, 7, 8, 10) if minor_ctx else MAJOR_SCALE
    return scale[deg] + acc


def _key_pc(name: str) -> tuple[int, bool]:
    base = TONICS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
    pc = base[name[0].lower()] + name[1:].count("#") - name[1:].count("b")
    return pc % 12, name[0].islower()


def _local_tonic(globalkey: str, localkey: str) -> tuple[int, bool]:
    g, gmin = _key_pc(globalkey)
    pc, minor = g, gmin
    for part in reversed(localkey.split("/")):
        core = part.lstrip("b#")
        pc = (pc + _numeral_pc(part, minor)) % 12
        minor = core.islower()
    return pc, minor


DCML_FAMILY = {"M": "maj", "Mm7": "maj", "MM7": "maj", "m": "min", "mm7": "min", "mM7": "min",
               "o": "dim", "%7": "dim", "o7": "dim", "+": "aug", "+7": "aug", "+M7": "aug"}


def validate_chords(root: Path, limit: int | None = None) -> dict:
    """Root / bass / quality agreement with DCML labels, per unit start."""
    rows = [json.loads(l) for l in (root / "manifests" / "baroque_manifest.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("harmonies") and r["status"] == "ok"]
    per_src = defaultdict(lambda: Counter())
    for r in rows[:limit]:
        score = json.loads((root / "data" / "canonical_baroque" / f"{r['piece_id']}.json").read_text())
        with (root / r["harmonies"]).open() as fh:
            hs = [h for h in csv.DictReader(fh, delimiter="\t")
                  if h["quarterbeats"] != "" and h.get("root") not in ("", None)
                  and h.get("chord_type") in DCML_FAMILY]
        if not hs:
            continue
        lab = label_piece(score, r["mode"], fugue=False)
        unit = lab["unit_steps"]
        starts = [int(Fraction(h["quarterbeats"]) * STEPS_PER_Q) for h in hs]
        c = per_src[r["source_name"]]
        k = 0
        for i, (rt, q, bs) in enumerate(lab["chords"]):
            t = i * unit
            while k + 1 < len(starts) and starts[k + 1] <= t:
                k += 1
            if starts[k] > t or rt is None:
                continue
            h = hs[k]
            try:
                lt, _ = _local_tonic(h["globalkey"], h["localkey"])
            except (KeyError, IndexError):
                continue
            g_root = (lt + 7 * int(float(h["root"]))) % 12
            g_bass = (lt + 7 * int(float(h["bass_note"]))) % 12
            c["n"] += 1
            c["root"] += rt == g_root
            c["bass"] += bs == g_bass
            c["family"] += rt == g_root and FAMILY[q] == DCML_FAMILY[h["chord_type"]]
    out = {}
    tot = Counter()
    for s, c in per_src.items():
        tot.update(c)
        out[s] = {k: round(c[k] / c["n"], 3) for k in ("root", "bass", "family")} | {"units": c["n"]}
    out["all"] = {k: round(tot[k] / max(1, tot["n"]), 3) for k in ("root", "bass", "family")} | {"units": tot["n"]}
    return out


def validate_algomus(root: Path) -> dict:
    """Cadence, pedal and countersubject detection against Algomus (WTC I)."""
    splits = {json.loads(l)["piece_id"]: json.loads(l) for l in
              (root / "manifests" / "splits.jsonl").read_text().splitlines() if l.strip()}
    res = Counter()
    for f in sorted((root / "data" / "annotations").glob("*.json")):
        pid = f.stem
        ann = [a for a in json.loads(f.read_text())["annotations"]
               if a.get("analytical_status") == "accepted"]
        score = json.loads((root / "data" / "canonical" / f"{pid}.json").read_text())
        lab = label_piece(score, splits[pid]["stratification_fields"]["mode"])
        tol = 2 * STEPS_PER_Q
        gold_c = [int(parse_frac(a["onset_q"]) * STEPS_PER_Q) for a in ann if a["type"] == "cadence"]
        pred_c = [c["step"] for c in lab["cadences"]]
        res["cad_gold"] += len(gold_c)
        res["cad_pred"] += len(pred_c)
        res["cad_hit_g"] += sum(any(abs(g - p) <= tol for p in pred_c) for g in gold_c)
        res["cad_hit_p"] += sum(any(abs(g - p) <= tol for g in gold_c) for p in pred_c)
        gold_p = [(int(parse_frac(a["onset_q"]) * STEPS_PER_Q),
                   int((parse_frac(a["onset_q"]) + parse_frac(a["duration_q"] or "0")) * STEPS_PER_Q))
                  for a in ann if a["type"] == "pedal"]
        pred_p = [(p["start"], p["end"]) for p in lab["pedals"]]
        ov = lambda x, y: min(x[1], y[1]) - max(x[0], y[0]) > 0
        res["ped_gold"] += len(gold_p)
        res["ped_pred"] += len(pred_p)
        res["ped_hit_g"] += sum(any(ov(g, p) for p in pred_p) for g in gold_p)
        res["ped_hit_p"] += sum(any(ov(g, p) for g in gold_p) for p in pred_p)
        slot = {v["voice_id"]: i for i, v in enumerate(
            sorted(score["voices"], key=lambda v: -float(v.get("mean_midi") or 0)))}
        gold_cs = [(slot.get(a["voice_id"]), int(parse_frac(a["onset_q"]) * STEPS_PER_Q),
                    int((parse_frac(a["onset_q"]) + parse_frac(a["duration_q"] or "0")) * STEPS_PER_Q))
                   for a in ann if a["type"] == "countersubject" and a.get("voice_id") in slot]
        pred_cs = [(c["voice"], c["start"], c["end"]) for c in lab["countersubjects"]]
        hit = lambda g, p: g[0] == p[0] and min(g[2], p[2]) - max(g[1], p[1]) >= 0.5 * (p[2] - p[1])
        res["cs_gold"] += len(gold_cs)
        res["cs_pred"] += len(pred_cs)
        res["cs_hit_g"] += sum(any(hit(g, p) for p in pred_cs) for g in gold_cs)
        res["cs_hit_p"] += sum(any(hit(g, p) for g in gold_cs) for p in pred_cs)
    pr = lambda k: {"precision": round(res[f"{k}_hit_p"] / max(1, res[f"{k}_pred"]), 3),
                    "recall": round(res[f"{k}_hit_g"] / max(1, res[f"{k}_gold"]), 3),
                    "gold": res[f"{k}_gold"], "pred": res[f"{k}_pred"]}
    return {"cadence": pr("cad"), "pedal": pr("ped"), "countersubject": pr("cs")}


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def corpus_scores(root: Path):
    """(score, mode, kind, split) for the original corpus and the Baroque expansion."""
    for line in (root / "manifests" / "splits.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        f = root / "data" / "canonical" / f"{r['piece_id']}.json"
        if f.exists():
            s = json.loads(f.read_text())
            yield s, r["stratification_fields"]["mode"], s.get("movement_type"), r["split"]
    bs = root / "manifests" / "baroque_splits.jsonl"
    if bs.exists():
        for line in bs.read_text().splitlines():
            r = json.loads(line)
            s = json.loads((root / "data" / "canonical_baroque" / f"{r['piece_id']}.json").read_text())
            yield s, r["stratification_fields"]["mode"], r["kind"], r["split"]


def _label_job(args):
    root, score, mode, kind = args
    try:
        lab = label_piece(score, mode, fugue=(kind == "fugue"))
    except Exception as e:  # noqa: BLE001
        return score["piece_id"], f"{type(e).__name__}: {e}"
    (root / "data" / "labels" / f"{score['piece_id']}.json").write_text(json.dumps(lab) + "\n")
    return score["piece_id"], None


def main(argv=None) -> int:
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--jobs", type=int, default=24)
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    if args.validate:
        print(json.dumps({"algomus": validate_algomus(root), "dcml_chords": validate_chords(root)},
                         indent=1))
        return 0
    (root / "data" / "labels").mkdir(parents=True, exist_ok=True)
    jobs = [(root, s, m, k) for s, m, k, _ in corpus_scores(root)]
    with Pool(args.jobs) as pool:
        res = pool.map(_label_job, jobs, chunksize=2)
    bad = [(p, e) for p, e in res if e]
    print(f"labelled {len(res) - len(bad)} pieces, {len(bad)} failed")
    for p, e in bad[:10]:
        print(" ", p, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
