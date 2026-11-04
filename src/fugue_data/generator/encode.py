"""Encode canonical scores into token + feature arrays for the generator.

Layout of one piece, time-major:  step 0 voice 1, step 0 voice 2, ..., step 1 ...
Voices are ordered highest first by mean pitch. Every music position carries:

  time features (always known)       plan features (0 = unknown)
  - voice slot                       - this voice's role now: free / subject /
  - step within the beat               answer / countersubject
  - beat within the bar              - steps to the next entry (any voice)
  - bar index                        - steps to the next cadence
  - progress through the piece       - key area of the next cadence
                                     - pedal point active

Plan features come from aligned structural labels (Algomus, WTC I). Pieces
without labels carry 0 ("unknown") throughout; the model learns both regimes,
and at training time labelled pieces are randomly dropped to unknown as well
(plan dropout), so plan-free generation and plan-following generation are one
model.

A prefix precedes the music: corpus, voice count, meter, mode, tonic, length,
whether a plan is present, and -- when the subject is known -- the subject itself
as a single-voice line, so the theme stays visible after it scrolls out of the
context window.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..normalize.rational import parse_frac
from ..review.propose import measure_table
from ..tokenization.views import merge_tied

STEPS_PER_Q = 4                       # sixteenth-note grid
MIDI_LO, MIDI_HI = 21, 108

SPECIAL = ["<PAD>", "<BOS>", "<SEP>", "REST", "HOLD"]
NOTES = [f"NOTE_ON_{m}" for m in range(MIDI_LO, MIDI_HI + 1)]
META = (["CORPUS_fugue", "CORPUS_chorale"] + [f"NV_{k}" for k in range(1, 7)]
        + ["MODE_major", "MODE_minor"] + [f"TONIC_{k}" for k in range(12)]
        + [f"LEN_{k}" for k in range(8)] + ["PLAN_yes", "PLAN_no",
                                            "SUBJ", "/SUBJ"])
METERS = ["2_2", "2_4", "3_2", "3_4", "3_8", "4_2", "4_4", "6_4", "6_8", "6_16",
          "9_8", "12_8", "12_16", "other"]
# CORPUS_other (Baroque pieces that are neither fugue nor chorale) is appended
# last so token ids of older checkpoints stay valid.
VOCAB = SPECIAL + NOTES + META + [f"METER_{m}" for m in METERS] + ["CORPUS_other"]
STOI = {t: i for i, t in enumerate(VOCAB)}
NOTE0 = STOI[NOTES[0]]

# Feature tables: name -> number of embedding rows (index 0 = n/a or unknown).
FEATURES = {"voice": 7, "sub": 17, "beat": 17, "bar": 257, "prog": 33,
            "role": 6, "to_entry": 17, "to_cad": 17, "cad_key": 16, "pedal": 3,
            "seg": 4, "pnote": 90, "epc": 13, "key": 25,
            "next_key": 25, "to_key": 17, "seq": 10,
            "chord": 110, "next_chord": 110, "bass": 14, "active": 3}
# chord / next_chord: 0 unknown, 1 silence, 2 + 12 * quality + root (relative
# to the home tonic; qualities as labels.QUAL_NAMES). bass: 0 unknown, 1 none,
# 2 + pitch class relative to the tonic. active (per voice): 0 unknown, 1 the
# voice rests this bar, 2 it sounds (labels.texture).
ROLE = {"free": 1, "subject": 2, "answer": 3, "countersubject": 4}
SEG = {"prefix": 1, "subject": 2, "music": 3}
CAD_KEYS = ["I", "i", "II", "ii", "III", "iii", "IV", "iv", "V", "v", "VI", "vi",
            "VII", "vii", "other"]


def _bucket(steps: int) -> int:
    """Distance in steps -> 1..16, roughly logarithmic (beyond 16 bars -> 16)."""
    if steps < 0:
        return 0
    edges = [0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    for k, e in enumerate(edges):
        if steps <= e:
            return k + 1
    return 16


def tonic_pc(fifths: int, mode: str) -> int:
    major = (7 * fifths) % 12
    return major if mode == "major" else (major + 9) % 12


def _meter_tok(meter: tuple[int, int]) -> str:
    key = f"{meter[0]}_{meter[1]}"
    return f"METER_{key if key in METERS else 'other'}"


def _beat_q(meter: tuple[int, int]) -> Fraction:
    num, den = meter
    unit = Fraction(4, den)
    return unit * 3 if den >= 8 and num % 3 == 0 and num > 3 else unit


def local_keys(grid: np.ndarray, bar_steps: int, tonic: int) -> np.ndarray:
    """Per-step local key relative to the home tonic: 1 + 2*degree + minor.

    Krumhansl-Schmuckler over a three-bar window centred on each bar, weighted
    by sounding duration. Noisy at the edges of a modulation, but it is what
    makes "go to the relative major here" expressible as conditioning.
    """
    from ..corruption.windows import _KK_MAJOR, _KK_MINOR, _corr
    n = len(grid)
    pitch = np.full(grid.shape, -1)
    for v in range(grid.shape[1]):
        cur = -1
        for s in range(n):
            t = int(grid[s, v])
            if NOTE0 <= t < NOTE0 + 88:
                cur = MIDI_LO + t - NOTE0
            elif t == STOI["REST"]:
                cur = -1
            pitch[s, v] = cur
    out = np.zeros(n, dtype=np.int16)
    bar_steps = max(1, bar_steps)
    for b0 in range(0, n, bar_steps):
        w = pitch[max(0, b0 - bar_steps):b0 + 2 * bar_steps]
        h = np.bincount(w[w >= 0] % 12, minlength=12).astype(float)
        if h.sum() == 0:
            continue
        best = (-2.0, 0)
        for t in range(12):
            rot = list(h[t:]) + list(h[:t])
            for mi, prof in enumerate((_KK_MAJOR, _KK_MINOR)):
                r = _corr(rot, prof)
                if r > best[0]:
                    best = (r, 1 + 2 * ((t - tonic) % 12) + mi)
        out[b0:b0 + bar_steps] = best[1]
    return out


def key_lookahead(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each step: the next key region's key, and steps until it starts.

    The current key alone tells the model nothing it can't hear; what it needs
    in order to prepare a modulation is warning that one is coming.
    """
    n = len(key)
    nxt = np.zeros(n, dtype=np.int16)
    dist = np.full(n, 16, dtype=np.int16)
    k_next, s_next = 0, None
    for s in range(n - 1, -1, -1):
        if s + 1 < n and key[s + 1] != key[s]:
            k_next, s_next = int(key[s + 1]), s + 1
        if s_next is not None:
            nxt[s] = k_next
            dist[s] = _bucket(s_next - s)
    return nxt, dist


def sequence_feature(grid: np.ndarray, beat_steps: int) -> np.ndarray:
    """(n_steps, nv): 1 = not in a sequence; 2-5 = first statement of a unit of
    1-4 beats; 6-9 = a restatement of such a unit (copy it, transposed)."""
    from .sequences import find_sequences
    out = np.ones(grid.shape, dtype=np.int16)
    for q in find_sequences(grid, beat_steps):
        a, D, v = q["start"], q["unit"], q["voice"]
        beats = min(4, max(1, D // max(1, beat_steps)))
        out[a:a + D, v] = 1 + beats
        out[a + D:a + q["statements"] * D, v] = 5 + beats
    return out


def label_arrays(lab: dict, n_steps: int, nv: int, tonic: int) -> dict:
    """Per-step plan arrays from a `labels.label_piece` result: chords, bass,
    texture, and detected cadences and pedals."""
    from .labels import QUAL_NAMES
    unit, bar = lab["unit_steps"], lab["bar_steps"]
    codes, basses = [], []
    for r, q, b in lab["chords"]:
        codes.append(1 if r is None else 2 + 12 * QUAL_NAMES.index(q) + (r - tonic) % 12)
        basses.append(1 if b is None else 2 + (b - tonic) % 12)
    codes = np.array(codes + [1], dtype=np.int16)
    basses = np.array(basses + [1], dtype=np.int16)
    idx = np.minimum(np.arange(n_steps) // unit, len(codes) - 2)
    chord = codes[idx]
    # the next different chord, so the model can prepare its voice-leading
    nxt = np.empty(len(codes), dtype=np.int16)
    cur = 1
    for i in range(len(codes) - 1, -1, -1):
        nxt[i] = cur
        if i and codes[i] != codes[i - 1]:
            cur = codes[i]
    active = np.zeros((n_steps, nv), dtype=np.int16)
    tex = np.array(lab["texture"], dtype=np.int16).reshape(-1, nv) if lab["texture"] else None
    if tex is not None and tex.shape[1] == nv:
        bidx = np.minimum(np.arange(n_steps) // bar, len(tex) - 1)
        active[:] = 1 + tex[bidx]
    to_cad = np.full(n_steps, 16, dtype=np.int16)
    cad_key = np.zeros(n_steps, dtype=np.int16)
    cs = sorted((c["step"], CAD_KEYS.index(c["key_area"]) + 1) for c in lab["cadences"])
    ci = 0
    for k in range(n_steps):
        while ci < len(cs) and cs[ci][0] < k:
            ci += 1
        if ci < len(cs):
            to_cad[k] = _bucket(cs[ci][0] - k)
            cad_key[k] = cs[ci][1]
    pedal = np.ones(n_steps, dtype=np.int16)
    for q in lab["pedals"]:
        pedal[q["start"]:q["end"]] = 2
    return {"chord": chord, "next_chord": nxt[idx], "bass": basses[idx], "active": active,
            "to_cad": to_cad, "cad_key": cad_key, "pedal": pedal}


def encode_piece(score: dict, split_row: dict, annotations: list[dict] | None,
                 labels: dict | None = None):
    """Arrays for one piece. Returns None if it cannot be represented.

    `labels` (data/labels/<piece>.json) adds the chord/texture plan, and the
    detected cadences and pedals where no expert labels exist."""
    mm = (score.get("meter_map") or [{"numerator": 4, "denominator": 4}])[0]
    meter = (mm["numerator"], mm["denominator"])
    bar_q = Fraction(4 * meter[0], meter[1])
    beat_q = _beat_q(meter)
    total = parse_frac(score["quarter_length_total"])
    n_steps = int(total * STEPS_PER_Q)
    if n_steps <= 0:
        return None
    voices = sorted(score["voices"], key=lambda v: -float(v.get("mean_midi") or 0))
    nv = len(voices)
    if nv > 6:
        return None

    # --- notes onto the grid ------------------------------------------------ #
    grid = np.full((n_steps, nv), STOI["REST"], dtype=np.int16)
    dropped = 0
    vid_to_slot = {}
    for vi, v in enumerate(voices):
        vid_to_slot[v["voice_id"]] = vi
        notes = [n for n in v["notes"] if not n.get("grace")
                 and parse_frac(n["duration_q"]) > 0]
        taken = np.zeros(n_steps, dtype=bool)
        for o, d, m in merge_tied(notes):
            s = int(round(o * STEPS_PER_Q))
            e = max(s + 1, int(round((o + d) * STEPS_PER_Q)))
            if s >= n_steps or taken[s] or not (MIDI_LO <= m <= MIDI_HI):
                dropped += 1          # finer than a 16th, or a divisi second note
                continue
            e = min(e, n_steps)
            # A later note may already own steps inside [s, e): stop before it.
            nxt = np.flatnonzero(taken[s:e])
            if len(nxt):
                e = s + int(nxt[0])
            grid[s, vi] = NOTE0 + (m - MIDI_LO)
            grid[s + 1:e, vi] = STOI["HOLD"]
            taken[s:e] = True

    # --- time features ------------------------------------------------------- #
    bars = measure_table(score)
    first = 1 if len(bars) > 1 and bars[1][2] - bars[0][2] < bar_q else 0
    ref = bars[first][2] if bars else Fraction(0)
    t = np.arange(n_steps)
    q = [Fraction(int(k), STEPS_PER_Q) for k in t]
    rel = [x - ref for x in q]
    bar_idx = np.array([int((r // bar_q)) + (1 if first else 0) for r in rel])
    in_bar = [r % bar_q for r in rel]
    beat = np.array([int(x // beat_q) for x in in_bar])
    sub = np.array([int((x % beat_q) * STEPS_PER_Q) for x in in_bar])
    prog = (t * 32) // n_steps

    feats = {
        "sub": np.clip(sub + 1, 1, 16), "beat": np.clip(beat + 1, 1, 16),
        "bar": np.clip(bar_idx + 1, 1, 256), "prog": np.clip(prog + 1, 1, 32),
    }

    # --- plan features ------------------------------------------------------ #
    role = np.zeros((n_steps, nv), dtype=np.int16)
    to_entry = np.zeros(n_steps, dtype=np.int16)
    to_cad = np.zeros(n_steps, dtype=np.int16)
    cad_key = np.zeros(n_steps, dtype=np.int16)
    pedal = np.zeros(n_steps, dtype=np.int16)
    subject_line = None
    has_plan = False
    pnote = np.zeros((n_steps, nv), dtype=np.int16)
    epc = np.zeros((n_steps, nv), dtype=np.int16)
    mode = split_row["stratification_fields"]["mode"]
    km = (score.get("key_signature_map") or [{"fifths": 0}])[0]
    tonic_here = tonic_pc(km["fifths"], mode)
    acc = [a for a in (annotations or [])
           if a.get("analytical_status") in (None, "accepted")]
    if acc:
        has_plan = True
        role[:] = ROLE["free"]
        pedal[:] = 1
        entries, cads = [], []
        for a in acc:
            s = int(round(parse_frac(a["onset_q"]) * STEPS_PER_Q))
            d = int(round(parse_frac(a["duration_q"]) * STEPS_PER_Q)) if a.get("duration_q") else 0
            if a["type"] in ("subject", "countersubject") and a.get("voice_id") in vid_to_slot:
                r = (ROLE["countersubject"] if a["type"] == "countersubject"
                     else ROLE["answer"] if a.get("entry_function") == "answer"
                     else ROLE["subject"])
                role[max(0, s):min(n_steps, s + d), vid_to_slot[a["voice_id"]]] = r
                if a["type"] == "subject":
                    entries.append((s, a))
            elif a["type"] == "cadence":
                k = a.get("cadence_key_area") or "other"
                cads.append((s, CAD_KEYS.index(k) + 1 if k in CAD_KEYS else len(CAD_KEYS)))
            elif a["type"] == "pedal":
                pedal[max(0, s):min(n_steps, s + d)] = 2
        es = sorted(s for s, _ in entries)
        cs = sorted(cads)
        ei = ci = 0
        for k in range(n_steps):
            while ei < len(es) and es[ei] < k:
                ei += 1
            while ci < len(cs) and cs[ci][0] < k:
                ci += 1
            to_entry[k] = _bucket(es[ei] - k) if ei < len(es) else 16
            if ci < len(cs):
                to_cad[k] = _bucket(cs[ci][0] - k)
                cad_key[k] = cs[ci][1]
            else:
                to_cad[k] = 16
        # Entry notes as features: the entering voice's own planned note, and
        # for every other voice the pitch class the entry is sounding then.
        sounding = np.full((n_steps, nv), -1)
        for v in range(nv):
            cur = -1
            for s in range(n_steps):
                t = int(grid[s, v])
                if NOTE0 <= t < NOTE0 + 88:
                    cur = t - NOTE0
                elif t == STOI["REST"]:
                    cur = -1
                sounding[s, v] = cur
        for s, a in entries:
            v = vid_to_slot[a["voice_id"]]
            d = int(round(parse_frac(a["duration_q"]) * STEPS_PER_Q))
            for k in range(max(0, s), min(n_steps, s + d)):
                t = int(grid[k, v])
                if NOTE0 <= t < NOTE0 + 88:
                    pnote[k, v] = 2 + t - NOTE0
                elif t == STOI["HOLD"]:
                    pnote[k, v] = 1
                if sounding[k, v] >= 0:
                    pc = 1 + (MIDI_LO + sounding[k, v] - tonic_here) % 12
                    for u in range(nv):
                        if u != v and epc[k, u] == 0:
                            epc[k, u] = pc
        if entries:
            s0, a0 = min(entries, key=lambda x: x[0])
            d0 = int(round(parse_frac(a0["duration_q"]) * STEPS_PER_Q))
            col = grid[s0:s0 + d0, vid_to_slot[a0["voice_id"]]].copy()
            if len(col) and len(col) <= 128:
                subject_line = col

    lab = None
    if labels is not None and labels.get("n_steps"):
        lab = label_arrays(labels, n_steps, nv, tonic_here)
        expert_cads = any(a["type"] == "cadence" for a in acc)
        if not expert_cads:              # no expert cadences / pedals: use detected
            to_cad, cad_key, pedal = lab["to_cad"], lab["cad_key"], lab["pedal"]
    key = local_keys(grid, int(bar_q * STEPS_PER_Q), tonic_here)
    next_key, to_key = key_lookahead(key)
    seq = sequence_feature(grid, int(beat_q * STEPS_PER_Q))
    return {
        "piece_id": score["piece_id"], "split": split_row["split"],
        "corpus": {"fugue": "fugue", "chorale": "chorale"}.get(
            score.get("movement_type"), "other" if split_row.get("kind") else "chorale"),
        "nv": nv, "meter_tok": _meter_tok(meter), "mode": mode,
        "tonic": tonic_pc(km["fifths"], mode), "n_steps": n_steps,
        "grid": grid, "feats": feats, "role": role, "to_entry": to_entry,
        "to_cad": to_cad, "cad_key": cad_key, "pedal": pedal,
        "has_plan": has_plan, "subject": subject_line, "dropped_notes": dropped,
        "pnote": pnote, "epc": epc,
        "key": key, "next_key": next_key, "to_key": to_key, "seq": seq,
        "chord": lab["chord"] if lab else np.zeros(n_steps, dtype=np.int16),
        "next_chord": lab["next_chord"] if lab else np.zeros(n_steps, dtype=np.int16),
        "bass": lab["bass"] if lab else np.zeros(n_steps, dtype=np.int16),
        "active": lab["active"] if lab else np.zeros((n_steps, nv), dtype=np.int16),
        "cad_from": ("labels" if any(a["type"] == "cadence" for a in acc)
                     else "detected" if lab else None),
        "plan_source": ("labels" if has_plan and not any(
            a.get("source_name") == "detected" for a in acc) else
            "detected" if has_plan else None),
    }


def prefix_tokens(p: dict, tonic: int, plan: bool) -> list[int]:
    length_bucket = min(7, p["n_steps"] // 512)
    toks = ["<BOS>", f"CORPUS_{p['corpus']}", f"NV_{p['nv']}", p["meter_tok"],
            f"MODE_{p['mode']}", f"TONIC_{tonic}", f"LEN_{length_bucket}",
            "PLAN_yes" if plan else "PLAN_no"]
    return [STOI[t] for t in toks]


def load_corpus(root: Path, detect_entries: bool = True, baroque: bool = False,
                labels: bool = False, exclude_sources: tuple = ()) -> list[dict]:
    """All pieces. Fugues without labels get entries from the detector.

    baroque: also the expanded corpus (manifests/baroque_splits.jsonl), pieces
    whose voices are clean enough for the note model. labels: attach the
    chord / texture / cadence / pedal plan from data/labels. exclude_sources:
    Baroque source names to leave out (manifests/baroque_splits.jsonl `corpus`)."""
    from .entries import as_annotations, detect
    rows = []
    for line in (root / "manifests" / "splits.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows.append((root / "data" / "canonical" / f"{r['piece_id']}.json", r))
    if baroque:
        for line in (root / "manifests" / "baroque_splits.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["note_model"] and r["corpus"] not in exclude_sources:
                rows.append((root / "data" / "canonical_baroque" / f"{r['piece_id']}.json", r))
    out = []
    for f, row in rows:
        if not f.exists():
            continue
        score = json.loads(f.read_text())
        ap = root / "data" / "annotations" / f"{score['piece_id']}.json"
        ann = json.loads(ap.read_text())["annotations"] if ap.exists() else None
        if ann is None and score.get("movement_type") == "fugue" and detect_entries:
            ann = as_annotations(detect(score), score["piece_id"]) or None
        lab = None
        lp = root / "data" / "labels" / f"{score['piece_id']}.json"
        if labels and lp.exists():
            lab = json.loads(lp.read_text())
        p = encode_piece(score, row, ann, lab)
        if p is not None:
            out.append(p)
    return out
