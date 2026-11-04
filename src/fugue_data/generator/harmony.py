"""Harmony planner: a chord for every harmonic unit, given the form plan.

    PYTHONPATH=src python -m fugue_data.generator.harmony fit      # -> data/models/harmony.json

A hidden Markov model over chords relative to the local key:

- **States**: (root relative to the local tonic, quality, inversion), with
  quality in labels.QUAL_NAMES and inversion 0 (root in the bass) or 1 (third
  in the bass); state "N" for no chord.
- **Transitions**: counted from the chord labels (labels.py) of every training
  piece, separately for major and minor local keys, with add-k smoothing. A
  chord repeating is a transition too, so harmonic rhythm comes with it.
- **Emissions**: notes the plan fixes (subject entries) should be chord tones
  of the unit they start in: a fixed note that is a chord tone scores 0, one
  that is not scores log(EMIT_NCT).
- **Constraints**: cadences fix their last chords (PAC: predominant -> V(7) ->
  I; HC: -> V), sequences fix their roots (descending fifths, rising 5-6,
  falling steps), pedals restrict chords to those that sound acceptable over
  the pedal tone.

Chords are drawn by forward filtering / backward sampling, so the progression
is a sample from the model given the constraints, not one fixed answer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .labels import QUALS, QUAL_NAMES

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "data" / "models" / "harmony.json"
EMIT_NCT = 0.15
STATES = [(r, q, inv) for r in range(12) for q in QUAL_NAMES for inv in (0, 1)] + [None]
SIDX = {s: i for i, s in enumerate(STATES)}


def state_of(root, qual, bass, tonic):
    if root is None:
        return None
    r = (root - tonic) % 12
    inv = 1 if bass is not None and (bass - root) % 12 in (3, 4) else 0
    return (r, qual, inv)


def fit(out: Path = MODEL, k: float = 0.05) -> dict:
    splits = {}
    for f in ("splits.jsonl", "baroque_splits.jsonl"):
        for line in (ROOT / "manifests" / f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                splits[r["piece_id"]] = r["split"]
    S = len(STATES)
    counts = {"major": np.full((S, S), k), "minor": np.full((S, S), k)}
    init = {"major": np.full(S, k), "minor": np.full(S, k)}
    n_pieces = 0
    for f in sorted((ROOT / "data" / "labels").glob("*.json")):
        if splits.get(f.stem) != "train":
            continue
        lab = json.loads(f.read_text())
        ch, keys = lab["chords"], lab["chord_keys"]
        prev = None
        for (root, q, bass), (kt, minor) in zip(ch, keys):
            m = "minor" if minor else "major"
            st = SIDX[state_of(root, q, bass, kt)]
            if prev is None:
                init[m][st] += 1
            else:
                counts[m][prev, st] += 1
            prev = st
        n_pieces += 1
    model = {"states": [list(s) if s else None for s in STATES], "pieces": n_pieces}
    for m in ("major", "minor"):
        T = counts[m] / counts[m].sum(1, keepdims=True)
        model[m] = {"log_trans": np.log(T).round(4).tolist(),
                    "log_init": np.log(init[m] / init[m].sum()).round(4).tolist()}
    out.write_text(json.dumps(model))
    return model


_M = None


def load():
    global _M
    if _M is None:
        m = json.loads(MODEL.read_text())
        for mode in ("major", "minor"):
            m[mode]["log_trans"] = np.array(m[mode]["log_trans"])
            m[mode]["log_init"] = np.array(m[mode]["log_init"])
        _M = m
    return _M


def chord_pcs(state, tonic):
    r, q, inv = state
    return {(tonic + r + t) % 12 for t in QUALS[q]}


# Chords at home in a key: the diatonic ones plus the chromatic chords Bach
# uses routinely (secondary dominants and leading-tone chords, the
# Neapolitan; in minor the major V and the raised-6th chords of the melodic
# minor). (root relative to the tonic, qualities). Others are penalised by
# OUT_OF_KEY so they appear only when a constraint asks for them: the chord
# labels the model was fitted on are ~75% accurate, and sampling from the raw
# table produced E minor in G minor and B major in E-flat major.
VOCAB = {
    "major": {0: ("maj", "maj7", "dom7"), 2: ("min", "min7", "maj", "dom7"),
              4: ("min", "min7", "maj", "dom7"), 5: ("maj", "maj7", "min"),
              7: ("maj", "dom7"), 9: ("min", "min7", "maj", "dom7"),
              11: ("dim", "hdim7", "dim7"), 1: ("maj",), 6: ("dim", "dim7", "hdim7"),
              8: ("dim7",)},
    "minor": {0: ("min", "maj", "dom7"), 2: ("dim", "hdim7", "maj", "dom7", "min"),
              3: ("maj", "maj7"), 5: ("min", "min7", "maj"), 7: ("maj", "dom7", "min"),
              8: ("maj", "maj7"), 10: ("maj", "dom7"), 11: ("dim", "dim7", "hdim7"),
              1: ("maj",), 6: ("dim", "dim7", "hdim7"), 9: ("dim", "hdim7")},
}
OUT_OF_KEY = -6.0
MINOR_V = -1.5            # v (minor) as the dominant of a minor key: Bach uses V
_PRIOR = {}


def key_prior(minor: bool) -> np.ndarray:
    if minor not in _PRIOR:
        voc = VOCAB["minor" if minor else "major"]
        pr = np.zeros(len(STATES))
        for i, st in enumerate(STATES):
            if st is None:
                continue
            r, q, _ = st
            if q not in voc.get(r, ()):
                pr[i] = OUT_OF_KEY
            elif minor and r == 7 and q == "min":
                pr[i] = MINOR_V
        _PRIOR[minor] = pr
    return _PRIOR[minor]


def plan_chords(n_units: int, keys: list, fixed_notes: list, allowed: list | None,
                rng: np.random.Generator, temperature: float = 1.0) -> list:
    """Sample one state per unit.

    keys[u] = (tonic_pc, minor); fixed_notes[u] = pitch classes the plan fixes
    at unit u (may be empty); allowed[u] = None or a set of state indices the
    unit must take (cadences, sequences, pedals)."""
    m = load()
    S = len(STATES)
    emit = np.zeros((n_units, S))
    cache = {}
    for u in range(n_units):
        kt, minor = keys[u]
        for i, st in enumerate(STATES):
            if st is None:
                emit[u, i] = -20.0 if fixed_notes[u] else -3.0
                continue
            key = (kt, st)
            if key not in cache:
                cache[key] = chord_pcs(st, kt)
            pcs = cache[key]
            for pc in fixed_notes[u]:
                if pc not in pcs:
                    emit[u, i] += np.log(EMIT_NCT)
        emit[u] += key_prior(bool(minor))
        if allowed is not None and allowed[u] is not None:
            mask = np.full(S, -np.inf)
            mask[list(allowed[u])] = 0
            emit[u] += mask
    # Transitions are relative to the local key; across a key change, map the
    # previous state into the new key's frame by its absolute root.
    def trans(u):
        kt, minor = keys[u]
        T = m["minor" if minor else "major"]["log_trans"] / temperature
        if u > 0 and keys[u - 1][0] != kt:
            shift = (keys[u - 1][0] - kt) % 12
            perm = np.arange(S)
            for i, st in enumerate(STATES):
                if st is not None:
                    perm[i] = SIDX[((st[0] + shift) % 12, st[1], st[2])]
            out = np.full((S, S), -np.inf)
            out[np.arange(S)] = T[perm]
            return out
        return T
    alpha = np.zeros((n_units, S))
    kt0, minor0 = keys[0]
    alpha[0] = m["minor" if minor0 else "major"]["log_init"] + emit[0]
    Ts = [None] + [trans(u) for u in range(1, n_units)]
    for u in range(1, n_units):
        a = alpha[u - 1][:, None] + Ts[u]
        mx = a.max(0)
        alpha[u] = mx + np.log(np.exp(a - mx).sum(0) + 1e-300) + emit[u]
    states = [0] * n_units
    p = np.exp(alpha[-1] - alpha[-1].max())
    states[-1] = int(rng.choice(S, p=p / p.sum()))
    for u in range(n_units - 2, -1, -1):
        lp = alpha[u] + Ts[u + 1][:, states[u + 1]]
        p = np.exp(lp - lp.max())
        states[u] = int(rng.choice(S, p=p / p.sum()))
    return [STATES[s] for s in states]


def state_ids(r: int, quals, invs=(0, 1)) -> set:
    return {SIDX[(r % 12, q, i)] for q in quals for i in invs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit"])
    ap.parse_args(argv)
    m = fit()
    print(f"fitted on {m['pieces']} training pieces -> {MODEL}")
    T = np.exp(np.array(m["major"]["log_trans"]))
    I = SIDX[(0, "maj", 0)]
    top = np.argsort(T[SIDX[(7, "dom7", 0)]])[::-1][:6]
    print("major: after V7 ->", [(STATES[t], round(float(T[SIDX[(7, 'dom7', 0)], t]), 3)) for t in top])
    top = np.argsort(T[SIDX[(5, "maj", 0)]])[::-1][:6]
    print("major: after IV ->", [(STATES[t], round(float(T[SIDX[(5, 'maj', 0)], t]), 3)) for t in top])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
