"""Find melodic sequences on the note grid (R4).

A sequence here is one voice stating a short unit, then restating it at least
twice more, each time transposed by the same step: the standard episode device.
Works on the token grid, so the same detector measures Bach and generated
output.

For a unit of D steps (one to four beats) starting at an attack:
- the unit has at least three notes and is not a plain scale run (it must turn
  direction or leap);
- each restatement has the same rhythm (identical relative onsets);
- its intervals are within a semitone of the unit's (a diatonic sequence keeps
  the steps but changes some interval qualities);
- restatement k sits k * T semitones from the unit (within one semitone), with
  T != 0.

Detection is per voice; longer units are tried first and a step covered by a
sequence is not reused for a shorter one.
"""

from __future__ import annotations

import numpy as np

from .encode import MIDI_LO, NOTE0, STOI

N_NOTES = 88
UNIT_BEATS = (4, 3, 2, 1)
MIN_STATEMENTS = 3

# Per-voice feature values (0 = unknown / no plan)
NONE, FIRST, REPEAT = 1, 2, 3


def _notes(col: np.ndarray) -> list[tuple[int, int]]:
    return [(s, MIDI_LO + int(t) - NOTE0) for s, t in enumerate(col)
            if NOTE0 <= t < NOTE0 + N_NOTES]


def _unit(notes, onsets, t, D):
    i0 = np.searchsorted(onsets, t)
    i1 = np.searchsorted(onsets, t + D)
    return notes[i0:i1]


def _shape(u, t):
    rel = tuple(s - t for s, _ in u)
    ivs = [b[1] - a[1] for a, b in zip(u, u[1:])]
    return rel, ivs


def _is_scale_run(ivs) -> bool:
    return all(1 <= abs(i) <= 2 for i in ivs) and (all(i > 0 for i in ivs) or all(i < 0 for i in ivs))


def find_sequences(grid: np.ndarray, beat_steps: int) -> list[dict]:
    """[{voice, start, unit, statements, step}] in grid steps."""
    out = []
    n, nv = grid.shape
    for v in range(nv):
        notes = _notes(grid[:, v])
        if len(notes) < 9:
            continue
        onsets = np.array([s for s, _ in notes])
        used = np.zeros(n, dtype=bool)
        for beats in UNIT_BEATS:
            D = beats * beat_steps
            for t, _ in notes:
                if used[t] or t + MIN_STATEMENTS * D > n:
                    continue
                u = _unit(notes, onsets, t, D)
                if len(u) < 3 or u[0][0] != t:
                    continue
                rel, ivs = _shape(u, t)
                if _is_scale_run(ivs) or all(i == 0 for i in ivs):
                    continue
                k, T = 1, None
                while t + (k + 1) * D <= n:
                    w = _unit(notes, onsets, t + k * D, D)
                    rel_w, ivs_w = _shape(w, t + k * D)
                    if rel_w != rel or any(abs(a - b) > 1 for a, b in zip(ivs, ivs_w)):
                        break
                    shift = w[0][1] - u[0][1]
                    if T is None:
                        if shift == 0:
                            break
                        T = shift
                    elif abs(shift - k * T) > 1:
                        break
                    k += 1
                if k >= MIN_STATEMENTS:
                    out.append({"voice": v, "start": t, "unit": D, "statements": k, "step": T})
                    used[t:t + k * D] = True
    return sorted(out, key=lambda x: (x["start"], x["voice"]))


def sequence_roles(grid: np.ndarray, beat_steps: int) -> np.ndarray:
    """(n_steps, nv): NONE, or FIRST / REPEAT inside a detected sequence."""
    roles = np.full(grid.shape, NONE, dtype=np.int16)
    for q in find_sequences(grid, beat_steps):
        a, D = q["start"], q["unit"]
        roles[a:a + D, q["voice"]] = FIRST
        roles[a + D:a + q["statements"] * D, q["voice"]] = REPEAT
    return roles


def coverage(grid: np.ndarray, beat_steps: int) -> float:
    """Share of steps where at least one voice is inside a sequence."""
    r = sequence_roles(grid, beat_steps)
    return float((r != NONE).any(axis=1).mean())


__all__ = ["find_sequences", "sequence_roles", "coverage", "STOI"]
