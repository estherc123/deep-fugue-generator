"""How much of a piece is copied from the training set.

A line is a sequence of (interval, inter-onset time) pairs: transposition-
invariant, rhythm-aware. For every note of a generated voice we find the
longest run, starting there, that also occurs in some training voice at any
transposition. Two numbers summarise a piece:

- `longest_run`: the longest shared run, in intervals.
- `copied_share`: the share of notes inside a shared run of >= `run` intervals.

Neither means anything alone: Bach shares idioms with Bach. So the same numbers
are computed for held-out Bach against the training set, and a generated piece
is judged against that baseline. Copying is generated output sharing *more*
than real unseen Bach does.

The subject is excluded from a generated piece's count when it is given: its
entries are the input, written in by the plan, not the model's choice.

    PYTHONPATH=src python -m fugue_data.generator.memorization \\
        data/review/generator_samples/bach_wtc1_fugue_02/generated_1.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .encode import MIDI_LO, NOTE0, STOI, load_corpus

N_NOTES = 88
MAX_N = 64
REST = STOI["REST"]


def lines(grid: np.ndarray, skip: np.ndarray | None = None) -> list[list[tuple]]:
    """Per voice: list of (interval, ioi, counted) between consecutive notes.

    A rest ends a line. `skip` marks steps whose notes are not counted (forced
    entry notes); they still connect the line.
    """
    out = []
    for v in range(grid.shape[1]):
        cur, prev = [], None
        for s in range(grid.shape[0]):
            t = int(grid[s, v])
            if NOTE0 <= t < NOTE0 + N_NOTES:
                m = MIDI_LO + t - NOTE0
                if prev is not None:
                    cur.append((m - prev[0], s - prev[1],
                                skip is None or not skip[s, v]))
                prev = (m, s)
            elif t == REST:
                if len(cur) > 1:
                    out.append(cur)
                cur, prev = [], None
        if len(cur) > 1:
            out.append(cur)
    return out


class Index:
    """Hashes of every training n-gram, n = 1..MAX_N, for longest-match lookup."""

    def __init__(self, grids: list[np.ndarray]):
        self.sets = [set() for _ in range(MAX_N + 1)]
        for g in grids:
            for ln in lines(g):
                key = [(a, b) for a, b, _ in ln]
                for i in range(len(key)):
                    h = ()
                    for n in range(1, min(MAX_N, len(key) - i) + 1):
                        h = h + (key[i + n - 1],)
                        self.sets[n].add(hash(h))

    def longest_from(self, key: list, i: int) -> int:
        n, h = 0, ()
        while i + n < len(key) and n < MAX_N:
            h2 = h + (key[i + n],)
            if hash(h2) not in self.sets[n + 1]:
                break
            h, n = h2, n + 1
        return n


def score_grid(index: Index, grid: np.ndarray, run: int = 12,
               skip: np.ndarray | None = None) -> dict:
    longest = 0
    counted = covered = 0
    for ln in lines(grid, skip):
        key = [(a, b) for a, b, _ in ln]
        cov = np.zeros(len(key), dtype=bool)
        for i in range(len(key)):
            k = index.longest_from(key, i)
            # only runs made mostly of counted (model-written) notes count
            if k and sum(c for _, _, c in ln[i:i + k]) >= 0.5 * k:
                longest = max(longest, k)
                if k >= run:
                    cov[i:i + k] = True
        c = np.array([x[2] for x in ln])
        counted += int(c.sum())
        covered += int((cov & c).sum())
    return {"longest_run": longest, "copied_share": round(covered / max(1, counted), 4),
            "notes": counted}


def baseline(root: Path, run: int = 12) -> dict:
    """Held-out Bach fugues against the training fugues+chorales."""
    pieces = load_corpus(root)
    index = Index([p["grid"] for p in pieces if p["split"] == "train"])
    rows = {p["piece_id"]: score_grid(index, p["grid"], run)
            for p in pieces if p["split"] != "train" and p["corpus"] == "fugue"}
    lr = [r["longest_run"] for r in rows.values()]
    cs = [r["copied_share"] for r in rows.values()]
    return {"run": run, "pieces": rows,
            "longest_run_median": float(np.median(lr)), "longest_run_max": int(max(lr)),
            "copied_share_median": float(np.median(cs)), "copied_share_max": float(max(cs))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("grids", nargs="*", help=".npy grids (generated); none = baseline only")
    ap.add_argument("--run", type=int, default=12)
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    base = baseline(root, args.run)
    print(json.dumps({k: v for k, v in base.items() if k != "pieces"}))
    if args.grids:
        pieces = load_corpus(root)
        index = Index([p["grid"] for p in pieces if p["split"] == "train"])
        for f in args.grids:
            f = Path(f)
            forced = f.parent / "forced.npy"
            skip = (np.load(forced) >= 0) if forced.exists() else None
            print(f, json.dumps(score_grid(index, np.load(f), args.run, skip)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
