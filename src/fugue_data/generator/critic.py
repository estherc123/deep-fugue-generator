"""Critic: scores how much a passage sounds like real Bach (the reward model).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m fugue_data.generator.critic

A bidirectional transformer over a window of the note grid (same tokens as the
generator, time-major, voices highest first), mean-pooled to one score. It is
trained on pairs, real passage vs a worse version of the same bars, with a
pairwise loss (the real one should score higher):

- **Corrupted pairs** (data/generated/corrupted/pairs.jsonl, 3,982 pairs):
  a real 8-bar window vs a copy with the keys moved, the rhythm flattened,
  segments shuffled, filler counterpoint pasted in, or a subject return
  removed. Labels are true by construction.
- **Generator pairs** (data/generated/critic_negatives): a real passage vs
  the generator's own continuation of the same bars under the same plan. This
  is what makes the critic useful for ranking the generator's output: it sees
  the generator's habits, not only the operators'.

Both members of a pair get the same random transposition. Splits are the
source pieces' splits, so validation/test pairs come from pieces the critic
never saw. Reported: pair accuracy per corruption type and on generator pairs.

Used for best-of-N: `score_grid` averages the critic over 8-bar windows of a
generated region; `sample.py --n-best N` keeps the highest-scoring samples.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import MIDI_LO, NOTE0, STOI, VOCAB

WIN = 128            # steps per window: 8 bars of 4/4 at sixteenths
MAX_TOK = 1024
N_NOTES = 88


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def window_grid(voices: dict, length_q: Fraction) -> np.ndarray:
    """Pair-window notes ({voice_k: [[onset, dur, midi], ...]}) -> token grid."""
    keys = sorted(voices, key=lambda k: int(k.split("_")[1]))
    n = int(length_q * 4)
    g = np.full((n, len(keys)), STOI["REST"], dtype=np.int16)
    for v, k in enumerate(keys):
        taken = np.zeros(n, dtype=bool)
        for o, d, m in sorted(voices[k], key=lambda x: Fraction(x[0])):
            s = int(round(Fraction(o) * 4))
            e = max(s + 1, int(round((Fraction(o) + Fraction(d)) * 4)))
            if s >= n or taken[s] or not (MIDI_LO <= m <= MIDI_LO + N_NOTES - 1):
                continue
            e = min(e, n)
            nxt = np.flatnonzero(taken[s:e])
            if len(nxt):
                e = s + int(nxt[0])
            g[s, v] = NOTE0 + m - MIDI_LO
            g[s + 1:e, v] = STOI["HOLD"]
            taken[s:e] = True
    return g


def load_pairs(root: Path) -> list[dict]:
    out = []
    for line in (root / "data" / "generated" / "corrupted" / "pairs.jsonl").read_text().splitlines():
        r = json.loads(line)
        L = Fraction(r["window"]["length_q"])
        out.append({"real": window_grid(r["original"], L), "fake": window_grid(r["corrupted"], L),
                    "split": r["split"], "kind": r["corruption_id"]})
    neg = root / "data" / "generated" / "critic_negatives"
    for f in sorted(neg.glob("*.npz")):
        z = np.load(f)
        real, gens, pr = z["real"], z["generated"], int(z["prompt_steps"])
        n = len(real)
        start = int(math.ceil(pr / 16) * 16)
        starts = list(range(start, n - WIN + 1, WIN)) or ([max(0, n - WIN)] if n - start >= 48 else [])
        for a in starts:
            for g in gens:
                out.append({"real": real[a:a + WIN], "fake": g[a:a + WIN],
                            "split": str(z["split"]), "kind": f"generator:{z['corpus']}"})
    return out


def _flat(g: np.ndarray, shift: int = 0):
    """Grid -> (tokens, voice ids, step-in-window), capped at MAX_TOK."""
    n, nv = g.shape
    n = min(n, MAX_TOK // nv)
    g = g[:n].astype(np.int64)
    if shift:
        notes = (g >= NOTE0) & (g < NOTE0 + N_NOTES)
        g = np.where(notes, g + shift, g)
    tok = g.reshape(-1)
    voice = np.tile(np.arange(1, nv + 1), n)
    step = np.repeat(np.arange(n), nv)
    return tok, voice, step


def _shift_ok(g: np.ndarray, shift: int) -> bool:
    m = g[(g >= NOTE0) & (g < NOTE0 + N_NOTES)]
    return not len(m) or (m.min() + shift >= NOTE0 and m.max() + shift < NOTE0 + N_NOTES)


def batchify(grids: list[np.ndarray], shifts: list[int], dev: str):
    flats = [_flat(g, s) for g, s in zip(grids, shifts)]
    L = max(len(f[0]) for f in flats)
    tok = torch.full((len(flats), L), STOI["<PAD>"], dtype=torch.long)
    voice = torch.zeros((len(flats), L), dtype=torch.long)
    step = torch.zeros((len(flats), L), dtype=torch.long)
    for i, (t, v, s) in enumerate(flats):
        tok[i, :len(t)] = torch.from_numpy(t)
        voice[i, :len(t)] = torch.from_numpy(v)
        step[i, :len(t)] = torch.from_numpy(s)
    return tok.to(dev), voice.to(dev), step.to(dev)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Critic(nn.Module):
    def __init__(self, d: int = 256, layers: int = 6, heads: int = 4, drop: float = 0.1):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), d)
        self.voice = nn.Embedding(8, d)
        self.step = nn.Embedding(MAX_TOK, d)
        self.sub = nn.Embedding(16, d)              # position in the bar (4/4 assumed)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, tok, voice, step):
        pad = tok == STOI["<PAD>"]
        x = self.tok(tok) + self.voice(voice) + self.step(step) + self.sub(step % 16)
        h = self.norm(self.enc(x, src_key_padding_mask=pad))
        keep = (~pad).unsqueeze(-1).float()
        return self.head((h * keep).sum(1) / keep.sum(1).clamp(min=1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@torch.no_grad()
def pair_accuracy(model, pairs, dev, bs=64) -> dict:
    model.eval()
    by = {}
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        sr = model(*batchify([p["real"] for p in chunk], [0] * len(chunk), dev))
        sf = model(*batchify([p["fake"] for p in chunk], [0] * len(chunk), dev))
        for p, a, b in zip(chunk, sr.tolist(), sf.tolist()):
            k = p["kind"]
            by.setdefault(k, [0, 0])
            by[k][0] += a > b
            by[k][1] += 1
    model.train()
    out = {k: round(c / n, 3) for k, (c, n) in sorted(by.items())}
    tot = [sum(v[0] for v in by.values()), sum(v[1] for v in by.values())]
    out["all"] = round(tot[0] / max(1, tot[1]), 3)
    out["n"] = tot[1]
    return out


def train(root: Path, steps: int, out: Path, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda"
    pairs = load_pairs(root)
    tr = [p for p in pairs if p["split"] == "train"]
    va = [p for p in pairs if p["split"] == "validation"]
    te = [p for p in pairs if p["split"] == "test"]
    print(json.dumps({"pairs": {"train": len(tr), "validation": len(va), "test": len(te)},
                      "generator_pairs_train": sum(p["kind"].startswith("generator") for p in tr)}),
          flush=True)
    model = Critic().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)
    best, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(tr), 48)
        real, fake, sh = [], [], []
        for i in idx:
            p = tr[i]
            s = int(rng.integers(-5, 7))
            if not (_shift_ok(p["real"], s) and _shift_ok(p["fake"], s)):
                s = 0
            real.append(p["real"]); fake.append(p["fake"]); sh.append(s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sr = model(*batchify(real, sh, dev))
            sf = model(*batchify(fake, sh, dev))
        loss = F.softplus(-(sr.float() - sf.float())).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            acc = pair_accuracy(model, va, dev)
            print(json.dumps({"step": step, "loss": round(loss.item(), 4), "val": acc,
                              "t": round(time.time() - t0)}), flush=True)
            if acc["all"] > best + 1e-4:
                best, bad = acc["all"], 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 8:
                    break
    model.load_state_dict(best_state)
    res = {"best_val": best, "validation": pair_accuracy(model, va, dev),
           "test": pair_accuracy(model, te, dev), "steps_run": step}
    torch.save({"state": model.state_dict(), "result": res}, out)
    print(json.dumps(res), flush=True)
    return res


# --------------------------------------------------------------------------- #
# Use
# --------------------------------------------------------------------------- #
def load_critic(path: Path, dev: str) -> Critic:
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = Critic().to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


@torch.no_grad()
def score_grid(model: Critic, grid: np.ndarray, start: int = 0, dev: str = "cuda") -> float:
    """Mean critic score over 8-bar windows (hop 4 bars) from `start` on."""
    n = len(grid)
    starts = list(range(start, max(start, n - WIN) + 1, WIN // 2))
    wins = [grid[a:a + WIN] for a in starts if len(grid[a:a + WIN]) >= 32]
    if not wins:
        return float("nan")
    return float(model(*batchify(wins, [0] * len(wins), dev)).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", default="data/models/critic_v1.pt")
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    train(root, args.steps, root / args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
