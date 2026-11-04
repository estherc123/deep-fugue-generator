"""Does the music get worse the longer the model writes? (Owner: "it is good
where it is close to the Bach reference and becomes progressively worse".)

    PYTHONPATH=src python -m fugue_data.generator.drift gen --gpu 0 --temperature 1.0
    PYTHONPATH=src python -m fugue_data.generator.drift report

Continues every test fugue for 24 bars from the middle of the piece (Bach up
to there as the prompt) under its own plan, with the current decoding. Then,
per bar since the prompt, against Bach's own bars:

- `nll`: the model's surprise at the music (mean per-token negative
  log-likelihood under the generator, read with the same plan). Rising
  generated-vs-Bach gap = the model is wandering where it was not trained.
- `harsh`: share of beat-onset voice pairs forming a second, seventh or
  tritone (dissonance on the beat, unprepared or not).
- `chord`: share of harmonic units whose root matches the chord plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "review" / "drift"
BARS = 24
KEYS = ("role", "to_entry", "to_cad", "cad_key", "pedal", "pnote", "epc", "key", "next_key",
        "to_key", "seq", "chord", "next_chord", "bass", "active")


def _cut(p, end):
    return {**p, "n_steps": end, "grid": p["grid"][:end],
            "feats": {k: v[:end] for k, v in p["feats"].items()},
            **{k: p[k][:end] for k in KEYS}}


def gen(model_path: str, temperature: float, n: int, tag: str, chord_bias: float = 0.0,
        key_bias: float = 3.0, melodic_rules: bool = False, bar_select: int = 0) -> None:
    from .encode import load_corpus
    from .sample import generate, load_model
    pieces = [p for p in load_corpus(ROOT, baroque=True, labels=True)
              if p["split"] == "test" and p["corpus"] == "fugue"]
    model, _ = load_model(ROOT / model_path, "cuda")
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    for p in pieces:
        bars = p["feats"]["bar"]
        b0, b1 = int(bars[0]), int(bars[-1])
        if b1 - b0 < BARS + 8:
            st = b0 + 4
        else:
            st = b0 + (b1 - b0 - BARS) // 2
        prompt = int(np.searchsorted(bars, st))
        end = int(np.searchsorted(bars, st + BARS))
        q = _cut(p, end)
        with torch.no_grad():
            grids, _ = generate(model, q, prompt, n, "cuda", temperature=temperature, key_bias=key_bias,
                                seed=3, force_sequences=True, no_crossing=True, repeat_penalty=2.0,
                                chord_bias=chord_bias, melodic_rules=melodic_rules,
                                bar_select=bar_select)
        np.savez_compressed(d / f"{p['piece_id']}.npz", generated=grids, real=q["grid"],
                            prompt_steps=prompt)
        print(tag, p["piece_id"], flush=True)


@torch.no_grad()
def token_nll(model, p, grid, dev="cuda"):
    """Per-step mean NLL of `grid` under the model with p's plan (chunks of ctx)."""
    from .train import build_sequence, collate
    import torch.nn.functional as F
    q = {**p, "grid": grid}
    n, nv = len(grid), p["nv"]
    out = np.full(n, np.nan)
    step = 0
    ctx = model.ctx
    # overlapping windows: score only the second half of each (enough left context)
    half = None
    while step < n:
        seq, feats, loss = build_sequence(q, max(0, step - (half or 0)), ctx, 0, True)
        pre = int((~loss).sum())
        steps_in = (len(seq) - pre) // nv
        half = half or steps_in // 2
        x, fx, y = collate([(seq, feats, loss)], ctx, dev)
        logits = model(x, fx).float()
        l = F.cross_entropy(logits[0], y[0], ignore_index=-100, reduction="none").cpu().numpy()
        first = max(0, step - half)
        for s in range(first, min(n, first + steps_in)):
            if s < step:
                continue
            a = pre - 1 + (s - first) * nv
            out[s] = float(l[a:a + nv].mean())
        step = first + steps_in
    return out


def _harsh(grid, beat):
    from .encode import MIDI_LO, NOTE0, STOI
    n, nv = grid.shape
    snd = np.full(grid.shape, -1)
    for v in range(nv):
        cur = -1
        for s in range(n):
            t = int(grid[s, v])
            if NOTE0 <= t < NOTE0 + 88:
                cur = MIDI_LO + t - NOTE0
            elif t == STOI["REST"]:
                cur = -1
            snd[s, v] = cur
    out = np.full(n, np.nan)
    for s in range(0, n, beat):
        live = [m for m in snd[s] if m >= 0]
        pairs = [(a, b) for i, a in enumerate(live) for b in live[i + 1:]]
        if pairs:
            out[s:s + beat] = np.mean([abs(a - b) % 12 in (1, 2, 6, 10, 11) for a, b in pairs])
    return out


def report(model_path: str, tags: list[str]) -> None:
    from .encode import load_corpus
    from .labels import chords, harmonic_unit, notes_from_grid
    from .report import bar_steps, beat_steps_of
    from .sample import load_model
    model, _ = load_model(ROOT / model_path, "cuda")
    pieces = {p["piece_id"]: p for p in load_corpus(ROOT, baroque=True, labels=True)
              if p["split"] == "test"}
    res = {}
    for tag in ["bach"] + tags:
        src = tags[0] if tag == "bach" else tag
        rows = {"nll": [], "harsh": [], "chord": []}
        for f in sorted((OUT / src).glob("*.npz")):
            z = np.load(f)
            p = pieces[f.stem]
            pr = int(z["prompt_steps"])
            bs, beat = bar_steps(p), beat_steps_of(p)
            mm = p["meter_tok"].split("_")
            unit = harmonic_unit((int(mm[1]), int(mm[2]))) if p["meter_tok"] != "METER_other" else 4
            real = z["real"]
            n = len(real)
            plan = chords(notes_from_grid(real), n, unit, bs)
            grids = [real] if tag == "bach" else list(z["generated"])
            for g in grids:
                q = {**p, **{k: (p[k][:n] if k != "feats" else {kk: vv[:n] for kk, vv in p["feats"].items()})
                             for k in KEYS + ("feats",)}, "n_steps": n}
                nll = token_nll(model, q, g)
                harsh = _harsh(g, beat)
                ch = chords(notes_from_grid(g), n, unit, bs)
                agree = np.full(n, np.nan)
                for i, (a, b) in enumerate(zip(plan, ch)):
                    if a["root"] is not None:
                        agree[i * unit:(i + 1) * unit] = a["root"] == b["root"]
                per = {"nll": [], "harsh": [], "chord": []}
                for b in range(BARS):
                    sl = slice(pr + b * bs, pr + (b + 1) * bs)
                    for k, arr in (("nll", nll), ("harsh", harsh), ("chord", agree)):
                        seg = arr[sl]
                        per[k].append(float(np.nanmean(seg)) if len(seg) and not np.isnan(seg).all() else np.nan)
                for k in rows:
                    rows[k].append(per[k])
        res[tag] = {k: np.nanmean(np.array(v, dtype=float), axis=0).round(3).tolist() for k, v in rows.items()}
    (OUT / "report.json").write_text(json.dumps(res, indent=1))
    print("bars since prompt:     " + " ".join(f"{b:>5d}" for b in range(1, BARS + 1, 2)))
    for k in ("nll", "harsh", "chord"):
        for tag, r in res.items():
            print(f"{k:6s} {tag:14s} " + " ".join(f"{x:5.2f}" for x in r[k][::2]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "report"])
    ap.add_argument("--model", default="data/models/generator_v7.pt")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--chord-bias", type=float, default=0.0)
    ap.add_argument("--key-bias", type=float, default=3.0)
    ap.add_argument("--melodic-rules", action="store_true")
    ap.add_argument("--bar-select", type=int, default=0)
    ap.add_argument("--tags", nargs="*", default=["t1.0"])
    args = ap.parse_args(argv)
    if args.cmd == "gen":
        gen(args.model, args.temperature, args.n, args.tag or f"t{args.temperature}",
            args.chord_bias, args.key_bias, args.melodic_rules, args.bar_select)
    else:
        report(args.model, args.tags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
