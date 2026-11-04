"""Preference pairs that teach the generator the rules (rule distillation).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m fugue_data.generator.rule_pairs --worker 0 --workers 4

For a training piece and an opening (Bach up to a bar, its plan after that),
the generator writes 8 bars twice:

- **chosen**: with the rule masks and bar-by-bar selection (K=4) on;
- **rejected**: free sampling, plan on, no masks;

and the pair is kept only when the rejected one breaks at least one forbidden
rule (bach_rules.FORBIDDEN) and the chosen one breaks none. DPO on these
pairs (`dpo.py --pairs-dir data/generated/rule_pairs`) moves the model's own
distribution toward rule-abiding continuations, so the masks have to step
in less often and the notes they leave are the model's first choices.
Saved in the critic_negatives format (`real` = chosen, `generated` = [rejected]).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .bach_rules import FORBIDDEN, check, piece_args
from .encode import load_corpus
from .sample import generate, load_model, plan_tokens

ROOT = Path(__file__).resolve().parents[3]
BARS = 8
KEYS = ("role", "to_entry", "to_cad", "cad_key", "pedal", "pnote", "epc", "key", "next_key",
        "to_key", "seq", "chord", "next_chord", "bass", "active")


def _cut(p, end):
    return {**p, "n_steps": end, "grid": p["grid"][:end],
            "feats": {k: v[:end] for k, v in p["feats"].items()}, **{k: p[k][:end] for k in KEYS}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--model", default="data/models/generator_v7_dpo_b.pt")
    ap.add_argument("--per-piece", type=int, default=3)
    args = ap.parse_args(argv)
    out = ROOT / "data" / "generated" / "rule_pairs"
    out.mkdir(parents=True, exist_ok=True)
    pieces = [p for p in load_corpus(ROOT, baroque=True, labels=True)
              if p["split"] in ("train", "validation") and p["corpus"] in ("fugue", "other")
              and p["nv"] >= 3]
    pieces.sort(key=lambda p: p["piece_id"])
    mine = pieces[args.worker::args.workers]
    model, _ = load_model(ROOT / args.model, "cuda")
    rng = np.random.default_rng(args.worker)
    kept = tried = 0
    for p in mine:
        bars = p["feats"]["bar"]
        b0, b1 = int(bars[0]), int(bars[-1])
        if b1 - b0 < BARS + 4:
            continue
        for j in range(args.per_piece):
            dst = out / f"{p['piece_id']}@{j}.npz"
            if dst.exists():
                continue
            st = int(rng.integers(b0 + 2, b1 - BARS))
            prompt = int(np.searchsorted(bars, st))
            end = int(np.searchsorted(bars, st + BARS))
            q = _cut(p, end)
            seed = int(rng.integers(1 << 30))
            with torch.no_grad():
                good, _ = generate(model, q, prompt, 1, "cuda", temperature=0.85, key_bias=1.5,
                                   chord_bias=0.75, seed=seed, force_sequences=False, no_crossing=True,
                                   repeat_penalty=2.0, melodic_rules=True, bar_select=4)
                bad, _ = generate(model, q, prompt, 2, "cuda", temperature=0.85, seed=seed + 1,
                                  mask_parallels=False)
            fixed = plan_tokens(q) >= 0 if q["has_plan"] else None
            kw = piece_args(q)
            tried += 1
            hg = [h for h in check(good[0], start=prompt, fixed=fixed, **kw) if h[0] in FORBIDDEN]
            rej = [b for b in bad
                   if any(h[0] in FORBIDDEN for h in check(b, start=prompt, fixed=fixed, **kw))]
            if hg or not rej:
                continue
            np.savez_compressed(dst, real=good[0], generated=np.stack(rej), prompt_steps=prompt,
                                split=p["split"], corpus=p["corpus"])
            kept += 1
            print(p["piece_id"], j, "kept", kept, "of", tried, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
