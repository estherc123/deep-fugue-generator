"""Generated passages for training the critic: the generator's own output over
the same bars as a real piece, so the critic learns the generator's faults,
not only the corruption operators'.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m fugue_data.generator.critic_negatives \
        --worker 0 --workers 4

Each piece in the train and validation splits gets one 16-bar run (prompt =
its first entry, or the first bar), two samples, under its own plan with the
current best decoding (key pull 3, repeat penalty 2, no crossing, parallels
masked, planned sequences written in). Saved to data/generated/critic_negatives/.
Splits are kept: the critic is validated on pieces it did not train on.

--fugue-offsets: for fugues only, also continue from bar 17, 33, 49, ... with
all of Bach's music before that point as the prompt, so the critic sees the
generator's middle entries and episodes, not only its first 16 bars. Saved as
<piece>@<bar>.npz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .encode import load_corpus
from .sample import first_entry_end, generate, load_model

BARS = 16


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--model", default="data/models/generator_v5.pt")
    ap.add_argument("--others", type=int, default=120, help="non-fugue train pieces to use")
    ap.add_argument("--chorales", type=int, default=80)
    ap.add_argument("--fugue-offsets", action="store_true")
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    out = root / "data" / "generated" / "critic_negatives"
    out.mkdir(parents=True, exist_ok=True)
    pieces = load_corpus(root, baroque=True, labels=True,
                         exclude_sources=("vivaldi-op6", "dcml-corelli"))
    rng = np.random.default_rng(0)
    pick = []
    for split in ("train", "validation"):
        ps = [p for p in pieces if p["split"] == split]
        f = [p for p in ps if p["corpus"] == "fugue"]
        o = [p for p in ps if p["corpus"] == "other"]
        c = [p for p in ps if p["corpus"] == "chorale"]
        if split == "train":
            o = list(rng.permutation(o))[:args.others]
            c = list(rng.permutation(c))[:args.chorales]
        pick += f + o + c
    pick.sort(key=lambda p: p["piece_id"])
    mine = pick[args.worker::args.workers]
    dev = "cuda"
    model, _ = load_model(root / args.model, dev)
    jobs = []
    for p in mine:
        if args.fugue_offsets:
            if p["corpus"] != "fugue":
                continue
            last = int(p["feats"]["bar"][-1])
            jobs += [(p, b) for b in range(int(p["feats"]["bar"][0]) + 16, last - 4, 16)]
        else:
            jobs.append((p, None))
    for p, b0 in jobs:
        dst = out / (f"{p['piece_id']}.npz" if b0 is None else f"{p['piece_id']}@{b0}.npz")
        if dst.exists():
            continue
        bar = int(p["feats"]["bar"][0]) if b0 is None else b0
        steps = int(np.searchsorted(p["feats"]["bar"], bar + BARS))
        q = {**p, "n_steps": steps, "grid": p["grid"][:steps],
             "feats": {k: v[:steps] for k, v in p["feats"].items()},
             **{k: p[k][:steps] for k in ("role", "to_entry", "to_cad", "cad_key", "pedal",
                                          "pnote", "epc", "key", "next_key", "to_key", "seq",
                                          "chord", "next_chord", "bass", "active")}}
        if b0 is None:
            prompt = first_entry_end(q) if q["has_plan"] else 0
            bar_steps = int(np.searchsorted(p["feats"]["bar"], bar + 1))
            prompt = min(max(prompt, bar_steps), steps // 2)
        else:
            prompt = int(np.searchsorted(p["feats"]["bar"], bar))
        try:
            with torch.no_grad():
                grids, _ = generate(model, q, prompt, 2, dev, key_bias=3.0, seed=1 + (b0 or 0),
                                    force_sequences=True, no_crossing=True, repeat_penalty=2.0)
        except Exception as e:  # noqa: BLE001
            print(p["piece_id"], b0, "failed", e, flush=True)
            continue
        np.savez_compressed(dst, generated=grids, real=q["grid"], prompt_steps=prompt,
                            split=p["split"], corpus=p["corpus"])
        print(p["piece_id"], b0, p["split"], steps, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
