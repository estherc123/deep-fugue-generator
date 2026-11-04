"""Blind A/B between two generator setups (model + decoding), same openings.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m fugue_data.generator.ab build --name v7_vs_new
    PYTHONPATH=src python -m fugue_data.generator.ab serve --name v7_vs_new --port 8779
    PYTHONPATH=src python -m fugue_data.generator.ab score --name v7_vs_new

Cases are the held-out fugues of `critic_ab` (from the first entry, and from
the middle where the piece is long enough), BARS bars each, LEAD_BARS of Bach
before as lead-in. One sample per setup per case, same seed. Sides shuffled;
key.json is never served. Page and server are critic_ab's.
"""

from __future__ import annotations

import argparse
import http.server
import json
import random
from math import comb
from pathlib import Path

import numpy as np

from . import critic_ab

ROOT = Path(__file__).resolve().parents[3]
BARS = 20
LEAD_BARS = 4

SETUPS = {
    "old": {"model": "data/models/generator_v7.pt", "temperature": 1.0, "chord_bias": 0.0},
    "new": {"model": "data/models/generator_v7_dpo_b.pt", "temperature": 0.85, "chord_bias": 1.5},
}
COMMON = {"key_bias": 3.0, "force_sequences": True, "no_crossing": True, "repeat_penalty": 2.0}


def build(name: str, seed: int = 11) -> None:
    import torch
    from ..review.export_audio import render_wav
    from .encode import load_corpus
    from .sample import first_entry_end, generate, grid_to_score, load_model
    out = ROOT / "data" / "review" / f"ab_{name}"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    pieces = [p for p in load_corpus(ROOT, baroque=True, labels=True)
              if p["split"] == "test" and p["corpus"] == "fugue"]
    models = {k: load_model(ROOT / v["model"], "cuda")[0] for k, v in SETUPS.items()}
    rng = random.Random(seed)
    cases = []
    for p in sorted(pieces, key=lambda p: p["piece_id"]):
        bars = p["feats"]["bar"]
        b_first, b_last = int(bars[0]), int(bars[-1])
        starts = ["entry"]
        if b_last - b_first >= 2 * BARS + LEAD_BARS:
            starts.append(b_first + (b_last - b_first) // 2 - BARS // 2)
        for st in starts:
            if st == "entry":
                prompt = max(first_entry_end(p) if p["has_plan"] else 0,
                             int(np.searchsorted(bars, b_first + 1)))
                b0 = int(bars[min(prompt, len(bars) - 1)])
            else:
                prompt, b0 = int(np.searchsorted(bars, st)), st
            end = int(np.searchsorted(bars, b0 + BARS))
            if end - prompt < 32:
                continue
            q = critic_ab_cut(p, end)
            lead = int(np.searchsorted(bars, max(b_first, int(bars[prompt]) - LEAD_BARS)))
            cid = f"c{len(cases) + 1:02d}"
            sides = list(SETUPS)
            rng.shuffle(sides)
            for label, setup in zip("AB", sides):
                cfg = SETUPS[setup]
                with torch.no_grad():
                    g, _ = generate(models[setup], q, prompt, 1, "cuda", temperature=cfg["temperature"],
                                    chord_bias=cfg["chord_bias"], seed=seed, **COMMON)
                np.save(out / "audio" / f"{cid}_{label}.npy", g[0])
                render_wav(grid_to_score(g[0][lead:], f"{cid}{label}"), out / "audio" / f"{cid}_{label}.wav")
            cases.append({"id": cid, "piece": p["piece_id"], "start": st, "A": sides[0], "B": sides[1],
                          "lead_bars": int(bars[prompt]) - int(bars[lead]), "prompt_steps": prompt})
            print(cid, p["piece_id"], st, flush=True)
    items = [{"id": c["id"], "lead_bars": c["lead_bars"]} for c in cases]
    rng.shuffle(items)
    (out / "items.json").write_text(json.dumps(items, indent=1))
    (out / "key.json").write_text(json.dumps({"setups": SETUPS, "common": COMMON, "cases": cases},
                                             indent=1))
    page = critic_ab.PAGE.replace("<title>Critic Blind Test</title>", "<title>Generator Blind Test</title>") \
        .replace("then 16 generated bars", f"then {BARS} generated bars")
    (out / "index.html").write_text(page)


def critic_ab_cut(p, end):
    keys = ("role", "to_entry", "to_cad", "cad_key", "pedal", "pnote", "epc", "key", "next_key",
            "to_key", "seq", "chord", "next_chord", "bass", "active")
    return {**p, "n_steps": end, "grid": p["grid"][:end],
            "feats": {k: v[:end] for k, v in p["feats"].items()}, **{k: p[k][:end] for k in keys}}


def score(name: str) -> dict:
    out = ROOT / "data" / "review" / f"ab_{name}"
    key = {c["id"]: c for c in json.loads((out / "key.json").read_text())["cases"]}
    ans = json.loads((out / "answers.json").read_text())
    wins = {"new": 0, "old": 0}
    same = 0
    for cid, a in ans.items():
        if a == "same":
            same += 1
        else:
            wins[key[cid][a]] += 1
    n = wins["new"] + wins["old"]
    p = sum(comb(n, k) for k in range(wins["new"], n + 1)) / 2 ** n if n else 1.0
    res = {"answered": len(ans), **{f"{k}_preferred": v for k, v in wins.items()},
           "no_difference": same, "p_value_new_better_one_sided": round(p, 4)}
    print(json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "serve", "score"])
    ap.add_argument("--name", default="v7_vs_new")
    ap.add_argument("--port", type=int, default=8779)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(args.name)
    elif args.cmd == "serve":
        critic_ab.OUT = ROOT / "data" / "review" / f"ab_{args.name}"
        http.server.ThreadingHTTPServer(("127.0.0.1", args.port), critic_ab.Handler).serve_forever()
    else:
        score(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
