"""Write a fugue from a subject alone.

    # a held-out Bach subject (Bach's own fugue on it is rendered for comparison)
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m fugue_data.generator.compose \\
        --subject-from bach_wtc2_fugue_19 --samples 3

    # your own subject: pitch:duration in quarters, onsets follow on
    ... compose --subject "C4:1 Eb4:1 D4:1/2 C4:1/2 B3:1 C4:1 D4:1 G3:2" \\
        --meter 4/4 --mode minor --voices 3

The form (entries, key path, sequences) is planned from a training fugue's
template (`plan.py`); the model writes everything except the entries.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from ..review.export_audio import render_wav
from ..review.render import write_midi
from .encode import STEPS_PER_Q, load_corpus
from .entries import detect
from .plan import choose_template, make_plan
from .sample import generate, grid_to_score, load_model, plan_tokens

_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_subject(text: str, start: Fraction = Fraction(0)) -> list:
    out, t = [], start
    for tok in text.split():
        name, dur = tok.split(":")
        pc = _NAMES[name[0].upper()]
        rest = name[1:]
        while rest and rest[0] in "#b":
            pc += 1 if rest[0] == "#" else -1
            rest = rest[1:]
        midi = 12 * (int(rest) + 1) + pc
        d = Fraction(dur)
        out.append((t, d, midi))
        t += d
    return out


def subject_of(root: Path, pid: str) -> list:
    score = json.loads((root / "data" / "canonical" / f"{pid}.json").read_text())
    ap = root / "data" / "annotations" / f"{pid}.json"
    ann = json.loads(ap.read_text())["annotations"] if ap.exists() else None
    det = detect(score, ann)
    return [(Fraction(o), Fraction(d), int(m)) for o, d, m, _ in det["subject"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/models/generator_v4.pt")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subject-from", help="take the subject of this (held-out) piece")
    g.add_argument("--subject", help='e.g. "C4:1 Eb4:1/2 ..." (pitch:quarters)')
    ap.add_argument("--meter", default="4/4")
    ap.add_argument("--mode", choices=["major", "minor"], default="minor")
    ap.add_argument("--voices", type=int, default=3)
    ap.add_argument("--template", help="training fugue to take the form from (default: chosen)")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--key-bias", type=float, default=3.0)
    ap.add_argument("--seq-bias", type=float, default=0.0)
    ap.add_argument("--no-force-sequences", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="data/review/compositions")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    pieces = load_corpus(root)
    rng = np.random.default_rng(args.seed)
    if args.subject_from:
        src = next(p for p in pieces if p["piece_id"] == args.subject_from)
        if src["split"] == "train":
            print(f"warning: {args.subject_from} is a training piece")
        subject = subject_of(root, args.subject_from)
        num, den = (int(x) for x in src["meter_tok"].split("_")[1:3])
        meter, mode, tonic, nv = (num, den), src["mode"], src["tonic"], src["nv"]
        name = args.name or args.subject_from
    else:
        subject = parse_subject(args.subject)
        meter = tuple(int(x) for x in args.meter.split("/"))
        mode, nv = args.mode, args.voices
        h = np.zeros(12)
        for _, d, m in subject:
            h[m % 12] += float(d)
        tonic = subject[-1][2] % 12 if mode else int(h.argmax())
        name = args.name or "subject"
    s_steps = int((subject[-1][0] + subject[-1][1] - subject[0][0]) * STEPS_PER_Q)
    template = (next(p for p in pieces if p["piece_id"] == args.template) if args.template
                else choose_template(pieces, nv, s_steps, rng, exclude=args.subject_from))
    p = make_plan(subject, meter, mode, tonic, template, name)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(root / args.model, dev)
    grids, stats = generate(model, p, p["prompt_steps"], args.samples, dev,
                            args.temperature, args.top_p, plan=True, seed=args.seed,
                            key_bias=args.key_bias, seq_bias=args.seq_bias,
                            sequences=p["sequences"], force_sequences=not args.no_force_sequences)
    out = root / args.out / name
    out.mkdir(parents=True, exist_ok=True)
    n = p["n_steps"]
    if args.subject_from:
        real = grid_to_score(src["grid"], args.subject_from)
        render_wav(real, out / "bach.wav")
        write_midi(real, Fraction(0), Fraction(src["n_steps"], STEPS_PER_Q), out / "bach.mid")
    for k in range(args.samples):
        sc = grid_to_score(grids[k], f"{name}:gen{k}")
        render_wav(sc, out / f"generated_{k + 1}.wav")
        write_midi(sc, Fraction(0), Fraction(n, STEPS_PER_Q), out / f"generated_{k + 1}.mid")
        np.save(out / f"generated_{k + 1}.npy", grids[k])
    np.save(out / "forced.npy", plan_tokens(p))
    np.save(out / "plan_key.npy", p["key"])
    meta = {"name": name, "subject_from": args.subject_from, "subject": args.subject,
            "template": p["template"], "model": args.model, "n_steps": n,
            "prompt_steps": p["prompt_steps"], "prompt_quarters": p["prompt_steps"] / STEPS_PER_Q,
            "entries": int(sum(1 for v in range(p["nv"]) for s in range(n)
                               if p["pnote"][s, v] >= 2 and (s == 0 or p["role"][s - 1, v] < 2))),
            "planned_sequences": len(p["sequences"]), "split": "new",
            "temperature": args.temperature, "top_p": args.top_p,
            "key_bias": args.key_bias, "seq_bias": args.seq_bias, "force_sequences": not args.no_force_sequences, "seed": args.seed,
            "force_entries": True, "mask_parallels": True, "plan_source": "template", **stats}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
