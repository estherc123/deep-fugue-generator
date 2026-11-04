"""Compose a fugue from a subject with the form planner and the rule-aware sampler.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m fugue_data.generator.compose2 \\
        --subject-from bach_wtc1_fugue_02 --samples 2

    ... --subject "C5:1/2 B4:1/4 C5:1/4 G4:1/2 Ab4:1/2" --meter 4/4 --mode minor --tonic 0 --voices 3

Steps: `form_planner.plan_piece` (sections, keys, entries with a tonal answer,
episodes with sequences, stretto when the subject allows, pedals, cadences, a
chord for every beat from the harmony planner), then `sample.generate` with
the first entry as the prompt, entries and pedals written in, the chord /
key / texture plan as conditioning, the forbidden rules masked
(`melodic_rules`), and bar-by-bar selection over K candidates by the
Bach-calibrated rule checker (`bach_rules`). Output in
data/review/compositions_v2/<name>/, with plan.txt describing the form and
audit.json the rule audit of every sample.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from ..review.export_audio import render_wav
from ..review.render import write_midi
from .bach_rules import FORBIDDEN, RULES, check
from .encode import STEPS_PER_Q
from .form_planner import describe, plan_piece, subject_from_piece
from .sample import generate, grid_to_score, load_model, plan_tokens

ROOT = Path(__file__).resolve().parents[3]

DEFAULTS = dict(temperature=0.85, key_bias=1.5, chord_bias=0.75, force_sequences=True,
                no_crossing=True, repeat_penalty=2.0, melodic_rules=True, bar_select=8,
                copy_bias=8.0, select_lambda=0.1)


def audit_grid(g, p, start, fixed=None):
    meter = tuple(p["form"]["meter"])
    keys = [((p["tonic"] + (int(k) - 1) // 2) % 12, bool((int(k) - 1) % 2)) if k > 0
            else (p["tonic"], p["mode"] == "minor") for k in p["key"]]
    hits = check(g, meter=meter, tonic=p["tonic"], mode=p["mode"], keys=keys, start=start, final=True,
                 fixed=fixed)
    c = Counter(r for r, _, _ in hits)
    bars = (len(g) - start) / p["form"]["bar"]
    return {"bars": round(bars, 1), "forbidden": {r: c[r] for r in FORBIDDEN if c[r]},
            "per_100_bars": {r: round(100 * c[r] / bars, 1) for r in RULES if c[r]}}


def compose(subject, meter, mode, tonic, nv, name, model_path, samples, seed, out_root,
            reference=None, groups=None, subject_from=None, **kw):
    dev = "cuda"
    cfg = {**DEFAULTS, **{k: v for k, v in kw.items() if v is not None}}
    p = plan_piece(subject, meter, mode, tonic, nv, seed=seed, piece_id=name, groups=groups)
    model, ck = load_model(ROOT / model_path, dev)
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.txt").write_text(describe(p["form"]) + "\n")
    prompt = p["prompt_steps"]
    # density target: the subject's own rhythm, in every voice, a little thinner
    s_steps = p["form"]["subject_steps"]
    target = len(subject) / (s_steps / p["form"]["bar"]) * nv * 0.7
    cfg.setdefault("density_target", target)
    with torch.no_grad():
        grids, stats = generate(model, p, prompt, samples, dev, seed=seed,
                                sequences=p["sequences"], copies=p.get("copies"),
                                climax=p.get("climax"), final_chord=p.get("final_chord"), **cfg)
    audits = []
    n = p["n_steps"]
    fixed = plan_tokens(p) >= 0
    # proofreading: correct forbidden breaks the sampler let through
    from .repair import repair
    keys = [((tonic + (int(kk) - 1) // 2) % 12, bool((int(kk) - 1) % 2)) if kk > 0
            else (tonic, mode == "minor") for kk in p["key"]]
    kw = {"meter": tuple(meter), "tonic": tonic, "mode": mode, "keys": keys}
    fc = p.get("final_chord") or {}
    repaired = []
    for k in range(len(grids)):
        g2, ch = repair(grids[k], fixed, kw, p["form"]["bar"], start=prompt,
                        final_from=fc.get("from"), final_pcs=fc.get("pcs"), bass_pcs=fc.get("bass"))
        grids[k] = g2
        repaired.append(ch)
    for k, g in enumerate(grids, 1):
        sc = grid_to_score(g, f"{name}:gen{k}")
        render_wav(sc, out / f"generated_{k}.wav")
        write_midi(sc, Fraction(0), Fraction(n, STEPS_PER_Q), out / f"generated_{k}.mid")
        np.save(out / f"generated_{k}.npy", g)
        audits.append(audit_grid(g, p, prompt, fixed))
    np.save(out / "forced.npy", plan_tokens(p))
    # the plan as used, so reviews never have to re-create it
    (out / "plan.json").write_text(json.dumps({
        "copies": p.get("copies", []), "sequences": p.get("sequences", []),
        "prompt_steps": p["prompt_steps"], "n_steps": n, "groups": groups,
        "key": [int(x) for x in p["key"]], "chord": [int(x) for x in p["chord"]],
        "final_chord": {k: (sorted(v) if isinstance(v, set) else v) for k, v in (p.get("final_chord") or {}).items()},
        "cadences": p["form"]["cadences"], "climax_step": (p.get("climax") or {}).get("step")}, default=str))
    np.save(out / "plan_key.npy", p["key"])
    if reference is not None:
        render_wav(grid_to_score(reference, f"{name}:bach"), out / "bach.wav")
    meta = {"name": name, "model": model_path, "seed": seed, "n_steps": n, "prompt_steps": prompt,
            "prompt_quarters": prompt / STEPS_PER_Q, "plan_source": "planner", "split": "composed",
            "voices": nv, "meter": list(meter), "mode": mode, "tonic": tonic, "groups": groups,
            "subject_from": subject_from, **cfg,
            "audit": audits, "repaired_notes": repaired, "stats": {k: v for k, v in stats.items() if not isinstance(v, list)}, "form": {k: v for k, v in p["form"].items() if k != "stretto_options"}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return out, audits


def main(argv=None) -> int:
    from .compose import parse_subject
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subject-from")
    g.add_argument("--subject")
    ap.add_argument("--meter", default="4/4")
    ap.add_argument("--mode", default="minor")
    ap.add_argument("--tonic", type=int, default=0)
    ap.add_argument("--voices", type=int, default=4)
    ap.add_argument("--model", default="data/models/generator_v7_dpo_b.pt")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name")
    ap.add_argument("--out", default="data/review/compositions_v2")
    ap.add_argument("--bar-select", type=int)
    ap.add_argument("--chord-bias", type=float)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--copy-bias", type=float, help="0 = write copies in literally")
    ap.add_argument("--groups", type=int, help="middle-section entry groups (6-8: BWV 543 scale)")
    args = ap.parse_args(argv)
    ref = None
    if args.subject_from:
        subject, meter, mode, tonic, nv = subject_from_piece(args.subject_from)
        from .encode import load_corpus
        ref = next(x for x in load_corpus(ROOT, baroque=True, detect_entries=False)
                   if x["piece_id"] == args.subject_from)["grid"]
        name = args.name or f"{args.subject_from}_s{args.seed}"
    else:
        subject = parse_subject(args.subject)
        meter = tuple(int(x) for x in args.meter.split("/"))
        mode, tonic, nv = args.mode, args.tonic, args.voices
        name = args.name or f"custom_s{args.seed}"
    out, audits = compose(subject, meter, mode, tonic, nv, name, args.model, args.samples, args.seed,
                          ROOT / args.out, ref, groups=args.groups, subject_from=args.subject_from, bar_select=args.bar_select, chord_bias=args.chord_bias,
                          temperature=args.temperature, copy_bias=args.copy_bias)
    print(out)
    print((out / "plan.txt").read_text())
    for k, a in enumerate(audits, 1):
        print(f"sample {k}: {a['bars']} bars, forbidden {a['forbidden'] or 'none'}; per 100 bars {a['per_100_bars']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
