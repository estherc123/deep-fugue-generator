"""Self-review of planner compositions: does the music do what the plan asked?

    PYTHONPATH=src python -m fugue_data.generator.review_compositions data/review/compositions_v2

Per sample, against its plan (re-created from the saved seed) and Bach's
own fugue on the same subject:

- forbidden rule breaks (bach_rules.FORBIDDEN), and the Bach-rate rules per
  100 bars;
- `cad_hit`: share of planned cadences where the detected chord at the
  arrival has the planned root with the bass on it;
- `chord`: share of harmonic units whose detected root matches the plan;
- `cs` / `seq`: share of countersubject / sequence notes kept from the copy;
- `notes/bar` and `rest%` per voice (texture), next to Bach's;
- `climax`: where the soprano's highest note falls (0-1; research: 0.8-0.92).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .bach_rules import FORBIDDEN, RULES, check, sounding
from .encode import MIDI_LO, NOTE0, load_corpus
from .form_planner import plan_piece, subject_from_piece
from .labels import chords, harmonic_unit, notes_from_grid
from .sample import _in_scale, plan_tokens

ROOT = Path(__file__).resolve().parents[3]


def adherence(g, p):
    res = {}
    for name, items in (("cs", p.get("copies", [])), ("seq", p.get("sequences", []))):
        a = b = 0
        for q in items:
            if name == "cs":
                rng = range(q["start"], q["start"] + q["length"])
                off, vs, T = q["start"] - q["src_start"], q["src_voice"], q["shift"]
            else:
                rng = range(q["start"] + q["unit"], q["start"] + q["statements"] * q["unit"])
                off, vs, T = q["unit"], q["voice"], q["step"]
            for t in rng:
                t0 = int(g[t - off, vs])
                if NOTE0 <= t0 < NOTE0 + 88:
                    want = NOTE0 + _in_scale(MIDI_LO + t0 - NOTE0 + T, T, p, t) - MIDI_LO
                    b += 1
                    a += int(g[t, q["voice"]]) == want
        res[name] = round(a / b, 2) if b else None
    return res


def texture(g, bar):
    snd, att = sounding(g)
    nbar = len(g) / bar
    return round(att.sum() / nbar, 1), [round(float((snd[:, v] < 0).mean()) * 100) for v in range(g.shape[1])]


def review(d: Path) -> list[dict]:
    meta = json.loads((d / "meta.json").read_text())
    pid = meta["name"].rsplit("_s", 1)[0]
    sub, meter, mode, tonic, nv = subject_from_piece(pid)
    p = plan_piece(sub, meter, mode, tonic, nv, seed=meta["seed"], piece_id=meta["name"])
    fixed = plan_tokens(p) >= 0
    form = p["form"]
    bar, n = form["bar"], p["n_steps"]
    unit = harmonic_unit(meter)
    keys = [((tonic + (int(k) - 1) // 2) % 12, bool((int(k) - 1) % 2)) if k > 0 else (tonic, mode == "minor")
            for k in p["key"]]
    plan_roots = [None if int(c) < 2 else (tonic + (int(c) - 2) % 12) % 12 for c in p["chord"][::unit]]
    rows = []
    bach = next(x for x in load_corpus(ROOT, baroque=True, detect_entries=False) if x["piece_id"] == pid)
    bnb, brest = texture(bach["grid"], bar)
    for f in sorted(d.glob("generated_*.npy")):
        g = np.load(f)
        hits = check(g, meter=meter, tonic=tonic, mode=mode, keys=keys, start=p["prompt_steps"],
                     final=True, fixed=fixed)
        c = Counter(r for r, _, _ in hits)
        bars = (n - p["prompt_steps"]) / bar
        ch = chords(notes_from_grid(g), n, unit, bar)
        pairs = [(a, b["root"]) for a, b in zip(plan_roots, ch) if a is not None]
        chord_agree = sum(a == b for a, b in pairs) / max(1, len(pairs))
        cad_ok = cad_n = 0
        for cd in form["cadences"]:
            if cd["type"] not in ("PAC", "IAC", "final"):
                continue
            u = min(len(ch) - 1, cd["step"] // unit)
            want = (tonic + cd["key"][0]) % 12
            cad_n += 1
            cad_ok += ch[u]["root"] == want and ch[u]["bass"] == want
        snd, _ = sounding(g)
        top = np.where(snd[:, 0] >= 0, snd[:, 0], -1)
        nb, rest = texture(g, bar)
        rows.append({"sample": f"{d.name}/{f.stem}", "bars": round(n / bar),
                     "forbidden": {r: c[r] for r in FORBIDDEN if c[r]},
                     "rates": {r: round(100 * c[r] / bars, 1) for r in RULES if c[r] and r not in FORBIDDEN},
                     "cad_hit": round(cad_ok / max(1, cad_n), 2), "chord": round(chord_agree, 2),
                     **adherence(g, p), "notes_per_bar": nb, "bach_notes_per_bar": bnb,
                     "rest_pct": rest, "bach_rest_pct": brest,
                     "climax": round(int(np.argmax(top)) / n, 2)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args(argv)
    allrows = []
    for d in sorted(Path(args.root).iterdir()):
        if (d / "meta.json").exists():
            for r in review(d):
                allrows.append(r)
                print(json.dumps(r))
    forb = sum(sum(r["forbidden"].values()) for r in allrows)
    print(f"\n{len(allrows)} fugues, {forb} forbidden-rule breaks in all "
          f"({forb / max(1, len(allrows)):.2f} per fugue; target <= 0.2)")
    (Path(args.root) / "review.json").write_text(json.dumps(allrows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
