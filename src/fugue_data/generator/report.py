"""One table for generated samples: does the output do what the plan asks?

Per sample directory (from `sample.py`), against Bach's piece and its plan:

- `key_agree`: share of bars whose local key (same Krumhansl–Schmuckler
  measure as the `key` feature) matches the plan's key for that bar (R3).
- `key_away`: the same, only over bars the plan puts outside the home key: did
  it actually modulate where asked?
- `notes_per_bar`: density, next to Bach's.
- parallel fifths / octaves by the rule engine (`bach_empirical`).
- `longest_run`: longest melodic run shared with the training set, model-written
  notes only (R5; held-out Bach is the baseline).
- `seq_cover`: share of the piece where some voice is inside a melodic
  sequence (R4; Bach's fugues: median 0.15).
- `chord_root` / `chord_bass`: share of harmonic units whose detected chord
  root / bass matches the chord plan (Bach's chords, `labels.chords`). Bach's
  own row is 1 by construction; models without the chord layer show the
  chance-plus-key baseline.

The memorisation index covers every training piece, the Baroque expansion
included.

    PYTHONPATH=src python -m fugue_data.generator.report data/review/generator_samples_v2/*
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..constraints.engine import evaluate, load_profile
from ..constraints.fragment import Fragment
from .encode import NOTE0, STEPS_PER_Q, load_corpus, local_keys
from .memorization import Index, score_grid
from .sequences import coverage
from .sample import grid_to_score


def _fragment(grid: np.ndarray, pid: str, canon: dict) -> Fragment:
    sc = grid_to_score(grid, pid)
    for v in sc["voices"]:
        for k, n in enumerate(v["notes"]):
            n.update(note_id=f'{v["voice_id"]}:{k}', voice_id=v["voice_id"],
                     pitch_class=n["midi"] % 12, octave=n["midi"] // 12 - 1, grace=False)
    sc["meter_map"] = canon.get("meter_map")
    sc["key_signature_map"] = canon.get("key_signature_map")
    return Fragment.from_score(sc)


def bar_steps(p: dict) -> int:
    num, den = p["meter_tok"].split("_")[1:3] if p["meter_tok"] != "METER_other" else (4, 4)
    return int(Fraction(4 * int(num), int(den)) * STEPS_PER_Q)


def beat_steps_of(p: dict) -> int:
    """Steps per beat (dotted beat in compound meters), as in `encode._beat_q`."""
    from .encode import _beat_q
    if p["meter_tok"] == "METER_other":
        return STEPS_PER_Q
    num, den = (int(x) for x in p["meter_tok"].split("_")[1:3])
    return int(_beat_q((num, den)) * STEPS_PER_Q)


def report(dirs: list[Path], root: Path) -> list[dict]:
    from .labels import chords as chords_of, harmonic_unit, notes_from_grid
    pieces = {p["piece_id"]: p for p in load_corpus(root, baroque=True, labels=True)}
    index = Index([p["grid"] for p in pieces.values() if p["split"] == "train"])
    prof = load_profile("bach_empirical")
    rows = []
    for d in dirs:
        pid = d.name
        p = pieces[pid]
        cf = root / "data" / "canonical" / f"{pid}.json"
        if not cf.exists():
            cf = root / "data" / "canonical_baroque" / f"{pid}.json"
        canon = json.loads(cf.read_text())
        mm = (canon.get("meter_map") or [{"numerator": 4, "denominator": 4}])[0]
        unit = harmonic_unit((mm["numerator"], mm["denominator"]))
        meta = json.loads((d / "meta.json").read_text())
        n = meta["n_steps"]
        bs = bar_steps(p)
        composed = (d / "plan_key.npy").exists()
        plan_ch = chords_of(notes_from_grid(p["grid"][:n]), n, unit, bs)
        plan_keys = (np.load(d / "plan_key.npy") if composed else p["key"][:n])[::bs]
        forced = np.load(d / "forced.npy") if (d / "forced.npy").exists() else None
        # For a composition, Bach's own fugue on the same subject is the
        # reference row; it was not written to this plan, so no key scores.
        samples = [("bach", p["grid"] if composed else p["grid"][:n])] + [
            (f.stem, np.load(f)) for f in sorted(d.glob("generated_*.npy"))]
        for name, g in samples:
            c = Counter(v["rule_id"] for v in evaluate(
                _fragment(g, pid, canon), prof, only={"parallel_fifths", "parallel_octaves"}))
            skip = (forced >= 0) if (forced is not None and name != "bach") else None
            keys = local_keys(g, bs, p["tonic"])[::bs]
            match = len(keys) == len(plan_keys)
            away = (plan_keys != 1) & (plan_keys != 2) & (plan_keys != 0)
            notes = int(((g >= NOTE0) & (g < NOTE0 + 88)).sum())
            ch = chords_of(notes_from_grid(g[:n]), min(n, len(g)), unit, bs) if not composed else None
            lo = (meta.get("prompt_steps") or 0) // unit       # generated region only
            pairs = [(a, b) for a, b in zip(plan_ch[lo:], ch[lo:]) if a["root"] is not None] if ch else []
            rows.append({
                "dir": str(d), "piece": pid, "sample": name,
                "guidance": meta.get("guidance", 1.0),
                "key_agree": round(float((keys == plan_keys).mean()), 3) if match else None,
                "key_away": (round(float((keys == plan_keys)[away].mean()), 3)
                             if match and away.any() else None),
                "notes_per_bar": round(notes / (len(g) / bs), 1),
                "par5": c["parallel_fifths"], "par8": c["parallel_octaves"],
                "longest_run": score_grid(index, g, 12, skip)["longest_run"],
                "seq_cover": round(coverage(g, beat_steps_of(p)), 3),
                "chord_root": round(sum(a["root"] == b["root"] for a, b in pairs) / len(pairs), 3) if pairs else None,
                "chord_bass": round(sum(a["bass"] == b["bass"] for a, b in pairs) / len(pairs), 3) if pairs else None,
            })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    for r in report([Path(x) for x in args.dirs], root):
        ka = "-" if r["key_agree"] is None else f'{r["key_agree"]:.2f}'
        print(f'{r["dir"]:55s} {r["sample"]:12s} key_agree {ka}  '
              f'away {r["key_away"] if r["key_away"] is not None else "-"}  '
              f'notes/bar {r["notes_per_bar"]:5.1f}  par5 {r["par5"]} par8 {r["par8"]}  '
              f'longest_run {r["longest_run"]:2d}  seq_cover {r["seq_cover"]:.2f}  '
              f'chord root {r["chord_root"]} bass {r["chord_bass"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
