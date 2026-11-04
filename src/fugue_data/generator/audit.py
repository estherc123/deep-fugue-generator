"""Rule audit: every rule in the registry, per 100 bars, generated vs Bach.

    PYTHONPATH=src python -m fugue_data.generator.audit DIR [DIR ...]

DIR holds generated grids (*.npy, or *.npz with `generated`) named after the
piece (`<piece_id>.npz`, `<piece_id>/generated_k.npy`, or the A/B layout with
key.json). Bach's reference is the same pieces' own music over the same bars.

The question the owner set: generated fugues may break the rules Bach keeps
at most about once per five fugues. A rule Bach breaks at ~0 per 100 bars is
"forbidden" here; for the others, the target is Bach's own rate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..constraints.engine import evaluate, load_profile
from .encode import load_corpus
from .report import _fragment, bar_steps

ROOT = Path(__file__).resolve().parents[3]
SKIP = {"subject_entry_identity", "answer_transformation", "cadence_target_completion",
        "structural_target_reachable", "section_length_limit", "voice_participation",
        "max_dead_air", "timing_within_bounds"}   # need a plan / whole piece


def _profile():
    prof = dict(load_profile("bach_empirical"))
    prof["severities"] = {k: ("off" if k in SKIP else "diagnostic") for k in prof["rules"]}
    return prof


def _canon(pid):
    f = ROOT / "data" / "canonical" / f"{pid}.json"
    if not f.exists():
        f = ROOT / "data" / "canonical_baroque" / f"{pid}.json"
    return json.loads(f.read_text())


def count(grid, pid, canon, prof, start=0):
    g = grid.copy()
    # only the region from `start` on is judged: blank the prompt with rests
    from .encode import STOI
    g[:start] = STOI["REST"]
    vs = evaluate(_fragment(g, pid, canon), prof)
    return Counter(v["rule_id"] for v in vs)


def collect(d: Path):
    """[(piece_id, grid, start)] from a directory of generated output."""
    out = []
    key = d / "key.json"
    if key.exists():
        k = json.loads(key.read_text())
        for c in k["cases"]:
            for side in "AB":
                f = d / "audio" / f"{c['id']}_{side}.npy"
                if f.exists():
                    out.append((c["piece"], np.load(f), c["prompt_steps"], c[side]))
        return out
    for f in sorted(d.glob("*.npz")):
        z = np.load(f)
        for g in z["generated"]:
            out.append((f.stem.split("@")[0], g, int(z["prompt_steps"]), "gen"))
    for f in sorted(d.glob("*/generated_*.npy")):
        meta = json.loads((f.parent / "meta.json").read_text())
        out.append((f.parent.name, np.load(f), meta.get("prompt_steps", 0), "gen"))
    return out


def audit(dirs, show=True):
    prof = _profile()
    pieces = {p["piece_id"]: p for p in load_corpus(ROOT, baroque=True)}
    tot = {}
    bars = Counter()
    seen_bach = set()
    for d in dirs:
        for pid, g, start, tag in collect(Path(d)):
            p, canon = pieces[pid], _canon(pid)
            n = len(g)
            bs = bar_steps(p)
            name = f"{Path(d).name}:{tag}"
            tot.setdefault(name, Counter()).update(count(g, pid, canon, prof, start))
            bars[name] += (n - start) / bs
            if (pid, start, n) not in seen_bach:
                seen_bach.add((pid, start, n))
                tot.setdefault("bach", Counter()).update(count(p["grid"][:n], pid, canon, prof, start))
                bars["bach"] += (n - start) / bs
    rules = sorted({r for c in tot.values() for r in c})
    names = ["bach"] + [k for k in tot if k != "bach"]
    if show:
        print(f"{'rule (per 100 bars)':28s}" + "".join(f"{n[:22]:>24s}" for n in names))
        for r in rules:
            print(f"{r:28s}" + "".join(f"{100 * tot[n][r] / max(1, bars[n]):24.1f}" for n in names))
        print(f"{'bars':28s}" + "".join(f"{bars[n]:24.0f}" for n in names))
    return {n: {r: 100 * tot[n][r] / max(1, bars[n]) for r in rules} for n in names}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args(argv)
    audit(args.dirs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
