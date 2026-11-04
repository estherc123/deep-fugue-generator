"""Add generator samples to the listening site, as a second page.

Reuses the corruption page's layout: Bach on top, a generated version below,
the generated region (everything after the prompt) shaded, and the blind test
asking which one is Bach.

    PYTHONPATH=src python -m fugue_data.generator.samples_site \
        v1=data/review/generator_samples_v1 v2=data/review/generator_samples_v2

Each set is every piece directory under the given folder. Notes the plan wrote
in (subject entries) get the white outline.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..review.corruption_site import _TEMPLATE
from ..review.export_audio import QUARTER_SECONDS
from .encode import NOTE0, STEPS_PER_Q, load_corpus
from .sample import grid_to_score


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _compact(score: dict) -> list[list]:
    return [[[float(Fraction(n["onset_q"])), float(Fraction(n["duration_q"])), n["midi"]]
             for n in v["notes"]] for v in score["voices"]]


def _desc(label: str, pid: str, meta: dict, plan_txt: str | None = None) -> str:
    if meta.get("plan_source") == "planner":
        form = (plan_txt or "").strip().replace("\n", "; ")
        return (f"{label}. Composed from the subject of {pid} alone (Bach's own fugue on it is on "
                f"top; it was held out of training). The form planner chose the form: {form}. "
                f"Entries (tonal answer in the exposition) and pedal tones are written in "
                f"(outlined); a chord for every beat comes from the harmony planner; the model "
                f"writes the rest with forbidden voice-leading masked and {meta.get('bar_select')} "
                f"candidates per bar, keeping the one with the fewest rule breaks.")
    if meta.get("plan_source") == "template":
        return (f"{label}. Composed from the subject alone. Top: Bach's own fugue on this "
                f"subject ({pid}, held out from training). Bottom: the generator's "
                f"fugue: the subject is given, the form (entries, key path, sequences) "
                f"is planned from the training fugue {meta['template']}, entries are "
                f"written in (outlined), planned sequences restate the model's own "
                f"material, and the model writes the rest. Key bias {meta['key_bias']}, "
                f"parallels masked, temperature {meta['temperature']}.")
    plan = ("Bach's plan for this piece" if meta.get("plan_source") != "detected"
            else "the detected entry plan for this piece")
    entries = ("Subject entries are written in from the plan (outlined); the model "
               "writes every other note." if meta.get("force_entries")
               else "The model writes the entries itself.")
    chords = ("" if "chords" not in meta else
              " The model also follows Bach's chord per beat and which voices sound per bar (v5)."
              if meta["chords"] else " No chord plan (v5 with the chord layer switched off).")
    extras = (f" Key pull {meta.get('key_bias', 0)}"
              + (", planned sequences written in." if meta.get("force_sequences") else ".")
              if "key_bias" in meta else "")
    rules = ("Parallel fifths/octaves are masked during sampling."
             if meta.get("mask_parallels") else "No counterpoint rules enforced.")
    return (f"{label}. Prompted with Bach's first {meta['prompt_quarters']:g} quarters "
            f"(the opening subject entry), then generated under {plan}. {entries} "
            f"{rules}{chords}{extras} {pid} is a {meta['split']} piece: the model never trained on it. "
            f"Sampling: temperature {meta['temperature']}, top-p {meta['top_p']}.")


def _changed(gen_grid: np.ndarray, forced: np.ndarray | None) -> list[list[int]]:
    """Per voice, 1 for notes whose attack the plan forced."""
    out = []
    for v in range(gen_grid.shape[1]):
        flags = []
        for s in range(gen_grid.shape[0]):
            t = int(gen_grid[s, v])
            if NOTE0 <= t < NOTE0 + 88:
                flags.append(int(forced is not None and forced[s, v] == t))
        out.append(flags)
    return out


def build(sets: list[tuple[str, Path]], root: Path | None = None,
          site: Path | None = None) -> Path:
    root = root or _repo_root()
    site = site or root / "data" / "review" / "corruption_site"
    corpus = {p["piece_id"]: p for p in load_corpus(root, baroque=True)}
    rng = random.Random(5)
    items = []
    (site / "audio").mkdir(parents=True, exist_ok=True)
    dirs = [(label, d) for label, base in sets
            for d in sorted((root / base).iterdir()) if (d / "meta.json").exists()]
    for label, d in dirs:
        pid = d.name
        meta = json.loads((d / "meta.json").read_text())
        if pid not in corpus:
            import re
            pid = re.sub(r"_s\d+$", "", pid)
        forced = np.load(d / "forced.npy") if (d / "forced.npy").exists() else None
        p = corpus[pid]
        bach = _compact(grid_to_score(p["grid"][:meta["n_steps"]], pid))
        length_q = meta["n_steps"] / STEPS_PER_Q
        bar_q = float(Fraction(4 * int(p["meter_tok"].split("_")[1]),
                               int(p["meter_tok"].split("_")[2])))
        for k in range(1, 100):
            f = d / f"generated_{k}.npy"
            if not f.exists():
                break
            key = f"gen_{label}_{d.name}_{k}".replace(" ", "_")
            shutil.copy(d / "bach.wav", site / "audio" / f"{key}_original.wav")
            shutil.copy(d / f"generated_{k}.wav", site / "audio" / f"{key}_corrupted.wav")
            g = np.load(f)
            gen = _compact(grid_to_score(g, pid))
            items.append({
                "key": key, "op": label, "title": f"{label} — generated version {k}",
                "piece": pid, "split": meta["split"], "corpus": p["corpus"],
                "meter": p["meter_tok"].replace("METER_", "").replace("_", "/"),
                "bars": int(round(length_q / bar_q)), "barQ": bar_q, "startBar": 1,
                "lengthQ": length_q, "span": [meta["prompt_quarters"], length_q],
                "spanVoice": None,
                "desc": _desc(label, pid, meta, (d / "plan.txt").read_text()
                              if (d / "plan.txt").exists() else None),
                "pairId": key, "original": bach, "corrupted": gen,
                "changed": _changed(g, forced),
                "blindFlip": rng.random() < 0.5,
            })
    html = _TEMPLATE
    swaps = [
        ("<title>Corrupted Bach Pairs</title>", "<title>Generator Samples</title>"),
        ("<h1>Corrupted Bach pairs</h1>", "<h1>Generator samples</h1>"),
        ("Each pair is a real Bach window and a copy damaged by one operator. Both\n"
         "  sides pass the same hard counterpoint rules, so the damage is musical, not\n"
         "  rule-breaking.",
         "Bach's fugue on top; below, a generated version of the same fugue from the "
         "same opening and the same form plan. The shaded region is generated. "
         "Outlined notes are subject entries written in from the plan (v2); "
         "everything else is the model."),
        ('"Corrupted"', '"Generated"'), (" — Corrupted", " — Generated"),
        ("corrupted span", "generated region"),
        ("note not in the original", "subject entry from the plan"),
        ('cb.', 'gen.'),
    ]
    for a, b in swaps:
        assert a in html, a
        html = html.replace(a, b)
    start = html.index("const OPS = [")
    end = html.index("];", start) + 2
    ops = [["all", "All"]] + [[lab, lab] for lab, _ in sets]
    html = html[:start] + f"const OPS = {json.dumps(ops)};" + html[end:]
    html = (html.replace("__DATA__", json.dumps(items, separators=(",", ":")))
            .replace("__QSEC__", str(QUARTER_SECONDS)))
    (site / "generator.html").write_text(html)
    return site / "generator.html"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="label=folder of sample directories")
    args = ap.parse_args(argv)
    print(build([tuple(x.split("=", 1)) for x in args.sets]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
