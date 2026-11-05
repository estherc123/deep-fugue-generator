"""Share of harmonic units in the generated music whose detected chord fits the planned local key."""
import sys, json, glob, numpy as np
sys.path.insert(0, "src")
from fugue_data.generator.harmony import VOCAB
from fugue_data.generator.labels import chords, notes_from_grid, harmonic_unit
for d in sys.argv[1:]:
    ok = tot = 0; early_ok = early_tot = 0
    for m in sorted(glob.glob(d + "/*/meta.json")):
        meta = json.load(open(m)); pk = np.load(m.replace("meta.json", "plan_key.npy"))
        tonic = meta["tonic"]; meter = tuple(meta["meter"]); unit = harmonic_unit(meter)
        bar = int(4 * meter[0] / meter[1] * 4); pr = meta["prompt_steps"]
        for f in sorted(glob.glob(m.replace("meta.json", "generated_*.npy"))):
            g = np.load(f); n = len(g)
            ch = chords(notes_from_grid(g), n, unit, bar)
            for u, c in enumerate(ch):
                s = u * unit
                if s < pr or c["root"] is None or pk[s] < 1: continue
                kd, mi = (int(pk[s]) - 1) // 2, (int(pk[s]) - 1) % 2
                rel = (c["root"] - tonic - kd) % 12
                good = c["qual"] in VOCAB["minor" if mi else "major"].get(rel, ())
                tot += 1; ok += good
                if s < pr + 6 * bar: early_tot += 1; early_ok += good
    print(f"{d.split('/')[-1]}: music chords in the planned key {ok/tot:.1%}; in the 6 bars after the first entry {early_ok/early_tot:.1%}")
