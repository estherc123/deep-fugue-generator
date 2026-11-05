"""Beat-onset dissonance when exactly 2 voices sound: Bach (train fugues) vs a compositions folder."""
import sys, glob, json, numpy as np
from pathlib import Path
sys.path.insert(0, "src")
from fugue_data.generator.bach_rules import sounding
from fugue_data.generator.encode import load_corpus, _beat_q, STEPS_PER_Q
DISS = {1, 2, 6, 10, 11}
def rate(g, beat, a=0, b=None):
    snd, _ = sounding(g[a:b]); n = d = 0
    for t in range(0, len(snd)):
        if (a + t) % beat: continue
        ps = sorted(int(x) for x in snd[t] if x >= 0)
        if len(ps) != 2: continue
        iv = (ps[1] - ps[0]) % 12
        n += 1; d += iv in DISS or iv == 5
    return d, n
if sys.argv[1] == "bach":
    D = N = 0
    for p in load_corpus(Path("."), baroque=True, detect_entries=False):
        if p["corpus"] != "fugue" or p["split"] != "train" or p["meter_tok"] == "METER_other": continue
        num, den = (int(x) for x in p["meter_tok"].split("_")[1:3])
        beat = int(_beat_q((num, den)) * STEPS_PER_Q)
        d, n = rate(p["grid"], beat); D += d; N += n
    print("bach train fugues: 2-voice beat onsets", N, "dissonant", round(D / N, 3))
else:
    D = N = 0
    for f in glob.glob(sys.argv[1] + "/*/generated_*.npy"):
        meta = json.load(open(Path(f).parent / "meta.json"))
        beat = int(_beat_q(tuple(meta["meter"])) * STEPS_PER_Q) if "meter" in meta else 4
        d, n = rate(np.load(f), beat); D += d; N += n
    print(sys.argv[1], N, round(D / N, 3))
