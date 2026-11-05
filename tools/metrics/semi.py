"""Beat onsets with a semitone clash (m2 / M7 / m9 between any two sounding voices), per beat onset."""
import sys, json, glob, numpy as np
from pathlib import Path
sys.path.insert(0, "src")
from fugue_data.generator.bach_rules import sounding
from fugue_data.generator.encode import load_corpus, _beat_q, STEPS_PER_Q
def rate(g, beat):
    snd, _ = sounding(g); n = c = 0
    for s in range(0, len(g), beat):
        ps = [int(x) for x in snd[s] if x >= 0]
        if len(ps) < 2: continue
        n += 1
        c += any((a - b) % 12 in (1, 11) for i, a in enumerate(ps) for b in ps[i + 1:])
    return c, n
C = N = 0
if sys.argv[1] == "bach":
    for p in load_corpus(Path("."), baroque=True, detect_entries=False):
        if p["corpus"] != "fugue" or p["split"] != sys.argv[2] or p["meter_tok"] == "METER_other": continue
        num, den = (int(x) for x in p["meter_tok"].split("_")[1:3])
        c, n = rate(p["grid"], int(_beat_q((num, den)) * STEPS_PER_Q)); C += c; N += n
else:
    from fugue_data.generator.form_planner import subject_from_piece
    for d in glob.glob(sys.argv[1] + "/*/"):
        meta = json.load(open(d + "meta.json")); pid = meta.get("subject_from") or meta["name"].rsplit("_s", 1)[0]
        meter = subject_from_piece(pid)[1]; beat = int(_beat_q(tuple(meter)) * STEPS_PER_Q)
        for f in glob.glob(d + "generated_*.npy"):
            c, n = rate(np.load(f), beat); C += c; N += n
print(sys.argv[1:], "semitone clash per beat", round(C / N, 4), N)
