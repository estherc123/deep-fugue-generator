"""Share of notes outside the local key's extended scale (minor: 3 forms): Bach (labelled keys) vs generated (plan keys)."""
import sys, json, glob, numpy as np
from pathlib import Path
sys.path.insert(0, "src")
from fugue_data.generator.encode import NOTE0, MIDI_LO, load_corpus
from fugue_data.generator.labels import harmonic_unit
EXT = {0: [0, 2, 4, 5, 7, 9, 11], 1: [0, 2, 3, 5, 7, 8, 9, 10, 11]}
def rate_keys(g, keyat, mask=None):
    n = o = 0
    for s in range(len(g)):
        for v in range(g.shape[1]):
            t = int(g[s, v])
            if NOTE0 <= t < NOTE0 + 88 and (mask is None or not mask[s, v]):
                kt, mi = keyat(s)
                n += 1; o += (MIDI_LO + t - NOTE0 - kt) % 12 not in EXT[mi]
    return o, n
if sys.argv[1] == "bach":
    O = N = 0
    for p in load_corpus(Path("."), baroque=True, detect_entries=False):
        if p["corpus"] != "fugue" or p["split"] != "test": continue
        f = Path("data/labels") / (p["piece_id"] + ".json")
        if not f.exists() or p["meter_tok"] == "METER_other": continue
        lab = json.loads(f.read_text()); ks = lab["chord_keys"]
        num, den = (int(x) for x in p["meter_tok"].split("_")[1:3]); u = harmonic_unit((num, den))
        o, n = rate_keys(p["grid"], lambda s: tuple(ks[min(len(ks) - 1, s // u)]))
        O += o; N += n
    print("bach test fugues out-of-key", round(O / N, 4), N)
else:
    O = N = 0
    for d in glob.glob(sys.argv[1] + "/*/"):
        plan = json.load(open(d + "plan.json")); fx = np.load(d + "forced.npy") >= 0
        from fugue_data.generator.form_planner import subject_from_piece
        meta = json.load(open(d + "meta.json")); pid = meta.get("subject_from") or meta["name"].rsplit("_s", 1)[0]
        tonic = subject_from_piece(pid)[3]
        key = plan["key"]
        ka = lambda s: ((tonic + (int(key[s]) - 1) // 2) % 12, (int(key[s]) - 1) % 2) if key[s] > 0 else (tonic, 0)
        for f in glob.glob(d + "generated_*.npy"):
            o, n = rate_keys(np.load(f), ka, fx); O += o; N += n
    print(sys.argv[1], "free notes out-of-key", round(O / N, 4), N)
