"""Voice crossing, repeated notes and unisons: Bach (by corpus) vs generated samples."""
import sys, glob, collections
import numpy as np
from pathlib import Path
sys.path.insert(0, "src")
from fugue_data.generator.encode import load_corpus, NOTE0, MIDI_LO, STOI

HOLD, REST = STOI["HOLD"], STOI["REST"]

def stats(g, start=0):
    g = g[start:]
    n, nv = g.shape
    snd = np.full(g.shape, -1)
    for v in range(nv):
        cur = -1
        for s in range(n):
            t = int(g[s, v])
            if NOTE0 <= t < NOTE0 + 88: cur = MIDI_LO + t - NOTE0
            elif t == REST: cur = -1
            snd[s, v] = cur
    both = cross = uni = 0
    for v in range(nv - 1):
        m = (snd[:, v] >= 0) & (snd[:, v + 1] >= 0)
        both += m.sum(); cross += (m & (snd[:, v] < snd[:, v + 1])).sum(); uni += (m & (snd[:, v] == snd[:, v + 1])).sum()
    att = rep = 0
    simul = 0
    for v in range(nv):
        prev = None
        for s in range(n):
            t = int(g[s, v])
            if NOTE0 <= t < NOTE0 + 88:
                att += 1
                if prev == t: rep += 1
                prev = t
            elif t == REST: prev = None
    # steps where >=2 voices re-attack their own previous pitch at once
    rep_now = np.zeros(n, dtype=int)
    for v in range(nv):
        prev = None
        for s in range(n):
            t = int(g[s, v])
            if NOTE0 <= t < NOTE0 + 88:
                if prev == t: rep_now[s] += 1
                prev = t
            elif t == REST: prev = None
    return {"cross%": 100 * cross / max(1, both), "unison%": 100 * uni / max(1, both),
            "repeat%": 100 * rep / max(1, att), "multi_repeat_steps%": 100 * (rep_now >= 2).sum() / max(1, (snd >= 0).any(1).sum())}

def show(name, rows):
    a = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
    print(f"{name:42s} " + "  ".join(f"{k} {v:5.2f}" for k, v in a.items()) + f"  (n={len(rows)})")

ps = load_corpus(Path("."), baroque=True)
by = collections.defaultdict(list)
for p in ps:
    src = "WTC fugues" if p["piece_id"].startswith("bach_wtc") else ("orig chorales" if p["piece_id"].startswith("bach_chorale") else p["piece_id"].split("_")[0])
    by[src].append(stats(p["grid"]))
for k in sorted(by, key=lambda k: -len(by[k])): show("corpus " + k, by[k])
for d in sys.argv[1:]:
    rows = []
    for f in sorted(glob.glob(f"{d}/*/generated_*.npy")):
        import json
        meta = json.load(open(Path(f).parent / "meta.json"))
        rows.append(stats(np.load(f), meta.get("prompt_steps", 0)))
    if rows: show(d.replace("data/review/", ""), rows)
