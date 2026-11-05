"""spot.py DIR GEN BAR0 BAR1: per beat, planned key/chord, heard chord, notes, flags."""
import sys, json, numpy as np
sys.path.insert(0, "src")
from fugue_data.generator.labels import chords, notes_from_grid, harmonic_unit, QUAL_NAMES
from fugue_data.generator.bach_rules import sounding
from fugue_data.generator.encode import _beat_q, STEPS_PER_Q
N = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
Q = {'maj':'', 'min':'m', 'dim':'dim', 'aug':'+', 'dom7':'7', 'min7':'m7', 'hdim7':'ø7', 'dim7':'o7', 'maj7':'M7'}
d, k, b0, b1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
meta = json.load(open(d + "/meta.json")); plan = json.load(open(d + "/plan.json"))
from fugue_data.generator.form_planner import subject_from_piece
pid = meta.get("subject_from") or meta["name"].rsplit("_s", 1)[0]
_, meter, mode, tonic, nv = subject_from_piece(pid)
g = np.load(f"{d}/generated_{k}.npy"); fx = np.load(d + "/forced.npy")
beat = int(_beat_q(tuple(meter)) * STEPS_PER_Q); bar = int(meter[0] * 4 * STEPS_PER_Q / meter[1])
unit = harmonic_unit(meter); det = chords(notes_from_grid(g), len(g), unit, bar)
snd, att = sounding(g)
def kn(c):
    c = int(c); return '?' if c <= 0 else N[(tonic + (c - 1) // 2) % 12] + ('m' if (c - 1) % 2 else '')
def cn(c):
    c = int(c)
    if c < 2: return '-'
    q, r = divmod(c - 2, 12); return N[(tonic + r) % 12] + Q[QUAL_NAMES[q]]
def scale(c):
    c = int(c); t = (tonic + (c - 1) // 2) % 12; mi = (c - 1) % 2
    st = [0, 2, 3, 5, 7, 8, 10, 11, 9] if mi else [0, 2, 4, 5, 7, 9, 11]
    return {(t + x) % 12 for x in st}
print(d, f"v{k}", mode, "tonic", N[tonic], "| * = entry note, ! = out of planned key, X = beat clash")
for s in range(b0 * bar - bar, b1 * bar, beat):
    if s >= len(g): break
    ps = []
    for v in range(nv):
        seg = []
        for t in range(s, s + beat):
            if att[t, v]:
                m = int(snd[t, v]); pc = m % 12
                seg.append(N[pc] + str(m // 12 - 1) + ('*' if fx[t, v] >= 0 else '') + ('' if pc in scale(plan['key'][t]) else '!'))
            elif t == s and snd[t, v] >= 0: seg.append('~')
        ps.append(' '.join(seg) or '.')
    on = sorted(int(x) for x in snd[s] if x >= 0); clash = []
    for i in range(len(on)):
        for j in range(i + 1, len(on)):
            iv = (on[j] - on[i]) % 12
            if iv in (1, 2, 6, 10, 11) or (iv == 5 and i == 0): clash.append(f"{N[on[i]%12]}-{N[on[j]%12]}")
    c = det[min(len(det) - 1, s // unit)]
    heard = '-' if c['root'] is None else N[c['root']] + Q[c['qual']]
    print(f"b{s//bar+1:<3}{(s%bar)/beat+1:.0f} key {kn(plan['key'][s]):4s} plan {cn(plan['chord'][s]):6s} heard {heard:6s} | " +
          ' | '.join(x.ljust(22) for x in ps) + (' X ' + ','.join(clash) if clash else ''))
