"""Vertical dissonance in the 2nd-entry region (generated vs Bach), per piece and sample."""
import sys, json, numpy as np
sys.path.insert(0, "src")
from fugue_data.generator.form_planner import plan_piece, subject_from_piece
from fugue_data.generator.bach_rules import sounding
from fugue_data.generator.encode import load_corpus
ROOT='.'
corpus={p['piece_id']:p for p in load_corpus(__import__('pathlib').Path('.'),baroque=True,detect_entries=False)}
DISS={1,2,6,10,11}
def diss_rate(g,a,b,bar):
    snd,_=sounding(g[a:b]); n=0; d=0; ds=0; ns=0
    for t in range(len(snd)):
        ps=[int(x) for x in snd[t] if x>=0]
        if len(ps)<2: continue
        lo=min(ps); bad=0
        for i in range(len(ps)):
            for j in range(i+1,len(ps)):
                iv=abs(ps[i]-ps[j])%12
                if iv in DISS or (iv==5 and lo in (ps[i],ps[j])): bad=1
        n+=1; d+=bad
        if (a+t)%(bar//2)==0: ns+=1; ds+=bad
    return d/max(n,1), ds/max(ns,1)
import glob,os
root=sys.argv[1] if len(sys.argv)>1 else 'data/review/compositions_v5'
for dd in sorted(glob.glob(root+'/*/')):
    meta=json.load(open(dd+'meta.json')); pid=meta.get('subject_from') or meta['name'].rsplit('_s',1)[0]
    sub,meter,mode,tonic,nv=subject_from_piece(pid)
    p=plan_piece(sub,meter,mode,tonic,nv,seed=meta['seed'],piece_id=meta['name'],groups=meta.get('groups'))
    e=sorted(p['form']['entries'],key=lambda e:e['start'])[:2]
    a=e[1]['start']; b=a+(e[1]['start']-e[0]['start']); bar=p['form']['bar']
    bg=corpus[pid]['grid'] if pid in corpus else None
    row=[os.path.basename(dd[:-1]), f"bars {a/bar+1:.1f}-{b/bar+1:.1f}"]
    for k in (1,2):
        g=np.load(dd+f'generated_{k}.npy'); r=diss_rate(g,a,b,bar); row.append(f"g{k} {r[0]:.2f}/{r[1]:.2f}")
    if bg is not None:
        # Bach's own 2nd entry region: approximate with same span
        r=diss_rate(bg,a,min(b,len(bg)),bar); row.append(f"bach {r[0]:.2f}/{r[1]:.2f}")
    print(*row)
