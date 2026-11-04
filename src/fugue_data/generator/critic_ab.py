"""Blind A/B: does the critic's ranking match a listener's ear? (Step 2.)

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m fugue_data.generator.critic_ab build
    PYTHONPATH=src python -m fugue_data.generator.critic_ab serve     # http://127.0.0.1:8778
    PYTHONPATH=src python -m fugue_data.generator.critic_ab score

For held-out pieces only (test split: the generator, the critic and its
training passages never saw them), each case continues Bach for 16 bars, from
the end of the first entry or from the middle of the piece. N samples are
generated with the current best decoding; the critic keeps its best- and
worst-scoring one. The listener hears both, with the same 4 bars of Bach
before the generated passage as lead-in, sides shuffled, and picks the one
that sounds better. Best-vs-worst is the strongest contrast the critic can
offer: if a listener cannot hear it, the critic's ranking is not worth
training on.

key.json (which side is the critic's pick) is never served by the app.
"""

from __future__ import annotations

import argparse
import http.server
import json
import random
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "review" / "critic_ab"
BARS = 16
LEAD_BARS = 4


def build(model_path: str, critic_path: str, n: int, seed: int = 7) -> None:
    import torch
    from ..review.export_audio import render_wav
    from .critic import load_critic, score_grid
    from .encode import STEPS_PER_Q, load_corpus
    from .sample import first_entry_end, generate, grid_to_score, load_model

    dev = "cuda"
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    pieces = [p for p in load_corpus(ROOT, baroque=True, labels=True)
              if p["split"] == "test" and p["corpus"] == "fugue"]
    model, _ = load_model(ROOT / model_path, dev)
    critic = load_critic(ROOT / critic_path, dev)
    rng = random.Random(seed)
    items, key = [], []
    for p in sorted(pieces, key=lambda p: p["piece_id"]):
        bars = p["feats"]["bar"]
        b_first, b_last = int(bars[0]), int(bars[-1])
        starts = ["entry"]
        if b_last - b_first >= 2 * BARS + LEAD_BARS:
            starts.append(b_first + (b_last - b_first) // 2 - BARS // 2)
        for st in starts:
            if st == "entry":
                prompt = max(first_entry_end(p) if p["has_plan"] else 0,
                             int(np.searchsorted(bars, b_first + 1)))
                b0 = int(bars[min(prompt, len(bars) - 1)])
            else:
                prompt = int(np.searchsorted(bars, st))
                b0 = st
            end = int(np.searchsorted(bars, b0 + BARS))
            if end - prompt < 8 * STEPS_PER_Q:
                continue
            q = {**p, "n_steps": end, "grid": p["grid"][:end],
                 "feats": {k: v[:end] for k, v in p["feats"].items()},
                 **{k: p[k][:end] for k in ("role", "to_entry", "to_cad", "cad_key", "pedal",
                                            "pnote", "epc", "key", "next_key", "to_key", "seq",
                                            "chord", "next_chord", "bass", "active")}}
            with torch.no_grad():
                grids, _ = generate(model, q, prompt, n, dev, key_bias=3.0, seed=seed,
                                    force_sequences=True, no_crossing=True, repeat_penalty=2.0)
            sc = [score_grid(critic, g, prompt, dev) for g in grids]
            best, worst = int(np.argmax(sc)), int(np.argmin(sc))
            lead = int(np.searchsorted(bars, max(b_first, int(bars[prompt]) - LEAD_BARS)))
            cid = f"c{len(items) + 1:02d}"
            sides = [("best", best), ("worst", worst)]
            rng.shuffle(sides)
            for label, (role, k) in zip("AB", sides):
                sc_ = grid_to_score(grids[k][lead:], f"{cid}{label}")
                render_wav(sc_, OUT / "audio" / f"{cid}_{label}.wav")
            items.append({"id": cid, "piece": p["piece_id"],
                          "where": "after the exposition's first entry" if st == "entry"
                          else f"from bar {b0}",
                          "lead_seconds": None, "lead_bars": int(bars[prompt]) - int(bars[lead])})
            key.append({"id": cid, "A": sides[0][0], "B": sides[1][0],
                        "scores": [round(x, 4) for x in sc], "best": best, "worst": worst,
                        "piece": p["piece_id"], "start": st, "prompt_steps": prompt})
            print(cid, p["piece_id"], st, "scores", [round(x, 2) for x in sc], flush=True)
    rng.shuffle(items)
    (OUT / "items.json").write_text(json.dumps(items, indent=1))
    (OUT / "key.json").write_text(json.dumps({"model": model_path, "critic": critic_path,
                                              "n": n, "cases": key}, indent=1))
    (OUT / "index.html").write_text(PAGE)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Critic Blind Test</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e1dc;--acc:#2f5d8a;--pick:#e8f0f8}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9993;--card:#1f1f1d;--line:#33322f;--acc:#8fb4dc;--pick:#22303d}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:760px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}p{color:var(--mut);margin:4px 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0}.row b{width:18px}
audio{flex:1;min-width:220px}button{font:inherit;padding:6px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.on{background:var(--pick);border-color:var(--acc);color:var(--acc);font-weight:600}
.meta{font-size:13px;color:var(--mut)}#done{font-weight:600}
</style></head><body><main>
<h1>Which continuation sounds better?</h1>
<p>Each case: the same Bach lead-in, then 16 generated bars. Pick the one you prefer, or "no difference". Choices save automatically.</p>
<p id="done"></p><div id="list"></div></main>
<script>
let items=[],ans={};
async function load(){items=await (await fetch('items.json')).json();
 try{ans=await (await fetch('answers')).json()}catch(e){ans={}}; render()}
function render(){const L=document.getElementById('list');L.innerHTML='';
 items.forEach((it,i)=>{const d=document.createElement('div');d.className='card';
  d.innerHTML=`<div class="meta">Case ${i+1} of ${items.length} &middot; lead-in: ${it.lead_bars} bars of Bach</div>
   <div class="row"><b>A</b><audio controls preload="none" src="audio/${it.id}_A.wav"></audio></div>
   <div class="row"><b>B</b><audio controls preload="none" src="audio/${it.id}_B.wav"></audio></div>
   <div class="row">${['A','B','same'].map(c=>`<button data-c="${c}" class="${ans[it.id]===c?'on':''}">${c==='same'?'no difference':c+' is better'}</button>`).join('')}</div>`;
  d.querySelectorAll('button').forEach(b=>b.onclick=async()=>{ans[it.id]=b.dataset.c;
   await fetch('answers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(ans)});render()});
  L.appendChild(d)});
 document.getElementById('done').textContent=`${Object.keys(ans).length} of ${items.length} answered`}
load();
</script></body></html>"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(OUT), **k)

    def do_GET(self):
        if self.path.startswith("/key.json"):
            self.send_error(404)
            return
        if self.path == "/answers":
            f = OUT / "answers.json"
            body = f.read_bytes() if f.exists() else b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/answers":
            self.send_error(404)
            return
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        (OUT / "answers.json").write_text(json.dumps(data, indent=1))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


def score() -> dict:
    from math import comb
    key = {c["id"]: c for c in json.loads((OUT / "key.json").read_text())["cases"]}
    ans = json.loads((OUT / "answers.json").read_text())
    agree = disagree = same = 0
    for cid, a in ans.items():
        if a == "same":
            same += 1
        elif key[cid][a] == "best":
            agree += 1
        else:
            disagree += 1
    n = agree + disagree
    # one-sided binomial p-value that the listener prefers the critic's pick
    p = sum(comb(n, k) for k in range(agree, n + 1)) / 2 ** n if n else 1.0
    res = {"answered": len(ans), "critic_pick_preferred": agree, "critic_pick_rejected": disagree,
           "no_difference": same, "agreement": round(agree / n, 3) if n else None,
           "p_value_one_sided": round(p, 4)}
    print(json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "serve", "score"])
    ap.add_argument("--model", default="data/models/generator_v7.pt")
    ap.add_argument("--critic", default="data/models/critic_v2.pt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--port", type=int, default=8778)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(args.model, args.critic, args.n)
    elif args.cmd == "serve":
        http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    else:
        score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
