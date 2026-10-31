"""Serve the blind arbitration set (Spec §18.5).

Deliberately shows nothing about either earlier verdict, and reshuffles the sides
under a seed unrelated to the review app's, so a reviewer cannot reconstruct
their previous answer from position or from memory of the layout.

Run:  python -m fugue_data.review.arbitrate_app  [--port 8768]
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


from .synth import SYNTH_JS

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Blind re-judging</title><style>
:root{color-scheme:light}
body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#f6f6f5;color:#1a1a1a}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #e3e3e0;
padding:10px 16px;display:flex;gap:14px;align-items:center;z-index:5;flex-wrap:wrap}
h1{font-size:15px;margin:0}.stat{font-size:12px;color:#666}.stat b{color:#111}
main{max-width:1100px;margin:0 auto;padding:16px}
.card{background:#fff;border:1px solid #e3e3e0;border-radius:8px;padding:14px;margin-bottom:14px}
.sides{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:820px){.sides{grid-template-columns:1fr}}
.side{border:1px solid #eee;border-radius:6px;padding:8px;background:#fafafa}
svg{max-width:100%;height:auto}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
button{font:inherit;padding:8px 15px;border-radius:6px;border:1px solid #d4d4d0;background:#fff;cursor:pointer}
button.pick{background:#2563eb;color:#fff;border-color:#2563eb}
button.tie{background:#64748b;color:#fff;border-color:#64748b}
button.unsure{background:#a16207;color:#fff;border-color:#a16207}
button.poor{background:#b91c1c;color:#fff;border-color:#b91c1c}
button.play{background:#0f766e;color:#fff;border-color:#0f766e}
.done{background:#fbfdfb;border-color:#cfe6d4}
.badge{font-size:11px;background:#f0f0ee;padding:2px 8px;border-radius:10px}
.badge.set{background:#d1fae5;color:#065f46;font-weight:600}
button.chosen{outline:3px solid #10b981;outline-offset:1px}
.hint{font-size:11px;color:#888;margin-left:6px}
</style></head><body>
<header><h1>Blind re-judging</h1>
<span class="stat">done <b id="c">0</b> / <b id="t">0</b></span>
<span class="stat">These are pairs where two reviewers disagreed. Neither earlier
verdict is shown, and the sides are reshuffled.</span></header>
<main id="list">loading…</main>
<script>
const $=i=>document.getElementById(i);
let V={};
// Voice timbre is fixed per line, not per note.
//
// Choosing the waveform from a note's pitch meant one voice changed instrument
// whenever it crossed the threshold mid-phrase, which reads as a new entry
// rather than the same line continuing -- the opposite of what counterpoint
// playback needs. Each voice now keeps one timbre and one gain for its whole
// duration, and the octave reinforcement that makes a low line audible on small
// speakers is a property of the voice too.
//
// Sawtooth is dropped: all harmonics at 1/n is buzzy, and roughness reads as bad
// music rather than as a bad synthesiser. Triangle carries only odd harmonics at
// 1/n^2, and an added octave partial gives a small speaker something to
// reproduce without the buzz.
""" + SYNTH_JS + """
async function load(){
  const d=await (await fetch('/api/items')).json();
  V=d.verdicts||{}; $('t').textContent=d.items.length;
  $('c').textContent=Object.keys(V).length;
  const L=$('list'); L.innerHTML='';
  d.items.forEach((x,i)=>{
    const el=document.createElement('div');
    el.className='card'+(V[x.arb_id]?' done':''); el.id='c'+i;
    el.innerHTML=`<div class="stat">#${i+1}
      <span class="badge">${V[x.arb_id]||'pending'}</span></div>
      <div class="sides">
        <div class="side"><h3>Candidate 1</h3>${x.left_svg}
          <button class="play" data-i="${i}" data-s="left">▶ play 1</button></div>
        <div class="side"><h3>Candidate 2</h3>${x.right_svg}
          <button class="play" data-i="${i}" data-s="right">▶ play 2</button></div>
      </div>
      <div class="row">
        <button class="pick" data-i="${i}" data-c="left">prefer 1</button>
        <button class="pick" data-i="${i}" data-c="right">prefer 2</button>
        <button class="tie" data-i="${i}" data-c="tie">tie</button>
        <button class="poor" data-i="${i}" data-c="both_poor">both poor</button>
        <button class="unsure" data-i="${i}" data-c="cant_tell">can't tell</button>
      </div>`;
    L.appendChild(el);
  });
  window.D=d;
  L.querySelectorAll('button.play').forEach(b=>b.onclick=()=>
    play(d.items[b.dataset.i][b.dataset.s+'_notes'], 0.55, 0.15));
  L.querySelectorAll('button[data-c]').forEach(b=>b.onclick=()=>
    send(d.items[b.dataset.i].arb_id, b.dataset.c, b.dataset.i));
  // Show which button was already chosen, so a revisited card reads clearly.
  d.items.forEach((x,i)=>{
    if(!V[x.arb_id])return;
    const card=$('c'+i);
    card.querySelectorAll('button[data-c]').forEach(z=>
      z.classList.toggle('chosen', z.dataset.c===V[x.arb_id]));
  });
}
async function send(id,choice,idx){
  const r=await fetch('/api/verdict',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({arb_id:id,choice})});
  if(!r.ok){alert('Could not save that vote.');return;}
  const d=await r.json();
  // Update just this card. Re-rendering the whole list on every vote threw the
  // scroll position back to the top, which with 64 items made the page feel
  // stuck after each click.
  V[id]=choice;
  const card=$('c'+idx);
  card.classList.add('done');
  const b=card.querySelector('.badge');
  b.textContent=choice; b.classList.add('set');
  card.querySelectorAll('button[data-c]').forEach(x=>
    x.classList.toggle('chosen', x.dataset.c===choice));
  $('c').textContent=d.done;
}
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    arb_dir: Path = None  # type: ignore

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/items":
            items = json.loads((self.arb_dir / "items.json").read_text())
            vp = self.arb_dir / "verdicts.json"
            verdicts = json.loads(vp.read_text()) if vp.exists() else {}
            return self._send(200, json.dumps({"items": items, "verdicts": verdicts}))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if urlparse(self.path).path != "/api/verdict":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        vp = self.arb_dir / "verdicts.json"
        cur = json.loads(vp.read_text()) if vp.exists() else {}
        cur[body["arb_id"]] = body["choice"]
        tmp = vp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cur, indent=2, sort_keys=True))
        tmp.replace(vp)
        return self._send(200, json.dumps({"ok": True, "done": len(cur)}))


def serve(arb_dir: Path | None = None, port: int = 8768) -> None:
    Handler.arb_dir = arb_dir or (_repo_root() / "data" / "generated" / "arbitration")
    n = len(json.loads((Handler.arb_dir / "items.json").read_text()))
    print(f"Blind re-judging: {n} disputed pairs -> http://127.0.0.1:{port}/", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="fugue-data arbitrate")
    ap.add_argument("--port", type=int, default=8768)
    args = ap.parse_args()
    serve(port=args.port)
