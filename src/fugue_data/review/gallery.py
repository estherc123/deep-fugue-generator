"""A plain listening page: a list of fragments, each with audio and a piano roll.

No judgement, no verdicts written -- this exists so a person can hear what a
generator actually sounds like before deciding whether it is worth pursuing.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .synth import SYNTH_JS

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Listening gallery</title><style>
body{font:14px system-ui;margin:0;background:#faf9f7;color:#1a1a1a}
header{padding:16px 24px;background:#fff;border-bottom:1px solid #e3e0da}
h1{font-size:17px;margin:0 0 4px}
.sub{color:#6b6660;font-size:13px}
.item{background:#fff;margin:14px 24px;padding:14px 16px;border:1px solid #e3e0da;border-radius:8px}
.hd{display:flex;align-items:baseline;gap:12px;margin-bottom:8px}
.nm{font-weight:600}
.tag{font-size:11px;padding:2px 8px;border-radius:99px;background:#eceae5;color:#555}
.meta{color:#6b6660;font-size:12px;margin-left:auto;font-variant-numeric:tabular-nums}
button{font:13px system-ui;padding:6px 14px;border:1px solid #c9c4bb;background:#fff;
  border-radius:6px;cursor:pointer}
button:hover{background:#f2f0ec}
svg{max-width:100%;height:auto;display:block;margin-top:8px}
</style></head><body>
<header><h1 id="ttl">Listening gallery</h1><div class="sub" id="sub"></div></header>
<div id="list"></div>
<script>
""" + SYNTH_JS + """
const $=id=>document.getElementById(id);
let DATA=null;
async function load(){
  const r=await fetch('/data'); DATA=await r.json();
  $('ttl').textContent=DATA.title||'Listening gallery';
  $('sub').textContent=DATA.subtitle||'';
  const L=$('list'); L.innerHTML='';
  DATA.items.forEach((it,i)=>{
    const d=document.createElement('div'); d.className='item';
    const hd=document.createElement('div'); hd.className='hd';
    const nm=document.createElement('span'); nm.className='nm'; nm.textContent=it.name;
    hd.appendChild(nm);
    if(it.tag){const t=document.createElement('span');t.className='tag';t.textContent=it.tag;hd.appendChild(t);}
    const b=document.createElement('button'); b.textContent='play';
    b.onclick=()=>play(it.notes, it.secPerQuarter||0.55, 0.16);
    hd.appendChild(b);
    if(it.meta){const m=document.createElement('span');m.className='meta';m.textContent=it.meta;hd.appendChild(m);}
    d.appendChild(hd);
    if(it.svg){const w=document.createElement('div');w.innerHTML=it.svg;d.appendChild(w);}
    L.appendChild(d);
  });
}
load();
</script></body></html>"""


def serve(data_path: Path, port: int = 8771) -> None:
    payload = json.loads(Path(data_path).read_text())

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/data":
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            else:
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"listening gallery: http://127.0.0.1:{port}/  ({len(payload['items'])} items)")
    HTTPServer(("127.0.0.1", port), H).serve_forever()
