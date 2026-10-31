"""Local excerpt review application (Spec §14.5).

Stdlib only -- no Streamlit, no Verovio, no browser plugin. Notation is the
piano-roll SVG rendered at proposal time; playback is a small Web Audio
synthesiser driven by the excerpt's own note list, so "play" works offline in
any browser. The MIDI preview is also linked for an external player.

Required actions, all present: play, view notation, approve, reject, request
correction, assign tags, add a short comment.

Run:  python -m fugue_data.review.app  [--port 8765]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from ..normalize.rational import parse_frac
from .render import excerpt_notes


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class Store:
    """Review records on disk. Writes are atomic so an interrupted save cannot
    corrupt the decisions already made."""

    def __init__(self, root: Path):
        self.root = root
        self.path = root / "data" / "proposed" / "excerpts.jsonl"
        self.cfg = yaml.safe_load((root / "config" / "excerpts.yaml").read_text())

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def save(self, records: list[dict]) -> None:
        tmp = self.path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
            for r in sorted(records, key=lambda r: r["excerpt_id"])))
        tmp.replace(self.path)

    def decide(self, excerpt_id: str, status: str, tags: list[str], comment: str,
               reviewer: str) -> dict | None:
        records = self.load()
        hit = None
        for r in records:
            if r["excerpt_id"] == excerpt_id:
                r["review_status"] = status
                r["review_tags"] = tags
                r["review_comment"] = comment or None
                r["reviewer"] = reviewer or None
                r["reviewed_at_utc"] = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ")
                hit = r
        if hit:
            self.save(records)
        return hit

    def note_payload(self, rec: dict) -> list[dict]:
        """Notes of one excerpt, as the browser synthesiser needs them."""
        p = self.root / "data" / "canonical" / f"{rec['piece_id']}.json"
        score = json.loads(p.read_text())
        start, end = parse_frac(rec["start_onset_q"]), parse_frac(rec["end_onset_q"])
        out = []
        for n in excerpt_notes(score, start, end):
            if n.get("tie") in ("continue", "stop"):
                continue
            out.append({"m": n["midi"],
                        "t": float(max(parse_frac(n["onset_q"]), start) - start),
                        "d": float(n["_clip_off"] - n["_clip_on"]),
                        "v": n["_voice_index"]})
        return sorted(out, key=lambda r: (r["t"], r["m"]))


from .synth import SYNTH_JS

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Excerpt review — fugue-data</title><style>
:root{color-scheme:light}
body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#f6f6f5;color:#1a1a1a}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #e3e3e0;
padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;z-index:5}
h1{font-size:15px;margin:0;font-weight:650}
.stat{font-size:12px;color:#666}.stat b{color:#1a1a1a}
main{max-width:960px;margin:0 auto;padding:16px}
.card{background:#fff;border:1px solid #e3e3e0;border-radius:8px;padding:14px;margin-bottom:14px}
.meta{font-size:12px;color:#555;margin-bottom:8px}
.meta code{background:#f0f0ee;padding:1px 5px;border-radius:3px}
.reasons span{display:inline-block;background:#eef2ff;color:#3730a3;border-radius:10px;
padding:1px 9px;font-size:11px;margin-right:5px}
svg{max-width:100%;height:auto;border:1px solid #eee;border-radius:5px;background:#fafafa}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}
button{font:inherit;padding:6px 13px;border-radius:6px;border:1px solid #d4d4d0;
background:#fff;cursor:pointer}
button:hover{background:#f3f3f1}
button.ok{background:#059669;color:#fff;border-color:#059669}
button.no{background:#dc2626;color:#fff;border-color:#dc2626}
button.fix{background:#d97706;color:#fff;border-color:#d97706}
button.play{background:#2563eb;color:#fff;border-color:#2563eb}
input[type=text]{font:inherit;padding:6px 9px;border:1px solid #d4d4d0;border-radius:6px;flex:1;min-width:220px}
label.tag{font-size:12px;background:#f0f0ee;border-radius:12px;padding:3px 9px;cursor:pointer}
label.tag input{margin-right:4px}
.done{opacity:.5}
.badge{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:600}
.b-pending{background:#f0f0ee;color:#666}.b-approved{background:#d1fae5;color:#065f46}
.b-rejected{background:#fee2e2;color:#991b1b}.b-correction_required{background:#fef3c7;color:#92400e}
kbd{background:#f0f0ee;border:1px solid #ddd;border-bottom-width:2px;border-radius:4px;
padding:0 5px;font-size:11px}
</style></head><body>
<header><h1>Excerpt review</h1>
<span class="stat">Play it. Does it sound like a complete musical thought, or does it
start or stop in a strange place? That is the only question.</span>
<span class="stat">approved <b id="c-approved">0</b> / target 50–100</span>
<span class="stat">pending <b id="c-pending">0</b></span>
<span class="stat">rejected <b id="c-rejected">0</b></span>
<span class="stat">correction <b id="c-correction">0</b></span>
<span class="stat" style="margin-left:auto">reviewer
<input type="text" id="reviewer" style="min-width:130px" placeholder="your name"></span>
<span class="stat" id="saved" style="margin-left:8px;color:#2a7"></span>
<span class="stat"><kbd>a</kbd> approve <kbd>r</kbd> reject <kbd>c</kbd> correction
<kbd>space</kbd> play</span>
</header>
<div id="warn" style="display:none;position:sticky;top:0;z-index:50;
     padding:10px 16px;background:#fde8c8;border-bottom:1px solid #e0c48a;
     font:14px system-ui;cursor:pointer" onclick="document.getElementById('reviewer').focus()">
  Enter a <b>reviewer name</b> above before approving &mdash; §27 forbids counting
  a review nobody owns, so the buttons stay disabled until you do.
</div><main id="list">loading…</main>
<script>
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
const $=id=>document.getElementById(id);
function syncGate(){
  const ok=!!(($('reviewer')||{}).value||'').trim();
  const w=$('warn'); if(w) w.style.display = ok ? 'none' : 'block';
  document.querySelectorAll('button[data-s]').forEach(b=>{
    b.disabled=!ok; b.style.opacity = ok ? '1' : '0.35';
    b.style.cursor = ok ? 'pointer' : 'not-allowed';
    b.style.filter = ok ? 'none' : 'grayscale(1)';
    b.title = ok ? '' : 'Enter your reviewer name at the top first';
  });
}
function flash(msg,bad){
  const s=$('saved'); if(!s) return;
  s.style.color = bad ? '#c33' : '#2a7'; s.textContent = msg;
  setTimeout(()=>{ if(s.textContent===msg) s.textContent=''; }, 2500);
}
async function load(){
  const list=$('list');
  let d;
  try{
    const r=await fetch('/api/excerpts');
    if(!r.ok) throw new Error('HTTP '+r.status);
    d=await r.json();
  }catch(err){
    // Surface the failure instead of leaving "loading..." on screen forever.
    list.innerHTML='<div class="card"><b>Could not load excerpts.</b><br>'+
      String(err)+'<br><small>Check the terminal running the app.</small></div>';
    return;
  }
  const counts={pending:0,approved:0,rejected:0,correction_required:0};
  d.records.forEach(x=>counts[x.review_status]=(counts[x.review_status]||0)+1);
  $('c-approved').textContent=counts.approved; $('c-pending').textContent=counts.pending;
  $('c-rejected').textContent=counts.rejected;
  $('c-correction').textContent=counts.correction_required;
  list.innerHTML='';
  d.records.forEach((x,i)=>{
    const el=document.createElement('div');
    el.className='card'+(x.review_status!=='pending'?' done':'');
    el.id='card-'+i;
    el.innerHTML=`
      <div class="meta"><code>${x.piece_id}</code> mm.${x.start_measure}–${x.end_measure}
        · ${x.bars} bars · ${x.voice_count} voices · ${x.note_count} notes
        · split <b>${x.split||'—'}</b>
        ${x.clipped_boundary_notes?` · <span title="notes truncated by the window end">${x.clipped_boundary_notes} clipped</span>`:''}
        <span class="badge b-${x.review_status}">${x.review_status}</span></div>
      <div class="reasons">${x.proposal_reasons.map(r=>`<span>${r}</span>`).join('')}</div>
      <div class="svg">${x.svg||''}</div>
      <div class="row">
        <button class="play" data-i="${i}">▶ play</button>
        ${x.midi_preview_path?`<a href="/render/${x.midi_preview_path}" download>MIDI</a>`:''}
        ${d.tags.map(t=>`<label class="tag"><input type="checkbox" value="${t}"
           ${(x.review_tags||[]).includes(t)?'checked':''}>${t}</label>`).join('')}
      </div>
      <div class="row">
        <input type="text" class="comment" placeholder="short comment"
               value="${(x.review_comment||'').replace(/"/g,'&quot;')}">
        <button class="ok" data-s="approved" data-i="${i}">sounds right &mdash; keep</button>
        <button class="no" data-s="rejected" data-i="${i}">sounds wrong &mdash; drop</button>
        <button class="fix" data-s="correction_required" data-i="${i}">not sure</button>
      </div>`;
    list.appendChild(el);
  });
  window.DATA=d;
  list.querySelectorAll('button.play').forEach(b=>b.onclick=()=>play(d.records[b.dataset.i].notes, 0.55, 0.16));
  list.querySelectorAll('button[data-s]').forEach(b=>b.onclick=()=>decide(b.dataset.i,b.dataset.s));
  syncGate();
}
async function decide(i,status){
  const x=window.DATA.records[i], card=document.getElementById('card-'+i);
  const tags=[...card.querySelectorAll('.tag input:checked')].map(e=>e.value);
  const comment=card.querySelector('.comment').value;
  const who=($('reviewer').value||'').trim();
  if(!who){ syncGate(); $('reviewer').focus();
            flash('enter a reviewer name first', true); return; }
  if(card) card.style.opacity='0.5';
  try{
    const r=await fetch('/api/review',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({excerpt_id:x.excerpt_id,status,tags,comment,reviewer:who})});
    if(!r.ok) throw new Error('HTTP '+r.status+' '+(await r.text()).slice(0,120));
  }catch(err){
    if(card) card.style.opacity='1';
    flash('save failed: '+String(err).slice(0,80), true);
    return;
  }
  try{localStorage.setItem('fugue_reviewer',who)}catch(e){}
  flash('saved: '+status);
  await load();
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  const cards=[...document.querySelectorAll('.card')];
  const mid=window.innerHeight/2;
  const idx=cards.findIndex(c=>{const r=c.getBoundingClientRect();return r.bottom>mid});
  if(idx<0)return;
  if(e.key==='a')decide(idx,'approved');
  if(e.key==='r')decide(idx,'rejected');
  if(e.key==='c')decide(idx,'correction_required');
  if(e.key===' '){e.preventDefault();play(window.DATA.records[idx].notes, 0.55, 0.16)}
});
try{const w=localStorage.getItem('fugue_reviewer'); if(w)$('reviewer').value=w}catch(e){}
$('reviewer').addEventListener('input', syncGate);
syncGate();
if(!($('reviewer').value||'').trim()) $('reviewer').focus();
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    ear_mode = False
    shortlist: set = set()
    store: Store = None  # type: ignore

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/excerpts":
            recs = [r for r in self.store.load() if not r.get("auto_rejected")]
            # Two-stage review: an agent judges the analysis (does the window
            # contain what it claims, is the cut musically placed), and only the
            # windows it cleared reach a person, who judges one thing by ear --
            # does this sound like a complete musical thought. §14.6 needs a human
            # approval, not a human analysis.
            if self.shortlist:
                recs = [r for r in recs if r["excerpt_id"] in self.shortlist]
            q = parse_qs(u.query)
            if q.get("status"):
                recs = [r for r in recs if r["review_status"] in q["status"]]
            out = []
            base = self.store.root / "data" / "proposed"
            for r in recs:
                svg = ""
                if r.get("render_path") and (base / r["render_path"]).exists():
                    svg = (base / r["render_path"]).read_text()
                out.append({**r, "svg": svg, "notes": self.store.note_payload(r)})
            return self._send(200, json.dumps(
                {"records": out, "tags": (EAR_TAGS if self.ear_mode
                                         else self.store.cfg["review"]["tags"])}))
        if u.path.startswith("/render/"):
            rel = u.path[len("/render/"):]
            f = (self.store.root / "data" / "proposed" / rel).resolve()
            base = (self.store.root / "data" / "proposed").resolve()
            if base in f.parents and f.exists():
                ctype = "image/svg+xml" if f.suffix == ".svg" else "audio/midi"
                return self._send(200, f.read_bytes(), ctype)
            return self._send(404, json.dumps({"error": "not found"}))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if urlparse(self.path).path != "/api/review":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        rec = self.store.decide(
            body.get("excerpt_id", ""), body.get("status", "pending"),
            body.get("tags", []), body.get("comment", ""), body.get("reviewer", ""))
        if rec is None:
            return self._send(404, json.dumps({"error": "unknown excerpt"}))
        return self._send(200, json.dumps({"ok": True, "record": rec}))


# Plain-language tags for ear review. The configured vocabulary -- clean_cadence,
# strong_counterpoint, awkward_boundary -- asks for an analytical judgement this
# reviewer has twice said they cannot make, and §14.6 does not require tags at all:
# it requires a person to approve. The agent supplies the analysis; the person
# supplies the ear.
EAR_TAGS = [
    "sounds complete",
    "starts oddly",
    "stops oddly",
    "too short to tell",
    "sounds wrong (maybe a data bug)",
]


def serve(root: Path | None = None, port: int = 8765, *,
          shortlist: Path | None = None, ear_mode: bool = False) -> None:
    root = root or _repo_root()
    Handler.store = Store(root)
    Handler.ear_mode = ear_mode
    Handler.shortlist = set()
    if shortlist:
        import json as _json
        Handler.shortlist = set(_json.loads(Path(shortlist).read_text()))
    n = len([r for r in Handler.store.load() if not r.get("auto_rejected")])
    print(f"Excerpt review: {n} candidates -> http://127.0.0.1:{port}/",
          flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="fugue-data review-excerpts")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    serve(port=args.port)
