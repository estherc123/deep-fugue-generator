"""Local preference-pair review application (Spec §18.5).

Required behaviour, all present:
  - candidate source is blinded (the UI never names which sample is which);
  - left/right presentation is randomized per pair from `presentation_seed`;
  - playback and notation for both candidates;
  - ties and `both_poor` are first-class answers, so no winner is forced;
  - a sample of pairs is repeated to estimate reviewer consistency.

Run:  python -m fugue_data.review.pairs_app  [--port 8766]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..normalize.rational import parse_frac
from .render import piano_roll_svg
from .store import assigned_slice, read_jsonl, update


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class PairStore:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "data" / "generated" / "preference_candidates" / "pairs.jsonl"
        self._scores: dict[str, dict] = {}
        self._scores_mtime: float | None = None

    def _load_scores(self) -> dict[str, dict]:
        # Reload when the file changes. Caching on first use meant a long-running
        # server kept serving a stale candidate set: after the pool was
        # regenerated, every pair referencing a new candidate resolved to None and
        # was silently dropped from the page -- the reviewer simply never saw the
        # new material and nothing said so.
        cpath = (self.root / "data" / "generated" / "preference_candidates"
                 / "candidates.jsonl")
        mtime = cpath.stat().st_mtime if cpath.exists() else None
        if self._scores and mtime == self._scores_mtime:
            return self._scores
        self._scores = {}
        self._scores_mtime = mtime
        p = cpath
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._scores[r["sample_id"]] = r["score"]
        return self._scores

    def load(self) -> list[dict]:
        return read_jsonl(self.path)

    def revision_of(self) -> dict[str, str]:
        """pair_id -> the generator revision that produced its candidates."""
        scores_by_rev = {}
        cpath = (self.root / "data" / "generated" / "preference_candidates"
                 / "candidates.jsonl")
        if cpath.exists():
            for line in cpath.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    scores_by_rev[r["sample_id"]] = r.get("generator_revision", "unknown")
        out = {}
        for pr in self.load():
            out[pr["pair_id"]] = scores_by_rev.get(pr["candidate_a_id"], "unknown")
        return out

    def agent_claimed(self) -> set[str]:
        """Pairs an agent has already judged or is currently assigned.

        Without this the human queue offered every pair no human had labelled --
        2,896 of them, of which only 599 were actually reserved. A reviewer would
        have spent most of their effort on pairs an agent had already covered or
        was working on, which is both wasted and, for an assigned pair, a second
        judgment that is not independent.
        """
        base = self.root / "data" / "generated" / "preference_candidates"
        claimed: set[str] = set()
        ap = base / "agent_labels.jsonl"
        if ap.exists():
            claimed |= {json.loads(l)["pair_id"] for l in ap.read_text().splitlines()
                        if l.strip()}
        bdir = self.root / "data" / "generated" / "agent_batches"
        if bdir.exists():
            for f in sorted(bdir.glob("agent_batch_*.json")):
                try:
                    doc = json.loads(f.read_text())
                except Exception:
                    continue
                claimed |= {i["pair_id"] for i in doc.get("items", [])}
        return claimed

    def label(self, pair_id: str, label: str, comment: str, reviewer: str) -> dict | None:
        """Record one decision atomically across processes.

        Read-modify-write happens inside an exclusive lock, so two reviewers
        submitting at the same instant cannot each write a file built from a
        stale read and silently drop one of the labels.
        """
        if not (reviewer or "").strip():
            # A label with no reviewer is unattributable, and §18.6 turns on
            # being able to say a *person* made it. Refuse rather than store it.
            raise ValueError("reviewer is required")

        def mutate(pairs):
            hit = None
            for p in pairs:
                if p["pair_id"] == pair_id:
                    p["label"] = (label if label in ("A", "B", "tie", "both_poor",
                                                 "cant_tell") else None)
                    p["review_status"] = "reviewed" if p["label"] else "pending"
                    p["review_comment"] = comment or None
                    p["reviewer"] = reviewer.strip()
                    p["reviewed_at_utc"] = datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ")
                    hit = p
            return hit

        return update(self.path, mutate)

    def payload(self, p: dict) -> dict | None:
        scores = self._load_scores()
        a, b = scores.get(p["candidate_a_id"]), scores.get(p["candidate_b_id"])
        if a is None or b is None:
            return None
        # §18.5: randomize which candidate is shown on the left, from the stored
        # seed so the layout is stable across reloads but unpredictable.
        rng = random.Random(p.get("presentation_seed") or 0)
        flip = rng.random() < 0.5
        left_key, right_key = ("B", "A") if flip else ("A", "B")
        docs = {"A": a, "B": b}

        def side(key):
            doc = docs[key]
            end = parse_frac(doc["quarter_length_total"])
            return {
                # The key is never sent: the reviewer cannot tell A from B.
                "svg": piano_roll_svg(doc, Fraction(0), end, None, ""),
                "notes": sorted(
                    [{"m": n["midi"], "t": float(parse_frac(n["onset_q"])),
                      "d": float(parse_frac(n["duration_q"])), "v": i}
                     for i, v in enumerate(doc["voices"]) for n in v["notes"]],
                    key=lambda r: (r["t"], r["m"])),
            }

        return {
            "pair_id": p["pair_id"],
            "prompt_id": p["prompt_id"],
            "split": p["split"],
            "selection_reason": p["selection_reason"],
            "review_status": p["review_status"],
            "label": p["label"],
            "review_comment": p["review_comment"],
            "left": side(left_key), "right": side(right_key),
            "left_is": left_key, "right_is": right_key,
        }


from .synth import SYNTH_JS

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Preference review — fugue-data</title><style>
:root{color-scheme:light}
body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#f6f6f5;color:#1a1a1a}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #e3e3e0;padding:10px 16px;
display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:5}
h1{font-size:15px;margin:0;font-weight:650}.stat{font-size:12px;color:#666}.stat b{color:#111}
main{max-width:1200px;margin:0 auto;padding:16px}
.card{background:#fff;border:1px solid #e3e3e0;border-radius:8px;padding:14px;margin-bottom:16px}
.meta{font-size:12px;color:#555;margin-bottom:10px}
.meta code{background:#f0f0ee;padding:1px 5px;border-radius:3px}
.sides{display:grid;grid-template-columns:1fr 1fr;gap:12px}
/* Phones: stack the candidates and make the controls thumb-sized. A two-column
   grid on a 390px screen renders both piano rolls unreadably narrow. */
@media (max-width:820px){
  .sides{grid-template-columns:1fr}
  .side h3{font-size:14px}
  header{padding:8px 10px;gap:8px}
  main{padding:10px}
  button{padding:10px 16px;font-size:15px}
  .row{gap:6px}
  input[type=text]{min-width:140px}
  .kbdhint{display:none}
}
.side{border:1px solid #eee;border-radius:6px;padding:8px;background:#fafafa}
.side h3{margin:0 0 6px;font-size:13px;font-weight:600}
svg{max-width:100%;height:auto}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:12px}
button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid #d4d4d0;background:#fff;cursor:pointer}
button:hover{background:#f3f3f1}
button.pick{background:#2563eb;color:#fff;border-color:#2563eb}
button.tie{background:#64748b;color:#fff;border-color:#64748b}
button.unsure{background:#a16207;color:#fff;border-color:#a16207}
button.poor{background:#b91c1c;color:#fff;border-color:#b91c1c}
button.play{background:#0f766e;color:#fff;border-color:#0f766e}
input[type=text]{font:inherit;padding:6px 9px;border:1px solid #d4d4d0;border-radius:6px;flex:1;min-width:200px}
.badge{font-size:11px;padding:2px 8px;border-radius:10px;background:#f0f0ee;color:#555}
.done{opacity:.55}
kbd{background:#f0f0ee;border:1px solid #ddd;border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:11px}
</style></head><body>
<header><h1>Preference review</h1>
<span class="stat">reviewed <b id="c-rev">0</b> / 500 target</span>
<span class="stat">pending <b id="c-pen">0</b></span>
<span class="stat" style="margin-left:auto">reviewer <input type="text" id="reviewer" style="min-width:120px"></span>
<span class="stat kbdhint"><kbd>1</kbd> left <kbd>2</kbd> right <kbd>t</kbd> tie <kbd>p</kbd> both poor <kbd>q</kbd>/<kbd>w</kbd> play</span>
<span class="stat" id="saved" style="margin-left:8px;color:#2a7"></span>
</header>
<div id="warn" style="display:none;padding:10px 16px;background:#fde8c8;
     border-bottom:1px solid #e0c48a;font:14px system-ui">
  Enter a <b>reviewer name</b> above before labelling &mdash; a label with no
  reviewer cannot be used, so the buttons stay disabled until you do.
</div>
<main id="list">loading…</main>
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
async function load(){
  const list=$('list');
  let d;
  try{
    const who=encodeURIComponent($('reviewer').value.trim());
    const r=await fetch('/api/pairs?reviewer='+who);
    if(!r.ok) throw new Error('HTTP '+r.status);
    d=await r.json();
  }catch(err){
    // Surface the failure instead of leaving "loading..." on screen forever.
    list.innerHTML='<div class="card"><b>Could not load pairs.</b><br>'+
      String(err)+'<br><small>Check the terminal running the app.</small></div>';
    return;
  }
  $('c-rev').textContent=d.reviewed; $('c-pen').textContent=d.pending;
  list.innerHTML='';
  d.pairs.forEach((p,i)=>{
    const el=document.createElement('div');
    el.className='card'+(p.review_status==='reviewed'?' done':'');el.id='card-'+i;
    el.innerHTML=`<div class="meta">prompt <code>${p.prompt_id}</code> · split <b>${p.split}</b>
      · chosen because <b>${p.selection_reason}</b>
      <span class="badge">${p.review_status}${p.label?': '+p.label:''}</span></div>
      <div class="sides">
        <div class="side"><h3>Candidate 1</h3>${p.left.svg}
          <button class="play" data-i="${i}" data-s="left">▶ play 1</button></div>
        <div class="side"><h3>Candidate 2</h3>${p.right.svg}
          <button class="play" data-i="${i}" data-s="right">▶ play 2</button></div>
      </div>
      <div class="row">
        <button class="pick" data-i="${i}" data-l="left">prefer 1</button>
        <button class="pick" data-i="${i}" data-l="right">prefer 2</button>
        <button class="tie" data-i="${i}" data-l="tie">tie</button>
        <button class="poor" data-i="${i}" data-l="both_poor">both poor</button>
        <button class="unsure" data-i="${i}" data-l="cant_tell">can't tell</button>
        <input type="text" class="comment" placeholder="optional comment"
               value="${(p.review_comment||'').replace(/"/g,'&quot;')}">
      </div>`;
    list.appendChild(el);
  });
  window.D=d;
  list.querySelectorAll('button.play').forEach(b=>b.onclick=()=>
    play(d.pairs[b.dataset.i][b.dataset.s].notes, 0.5, 0.15));
  list.querySelectorAll('button[data-l]').forEach(b=>b.onclick=()=>
    decide(b.dataset.i,b.dataset.l));
  syncGate();
}
function reviewerName(){ const r=$('reviewer'); return r? r.value.trim() : ''; }
function syncGate(){
  const ok=!!reviewerName();
  const w=$('warn'); if(w) w.style.display = ok ? 'none' : 'block';
  document.querySelectorAll('button[data-l]').forEach(b=>{
    b.disabled=!ok; b.style.opacity = ok ? '1' : '0.45';
    b.style.cursor = ok ? 'pointer' : 'not-allowed';
  });
}
function flash(msg,bad){
  const s=$('saved'); if(!s) return;
  s.style.color = bad ? '#c33' : '#2a7'; s.textContent = msg;
  setTimeout(()=>{ if(s.textContent===msg) s.textContent=''; }, 2500);
}
async function decide(i,choice){
  const who=reviewerName();
  if(!who){ syncGate(); const r=$('reviewer'); if(r) r.focus();
            flash('enter a reviewer name first', true); return; }
  const p=window.D.pairs[i],card=document.getElementById('card-'+i);
  // Translate screen side back to the underlying candidate, which the reviewer
  // never sees: this is what keeps the comparison blind.
  const label = choice==='left'?p.left_is : choice==='right'?p.right_is : choice;
  if(card){ card.style.opacity='0.5'; }
  try{
    const r=await fetch('/api/label',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({pair_id:p.pair_id,label,
        comment:card?card.querySelector('.comment').value:'',
        reviewer:who})});
    if(!r.ok){ throw new Error('HTTP '+r.status+' '+(await r.text()).slice(0,120)); }
  }catch(err){
    if(card){ card.style.opacity='1'; }
    flash('save failed: '+String(err).slice(0,80), true);
    return;
  }
  try{localStorage.setItem('fugue_reviewer',who)}catch(e){}
  flash('saved: '+label);
  await load();
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  const cards=[...document.querySelectorAll('.card')],mid=window.innerHeight/2;
  const idx=cards.findIndex(c=>c.getBoundingClientRect().bottom>mid);
  if(idx<0)return;
  if(e.key==='1')decide(idx,'left'); if(e.key==='2')decide(idx,'right');
  if(e.key==='t')decide(idx,'tie'); if(e.key==='p')decide(idx,'both_poor');
  if(e.key==='q')play(window.D.pairs[idx].left.notes, 0.5, 0.15);
  if(e.key==='w')play(window.D.pairs[idx].right.notes, 0.5, 0.15);
});
try{const w=localStorage.getItem('fugue_reviewer'); if(w)$('reviewer').value=w}catch(e){}
$('reviewer').addEventListener('input', syncGate);
syncGate();
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    store: PairStore = None  # type: ignore
    page_limit = 40
    reviewers = 1          # >1 partitions the queue between concurrent reviewers
    human_only = True      # hide pairs an agent has judged or been assigned

    def log_message(self, *a):
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
        if u.path == "/api/pairs":
            allp = self.store.load()
            reviewed = sum(1 for p in allp if p["review_status"] == "reviewed")
            pending = [p for p in allp if p["review_status"] != "reviewed"]
            if self.human_only:
                claimed = self.store.agent_claimed()
                pending = [p for p in pending if p["pair_id"] not in claimed]
            # Split the queue between reviewers so two people working at once
            # cover twice the ground instead of labelling the same pairs.
            q = parse_qs(u.query)
            who = (q.get("reviewer") or [""])[0].strip()
            if who and self.reviewers > 1:
                keep = set(assigned_slice([p["pair_id"] for p in pending],
                                          who, self.reviewers))
                pending = [p for p in pending if p["pair_id"] in keep]
            # Shuffle, deterministically per reviewer, and interleave the
            # generator revisions.
            #
            # File order is by pair_id, and the app served the first N of it. That
            # meant new material appended to the pool sat at the end and was never
            # reached; a pair the reviewer skipped reappeared at the top on every
            # reload; and once the pool spans two generator revisions, a reviewer
            # would work through one generation before ever seeing the other,
            # letting their standard drift between the two.
            #
            # Seeding on the reviewer's name keeps each person's order stable
            # across reloads -- so the page does not reshuffle under them -- while
            # giving different reviewers different orders.
            rev = self.store.revision_of()
            seed = who or "anon"
            def rank(pr):
                h = hashlib.sha256(f"{seed}:{pr['pair_id']}".encode()).hexdigest()
                return int(h[:12], 16)
            buckets: dict[str, list] = {}
            for pr in sorted(pending, key=rank):
                buckets.setdefault(rev.get(pr["pair_id"], "unknown"), []).append(pr)
            interleaved = []
            keys = sorted(buckets)
            for i in range(max((len(v) for v in buckets.values()), default=0)):
                for k in keys:
                    if i < len(buckets[k]):
                        interleaved.append(buckets[k][i])
            pending = interleaved

            out = []
            for p in pending[:self.page_limit]:
                pay = self.store.payload(p)
                if pay:
                    out.append(pay)
            return self._send(200, json.dumps(
                {"pairs": out, "reviewed": reviewed, "pending": len(pending)}))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if urlparse(self.path).path != "/api/label":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            rec = self.store.label(body.get("pair_id", ""), body.get("label", ""),
                                   body.get("comment", ""), body.get("reviewer", ""))
        except ValueError as e:
            return self._send(400, json.dumps({"error": str(e)}))
        if rec is None:
            return self._send(404, json.dumps({"error": "unknown pair"}))
        return self._send(200, json.dumps({"ok": True, "record": rec}))


def serve(root: Path | None = None, port: int = 8766, *, host: str = "127.0.0.1",
          reviewers: int = 1, human_only: bool = True) -> None:
    root = root or _repo_root()
    Handler.store = PairStore(root)
    Handler.reviewers = reviewers
    Handler.human_only = human_only
    n = len(Handler.store.load())
    print(f"Preference review: {n} pairs -> http://127.0.0.1:{port}/",
          flush=True)
    HTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="fugue-data review-pairs")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--reviewers", type=int, default=1,
                    help="partition the queue between this many concurrent reviewers")
    ap.add_argument("--include-agent-pairs", action="store_true",
                    help="also show pairs an agent has judged or been assigned")
    args = ap.parse_args()
    serve(port=args.port, host=args.host, reviewers=args.reviewers,
          human_only=not args.include_agent_pairs)
