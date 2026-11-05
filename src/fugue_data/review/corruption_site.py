"""A local listening page for corrupted-Bach pairs.

Each pair is drawn as two piano rolls -- original over corrupted, one colour per
voice -- with a WAV for each side, so the damage can be both seen and heard.
A blind mode hides which side is Bach and asks the listener to pick, which is
the honest test of whether a corruption is audible at all.

    PYTHONPATH=src python -m fugue_data.review.corruption_site --per-type 4
    python -m http.server 8777 --bind 127.0.0.1 -d data/review/corruption_site
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from fractions import Fraction
from pathlib import Path

from .export_audio import QUARTER_SECONDS, render_wav

OPS = ["segment_shuffle", "foreign_key_area", "solver_filler", "rhythm_flatten",
       "subject_return_removal"]

TITLES = {
    "segment_shuffle": "Shuffled bars",
    "foreign_key_area": "Wrong key area",
    "solver_filler": "Solver filler",
    "rhythm_flatten": "Flattened rhythm",
    "subject_return_removal": "Missing subject return",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _q(x) -> float:
    return float(Fraction(x))


def _describe(r: dict) -> str:
    p, bar = r["params"], _q(r["window"]["length_q"]) / r["window"]["n_bars"]
    # Bars counted within the window (bar 1 = the window's first bar).
    s, e = (round(_q(x) / bar) + 1 for x in p["span"])
    span = f"window bars {s}–{e - 1}" if e - s > 1 else f"window bar {s}"
    op = r["corruption_id"]
    if op == "segment_shuffle":
        order = ", ".join(str(int(s) + k) for k in p["permutation"])
        return f"{span} replayed in the order {order}, all voices together."
    if op == "foreign_key_area":
        names = {1: "up a semitone", -1: "down a semitone", 6: "up a tritone",
                 -6: "down a tritone"}
        return f"{span} moved {names.get(p['shift'], str(p['shift']))} in every voice."
    if op == "solver_filler":
        return f"{span} replaced by CP-SAT solver material ({p['filler_sample_id']}) in the local key."
    if op == "rhythm_flatten":
        return (f"{span} reduced to one attack per beat "
                f"({p['notes_before']} → {p['notes_after']} notes).")
    if op == "subject_return_removal":
        return (f"Subject entry in {p['voice_id'].replace('_', ' ')} (entry #{p['entry_index'] + 1}) "
                f"overwritten with free material from elsewhere in the piece.")
    return ""


def _ordered(compact: dict, order: list[str]) -> list[list]:
    return [[[_q(o), _q(d), m] for o, d, m in compact[v]] for v in order]


def _score_for_wav(compact: dict, order: list[str], length_q: str) -> dict:
    return {"quarter_length_total": length_q,
            "voices": [{"voice_id": v, "notes": [
                {"onset_q": o, "duration_q": d, "midi": m, "tie": None}
                for o, d, m in compact[v]]} for v in order]}


def select(recs: list[dict], per_type: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for op in OPS:
        pool = [r for r in recs if r["corruption_id"] == op]
        fugues = [r for r in pool if r["corpus"] == "fugue"]
        chorales = [r for r in pool if r["corpus"] == "chorale"]
        n_ch = min(len(chorales), per_type // 4)          # mostly fugues
        pick = rng.sample(fugues, min(len(fugues), per_type - n_ch))
        pick += rng.sample(chorales, min(len(chorales), per_type - len(pick)))
        out += pick
    return out


def build(root: Path | None = None, *, per_type: int = 4, seed: int = 11,
          out_dir: Path | None = None) -> Path:
    root = root or _repo_root()
    out_dir = out_dir or root / "data" / "review" / "corruption_site"
    pairs_path = root / "data" / "generated" / "corrupted" / "pairs.jsonl"
    recs = [json.loads(l) for l in pairs_path.read_text().splitlines() if l.strip()]
    chosen = select(recs, per_type, seed)
    rng = random.Random(seed + 1)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    items = []
    for i, r in enumerate(chosen):
        # Highest voice first, by mean pitch of the original.
        def mean(v):
            ev = r["original"][v]
            return -sum(m for _, _, m in ev) / len(ev) if ev else 0
        order = sorted(r["original"], key=mean)
        key = f"p{i:02d}"
        for side in ("original", "corrupted"):
            render_wav(_score_for_wav(r[side], order, r["window"]["length_q"]),
                       out_dir / "audio" / f"{key}_{side}.wav")
        orig = {v: Counter(map(tuple, r["original"][v])) for v in order}
        changed = [[1 if orig[v][tuple(n)] == 0 else 0 for n in r["corrupted"][v]]
                   for v in order]
        bar = _q(r["window"]["length_q"]) / r["window"]["n_bars"]
        items.append({
            "key": key, "op": r["corruption_id"], "title": TITLES[r["corruption_id"]],
            "piece": r["piece_id"], "split": r["split"], "corpus": r["corpus"],
            "meter": r["meter"], "bars": r["window"]["n_bars"], "barQ": bar,
            "startBar": r["window"].get("start_measure") or "?",
            "lengthQ": _q(r["window"]["length_q"]),
            "span": [_q(x) for x in r["params"]["span"]],
            "spanVoice": order.index(r["params"]["voice_id"])
            if "voice_id" in r["params"] else None,
            "desc": _describe(r), "pairId": r["pair_id"],
            "original": _ordered(r["original"], order),
            "corrupted": _ordered(r["corrupted"], order),
            "changed": changed,
            "blindFlip": rng.random() < 0.5,
        })
    html = (_TEMPLATE.replace("__DATA__", json.dumps(items, separators=(",", ":")))
            .replace("__QSEC__", str(QUARTER_SECONDS)))
    (out_dir / "index.html").write_text(html)
    return out_dir


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corrupted Bach Pairs</title>
<style>
  :root {
    --bg: #0b0c10; --panel: #14161c; --line: #262a33; --text: #e6e6e6;
    --muted: #9097a3; --accent: #e8c547; --good: #5cc98a; --bad: #e86a5c;
    --band: rgba(255,255,255,0.07); --bandEdge: rgba(232,197,71,0.55);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { padding: 20px 16px 8px; max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0 0 4px; font-weight: 600; }
  header p { margin: 0; color: var(--muted); }
  .bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
         padding: 12px 16px; max-width: 1200px; margin: 0 auto; }
  button { background: var(--panel); color: var(--text); border: 1px solid var(--line);
           border-radius: 6px; padding: 6px 12px; font: inherit; cursor: pointer; }
  button:hover { border-color: #3a3f4b; }
  button.on { border-color: var(--accent); color: var(--accent); }
  .spacer { flex: 1; }
  #score { color: var(--muted); }
  main { max-width: 1200px; margin: 0 auto; padding: 0 16px 40px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
          padding: 14px; margin: 14px 0; }
  .head { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: baseline; }
  .head h2 { font-size: 16px; margin: 0; font-weight: 600; }
  .meta { color: var(--muted); font-size: 13px; }
  .desc { margin: 6px 0 10px; font-size: 14px; }
  .roll { margin-top: 10px; }
  .rollhead { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
  .label { font-weight: 600; min-width: 92px; }
  .pos { font-family: ui-monospace, monospace; font-size: 12px; opacity: 0.85; }
  .roll canvas { cursor: col-resize; touch-action: none; }
  .play { min-width: 74px; }
  canvas { width: 100%; height: 170px; display: block; border-radius: 6px;
           background: #07080b; }
  .pick { display: none; gap: 8px; align-items: center; margin-top: 10px; }
  body.blind .pick { display: flex; }
  .verdict { font-weight: 600; }
  .verdict.ok { color: var(--good); } .verdict.no { color: var(--bad); }
  body.blind .card:not(.revealed) .desc,
  body.blind .card:not(.revealed) .head h2,
  body.blind .card:not(.revealed) .meta .op { visibility: hidden; }
  .legend { color: var(--muted); font-size: 13px; padding: 0 16px; max-width: 1200px;
            margin: 0 auto; }
  .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
        margin: 0 4px 0 10px; vertical-align: -1px; }
</style>
</head>
<body>
<header>
  <h1>Corrupted Bach pairs</h1>
  <p>Each pair is a real Bach window and a copy damaged by one operator. Both
  sides pass the same hard counterpoint rules, so the damage is musical, not
  rule-breaking.</p>
</header>
<div class="bar" id="filters"></div>
<div class="bar">
  <button id="blind">Blind test: off</button>
  <span id="score"></span>
  <span class="spacer"></span>
  <button id="stop">Stop audio</button>
</div>
<div class="legend">Colour = voice (top to bottom).
  <span class="sw" style="background:rgba(255,255,255,0.18);border:1px solid #e8c547"></span>corrupted span
  <span class="sw" style="background:transparent;border:2px solid #fff"></span>note not in the original
</div>
<main id="list"></main>

<script>
const DATA = __DATA__;
const QSEC = __QSEC__;
const COLORS = ["#e8564f", "#f2b33d", "#55c26b", "#4aa3f0", "#b177e8", "#e86fb0"];
const OPS = [["all","All"],["segment_shuffle","Shuffled bars"],["foreign_key_area","Wrong key area"],
             ["solver_filler","Solver filler"],["rhythm_flatten","Flattened rhythm"],
             ["subject_return_removal","Missing subject return"]];

const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} },
};
let filter = store.get("cb.filter", "all");
let blind = store.get("cb.blind", false);
let picks = store.get("cb.picks", {});
let current = null;   // {audio, canvas, item, side}

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}

function draw(canvas, item, side, playQ) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const all = item.original.concat(item.corrupted).flat();
  let lo = Math.min(...all.map(n => n[2])) - 2, hi = Math.max(...all.map(n => n[2])) + 2;
  const pad = 6, x = q => pad + (q / item.lengthQ) * (w - 2 * pad);
  const rowH = (h - 2 * pad) / (hi - lo + 1);
  const y = m => pad + (hi - m) * rowH;
  const reveal = !blind || canvas.closest(".card").classList.contains("revealed");
  // corrupted span band (only on the corrupted side, and not in blind mode)
  if (side === "corrupted" && reveal) {
    g.fillStyle = "rgba(255,255,255,0.07)";
    g.fillRect(x(item.span[0]), 0, x(item.span[1]) - x(item.span[0]), h);
    g.strokeStyle = "rgba(232,197,71,0.55)"; g.lineWidth = 1;
    g.strokeRect(x(item.span[0]) + 0.5, 0.5, x(item.span[1]) - x(item.span[0]) - 1, h - 1);
  }
  // bar lines
  g.strokeStyle = "rgba(255,255,255,0.08)"; g.lineWidth = 1;
  for (let b = 0; b <= item.bars; b++) {
    const xx = Math.round(x(b * item.barQ)) + 0.5;
    g.beginPath(); g.moveTo(xx, 0); g.lineTo(xx, h); g.stroke();
  }
  const voices = item[side];
  voices.forEach((notes, vi) => {
    const col = COLORS[vi % COLORS.length];
    notes.forEach((n, ni) => {
      const x0 = x(n[0]), x1 = x(n[0] + n[1]), yy = y(n[2]);
      const nh = Math.max(3, rowH * 0.9);
      const sounding = playQ !== null && n[0] <= playQ && playQ < n[0] + n[1];
      g.fillStyle = col;
      g.globalAlpha = sounding ? 1 : 0.82;
      g.beginPath();
      g.roundRect ? g.roundRect(x0, yy, Math.max(2, x1 - x0 - 1.5), nh, Math.min(3, nh / 2))
                  : g.rect(x0, yy, Math.max(2, x1 - x0 - 1.5), nh);
      g.fill();
      if (sounding) { g.globalAlpha = 0.35; g.fillStyle = "#fff"; g.fill(); }
      g.globalAlpha = 1;
      if (side === "corrupted" && reveal && item.changed[vi][ni]) {
        g.strokeStyle = "#ffffff"; g.lineWidth = 1.5; g.stroke();
      }
    });
  });
  if (playQ !== null) {
    g.strokeStyle = "#ffffff"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x(playQ), 0); g.lineTo(x(playQ), h); g.stroke();
  }
}

// Stopping leaves the line where the music stopped; drag or click the roll to
// move it. Play starts from the line. The readout names the piece, version,
// bar and time, so a screenshot says exactly where.
function stopAll(reset) {
  if (current) {
    const c = current;
    c.audio.pause();
    current = null;
    c.button.textContent = "▶ Play";
    c.setPos(reset ? null : c.audio.currentTime / QSEC);
  }
}

function tick() {
  if (!current) return;
  current.show(Math.min(current.audio.currentTime / QSEC, current.item.lengthQ));
  if (!current.audio.paused) requestAnimationFrame(tick);
}

function rollFor(item, side, label) {
  const canvas = el("canvas");
  const audio = new Audio(`audio/${item.key}_${side}.wav`);
  audio.preload = "none";
  const button = el("button", { class: "play" }, "▶ Play");
  const readout = el("span", { class: "pos" });
  let posQ = null;
  const fmt = sec => `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;
  function show(q) {
    draw(canvas, item, side, q);
    if (q === null) { readout.textContent = ""; return; }
    const bar = Math.floor(q / item.barQ), beat = (q - bar * item.barQ) + 1;
    readout.textContent = `${item.where || item.piece + " · " + item.title.split(" — ").pop()} · ${lab.textContent} · ` +
      `bar ${bar + item.startBar} beat ${beat.toFixed(1)} · ${fmt(q * QSEC)}`;
  }
  function setPos(q) { posQ = q; show(q); }
  const st = { audio, canvas, item, side, button, show, setPos };
  function seekTo(q) {
    if (audio.readyState >= 1) audio.currentTime = q * QSEC;
    else audio.addEventListener("loadedmetadata", () => { audio.currentTime = q * QSEC; }, { once: true });
  }
  button.addEventListener("click", () => {
    if (current && current.audio === audio) { stopAll(); return; }
    stopAll();
    current = st;
    button.textContent = "■ Stop";
    const from = posQ !== null && posQ < item.lengthQ - 0.05 ? posQ : 0;
    seekTo(from);
    audio.play(); requestAnimationFrame(tick);
  });
  audio.addEventListener("ended", () => { if (current && current.audio === audio) stopAll(true); });
  function qAt(ev) {
    const r = canvas.getBoundingClientRect(), pad = 6;
    const f = (ev.clientX - r.left - pad) / (r.width - 2 * pad);
    return Math.max(0, Math.min(1, f)) * item.lengthQ;
  }
  let dragging = false;
  function move(ev) {
    const q = qAt(ev);
    if (current && current.audio === audio) { seekTo(q); show(q); posQ = q; }
    else setPos(q);
  }
  canvas.addEventListener("pointerdown", ev => { dragging = true; canvas.setPointerCapture(ev.pointerId); move(ev); });
  canvas.addEventListener("pointermove", ev => { if (dragging) move(ev); });
  canvas.addEventListener("pointerup", () => { dragging = false; });
  canvas.addEventListener("pointercancel", () => { dragging = false; });
  const lab = el("span", { class: "label" }, label);
  const wrap = el("div", { class: "roll" }, el("div", { class: "rollhead" }, button, lab, readout), canvas);
  wrap._redraw = () => show(posQ);
  wrap._label = lab;
  return wrap;
}

function card(item) {
  const c = el("section", { class: "card", "data-op": item.op });
  const verdict = el("span", { class: "verdict" });
  const head = el("div", { class: "head" },
    el("h2", {}, item.title),
    el("span", { class: "meta" },
      `${item.piece} · ${item.bars} bars from m. ${item.startBar} · ${item.meter} · ${item.split}`));
  const desc = el("p", { class: "desc" }, item.desc);
  const rOrig = rollFor(item, "original", "Original");
  const rCorr = rollFor(item, "corrupted", "Corrupted");
  const rolls = el("div");
  const pick = el("div", { class: "pick" }, el("span", {}, "Which one is Bach?"));
  const pickBtn = (letter) => el("button", { onclick: () => choose(letter) }, letter);
  const reveal = el("button", { onclick: () => { c.classList.add("revealed"); layout(); } }, "Reveal");
  pick.append(pickBtn("A"), pickBtn("B"), reveal, verdict);

  function bachLetter() { return item.blindFlip ? "B" : "A"; }
  function choose(letter) {
    picks[item.key] = letter === bachLetter();
    store.set("cb.picks", picks);
    c.classList.add("revealed");
    layout(); updateScore();
  }
  function layout() {
    rolls.replaceChildren();
    const revealed = !blind || c.classList.contains("revealed");
    let order = [rOrig, rCorr];
    if (blind && item.blindFlip) order = [rCorr, rOrig];
    if (blind) {
      order[0]._label.textContent = "A" + (revealed ? (order[0] === rOrig ? " — Original" : " — Corrupted") : "");
      order[1]._label.textContent = "B" + (revealed ? (order[1] === rOrig ? " — Original" : " — Corrupted") : "");
    } else {
      rOrig._label.textContent = "Original"; rCorr._label.textContent = "Corrupted";
    }
    rolls.append(...order);
    if (item.key in picks && blind) {
      verdict.textContent = picks[item.key] ? "Correct" : "Fooled you";
      verdict.className = "verdict " + (picks[item.key] ? "ok" : "no");
    } else { verdict.textContent = ""; }
    requestAnimationFrame(() => { rOrig._redraw(); rCorr._redraw(); });
  }
  c._layout = layout;
  c.append(head, desc, rolls, pick);
  return c;
}

function updateScore() {
  const keys = Object.keys(picks);
  const right = keys.filter(k => picks[k]).length;
  document.getElementById("score").textContent =
    blind ? (keys.length ? `Picked Bach correctly: ${right} / ${keys.length}` : "Listen to A and B, then pick the real Bach.") : "";
}

const list = document.getElementById("list");
const cards = DATA.map(card);
list.append(...cards);

const filters = document.getElementById("filters");
OPS.forEach(([k, label]) => {
  const n = k === "all" ? DATA.length : DATA.filter(d => d.op === k).length;
  const b = el("button", { "data-k": k }, `${label} (${n})`);
  b.addEventListener("click", () => { filter = k; store.set("cb.filter", k); applyFilter(); });
  filters.append(b);
});

function applyFilter() {
  filters.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.k === filter));
  cards.forEach(c => { c.style.display = filter === "all" || c.dataset.op === filter ? "" : "none"; });
  cards.forEach(c => { if (c.style.display !== "none") c._layout(); });
}

const blindBtn = document.getElementById("blind");
function applyBlind() {
  stopAll();
  document.body.classList.toggle("blind", blind);
  blindBtn.textContent = "Blind test: " + (blind ? "on" : "off");
  blindBtn.classList.toggle("on", blind);
  cards.forEach(c => { if (!blind) c.classList.remove("revealed"); else if (!(DATA[cards.indexOf(c)].key in picks)) c.classList.remove("revealed"); else c.classList.add("revealed"); });
  applyFilter(); updateScore();
}
blindBtn.addEventListener("click", () => { blind = !blind; store.set("cb.blind", blind); applyBlind(); });
document.getElementById("stop").addEventListener("click", () => stopAll());
window.addEventListener("resize", () => cards.forEach(c => { if (c.style.display !== "none") c._layout(); }));
applyBlind();
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-type", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)
    out = build(per_type=args.per_type, seed=args.seed)
    print(f"wrote {out}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
