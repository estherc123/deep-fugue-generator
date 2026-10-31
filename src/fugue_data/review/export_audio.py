"""Render generated fragments to self-contained audio for sharing (Spec §14.5).

The review apps synthesise playback in the browser and never write audio to disk,
so there is nothing to hand to someone who is not sitting at the app. MIDI alone
is awkward -- it needs a synthesiser on the listener's machine, and none is
installed here -- so fragments are rendered to WAV with the same triangle-wave
voicing the browser uses. What the listener hears matches what a reviewer hears.

A MIDI file is written alongside each WAV for anyone who wants to open the
fragment in notation software.

Blinding is preserved: files are named `candidate_1` / `candidate_2` in the same
shuffled order the app uses, and the mapping back to the underlying samples goes
in a separate `blinding_key.json` that need not be shared.
"""

from __future__ import annotations

import json
import math
import random
import struct
import wave
from fractions import Fraction
from pathlib import Path

from ..normalize.rational import parse_frac
from .render import piano_roll_svg, write_midi

SAMPLE_RATE = 44100
QUARTER_SECONDS = 0.55          # ~109 bpm, matching the review app's pacing


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _saw(freq: float, n: int) -> "list[float]":
    import numpy as np
    t = np.arange(n) / SAMPLE_RATE
    return 2.0 * ((t * freq) % 1.0) - 1.0


def _triangle(freq: float, n: int) -> "list[float]":
    import numpy as np
    t = np.arange(n) / SAMPLE_RATE
    # Triangle wave: the browser's oscillator type, so the timbre matches.
    return 2.0 * np.abs(2.0 * ((t * freq) % 1.0) - 1.0) - 1.0


def render_wav(score: dict, path: Path) -> float:
    """Synthesise a fragment to a 16-bit mono WAV. Returns its duration."""
    import numpy as np
    total_q = float(parse_frac(score.get("quarter_length_total", "0")))
    dur_s = total_q * QUARTER_SECONDS + 0.8
    buf = np.zeros(int(dur_s * SAMPLE_RATE), dtype=np.float64)

    # Timbre and gain are properties of the VOICE, fixed for its whole line --
    # deriving them from each note's pitch made a single voice change instrument
    # mid-phrase, which reads as a new entry rather than a continuing line.
    nv = len(score["voices"])
    for vi, v in enumerate(score["voices"]):
        from_bottom = (nv - 1) - vi
        is_bass = from_bottom == 0
        is_lower = from_bottom <= 1
        vgain = 0.16 * (1.15 if is_bass else 1.0 if is_lower else 1.0)
        voct = 0.95 if is_bass else 0.40 if is_lower else 0.0
        voct2 = 0.55 if is_bass else 0.18 if is_lower else 0.0
        atk = 0.040 if is_bass else 0.030 if is_lower else 0.018
        for note in v["notes"]:
            if note.get("tie") in ("continue", "stop"):
                continue
            onset = float(parse_frac(note["onset_q"])) * QUARTER_SECONDS
            length = max(0.10, float(parse_frac(note["duration_q"])) * QUARTER_SECONDS)
            freq = 440.0 * (2.0 ** ((note["midi"] - 69) / 12.0))
            n = int(length * SAMPLE_RATE)
            start = int(onset * SAMPLE_RATE)
            if start + n > len(buf):
                n = len(buf) - start
            if n <= 0:
                continue
            # Triangle only. Sawtooth carries every harmonic at 1/n and the buzz
            # reads as bad music rather than as a bad synthesiser; an octave
            # partial gives a small speaker something to reproduce without it.
            wave_ = _triangle(freq, n)
            if voct > 0:
                wave_ = wave_ + voct * _triangle(freq * 2.0, n)
            if voct2 > 0:
                wave_ = wave_ + voct2 * _triangle(freq * 4.0, n)
            env = np.ones(n)
            a = min(int(atk * SAMPLE_RATE), n)
            env[:a] = np.linspace(0.0, 1.0, a)
            d = min(int(0.30 * SAMPLE_RATE), n)
            env[-d:] *= np.linspace(1.0, 0.0, d)
            buf[start:start + n] += wave_ * env * vgain

    peak = float(np.max(np.abs(buf))) or 1.0
    buf = buf / max(peak, 1.0) * 0.89                     # normalise, avoid clipping
    pcm = (buf * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return dur_s


def export(root: Path | None = None, *, count: int = 25, out_dir: Path | None = None,
           only_revision: str | None = None, seed: int = 20260728) -> dict:
    root = root or _repo_root()
    d = root / "data" / "generated" / "preference_candidates"
    out = Path(out_dir) if out_dir else (root / "data" / "exports" / "audio")
    out.mkdir(parents=True, exist_ok=True)

    pairs = [json.loads(l) for l in (d / "pairs.jsonl").read_text().splitlines() if l.strip()]
    cands = {c["sample_id"]: c for c in
             (json.loads(l) for l in (d / "candidates.jsonl").read_text().splitlines()
              if l.strip())}

    pool = [p for p in pairs if p["candidate_a_id"] in cands and p["candidate_b_id"] in cands]
    if only_revision:
        pool = [p for p in pool
                if cands[p["candidate_a_id"]].get("generator_revision") == only_revision]
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool = pool[:count]

    key, rows = [], []
    for i, p in enumerate(pool):
        a, b = cands[p["candidate_a_id"]], cands[p["candidate_b_id"]]
        r = random.Random(p.get("presentation_seed") or 0)
        flip = r.random() < 0.5
        sides = {"1": (b if flip else a), "2": (a if flip else b)}
        tag = f"pair_{i + 1:03d}"
        entry = {"tag": tag, "pair_id": p["pair_id"],
                 "generator_revision": a.get("generator_revision", "unknown"),
                 "files": {}, "svg": {}}
        for side, cand in sides.items():
            wav = out / f"{tag}_candidate_{side}.wav"
            render_wav(cand["score"], wav)
            mid = out / f"{tag}_candidate_{side}.mid"
            write_midi(cand["score"], Fraction(0),
                       parse_frac(cand["score"]["quarter_length_total"]), mid)
            entry["files"][side] = wav.name
            entry["svg"][side] = piano_roll_svg(
                cand["score"], Fraction(0),
                parse_frac(cand["score"]["quarter_length_total"]), None, "")
        rows.append(entry)
        key.append({"tag": tag, "pair_id": p["pair_id"],
                    "candidate_1": sides["1"]["sample_id"],
                    "candidate_2": sides["2"]["sample_id"]})

    # The blinding map is written separately so the audio folder can be shared
    # without revealing which side is which.
    (out / "blinding_key.json").write_text(json.dumps(key, indent=2) + "\n")

    cards = []
    for e in rows:
        cards.append(f"""<div class="card">
  <h2>{e['tag']} <small>{e['generator_revision']}</small></h2>
  <div class="sides">
    <div><h3>Candidate 1</h3>{e['svg']['1']}
      <audio controls preload="none" src="{e['files']['1']}"></audio></div>
    <div><h3>Candidate 2</h3>{e['svg']['2']}
      <audio controls preload="none" src="{e['files']['2']}"></audio></div>
  </div>
</div>""")
    html = f"""<!doctype html><meta charset="utf-8">
<title>Generated counterpoint - listening set</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:0;background:#f6f6f5;color:#1a1a1a}}
main{{max-width:1100px;margin:0 auto;padding:16px}}
h1{{font-size:18px}} h2{{font-size:14px;margin:0 0 8px}}
h2 small{{font-weight:400;color:#888}} h3{{font-size:13px;margin:0 0 6px}}
.card{{background:#fff;border:1px solid #e3e3e0;border-radius:8px;padding:14px;margin-bottom:14px}}
.sides{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media(max-width:820px){{.sides{{grid-template-columns:1fr}}}}
svg{{max-width:100%;height:auto;border:1px solid #eee;border-radius:5px}}
audio{{width:100%;margin-top:8px}}
p.note{{color:#555}}
</style>
<main>
<h1>Generated counterpoint &mdash; {len(rows)} pairs</h1>
<p class="note">Each pair is two realisations of the <b>same</b> prompt &mdash; same
subject, key, meter and entry plan &mdash; differing only in the generator's random
seed. The question is which is the better piece of counterpoint, not whether it is
Bach. These come from a constraint solver: expect correct but plain writing, with
no functional harmonic direction. Candidate order is shuffled per pair.</p>
{''.join(cards)}
</main>"""
    (out / "index.html").write_text(html)

    total_mb = sum(f.stat().st_size for f in out.glob("*")) / 1e6
    return {"stage": "export_audio", "pairs": len(rows), "directory": str(out),
            "files": len(list(out.glob("*"))), "total_mb": round(total_mb, 1),
            "index": str(out / "index.html"),
            "blinding_key": str(out / "blinding_key.json")}
