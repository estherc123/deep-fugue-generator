"""Render an excerpt for human review: MIDI to play, piano roll to look at (§14.5).

The piano roll is emitted as inline SVG rather than engraved notation so the
review app stays dependency-light -- it needs no Verovio, MuseScore or browser
plugin. `music21` writes the MIDI.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from ..normalize.rational import parse_frac

_VOICE_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def excerpt_notes(score: dict, start: Fraction, end: Fraction) -> list[dict]:
    out = []
    for vi, v in enumerate(score["voices"]):
        for n in v["notes"]:
            o = parse_frac(n["onset_q"])
            d = parse_frac(n["duration_q"])
            if o + d <= start or o >= end:
                continue
            out.append({**n, "_voice_index": vi,
                        "_clip_on": max(o, start), "_clip_off": min(o + d, end)})
    return out


def write_midi(score: dict, start: Fraction, end: Fraction, path: Path) -> bool:
    """Write a MIDI preview of the window. Returns False if music21 is unavailable."""
    try:
        from music21 import stream, note as m21note, tempo
    except Exception:
        return False
    s = stream.Score()
    for vi, v in enumerate(score["voices"]):
        part = stream.Part(id=v["voice_id"])
        if vi == 0:
            part.append(tempo.MetronomeMark(number=72))
        for n in v["notes"]:
            o = parse_frac(n["onset_q"])
            d = parse_frac(n["duration_q"])
            if o + d <= start or o >= end:
                continue
            if n.get("tie") in ("continue", "stop"):
                continue
            nn = m21note.Note()
            nn.pitch.midi = n["midi"]
            nn.quarterLength = float(min(o + d, end) - max(o, start))
            part.insert(float(max(o, start) - start), nn)
        if len(part.notes):
            s.insert(0, part)
    path.parent.mkdir(parents=True, exist_ok=True)
    s.write("midi", fp=str(path))
    return True


def piano_roll_svg(score: dict, start: Fraction, end: Fraction,
                   annotations: list[dict] | None = None, title: str = "") -> str:
    notes = excerpt_notes(score, start, end)
    if not notes:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='400' height='40'>" \
               "<text x='8' y='24' font-family='sans-serif' font-size='13'>empty window</text></svg>"

    lo = min(n["midi"] for n in notes) - 2
    hi = max(n["midi"] for n in notes) + 2
    span = float(end - start)
    W, H = 900, max(160, (hi - lo) * 9)
    pad_l, pad_t = 44, 26
    xw = (W - pad_l - 12) / span if span else 1
    yh = (H - pad_t - 24) / max(1, hi - lo)

    def x(q): return pad_l + float(q - start) * xw
    def y(m): return pad_t + (hi - m) * yh

    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
             f"viewBox='0 0 {W} {H}' font-family='system-ui,sans-serif'>",
             f"<rect width='{W}' height='{H}' fill='#fafafa'/>"]
    if title:
        parts.append(f"<text x='8' y='16' font-size='12' fill='#444'>{title}</text>")

    # Staff-ish guide lines every octave, and bar lines from measure starts.
    for m in range(lo, hi + 1):
        if m % 12 == 0:
            parts.append(f"<line x1='{pad_l}' y1='{y(m):.1f}' x2='{W-12}' y2='{y(m):.1f}' "
                         f"stroke='#e2e2e2'/>")
            parts.append(f"<text x='6' y='{y(m)+4:.1f}' font-size='9' fill='#999'>C{m//12-1}</text>")

    seen_bars = set()
    for v in score["voices"]:
        for n in v["notes"]:
            o = parse_frac(n["onset_q"])
            if start <= o < end:
                bs = o - parse_frac(n["offset_in_measure_q"])
                if bs >= start and bs not in seen_bars:
                    seen_bars.add(bs)
                    parts.append(f"<line x1='{x(bs):.1f}' y1='{pad_t-6}' x2='{x(bs):.1f}' "
                                 f"y2='{H-20}' stroke='#c9c9c9' stroke-dasharray='3,3'/>")
                    parts.append(f"<text x='{x(bs)+3:.1f}' y='{pad_t-10}' font-size='9' "
                                 f"fill='#888'>m{n.get('measure_number')}</text>")

    for a in annotations or []:
        if a.get("analytical_status") not in (None, "accepted") or not a.get("duration_q"):
            continue
        s0 = parse_frac(a["onset_q"]); e0 = s0 + parse_frac(a["duration_q"])
        if e0 <= start or s0 >= end:
            continue
        xs, xe = x(max(s0, start)), x(min(e0, end))
        parts.append(f"<rect x='{xs:.1f}' y='{pad_t-4}' width='{max(1,xe-xs):.1f}' height='6' "
                     f"fill='#111' opacity='0.10'/>")
        parts.append(f"<text x='{xs+2:.1f}' y='{H-8}' font-size='9' fill='#666'>"
                     f"{a['type']}</text>")

    for n in notes:
        c = _VOICE_COLORS[n["_voice_index"] % len(_VOICE_COLORS)]
        w = max(2.0, (float(n["_clip_off"] - n["_clip_on"])) * xw - 1)
        parts.append(f"<rect x='{x(n['_clip_on']):.1f}' y='{y(n['midi']):.1f}' "
                     f"width='{w:.1f}' height='{max(3, yh-1):.1f}' rx='1.5' fill='{c}' "
                     f"opacity='0.85'><title>{n['step']}{n['octave']} "
                     f"{n['voice_id']} m{n.get('measure_number')}</title></rect>")
    parts.append("</svg>")
    return "".join(parts)
