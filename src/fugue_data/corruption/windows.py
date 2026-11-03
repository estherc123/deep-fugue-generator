"""Bar-aligned windows of real Bach in a compact, editable form.

A window holds tie-merged events per voice, shifted so the window starts at 0.
Notes sounding into the window from before its start are kept, clipped to the
window, so the original and every corrupted copy share the same boundary
treatment: whatever artefact clipping introduces, it introduces on both sides of
the pair, and cannot become a cue that separates them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

from ..constraints.fragment import Fragment
from ..normalize.rational import frac_str, parse_frac
from ..review.propose import measure_table
from ..tokenization.views import merge_tied

# (onset, duration, midi), onset relative to the window start.
Event = tuple[Fraction, Fraction, int]

# Krumhansl-Kessler key profiles (major, minor), tonic first.
_KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


@dataclass
class Window:
    piece_id: str
    start_q: Fraction                 # absolute onset of the window in the piece
    length_q: Fraction
    bar_q: Fraction                   # length of one bar in quarters
    meter: tuple[int, int]
    key_fifths: int
    movement_type: str
    is_extract: bool
    first_bar: int                    # index into the piece's bar table
    n_bars: int
    voices: dict[str, list[Event]]    # voice_id -> events, highest voice first
    meta: dict = field(default_factory=dict)

    def copy_with(self, voices: dict[str, list[Event]]) -> "Window":
        return Window(self.piece_id, self.start_q, self.length_q, self.bar_q,
                      self.meter, self.key_fifths, self.movement_type,
                      self.is_extract, self.first_bar, self.n_bars,
                      {v: sorted(e) for v, e in voices.items()}, dict(self.meta))

    # ---- views ---------------------------------------------------------- #
    def to_fragment(self) -> Fragment:
        """The window as a rule-engine Fragment (canonical note-dict shape)."""
        voices = []
        for vid, events in self.voices.items():
            notes = []
            for k, (o, d, m) in enumerate(events):
                notes.append({"note_id": f"{vid}:{k}", "voice_id": vid,
                              "onset_q": frac_str(o), "duration_q": frac_str(d),
                              "midi": int(m), "pitch_class": int(m) % 12,
                              "octave": int(m) // 12 - 1, "tie": None,
                              "grace": False})
            voices.append({"voice_id": vid, "notes": notes})
        return Fragment(piece_id=self.piece_id, voices=voices,
                        quarter_length_total=self.length_q, meter=self.meter,
                        key_fifths=self.key_fifths, measure_count=self.n_bars,
                        range_profile=("satb" if self.movement_type == "chorale"
                                       else "keyboard"),
                        is_extract=self.is_extract)

    def compact(self) -> dict:
        """JSON form: {voice_id: [[onset_q, duration_q, midi], ...]}."""
        return {v: [[frac_str(o), frac_str(d), int(m)] for o, d, m in ev]
                for v, ev in self.voices.items()}

    def note_count(self) -> int:
        return sum(len(e) for e in self.voices.values())


def bar_length_q(meter: tuple[int, int]) -> Fraction:
    return Fraction(4 * meter[0], meter[1])


def is_compound(meter: tuple[int, int]) -> bool:
    num, den = meter
    return den >= 8 and num % 3 == 0 and num > 3


def beat_length_q(meter: tuple[int, int]) -> Fraction:
    """One beat: a dotted value in compound meter, the denominator otherwise."""
    unit = Fraction(4, meter[1])
    return unit * 3 if is_compound(meter) else unit


def bars_for(min_bars: int, min_q: float, bar_q: Fraction) -> int:
    """At least `min_bars`, and enough bars to cover `min_q` quarters (3/8 etc.)."""
    return max(min_bars, math.ceil(Fraction(str(min_q)) / bar_q))


def voice_events(score: dict) -> dict[str, list[Event]]:
    """Tie-merged, grace-free absolute events per voice, highest voice first."""
    out = {}
    for v in sorted(score["voices"], key=lambda v: -float(v.get("mean_midi") or 0)):
        notes = [n for n in v["notes"] if not n.get("grace")
                 and parse_frac(n["duration_q"]) > 0]
        out[v["voice_id"]] = merge_tied(notes)
    return out


def extract(score: dict, events: dict[str, list[Event]], start: Fraction,
            n_bars: int, first_bar: int = 0) -> Window | None:
    """Window of exactly `n_bars` bars of the meter from `start`, or None past the end."""
    total = parse_frac(score.get("quarter_length_total", "0"))
    mm = (score.get("meter_map") or [{"numerator": 4, "denominator": 4}])[0]
    meter = (mm["numerator"], mm["denominator"])
    end = start + n_bars * bar_length_q(meter)
    if end > total:
        return None
    voices: dict[str, list[Event]] = {}
    for vid, ev in events.items():
        w = []
        for o, d, m in ev:
            e = o + d
            if e <= start or o >= end:
                continue
            a, b = max(o, start), min(e, end)
            w.append((a - start, b - a, m))
        voices[vid] = w
    km = (score.get("key_signature_map") or [{"fifths": 0}])[0]
    return Window(piece_id=score["piece_id"], start_q=start, length_q=end - start,
                  bar_q=bar_length_q(meter), meter=meter, key_fifths=km["fifths"],
                  movement_type=score.get("movement_type", "other"),
                  is_extract=bool((score.get("normalization") or {})
                                  .get("excluded_parts")),
                  first_bar=first_bar, n_bars=n_bars, voices=voices)


def windows_for(score: dict, n_bars_min: int, min_q: float, stride_bars: int):
    """Windows on the meter grid: whole bars of the meter, from a real downbeat.

    The score's own bar table is not a reliable grid. Unfolded repeats split one
    bar into two short ones (3 + 1 beats), and chorale extracts omit bars in
    which only the excluded instruments play. Counting table entries therefore
    gives windows of irregular length, while every operator edits in whole bars
    of the meter. So windows are laid on the meter grid from the first full bar,
    and kept only where the score's downbeats agree with that grid: an off-grid
    bar start is tolerated only as the short half of a split bar, i.e. when the
    next bar start returns to the grid.
    """
    bars = measure_table(score)
    if not bars:
        return
    mm = (score.get("meter_map") or [{"numerator": 4, "denominator": 4}])[0]
    bq = bar_length_q((mm["numerator"], mm["denominator"]))
    n_bars = bars_for(n_bars_min, min_q, bq)
    stride = max(1, round(stride_bars * n_bars / n_bars_min))
    events = voice_events(score)
    total = parse_frac(score.get("quarter_length_total", "0"))
    # A pickup bar is short; the grid starts at the first full bar.
    first = 1 if len(bars) > 1 and bars[1][2] - bars[0][2] < bq else 0
    ref = bars[first][2]
    starts = [b[2] for b in bars[first:]] + [total]
    # Bar starts that break the grid: off-grid and not followed by a return to it.
    broken = [s for s, nxt in zip(starts, starts[1:])
              if (s - ref) % bq and (nxt - ref) % bq]
    k = 0
    while ref + (k + n_bars) * bq <= total:
        start = ref + k * bq
        end = start + n_bars * bq
        if not any(start <= s < end for s in broken):
            w = extract(score, events, start, n_bars, first_bar=first + k)
            if w is not None:
                # The score's own measure number at the window's downbeat.
                w.meta["start_measure"] = next(
                    (num for _, num, st in reversed(bars) if st <= start), None)
                yield w
        k += stride


# --------------------------------------------------------------------------- #
# Span helpers -- every corruption edits a [start, end) span of the window
# --------------------------------------------------------------------------- #
def cut(events: list[Event], start: Fraction, end: Fraction):
    """Split events into (before, inside, after) at the span edges.

    A note crossing an edge is divided there, so each part can be edited or moved
    independently without leaving a note sounding outside its own span.
    """
    before, inside, after = [], [], []
    for o, d, m in events:
        e = o + d
        if e <= start:
            before.append((o, d, m))
        elif o >= end:
            after.append((o, d, m))
        else:
            if o < start:
                before.append((o, start - o, m))
            a, b = max(o, start), min(e, end)
            inside.append((a, b - a, m))
            if e > end:
                after.append((end, e - end, m))
    return before, inside, after


def replace_span(events: list[Event], start: Fraction, end: Fraction,
                 new_inside: list[Event]) -> list[Event]:
    before, _, after = cut(events, start, end)
    return sorted(before + new_inside + after)


def rejoin(events: list[Event], original: list[Event], bounds: dict) -> list[Event]:
    """Re-merge halves of one original note that `cut` divided.

    `cut` splits a note crossing a boundary. If the edit leaves both halves
    contiguous again, the corrupted copy would re-attack where Bach held the
    note -- an artefact found only on the corrupted side, i.e. a free cue for a
    judge. `bounds` maps a boundary in the corrupted window to the time in the
    original it came from (the same time unless the operator moved material).
    Two contiguous same-pitch events meeting at such a boundary are merged only
    if one original note of that pitch sounded across the original time: a
    same-pitch meeting the edit created by chance is a real new attack.
    """
    def held_across(t: Fraction, m: int) -> bool:
        return any(o < t < o + d and pm == m for o, d, pm in original)

    out: list[Event] = []
    for o, d, m in sorted(events):
        if out:
            po, pd, pm = out[-1]
            src = bounds.get(o)
            if pm == m and po + pd == o and src is not None and held_across(src, m):
                out[-1] = (po, pd + d, m)
                continue
        out.append((o, d, m))
    return out


def sounding_at(events: list[Event], t: Fraction) -> int | None:
    """Highest pitch sounding at `t` in one voice, or None during a rest."""
    hits = [m for o, d, m in events if o <= t < o + d]
    return max(hits) if hits else None


# --------------------------------------------------------------------------- #
# Key finding
# --------------------------------------------------------------------------- #
def pc_histogram(voices: dict[str, list[Event]]) -> list[float]:
    h = [0.0] * 12
    for ev in voices.values():
        for _, d, m in ev:
            h[m % 12] += float(d)
    return h


def _corr(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / 12, sum(y) / 12
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else 0.0


def estimate_key(voices: dict[str, list[Event]]) -> tuple[int, str, float]:
    """(tonic_pc, mode, correlation) by Krumhansl-Schmuckler."""
    h = pc_histogram(voices)
    best = (0, "major", -2.0)
    for tonic in range(12):
        rot = h[tonic:] + h[:tonic]
        for mode, prof in (("major", _KK_MAJOR), ("minor", _KK_MINOR)):
            r = _corr(rot, prof)
            if r > best[2]:
                best = (tonic, mode, r)
    return best
