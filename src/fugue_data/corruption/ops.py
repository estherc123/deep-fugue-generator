"""The five corruption operators.

Each operator proposes candidate edits of a window; it never decides acceptance.
The runner re-validates every candidate with the rule engine and keeps the one
that adds the least soft/diagnostic damage, so the surviving corruption is the
*least* detectable by rules -- the damage has to be found musically.

Operators return a list of (voices, params) candidates. `voices` maps voice_id to
window-relative events; `params` records exactly what was done, so a pair can be
reproduced or audited without re-running the generator.

  segment_shuffle        permute the bars of a free (non-entry) region:
                         harmony inside each bar survives, succession does not
  foreign_key_area       shift a span by a semitone or tritone in every voice:
                         a key area the context never prepared
  solver_filler          replace a span with CP-SAT material in the local key:
                         legal, idiomatic-ish, and not Bach
  rhythm_flatten         reduce a span to one attack per beat: pitches at the
                         beat survive, surface rhythm does not
  subject_return_removal replace a post-exposition subject entry with free
                         material from the same voice elsewhere in the piece
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

from .windows import (Event, Window, beat_length_q, cut, rejoin, replace_span,
                      sounding_at)

_REGISTRY: dict[str, callable] = {}


def operator(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def registry() -> dict:
    return dict(_REGISTRY)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _interior_spans(win: Window, span_bars: int, margin: int = 1):
    """Bar-aligned spans that leave `margin` untouched bars at each window edge.

    The untouched context is the point: a corruption is only audible as a
    corruption against music that leads into and out of it.
    """
    lo, hi = margin, win.n_bars - span_bars - margin
    if hi < lo:
        lo, hi = 0, win.n_bars - span_bars
    for b in range(lo, hi + 1):
        yield b * win.bar_q, (b + span_bars) * win.bar_q


def _entry_overlap(entries, start: Fraction, end: Fraction) -> Fraction:
    return sum((max(Fraction(0), min(e, end) - max(s, start))
                for _, s, e, _ in entries), Fraction(0))


def _span_bar_choices(cfg: dict) -> list[int]:
    lo, hi = cfg.get("span_bars", [1, 2])
    return list(range(lo, hi + 1))


def _transpose(events: list[Event], shift: int) -> list[Event]:
    return [(o, d, m + shift) for o, d, m in events]


def _mean_pitch(events: list[Event]) -> float | None:
    tot = sum(float(d) for _, d, _ in events)
    return sum(float(d) * m for _, d, m in events) / tot if tot else None


# --------------------------------------------------------------------------- #
@operator("segment_shuffle")
def segment_shuffle(win: Window, ctx: dict, rng: random.Random, cfg: dict):
    """Derange the bars of a region, all voices together.

    Regions overlapping a labelled subject entry least are preferred, so in the
    annotated fugues the shuffle lands in episodes -- where Bach's sequences give
    the succession of bars a direction that a permutation destroys.
    """
    out = []
    lo, hi = cfg.get("region_bars", [3, 4])
    for n in range(lo, hi + 1):
        spans = list(_interior_spans(win, n))
        if not spans:
            continue
        entries = ctx.get("entries") or []
        spans.sort(key=lambda s: _entry_overlap(entries, *s))
        best = _entry_overlap(entries, *spans[0])
        spans = [s for s in spans if _entry_overlap(entries, *s) == best]
        rng.shuffle(spans)
        for start, end in spans[:3]:
            perms = [p for p in itertools.permutations(range(n))
                     if all(p[i] != i for i in range(n))]
            rng.shuffle(perms)
            for perm in perms[:3]:
                voices = {}
                for vid, ev in win.voices.items():
                    _, inside, _ = cut(ev, start, end)
                    units = []
                    for k in range(n):
                        a = start + k * win.bar_q
                        _, u, _ = cut(inside, a, a + win.bar_q)
                        units.append([(o - a, d, m) for o, d, m in u])
                    new = []
                    for slot, src in enumerate(perm):
                        a = start + slot * win.bar_q
                        new += [(o + a, d, m) for o, d, m in units[src]]
                    # A bar boundary only joins one original note if the two
                    # bars either side of it were neighbours in the original.
                    bounds = {start + (j + 1) * win.bar_q:
                              start + (perm[j] + 1) * win.bar_q
                              for j in range(n - 1) if perm[j + 1] == perm[j] + 1}
                    voices[vid] = rejoin(replace_span(ev, start, end, new), ev, bounds)
                out.append((voices, {"span": [start, end], "region_bars": n,
                                     "permutation": list(perm),
                                     "entry_overlap_q": str(best)}))
    return out


@operator("foreign_key_area")
def foreign_key_area(win: Window, ctx: dict, rng: random.Random, cfg: dict):
    """Transpose a span, every voice by the same interval.

    Vertical intervals inside the span are untouched, so the passage stays as
    consonant as Bach wrote it; only its relation to the surrounding key breaks.
    Semitone and tritone shifts are the defaults because the keys they reach are
    the ones a Baroque tonal plan does not visit from the local tonic.
    """
    out = []
    for n in _span_bar_choices(cfg):
        for start, end in _interior_spans(win, n):
            for shift in cfg.get("shifts", [1, -1, 6, -6]):
                voices = {}
                for vid, ev in win.voices.items():
                    _, inside, _ = cut(ev, start, end)
                    voices[vid] = replace_span(ev, start, end, _transpose(inside, shift))
                out.append((voices, {"span": [start, end], "shift": shift}))
    return out


@operator("solver_filler")
def solver_filler(win: Window, ctx: dict, rng: random.Random, cfg: dict):
    """Replace a span with bars cut from a hard-valid CP-SAT fragment.

    Fragments are matched on voice count, bar length and simple/compound meter,
    taken from the same split as the piece, and transposed into the window's
    estimated local key (same mode) so the filler is not also a key corruption.
    Each voice is moved by octaves to sit where the Bach voice it replaces sat.
    """
    pool = ctx.get("filler_pool") or []
    tonic, mode = ctx["local_key"][:2]
    pool = [f for f in pool if f["mode"] == mode]
    if not pool:
        return []
    out = []
    picks = rng.sample(pool, min(len(pool), cfg.get("fragments_per_window", 8)))
    vids = list(win.voices)
    for frag in picks:
        n = rng.choice([b for b in _span_bar_choices(cfg) if b <= frag["n_bars"]]
                       or [frag["n_bars"]])
        spans = list(_interior_spans(win, n))
        if not spans:
            continue
        start, end = rng.choice(spans)
        fb = rng.randrange(0, frag["n_bars"] - n + 1)
        f0, f1 = fb * win.bar_q, (fb + n) * win.bar_q
        pc_shift = (tonic - frag["tonic_pc"]) % 12
        voices = {}
        for vid, fvid in zip(vids, frag["voice_order"]):
            ev = win.voices[vid]
            _, src, _ = cut(frag["voices"][fvid], f0, f1)
            src = [(o - f0 + start, d, m + pc_shift) for o, d, m in src]
            _, orig_inside, _ = cut(ev, start, end)
            target = _mean_pitch(orig_inside) or _mean_pitch(ev)
            have = _mean_pitch(src)
            if target is not None and have is not None:
                src = _transpose(src, 12 * round((target - have) / 12))
            voices[vid] = replace_span(ev, start, end, src)
        out.append((voices, {"span": [start, end], "filler_sample_id": frag["sample_id"],
                             "filler_bars": [fb, fb + n], "pc_shift": pc_shift}))
    return out


@operator("rhythm_flatten")
def rhythm_flatten(win: Window, ctx: dict, rng: random.Random, cfg: dict):
    """One attack per beat, at the pitch sounding on the beat.

    What survives is roughly a beat-level reduction: the harmony on each beat is
    Bach's, the passing and neighbour motion that animates it is gone. A pitch
    Bach held across a beat (or across the span edges) stays held; a pitch he
    repeated stays repeated.
    """
    out = []
    beat = beat_length_q(win.meter)
    need = cfg.get("min_note_reduction", 0.3)
    for n in _span_bar_choices(cfg):
        for start, end in _interior_spans(win, n):
            voices, before_n, after_n = {}, 0, 0
            for vid, ev in win.voices.items():
                _, inside, _ = cut(ev, start, end)
                new: list[Event] = []
                t = start
                while t < end:
                    m = sounding_at(ev, t)
                    if m is not None:
                        new.append((t, min(beat, end - t), m))
                    t += beat
                bounds = {start + k * beat: start + k * beat
                          for k in range(int((end - start) / beat) + 1)}
                merged = rejoin(replace_span(ev, start, end, new), ev, bounds)
                before_n += len(inside)
                after_n += len(cut(merged, start, end)[1])
                voices[vid] = merged
            if before_n and (before_n - after_n) / before_n >= need:
                out.append((voices, {"span": [start, end], "beat_q": str(beat),
                                     "notes_before": before_n, "notes_after": after_n}))
    return out


@operator("subject_return_removal")
def subject_return_removal(win: Window, ctx: dict, rng: random.Random, cfg: dict):
    """Overwrite a post-exposition entry with the same voice's free material.

    Only for pieces with labelled entries. The donor comes from elsewhere in the
    same piece and voice, outside every labelled subject entry in that voice, at
    the same metric position, so its rhythm and figuration are Bach's own and the
    only thing missing is the return of the subject.
    """
    entries = ctx.get("entries") or []
    if not entries:
        return []
    returns = [(vid, s, e, idx) for vid, s, e, idx in entries
               if idx >= ctx["voice_count"] and s >= win.bar_q
               and e <= win.length_q and vid in win.voices]
    if not returns:
        return []
    out = []
    piece_ev = ctx["piece_events"]
    abs_entries = ctx["abs_entries"]
    total = ctx["piece_length_q"]
    w0, w1 = win.start_q, win.start_q + win.length_q
    for vid, s, e, idx in returns:
        length = e - s
        busy = [(a, b) for v, a, b, _ in abs_entries if v == vid]
        starts = []
        p = (win.start_q + s) % win.bar_q
        while p + length <= total:
            clear = all(p + length <= a or p >= b for a, b in busy)
            outside = p + length <= w0 or p >= w1
            if clear and outside:
                starts.append(p)
            p += win.bar_q
        rng.shuffle(starts)
        ev = win.voices[vid]
        _, orig, _ = cut(ev, s, e)
        orig_mean = _mean_pitch(orig)
        for p in starts[:cfg.get("donors_per_entry", 6)]:
            _, donor, _ = cut(piece_ev[vid], p, p + length)
            sounding = sum((d for _, d, _ in donor), Fraction(0))
            if sounding < length * Fraction(str(cfg.get("min_donor_sounding", 0.6))):
                continue
            donor = [(o - p + s, d, m) for o, d, m in donor]
            dmean = _mean_pitch(donor)
            for shift in cfg.get("donor_shifts", [-12, -7, -5, 0, 5, 7, 12]):
                if orig_mean is not None and dmean is not None \
                        and abs(dmean + shift - orig_mean) > cfg.get("max_register_move", 7):
                    continue
                voices = dict(win.voices)
                voices[vid] = replace_span(ev, s, e, _transpose(donor, shift))
                out.append((voices, {"span": [s, e], "voice_id": vid,
                                     "entry_index": idx,
                                     "donor_start_q": str(p), "donor_shift": shift}))
    return out
