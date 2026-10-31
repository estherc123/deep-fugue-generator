"""Propose 2-8 bar real-Bach excerpt candidates (Spec §14.1-§14.3).

Proposal is automatic; *approval* is not. Every record leaves this stage with
`review_status: pending`, and §27 forbids any downstream export presenting a
pending excerpt as human-approved.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction

from ..normalize.rational import frac_str, parse_frac
from ..tokenization.views import merge_tied

# Intervals (mod 12) treated as dissonances needing resolution at a boundary.
_DISSONANT = {1, 2, 6, 10, 11}


# --------------------------------------------------------------------------- #
# Score helpers
# --------------------------------------------------------------------------- #
def measure_table(score: dict) -> list[tuple[int, str, Fraction]]:
    starts: dict[int, tuple[str, Fraction]] = {}
    for v in score["voices"]:
        for n in v["notes"]:
            idx = n.get("measure_index_unfolded")
            if idx is None:
                continue
            st = parse_frac(n["onset_q"]) - parse_frac(n["offset_in_measure_q"])
            prev = starts.get(idx)
            if prev is None or st < prev[1]:
                starts[idx] = (str(n.get("measure_number")), st)
    return [(i, num, st) for i, (num, st) in sorted(starts.items())]


def _bar_index_at(onset: Fraction, bars) -> int:
    lo = 0
    for k, (_, _, st) in enumerate(bars):
        if st <= onset:
            lo = k
        else:
            break
    return lo


def notes_in(score: dict, start: Fraction, end: Fraction) -> list[dict]:
    out = []
    for v in score["voices"]:
        for n in v["notes"]:
            o = parse_frac(n["onset_q"])
            if start <= o < end:
                out.append(n)
    return out


def sounding_voices(score: dict, start: Fraction, end: Fraction) -> set[str]:
    out = set()
    for v in score["voices"]:
        for n in v["notes"]:
            o = parse_frac(n["onset_q"])
            if o + parse_frac(n["duration_q"]) > start and o < end:
                out.add(v["voice_id"])
                break
    return out


# --------------------------------------------------------------------------- #
# §14.2 candidate triggers
# --------------------------------------------------------------------------- #
def _window_around(bars, bar_idx: int, span: int, total: Fraction) -> tuple[Fraction, Fraction] | None:
    if bar_idx < 0 or bar_idx >= len(bars):
        return None
    start = bars[bar_idx][2]
    end_idx = bar_idx + span
    end = bars[end_idx][2] if end_idx < len(bars) else total
    if end <= start:
        return None
    return start, end


def propose_windows(score: dict, annotations: list[dict], cfg: dict) -> list[dict]:
    """Candidate (start, end, reasons) windows before rejection filtering."""
    pcfg = cfg["proposal"]
    trig = pcfg["triggers"]
    bars = measure_table(score)
    if len(bars) < pcfg["min_bars"]:
        return []
    total = parse_frac(score.get("quarter_length_total", "0"))
    cap = pcfg["max_per_trigger_per_piece"]
    counts: dict[str, int] = {}
    found: dict[tuple[str, str], set[str]] = {}

    def add(reason: str, start: Fraction, end: Fraction):
        if counts.get(reason, 0) >= cap:
            return
        key = (frac_str(start), frac_str(end))
        if key not in found:
            counts[reason] = counts.get(reason, 0) + 1
        found.setdefault(key, set()).add(reason)

    accepted = [a for a in annotations
                if a.get("analytical_status") in (None, "accepted")]
    by_type: dict[str, list[dict]] = {}
    for a in accepted:
        by_type.setdefault(a["type"], []).append(a)

    def span_for(a) -> tuple[Fraction, Fraction]:
        s = parse_frac(a["onset_q"])
        d = parse_frac(a["duration_q"]) if a.get("duration_q") else Fraction(0)
        return s, s + d

    # Complete subject / answer entries: the bar the entry starts in, through the
    # bar it ends in, padded to at least min_bars.
    if trig.get("subject_entry"):
        for a in by_type.get("subject", []) + by_type.get("answer", []):
            s, e = span_for(a)
            bi, be = _bar_index_at(s, bars), _bar_index_at(max(e - Fraction(1, 64), s), bars)
            span = max(pcfg["min_bars"], min(pcfg["max_bars"], be - bi + 1))
            w = _window_around(bars, bi, span, total)
            if w:
                add("subject_entry", *w)

    # Countersubject sounding against an entry.
    if trig.get("countersubject_combination"):
        for a in by_type.get("countersubject", []):
            s, e = span_for(a)
            against = [x for x in by_type.get("subject", []) if
                       span_for(x)[0] < e and span_for(x)[1] > s]
            if not against:
                continue
            bi = _bar_index_at(s, bars)
            be = _bar_index_at(max(e - Fraction(1, 64), s), bars)
            span = max(pcfg["min_bars"], min(pcfg["max_bars"], be - bi + 1))
            w = _window_around(bars, bi, span, total)
            if w:
                add("countersubject_combination", *w)

    # Exposition boundary: the bars around the last entry of the opening run.
    if trig.get("exposition_boundary") and by_type.get("subject"):
        entries = sorted(by_type["subject"], key=lambda a: parse_frac(a["onset_q"]))
        n_voices = score["voice_count"]
        if len(entries) >= n_voices:
            last = entries[n_voices - 1]
            bi = _bar_index_at(span_for(last)[0], bars)
            w = _window_around(bars, max(0, bi - 1), pcfg["max_bars"] // 2 + 1, total)
            if w:
                add("exposition_boundary", *w)

    # Cadence arrivals: the approach into the cadence, not just the chord.
    if trig.get("cadence_arrival"):
        for a in by_type.get("cadence", []):
            bi = _bar_index_at(parse_frac(a["onset_q"]), bars)
            w = _window_around(bars, max(0, bi - 2), 3, total)
            if w:
                add("cadence_arrival", *w)

    if trig.get("pedal_point"):
        for a in by_type.get("pedal", []):
            s, e = span_for(a)
            bi = _bar_index_at(s, bars)
            be = _bar_index_at(max(e - Fraction(1, 64), s), bars)
            span = max(pcfg["min_bars"], min(pcfg["max_bars"], be - bi + 1))
            w = _window_around(bars, bi, span, total)
            if w:
                add("pedal_point", *w)

    if trig.get("short_episode"):
        for a in by_type.get("episode", []):
            s, e = span_for(a)
            bi = _bar_index_at(s, bars)
            be = _bar_index_at(max(e - Fraction(1, 64), s), bars)
            if be - bi + 1 <= pcfg["max_bars"]:
                w = _window_around(bars, bi, max(pcfg["min_bars"], be - bi + 1), total)
                if w:
                    add("short_episode", *w)

    # Stretto: two subject entries overlapping in different voices -- derived from
    # the annotations rather than labelled, since Algomus has no stretto tag.
    if trig.get("stretto"):
        subs = sorted(by_type.get("subject", []), key=lambda a: parse_frac(a["onset_q"]))
        for i, a in enumerate(subs):
            sa, ea = span_for(a)
            for b in subs[i + 1:]:
                sb, eb = span_for(b)
                if sb >= ea:
                    break
                if a.get("voice_id") and b.get("voice_id") and a["voice_id"] != b["voice_id"]:
                    bi = _bar_index_at(sa, bars)
                    be = _bar_index_at(max(eb - Fraction(1, 64), sa), bars)
                    span = max(pcfg["min_bars"], min(pcfg["max_bars"], be - bi + 1))
                    w = _window_around(bars, bi, span, total)
                    if w:
                        add("stretto", *w)
                    break

    # Texture change: the count of sounding voices changes cleanly at a barline.
    if trig.get("texture_change"):
        prev = None
        for k, (_, _, st) in enumerate(bars):
            end = bars[k + 1][2] if k + 1 < len(bars) else total
            cur = len(sounding_voices(score, st, end))
            if prev is not None and cur != prev:
                w = _window_around(bars, max(0, k - 1), pcfg["min_bars"] + 1, total)
                if w:
                    add("texture_change", *w)
            prev = cur

    # Sequence: a voice repeats a melodic interval pattern at a shifted pitch level.
    if trig.get("sequence"):
        for v in score["voices"]:
            merged = merge_tied(v["notes"])
            if len(merged) < 8:
                continue
            midis = [m[2] for m in merged]
            ivs = [b - a for a, b in zip(midis, midis[1:])]
            for i in range(len(ivs) - 6):
                a, b = ivs[i:i + 3], ivs[i + 3:i + 6]
                if a == b and any(x != 0 for x in a):
                    shift = midis[i + 3] - midis[i]
                    if shift != 0:
                        st = merged[i][0]
                        bi = _bar_index_at(st, bars)
                        w = _window_around(bars, bi, pcfg["min_bars"] + 1, total)
                        if w:
                            add("sequence", *w)
                        break

    out = []
    for (s, e), reasons in sorted(found.items()):
        out.append({"start_onset_q": s, "end_onset_q": e,
                    "proposal_reasons": sorted(reasons)})
    return out


# --------------------------------------------------------------------------- #
# §14.3 automatic rejection
# --------------------------------------------------------------------------- #
def rejection_reasons(score: dict, annotations: list[dict], start: Fraction,
                      end: Fraction, cfg: dict) -> list[str]:
    r = cfg["rejection"]
    reasons: list[str] = []

    if r.get("reject_parser_warnings"):
        codes = {w.get("code") for w in score.get("warnings", [])}
        blocking = {c for c in codes if c and not c.startswith("voice_")}
        if blocking:
            reasons.append(f"parser_warnings:{sorted(blocking)[0]}")

    notes = notes_in(score, start, end)
    if len(notes) < r.get("min_notes", 1):
        reasons.append("too_few_notes")

    # §14.3 rejects a boundary inside a tied note "unless context is preserved".
    # The two boundaries are not symmetric. A window that BEGINS inside a tie
    # group presents a note whose attack the listener never hears, so its context
    # genuinely is missing and it is rejected. A window that ENDS inside one just
    # truncates a note that has already been heard to begin; the render and MIDI
    # clip it, and `clipped_boundary_notes` on the record declares it -- so the
    # context is preserved and the candidate stands. Rejecting both would discard
    # nearly every window in this corpus, where ties cross most barlines.
    if r.get("reject_tied_boundary"):
        for v in score["voices"]:
            if any(n.get("tie") in ("continue", "stop")
                   and parse_frac(n["onset_q"]) <= start
                   < parse_frac(n["onset_q"]) + parse_frac(n["duration_q"])
                   for n in v["notes"]):
                reasons.append("begins_inside_tie")
                break

    # Cuts through an annotated subject entry.
    if r.get("reject_cut_subject"):
        for a in annotations:
            if a.get("type") not in ("subject", "answer"):
                continue
            if a.get("analytical_status") not in (None, "accepted"):
                continue
            s = parse_frac(a["onset_q"])
            e = s + (parse_frac(a["duration_q"]) if a.get("duration_q") else Fraction(0))
            if s < start < e or s < end < e:
                reasons.append("cuts_subject_entry")
                break

    # Voice identity is about being able to tell the lines apart, not about every
    # voice sounding: in a fugue exposition the later voices have not entered yet,
    # and voices rest for extended periods by design (§8.4). So the test is that
    # at least two distinct voices sound and each keeps its canonical identity.
    if r.get("reject_voice_identity_loss"):
        sounding = sounding_voices(score, start, end)
        if len(sounding) < r.get("min_sounding_voices", 2):
            reasons.append("voice_identity_not_preserved")

    if r.get("reject_unresolved_suspension") and ends_on_unresolved_suspension(
            score, start, end):
        reasons.append("ends_on_unresolved_suspension")

    return sorted(set(reasons))


def ends_on_unresolved_suspension(score: dict, start: Fraction, end: Fraction) -> bool:
    """True when the window stops on a suspension that never resolves inside it.

    A suspension is a specific figure, not any dissonance: a note *held over* from
    an earlier attack that is dissonant against the lowest sounding pitch and then
    steps down. So this looks for a note that (a) is still sounding at the final
    instant, (b) attacked strictly before it, (c) is dissonant against the bass,
    and (d) has no stepwise descent in its own voice inside the window. Treating
    every dissonant vertical as a rejection instead would throw out most of the
    corpus -- passing sevenths and ninths are ordinary in this style.
    """
    last = end - Fraction(1, 64)
    sounding: list[tuple[str, int, Fraction]] = []
    for v in score["voices"]:
        for n in v["notes"]:
            o = parse_frac(n["onset_q"])
            if o <= last < o + parse_frac(n["duration_q"]):
                sounding.append((v["voice_id"], n["midi"], o))
    if len(sounding) < 2:
        return False
    bass = min(m for _, m, _ in sounding)

    # "Held over" means sustained across the last beat, not merely attacked
    # earlier in the bar -- otherwise almost every note sounding at the end
    # qualifies and the test degenerates into "any dissonance at the end".
    beat = Fraction(1)
    for vid, midi, onset in sounding:
        if midi == bass or onset > end - beat:
            continue                                  # not held over the beat
        if (midi - bass) % 12 not in _DISSONANT:
            continue                                  # not a dissonance to resolve
        voice = next(v for v in score["voices"] if v["voice_id"] == vid)
        resolves = any(
            onset < parse_frac(n["onset_q"]) < end and 0 < midi - n["midi"] <= 2
            for n in voice["notes"])
        if not resolves:
            return True
    return False


def snap_window(score: dict, annotations: list[dict], start: Fraction, end: Fraction,
                bars, cfg: dict) -> tuple[Fraction, Fraction] | None:
    """Widen a window outward so it no longer cuts a tie or a subject entry.

    §14.3 rejects a boundary inside a tied note *unless context is preserved*, so
    the productive response is to preserve the context -- extend to the next bar
    line -- and only reject when that cannot be done within the bar budget.
    """
    max_bars = cfg["proposal"]["max_bars"]
    bar_starts = [st for _, _, st in bars]
    total = parse_frac(score.get("quarter_length_total", "0"))

    def crosses(a: Fraction, b: Fraction) -> bool:
        for v in score["voices"]:
            for n in v["notes"]:
                o = parse_frac(n["onset_q"])
                e = o + parse_frac(n["duration_q"])
                if n.get("tie") in ("start", "continue") and o < b < e:
                    return True
                if n.get("tie") in ("continue", "stop") and o < a < e:
                    return True
        for ann in annotations:
            if ann.get("type") not in ("subject", "answer"):
                continue
            if ann.get("analytical_status") not in (None, "accepted"):
                continue
            s0 = parse_frac(ann["onset_q"])
            e0 = s0 + (parse_frac(ann["duration_q"]) if ann.get("duration_q") else Fraction(0))
            if s0 < a < e0 or s0 < b < e0:
                return True
        return False

    orig_end = end
    end = _musical_end(score, annotations, start, end, bars, cfg, total)

    def cuts_tie_at_end(b: Fraction) -> bool:
        for v in score["voices"]:
            for n in v["notes"]:
                o = parse_frac(n["onset_q"])
                if n.get("tie") in ("start", "continue") \
                        and o < b < o + parse_frac(n["duration_q"]):
                    return True
        return False

    closes = set(_close_points(score, annotations, bars, total))

    def acceptable(a: Fraction, b: Fraction) -> bool:
        if not crosses(a, b):
            return True
        # An end that is a sanctioned musical close is allowed to fall inside a
        # later entry. In a fugue the entries overlap, so a section closes on the
        # cadence precisely BECAUSE the next entry has already begun -- refusing
        # that rejected 64% of windows. A tie is still a hard stop: ending inside
        # one leaves a sound with no resolution (§14.3).
        return b in closes and not cuts_tie_at_end(b) and not crosses(a, a)

    for _ in range(max_bars):
        if acceptable(start, end):
            return start, end
        # Widen the end to the next MUSICAL close, not the next barline.
        #
        # _musical_end places the end one beat past the last entry landing, but in
        # a fugue the entries overlap, so that point often falls inside the next
        # entry and `crosses` fires. Snapping to the next barline from there threw
        # the musical end away again -- window 0-12 became 0-13 and then 0-24.
        # Barlines remain the fallback when no musical close is available.
        later = [c for c in _close_points(score, annotations, bars, total) if c > end]
        later += [st for st in bar_starts if st > end]
        later = sorted(set(later))
        moved = False
        if later:
            cand_end = later[0]
            if _bar_span(bar_starts, start, cand_end) <= max_bars:
                end = cand_end
                moved = True
        if not moved:
            earlier = [st for st in bar_starts if st < start]
            if earlier and _bar_span(bar_starts, earlier[-1], end) <= max_bars:
                start = earlier[-1]
                moved = True
        if not moved:
            return None
    if acceptable(start, end):
        return start, min(end, total)
    # Falling back to None is not neutral: the caller then keeps the raw,
    # barline-snapped window, which is the bug this function exists to fix. A
    # window that ends at the last musical close is better than one that ends on
    # an arbitrary barline, even when some entry is still sounding across it.
    best = _musical_end(score, annotations, start, orig_end, bars, cfg, total)
    if best > start and _bar_span(bar_starts, start, best) <= max_bars:
        return start, min(best, total)
    return None


def _close_points(score: dict, annotations: list[dict], bars,
                  total: Fraction) -> list[Fraction]:
    """Points at which a fugue section can legitimately close.

    One beat past each annotated entry's landing and each cadence arrival, plus
    the places a straddling tie resolves. Cached per score so the widening loop
    does not recompute it on every iteration.
    """
    key = id(score)
    hit = _CLOSE_CACHE.get(key)
    if hit is not None:
        return hit
    beat = _beat_length(score)
    pts: set[Fraction] = set()
    for ann in annotations:
        if ann.get("analytical_status") not in (None, "accepted"):
            continue
        s0 = parse_frac(ann["onset_q"])
        kind = ann.get("type")
        if kind in ("subject", "answer", "countersubject") and ann.get("duration_q"):
            pts.add(s0 + parse_frac(ann["duration_q"]) + beat)
        elif kind == "cadence":
            pts.add(s0 + beat)
    for v in score["voices"]:
        for n in v["notes"]:
            if n.get("tie") in ("start", "continue"):
                pts.add(parse_frac(n["onset_q"]) + parse_frac(n["duration_q"]))
    out = sorted(x for x in pts if 0 < x <= (total or x))
    _CLOSE_CACHE[key] = out
    return out


_CLOSE_CACHE: dict = {}


def _musical_end(score: dict, annotations: list[dict], start: Fraction,
                 end: Fraction, bars, cfg: dict, total: Fraction) -> Fraction:
    """Place the window end where the music closes, not on the next barline.

    Snapping the end to a barline is one bug with two faces. A fugue section
    closes on the downbeat the cadence lands on, and that downbeat usually also
    begins the next episode. Snapping therefore either swallows the whole
    following bar ("runs on into the next episode", 22 windows) or stops on the
    barline just before the landing beat and cuts the resolution off (35 windows)
    -- depending only on where the cadence happens to fall. 57 of 91 unresolved
    corrections were these two faces of the same mistake.

    So: find every point the window is obliged to reach -- each annotated entry's
    landing and each cadence arrival inside it -- take the latest, and give it one
    beat to land on. Then let a tie straddling that point resolve, capped at half
    a bar so one long tie cannot drag the window open indefinitely.
    """
    beat = _beat_length(score)
    bar_len = beat * _beats_per_bar(bars, end, beat)

    required: list[Fraction] = []
    for ann in annotations:
        if ann.get("analytical_status") not in (None, "accepted"):
            continue
        kind = ann.get("type")
        s0 = parse_frac(ann["onset_q"])
        if kind in ("subject", "answer", "countersubject"):
            d = parse_frac(ann["duration_q"]) if ann.get("duration_q") else Fraction(0)
            # The entry's span ends BEFORE the note that resolves it -- measured,
            # 190 of 191 annotated entries. One beat carries the landing note.
            if d and start < s0 + d <= end:
                required.append(s0 + d + beat)
        elif kind == "cadence" and start < s0 <= end:
            required.append(s0 + beat)
    if required:
        end = max(end, max(required))

    # Let a tie straddling the end resolve, within half a bar.
    cap = end + bar_len / 2
    for v in score["voices"]:
        for n in v["notes"]:
            o = parse_frac(n["onset_q"])
            e = o + parse_frac(n["duration_q"])
            if n.get("tie") in ("start", "continue") and o < end < e and e <= cap:
                end = max(end, e)

    # Stopping one or two bars short of the final cadence leaves the piece
    # hanging (11 windows). If the end is that close, take it to the end.
    if total and 0 < total - end <= 3 * bar_len:
        end = total
    return min(end, total) if total else end


def _beat_length(score: dict) -> Fraction:
    """The felt beat, which in a compound meter is dotted.

    Returning 4/denominator treats 6/8 as having an eighth-note beat, so "one beat
    past the landing" stopped a third of the way through the real beat -- several
    fugue 15 windows ended on a V7 because of it. In 6/8, 9/8, 12/8 and their /16
    equivalents the beat is three of the notated units.
    """
    mm = (score.get("meter_map") or [{}])[0]
    den = int(mm.get("denominator") or 4)
    num = int(mm.get("numerator") or 4)
    if den <= 0:
        den = 4
    unit = Fraction(4, den)
    if den >= 8 and num > 3 and num % 3 == 0:
        return unit * 3
    return unit


def _beats_per_bar(bars, at: Fraction, beat: Fraction) -> int:
    starts = [st for _, _, st in bars]
    for a, b in zip(starts, starts[1:]):
        if a <= at < b:
            return max(1, int((b - a) / beat))
    if len(starts) >= 2:
        return max(1, int((starts[1] - starts[0]) / beat))
    return 4


def _bar_span(bar_starts, a: Fraction, b: Fraction) -> int:
    return sum(1 for st in bar_starts if a <= st < b) or 1


def excerpt_id(piece_id: str, start: Fraction, end: Fraction) -> str:
    h = hashlib.sha256(f"{piece_id}:{start}:{end}".encode()).hexdigest()[:10]
    return f"ex:{piece_id}:{h}"


def clipped_boundary_notes(score: dict, start: Fraction, end: Fraction) -> int:
    """Notes still sounding at the window end, truncated by it.

    Declared on every review record so a clipped release is visible to the
    reviewer rather than silently absorbed (§14.3 "context is preserved").
    """
    n = 0
    for v in score["voices"]:
        for note in v["notes"]:
            o = parse_frac(note["onset_q"])
            if o < end < o + parse_frac(note["duration_q"]):
                n += 1
    return n
