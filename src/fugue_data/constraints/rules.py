"""Rule implementations (Spec §15.5).

Each function takes (fragment, params) and yields violation dicts without a
severity: the engine attaches severity from the active profile, so the same
detection can be hard in one profile and diagnostic in another (§15.2).

Design note on §15.5's warning. Voice crossing, direct fifths, large leaps,
dissonance and repeated notes are deliberately NOT unconditional: each is either
a soft_preference, or a conditional_constraint whose detector first excludes the
contexts in which Bach uses the figure freely.
"""

from __future__ import annotations

from fractions import Fraction

from ..normalize.rational import frac_str, parse_frac

PERFECT_FIFTH, OCTAVE = 7, 12
_CONSONANT = {0, 3, 4, 7, 8, 9}          # unison, 3rds, P5, 6ths (mod 12)
_REGISTRY: dict[str, callable] = {}


def rule(fn):
    _REGISTRY[fn.__name__] = fn
    return fn


def registry() -> dict:
    return dict(_REGISTRY)


def _v(rule_id, frag, onset, voice_ids, note_ids, message, **evidence):
    return {"rule_id": rule_id, "piece_id": frag.piece_id,
            "voice_ids": list(voice_ids), "note_ids": list(note_ids),
            "onset_q": frac_str(onset), "measure_number": None,
            "evidence": evidence, "message": message}


# --------------------------------------------------------------------------- #
# Notational invariants
# --------------------------------------------------------------------------- #
@rule
def r_timing_nonnegative_onset(frag, p):
    for v in frag.voices:
        for n in v["notes"]:
            o = parse_frac(n["onset_q"])
            if o < 0:
                yield _v("timing_nonnegative_onset", frag, o, [v["voice_id"]],
                         [n["note_id"]], f"negative onset {n['onset_q']}")


@rule
def r_timing_positive_duration(frag, p):
    for v in frag.voices:
        for n in v["notes"]:
            if n.get("grace"):
                continue
            d = parse_frac(n["duration_q"])
            if d <= 0:
                yield _v("timing_positive_duration", frag, parse_frac(n["onset_q"]),
                         [v["voice_id"]], [n["note_id"]],
                         f"non-positive duration {n['duration_q']}")


@rule
def r_timing_within_bounds(frag, p):
    total = frag.quarter_length_total
    for v in frag.voices:
        for n in v["notes"]:
            end = parse_frac(n["onset_q"]) + parse_frac(n["duration_q"])
            if total and end > total:
                yield _v("timing_within_bounds", frag, parse_frac(n["onset_q"]),
                         [v["voice_id"]], [n["note_id"]],
                         f"note ends at {frac_str(end)}, past fragment end {frac_str(total)}")


@rule
def r_voice_monophonic(frag, p):
    for v in frag.voices:
        notes = frag.notes_sorted(v)
        for a, b in zip(notes, notes[1:]):
            ao, ad = parse_frac(a["onset_q"]), parse_frac(a["duration_q"])
            bo = parse_frac(b["onset_q"])
            if bo < ao + ad and not (a.get("grace") or b.get("grace")):
                yield _v("voice_monophonic", frag, bo, [v["voice_id"]],
                         [a["note_id"], b["note_id"]],
                         f"{v['voice_id']} sounds two pitches at {frac_str(bo)}",
                         overlap_q=frac_str(ao + ad - bo))


# --------------------------------------------------------------------------- #
# Range and spacing
# --------------------------------------------------------------------------- #
@rule
def r_voice_range(frag, p):
    ranges = p.get("ranges", {})
    for v in frag.voices:
        lo, hi = ranges.get(v["voice_id"], ranges.get("default", (0, 127)))
        for n in v["notes"]:
            if not (lo <= n["midi"] <= hi):
                yield _v("voice_range", frag, parse_frac(n["onset_q"]), [v["voice_id"]],
                         [n["note_id"]],
                         f"{v['voice_id']} midi {n['midi']} outside [{lo}, {hi}]",
                         midi=n["midi"], low=lo, high=hi)


@rule
def r_voice_spacing(frag, p):
    """Adjacent *upper* voices only: wide bass spacing is idiomatic, not an error."""
    limit = p.get("voice_spacing_max_semitones", 16)
    ids = frag.voice_ids()
    for t in frag.onsets():
        sounding = dict(frag.sounding_at(t))
        for a, b in zip(ids, ids[1:]):
            if b == ids[-1]:                     # the pair touching the bass is exempt
                continue
            if a in sounding and b in sounding:
                gap = sounding[a]["midi"] - sounding[b]["midi"]
                if gap > limit:
                    yield _v("voice_spacing", frag, t, [a, b],
                             [sounding[a]["note_id"], sounding[b]["note_id"]],
                             f"{a}-{b} spaced {gap} semitones (limit {limit})", gap=gap)


@rule
def r_voice_overlap(frag, p):
    """A lower voice rising above where the upper voice just was, or the reverse."""
    ids = frag.voice_ids()
    by_id = {v["voice_id"]: frag.notes_sorted(v) for v in frag.voices}
    for a, b in zip(ids, ids[1:]):
        for nb in by_id[b]:
            ob = parse_frac(nb["onset_q"])
            prev_a = [n for n in by_id[a] if parse_frac(n["onset_q"]) < ob]
            if not prev_a:
                continue
            if nb["midi"] > prev_a[-1]["midi"]:
                yield _v("voice_overlap", frag, ob, [a, b],
                         [prev_a[-1]["note_id"], nb["note_id"]],
                         f"{b} rises above the previous {a} pitch "
                         f"({nb['midi']} > {prev_a[-1]['midi']})")


@rule
def r_voice_crossing(frag, p):
    ids = frag.voice_ids()
    for t in frag.onsets():
        s = dict(frag.sounding_at(t))
        for a, b in zip(ids, ids[1:]):
            if a in s and b in s and s[b]["midi"] > s[a]["midi"]:
                yield _v("voice_crossing", frag, t, [a, b],
                         [s[a]["note_id"], s[b]["note_id"]],
                         f"{b} is above {a} at {frac_str(t)}")


# --------------------------------------------------------------------------- #
# Parallels
# --------------------------------------------------------------------------- #
def _pairs_in_motion(frag):
    """Successive simultaneities for each voice pair, where both voices move.

    "Successive" means genuinely adjacent: if either voice rests between the two
    simultaneities, they are not consecutive motion and parallel motion cannot be
    said to occur across the gap. Omitting that check reported parallel fifths in
    real Bach across a phrase break -- the corpus said the detector was wrong
    before any profile said so (§15.6).
    """
    ids = frag.voice_ids()
    onsets = frag.onsets()
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            prev = None
            prev_t = None
            for t in onsets:
                s = dict(frag.sounding_at(t))
                if a not in s or b not in s:
                    prev = None
                    prev_t = None
                    continue
                cur = (s[a], s[b])
                if prev is not None:
                    pa, pb = prev
                    # Both notes of the earlier pair must still be sounding right up
                    # to this onset; a gap means a rest intervened.
                    contiguous = all(
                        parse_frac(n["onset_q"]) + parse_frac(n["duration_q"]) >= t
                        for n in (pa, pb))
                    if contiguous and pa["midi"] != cur[0]["midi"] \
                            and pb["midi"] != cur[1]["midi"]:
                        yield a, b, prev, cur, t
                prev = cur
                prev_t = t


def _on_beat(frag, t: Fraction) -> bool:
    """True when `t` falls on a beat of the prevailing meter."""
    beat = Fraction(4, frag.meter[1])
    return t % beat == 0


def _parallel(frag, interval, rule_id, label, params=None):
    params = params or {}
    # Parallels between ornamental offbeat notes are not the prohibited figure.
    # BWV 848 m39 has textbook parallel fifths between two offbeat sixteenths in
    # a florid line -- the corpus insisting the unconditioned rule is too strong.
    # The strict profile can switch this context off and count every parallel.
    require_metric = params.get("parallel_requires_metric_onset", True)
    for a, b, prev, cur, t in _pairs_in_motion(frag):
        i0 = abs(prev[0]["midi"] - prev[1]["midi"]) % 12
        i1 = abs(cur[0]["midi"] - cur[1]["midi"]) % 12
        if i0 == interval and i1 == interval:
            # Contrary motion into the same interval class is not a parallel.
            d0 = cur[0]["midi"] - prev[0]["midi"]
            d1 = cur[1]["midi"] - prev[1]["midi"]
            if require_metric:
                prev_t = max(parse_frac(prev[0]["onset_q"]),
                             parse_frac(prev[1]["onset_q"]))
                if not (_on_beat(frag, prev_t) or _on_beat(frag, t)):
                    continue
            if d0 * d1 > 0:
                yield _v(rule_id, frag, t, [a, b],
                         [prev[0]["note_id"], prev[1]["note_id"],
                          cur[0]["note_id"], cur[1]["note_id"]],
                         f"parallel {label} between {a} and {b} at {frac_str(t)}",
                         from_interval=i0, to_interval=i1)


@rule
def r_parallel_fifths(frag, p):
    yield from _parallel(frag, PERFECT_FIFTH, "parallel_fifths", "perfect fifths", p)


@rule
def r_parallel_octaves(frag, p):
    yield from _parallel(frag, 0, "parallel_octaves", "octaves or unisons", p)


@rule
def r_direct_fifths_outer(frag, p):
    """Similar motion into a perfect consonance between the outer voices, with a
    leap on top. Never hard: §15.5 names direct fifths as context-dependent."""
    ids = frag.voice_ids()
    if len(ids) < 2:
        return
    top, bottom = ids[0], ids[-1]
    for a, b, prev, cur, t in _pairs_in_motion(frag):
        if (a, b) != (top, bottom):
            continue
        i1 = abs(cur[0]["midi"] - cur[1]["midi"]) % 12
        if i1 not in (0, PERFECT_FIFTH):
            continue
        d_top = cur[0]["midi"] - prev[0]["midi"]
        d_bot = cur[1]["midi"] - prev[1]["midi"]
        if d_top * d_bot > 0 and abs(d_top) > 2:
            yield _v("direct_fifths_outer", frag, t, [a, b],
                     [cur[0]["note_id"], cur[1]["note_id"]],
                     f"outer voices reach a perfect consonance by similar motion "
                     f"with a leap of {d_top} semitones")


# --------------------------------------------------------------------------- #
# Melody
# --------------------------------------------------------------------------- #
@rule
def r_melodic_leap_limit(frag, p):
    limit = p.get("max_leap_semitones", 12)
    allow_octave = p.get("allow_octave_leap", True)
    for v in frag.voices:
        merged = frag.merged(v)
        for (o0, d0, m0), (o1, _, m1) in zip(merged, merged[1:]):
            leap = abs(m1 - m0)
            if leap <= limit:
                continue
            if allow_octave and leap == OCTAVE:
                continue
            if o1 > o0 + d0:                      # leap across a rest is idiomatic
                continue
            yield _v("melodic_leap_limit", frag, o1, [v["voice_id"]], [],
                     f"{v['voice_id']} leaps {leap} semitones (limit {limit})",
                     leap=leap, limit=limit)


@rule
def r_augmented_interval_melodic(frag, p):
    for v in frag.voices:
        merged = frag.merged(v)
        for (o0, _, m0), (o1, _, m1) in zip(merged, merged[1:]):
            if abs(m1 - m0) in (3, 6) and abs(m1 - m0) == 6:
                yield _v("augmented_interval_melodic", frag, o1, [v["voice_id"]], [],
                         f"{v['voice_id']} moves by a tritone", interval=m1 - m0)


@rule
def r_leap_recovery(frag, p):
    for v in frag.voices:
        merged = frag.merged(v)
        for (o0, _, m0), (o1, _, m1), (o2, _, m2) in zip(merged, merged[1:], merged[2:]):
            leap = m1 - m0
            if abs(leap) < PERFECT_FIFTH:
                continue
            step = m2 - m1
            if step == 0 or (leap > 0 and step > 0) or (leap < 0 and step < 0) or abs(step) > 2:
                yield _v("leap_recovery", frag, o2, [v["voice_id"]], [],
                         f"{v['voice_id']} leaps {leap} then moves {step} "
                         f"instead of stepping back")


# --------------------------------------------------------------------------- #
# Tendency tones and dissonance
# --------------------------------------------------------------------------- #
@rule
def r_leading_tone_resolution(frag, p):
    """Only in outer voices at a cadential arrival: §15.5 forbids a blanket rule."""
    targets = frag.plan.get("cadence_targets") or []
    if not targets:
        return
    ids = frag.voice_ids()
    outer = {ids[0], ids[-1]}
    lt_pc = (frag.tonic_pc - 1) % 12
    for target in targets:
        t = parse_frac(str(target.get("onset_q", "0")))
        for vid, n in frag.sounding_at(max(Fraction(0), t - Fraction(1, 2))):
            if vid not in outer or n["pitch_class"] != lt_pc:
                continue
            voice = next(v for v in frag.voices if v["voice_id"] == vid)
            after = [x for x in frag.notes_sorted(voice)
                     if parse_frac(x["onset_q"]) >= parse_frac(n["onset_q"]) +
                     parse_frac(n["duration_q"])]
            if not after or after[0]["midi"] - n["midi"] != 1:
                yield _v("leading_tone_resolution", frag, t, [vid], [n["note_id"]],
                         f"leading tone in {vid} does not resolve up by semitone "
                         f"at the cadence")


def classify_dissonance(frag, vid: str, n: dict, bass_midi: int) -> str:
    """suspension | passing | neighbour | anticipation | unprepared | consonant."""
    iv = (n["midi"] - bass_midi) % 12
    if iv in _CONSONANT:
        return "consonant"
    voice = next(v for v in frag.voices if v["voice_id"] == vid)
    notes = frag.notes_sorted(voice)
    idx = next((i for i, x in enumerate(notes) if x["note_id"] == n["note_id"]), None)
    if idx is None:
        return "unprepared"
    prev = notes[idx - 1] if idx > 0 else None
    nxt = notes[idx + 1] if idx + 1 < len(notes) else None
    if prev is None:
        return "unprepared"
    approach = n["midi"] - prev["midi"]
    if nxt is None:
        return "unprepared"
    departure = nxt["midi"] - n["midi"]
    if approach == 0:
        # A dissonance repeated from the previous note IS a suspension -- it was
        # prepared as a consonance and held over. Whether it then resolves is
        # `suspension_resolution`'s question, not this classifier's. Folding the
        # resolution test in here made an *unresolved* suspension come back as
        # "unprepared", so the one rule meant to catch it could never fire.
        return "suspension"
    if abs(approach) <= 2 and abs(departure) <= 2:
        if approach * departure > 0:
            return "passing"
        return "neighbour"
    if abs(approach) <= 2 and departure == 0:
        return "anticipation"
    return "unprepared"


@rule
def r_dissonance_treatment(frag, p):
    for t in frag.onsets():
        sounding = frag.sounding_at(t)
        if len(sounding) < 2:
            continue
        bass = min(n["midi"] for _, n in sounding)
        for vid, n in sounding:
            if n["midi"] == bass:
                continue
            kind = classify_dissonance(frag, vid, n, bass)
            if kind == "unprepared":
                yield _v("dissonance_treatment", frag, t, [vid], [n["note_id"]],
                         f"{vid} enters a dissonance against the bass without "
                         f"preparation or stepwise approach", classification=kind)


@rule
def r_suspension_resolution(frag, p):
    """A suspension is prepared, struck over, and resolved down by step.

    All three parts matter. Checking only "is this repeated note dissonant now"
    reported a 93% false-positive rate against real Bach: any note held while the
    bass moves is dissonant against *some* passing harmony, and that is ordinary
    writing, not a suspension. So the figure is required to be (a) consonant when
    it was attacked -- the preparation -- and (b) dissonant on a metrically strong
    beat, which is where a suspension bites.
    """
    for t in frag.onsets():
        if not _on_beat(frag, t):
            continue                       # a suspension lands on a strong beat
        sounding = frag.sounding_at(t)
        if len(sounding) < 2:
            continue
        bass = min(n["midi"] for _, n in sounding)
        for vid, n in sounding:
            if n["midi"] == bass:
                continue
            if classify_dissonance(frag, vid, n, bass) != "suspension":
                continue
            # The preparation: consonant against the bass when it was struck.
            at = parse_frac(n["onset_q"])
            prep = frag.sounding_at(at)
            if prep:
                prep_bass = min(x["midi"] for _, x in prep)
                if (n["midi"] - prep_bass) % 12 not in _CONSONANT:
                    continue               # never a consonance: not a suspension
            voice = next(v for v in frag.voices if v["voice_id"] == vid)
            notes = frag.notes_sorted(voice)
            idx = next(i for i, x in enumerate(notes) if x["note_id"] == n["note_id"])
            nxt = notes[idx + 1] if idx + 1 < len(notes) else None
            if nxt is None or not (-2 <= nxt["midi"] - n["midi"] < 0):
                yield _v("suspension_resolution", frag, t, [vid], [n["note_id"]],
                         f"suspension in {vid} does not resolve down by step")


# --------------------------------------------------------------------------- #
# Fugal structure -- silent unless the fragment carries a plan
# --------------------------------------------------------------------------- #
def _interval_seq(events):
    return [b[2] - a[2] for a, b in zip(events, events[1:])]


@rule
def r_subject_entry_identity(frag, p):
    plan = frag.plan.get("entry_plan") or []
    subject = frag.plan.get("subject")
    if not plan or not subject:
        return
    want = list(subject.get("intervals", []))
    for entry in plan:
        # Answers are transformations by design; `answer_transformation` judges
        # them. Checking them here too would flag every real answer twice.
        if entry.get("role") == "answer":
            continue
        vid = entry["voice_id"]
        at = parse_frac(str(entry["onset_q"]))
        voice = next((v for v in frag.voices if v["voice_id"] == vid), None)
        if voice is None:
            yield _v("subject_entry_identity", frag, at, [vid], [],
                     f"scheduled entry voice {vid} does not exist")
            continue
        events = [e for e in frag.merged(voice) if e[0] >= at][:len(want) + 1]
        if len(events) < len(want) + 1:
            yield _v("subject_entry_identity", frag, at, [vid], [],
                     f"scheduled entry in {vid} at {frac_str(at)} is incomplete")
            continue
        if events[0][0] != at:
            yield _v("subject_entry_identity", frag, at, [vid], [],
                     f"scheduled entry in {vid} does not begin at {frac_str(at)}")
            continue
        got = _interval_seq(events)
        if got != want:
            yield _v("subject_entry_identity", frag, at, [vid], [],
                     f"entry in {vid} does not match the subject intervals",
                     expected=want, got=got)


@rule
def r_answer_transformation(frag, p):
    plan = frag.plan.get("entry_plan") or []
    subject = frag.plan.get("subject")
    if not plan or not subject:
        return
    want = list(subject.get("intervals", []))
    for entry in plan:
        if entry.get("role") != "answer":
            continue
        vid = entry["voice_id"]
        at = parse_frac(str(entry["onset_q"]))
        voice = next((v for v in frag.voices if v["voice_id"] == vid), None)
        if voice is None:
            continue
        events = [e for e in frag.merged(voice) if e[0] >= at][:len(want) + 1]
        if len(events) < len(want) + 1:
            continue
        got = _interval_seq(events)
        real = got == want
        # A tonal answer adjusts only the opening degree.
        tonal = len(got) == len(want) and got[1:] == want[1:] and abs(got[0] - want[0]) <= 2
        if not (real or tonal):
            yield _v("answer_transformation", frag, at, [vid], [],
                     f"answer in {vid} is neither a real nor a tonal transformation "
                     f"of the subject", expected=want, got=got)


@rule
def r_cadence_target_completion(frag, p):
    for target in frag.plan.get("cadence_targets") or []:
        t = parse_frac(str(target.get("onset_q", "0")))
        sounding = frag.sounding_at(max(Fraction(0), t - Fraction(1, 64)))
        if not sounding:
            yield _v("cadence_target_completion", frag, t, [], [],
                     f"scheduled cadence at {frac_str(t)} has no sounding voices")
            continue
        pcs = {n["pitch_class"] for _, n in sounding}
        want_pc = target.get("tonic_pc", frag.tonic_pc)
        bass = min(sounding, key=lambda x: x[1]["midi"])[1]
        if bass["pitch_class"] != want_pc or want_pc not in pcs:
            yield _v("cadence_target_completion", frag, t, [], [],
                     f"scheduled cadence at {frac_str(t)} does not arrive on "
                     f"pitch class {want_pc}", bass_pc=bass["pitch_class"])


@rule
def r_voice_participation(frag, p):
    need = p.get("min_voice_participation", 0.2)
    total = frag.quarter_length_total
    if not total:
        return
    for v in frag.voices:
        sounding = sum(parse_frac(n["duration_q"]) for n in v["notes"])
        share = float(sounding / total)
        if share < need:
            yield _v("voice_participation", frag, Fraction(0), [v["voice_id"]], [],
                     f"{v['voice_id']} sounds for {share:.0%} of the fragment "
                     f"(minimum {need:.0%})", share=round(share, 3))


@rule
def r_max_dead_air(frag, p):
    """Bounded silence -- but only for fragments that are musically complete.

    The BCFB chorales are whole cantata movements from which the instrumental
    parts were deliberately excluded (§10.2), so the vocal parts fall silent for
    the introduction and the interludes. That silence is where the orchestra
    plays, not dead air, and treating it as a violation would flag Bach for music
    he wrote and the pipeline chose not to import. An extract therefore declares
    itself and the rule stands down, rather than the rule being weakened for
    everyone (§15.5: contextual conditions, not blanket severity).
    """
    if getattr(frag, "is_extract", False):
        return
    limit = Fraction(str(p.get("max_dead_air_q", 4)))
    spans = []
    for v in frag.voices:
        for n in v["notes"]:
            o = parse_frac(n["onset_q"])
            spans.append((o, o + parse_frac(n["duration_q"])))
    if not spans:
        return
    spans.sort()
    cursor = Fraction(0)
    for s, e in spans:
        if s - cursor > limit:
            yield _v("max_dead_air", frag, cursor, [], [],
                     f"{frac_str(s - cursor)} quarters of silence from "
                     f"{frac_str(cursor)} (limit {frac_str(limit)})")
        cursor = max(cursor, e)
    if frag.quarter_length_total - cursor > limit:
        yield _v("max_dead_air", frag, cursor, [], [],
                 f"fragment ends with {frac_str(frag.quarter_length_total - cursor)} "
                 f"quarters of silence")


@rule
def r_section_length_limit(frag, p):
    bounds = frag.plan.get("length_bars")
    if not bounds or frag.measure_count is None:
        return
    lo, hi = bounds
    if not (lo <= frag.measure_count <= hi):
        yield _v("section_length_limit", frag, Fraction(0), [], [],
                 f"fragment is {frag.measure_count} bars, outside [{lo}, {hi}]")


@rule
def r_structural_target_reachable(frag, p):
    """No state from which a required structural target can no longer be met."""
    total = frag.quarter_length_total
    subject_len = Fraction(str((frag.plan.get("subject") or {}).get("length_q", 0)))
    for entry in frag.plan.get("entry_plan") or []:
        at = parse_frac(str(entry["onset_q"]))
        if at + subject_len > total:
            yield _v("structural_target_reachable", frag, at, [entry["voice_id"]], [],
                     f"scheduled entry at {frac_str(at)} cannot complete before the "
                     f"fragment ends at {frac_str(total)}")
    for target in frag.plan.get("cadence_targets") or []:
        t = parse_frac(str(target.get("onset_q", "0")))
        if t > total:
            yield _v("structural_target_reachable", frag, t, [], [],
                     f"scheduled cadence at {frac_str(t)} lies past the fragment end")
