"""Rule-aware mutators for controlled invalid generation (Spec §17.3).

Each mutator takes a hard-valid fragment and edits it so that exactly one named
hard rule is broken. A mutator only proposes; §17.2 requires the result to be
re-validated, and acceptance demands the intended violation appear and no
unintended one come with it.
"""

from __future__ import annotations

import copy
import random
from fractions import Fraction

from ..normalize.rational import frac_str, parse_frac

_REGISTRY: dict[str, dict] = {}


def mutator(mutation_id: str, target_rule_id: str):
    def deco(fn):
        _REGISTRY[mutation_id] = {"fn": fn, "target_rule_id": target_rule_id,
                                  "mutation_id": mutation_id}
        return fn
    return deco


def registry() -> dict:
    return dict(_REGISTRY)


def _notes_by_voice(doc: dict) -> dict[str, list[dict]]:
    return {v["voice_id"]: sorted(v["notes"], key=lambda n: parse_frac(n["onset_q"]))
            for v in doc["voices"]}


def _set_midi(note: dict, midi: int) -> None:
    note["midi"] = int(midi)
    note["pitch_class"] = int(midi) % 12
    note["octave"] = int(midi) // 12 - 1


def _simultaneous(doc: dict):
    """(onset, {voice_id: note}) for onsets where at least two voices attack."""
    by_onset: dict[Fraction, dict[str, dict]] = {}
    for v in doc["voices"]:
        for n in v["notes"]:
            by_onset.setdefault(parse_frac(n["onset_q"]), {})[v["voice_id"]] = n
    return sorted((t, s) for t, s in by_onset.items() if len(s) >= 2)


def _consecutive_pairs(doc: dict):
    """Adjacent simultaneities sharing two voices that both move."""
    sims = _simultaneous(doc)
    for (t0, s0), (t1, s1) in zip(sims, sims[1:]):
        shared = sorted(set(s0) & set(s1))
        for i, a in enumerate(shared):
            for b in shared[i + 1:]:
                yield t0, t1, a, b, s0, s1


# --------------------------------------------------------------------------- #
def _force_parallel(doc, interval: int, rule_label: str):
    """Rewrite an upper voice so two adjacent simultaneities move in parallel.

    Positions where the edit would change nothing -- the pair already moves in
    parallel at this interval -- are skipped rather than returned: a mutation that
    leaves the fragment identical is not a controlled negative, and returning one
    just burns an attempt that the runner then has to reject.
    """
    for t0, t1, a, b, s0, s1 in _consecutive_pairs(doc):
        base0, base1 = s0[b]["midi"], s1[b]["midi"]
        if base0 == base1:
            continue                       # lower voice must move too
        want0, want1 = base0 + interval, base1 + interval
        if s0[a]["midi"] == want0 and s1[a]["midi"] == want1:
            continue                       # already parallel here: no-op
        if not (21 <= want0 <= 107 and 21 <= want1 <= 107):
            continue
        _set_midi(s0[a], want0)
        _set_midi(s1[a], want1)
        return {"voice_ids": [a, b], "onset_q": frac_str(t1),
                "detail": f"forced {rule_label} in {a} above {b}"}
    return None


@mutator("create_parallel_fifth", "parallel_fifths")
def m_parallel_fifth(doc, rng, params):
    return _force_parallel(doc, 7, "perfect fifths")


@mutator("create_parallel_octave", "parallel_octaves")
def m_parallel_octave(doc, rng, params):
    return _force_parallel(doc, 12, "octaves")


@mutator("exceed_voice_range", "voice_range")
def m_exceed_range(doc, rng, params):
    ranges = params.get("ranges", {})
    for v in doc["voices"]:
        lo, hi = ranges.get(v["voice_id"], ranges.get("default", (0, 127)))
        notes = sorted(v["notes"], key=lambda n: parse_frac(n["onset_q"]))
        if not notes:
            continue
        n = notes[len(notes) // 2]
        _set_midi(n, min(107, hi + 6))
        return {"voice_ids": [v["voice_id"]], "onset_q": n["onset_q"],
                "detail": f"pushed {v['voice_id']} to midi {n['midi']} above {hi}"}
    return None


@mutator("create_prohibited_melodic_leap", "melodic_leap_limit")
def m_prohibited_leap(doc, rng, params):
    limit = params.get("max_leap_semitones", 12)
    for v in doc["voices"]:
        notes = sorted(v["notes"], key=lambda n: parse_frac(n["onset_q"]))
        for a, b in zip(notes, notes[1:]):
            # Only where the notes are contiguous: a leap across a rest is idiomatic.
            if parse_frac(a["onset_q"]) + parse_frac(a["duration_q"]) != parse_frac(b["onset_q"]):
                continue
            target = a["midi"] + limit + 3
            if 21 <= target <= 107:
                _set_midi(b, target)
                return {"voice_ids": [v["voice_id"]], "onset_q": b["onset_q"],
                        "detail": f"leap of {limit + 3} semitones in {v['voice_id']}"}
    return None


@mutator("create_illegal_voice_overlap", "voice_overlap")
def m_voice_overlap(doc, rng, params):
    ids = [v["voice_id"] for v in doc["voices"]]
    by = _notes_by_voice(doc)
    for a, b in zip(ids, ids[1:]):
        for nb in by[b]:
            ob = parse_frac(nb["onset_q"])
            prev_a = [n for n in by[a] if parse_frac(n["onset_q"]) < ob]
            if not prev_a:
                continue
            target = prev_a[-1]["midi"] + 2
            if nb["midi"] < target <= 107:
                _set_midi(nb, target)
                return {"voice_ids": [a, b], "onset_q": nb["onset_q"],
                        "detail": f"{b} pushed above the previous {a} pitch"}
    return None


@mutator("break_leading_tone_resolution", "leading_tone_resolution")
def m_break_leading_tone(doc, rng, params):
    tonic = params.get("_tonic_pc", 0)
    lt = (tonic - 1) % 12
    ids = [v["voice_id"] for v in doc["voices"]]
    outer = {ids[0], ids[-1]}
    for v in doc["voices"]:
        if v["voice_id"] not in outer:
            continue
        notes = sorted(v["notes"], key=lambda n: parse_frac(n["onset_q"]))
        for i, n in enumerate(notes[:-1]):
            if n["pitch_class"] == lt and notes[i + 1]["midi"] - n["midi"] == 1:
                _set_midi(notes[i + 1], n["midi"] - 2)       # falls away instead
                return {"voice_ids": [v["voice_id"]], "onset_q": notes[i + 1]["onset_q"],
                        "detail": "leading tone falls instead of resolving up"}
    return None


def _entry_spans(params) -> list[tuple[str, Fraction]]:
    return [(e["voice_id"], parse_frac(str(e["onset_q"])))
            for e in (params.get("_entry_plan") or [])]


@mutator("create_unresolved_suspension", "suspension_resolution")
def m_unresolved_suspension(doc, rng, params):
    """Prepare a dissonance, hold it over, then leap away instead of stepping down.

    Deliberately works only on notes outside a scheduled entry: rewriting pitches
    inside an entry breaks the subject too, and §17.2 rejects a mutation that
    carries an unintended second violation.
    """
    ids = [v["voice_id"] for v in doc["voices"]]
    by = _notes_by_voice(doc)
    bass_id = ids[-1]
    entry_starts = dict(_entry_spans(params))
    for vid in ids[:-1]:
        notes = by[vid]
        guard = entry_starts.get(vid)
        for i in range(1, len(notes) - 1):
            prev, cur, nxt = notes[i - 1], notes[i], notes[i + 1]
            if guard is not None and parse_frac(prev["onset_q"]) >= guard:
                continue                                    # inside the entry
            bass = [n for n in by[bass_id]
                    if parse_frac(n["onset_q"]) <= parse_frac(cur["onset_q"])]
            if not bass:
                continue
            target = bass[-1]["midi"] + 14                  # a ninth: dissonant
            if not (21 <= target <= 100):
                continue
            _set_midi(prev, target)                         # prepared
            _set_midi(cur, target)                          # held over
            _set_midi(nxt, target + 5)                      # leaps away
            return {"voice_ids": [vid], "onset_q": cur["onset_q"],
                    "detail": "suspension leaps away instead of stepping down"}
    return None


@mutator("alter_required_subject_note", "subject_entry_identity")
def m_alter_subject(doc, rng, params):
    plan = params.get("_entry_plan") or []
    subject_entries = [e for e in plan if e.get("role") != "answer"]
    if not subject_entries:
        return None
    entry = subject_entries[0]
    vid = entry["voice_id"]
    at = parse_frac(str(entry["onset_q"]))
    voice = next((v for v in doc["voices"] if v["voice_id"] == vid), None)
    if voice is None:
        return None
    notes = [n for n in sorted(voice["notes"], key=lambda n: parse_frac(n["onset_q"]))
             if parse_frac(n["onset_q"]) >= at]
    if len(notes) < 2:
        return None
    n = notes[1]
    _set_midi(n, n["midi"] + 1)
    return {"voice_ids": [vid], "onset_q": n["onset_q"],
            "detail": "second note of the subject altered by a semitone"}


@mutator("break_cadence_target", "cadence_target_completion")
def m_break_cadence(doc, rng, params):
    target = params.get("_cadence_target")
    if not target:
        return None
    bass_id = doc["voices"][-1]["voice_id"]
    notes = sorted(doc["voices"][-1]["notes"], key=lambda n: parse_frac(n["onset_q"]))
    if not notes:
        return None
    n = notes[-1]
    _set_midi(n, n["midi"] + 2)             # bass leaves the tonic
    return {"voice_ids": [bass_id], "onset_q": n["onset_q"],
            "detail": "final bass moved off the cadential tonic"}


@mutator("create_invalid_duration", "timing_positive_duration")
def m_invalid_duration(doc, rng, params):
    """Notation-only subset (§17.3): a structurally invalid duration."""
    for v in doc["voices"]:
        notes = sorted(v["notes"], key=lambda n: parse_frac(n["onset_q"]))
        if len(notes) < 2:
            continue
        n = notes[len(notes) // 2]
        n["duration_q"] = "0"
        return {"voice_ids": [v["voice_id"]], "onset_q": n["onset_q"],
                "detail": "duration set to zero"}
    return None


def apply_mutation(doc: dict, mutation_id: str, rng: random.Random,
                   params: dict) -> tuple[dict, dict] | None:
    spec = _REGISTRY[mutation_id]
    mutated = copy.deepcopy(doc)
    info = spec["fn"](mutated, rng, params)
    if info is None:
        return None
    return mutated, {**info, "mutation_id": mutation_id,
                     "target_rule_id": spec["target_rule_id"]}


# --------------------------------------------------------------------------- #
# Additional mutators beyond the §17.3 starter list.
#
# §17.2 requires the result to carry exactly one intended *hard* violation, so a
# mutator is only usable when its target rule is hard in the active profile.
# Under `high_baroque_conservative` three of the §17.3 mutators -- leading-tone
# resolution, melodic leap and voice overlap -- target rules that are soft, by
# deliberate calibration against real Bach (§15.6). They stay in the registry and
# are usable under `species_strict`; these three fill the balance with rules that
# are hard in the conservative profile, so no single rule need exceed §17.4's 15%.
# --------------------------------------------------------------------------- #
@mutator("create_voice_polyphony", "voice_monophonic")
def m_voice_polyphony(doc, rng, params):
    for v in doc["voices"]:
        notes = sorted(v["notes"], key=lambda n: parse_frac(n["onset_q"]))
        if len(notes) < 2:
            continue
        i = len(notes) // 2
        a = notes[i]
        extra = dict(a)
        extra["note_id"] = a["note_id"] + ":dup"
        _set_midi(extra, min(107, a["midi"] + 4))
        v["notes"].append(extra)
        return {"voice_ids": [v["voice_id"]], "onset_q": a["onset_q"],
                "detail": f"second simultaneous pitch added to {v['voice_id']}"}
    return None


@mutator("silence_a_voice", "voice_participation")
def m_silence_voice(doc, rng, params):
    need = params.get("min_voice_participation", 0.2)
    for v in doc["voices"]:
        notes = sorted(v["notes"], key=lambda n: parse_frac(n["onset_q"]))
        if len(notes) < 3:
            continue
        total = parse_frac(doc["quarter_length_total"])
        keep: list[dict] = []
        acc = Fraction(0)
        for n in notes:
            if acc + parse_frac(n["duration_q"]) > total * Fraction(need).limit_denominator(100):
                break
            keep.append(n)
            acc += parse_frac(n["duration_q"])
        if len(keep) < len(notes):
            v["notes"] = keep[:max(1, len(keep) - 1)]
            return {"voice_ids": [v["voice_id"]], "onset_q": "0",
                    "detail": f"{v['voice_id']} reduced below the participation floor"}
    return None


@mutator("alter_answer_transformation", "answer_transformation")
def m_alter_answer(doc, rng, params):
    plan = params.get("_entry_plan") or []
    answers = [e for e in plan if e.get("role") == "answer"]
    if not answers:
        return None
    entry = answers[0]
    vid = entry["voice_id"]
    at = parse_frac(str(entry["onset_q"]))
    voice = next((v for v in doc["voices"] if v["voice_id"] == vid), None)
    if voice is None:
        return None
    notes = [n for n in sorted(voice["notes"], key=lambda n: parse_frac(n["onset_q"]))
             if parse_frac(n["onset_q"]) >= at]
    if len(notes) < 3:
        return None
    n = notes[2]
    _set_midi(n, n["midi"] + 3)
    return {"voice_ids": [vid], "onset_q": n["onset_q"],
            "detail": "answer altered beyond a real or tonal transformation"}
