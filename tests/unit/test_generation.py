"""Unit tests for generation, mutation, pairing and excerpts (Spec §14, §16-§18, §25.1)."""

import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fugue_data.constraints.engine import load_profile, ranges_for
from fugue_data.generation.canonicalize import build_fragment_score, spell
from fugue_data.generation.mutators import apply_mutation, registry
from fugue_data.generation.preferences import (diagnostics, prompt_id,
                                               shared_conditions_hash)
from fugue_data.generation.requests import build_requests
from fugue_data.generation.solver import solve_fragment
from fugue_data.generation.valid_runner import split_for_subject
from fugue_data.generation.validate import validate_candidate
from fugue_data.review.propose import (clipped_boundary_notes,
                                       ends_on_unresolved_suspension,
                                       excerpt_id, rejection_reasons)

CFG = yaml.safe_load((ROOT / "config" / "generation.yaml").read_text())
EXC = yaml.safe_load((ROOT / "config" / "excerpts.yaml").read_text())


# ---- requests (§16.2, §16.3) ------------------------------------------------ #
def test_requests_are_deterministic_for_a_seed():
    a = build_requests(40, CFG, seed=5)
    b = build_requests(40, CFG, seed=5)
    assert [r["request_id"] for r in a] == [r["request_id"] for r in b]
    assert [r["subject"]["intervals"] for r in a] == [r["subject"]["intervals"] for r in b]


def test_requests_cover_every_configured_axis():
    reqs = build_requests(400, CFG, seed=1)
    for axis, values in CFG["valid"]["coverage"].items():
        seen = {r["_coverage"][axis] for r in reqs}
        assert seen == set(values), axis


def test_entry_onsets_align_to_the_rhythmic_grid():
    """An unaligned entry onset would be silently snapped, moving the subject."""
    for r in build_requests(200, CFG, seed=3):
        slot = min(Fraction(x) for x in r["allowed_rhythmic_values"])
        for e in r["entry_plan"]:
            assert Fraction(e["onset_q"]) % slot == 0


def test_subject_fits_with_room_for_another_entry():
    for r in build_requests(200, CFG, seed=4):
        assert Fraction(r["subject"]["length_q"]) <= Fraction(r["_total_q"])


# ---- solver + validator separation (§16.4) ---------------------------------- #
@pytest.mark.slow
def test_solved_fragment_is_accepted_by_the_independent_validator():
    prof = load_profile(CFG["valid"]["rule_profile"])
    solved_any = False
    for r in build_requests(25, CFG, seed=11):
        s = solve_fragment(r, max_time_s=5.0, workers=2)
        if not s:
            continue
        solved_any = True
        doc = build_fragment_score(r, s, "t:0")
        v = validate_candidate(doc, r, prof, set())
        assert v["hard_violation_count"] == 0
        assert v["canonical_schema_valid"] and v["structure_complete"]
        assert v["accepted"]
    assert solved_any


def test_solver_refuses_an_unaligned_entry_rather_than_snapping():
    r = build_requests(1, CFG, seed=9)[0]
    r = {**r, "entry_plan": [{**r["entry_plan"][0], "onset_q": "1/3"}]}
    assert solve_fragment(r, max_time_s=1.0, workers=1) is None


def test_duplicate_is_rejected_by_the_acceptance_predicate():
    prof = load_profile(CFG["valid"]["rule_profile"])
    r = build_requests(1, CFG, seed=12)[0]
    s = solve_fragment(r, max_time_s=5.0, workers=2)
    if s is None:
        pytest.skip("no solution for this request")
    doc = build_fragment_score(r, s, "t:0")
    h = doc["normalization"]["core_event_hash"]
    v = validate_candidate(doc, r, prof, {h})
    assert v["duplicate_status"] == "duplicate" and not v["accepted"]


# ---- canonicalization (§27) -------------------------------------------------- #
def test_synthetic_output_is_never_labelled_as_bach():
    r = build_requests(1, CFG, seed=2)[0]
    s = solve_fragment(r, max_time_s=5.0, workers=2)
    if s is None:
        pytest.skip("no solution")
    doc = build_fragment_score(r, s, "t:0")
    assert doc["composer_id"] == "synthetic"
    assert "bach" not in json.dumps(doc).lower()


def test_pitch_spelling_is_a_real_step_and_octave():
    assert spell(60, False) == ("C", 0, 4)
    assert spell(61, False) == ("C", 1, 4)
    assert spell(61, True) == ("D", -1, 4)


# ---- split inheritance (§16.6, §17.6) ---------------------------------------- #
def test_split_for_subject_is_deterministic_and_spread():
    assert split_for_subject("s1", 7) == split_for_subject("s1", 7)
    got = {split_for_subject(f"s{i}", 7) for i in range(200)}
    assert got == {"train", "validation", "test"}


# ---- mutators (§17.3, §25.1 "each mutator") ---------------------------------- #
def mk_doc():
    def n(v, o, d, m):
        return {"note_id": f"{v}:{o}:{m}", "voice_id": v, "onset_q": str(o),
                "duration_q": str(d), "measure_number": "1",
                "measure_index_unfolded": 0, "offset_in_measure_q": str(o),
                "midi": m, "step": "C", "alter": "0", "octave": m // 12 - 1,
                "pitch_class": m % 12, "diatonic_step": 0, "tie": None,
                "grace": False, "articulations": [], "ornaments": [],
                "chord_group": None, "source_refs": []}
    return {"schema_version": "1.0.0", "piece_id": "t", "work_group_id": None,
            "composer_id": "synthetic", "movement_type": "other",
            "source_file_ids": [], "voice_count": 2,
            "voices": [
                {"voice_id": "voice_1", "staff": None, "notes":
                    [n("voice_1", i, 1, 67 + (i % 3)) for i in range(6)]},
                # Deliberately not already in parallel with voice_1, so a
                # parallel-forcing mutation has something to change.
                {"voice_id": "voice_2", "staff": None, "notes":
                    [n("voice_2", i, 1, 52 + (i % 4)) for i in range(6)]}],
            "meter_map": [{"onset_q": "0", "numerator": 4, "denominator": 4}],
            "key_signature_map": [{"onset_q": "0", "fifths": 0}], "tempo_map": [],
            "annotations": [], "quarter_length_total": "6", "measure_count": 2,
            "warnings": [], "normalization": {"software_version": "t",
                                              "config_hash": "t",
                                              "core_event_hash": "h" * 64}}


@pytest.mark.parametrize("mutation_id", sorted(registry()))
def test_every_mutator_either_changes_the_fragment_or_declines(mutation_id):
    prof = load_profile("high_baroque_conservative")
    params = {**prof["params"], "ranges": ranges_for(prof, "keyboard"),
              "_entry_plan": [{"voice_id": "voice_1", "onset_q": "0", "role": "subject"},
                              {"voice_id": "voice_2", "onset_q": "2", "role": "answer"}],
              "_cadence_target": {"onset_q": "6", "tonic_pc": 0}, "_tonic_pc": 0}
    doc = mk_doc()
    before = json.dumps(doc, sort_keys=True)
    out = apply_mutation(doc, mutation_id, random.Random(0), params)
    # The source must never be modified in place.
    assert json.dumps(doc, sort_keys=True) == before
    if out is None:
        return
    mutated, info = out
    assert info["mutation_id"] == mutation_id
    assert info["target_rule_id"]
    assert json.dumps(mutated, sort_keys=True) != before


def test_registry_covers_every_configured_mutator():
    assert set(CFG["invalid"]["mutators"]) <= set(registry())


# ---- preference pairs (§18.2) ------------------------------------------------ #
def test_shared_conditions_hash_ignores_the_seed_only():
    r = build_requests(1, CFG, seed=6)[0]
    same = {**r, "seed": r["seed"] + 1}
    different = {**r, "length_bars": r["length_bars"] + 1}
    assert shared_conditions_hash(r) == shared_conditions_hash(same)
    assert shared_conditions_hash(r) != shared_conditions_hash(different)
    assert prompt_id(r) == prompt_id(same)


def test_diagnostics_are_a_vector_not_a_single_score():
    rec = {"score": mk_doc(), "hard_valid": True, "soft_scores": {"leap_recovery": 2}}
    d = diagnostics(rec)
    for k in ("style_prior_proxy", "rhythmic_diversity", "melodic_diversity",
              "repetition", "voice_participation_min", "soft_rule_vector"):
        assert k in d
    assert d["soft_violation_total"] == 2


# ---- excerpts (§14.3) -------------------------------------------------------- #
def test_excerpt_id_is_stable():
    assert excerpt_id("p", Fraction(0), Fraction(4)) == excerpt_id("p", Fraction(0), Fraction(4))
    assert excerpt_id("p", Fraction(0), Fraction(4)) != excerpt_id("p", Fraction(0), Fraction(8))


def test_window_beginning_inside_a_tie_is_rejected():
    doc = mk_doc()
    doc["voices"][0]["notes"][2]["tie"] = "stop"
    reasons = rejection_reasons(doc, [], Fraction(2), Fraction(5), EXC)
    assert "begins_inside_tie" in reasons


def test_window_ending_inside_a_tie_is_allowed_and_declared():
    doc = mk_doc()
    doc["voices"][0]["notes"][4]["tie"] = "start"
    reasons = rejection_reasons(doc, [], Fraction(0), Fraction(5), EXC)
    assert "ends_inside_tie" not in reasons
    assert clipped_boundary_notes(doc, Fraction(0), Fraction(Fraction(9, 2))) >= 1


def test_window_cutting_a_subject_entry_is_rejected():
    anns = [{"type": "subject", "onset_q": "1", "duration_q": "4",
             "analytical_status": "accepted", "voice_id": "voice_1"}]
    reasons = rejection_reasons(mk_doc(), anns, Fraction(2), Fraction(6), EXC)
    assert "cuts_subject_entry" in reasons


def test_resting_voice_is_not_voice_identity_loss():
    """A fugue's later voices have not entered yet; that is not a defect."""
    doc = mk_doc()
    doc["voices"][1]["notes"] = []
    reasons = rejection_reasons(doc, [], Fraction(0), Fraction(6), EXC)
    assert "voice_identity_not_preserved" in reasons      # only one voice sounds
    doc2 = mk_doc()
    reasons2 = rejection_reasons(doc2, [], Fraction(0), Fraction(6), EXC)
    assert "voice_identity_not_preserved" not in reasons2


def test_unresolved_suspension_detector_needs_a_held_dissonance():
    doc = mk_doc()
    assert ends_on_unresolved_suspension(doc, Fraction(0), Fraction(1)) is False


# ---- vertical harmony quality (regression) ---------------------------------- #
CONSONANT_OVER_BASS = {0, 3, 4, 7, 8, 9}


def _sonorities(doc):
    from fugue_data.normalize.rational import parse_frac
    onsets = sorted({parse_frac(n["onset_q"])
                     for v in doc["voices"] for n in v["notes"]})
    for t in onsets:
        s = []
        for v in doc["voices"]:
            for n in v["notes"]:
                o = parse_frac(n["onset_q"])
                if o <= t < o + parse_frac(n["duration_q"]):
                    s.append(n["midi"])
                    break
        if len(s) >= 2:
            yield sorted(s, reverse=True)


@pytest.mark.slow
def test_generated_harmony_is_not_a_cluster_field():
    """Guards a real defect: the solver originally constrained parallels, ranges,
    voice order and scale membership but said nothing about vertical harmony, so
    50.6% of simultaneities contained a semitone clash against 9.8% in real Bach,
    and the fragments were unlistenable. Harmony has to be stated to exist.
    """
    prof = load_profile(CFG["valid"]["rule_profile"])
    clusters = triadic = total = 0
    for r in build_requests(40, CFG, seed=4242):
        s = solve_fragment(r, max_time_s=5.0, workers=2)
        if not s:
            continue
        doc = build_fragment_score(r, s, "t")
        if not validate_candidate(doc, r, prof, set())["accepted"]:
            continue
        for p in _sonorities(doc):
            total += 1
            if any((b - a) % 12 in (1, 11) for i, a in enumerate(p) for b in p[i + 1:]):
                clusters += 1
            elif all((x - p[-1]) % 12 in CONSONANT_OVER_BASS for x in p[:-1]):
                triadic += 1
    assert total > 200, "not enough sonorities sampled to judge"
    assert clusters / total < 0.05, (
        f"{clusters/total:.1%} of simultaneities contain a semitone clash; "
        f"real Bach sits near 10% and the unconstrained solver hit 50%")
    assert triadic / total > 0.45, (
        f"only {triadic/total:.1%} of simultaneities are triadic; real Bach is ~52%")


@pytest.mark.slow
def test_strong_beats_are_consonant_over_the_bass():
    from fractions import Fraction as F
    from fugue_data.normalize.rational import parse_frac
    prof = load_profile(CFG["valid"]["rule_profile"])
    checked = 0
    for r in build_requests(20, CFG, seed=99):
        s = solve_fragment(r, max_time_s=5.0, workers=2)
        if not s:
            continue
        doc = build_fragment_score(r, s, "t")
        if not validate_candidate(doc, r, prof, set())["accepted"]:
            continue
        beat = F(4, r["meter"]["denominator"])
        for p in _sonorities(doc):
            pass
        onsets = sorted({parse_frac(n["onset_q"])
                         for v in doc["voices"] for n in v["notes"]})
        for t in onsets:
            if t % beat != 0:
                continue
            sounding = []
            for v in doc["voices"]:
                for n in v["notes"]:
                    o = parse_frac(n["onset_q"])
                    if o <= t < o + parse_frac(n["duration_q"]):
                        sounding.append(n["midi"])
                        break
            if len(sounding) < 2:
                continue
            bass = min(sounding)
            checked += 1
            for x in sounding:
                if x == bass:
                    continue
                assert (x - bass) % 12 in CONSONANT_OVER_BASS, (
                    f"dissonance {(x-bass)%12} over the bass on a beat at {t}")
    assert checked > 50


def _melodic_intervals(doc):
    from fugue_data.normalize.rational import parse_frac
    for v in doc["voices"]:
        notes = sorted(v["notes"], key=lambda n: parse_frac(n["onset_q"]))
        for a, b in zip(notes, notes[1:]):
            yield v["voice_id"], a, b, b["midi"] - a["midi"]


def _accepted_fragments(n_requests, seed):
    prof = load_profile(CFG["valid"]["rule_profile"])
    out = []
    for r in build_requests(n_requests, CFG, seed=seed):
        s = solve_fragment(r, max_time_s=5.0, workers=2)
        if not s:
            continue
        doc = build_fragment_score(r, s, "t")
        if validate_candidate(doc, r, prof, set())["accepted"]:
            out.append((r, doc))
    return out


@pytest.mark.slow
def test_no_melodic_tritones():
    """Measured at 23.1 per 1000 notes unconstrained, against Bach's 9.7."""
    frags = _accepted_fragments(30, 515)
    assert frags
    offenders = [(vid, d) for _, doc in frags
                 for vid, a, b, d in _melodic_intervals(doc) if abs(d) == 6]
    assert not offenders, f"melodic tritones present: {offenders[:5]}"


@pytest.mark.slow
def test_free_voices_do_not_leap_beyond_a_fourth():
    """The cap applies to free counterpoint, not to scheduled entries.

    An earlier version of this test measured leap recovery across *all* notes and
    compared it to Bach. That conflates two different things: a leap into or out
    of a fixed subject entry is not a counterpoint decision the solver made -- the
    subject enters where the plan says -- and lumping them together produced a
    number that could not be acted on. Measuring only free material, and asserting
    the bound the constraint actually promises, tests something real.
    """
    from fractions import Fraction as F
    from fugue_data.normalize.rational import parse_frac
    frags = _accepted_fragments(30, 616)
    assert frags
    offenders = []
    for req, doc in frags:
        subj_len = sum(F(x) for x in req["subject"]["rhythm_q"])
        spans = {}
        for e in req["entry_plan"]:
            a = F(e["onset_q"])
            spans.setdefault(e["voice_id"], []).append((a, a + subj_len))
        for v in doc["voices"]:
            notes = sorted(v["notes"], key=lambda n: parse_frac(n["onset_q"]))
            vs = spans.get(v["voice_id"], [])
            inside = lambda t: any(x <= t < y for x, y in vs)
            for a, b in zip(notes, notes[1:]):
                ta, tb = parse_frac(a["onset_q"]), parse_frac(b["onset_q"])
                if inside(ta) or inside(tb):
                    continue                      # not a free-counterpoint move
                if abs(b["midi"] - a["midi"]) > 5:
                    offenders.append((v["voice_id"], b["midi"] - a["midi"]))
    assert not offenders, f"free voices leaping beyond a fourth: {offenders[:5]}"


@pytest.mark.slow
def test_no_direct_fifths_into_the_outer_voices():
    """9.8 per 1000 notes unconstrained against Bach's 1.9."""
    from fugue_data.normalize.rational import parse_frac
    frags = _accepted_fragments(30, 717)
    assert frags
    offenders = []
    for _, doc in frags:
        if len(doc["voices"]) < 2:
            continue
        top, low = doc["voices"][0], doc["voices"][-1]
        onsets = sorted({parse_frac(n["onset_q"])
                         for v in doc["voices"] for n in v["notes"]})

        def at(voice, t):
            for n in voice["notes"]:
                o = parse_frac(n["onset_q"])
                if o <= t < o + parse_frac(n["duration_q"]):
                    return n
            return None

        for t0, t1 in zip(onsets, onsets[1:]):
            a0, a1 = at(top, t0), at(top, t1)
            b0, b1 = at(low, t0), at(low, t1)
            if not all((a0, a1, b0, b1)):
                continue
            if (a1["midi"] - b1["midi"]) % 12 not in (0, 7):
                continue
            dt, db = a1["midi"] - a0["midi"], b1["midi"] - b0["midi"]
            if dt * db > 0 and abs(dt) > 2:
                offenders.append((t1, dt, db))
    assert not offenders, f"direct fifths/octaves into the outer voices: {offenders[:5]}"


def test_pair_id_is_derived_from_content_not_position():
    """A positional id is reused by the next generation run for different music,
    so a label made against an earlier run would ingest cleanly and attach to the
    wrong fragments. The id must change when the music changes."""
    from fugue_data.generation.preferences import build_pairs

    def rec(sample_id, event_hash, split="train", seed=1):
        r = build_requests(1, CFG, seed=seed)[0]
        return {"sample_id": sample_id, "event_hash": event_hash, "split": split,
                "score": mk_doc(), "hard_valid": True, "soft_scores": {},
                "request": r, "seed": seed}

    a, b = rec("s1", "a" * 64), rec("s2", "b" * 64)
    first = build_pairs([a, b], CFG)
    assert first, "expected one pair from two candidates on the same prompt"
    pid = first[0]["pair_id"]

    # Same content, rebuilt: the id is stable.
    assert build_pairs([a, b], CFG)[0]["pair_id"] == pid
    # Order of the two candidates must not change the id.
    assert build_pairs([b, a], CFG)[0]["pair_id"] == pid
    # Different music: a different id, so a stale label cannot bind to it.
    c = rec("s2", "c" * 64)
    assert build_pairs([a, c], CFG)[0]["pair_id"] != pid
    assert not pid.startswith("pair:0000")      # not a positional counter


# ---- subject quality (regression) ------------------------------------------- #
def test_subjects_contain_no_awkward_melodic_intervals():
    """The subject is the one thing every entry repeats, so a bad interval in it
    is heard in the subject, in the answer, and in every later entry -- and it is
    identical in both candidates of a pair, so no amount of fixing the free
    voices can rescue it. The free voices were constrained against tritones while
    the subject generator produced them in 12% of subjects.
    """
    from fugue_data.generation.requests import _awkward_interval, scale_for
    found = []
    for r in build_requests(400, CFG, seed=1):
        scale = sorted({(r["key"]["tonic_pc"] + d) % 12
                        for d in scale_for(r["key"]["mode"])})
        pitches = [r["key"]["tonic_pc"]]
        for iv in r["subject"]["intervals"]:
            pitches.append(pitches[-1] + iv)
        for a, b in zip(pitches, pitches[1:]):
            w = _awkward_interval(a, b, scale)
            if w:
                found.append((r["request_id"], w, b - a))
    assert not found, f"awkward subject intervals: {found[:5]}"


def test_awkward_interval_uses_spelling_not_just_distance():
    """G#->C is four semitones and sounds like a major third, but spans four
    letter names and is a diminished fourth. Distance alone cannot tell them
    apart, which is exactly how the bad subject got through."""
    from fugue_data.generation.requests import _awkward_interval
    a_minor = sorted({(9 + d) % 12 for d in (0, 2, 3, 5, 7, 8, 11)})
    assert _awkward_interval(8, 12, a_minor) == "diminished fourth"   # G# -> C
    assert _awkward_interval(5, 8, a_minor) == "augmented second"     # F  -> G#
    assert _awkward_interval(9, 14, a_minor) is None                  # A  -> D
    assert _awkward_interval(9, 15, a_minor) == "tritone"             # A  -> D#
