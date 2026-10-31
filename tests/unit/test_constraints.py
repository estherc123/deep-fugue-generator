"""Unit tests for the hard-constraint framework (Spec §15, §25.1)."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fugue_data.constraints.engine import (PROFILES, evaluate, hard_violations,
                                           load_profile, ranges_for, ruleset_hash,
                                           validate_registry)
from fugue_data.constraints.fragment import Fragment
from fugue_data.constraints.rules import classify_dissonance, registry


def mk_note(voice, onset, dur, midi, tie=None, nid=None):
    return {"note_id": nid or f"{voice}:{onset}:{midi}", "voice_id": voice,
            "onset_q": str(onset), "duration_q": str(dur), "measure_number": "1",
            "measure_index_unfolded": 0, "offset_in_measure_q": str(onset),
            "midi": midi, "step": "C", "alter": "0", "octave": midi // 12 - 1,
            "pitch_class": midi % 12, "diatonic_step": 0, "tie": tie,
            "grace": False, "articulations": [], "ornaments": [],
            "chord_group": None, "source_refs": []}


def frag(voices, total=4, plan=None, **kw):
    f = Fragment(piece_id="t",
                 voices=[{"voice_id": f"voice_{i+1}", "staff": None, "notes": ns}
                         for i, ns in enumerate(voices)],
                 quarter_length_total=Fraction(total),
                 plan=plan or {})
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def fire(f, rule_id, profile="species_strict"):
    vs = evaluate(f, load_profile(profile), only={rule_id})
    return [v for v in vs if v["rule_id"] == rule_id]


# ---- registry and profiles (§15.3, §3.4) ----------------------------------- #
def test_registry_validates_and_every_rule_cites_a_source():
    assert validate_registry(ROOT) == []


def test_all_three_profiles_exist_with_every_rule():
    ids = set(load_profile("species_strict")["rules"])
    for p in PROFILES:
        assert set(load_profile(p)["severities"]) == ids


def test_ruleset_hash_differs_per_profile_and_is_stable():
    hashes = {p: ruleset_hash(load_profile(p)) for p in PROFILES}
    assert len(set(hashes.values())) == 3
    assert ruleset_hash(load_profile("bach_empirical")) == hashes["bach_empirical"]


def test_off_severity_is_a_string_not_a_boolean():
    """Bare `off` in YAML parses as False and would silently skip the lookup."""
    for p in PROFILES:
        for v in load_profile(p)["severities"].values():
            assert isinstance(v, str)


def test_range_profile_selection():
    prof = load_profile("bach_empirical")
    assert ranges_for(prof, "satb")["voice_1"] != ranges_for(prof, "keyboard")["voice_1"]


# ---- notational invariants -------------------------------------------------- #
def test_negative_onset_detected():
    assert fire(frag([[mk_note("voice_1", -1, 1, 60)]]), "timing_nonnegative_onset")


def test_zero_duration_detected():
    assert fire(frag([[mk_note("voice_1", 0, 0, 60)]]), "timing_positive_duration")


def test_note_past_end_detected():
    assert fire(frag([[mk_note("voice_1", 0, 99, 60)]], total=4), "timing_within_bounds")


def test_monophonic_overlap_detected():
    f = frag([[mk_note("voice_1", 0, 2, 60), mk_note("voice_1", 1, 2, 64)]])
    assert fire(f, "voice_monophonic")


# ---- parallels (§25.1 perfect-parallel detection) --------------------------- #
def test_parallel_fifths_detected():
    f = frag([[mk_note("voice_1", 0, 1, 67), mk_note("voice_1", 1, 1, 69)],
              [mk_note("voice_2", 0, 1, 60), mk_note("voice_2", 1, 1, 62)]])
    assert fire(f, "parallel_fifths")


def test_parallel_octaves_detected():
    f = frag([[mk_note("voice_1", 0, 1, 72), mk_note("voice_1", 1, 1, 74)],
              [mk_note("voice_2", 0, 1, 60), mk_note("voice_2", 1, 1, 62)]])
    assert fire(f, "parallel_octaves")


def test_contrary_motion_into_a_fifth_is_not_parallel():
    """A fifth reached by contrary motion is not the prohibited figure.

    In semitones the interval can only be held across contrary motion by landing
    on the compound form -- a fifth (7) opening out to a twelfth (19), which is
    still 7 mod 12 and so still looks parallel to an interval-class test.
    """
    f = frag([[mk_note("voice_1", 0, 1, 67), mk_note("voice_1", 1, 1, 73)],
              [mk_note("voice_2", 0, 1, 60), mk_note("voice_2", 1, 1, 54)]])
    assert not fire(f, "parallel_fifths")


def test_fifths_across_a_rest_are_not_parallel():
    """Real Bach reported parallel fifths across a phrase break until the
    detector required the two simultaneities to actually be consecutive."""
    f = frag([[mk_note("voice_1", 0, 1, 67), mk_note("voice_1", 3, 1, 69)],
              [mk_note("voice_2", 0, 1, 60), mk_note("voice_2", 3, 1, 62)]], total=8)
    assert not fire(f, "parallel_fifths")


def test_offbeat_parallels_excluded_only_where_the_profile_says_so():
    """BWV 848 m39 has parallel fifths between two ornamental offbeat sixteenths."""
    f = frag([[mk_note("voice_1", Fraction(1, 4), Fraction(1, 4), 67),
               mk_note("voice_1", Fraction(1, 2), Fraction(1, 4), 69)],
              [mk_note("voice_2", Fraction(1, 4), Fraction(1, 4), 60),
               mk_note("voice_2", Fraction(1, 2), Fraction(1, 4), 62)]])
    assert not fire(f, "parallel_fifths", "high_baroque_conservative")
    assert fire(f, "parallel_fifths", "species_strict")


# ---- dissonance classification ---------------------------------------------- #
def test_classify_passing_and_neighbour():
    # 60 - 62 - 64: the middle note is approached and left by step in one
    # direction, so it is a passing note against a C bass.
    f = frag([[mk_note("voice_1", 0, 1, 60), mk_note("voice_1", 1, 1, 62),
               mk_note("voice_1", 2, 1, 64)],
              [mk_note("voice_2", 0, 3, 48)]])
    v = f.voices[0]["notes"]
    assert classify_dissonance(f, "voice_1", v[1], 48) == "passing"
    # 60 - 62 - 60: step away and back is a neighbour note.
    g = frag([[mk_note("voice_1", 0, 1, 60), mk_note("voice_1", 1, 1, 62),
               mk_note("voice_1", 2, 1, 60)],
              [mk_note("voice_2", 0, 3, 48)]])
    assert classify_dissonance(g, "voice_1", g.voices[0]["notes"][1], 48) == "neighbour"


def test_a_voices_first_note_has_no_preparation_to_judge():
    f = frag([[mk_note("voice_1", 0, 1, 62)], [mk_note("voice_2", 0, 1, 60)]])
    assert classify_dissonance(f, "voice_1", f.voices[0]["notes"][0], 60) == "unprepared"


def test_unresolved_suspension_is_classified_as_a_suspension_not_unprepared():
    """Folding the resolution test into the classifier made the one rule meant to
    catch an unresolved suspension unable to fire at all."""
    f = frag([[mk_note("voice_1", 0, 1, 62), mk_note("voice_1", 1, 1, 62),
               mk_note("voice_1", 2, 1, 67)],
              [mk_note("voice_2", 0, 3, 60)]])
    v = f.voices[0]["notes"]
    assert classify_dissonance(f, "voice_1", v[1], 60) == "suspension"


def test_suspension_must_be_prepared_as_a_consonance():
    f = frag([[mk_note("voice_1", 0, 1, 61), mk_note("voice_1", 1, 1, 61),
               mk_note("voice_1", 2, 1, 66)],
              [mk_note("voice_2", 0, 3, 60)]])
    # Never consonant against the bass, so not a suspension to resolve.
    assert not fire(f, "suspension_resolution")


# ---- structural rules stay silent without a plan ---------------------------- #
@pytest.mark.parametrize("rule_id", ["subject_entry_identity", "answer_transformation",
                                     "cadence_target_completion",
                                     "structural_target_reachable",
                                     "section_length_limit"])
def test_structural_rules_silent_on_real_music_without_a_plan(rule_id):
    f = frag([[mk_note("voice_1", 0, 1, 60)], [mk_note("voice_2", 0, 1, 48)]])
    assert fire(f, rule_id) == []


def test_subject_entry_identity_detects_a_wrong_interval():
    plan = {"subject": {"intervals": [2, 2], "length_q": 3},
            "entry_plan": [{"voice_id": "voice_1", "onset_q": "0", "role": "subject"}]}
    f = frag([[mk_note("voice_1", 0, 1, 60), mk_note("voice_1", 1, 1, 62),
               mk_note("voice_1", 2, 1, 67)]], plan=plan)
    assert fire(f, "subject_entry_identity")


def test_answers_are_judged_by_answer_transformation_not_entry_identity():
    plan = {"subject": {"intervals": [2, 2], "length_q": 3},
            "entry_plan": [{"voice_id": "voice_1", "onset_q": "0", "role": "answer"}]}
    f = frag([[mk_note("voice_1", 0, 1, 67), mk_note("voice_1", 1, 1, 69),
               mk_note("voice_1", 2, 1, 71)]], plan=plan)
    assert fire(f, "subject_entry_identity") == []
    assert fire(f, "answer_transformation") == []      # a real answer


def test_unreachable_structural_target_detected():
    plan = {"subject": {"intervals": [2], "length_q": 8},
            "entry_plan": [{"voice_id": "voice_1", "onset_q": "3", "role": "subject"}]}
    f = frag([[mk_note("voice_1", 3, 1, 60)]], total=4, plan=plan)
    assert fire(f, "structural_target_reachable")


# ---- context conditions §15.5 warns about ------------------------------------ #
def test_dead_air_stands_down_for_an_extract():
    """BCFB chorales are cantata movements with the instruments excluded, so the
    vocal parts legitimately fall silent."""
    notes = [[mk_note("voice_1", 40, 1, 60)]]
    assert fire(frag(notes, total=48, is_extract=False), "max_dead_air")
    assert fire(frag(notes, total=48, is_extract=True), "max_dead_air") == []


def test_voice_crossing_is_never_hard():
    for p in PROFILES:
        assert load_profile(p)["severities"]["voice_crossing"] != "hard"


def test_hard_violations_filter():
    vs = [{"severity": "hard"}, {"severity": "soft"}, {"severity": "diagnostic"}]
    assert len(hard_violations(vs)) == 1
