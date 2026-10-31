"""Unit tests for annotation alignment (Spec §11, §25.1)."""

import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fugue_data.annotations.align import (AlignmentError, align_dez,
                                          assign_alternative_groups, locate,
                                          measure_table, voice_note_coverage)
from fugue_data.annotations.dezrann import (load_config, map_label,
                                            parse_dez_filename, _tag_words)
from fugue_data.annotations.entries import classify_entries
from fugue_data.validation.annotations import validate_annotations


CFG = load_config()


# --------------------------------------------------------------------------- #
# Fixtures: a tiny 2-voice score and a matching .dez analysis.
# --------------------------------------------------------------------------- #
def mk_note(voice, onset, dur, midi, measure, offset, diatonic=36, step="C", octave=5):
    return {
        "note_id": f"{voice}:{onset}", "voice_id": voice,
        "onset_q": str(onset), "duration_q": str(dur),
        "measure_number": str(measure), "measure_index_unfolded": measure - 1,
        "offset_in_measure_q": str(offset), "midi": midi, "step": step,
        "alter": "0", "octave": octave, "pitch_class": midi % 12,
        "diatonic_step": diatonic, "tie": None, "grace": False,
        "articulations": [], "ornaments": [], "chord_group": None, "source_refs": [],
    }


def mk_score(piece_id="bach_wtc1_fugue_02", work="BWV847"):
    # Two voices, 2 bars of 4/4. voice_1 enters in bar 2, voice_2 in bar 1.
    v2 = [mk_note("voice_2", i, 1, 60 + i, 1, i, 36 + i) for i in range(4)]
    v1 = [mk_note("voice_1", 4 + i, 1, 67 + i, 2, i, 40 + i) for i in range(4)]
    return {
        "schema_version": "1.0.0", "piece_id": piece_id, "work_group_id": work,
        "composer_id": "bach_js", "movement_type": "fugue", "source_file_ids": [],
        "voice_count": 2, "staff_count": 2,
        "voices": [{"voice_id": "voice_1", "staff": 1, "mean_midi": 68.5, "notes": v1},
                   {"voice_id": "voice_2", "staff": 2, "mean_midi": 61.5, "notes": v2}],
        "meter_map": [{"onset_q": "0", "numerator": 4, "denominator": 4}],
        "key_signature_map": [{"onset_q": "0", "fifths": 0}], "tempo_map": [],
        "annotations": [], "quarter_length_total": "8", "measure_count": 2,
        "warnings": [],
        "normalization": {"software_version": "t", "config_hash": "t", "event_hash": "t"},
    }


class FakeDez:
    def __init__(self, labels, piece_id="bach_wtc1_fugue_02", work="BWV847", kind="ref"):
        self.labels = labels
        self.piece_id = piece_id
        self.work_group_id = work
        self.annotator_kind = kind
        self.path = Path("02-bwv847-ref.dez")
        self.fugue_index = 2


# ---- source filename identity --------------------------------------------- #
def test_parse_dez_filename():
    assert parse_dez_filename("02-bwv847-ref.dez") == (2, "BWV847", "ref")
    assert parse_dez_filename("24-bwv869-computed.dez") == (24, "BWV869", "computed")
    assert parse_dez_filename("synchro-01.json") is None


# ---- vocabulary mapping (§11.2) ------------------------------------------- #
@pytest.mark.parametrize("raw_type,expected", [
    ("S", "subject"), ("S-inv", "subject"), ("S2", "subject"),
    ("CS1", "countersubject"), ("CS2-inc", "countersubject"),
    ("Cadence", "cadence"), ("Pedal", "pedal"),
])
def test_map_label_types(raw_type, expected):
    assert map_label({"type": raw_type, "tag": raw_type}, CFG).type == expected


def test_map_structure_disambiguated_by_tag():
    assert map_label({"type": "Structure", "tag": "episode"}, CFG).type == "episode"
    assert map_label({"type": "Structure", "tag": "codetta"}, CFG).type == "other"


def test_map_transformations_from_type_and_tag():
    m = map_label({"type": "S-invinc", "tag": "S-invinc varied"}, CFG)
    assert m.type == "subject"
    assert m.transformations == ["incomplete", "inverted", "varied"]


def test_ignore_becomes_rejected_candidate_keeping_family():
    m = map_label({"type": "ignore", "tag": "S-inc-IGN"}, CFG)
    assert m.analytical_status == "rejected_candidate"
    assert m.type == "subject"          # the near-miss stays interpretable
    assert "incomplete" in m.transformations


def test_cadence_tag_parsed_into_key_and_type():
    m = map_label({"type": "Cadence", "tag": " v:PAC"}, CFG)
    assert (m.cadence_key_area, m.cadence_type) == ("v", "PAC")
    assert m.confidence is None


def test_uncertain_cadence_lowers_confidence():
    m = map_label({"type": "cad:PAC ?", "tag": " I:PAC ?"}, CFG)
    assert m.type == "cadence" and m.confidence == 0.5


def test_unknown_vocabulary_is_flagged_not_guessed():
    m = map_label({"type": "seqnext", "tag": "2"}, CFG)
    assert m.unmapped is True and m.type == "other"


def test_tag_words_strips_restated_name_and_ign():
    assert _tag_words("CS1 varied-end") == ["varied-end"]
    assert _tag_words("S-var augmented+rhythmicvar-IGN") == ["augmented+rhythmicvar"]


# ---- onset -> measure conversion (§25.1) ---------------------------------- #
def test_measure_table_and_locate():
    table = measure_table(mk_score())
    assert [(m.index, m.number, m.start_q) for m in table] == [
        (0, "1", Fraction(0)), (1, "2", Fraction(4))]
    assert locate(Fraction(0), table) == ("1", Fraction(0))
    assert locate(Fraction(9, 2), table) == ("2", Fraction(1, 2))


def test_locate_before_first_measure_returns_none():
    assert locate(Fraction(-1), measure_table(mk_score())) == (None, None)


# ---- voice coverage -------------------------------------------------------- #
def test_voice_note_coverage_full_and_empty():
    score = mk_score()
    v1, v2 = score["voices"]
    assert voice_note_coverage(v2, Fraction(0), Fraction(4)) == 1.0
    assert voice_note_coverage(v1, Fraction(0), Fraction(4)) == 0.0
    assert voice_note_coverage(v2, Fraction(0), Fraction(0)) is None


def test_voice_note_coverage_merges_overlaps_not_sums():
    # Two simultaneous notes (divisi) must not report 200% coverage.
    voice = {"voice_id": "voice_1", "notes": [
        mk_note("voice_1", 0, 2, 60, 1, 0), mk_note("voice_1", 0, 2, 64, 1, 0)]}
    assert voice_note_coverage(voice, Fraction(0), Fraction(2)) == 1.0


# ---- alignment (§11.1, §11.4) --------------------------------------------- #
def test_align_maps_staff_to_voice_top_down():
    dez = FakeDez([{"type": "S", "tag": "S", "staff": "2", "start": 0, "duration": 4},
                   {"type": "S", "tag": "S", "staff": "1", "start": 4, "duration": 4}])
    anns, warns = align_dez(dez, mk_score(), CFG)
    assert [a["voice_id"] for a in anns] == ["voice_2", "voice_1"]
    assert all(a["voice_note_coverage"] == 1.0 for a in anns)
    assert warns == []


def test_align_rejects_wrong_piece():
    with pytest.raises(AlignmentError):
        align_dez(FakeDez([], piece_id="bach_wtc1_fugue_09"), mk_score(), CFG)


def test_align_rejects_work_id_mismatch_not_just_filename():
    # §11.1: never align on filename alone -- the BWV must agree too.
    with pytest.raises(AlignmentError):
        align_dez(FakeDez([], work="BWV999"), mk_score(), CFG)


def test_align_drops_out_of_bounds_onset_with_warning():
    dez = FakeDez([{"type": "S", "tag": "S", "staff": "1", "start": 99, "duration": 1}])
    anns, warns = align_dez(dez, mk_score(), CFG)
    assert anns == []
    assert warns[0]["code"] == "annotation_onset_out_of_bounds"


def test_align_flags_unresolvable_staff():
    dez = FakeDez([{"type": "S", "tag": "S", "staff": "5", "start": 0, "duration": 4}])
    anns, warns = align_dez(dez, mk_score(), CFG)
    assert anns == []
    assert warns[0]["code"] == "annotation_voice_unresolved"


def test_align_flags_low_voice_coverage():
    # A label on the voice that is silent there is the signature of a bad mapping.
    dez = FakeDez([{"type": "S", "tag": "S", "staff": "1", "start": 0, "duration": 4}])
    anns, warns = align_dez(dez, mk_score(), CFG)
    assert anns[0]["voice_note_coverage"] == 0.0
    assert warns[0]["code"] == "annotation_low_voice_coverage"


def test_cadence_zero_duration_becomes_point_event_not_dropped():
    dez = FakeDez([{"type": "cad:PAC ?", "tag": " I:PAC ?", "start": 4, "duration": 0}])
    anns, warns = align_dez(dez, mk_score(), CFG)
    assert len(anns) == 1 and anns[0]["duration_q"] is None
    assert warns[0]["code"] == "annotation_zero_duration_point_event"
    assert anns[0]["raw"]["duration"] == 0      # source preserved verbatim


def test_span_overrunning_piece_is_clipped_with_warning():
    dez = FakeDez([{"type": "S", "tag": "S", "staff": "2", "start": 4, "duration": 99}])
    anns, warns = align_dez(dez, mk_score(), CFG)
    assert anns[0]["duration_q"] == "4"
    assert warns[0]["code"] == "annotation_span_clipped"


def test_align_preserves_raw_label_verbatim():
    raw = {"type": "S-inv", "tag": "S-inv varied", "staff": "2", "start": 0, "duration": 4}
    anns, _ = align_dez(FakeDez([dict(raw)]), mk_score(), CFG)
    assert anns[0]["raw"]["type"] == "S-inv"
    assert anns[0]["raw"]["tag"] == "S-inv varied"


def test_annotation_ids_are_stable_across_label_order():
    a = [{"type": "S", "tag": "S", "staff": "2", "start": 0, "duration": 4},
         {"type": "S", "tag": "S", "staff": "1", "start": 4, "duration": 4}]
    first, _ = align_dez(FakeDez(list(a)), mk_score(), CFG)
    second, _ = align_dez(FakeDez(list(reversed(a))), mk_score(), CFG)
    assert [x["annotation_id"] for x in first] == [x["annotation_id"] for x in second]
    assert [x["onset_q"] for x in first] == [x["onset_q"] for x in second]


# ---- analytical ambiguity (§11.3) ----------------------------------------- #
def test_alternative_group_links_accepted_and_rejected_overlap():
    dez = FakeDez([
        {"type": "S", "tag": "S", "staff": "2", "start": 0, "duration": 4},
        {"type": "ignore", "tag": "S-inc-IGN", "staff": "2", "start": 2, "duration": 2},
    ])
    anns, _ = align_dez(dez, mk_score(), CFG)
    assert assign_alternative_groups(anns, "p") == 1
    assert len({a["analytical_alternative_group"] for a in anns}) == 1
    # Neither side is discarded.
    assert {a["analytical_status"] for a in anns} == {"accepted", "rejected_candidate"}


def test_no_alternative_group_without_disagreement():
    dez = FakeDez([
        {"type": "S", "tag": "S", "staff": "2", "start": 0, "duration": 4},
        {"type": "S", "tag": "S", "staff": "2", "start": 2, "duration": 2},
    ])
    anns, _ = align_dez(dez, mk_score(), CFG)
    assert assign_alternative_groups(anns, "p") == 0


# ---- derived subject / answer --------------------------------------------- #
def test_entry_function_is_derived_and_labelled_as_such():
    # voice_2 head diatonic_step 36; voice_1 head 40 -> +4 = a fifth up = answer.
    dez = FakeDez([{"type": "S", "tag": "S", "staff": "2", "start": 0, "duration": 4},
                   {"type": "S", "tag": "S", "staff": "1", "start": 4, "duration": 4}])
    score = mk_score()
    anns, _ = align_dez(dez, score, CFG)
    summary = classify_entries(anns, score, CFG)
    assert (anns[0]["entry_function"], anns[1]["entry_function"]) == ("subject", "answer")
    assert summary["subject"] == 1 and summary["answer"] == 1
    # §27: inference must never masquerade as source data.
    for a in anns:
        assert a["entry_function_derivation"].startswith("computed:")
        assert a["type"] == "subject"          # source-given type is untouched
        assert a["raw"]["type"] == "S"


def test_entry_function_undetermined_at_other_levels():
    score = mk_score()
    score["voices"][0]["notes"] = [mk_note("voice_1", 4 + i, 1, 64 + i, 2, i, 38 + i)
                                   for i in range(4)]   # +2 = a third, neither level
    dez = FakeDez([{"type": "S", "tag": "S", "staff": "2", "start": 0, "duration": 4},
                   {"type": "S", "tag": "S", "staff": "1", "start": 4, "duration": 4}])
    anns, _ = align_dez(dez, score, CFG)
    classify_entries(anns, score, CFG)
    assert anns[1]["entry_function"] is None
    assert anns[1]["entry_head_diatonic_interval"] == 2


# ---- validation gates (§11.4) --------------------------------------------- #
def mk_doc(annotations, piece_id="bach_wtc1_fugue_02"):
    return {
        "schema_version": "1.0.0", "piece_id": piece_id, "work_group_id": "BWV847",
        "source_name": "algomus", "annotation_map_version": "1.0.0",
        "source_file_ids": [], "annotations": annotations,
        "alignment": {"voice_mapping_strategy": "staff_index_top_down",
                      "piece_quarter_length": "8", "voice_count": 2},
        "warnings": [],
    }


def base_ann(**kw):
    a = {"annotation_id": "a1", "piece_id": "bach_wtc1_fugue_02", "type": "subject",
         "voice_id": "voice_1", "onset_q": "4", "duration_q": "4",
         "source_name": "algomus", "source_reference": "f.dez#0",
         "annotator": "reference_human", "analytical_status": "accepted",
         "raw": {"type": "S", "tag": "S"}}
    a.update(kw)
    return a


def test_valid_document_passes():
    assert validate_annotations(mk_doc([base_ann()]), mk_score()) == []


def test_gate_onset_outside_score():
    errs = validate_annotations(mk_doc([base_ann(onset_q="99")]), mk_score())
    assert any("onset out of bounds" in e for e in errs)


def test_gate_unresolved_voice():
    errs = validate_annotations(mk_doc([base_ann(voice_id="voice_9")]), mk_score())
    assert any("unresolved voice" in e for e in errs)


def test_gate_non_positive_duration():
    errs = validate_annotations(mk_doc([base_ann(duration_q="0")]), mk_score())
    assert any("non-positive duration" in e for e in errs)


def test_gate_derived_function_must_declare_derivation():
    errs = validate_annotations(
        mk_doc([base_ann(entry_function="answer")]), mk_score())
    assert any("without derivation note" in e for e in errs)


def test_gate_singleton_alternative_group():
    errs = validate_annotations(
        mk_doc([base_ann(analytical_alternative_group="g1")]), mk_score())
    assert any("single member" in e for e in errs)


def test_gate_piece_id_mismatch():
    errs = validate_annotations(mk_doc([], piece_id="bach_wtc1_fugue_09"), mk_score())
    assert any("piece_id mismatch" in e for e in errs)


def test_point_event_duration_null_is_valid():
    assert validate_annotations(
        mk_doc([base_ann(type="cadence", voice_id=None, duration_q=None)]), mk_score()) == []
