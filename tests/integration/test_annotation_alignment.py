"""Integration: one WTC I fugue end-to-end through annotation alignment (§25.2).

Runs against the real pinned Algomus checkout and the real canonical scores, so
it is skipped when the corpus has not been acquired/normalized yet.
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fugue_data.annotations.align import align_dez, assign_alternative_groups
from fugue_data.annotations.dezrann import discover_dez_files, load_config
from fugue_data.annotations.entries import classify_entries
from fugue_data.validation.annotations import validate_annotations

PIECE = "bach_wtc1_fugue_02"          # BWV 847, C minor -- the textbook 3-voice fugue

pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "canonical" / f"{PIECE}.json").exists(),
    reason="corpus not acquired/normalized in this checkout",
)


@pytest.fixture(scope="module")
def aligned():
    cfg = load_config(ROOT)
    score = json.loads((ROOT / "data" / "canonical" / f"{PIECE}.json").read_text())
    dez = next(f for f in discover_dez_files(ROOT) if f.piece_id == PIECE)
    anns, warns = align_dez(dez, score, cfg)
    assign_alternative_groups(anns, PIECE)
    summary = classify_entries(anns, score, cfg)
    return score, anns, warns, summary


def test_alignment_is_clean(aligned):
    score, anns, warns, _ = aligned
    assert warns == []
    assert len(anns) == 30


def test_passes_validation_gates(aligned):
    score, anns, _, _ = aligned
    doc = {
        "schema_version": "1.0.0", "piece_id": PIECE,
        "work_group_id": score["work_group_id"], "source_name": "algomus",
        "annotation_map_version": load_config(ROOT)["annotation_map_version"],
        "source_file_ids": [], "annotations": anns,
        "alignment": {"voice_mapping_strategy": "staff_index_top_down",
                      "piece_quarter_length": score["quarter_length_total"],
                      "voice_count": score["voice_count"]},
        "warnings": [],
    }
    assert validate_annotations(doc, score) == []


def test_exposition_matches_the_published_analysis(aligned):
    """Subject (alto, m1) -> answer (soprano, m3) -> subject (bass, m7)."""
    _, anns, _, _ = aligned
    entries = [a for a in anns if a["type"] == "subject"][:3]
    assert [(a["measure_number_source"], a["voice_id"]) for a in entries] == [
        ("1", "voice_2"), ("3", "voice_1"), ("7", "voice_3")]
    assert [a["entry_function"] for a in entries] == ["subject", "answer", "subject"]


def test_two_countersubjects_and_closing_tonic_pedal(aligned):
    _, anns, _, _ = aligned
    assert {a["countersubject_index"] for a in anns if a["type"] == "countersubject"} == {1, 2}
    pedals = [a for a in anns if a["type"] == "pedal"]
    assert len(pedals) == 1
    assert pedals[0]["pedal_degree"] == "I"
    assert pedals[0]["measure_number_source"] == "29"


def test_final_cadence_is_picardy_major(aligned):
    _, anns, _, _ = aligned
    cadences = [a for a in anns if a["type"] == "cadence"]
    assert [a["cadence_type"] for a in cadences] == ["PAC", "rIAC", "IAC", "PAC"]
    # C minor fugue, major-mode final cadence: the Picardy third.
    assert cadences[-1]["cadence_key_area"] == "I"


def test_every_annotation_lands_on_real_notes(aligned):
    score, anns, _, _ = aligned
    voiced = [a for a in anns if a["voice_note_coverage"] is not None]
    assert len(voiced) == 20
    assert min(a["voice_note_coverage"] for a in voiced) > 0.8


def test_onsets_are_exact_rationals_on_the_score_time_base(aligned):
    score, anns, _, _ = aligned
    total = Fraction(score["quarter_length_total"])
    for a in anns:
        onset = Fraction(a["onset_q"])          # parses only if exactly rational
        assert 0 <= onset <= total
        if a["duration_q"] is not None:
            assert Fraction(a["duration_q"]) > 0
