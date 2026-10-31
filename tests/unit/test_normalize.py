"""Unit tests for normalization primitives (Spec §25.1)."""

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fugue_data.normalize.rational import to_fraction, frac_str, parse_frac
from fugue_data.normalize.model import Note
from fugue_data.normalize.hashing import (
    voice_melodic_hash, voice_rhythm_hash, voice_event_hash, _merge_ties,
)
from fugue_data.validation.canonical import validate_canonical


def mk_note(onset, dur, midi, tie=None, grace=False, step="C", octave=4):
    f = Fraction
    return Note(
        note_id=f"n{onset}_{midi}", voice_id="voice_1",
        onset_q=f(onset), duration_q=f(dur), measure_number="1",
        measure_index_unfolded=0, offset_in_measure_q=f(onset),
        midi=midi, step=step, alter=f(0), octave=octave,
        pitch_class=midi % 12, diatonic_step=1, tie=tie, grace=grace,
    )


# ---- rational-time conversion --------------------------------------------- #
def test_rational_dyadic_exact():
    assert to_fraction(0.5) == Fraction(1, 2)
    assert to_fraction(0.25) == Fraction(1, 4)
    assert to_fraction(0.75) == Fraction(3, 4)


def test_rational_tuplet_exact():
    assert to_fraction(Fraction(1, 3)) == Fraction(1, 3)
    assert to_fraction(1 / 3) == Fraction(1, 3)  # snaps float to musical denom


def test_frac_str_roundtrip():
    for s in ["0", "2", "3/8", "13/2", "1/3"]:
        assert frac_str(parse_frac(s)) == s


# ---- tie merging (§25.1) -------------------------------------------------- #
def test_tie_merge_combines_duration():
    notes = [mk_note(0, 1, 60, tie="start"), mk_note(1, 1, 60, tie="stop")]
    merged = _merge_ties(notes)
    assert merged == [(Fraction(0), Fraction(2), 60)]


def test_tied_note_hashes_equal_untied():
    tied = [mk_note(0, 1, 60, tie="start"), mk_note(1, 1, 60, tie="stop"),
            mk_note(2, 1, 62)]
    untied = [mk_note(0, 2, 60), mk_note(2, 1, 62)]
    assert voice_melodic_hash(tied) == voice_melodic_hash(untied)
    assert voice_rhythm_hash(tied) == voice_rhythm_hash(untied)


# ---- transposition invariance --------------------------------------------- #
def test_melodic_hash_transposition_invariant():
    base = [mk_note(0, 1, 60), mk_note(1, 1, 64), mk_note(2, 1, 67)]
    up5 = [mk_note(0, 1, 65), mk_note(1, 1, 69), mk_note(2, 1, 72)]
    assert voice_melodic_hash(base) == voice_melodic_hash(up5)


def test_event_hash_not_transposition_invariant():
    base = [mk_note(0, 1, 60), mk_note(1, 1, 64)]
    up = [mk_note(0, 1, 61), mk_note(1, 1, 65)]
    assert voice_event_hash(base) != voice_event_hash(up)


# ---- validation gates (§10.4) --------------------------------------------- #
def _doc(notes, warnings=None):
    return {
        "schema_version": "1.0.0", "piece_id": "p", "work_group_id": "BWV1",
        "composer_id": "bach_js", "movement_type": "fugue", "source_file_ids": ["s"],
        "voice_count": 1, "staff_count": 1,
        "voices": [{"voice_id": "voice_1", "staff": "1", "mean_midi": 60.0,
                    "notes": [n.to_dict() for n in notes]}],
        "meter_map": [], "key_signature_map": [],
        "warnings": warnings or [],
        "normalization": {"software_version": "0.1.0", "config_hash": "x",
                          "event_hash": "y"},
    }


def test_validation_flags_zero_duration():
    errs = validate_canonical(_doc([mk_note(0, 0, 60)]))
    assert any("zero-duration" in e for e in errs)


def test_validation_allows_grace_zero_duration():
    errs = validate_canonical(_doc([mk_note(0, 0, 60, grace=True), mk_note(0, 1, 62)]))
    assert not any("zero-duration" in e for e in errs)


def test_validation_flags_unexplained_overlap():
    notes = [mk_note(0, 2, 60), mk_note(1, 1, 64)]  # second starts before first ends
    errs = validate_canonical(_doc(notes))
    assert any("unexplained monophonic overlap" in e for e in errs)


def test_validation_accepts_explained_overlap():
    notes = [mk_note(0, 2, 60), mk_note(1, 1, 64)]
    warns = [{"code": "voice_partial_overlap", "voice_id": "voice_1", "onset_q": "1",
              "message": "divisi"}]
    errs = validate_canonical(_doc(notes, warnings=warns))
    assert not any("unexplained monophonic overlap" in e for e in errs)
