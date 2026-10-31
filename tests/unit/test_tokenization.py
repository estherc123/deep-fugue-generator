"""Unit tests for model-view tokenization (Spec §19, §25.1)."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fugue_data.tokenization.runner import decode_grid
from fugue_data.tokenization.views import (GridUnsupported, choose_denominator,
                                           event_records, grid_tokens, merge_tied)
from fugue_data.tokenization.vocab import (build_vocabulary, vocabulary_hash,
                                           vocabulary_index)

CFG = yaml.safe_load((ROOT / "config" / "tokenizer.yaml").read_text())


def mk_note(voice, onset, dur, midi, tie=None, measure=1, offset=None, step="C", octv=5):
    return {"note_id": f"{voice}:{onset}:{midi}", "voice_id": voice,
            "onset_q": str(onset), "duration_q": str(dur), "measure_number": str(measure),
            "measure_index_unfolded": measure - 1,
            "offset_in_measure_q": str(onset if offset is None else offset),
            "midi": midi, "step": step, "alter": "0", "octave": octv,
            "pitch_class": midi % 12, "diatonic_step": 35, "tie": tie, "grace": False,
            "articulations": [], "ornaments": [], "chord_group": None, "source_refs": []}


def mk_score(voices, total, measures=1):
    return {"schema_version": "1.0.0", "piece_id": "p", "work_group_id": "W",
            "composer_id": "bach_js", "movement_type": "fugue", "source_file_ids": [],
            "voice_count": len(voices),
            "voices": [{"voice_id": f"voice_{i+1}", "staff": 1, "notes": ns}
                       for i, ns in enumerate(voices)],
            "meter_map": [{"onset_q": "0", "numerator": 4, "denominator": 4}],
            "key_signature_map": [{"onset_q": "0", "fifths": -3}], "tempo_map": [],
            "annotations": [], "quarter_length_total": str(total),
            "measure_count": measures, "warnings": [],
            "normalization": {"software_version": "t", "config_hash": "t",
                              "core_event_hash": "h" * 64}}


# ---- vocabulary (§19.4) ---------------------------------------------------- #
def test_vocabulary_is_closed_and_unique():
    v = build_vocabulary()
    assert len(v) == len(set(v))
    assert vocabulary_index(v)["<PAD>"] == 0


def test_vocabulary_hash_is_stable_and_order_sensitive():
    v = build_vocabulary()
    assert vocabulary_hash(v) == vocabulary_hash(build_vocabulary())
    assert vocabulary_hash(v) != vocabulary_hash(list(reversed(v)))


# ---- quantization policy (§19.2) ------------------------------------------- #
def test_sixteenths_chosen_when_exactly_representable():
    s = mk_score([[mk_note("voice_1", 0, "1/4", 60)]], 1)
    assert choose_denominator(s, [4, 8, 12]) == 4


def test_finer_denominator_chosen_for_triplets_never_rounded():
    s = mk_score([[mk_note("voice_1", 0, "1/3", 60)]], 1)
    assert choose_denominator(s, [4, 8, 12, 24]) == 12


def test_grid_unsupported_rather_than_quantized():
    s = mk_score([[mk_note("voice_1", 0, "1/5", 60)]], 1)
    with pytest.raises(GridUnsupported):
        choose_denominator(s, [4, 8, 12])


# ---- tie merging (§25.1) --------------------------------------------------- #
def test_merge_tied_folds_a_note_split_across_a_barline():
    notes = [mk_note("voice_1", 0, 4, 60, tie="start"),
             mk_note("voice_1", 4, 4, 60, tie="stop", measure=2)]
    assert merge_tied(notes) == [(Fraction(0), Fraction(8), 60)]


def test_tied_note_is_one_attack_in_the_grid():
    s = mk_score([[mk_note("voice_1", 0, 4, 60, tie="start"),
                   mk_note("voice_1", 4, 4, 60, tie="stop", measure=2)]], 8, measures=2)
    toks, policy = grid_tokens(s, CFG)
    assert toks.count("NOTE_ON_60") == 1
    assert toks.count("HOLD") == 31              # 32 steps, one of them the attack


# ---- grid structure (§19.2) ------------------------------------------------ #
def test_grid_emits_key_meter_bar_beat_and_end_of_section():
    s = mk_score([[mk_note("voice_1", 0, 4, 60)]], 4)
    toks, _ = grid_tokens(s, CFG)
    assert toks[0] == "KEY_-3" and toks[1] == "METER_4_4"
    assert "BAR" in toks and "BEAT_1" in toks
    assert toks[-1] == "<END_OF_SECTION>"


def test_grid_marks_rest_hold_and_note_on_per_voice():
    s = mk_score([[mk_note("voice_1", 0, 1, 60)],
                  [mk_note("voice_2", 1, 1, 55)]], 2)
    toks, _ = grid_tokens(s, CFG)
    assert "NOTE_ON_60" in toks and "NOTE_ON_55" in toks
    assert "REST" in toks and "HOLD" in toks


def test_grid_round_trips_to_the_original_attacks():
    s = mk_score([[mk_note("voice_1", 0, 1, 60), mk_note("voice_1", 2, 1, 64)],
                  [mk_note("voice_2", 1, 1, 48)]], 4)
    toks, policy = grid_tokens(s, CFG)
    decoded = decode_grid(toks, 2, policy["steps_per_quarter"])
    assert sorted(decoded["voice_1"]) == [(Fraction(0), 60), (Fraction(2), 64)]
    assert sorted(decoded["voice_2"]) == [(Fraction(1), 48)]


def test_intra_voice_collision_is_recorded_not_silently_dropped():
    """One pitch per voice per step is a real grid limit; it must be declared."""
    s = mk_score([[mk_note("voice_1", 0, 1, 60), mk_note("voice_1", 0, 1, 67)]], 1)
    _, policy = grid_tokens(s, CFG)
    assert policy["voice_collisions"] == 1
    assert policy["voice_collision_detail"][0]["dropped"] == 60


def test_structural_annotation_tokens_only_for_accepted_labels():
    s = mk_score([[mk_note("voice_1", 0, 4, 60)]], 4)
    anns = [{"type": "subject", "onset_q": "0", "duration_q": "2",
             "analytical_status": "accepted", "voice_id": "voice_1"},
            {"type": "subject", "onset_q": "2", "duration_q": "2",
             "analytical_status": "rejected_candidate", "voice_id": "voice_1"}]
    toks, _ = grid_tokens(s, CFG, anns)
    assert toks.count("ANN_SUBJECT_START") == 1


# ---- event view (§19.3) ---------------------------------------------------- #
def test_event_records_carry_required_fields():
    s = mk_score([[mk_note("voice_1", 0, 1, 60), mk_note("voice_1", 1, 1, 64)]], 2)
    recs, policy = event_records(s, CFG)
    assert policy["quantization"].startswith("none")
    r0, r1 = recs
    for field in ("voice_id", "onset_delta_q", "duration_q", "bar", "beat", "tie",
                  "step", "octave", "semitone_delta", "sections"):
        assert field in r0
    assert r0["onset_delta_q"] is None and r1["onset_delta_q"] == "1"
    assert r1["semitone_delta"] == 4


def test_event_view_keeps_notes_the_grid_cannot_hold():
    """§19.1: the canonical layer is not discarded, so divisi survives somewhere."""
    s = mk_score([[mk_note("voice_1", 0, 1, 60), mk_note("voice_1", 0, 1, 67)]], 1)
    recs, _ = event_records(s, CFG)
    assert sorted(r["midi"] for r in recs) == [60, 67]


def test_bar_tokens_survive_a_bar_where_every_voice_rests():
    """Bar position comes from the meter, not from where notes happen to fall.

    Deriving it from notes meant a bar in which all voices rest emitted no BAR
    token, the running bar-start never advanced, and the beat counter carried on
    across bars -- producing BEAT_17 and beyond, which are not in the vocabulary.
    The BCFB chorales are cantata movements whose vocal parts rest through
    instrumental bars, so five of them were affected and every gate passed.
    """
    from fugue_data.tokenization.vocab import build_vocabulary
    # 4 bars of 4/4 with nothing sounding in bars 2 and 3.
    s = mk_score([[mk_note("voice_1", 0, 1, 60),
                   mk_note("voice_1", 12, 1, 62, measure=4, offset=0)]], 16, measures=4)
    toks, _ = grid_tokens(s, CFG)
    declared = set(build_vocabulary())
    assert toks.count("BAR") == 4, f"expected 4 bar tokens, got {toks.count('BAR')}"
    offenders = sorted({t for t in toks if t not in declared})
    assert not offenders, f"tokens outside the vocabulary: {offenders}"
