"""Unit tests for deduplication and splitting (Spec §12, §13, §25.1)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fugue_data.splitting.dedup import (LEVELS, build_duplicate_groups,
                                        piece_signatures)
from fugue_data.splitting.features import (derive_mode, length_bucket,
                                           length_thresholds, meter_of,
                                           piece_features)
from fugue_data.splitting.split import (assign_splits, assignment_hash,
                                        build_split_units, split_manifest_rows,
                                        stratum_key)
from fugue_data.validation.splits import validate_splits


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def mk_note(voice, onset, dur, midi, pc=None):
    return {"note_id": f"{voice}:{onset}", "voice_id": voice,
            "onset_q": str(onset), "duration_q": str(dur),
            "measure_number": "1", "measure_index_unfolded": 0,
            "offset_in_measure_q": "0", "midi": midi, "step": "C", "alter": "0",
            "octave": midi // 12 - 1, "pitch_class": midi % 12 if pc is None else pc,
            "diatonic_step": 0, "tie": None, "grace": False,
            "articulations": [], "ornaments": [], "chord_group": None,
            "source_refs": []}


def mk_score(piece_id, work, midis=(60, 62, 64, 65), sha="a" * 64,
             event="e" * 64, melodic="m" * 64, rhythm="r" * 64, bars=10):
    notes = [mk_note("voice_1", i, 1, m) for i, m in enumerate(midis)]
    return {
        "schema_version": "1.0.0", "piece_id": piece_id, "work_group_id": work,
        "composer_id": "bach_js", "movement_type": "chorale", "source_file_ids": [],
        "voice_count": 1, "voices": [{"voice_id": "voice_1", "staff": 1, "notes": notes}],
        "meter_map": [{"onset_q": "0", "numerator": 4, "denominator": 4}],
        "key_signature_map": [], "tempo_map": [], "annotations": [],
        "quarter_length_total": str(len(midis)), "measure_count": bars, "warnings": [],
        "_source": [{"sha256": sha}],
        "normalization": {"software_version": "t", "config_hash": "t",
                          "core_event_hash": event, "event_hash": event,
                          "transposition_melodic_hash": melodic, "rhythm_hash": rhythm,
                          "per_voice_hashes": {}},
    }


def features_for(scores):
    feats = [piece_features(s) for s in scores]
    for f in feats:
        f["corpus"] = "bcfb"
        f["length_bucket"] = "short"
    return feats


# ---- features -------------------------------------------------------------- #
def test_meter_of():
    assert meter_of(mk_score("p", "W")) == "4/4"
    assert meter_of({"meter_map": []}) == "unknown"


def test_derive_mode_minor_and_major():
    # Tonic C (final bass note), corpus rich in E-flat -> minor.
    minor = mk_score("p", "W", midis=(60, 63, 63, 63, 60))
    assert derive_mode(minor) == "minor"
    major = mk_score("p", "W", midis=(60, 64, 64, 64, 60))
    assert derive_mode(major) == "major"


def test_derive_mode_survives_picardy_third():
    """A major final chord must not flip a minor-mode piece (§13.2)."""
    # Overwhelmingly E-flat (minor 3rd) with a single E-natural at the end.
    notes = [mk_note("voice_1", i, 1, 63) for i in range(8)]
    notes.append(mk_note("voice_1", 8, 1, 64))
    notes.append(mk_note("voice_1", 9, 1, 60))          # final bass note = tonic C
    score = mk_score("p", "W")
    score["voices"] = [{"voice_id": "voice_1", "staff": 1, "notes": notes}]
    assert derive_mode(score) == "minor"


@pytest.mark.slow
def test_derive_mode_matches_known_wtc_ordering():
    """The WTC's own ordering fixes every fugue's mode: odd = major, even = minor."""
    canonical = ROOT / "data" / "canonical"
    files = sorted(canonical.glob("bach_wtc*_fugue_*.json"))
    if not files:
        pytest.skip("corpus not normalized in this checkout")
    for p in files:
        idx = int(p.stem.split("_")[-1])
        expected = "major" if idx % 2 == 1 else "minor"
        assert derive_mode(json.loads(p.read_text())) == expected, p.stem


def test_length_thresholds_and_buckets():
    lo, hi = length_thresholds([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert length_bucket(1, (lo, hi)) == "short"
    assert length_bucket(9, (lo, hi)) == "long"
    assert length_bucket(None, (lo, hi)) == "unknown"


# ---- deduplication (§12) --------------------------------------------------- #
def test_all_six_levels_are_reported():
    sigs = [piece_signatures(mk_score("a", "W1")), piece_signatures(mk_score("b", "W2"))]
    r = build_duplicate_groups(sigs, ["canonical_event_hash"])
    assert set(r["level_census"]) == set(LEVELS)


def test_identical_checksum_groups_pieces():
    sigs = [piece_signatures(mk_score("a", "W1", sha="f" * 64)),
            piece_signatures(mk_score("b", "W2", sha="f" * 64, event="z" * 64))]
    r = build_duplicate_groups(sigs, ["source_checksum"])
    assert len(r["groups"]) == 1
    assert r["groups"][0]["members"] == ["a", "b"]
    assert "source_checksum" in r["groups"][0]["matched_levels"]


def test_identical_event_hash_groups_across_different_work_ids():
    """The BWV 125.06 / BWV 72.06 case: different works, identical music."""
    sigs = [piece_signatures(mk_score("a", "W1", sha="1" * 64)),
            piece_signatures(mk_score("b", "W2", sha="2" * 64))]
    r = build_duplicate_groups(sigs, ["canonical_event_hash"])
    assert r["groups"][0]["members"] == ["a", "b"]
    assert r["groups"][0]["work_group_ids"] == ["W1", "W2"]


def test_work_identity_groups_ab_variants():
    sigs = [piece_signatures(mk_score("a", "W", sha="1" * 64, event="1" * 64)),
            piece_signatures(mk_score("b", "W", sha="2" * 64, event="2" * 64))]
    r = build_duplicate_groups(sigs, ["work_identity"])
    assert r["groups"][0]["members"] == ["a", "b"]


def test_non_blocking_level_does_not_group():
    """A shared rhythm is a similarity signal, not an identity claim."""
    sigs = [piece_signatures(mk_score("a", "W1", sha="1" * 64, event="1" * 64,
                                      melodic="1" * 64)),
            piece_signatures(mk_score("b", "W2", sha="2" * 64, event="2" * 64,
                                      melodic="2" * 64))]
    r = build_duplicate_groups(sigs, ["canonical_event_hash"])
    assert r["groups"] == []
    assert r["level_census"]["rhythm_hash"]["match_groups"] == 1     # still reported
    assert r["level_census"]["rhythm_hash"]["blocking"] is False


def test_nothing_is_deleted_and_a_representative_is_named():
    sigs = [piece_signatures(mk_score("b", "W")), piece_signatures(mk_score("a", "W"))]
    g = build_duplicate_groups(sigs, ["work_identity"])["groups"][0]
    assert g["members"] == ["a", "b"]              # both retained
    assert g["preferred_piece_id"] == "a"          # deterministic representative


# ---- split units ----------------------------------------------------------- #
def test_split_unit_fuses_duplicate_group():
    feats = features_for([mk_score("a", "W1"), mk_score("b", "W2")])
    units = build_split_units(feats, {"a": "dup:0001", "b": "dup:0001"})
    assert len(units) == 1
    assert units[0]["weight"] == 2                 # consumes two works from target
    assert units[0]["piece_ids"] == ["a", "b"]


def test_split_unit_groups_variants_of_one_work():
    feats = features_for([mk_score("a", "W"), mk_score("b", "W")])
    units = build_split_units(feats, {})
    assert len(units) == 1 and units[0]["weight"] == 1


def test_stratum_key_is_stable():
    f = {"mode": "minor", "meter": "4/4", "book": 1}
    assert stratum_key(f, ["book", "mode"]) == "book=1|mode=minor"


# ---- assignment (§13) ------------------------------------------------------ #
def make_units(n, weight_two_at=()):
    feats = []
    for i in range(n):
        s = mk_score(f"p{i:03d}", f"W{i:03d}")
        feats.append(s)
    f = features_for(feats)
    groups = {}
    for i in weight_two_at:
        groups[f"p{i:03d}"] = f"dup:{i}"
        groups[f"p{i+1:03d}"] = f"dup:{i}"
    return build_split_units(f, groups)


def test_assignment_hits_targets_exactly():
    units = make_units(48)
    units = assign_splits(units, {"train": 36, "validation": 6, "test": 6}, ["mode"], 7)
    counts = {}
    for u in units:
        counts[u["split"]] = counts.get(u["split"], 0) + u["weight"]
    assert counts == {"train": 36, "validation": 6, "test": 6}


def test_assignment_is_deterministic_for_a_seed():
    a = assign_splits(make_units(30), {"train": 20, "validation": 5, "test": 5}, ["mode"], 3)
    b = assign_splits(make_units(30), {"train": 20, "validation": 5, "test": 5}, ["mode"], 3)
    assert {u["split_unit_id"]: u["split"] for u in a} == \
           {u["split_unit_id"]: u["split"] for u in b}


def test_different_seed_changes_assignment():
    a = assign_splits(make_units(30), {"train": 20, "validation": 5, "test": 5}, ["mode"], 3)
    b = assign_splits(make_units(30), {"train": 20, "validation": 5, "test": 5}, ["mode"], 4)
    assert {u["split_unit_id"]: u["split"] for u in a} != \
           {u["split_unit_id"]: u["split"] for u in b}


def make_strata_units(sizes: dict[str, int]):
    """Units labelled into named strata of controlled sizes."""
    feats = []
    i = 0
    for stratum, n in sorted(sizes.items()):
        for _ in range(n):
            f = piece_features(mk_score(f"p{i:04d}", f"W{i:04d}"))
            f["corpus"] = "bcfb"
            f["length_bucket"] = "short"
            f["mode"] = stratum          # `mode` is the stratification field used
            feats.append(f)
            i += 1
    return build_split_units(feats, {}), {f["piece_id"]: f["mode"] for f in feats}


def test_small_strata_are_not_swallowed_whole_by_train():
    """The bug this guards: allocating on *absolute* deficit hands the first ~100
    units straight to train (111 vs 14 vs 14), so any stratum that exhausts early
    lands entirely in train -- which is how all 37 long chorales and all 16 in 3/4
    ended up with zero validation and test representation."""
    sizes = {"big": 100, "s1": 5, "s2": 5, "s3": 5, "s4": 5, "s5": 5}
    units, stratum_of = make_strata_units(sizes)
    units = assign_splits(units, {"train": 100, "validation": 12, "test": 13},
                          ["mode"], 11)
    per_stratum = {s: {"train": 0, "validation": 0, "test": 0} for s in sizes}
    for u in units:
        for pid in u["piece_ids"]:
            per_stratum[stratum_of[pid]][u["split"]] += 1

    # The small strata all exhaust within the first ~30 units of the interleave,
    # which absolute-deficit greed would assign entirely to train. Held out
    # together they must reach validation and test.
    small_held_out = sum(per_stratum[s]["validation"] + per_stratum[s]["test"]
                         for s in ("s1", "s2", "s3", "s4", "s5"))
    assert small_held_out > 0

    # And the large stratum must track the target ratio rather than filling train
    # first. (An individual 5-member stratum landing all-train is ordinary chance
    # at an 80% train rate, so the guard is on the aggregate, not on each one.)
    big = per_stratum["big"]
    assert big["train"] == 80 and big["validation"] == 10 and big["test"] == 10


def test_weight_two_unit_keeps_targets_exact():
    units = make_units(20, weight_two_at=(0,))
    units = assign_splits(units, {"train": 15, "validation": 3, "test": 2}, ["mode"], 5)
    counts = {}
    for u in units:
        counts[u["split"]] = counts.get(u["split"], 0) + u["weight"]
    assert counts == {"train": 15, "validation": 3, "test": 2}


def test_assignment_hash_is_order_independent_and_verifies():
    h1 = assignment_hash(7, "u", ["b", "a"], "train")
    h2 = assignment_hash(7, "u", ["a", "b"], "train")
    assert h1 == h2
    assert h1 != assignment_hash(7, "u", ["a", "b"], "test")
    assert h1 != assignment_hash(8, "u", ["a", "b"], "train")


# ---- leakage gates (§13.5) ------------------------------------------------- #
def rows_for(units, seed=7, corpus="bcfb"):
    return split_manifest_rows(units, seed, corpus)


def test_clean_split_passes_all_gates():
    units = assign_splits(make_units(20), {"train": 14, "validation": 3, "test": 3},
                          ["mode"], 7)
    rows = rows_for(units)
    errs = validate_splits(rows, [], {"bcfb": {"train": 14, "validation": 3, "test": 3}})
    assert errs == []


def test_gate_catches_work_group_in_two_splits():
    units = assign_splits(make_units(20), {"train": 14, "validation": 3, "test": 3},
                          ["mode"], 7)
    rows = rows_for(units)
    rows[0]["work_group_id"] = rows[1]["work_group_id"]
    errs = validate_splits(rows, [], {})
    assert any("work_group_id in multiple splits" in e for e in errs) or \
           rows[0]["split"] == rows[1]["split"]


def test_gate_catches_duplicate_group_crossing_splits():
    units = assign_splits(make_units(20), {"train": 14, "validation": 3, "test": 3},
                          ["mode"], 7)
    rows = rows_for(units)
    a, b = next((x, y) for x in rows for y in rows if x["split"] != y["split"])
    errs = validate_splits(
        rows, [{"duplicate_group_id": "dup:1", "members": [a["piece_id"], b["piece_id"]]}], {})
    assert any("duplicate group crosses splits" in e for e in errs)


def test_gate_catches_tampered_assignment_hash():
    units = assign_splits(make_units(20), {"train": 14, "validation": 3, "test": 3},
                          ["mode"], 7)
    rows = rows_for(units)
    rows[0]["assignment_hash"] = "0" * 64
    errs = validate_splits(rows, [], {})
    assert any("assignment_hash mismatch" in e for e in errs)


def test_gate_catches_missed_target_counts():
    units = assign_splits(make_units(20), {"train": 14, "validation": 3, "test": 3},
                          ["mode"], 7)
    rows = rows_for(units)
    errs = validate_splits(rows, [], {"bcfb": {"train": 99, "validation": 3, "test": 3}})
    assert any("expected 99 works" in e for e in errs)
