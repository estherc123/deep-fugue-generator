"""Unit tests for acquisition piece-identity classification (Spec §7, §25.1)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fugue_data.acquire.acquire import classify, wtc_bwv, _bcfb_chorale_id, infer_format


def test_wtc_bwv_mapping():
    assert wtc_bwv(1, "f", 1) == "BWV846"
    assert wtc_bwv(1, "f", 24) == "BWV869"
    assert wtc_bwv(2, "f", 1) == "BWV870"
    assert wtc_bwv(2, "f", 24) == "BWV893"


def test_wtc_fugue_vs_prelude():
    fugue = classify("humdrum-wtc", "kern/wtc1f01.krn", "kern")
    assert fugue.movement_claimed == "fugue"
    assert fugue.work_id_claimed == "BWV846"
    assert fugue.piece_id == "bach_wtc1_fugue_01"
    assert fugue.included_in_default is True

    prelude = classify("humdrum-wtc", "kern/wtc1p01.krn", "kern")
    assert prelude.movement_claimed == "prelude"
    assert prelude.included_in_default is False  # excluded from default corpus


def test_bcfb_chorale_id_grouping():
    # a/b variants share one work_group_id
    assert _bcfb_chorale_id("BWV_10.07a_FB") == ("BWV10.07", "a")
    assert _bcfb_chorale_id("BWV_10.07b_FB") == ("BWV10.07", "b")
    # dot-less standalone chorale
    assert _bcfb_chorale_id("BWV_250_FB") == ("BWV250", "")


def test_bcfb_master_vs_derived():
    master = classify("bcfb", "FB_source/musicXML_master/BWV_10.07a_FB.musicxml", "musicxml")
    assert master.movement_claimed == "chorale"
    assert master.work_id_claimed == "BWV10.07"
    assert master.included_in_default is True

    kern = classify("bcfb", "FB_source/translated_kern/BWV_10.07a_FB.krn", "kern")
    assert kern.work_id_claimed == "BWV10.07"
    assert kern.included_in_default is False  # secondary format, not default corpus

    editorial = classify(
        "bcfb", "FB_source/musicXML_master/editorial_FB_only/BWV_10.07a_FB.musicxml", "musicxml")
    assert editorial.included_in_default is False  # derived, retained but excluded


def test_algomus_dez_vs_synchro():
    ref = classify("algomus", "fugues/bach-wtc-i/01-bwv846-ref.dez", "json")
    assert ref.movement_claimed == "annotation"
    assert ref.piece_id == "bach_wtc1_fugue_01"
    assert ref.variant_id == "algomus-ref"

    synchro = classify(
        "algomus", "fugues/bach-wtc-i/synchro/01-bwv846-All_of_Bach-synchro.json", "json")
    assert synchro.piece_id is None  # audio-sync is not a structural annotation


def test_infer_format():
    assert infer_format(Path("x.krn")) == "kern"
    assert infer_format(Path("x.musicxml")) == "musicxml"
    assert infer_format(Path("x.dez")) == "json"
    assert infer_format(Path("x.foo")) == "other"
