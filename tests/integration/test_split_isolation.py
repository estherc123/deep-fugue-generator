"""Integration: split isolation over the real corpus (Spec §13.5, §21).

Skipped when the corpus has not been normalized in this checkout.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fugue_data.splitting.runner import run_split
from fugue_data.validation.splits import validate_splits

pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "canonical" / "bach_wtc1_fugue_01.json").exists(),
    reason="corpus not normalized in this checkout",
)


@pytest.fixture(scope="module")
def built():
    summary = run_split(ROOT)
    rows = [json.loads(l) for l in (ROOT / "manifests" / "splits.jsonl").read_text().splitlines()]
    groups = json.loads((ROOT / "manifests" / "duplicate_groups.json").read_text())["groups"]
    return summary, rows, groups


def test_no_leakage(built):
    summary, _, _ = built
    assert summary["leakage_errors"] == []


def test_targets_met_exactly(built):
    summary, _, _ = built
    assert summary["corpora"]["wtc"]["works_by_split"] == {
        "train": 36, "validation": 6, "test": 6}
    assert summary["corpora"]["bcfb"]["works_by_split"] == {
        "train": 111, "validation": 14, "test": 14}


def test_every_piece_assigned_once(built):
    _, rows, _ = built
    assert len(rows) == 191
    assert len({r["piece_id"] for r in rows}) == 191


def test_known_bcfb_variant_pairs_share_a_split(built):
    """The four a/b encoded variants are one work each (§13.3)."""
    _, rows, _ = built
    by_piece = {r["piece_id"]: r for r in rows}
    for work in ("10.07", "38.06", "161.06", "177.05"):
        a = by_piece[f"bach_chorale_bwv{work}a"]
        b = by_piece[f"bach_chorale_bwv{work}b"]
        assert a["split"] == b["split"], work
        assert a["split_unit_id"] == b["split_unit_id"]


def test_content_duplicates_across_different_bwv_share_a_split(built):
    """The two duplicates dedup found that work identity alone would have missed."""
    _, rows, _ = built
    by_piece = {r["piece_id"]: r for r in rows}
    for a, b in (("bach_chorale_bwv117.04", "bach_chorale_bwv117.09"),
                 ("bach_chorale_bwv125.06", "bach_chorale_bwv72.06")):
        assert by_piece[a]["split"] == by_piece[b]["split"]
        assert by_piece[a]["split_unit_id"] == by_piece[b]["split_unit_id"]


def test_run_is_reproducible(built):
    summary, rows, _ = built
    again = run_split(ROOT)
    rows2 = [json.loads(l) for l in (ROOT / "manifests" / "splits.jsonl").read_text().splitlines()]
    assert rows == rows2
    assert again["corpora"] == summary["corpora"]


def test_seed_is_recorded_on_every_row(built):
    _, rows, _ = built
    assert {r["split_seed"] for r in rows} == {20260728}


def test_gates_reject_a_tampered_manifest(built):
    """The gates must actually bite, not merely pass on clean input."""
    _, rows, groups = built
    tampered = [dict(r) for r in rows]
    victim = next(r for r in tampered if r["split"] != "test")
    victim["split"] = "test"
    errs = validate_splits(tampered, groups, {})
    assert errs, "tampering with a split assignment must be detected"
