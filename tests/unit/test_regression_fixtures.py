"""Regression fixtures (Spec §25.3).

These pin representative canonical outputs, the annotation alignment of the
textbook fugue, the tokenizer vocabulary hash and the three ruleset hashes.

If one of these fails, that is not a flaky test: the pipeline's output has
changed. §25.3 requires a documented migration, a schema or normalizer version
change, and review of the affected exports before the fixture is updated.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fugue_data.constraints.engine import PROFILES, load_profile, ruleset_hash
from fugue_data.tokenization.vocab import build_vocabulary, vocabulary_hash

FIX = json.loads((ROOT / "tests" / "fixtures" / "regression_hashes.json").read_text())
CANON = ROOT / "data" / "canonical"
needs_corpus = pytest.mark.skipif(
    not (CANON / "bach_wtc1_fugue_01.json").exists(),
    reason="corpus not built in this checkout")


@needs_corpus
@pytest.mark.parametrize("piece_id", sorted(FIX["pieces"]))
def test_canonical_output_is_unchanged(piece_id):
    want = FIX["pieces"][piece_id]
    d = json.loads((CANON / f"{piece_id}.json").read_text())
    n = d["normalization"]
    got = {
        "work_group_id": d["work_group_id"],
        "voice_count": d["voice_count"],
        "measure_count": d["measure_count"],
        "quarter_length_total": d["quarter_length_total"],
        "note_count": sum(len(v["notes"]) for v in d["voices"]),
        "core_event_hash": n["core_event_hash"],
        "rhythm_hash": n["rhythm_hash"],
        "transposition_melodic_hash": n["transposition_melodic_hash"],
    }
    assert got == want, (
        f"{piece_id} canonical output changed. §25.3 requires a documented "
        f"migration and a version bump before updating this fixture.")


@needs_corpus
def test_annotation_alignment_is_unchanged():
    want = FIX["annotations"]["bach_wtc1_fugue_02"]
    d = json.loads((ROOT / "data" / "annotations" / "bach_wtc1_fugue_02.json").read_text())
    assert len(d["annotations"]) == want["count"]
    assert d["alignment"]["counts_by_type"] == want["counts_by_type"]
    assert round(d["alignment"]["mean_voice_note_coverage"], 6) == \
        want["mean_voice_note_coverage"]


def test_tokenizer_vocabulary_hash_is_unchanged():
    assert vocabulary_hash(build_vocabulary()) == FIX["vocabulary_hash"], (
        "tokenizer vocabulary changed; bump tokenizer_version and re-export")


@pytest.mark.parametrize("profile", PROFILES)
def test_ruleset_hash_is_unchanged(profile):
    assert ruleset_hash(load_profile(profile)) == FIX["ruleset_hashes"][profile], (
        f"{profile} ruleset changed; bump ruleset_version and re-validate "
        f"empirically (§15.6) before updating this fixture")
