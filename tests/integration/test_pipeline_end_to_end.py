"""Integration tests across the built pipeline (Spec §25.2).

§25.2 asks for: one WTC I fugue with Algomus annotations (covered in
test_annotation_alignment.py), one WTC II fugue, one BCFB chorale with alternate
version behaviour, one deliberately malformed score, one valid synthetic
fragment, one controlled invalid fragment, one reviewed excerpt and one reviewed
preference pair. The last two are asserted as *state machine* behaviour, since no
human has reviewed anything yet and the pipeline must not pretend otherwise.
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fugue_data.constraints.engine import evaluate, hard_violations, load_profile
from fugue_data.constraints.fragment import Fragment
from fugue_data.generation.validate import validate_candidate
from fugue_data.validation.canonical import validate_canonical

CANON = ROOT / "data" / "canonical"
pytestmark = pytest.mark.skipif(
    not (CANON / "bach_wtc2_fugue_01.json").exists(),
    reason="corpus not built in this checkout")


def _stage_complete(summary_name: str, *data_paths: str) -> bool:
    """A stage is done when its summary manifest AND its output both exist.

    The summary alone is not enough: manifests/ survives a `rm -rf data/generated`,
    so a stale summary from an earlier build makes a missing corpus look finished.
    Shard files alone are not enough either: they appear immediately and fill over
    the next hour, so a mid-flight corpus looks present but is incomplete. Both
    conditions together are what "this stage actually finished" means.
    """
    if not (ROOT / "manifests" / summary_name).exists():
        return False
    return all((ROOT / p).exists() for p in data_paths)


needs_generated = pytest.mark.skipif(
    not _stage_complete("generation_valid_summary.json",
                        "data/generated/valid/checkpoint.json"),
    reason="valid generation has not completed in this checkout")

needs_invalid = pytest.mark.skipif(
    not _stage_complete("generation_invalid_summary.json",
                        "data/generated/invalid"),
    reason="invalid generation has not completed in this checkout")


def _first_shard_record(kind: str):
    d = ROOT / "data" / "generated" / kind
    for shard in sorted(d.glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                return json.loads(line)
    return None


# ---- one WTC II fugue -------------------------------------------------------- #
def test_wtc2_fugue_normalizes_and_validates():
    d = json.loads((CANON / "bach_wtc2_fugue_01.json").read_text())
    assert validate_canonical(d) == []
    assert d["voice_count"] >= 2
    assert Fraction(d["quarter_length_total"]) > 0


# ---- one BCFB chorale with alternate version behaviour ----------------------- #
def test_bcfb_variant_pair_shares_a_work_and_a_split():
    a = json.loads((CANON / "bach_chorale_bwv10.07a.json").read_text())
    b = json.loads((CANON / "bach_chorale_bwv10.07b.json").read_text())
    assert a["work_group_id"] == b["work_group_id"] == "BWV10.07"
    # Distinct encodings: same work, different content.
    assert a["normalization"]["core_event_hash"] != b["normalization"]["core_event_hash"]
    rows = {json.loads(l)["piece_id"]: json.loads(l)
            for l in (ROOT / "manifests" / "splits.jsonl").read_text().splitlines()
            if l.strip()}
    assert rows[a["piece_id"]]["split"] == rows[b["piece_id"]]["split"]


# ---- one deliberately malformed score ---------------------------------------- #
def test_malformed_score_is_rejected_by_validation():
    d = json.loads((CANON / "bach_wtc2_fugue_01.json").read_text())
    d["voices"][0]["notes"][0]["duration_q"] = "-1"
    errors = validate_canonical(d)
    assert any("negative duration" in e for e in errors)


def test_malformed_score_fails_the_rule_engine_too():
    d = json.loads((CANON / "bach_wtc2_fugue_01.json").read_text())
    d["voices"][0]["notes"][0]["duration_q"] = "0"
    f = Fragment.from_score(d)
    vs = evaluate(f, load_profile("bach_empirical"), only={"timing_positive_duration"})
    assert vs and vs[0]["severity"] == "hard"


# ---- one valid synthetic fragment -------------------------------------------- #
@needs_generated
def test_valid_synthetic_fragment_revalidates_from_disk():
    rec = _first_shard_record("valid")
    assert rec is not None
    assert rec["source_type"] == "synthetic_valid" and rec["hard_valid"]
    assert rec["score"]["composer_id"] == "synthetic"     # §27
    prof = load_profile(rec["rule_profile"])
    v = validate_candidate(rec["score"], rec["request"], prof, set())
    assert v["hard_violation_count"] == 0
    assert v["canonical_schema_valid"] and v["structure_complete"]


# ---- one controlled invalid fragment ------------------------------------------ #
@needs_invalid
def test_controlled_invalid_has_exactly_its_intended_violation():
    rec = _first_shard_record("invalid")
    assert rec is not None
    assert len(rec["violations"]) == 1
    assert rec["violations"][0]["rule_id"] == rec["target_rule_id"]
    assert rec["pre_validation_hash"] != rec["post_validation_hash"]
    prof = load_profile(rec["rule_profile"])
    f = Fragment.from_score(rec["score"])
    f.range_profile = "keyboard"
    hard = hard_violations(evaluate(f, prof, only={rec["target_rule_id"]}))
    # Structural rules need the plan to fire; the rest must reproduce on re-run.
    if rec["target_rule_id"] not in ("subject_entry_identity", "answer_transformation",
                                     "cadence_target_completion"):
        assert hard, rec["target_rule_id"]


@needs_invalid
def test_negative_sits_in_its_parents_split():
    valid = {}
    for shard in sorted((ROOT / "data" / "generated" / "valid").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                valid[r["sample_id"]] = r["split"]
    checked = 0
    for shard in sorted((ROOT / "data" / "generated" / "invalid").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["parent_valid_sample_id"] in valid:
                assert r["split"] == valid[r["parent_valid_sample_id"]]
                checked += 1
    assert checked > 100


# ---- one reviewed excerpt: the state machine, not a fabricated approval ------- #
def test_excerpt_review_state_machine():
    rows = [json.loads(l) for l in
            (ROOT / "data" / "proposed" / "excerpts.jsonl").read_text().splitlines()
            if l.strip()]
    live = [r for r in rows if not r["auto_rejected"]]
    assert len(live) >= 200                                   # §14.6
    assert all(r["review_status"] in
               ("pending", "approved", "rejected", "correction_required") for r in live)
    # Nothing may claim human approval without a reviewer and a timestamp.
    for r in live:
        if r["review_status"] == "approved":
            assert r["reviewer"] and r["reviewed_at_utc"]


def test_approved_excerpts_are_the_only_ones_exported_as_human_reviewed():
    import zstandard as zstd
    approved = {r["excerpt_id"] for r in
                (json.loads(l) for l in
                 (ROOT / "data" / "proposed" / "excerpts.jsonl").read_text().splitlines()
                 if l.strip())
                if r["review_status"] == "approved"}
    style = ROOT / "data" / "exports" / "style"
    if not style.exists():
        pytest.skip("exports not built")
    for f in style.glob("*.jsonl.zst"):
        for line in zstd.ZstdDecompressor().decompress(f.read_bytes()).decode().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("source_type", "").endswith("human_approved"):
                assert r["id"] in approved


# ---- one reviewed preference pair: same discipline ---------------------------- #
@pytest.mark.skipif(
    not _stage_complete("preference_summary.json",
                        "data/generated/preference_candidates/pairs.jsonl"),
    reason="preference pairs have not been generated in this checkout")
def test_preference_pair_state_machine_and_prompt_identity():
    from fugue_data.generation.preferences import shared_conditions_hash
    pairs = [json.loads(l) for l in
             (ROOT / "data" / "generated" / "preference_candidates"
              / "pairs.jsonl").read_text().splitlines() if l.strip()]
    assert len(pairs) >= 500
    cands = {r["sample_id"]: r for r in
             (json.loads(l) for l in
              (ROOT / "data" / "generated" / "preference_candidates"
               / "candidates.jsonl").read_text().splitlines() if l.strip())}
    for p in pairs[:200]:
        a, b = cands[p["candidate_a_id"]], cands[p["candidate_b_id"]]
        # §18.2: identical conditions, different seeds.
        assert shared_conditions_hash(a["request"]) == shared_conditions_hash(b["request"])
        assert a["seed"] != b["seed"]
        assert a["split"] == b["split"] and a["split"] != "test"
        if p["review_status"] == "reviewed":
            assert p["label"] in ("A", "B", "tie", "both_poor")


def test_unreviewed_pairs_are_not_exported():
    d = ROOT / "data" / "exports" / "preferences"
    if not d.exists():
        pytest.skip("exports not built")
    for f in d.glob("*_pairs.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                assert r["review_status"] == "reviewed" and r["label"] is not None
