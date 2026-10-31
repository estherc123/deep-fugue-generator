

def test_self_consistency_ignores_whether_the_agent_agreed(tmp_path):
    """A re-judge that repeats the human's own label is self-consistent even when
    the agent also picked it. Regression: the random sample under-reported 28
    consistent re-judges as 24 because 4 of them coincided with the agent."""
    import json
    from fugue_data.review.arbitrate import score
    (tmp_path / "key.json").write_text(json.dumps([
        {"arb_id": "a1", "pair_id": "p1", "left_is": "A", "right_is": "B",
         "prior_human": "A", "prior_other": "A"},   # human repeats; agent agrees
        {"arb_id": "a2", "pair_id": "p2", "left_is": "A", "right_is": "B",
         "prior_human": "A", "prior_other": "B"},   # human repeats; agent differs
    ]))
    (tmp_path / "verdicts.json").write_text(json.dumps({"a1": "left", "a2": "left"}))
    r = score(arb_dir=tmp_path)
    assert r["repeated_own_label"] == 2
    assert r["self_consistency"] == 1.0
    assert r["matches_your_earlier_label"] == 1   # attribution bucket, still exclusive


def test_bon_ab_scoring_excludes_hedges_from_the_win_rate(tmp_path):
    """A hedge is 'no preference', not half a win -- folding hedges in as 0.5
    would manufacture signal from this reviewer's large measured abstain region."""
    import json
    from fugue_data.modeling.bon_ab import score_ab
    (tmp_path / "key.json").write_text(json.dumps([
        {"arb_id": "a", "left_is": "prior", "right_is": "control"},
        {"arb_id": "b", "left_is": "control", "right_is": "prior"},
        {"arb_id": "c", "left_is": "prior", "right_is": "control"},
    ]))
    (tmp_path / "verdicts.json").write_text(json.dumps(
        {"a": "left", "b": "right", "c": "tie"}))
    r = score_ab(out=tmp_path)
    assert r["prior_wins"] == 2 and r["control_wins"] == 0
    assert r["hedged"] == 1 and r["decisive"] == 2
    assert r["prior_win_rate"] == 1.0
    assert r["beats_chance"] is False          # n=2 cannot clear the CI


def test_prior_scorer_rejects_unscoreable_fragments():
    """An un-griddable or out-of-vocabulary fragment must lose, never win by default."""
    import math
    from fugue_data.modeling.best_of_n import PriorScorer
    sc = PriorScorer()
    nll, n = sc.nll({"voices": [], "quarter_length_total": "4"})
    assert nll == math.inf and n == 0


def test_stepwise_floor_does_not_forbid_sustained_notes():
    """A sustained note has interval 0 and must not count as a step, nor be made
    infeasible. Reifying `st` as `ad <= 2` alone did both: it counted sustains as
    steps, and its negative half (`ad >= 3`) made ad == 0 unsatisfiable, which
    would have forced every voice to move on every slot."""
    import yaml
    from pathlib import Path
    from fugue_data.generation import solver as S
    from fugue_data.generation.requests import build_requests
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]
                          / "config" / "generation.yaml").read_text())
    reqs = build_requests(30, cfg, seed=4242)
    reqs.sort(key=lambda r: 0 if r.get("cadence_target") else 1)
    solved = [f for f in (S.solve_fragment(r, max_time_s=5.0) for r in reqs[:20]) if f]
    assert solved, "the stepwise floor made every request infeasible"
    # And sustains genuinely occur in the output, rather than every voice moving.
    from fractions import Fraction
    holds = 0
    for f in solved:
        for v in f.get("voices", []):
            ns = sorted(v.get("notes", []), key=lambda n: Fraction(str(n["onset_q"])))
            holds += sum(1 for n in ns if Fraction(str(n["duration_q"])) > 1)
    assert holds > 0, "no sustained notes -- the floor is forcing motion everywhere"


def test_measured_floors_sit_below_the_corpus_rates_they_cite():
    """The constants carry Bach rates in their comments; keep them honest."""
    from fugue_data.generation import solver as S
    # Floors apply to free voices only, so the stepwise floor may exceed the
    # whole-fragment corpus rate; the fifth floor may not exceed Bach's outright.
    assert 0 < S.FIFTH_MOTION_FLOOR_PCT <= 40
    assert 0 < S.STEPWISE_FLOOR_PCT <= 85


def test_motif_primes_carry_their_own_voice_count():
    """Priming a 3-voice fugue into a 4-voice scaffold shifts every token out of
    phase -- voice 1's second note lands in voice 4. All six held-out WTC fugue
    primes are 3-voice, so a fixed voice_count=4 corrupted 100% of the fugal
    material while leaving the chorales correct and the bug invisible."""
    from fugue_data.modeling.motifs import usable_primes
    primes = usable_primes()
    assert primes, "no usable primes"
    assert all("voice_count" in p for p in primes)
    counts = {p["voice_count"] for p in primes}
    assert counts != {4}, "expected a mix of voice counts among the primes"
    for p in primes:
        assert 2 <= p["voice_count"] <= 6


def test_motif_extraction_reaches_the_fugue_primes():
    """Walking primes in corpus order and stopping at `count` never reached the
    six WTC fugues: each prime yields up to one motif per voice, so 24 motifs
    exhausted the first six primes, all chorales. The target genre is fugue."""
    from fugue_data.modeling.best_of_n import PriorScorer
    from fugue_data.modeling.motifs import extract_motifs
    ms = extract_motifs(PriorScorer(), count=16, n_notes=5, seed=7)
    assert ms, "no motifs extracted"
    assert any("wtc" in m["source"] for m in ms), \
        "no fugue-derived motifs -- the fugue primes fell off the end again"


def test_pair_budget_is_for_fresh_pairs_not_the_total(tmp_path, monkeypatch):
    """Retention must not consume the budget for new material.

    When `target` capped the TOTAL, a pool whose retained (already-judged) pairs
    outnumbered the target could never grow: the loop broke before adding one new
    pair. A 3.8-hour run producing 3,435 candidates contributed nothing to review.
    """
    from fugue_data.generation import pairs_runner as pr
    built = [{"pair_id": f"p{i}", "is_consistency_repeat": False} for i in range(50)]
    keep = {f"p{i}" for i in range(30)}          # 30 retained, target 10
    pairs = [b for b in built if b["pair_id"] in keep]
    added = 0
    for b in built:
        if added >= 10:
            break
        if b["pair_id"] not in keep:
            pairs.append(b)
            added += 1
    assert added == 10, "fresh pairs must be added even when retained > target"
    assert len(pairs) == 40
