"""Build a blind A/B: prior-ranked best-of-N vs an unranked pick.

The comparison is deliberately narrow. Both sides come from the SAME request and
the SAME pool of N solver outputs; the only difference is that one side was chosen
by the style prior and the other was chosen at random. That isolates the ranking.
Comparing "best of 12" against "a single solve" would instead confound the ranker
with the extra solver attempts, and would flatter the prior.

Sides are shuffled per item and the assignment is written to key.json, which the
review app never serves.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from ..normalize.rational import parse_frac
from ..review.arbitrate import _notes
from ..review.render import piano_roll_svg
from .best_of_n import PriorScorer, best_of_n, rank_adjusted, fit_density_baseline, hold_fraction


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build(root: Path | None = None, *, n_requests: int = 40, n: int = 12,
          seed: int = 4242, out: Path | None = None,
          max_time_s: float = 5.0) -> dict:
    root = root or _repo_root()
    out = Path(out) if out else (root / "data" / "generated" / "bon_ab")
    out.mkdir(parents=True, exist_ok=True)
    scorer = PriorScorer(root)

    src = root / "data" / "generated" / "preference_candidates" / "candidates.jsonl"
    # Fresh requests from the §16 grid rather than the ones stored on existing
    # candidates: serialization strips the derived keys (_total_q, _register_shift)
    # the solver needs, and rebuilding them here would duplicate requests.py.
    import yaml
    from ..generation.requests import build_requests
    cfg = yaml.safe_load((root / "config" / "generation.yaml").read_text())
    requests = build_requests(n_requests * 6, cfg, seed=seed)
    # Closed prompts first, as in pair_candidates: an unresolved ending dominates
    # a listener's impression and would drown out the difference being tested.
    requests.sort(key=lambda r: 0 if r.get("cadence_target") else 1)

    rng = random.Random(seed)

    # A density baseline fit once over a broad sample, so every request is judged
    # on the same curve rather than on a 12-point fit of its own.
    pool = []
    for i, line in enumerate(src.read_text().splitlines()):
        if i >= 300 or not line.strip():
            break
        c = json.loads(line)
        nll, _ = scorer.nll(c["score"])
        pool.append((hold_fraction(scorer, c["score"]), nll))
    baseline = fit_density_baseline(pool)

    items, key, stats = [], [], []
    for req in requests:
        if len(items) >= n_requests:
            break
        winner, ranked = best_of_n(req, n, scorer, base_seed=rng.randrange(2**30),
                                   max_time_s=max_time_s)
        if len(ranked) < 2:
            continue                       # nothing to choose between
        ranked = rank_adjusted([r.candidate for r in ranked], scorer, baseline)
        best = ranked[0]
        # The control: a uniform pick from the same pool, best excluded, so the
        # two sides can never be the same fragment.
        control = rng.choice(ranked[1:])
        flip = rng.random() < 0.5
        left, right = (control, best) if flip else (best, control)
        aid = hashlib.sha256(f"{seed}:{len(items)}".encode()).hexdigest()[:10]
        items.append({
            "arb_id": aid,
            "left_svg": piano_roll_svg(left.candidate["score"], parse_frac("0"),
                parse_frac(left.candidate["score"]["quarter_length_total"]), None, ""),
            "right_svg": piano_roll_svg(right.candidate["score"], parse_frac("0"),
                parse_frac(right.candidate["score"]["quarter_length_total"]), None, ""),
            "left_notes": _notes(left.candidate["score"]),
            "right_notes": _notes(right.candidate["score"]),
        })
        key.append({"arb_id": aid,
                    "left_is": "control" if flip else "prior",
                    "right_is": "prior" if flip else "control",
                    "prior_human": None, "prior_other": None,
                    "solved": len(ranked),
                    "prior_score": round(best.nll, 4),
                    "control_score": round(control.nll, 4)})
        stats.append((best.nll, control.nll, best.raw_nll, control.raw_nll))

    (out / "items.json").write_text(json.dumps(items, indent=2) + "\n")
    (out / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    return {"items": len(items), "dir": str(out),
            "mean_prior_score": round(sum(s[0] for s in stats) / max(1, len(stats)), 4),
            "mean_control_score": round(sum(s[1] for s in stats) / max(1, len(stats)), 4)}


if __name__ == "__main__":
    import sys
    kw = dict(a.lstrip("-").split("=") for a in sys.argv[1:])
    print(json.dumps(build(n_requests=int(kw.get("n_requests", 40)),
                           n=int(kw.get("n", 12))), indent=2))


def score_ab(root: Path | None = None, *, out: Path | None = None) -> dict:
    """Score the blind A/B once verdicts exist.

    Reported as a win rate for the prior-ranked side over decisive verdicts only.
    Ties and both-poor are counted and excluded from the rate rather than split,
    because on this reviewer's measured behaviour a hedge means "no preference",
    not "half a preference" -- 37% of forced binary choices became hedges when the
    option existed, so folding them in as half-wins would manufacture signal.
    """
    import math
    root = root or _repo_root()
    d = Path(out) if out else (root / "data" / "generated" / "bon_ab")
    key = {k["arb_id"]: k for k in json.loads((d / "key.json").read_text())}
    vp = d / "verdicts.json"
    if not vp.exists():
        return {"error": "no verdicts.json yet", "dir": str(d)}
    verdicts = json.loads(vp.read_text())

    prior_wins = control_wins = hedged = 0
    for aid, choice in verdicts.items():
        k = key.get(aid)
        if not k:
            continue
        if choice in ("tie", "both_poor", "cant_tell"):
            hedged += 1
        elif choice == "left":
            prior_wins += k["left_is"] == "prior"; control_wins += k["left_is"] != "prior"
        elif choice == "right":
            prior_wins += k["right_is"] == "prior"; control_wins += k["right_is"] != "prior"
    dec = prior_wins + control_wins
    res = {"judged": len(verdicts), "decisive": dec, "hedged": hedged,
           "prior_wins": prior_wins, "control_wins": control_wins}
    if dec:
        p = prior_wins / dec
        # Wilson, not Wald: the Wald interval collapses to zero width at p=0 or
        # p=1, so a 2-for-2 sweep would report itself as significant. Wilson stays
        # honest at the extremes and at the small n this comparison will have.
        z = 1.96
        denom = 1 + z * z / dec
        centre = (p + z * z / (2 * dec)) / denom
        half = (z / denom) * math.sqrt(p * (1 - p) / dec + z * z / (4 * dec * dec))
        lo, hi = centre - half, centre + half
        res["prior_win_rate"] = round(p, 3)
        res["ci95"] = [round(max(0.0, lo), 3), round(min(1.0, hi), 3)]
        res["beats_chance"] = bool(lo > 0.5)
        res["reading"] = (
            "The prior-ranked side wins significantly." if lo > 0.5 else
            "Not distinguishable from chance at this sample size -- the prior is "
            "not demonstrably improving what a listener hears, whatever it does to "
            "held-out perplexity.")
    return res
