"""Best-of-N generation: rank CP-SAT output under the Bach style prior.

The solver treats every hard-valid fragment as equally acceptable -- it has no
notion of which of two legal continuations sounds more like Bach. The prior does,
having been fit on 137 real works (test content perplexity 2.99 vs 4.72 n-gram).
So: solve one request under N seeds, keep the fragment the prior likes best.

The prior is a *ranker* here, never a filter -- every candidate it chooses from
has already passed the same hard gates as before, so this cannot admit anything
the validator would reject. It reorders; it does not permit.

Scoring is mean negative log-likelihood over CONTENT tokens only (pitch and
articulation decisions). The grid interleaves VOICE_n / BAR / BEAT markers in a
fixed pattern that any model predicts almost for free; including them would let a
fragment look good by being long and regular. Lower is better.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ..tokenization.views import grid_tokens, GridUnsupported
from .style_data import is_content
from .style_prior import StylePrior, device


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass
class Ranked:  # raw_nll / hold_fraction are attached by rank_adjusted
    """One scored candidate. ``nll`` is mean NLL per content token; lower = better."""
    nll: float
    content_tokens: int
    seed: int
    candidate: dict


class PriorScorer:
    """Scores a canonical score dict under the trained style prior."""

    def __init__(self, root: Path | None = None, ckpt: Path | None = None):
        self.root = root or _repo_root()
        ck = torch.load(ckpt or (self.root / "data" / "models" / "style_prior.pt"),
                        map_location="cpu")
        cfg = ck["config"]
        self.tokens = ck["tokens"]
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.content = [is_content(t) for t in self.tokens]
        self.dev = device()
        self.ctx = cfg["ctx"]
        self.model = StylePrior(cfg["vocab"], cfg["d"], cfg["layers"],
                                cfg["heads"], cfg["ctx"], cfg["drop"]).to(self.dev)
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.tok_cfg = yaml.safe_load(
            (self.root / "config" / "tokenizer.yaml").read_text())

    def tokenize(self, score: dict) -> list[str] | None:
        try:
            toks, _meta = grid_tokens(score, self.tok_cfg)
        except GridUnsupported:
            return None
        return toks

    @torch.no_grad()
    def nll(self, score: dict) -> tuple[float, int]:
        """Mean NLL per content token, and how many content tokens there were.

        Returns ``(inf, 0)`` for a fragment that cannot be gridded or carries a
        token the prior never saw -- an unscoreable candidate must lose, never
        win by default.
        """
        toks = self.tokenize(score)
        if not toks or any(t not in self.stoi for t in toks):
            return math.inf, 0
        ids = torch.tensor([self.stoi[t] for t in toks], dtype=torch.long)
        cm = torch.tensor([self.content[i] for i in ids.tolist()], dtype=torch.bool)
        lp = n = 0.0
        for start in range(0, len(ids) - 1, self.ctx):
            chunk = ids[start:start + self.ctx + 1]
            if len(chunk) < 2:
                continue
            logits = self.model(chunk[:-1].unsqueeze(0).to(self.dev))[0]
            step = -F.cross_entropy(logits, chunk[1:].to(self.dev),
                                    reduction="none").cpu()
            m = cm[start + 1:start + 1 + len(step)]
            lp += step[m].sum().item(); n += int(m.sum())
        if n == 0:
            return math.inf, 0
        return -lp / n, int(n)


def rank(candidates: list[dict], scorer: PriorScorer) -> list[Ranked]:
    """Score already-generated candidates, best first."""
    out = [Ranked(*scorer.nll(c["score"]), c.get("seed", -1), c) for c in candidates]
    return sorted(out, key=lambda r: r.nll)


def best_of_n(request: dict, n: int, scorer: PriorScorer, *, base_seed: int,
              max_time_s: float = 5.0) -> tuple[Ranked | None, list[Ranked]]:
    """Solve ``request`` under ``n`` seeds and return (winner, all_ranked).

    Distinct seeds give the solver distinct search orders, so the N fragments
    differ musically while satisfying identical constraints. Seeds that fail to
    solve are simply absent -- a request that yields fewer than 2 fragments gives
    the prior nothing to choose between, and the caller should treat that as a
    non-comparison rather than a best-of-N win.
    """
    from ..generation.solver import solve_fragment

    got: list[dict] = []
    for i in range(n):
        req = dict(request)
        req["_solver_seed"] = base_seed + i
        frag = solve_fragment(req, max_time_s=max_time_s)
        if frag:
            got.append({"score": frag, "seed": req["seed"], "request": request})
    if not got:
        return None, []
    ranked = rank(got, scorer)
    return ranked[0], ranked


# ---------------------------------------------------------------------------
# Density adjustment.
#
# Raw NLL is not a usable ranking signal on its own: measured over 200 generated
# fragments, corr(NLL, HOLD-fraction) = -0.83. The prior's favourite fragments are
# 85% held notes averaging 23 notes; its least favourite are dense, averaging 61.
# Long held notes are simply easy to predict, so ranking by raw NLL would select
# the most static music available -- reintroducing exactly the simultaneous
# stagnation the solver was revised to remove.
#
# So we rank by the residual against a density baseline: how much better this
# fragment scores than other fragments of the SAME note density. Density is a
# consequence of the request's allowed_rhythmic_values, not a mark of quality, so
# removing it costs no real signal. The baseline is quadratic in HOLD-fraction and
# is fit on the candidate pool itself, which spans the full density range.
# ---------------------------------------------------------------------------

def hold_fraction(scorer: "PriorScorer", score: dict) -> float:
    toks = scorer.tokenize(score) or []
    hold = sum(1 for t in toks if t == "HOLD")
    on = sum(1 for t in toks if t.startswith("NOTE_ON_"))
    return hold / max(1, hold + on)


def fit_density_baseline(pairs: list[tuple[float, float]]):
    """``pairs`` are (hold_fraction, nll). Returns f(hold_fraction) -> expected nll."""
    import numpy as np
    finite = [(h, v) for h, v in pairs if math.isfinite(v)]
    if len(finite) < 8:
        return lambda h: 0.0          # too little data to adjust; fall back to raw
    h = np.array([p[0] for p in finite]); v = np.array([p[1] for p in finite])
    deg = 2 if len(finite) >= 20 else 1
    coef = np.polyfit(h, v, deg)
    return lambda x: float(np.polyval(coef, x))


def rank_adjusted(candidates: list[dict], scorer: "PriorScorer",
                  baseline=None) -> list[Ranked]:
    """Rank by density-adjusted NLL (residual vs same-density expectation).

    ``baseline`` may be a function from a larger pool; otherwise it is fit on
    these candidates. Best (most negative residual) first.
    """
    scored = []
    for c in candidates:
        nll, n = scorer.nll(c["score"])
        scored.append((c, nll, n, hold_fraction(scorer, c["score"])))
    if baseline is None:
        baseline = fit_density_baseline([(h, v) for _, v, _, h in scored])
    out = []
    for c, nll, n, h in scored:
        resid = math.inf if not math.isfinite(nll) else nll - baseline(h)
        r = Ranked(resid, n, c.get("_solver_seed", c.get("seed", -1)), c)
        r.raw_nll = nll; r.hold_fraction = h        # keep both for reporting
        out.append(r)
    return sorted(out, key=lambda r: r.nll)
