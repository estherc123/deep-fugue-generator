"""Does adding agent preferences to human preferences help or hurt?

Two reward models over identical features and architecture, differing only in
what they were trained on:

    H    human preference labels only
    H+A  human labels plus every agent label

Both are scored on the SAME held-out human labels, because the question is which
better predicts a person's preference -- the only target that matters for a model
meant to please a listener. Agent pairs overlapping the test fold are dropped from
H+A's training set, or it would be scored on pairs it had effectively seen.

With only ~315 decisive human labels, a single split is noise. The comparison is
repeated over many random splits and the paired difference is reported, since the
same folds are used for both models and the fold-to-fold variation cancels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import NAMES
from .reward import build_dataset, fit, predict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def compare(root: Path | None = None, *, repeats: int = 40, test_frac: float = 0.3,
            seed: int = 0) -> dict:
    root = root or _repo_root()
    (Xa, ya, ida), (Xh, yh, idh), pairs, cands, _ = build_dataset(root)
    rng = np.random.default_rng(seed)
    nh = len(yh)
    if nh < 40:
        return {"error": f"only {nh} human decisive labels; too few to compare"}

    accH, accHA, diffs = [], [], []
    for r in range(repeats):
        perm = rng.permutation(nh)
        ntest = max(10, int(nh * test_frac))
        te, tr = perm[:ntest], perm[ntest:]
        test_ids = {idh[i] for i in te}

        mH = fit(Xh[tr], yh[tr], seed=r)
        # Agent rows whose pair is in the human test fold are excluded, so H+A is
        # not evaluated on material it trained on under a different labeller.
        keep = [i for i, p in enumerate(ida) if p not in test_ids]
        Xcomb = np.vstack([Xh[tr], Xa[keep]])
        ycomb = np.concatenate([yh[tr], ya[keep]])
        mHA = fit(Xcomb, ycomb, seed=r)

        pH = (predict(mH, Xh[te]) > 0.5).astype(float)
        pHA = (predict(mHA, Xh[te]) > 0.5).astype(float)
        a1 = float((pH == yh[te]).mean())
        a2 = float((pHA == yh[te]).mean())
        accH.append(a1); accHA.append(a2); diffs.append(a2 - a1)

    accH = np.array(accH); accHA = np.array(accHA); diffs = np.array(diffs)
    # Paired comparison: the same folds are used for both, so the per-fold
    # difference is the statistic, not the difference of the means.
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    t = float(diffs.mean() / se) if se > 0 else float("nan")

    mH_full = fit(Xh, yh, seed=99)
    keep_all = list(range(len(ya)))
    mHA_full = fit(np.vstack([Xh, Xa[keep_all]]),
                   np.concatenate([yh, ya[keep_all]]), seed=99)

    def top_weights(m, k=8):
        order = np.argsort(-np.abs(m["w"]))[:k]
        return [{"feature": NAMES[i], "weight": round(float(m["w"][i]), 3)} for i in order]

    return {
        "stage": "reward_model_comparison",
        "human_decisive_labels": int(nh),
        "agent_decisive_labels": int(len(ya)),
        "repeats": repeats,
        "test_fold_size": max(10, int(nh * test_frac)),
        "H_accuracy_mean": round(float(accH.mean()), 4),
        "H_accuracy_sd": round(float(accH.std(ddof=1)), 4),
        "HA_accuracy_mean": round(float(accHA.mean()), 4),
        "HA_accuracy_sd": round(float(accHA.std(ddof=1)), 4),
        "paired_difference_mean": round(float(diffs.mean()), 4),
        "paired_difference_se": round(float(se), 4),
        "paired_t": round(t, 2),
        "folds_where_HA_better": int((diffs > 0).sum()),
        "folds_where_H_better": int((diffs < 0).sum()),
        "chance": 0.5,
        "H_top_weights": top_weights(mH_full),
        "HA_top_weights": top_weights(mHA_full),
    }
