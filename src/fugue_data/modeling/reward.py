"""A Bradley-Terry preference model over fragment features.

Trained on pairwise judgments: the feature difference between the two candidates
predicts which was preferred. Logistic regression with L2, fitted by gradient
descent -- small enough not to need a framework, and transparent enough that the
learned weights can be read as "this model thinks more of X is better", which is
the interesting output here.

The point of the experiment is not the accuracy on held-out labels from the same
reviewer -- that only shows the reviewer is self-consistent. It is whether a model
trained on ONE reviewer's labels predicts a DIFFERENT reviewer's preferences. That
is what says the labels captured something about the music rather than about the
labeller.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .features import NAMES, features


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_dataset(root: Path | None = None):
    root = root or _repo_root()
    d = root / "data" / "generated" / "preference_candidates"
    pairs = {json.loads(l)["pair_id"]: json.loads(l)
             for l in (d / "pairs.jsonl").read_text().splitlines() if l.strip()}
    cands = {c["sample_id"]: c for c in
             (json.loads(l) for l in (d / "candidates.jsonl").read_text().splitlines()
              if l.strip())}
    feat: dict[str, np.ndarray] = {}

    def f(sid):
        if sid not in feat:
            feat[sid] = np.array(features(cands[sid]["score"]), dtype=float)
        return feat[sid]

    agent: dict[str, str] = {}
    ap = d / "agent_labels.jsonl"
    if ap.exists():
        for line in ap.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["label"] in ("A", "B"):
                    agent[r["pair_id"]] = r["label"]

    human = {p: v["label"] for p, v in pairs.items()
             if v.get("label") in ("A", "B") and v["review_status"] == "reviewed"}

    def rows(labels):
        X, y, ids = [], [], []
        for pid, lab in labels.items():
            p = pairs.get(pid)
            if not p or p["candidate_a_id"] not in cands or p["candidate_b_id"] not in cands:
                continue
            X.append(f(p["candidate_a_id"]) - f(p["candidate_b_id"]))
            y.append(1.0 if lab == "A" else 0.0)
            ids.append(pid)
        return np.array(X), np.array(y), ids

    return rows(agent), rows(human), pairs, cands, feat


def fit(X, y, *, l2=1.0, iters=4000, lr=0.08, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = rng.normal(0, 0.01, Xs.shape[1])
    n = len(y)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w)))
        grad = Xs.T @ (p - y) / n + l2 * w / n
        w -= lr * grad
    return {"w": w, "mu": mu, "sd": sd}


def predict(model, X):
    Xs = (X - model["mu"]) / model["sd"]
    return 1.0 / (1.0 + np.exp(-(Xs @ model["w"])))


def accuracy(model, X, y):
    if len(y) == 0:
        return float("nan")
    return float(((predict(model, X) > 0.5) == (y > 0.5)).mean())


def run(root: Path | None = None, *, folds: int = 5, seed: int = 0) -> dict:
    root = root or _repo_root()
    (Xa, ya, ida), (Xh, yh, idh), pairs, cands, feat = build_dataset(root)
    overlap = set(ida) & set(idh)

    # Train only on agent pairs the human never judged, so the human evaluation
    # is genuinely held out rather than partly memorised.
    keep = [i for i, p in enumerate(ida) if p not in overlap]
    Xtr, ytr = Xa[keep], ya[keep]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    cv = []
    for k in range(folds):
        te = idx[k::folds]
        tr = np.array([i for i in idx if i not in set(te)])
        m = fit(Xtr[tr], ytr[tr])
        cv.append(accuracy(m, Xtr[te], ytr[te]))

    model = fit(Xtr, ytr)
    ov_idx = [i for i, p in enumerate(idh) if p in overlap]
    acc_human_overlap = accuracy(model, Xh[ov_idx], yh[ov_idx]) if ov_idx else float("nan")
    acc_human_all = accuracy(model, Xh, yh)

    # Same model architecture trained on the human labels, as a reference point.
    hm = fit(Xh, yh) if len(yh) > 30 else None
    idxh = rng.permutation(len(yh))
    cvh = []
    for k in range(folds):
        te = idxh[k::folds]
        tr = np.array([i for i in idxh if i not in set(te)])
        if len(tr) < 20:
            continue
        m = fit(Xh[tr], yh[tr])
        cvh.append(accuracy(m, Xh[te], yh[te]))

    order = np.argsort(-np.abs(model["w"]))
    weights = [{"feature": NAMES[i], "weight": round(float(model["w"][i]), 3)}
               for i in order[:10]]

    return {
        "stage": "reward_model",
        "train_pairs_agent": int(len(ytr)),
        "eval_pairs_human_heldout": len(ov_idx),
        "eval_pairs_human_all": int(len(yh)),
        "agent_cv_accuracy": round(float(np.mean(cv)), 4),
        "human_accuracy_on_overlap": round(float(acc_human_overlap), 4),
        "human_accuracy_all": round(float(acc_human_all), 4),
        "human_model_cv_accuracy": (round(float(np.mean(cvh)), 4) if cvh else None),
        "chance": 0.5,
        "top_weights": weights,
    }
