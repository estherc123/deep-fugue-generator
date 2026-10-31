"""Reviewer agreement analysis for preference labels (Spec §18.5).

§18.5 requires repeating some pairs "to estimate reviewer consistency". That
covers a reviewer against itself. With several reviewers there is a second
question -- whether they agree with *each other* -- and a third that matters more
than either: whether the machine reviewers agree with the human.

Raw agreement is not enough on its own. Four reviewers who all answer "B" most of
the time will agree often by chance, so Cohen's kappa is reported alongside it,
correcting for each pair of reviewers' own marginals.
"""

from __future__ import annotations

import collections
import itertools
import json
from pathlib import Path

LABELS = ("A", "B", "tie", "both_poor", "cant_tell")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _kappa(a: dict, b: dict, pairs: list[str]) -> tuple[float, float, int]:
    both = [p for p in pairs if p in a and p in b]
    if not both:
        return float("nan"), float("nan"), 0
    n = len(both)
    po = sum(1 for p in both if a[p] == b[p]) / n
    ca = collections.Counter(a[p] for p in both)
    cb = collections.Counter(b[p] for p in both)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in LABELS)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return po, kappa, n


def analyse(root: Path | None = None) -> dict:
    root = root or _repo_root()
    d = root / "data" / "generated" / "preference_candidates"

    agent_rows = []
    ap = d / "agent_labels.jsonl"
    if ap.exists():
        agent_rows = [json.loads(l) for l in ap.read_text().splitlines() if l.strip()]

    human: dict[str, str] = {}
    human_by: dict[str, str] = {}
    for line in (d / "pairs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        if p.get("review_status") == "reviewed" and p.get("label"):
            human[p["pair_id"]] = p["label"]
            human_by[p["pair_id"]] = p.get("reviewer") or "human"

    by_rev: dict[str, dict[str, str]] = collections.defaultdict(dict)
    conf: dict[str, list[float]] = collections.defaultdict(list)
    for r in agent_rows:
        by_rev[r["reviewer"]][r["pair_id"]] = r["label"]
        if r.get("confidence") is not None:
            conf[r["reviewer"]].append(r["confidence"])
    if human:
        by_rev["human"] = human

    reviewers = sorted(by_rev)
    dist = {}
    for rv in reviewers:
        c = collections.Counter(by_rev[rv].values())
        n = max(1, sum(c.values()))
        dist[rv] = {
            "n": sum(c.values()),
            **{k: round(c[k] / n, 4) for k in LABELS},
            "mean_confidence": (round(sum(conf[rv]) / len(conf[rv]), 3)
                                if conf.get(rv) else None),
        }

    split = root / "manifests" / "agent_batch_split.json"
    shared = (json.loads(split.read_text())["shared_pair_ids"]
              if split.exists() else [])

    pairwise = {}
    for a, b in itertools.combinations(reviewers, 2):
        common = sorted(set(by_rev[a]) & set(by_rev[b]))
        po, k, n = _kappa(by_rev[a], by_rev[b], common)
        if n:
            pairwise[f"{a} vs {b}"] = {
                "overlap": n, "raw_agreement": round(po, 4),
                "cohens_kappa": round(k, 4)}

    agent_only = [r for r in reviewers if r != "human"]
    ks = [v["cohens_kappa"] for kx, v in pairwise.items()
          if "human" not in kx and v["overlap"]]
    human_ks = [v["cohens_kappa"] for kx, v in pairwise.items() if "human" in kx]

    unanimity = {}
    if shared and len(agent_only) > 1:
        full = [p for p in shared if all(p in by_rev[a] for a in agent_only)]
        c = collections.Counter(
            len({by_rev[a][p] for a in agent_only}) for p in full)
        unanimity = {f"{k}_distinct_labels": v for k, v in sorted(c.items())}
        unanimity["pairs_compared"] = len(full)

    summary = {
        "stage": "reviewer_agreement",
        "reviewers": reviewers,
        "label_distribution": dist,
        "shared_calibration_pairs": len(shared),
        "pairwise": pairwise,
        "mean_agent_kappa": round(sum(ks) / len(ks), 4) if ks else None,
        "agent_unanimity_on_shared": unanimity,
        "human_vs_agent_overlap": sum(
            v["overlap"] for kx, v in pairwise.items() if "human" in kx),
        "mean_human_agent_kappa": (round(sum(human_ks) / len(human_ks), 4)
                                   if human_ks else None),
        "interpretation": {
            "kappa_scale": "0.0-0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, "
                           "0.6-0.8 substantial, 0.8-1.0 near perfect",
        },
    }
    if not human_ks:
        summary["warnings"] = [
            "No pair carries both a human and an agent label, so agent quality "
            "cannot be checked against human judgment. Agent batches excluded "
            "human-labelled pairs to avoid double-labelling, which also removed "
            "every point of comparison. Give one agent a sample of the "
            "human-labelled pairs to close this."]
    (root / "manifests" / "reviewer_agreement.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
