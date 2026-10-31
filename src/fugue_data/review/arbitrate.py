"""Blind re-judging of pairs where two reviewers disagreed (Spec §18.5).

When two reviewers disagree at chance level, the disagreement alone says nothing
about which of them is measuring the music: it could equally be that one of them
is inconsistent with *itself*. So the disputed pairs are put back in front of a
reviewer with the sides reshuffled under a fresh seed and no sight of either
earlier verdict, and the new answers are scored three ways --

  against the reviewer's own earlier label  -> their self-consistency
  against the other reviewer's label        -> whether the other one was right
  neither                                   -> the pair is genuinely ambiguous

Without the self-consistency arm the result is uninterpretable, which is why the
same pairs are re-served rather than simply asking who was right.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from ..normalize.rational import parse_frac
from .render import piano_roll_svg
from .store import update


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build(root: Path | None = None, *, pair_ids: list[str], other: str,
          out: Path | None = None, seed: int = 99991) -> dict:
    root = root or _repo_root()
    d = root / "data" / "generated" / "preference_candidates"
    out = Path(out) if out else (root / "data" / "generated" / "arbitration")
    out.mkdir(parents=True, exist_ok=True)

    pairs = {json.loads(l)["pair_id"]: json.loads(l)
             for l in (d / "pairs.jsonl").read_text().splitlines() if l.strip()}
    cands = {c["sample_id"]: c for c in
             (json.loads(l) for l in (d / "candidates.jsonl").read_text().splitlines()
              if l.strip())}
    theirs = {}
    for l in (d / "agent_labels.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r["reviewer"] == other:
                theirs[r["pair_id"]] = r["label"]

    rng = random.Random(seed)
    items, key = [], []
    for pid in pair_ids:
        p = pairs.get(pid)
        if not p:
            continue
        a, b = cands.get(p["candidate_a_id"]), cands.get(p["candidate_b_id"])
        if not a or not b:
            continue
        # A fresh shuffle, independent of the app's presentation_seed, so a
        # reviewer who remembers "I preferred the left one" cannot reproduce
        # their earlier answer from position.
        flip = rng.random() < 0.5
        left, right = (b, a) if flip else (a, b)
        items.append({
            "arb_id": hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()[:10],
            "left_svg": piano_roll_svg(left["score"], parse_frac("0"),
                                       parse_frac(left["score"]["quarter_length_total"]),
                                       None, ""),
            "right_svg": piano_roll_svg(right["score"], parse_frac("0"),
                                        parse_frac(right["score"]["quarter_length_total"]),
                                        None, ""),
            "left_notes": _notes(left["score"]),
            "right_notes": _notes(right["score"]),
        })
        key.append({
            "arb_id": items[-1]["arb_id"], "pair_id": pid,
            # left/right -> which underlying candidate, so a new answer can be
            # compared with both earlier verdicts.
            "left_is": "B" if flip else "A", "right_is": "A" if flip else "B",
            "prior_human": p.get("label"), "prior_other": theirs.get(pid),
            "other_reviewer": other,
        })
    (out / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    (out / "items.json").write_text(json.dumps(items, indent=2) + "\n")
    return {"items": len(items), "dir": str(out)}


def _notes(score: dict) -> list[dict]:
    out = []
    for i, v in enumerate(score["voices"]):
        for n in v["notes"]:
            if n.get("tie") in ("continue", "stop"):
                continue
            out.append({"m": n["midi"], "t": float(parse_frac(n["onset_q"])),
                        "d": float(parse_frac(n["duration_q"])), "v": i})
    return sorted(out, key=lambda r: (r["t"], r["m"]))


def score(root: Path | None = None, *, arb_dir: Path | None = None) -> dict:
    """Score the re-judgements once they exist."""
    root = root or _repo_root()
    ad = Path(arb_dir) if arb_dir else (root / "data" / "generated" / "arbitration")
    key = {k["arb_id"]: k for k in json.loads((ad / "key.json").read_text())}
    vpath = ad / "verdicts.json"
    if not vpath.exists():
        return {"error": "no verdicts.json yet"}
    verdicts = json.loads(vpath.read_text())

    agree_self = agree_other = neither = self_hits = 0
    rows = []
    for arb_id, choice in verdicts.items():
        k = key.get(arb_id)
        if not k or choice not in ("left", "right", "tie", "both_poor"):
            continue
        resolved = (k["left_is"] if choice == "left"
                    else k["right_is"] if choice == "right" else choice)
        m_self = resolved == k["prior_human"]
        m_other = resolved == k["prior_other"]
        # Self-consistency is a property of the human alone: it must not depend on
        # whether the agent happened to agree. The three attribution buckets below
        # stay mutually exclusive (they answer "who did this re-judge side with"),
        # but on a RANDOM sample human and agent often agree, so counting those
        # as "neither" would understate self-consistency. On the max-disagreement
        # set the two are exclusive by construction and this changes nothing.
        if m_self:
            self_hits += 1
        if m_self and not m_other:
            agree_self += 1
        elif m_other and not m_self:
            agree_other += 1
        else:
            neither += 1
        rows.append({"pair_id": k["pair_id"], "rejudged": resolved,
                     "prior_human": k["prior_human"], "prior_other": k["prior_other"]})
    n = max(1, agree_self + agree_other + neither)
    return {
        "stage": "arbitration",
        "rejudged": len(rows),
        "matches_your_earlier_label": agree_self,
        "matches_the_agent_instead": agree_other,
        "matches_neither": neither,
        "self_consistency": round(self_hits / n, 3),
        "repeated_own_label": self_hits,
        "agent_vindicated": round(agree_other / n, 3),
        "reading": ("High self-consistency with low agent agreement means the agent "
                    "is measuring something else. Low self-consistency on both means "
                    "these pairs are genuinely too close to call, and the original "
                    "kappa of 0 says more about the pairs than about the agent."),
        "detail": rows,
    }
