"""Build and store preference-pair candidates (Spec §18).

Candidates are generated automatically; labels are not. Every pair leaves this
stage `pending`, and §18.6 forbids an unreviewed pair entering a human-preference
export. Re-running preserves any label already recorded.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import jsonschema
import yaml

from ..review.store import locked

from .preferences import build_pairs, shared_conditions_hash

_HUMAN_FIELDS = ("label", "review_status", "reviewer", "review_comment",
                 "reviewed_at_utc")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _iter_candidates(root: Path):
    """Preference candidates, not the §16 set.

    The §16 fragments each carry their own subject, so no two share a prompt and
    §18.2's identical-conditions requirement can never be met from them. Pairs are
    built from the dedicated per-prompt pass instead.
    """
    p = root / "data" / "generated" / "preference_candidates" / "candidates.jsonl"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def generate_pairs(root: Path | None = None, *, target: int | None = None) -> dict:
    root = root or _repo_root()
    cfg = yaml.safe_load((root / "config" / "generation.yaml").read_text())
    pcfg = cfg["preferences"]
    target = target or pcfg["target_pairs"]      # fresh pairs to add, see below
    schema = json.loads((root / "schemas" / "preference_pair.schema.json").read_text())

    out_dir = root / "data" / "generated" / "preference_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    store = out_dir / "pairs.jsonl"

    existing: dict[str, dict] = {}
    if store.exists():
        for line in store.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["pair_id"]] = r

    records = list(_iter_candidates(root))
    if not records:
        raise RuntimeError("no preference candidates; run generate_pair_candidates first")

    # Any pair that already carries a human or agent judgment is retained
    # unconditionally, ahead of the target cap. Rebuilding the list and slicing
    # it to `target` silently dropped 127 of 254 human labels the first time this
    # ran with an enlarged candidate pool: the labelled pairs were still valid,
    # they simply fell outside the cap. Review effort is the scarcest input here
    # and must never be discarded by a truncation.
    built = build_pairs(records, cfg)
    built_by_id = {p["pair_id"]: p for p in built}
    # Retention covers agent judgments as well as human ones. Keying only on the
    # human `label` field orphaned 1,251 agent labels the last time the pool was
    # rebuilt: the pairs they referred to were unlabelled by humans, so nothing
    # protected them, and the rebuild replaced them. Any pair carrying *any*
    # judgment survives regeneration.
    agent_path = (root / "data" / "generated" / "preference_candidates"
                  / "agent_labels.jsonl")
    agent_judged: set[str] = set()
    if agent_path.exists():
        agent_judged = {json.loads(l)["pair_id"]
                        for l in agent_path.read_text().splitlines() if l.strip()}
    labelled_prior = [r for r in existing.values()
                      if r.get("label") or r.get("review_status") == "reviewed"
                      or r["pair_id"] in agent_judged]
    keep_ids = {r["pair_id"] for r in labelled_prior}
    pairs = [built_by_id.get(r["pair_id"], r) for r in labelled_prior]
    # `target` is a budget for FRESH pairs, not a cap on the total.
    #
    # It used to cap the total, which quietly made the pool un-growable: once the
    # retained (already-judged) pairs outnumbered the target, `len(pairs) >= target`
    # was true before the loop began and not one new pair was ever added. A run
    # that solved 3,435 new candidates over 3.8 hours contributed exactly nothing
    # to review. Retention must not consume the budget for new material -- the two
    # serve opposite purposes.
    added = 0
    for p_ in built:
        if added >= target:
            break
        if p_["pair_id"] not in keep_ids:
            pairs.append(p_)
            added += 1

    # §18.5: repeat a sample of pairs so reviewer consistency can be estimated.
    rng = random.Random(pcfg["seed"])
    repeats = []
    if len(pairs) >= 20:
        for p in rng.sample(pairs, max(1, len(pairs) // 20)):
            r = dict(p)
            r["pair_id"] = p["pair_id"] + ":repeat"
            r["is_consistency_repeat"] = True
            r["presentation_seed"] = rng.randrange(2 ** 31)
            repeats.append(r)
    pairs = pairs + repeats

    validator = jsonschema.Draft7Validator(schema)
    errors: list[str] = []
    for p in pairs:
        prior = existing.get(p["pair_id"])
        if prior and not p["is_consistency_repeat"]:
            for f in _HUMAN_FIELDS:
                p[f] = prior.get(f, p[f])
        for e in validator.iter_errors(p):
            errors.append(f"{p['pair_id']}: {e.message}")

    # Write under the same advisory lock the review app uses. Rebuilding the pool
    # while someone is reviewing is normal -- generation takes an hour and nobody
    # should have to stop working -- but an unlocked overwrite would silently drop
    # any label submitted during the write. Re-reading inside the lock also picks
    # up labels made after this run started.
    with locked(store):
        if store.exists():
            fresh = {}
            for line in store.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("label") or r.get("review_status") == "reviewed":
                        fresh[r["pair_id"]] = r
            for p_ in pairs:
                late = fresh.get(p_["pair_id"])
                if late and not p_.get("label"):
                    for f in _HUMAN_FIELDS:
                        p_[f] = late.get(f, p_.get(f))
            # A pair labelled after this run began, which the rebuild did not
            # produce, is still carried through rather than dropped.
            have = {p_["pair_id"] for p_ in pairs}
            for pid, r in fresh.items():
                if pid not in have:
                    pairs.append(r)
        store.write_text("".join(
            json.dumps(p, sort_keys=True, separators=(",", ":")) + "\n" for p in pairs))

    by_reason: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for p in pairs:
        by_reason[p["selection_reason"]] = by_reason.get(p["selection_reason"], 0) + 1
        by_split[p["split"]] = by_split.get(p["split"], 0) + 1
        by_status[p["review_status"]] = by_status.get(p["review_status"], 0) + 1

    labelled = sum(1 for p in pairs if p["review_status"] == "reviewed")
    summary = {
        "stage": "generate_preference_candidates",
        "fresh_pairs_target": target,
        "fresh_pairs_added": added,
        "pairs": len(pairs),
        "consistency_repeats": len(repeats),
        "prompts": len({p["prompt_id"] for p in pairs}),
        "by_selection_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
        "by_split": dict(sorted(by_split.items())),
        "by_review_status": dict(sorted(by_status.items())),
        "labelled": labelled,
        "schema_errors": errors[:10],
        "failures": ([f"only {len(pairs)} pairs, target {target} (§18.6)"]
                     if len(pairs) < min(500, target) else []) +
                    ([f"{len(errors)} schema errors"] if errors else []),
        # §18.6's labelled-pool target is a human checkpoint, reported not claimed.
        "human_checkpoint": {
            "requirement": "at least 500 reviewed pairs for the first iteration (§18.6)",
            "reviewed": labelled,
            "outstanding": max(0, 500 - labelled),
            "command": "python -m fugue_data.review.pairs_app",
        },
    }
    (root / "manifests" / "preference_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
