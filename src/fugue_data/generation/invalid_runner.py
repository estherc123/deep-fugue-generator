"""Generate the controlled single-violation set (Spec §17).

§17.2's acceptance is strict and is enforced here, not assumed:
  - the source fragment has zero hard violations;
  - the result has exactly one hard violation, and it is the intended rule;
  - no unintended hard violation came with it.

A negative always inherits its parent's split, so a positive and the negatives
derived from it never straddle a train/test boundary (§17.6).
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import yaml

from ..constraints.engine import evaluate, hard_violations, load_profile, ranges_for
from ..constraints.fragment import Fragment
from ..normalize.hashing import compute_hashes
from ..validation.canonical import validate_canonical
from .mutators import apply_mutation, registry

SHARD_SIZE = 1000


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _iter_valid(root: Path):
    d = root / "data" / "generated" / "valid"
    for shard in sorted(d.glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def _plan_of(rec: dict) -> dict:
    req = rec["request"]
    return {"subject": req["subject"], "entry_plan": req["entry_plan"],
            "cadence_targets": [req["cadence_target"]] if req.get("cadence_target") else [],
            "length_bars": [req["length_bars"], req["length_bars"]]}


def _fragment(doc: dict, rec: dict) -> Fragment:
    req = rec["request"]
    f = Fragment.from_score(doc, plan=_plan_of(rec))
    f.tonic_pc = req["key"]["tonic_pc"]
    f.mode = req["key"]["mode"]
    f.range_profile = req.get("range_profile", "keyboard")
    return f


def _content_hash(doc: dict) -> str:
    payload = json.dumps([[ (n["onset_q"], n["duration_q"], n["midi"])
                            for n in sorted(v["notes"],
                                            key=lambda n: (n["onset_q"], n["midi"]))]
                          for v in doc["voices"]], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def generate_invalid(root: Path | None = None, *, count: int | None = None,
                     progress_every: int = 1000) -> dict:
    root = root or _repo_root()
    cfg = yaml.safe_load((root / "config" / "generation.yaml").read_text())
    icfg = cfg["invalid"]
    target = count or icfg["target_count"]
    profile = load_profile(cfg["valid"]["rule_profile"])
    sev = profile["severities"]

    muts = registry()
    usable = sorted(m for m, s in muts.items()
                    if sev.get(s["target_rule_id"]) == "hard")
    skipped = {m: sev.get(s["target_rule_id"]) for m, s in sorted(muts.items())
               if sev.get(s["target_rule_id"]) != "hard"}
    if not usable:
        raise RuntimeError("no mutator targets a hard rule in this profile")

    quota = int(target * icfg["max_share_per_rule"])
    per_mutator_target = target // len(usable) + 1

    out_dir = root / "data" / "generated" / "invalid"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("shard_*.jsonl"):
        f.unlink()

    rng = random.Random(icfg["seed"])
    sources = list(_iter_valid(root))
    if not sources:
        raise RuntimeError("no hard-valid fragments to mutate; run generate_valid first")
    rng.shuffle(sources)

    accepted = 0
    per_rule: dict[str, int] = {}
    per_mutator: dict[str, int] = {}
    rejects: dict[str, int] = {}
    seen: set[str] = set()
    buf: list[dict] = []
    t0 = time.time()

    def flush():
        nonlocal buf
        if not buf:
            return
        path = out_dir / f"shard_{accepted // SHARD_SIZE:04d}.jsonl"
        with path.open("a") as fh:
            for r in buf:
                fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
        buf = []

    si = 0
    attempts = 0
    rotation = 0
    retired: set[str] = set()
    consecutive_fail: dict[str, int] = {}
    give_up_after = 200
    max_attempts = target * 12
    while accepted < target and attempts < max_attempts:
        attempts += 1
        src = sources[si % len(sources)]
        si += 1

        # Pick the mutator furthest below its share, so §17.4's balance holds by
        # construction. Ties rotate rather than always resolving the same way:
        # picking the strict minimum locks onto a mutator that never succeeds --
        # one broken mutator then consumes every remaining attempt and the run
        # ends with a handful of samples. A mutator that fails `give_up_after`
        # times in a row is retired and reported instead of stalling the batch.
        candidates = [m for m in usable
                      if per_mutator.get(m, 0) < min(per_mutator_target, quota)
                      and m not in retired]
        if not candidates:
            candidates = [m for m in usable
                          if per_mutator.get(m, 0) < quota and m not in retired]
        if not candidates:
            break
        low = min(per_mutator.get(m, 0) for m in candidates)
        tied = sorted(m for m in candidates if per_mutator.get(m, 0) == low)
        mid = tied[rotation % len(tied)]
        rotation += 1

        src_doc = src["score"]
        # §17.2 step 1: the source must itself be clean.
        src_frag = _fragment(src_doc, src)
        if hard_violations(evaluate(src_frag, profile)):
            rejects["source_not_clean"] = rejects.get("source_not_clean", 0) + 1
            continue

        req = src["request"]
        params = {**profile["params"],
                  "ranges": ranges_for(profile, req.get("range_profile")),
                  "_entry_plan": req["entry_plan"],
                  "_cadence_target": req.get("cadence_target"),
                  "_tonic_pc": req["key"]["tonic_pc"]}
        applied = apply_mutation(src_doc, mid, rng, params)
        if applied is None:
            rejects["mutation_not_applicable"] = rejects.get("mutation_not_applicable", 0) + 1
            continue
        mutated, info = applied

        pre_hash = _content_hash(src_doc)
        post_hash = _content_hash(mutated)
        if pre_hash == post_hash:
            rejects["mutation_no_effect"] = rejects.get("mutation_no_effect", 0) + 1
            continue
        if post_hash in seen:
            rejects["duplicate"] = rejects.get("duplicate", 0) + 1
            continue

        sample_id = f"syni:{accepted:06d}"
        mutated["piece_id"] = sample_id
        # Re-hash: the content changed, so the stored hashes must too.
        frag = _fragment(mutated, src)
        schema_errors = validate_canonical(mutated)
        violations = evaluate(frag, profile)
        hard = hard_violations(violations)

        intended = [v for v in hard if v["rule_id"] == info["target_rule_id"]]
        unintended = [v for v in hard if v["rule_id"] != info["target_rule_id"]]

        # §17.2 step 4: exactly one intended hard violation, nothing unintended.
        if len(intended) != 1 or unintended:
            key = ("no_intended_violation" if not intended
                   else "multiple_intended" if len(intended) > 1
                   else f"unintended:{unintended[0]['rule_id']}")
            rejects[key] = rejects.get(key, 0) + 1
            consecutive_fail[mid] = consecutive_fail.get(mid, 0) + 1
            if consecutive_fail[mid] >= give_up_after:
                retired.add(mid)
            continue
        # A notation-only mutation may legitimately fail the canonical schema; any
        # other schema failure means the mutation broke more than it intended.
        if schema_errors and info["target_rule_id"] != "timing_positive_duration":
            rejects["schema"] = rejects.get("schema", 0) + 1
            continue

        seen.add(post_hash)
        consecutive_fail[mid] = 0
        rec = {
            # §17.5 required record
            "sample_id": sample_id,
            "parent_valid_sample_id": src["sample_id"],
            "mutation_id": info["mutation_id"],
            "target_rule_id": info["target_rule_id"],
            "mutation_seed": icfg["seed"],
            "pre_validation_hash": pre_hash,
            "post_validation_hash": post_hash,
            "violations": intended,
            "source_type": "synthetic_controlled_invalid",
            # §17.6: a negative always sits in its parent's split.
            "split": src["split"],
            "parent_subject_id": src["parent_subject_id"],
            "rule_profile": profile["profile"],
            "ruleset_hash": src["ruleset_hash"],
            "mutation_detail": info.get("detail"),
            "notation_only": info["target_rule_id"] == "timing_positive_duration",
            "score": mutated,
        }
        buf.append(rec)
        per_rule[info["target_rule_id"]] = per_rule.get(info["target_rule_id"], 0) + 1
        per_mutator[mid] = per_mutator.get(mid, 0) + 1
        accepted += 1
        if len(buf) >= 100:
            flush()
        if accepted % progress_every == 0:
            flush()
            print(f"[generate_invalid] {accepted}/{target} ({time.time()-t0:.0f}s)",
                  flush=True)
    flush()

    splits: dict[str, int] = {}
    for shard in sorted(out_dir.glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                s = json.loads(line)["split"]
                splits[s] = splits.get(s, 0) + 1

    over = {r: n / max(1, accepted) for r, n in per_rule.items()
            if n / max(1, accepted) > icfg["max_share_per_rule"]}
    summary = {
        "stage": "generate_invalid",
        "target": target,
        "accepted": accepted,
        "attempts": attempts,
        "rule_profile": profile["profile"],
        "mutators_usable": usable,
        "mutators_skipped_not_hard_in_profile": skipped,
        "per_rule_counts": dict(sorted(per_rule.items(), key=lambda kv: -kv[1])),
        "per_rule_share": {r: round(n / max(1, accepted), 4)
                           for r, n in sorted(per_rule.items())},
        "max_share_per_rule": icfg["max_share_per_rule"],
        "rules_over_share": over,
        "per_mutator_counts": dict(sorted(per_mutator.items())),
        "reject_reasons": dict(sorted(rejects.items(), key=lambda kv: -kv[1])),
        "mutators_retired": sorted(retired),
        "split_counts": dict(sorted(splits.items())),
        "elapsed_s": round(time.time() - t0, 1),
        "failures": ([f"only {accepted} accepted, target {target} (§17.6)"]
                     if accepted < target else [])
                    + ([f"rules over the {icfg['max_share_per_rule']:.0%} share: {over}"]
                       if over else []),
    }
    (root / "manifests" / "generation_invalid_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
