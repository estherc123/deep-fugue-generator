"""Generate the hard-valid synthetic set (Spec §16).

Checkpointed and resumable (§26): fragments are appended to numbered shards and a
checkpoint records how many requests have been consumed, so an interrupted batch
continues instead of restarting. Re-running after completion is a no-op.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import yaml

from ..constraints.engine import load_profile, ruleset_hash
from .canonicalize import build_fragment_score
from .requests import build_requests
from .solver import solve_fragment
from .validate import validate_candidate

SHARD_SIZE = 1000


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def split_for_subject(subject_id: str, seed: int) -> str:
    """Deterministic split for a synthetic fragment, keyed on its parent subject.

    Everything derived from one subject lands in one split, which is what keeps
    §16.6 ("no split leakage through parent subjects") and §17.6 (a controlled
    negative shares its parent's split) true by construction rather than by check.
    """
    h = int(hashlib.sha256(f"{seed}:{subject_id}".encode()).hexdigest()[:8], 16)
    r = h % 100
    return "train" if r < 80 else ("validation" if r < 90 else "test")


def generate_valid(root: Path | None = None, *, count: int | None = None,
                   profile_name: str | None = None, resume: bool = True,
                   progress_every: int = 1000) -> dict:
    root = root or _repo_root()
    cfg = yaml.safe_load((root / "config" / "generation.yaml").read_text())
    vcfg = cfg["valid"]
    target = count or vcfg["target_count"]
    pname = profile_name or vcfg["rule_profile"]
    profile = load_profile(pname)
    rhash = ruleset_hash(profile)
    seed = vcfg["seed"]

    out_dir = root / "data" / "generated" / "valid"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.json"

    accepted: list[dict] = []
    seen_hashes: set[str] = set()
    consumed = 0
    if resume and ckpt_path.exists():
        ck = json.loads(ckpt_path.read_text())
        consumed = ck.get("requests_consumed", 0)
        seen_hashes = set(ck.get("event_hashes", []))
        accepted_count = ck.get("accepted", 0)
        if accepted_count >= target:
            return {**ck, "stage": "generate_valid", "resumed_complete": True}
    else:
        for f in out_dir.glob("shard_*.jsonl"):
            f.unlink()
        accepted_count = 0

    # Ask for headroom: not every request yields a solution, and the acceptance
    # rate moves whenever the solver's constraints change. At 1.6x the pool ran
    # dry at 9,765 of 10,000 once the harmony, rhythm and cadence constraints
    # tightened the solve rate to around 60%. Sized for the worst rate observed,
    # with margin; unused requests cost nothing because the loop stops on target.
    pool = max(int(target * 2.6), target + 2000)
    requests = build_requests(pool, cfg, seed=seed)

    stats = {"solved": 0, "no_solution": 0, "rejected": 0, "duplicates": 0}
    reject_reasons: dict[str, int] = {}
    coverage: dict[str, dict[str, int]] = {}
    shard_buf: list[dict] = []
    t0 = time.time()

    def flush():
        nonlocal shard_buf
        if not shard_buf:
            return
        idx = accepted_count // SHARD_SIZE
        path = out_dir / f"shard_{idx:04d}.jsonl"
        with path.open("a") as fh:
            for rec in shard_buf:
                fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        shard_buf = []

    i = consumed
    while accepted_count < target and i < len(requests):
        req = requests[i]
        i += 1
        solved = solve_fragment(req, max_time_s=vcfg["max_solve_time_s"],
                                workers=vcfg["solver_workers"])
        if not solved:
            stats["no_solution"] += 1
            continue
        stats["solved"] += 1
        sample_id = f"synv:{accepted_count:06d}"
        doc = build_fragment_score(req, solved, sample_id)
        v = validate_candidate(doc, req, profile, seen_hashes)
        if not v["accepted"]:
            stats["rejected"] += 1
            if v["duplicate_status"] != "unique":
                stats["duplicates"] += 1
                reject_reasons["duplicate"] = reject_reasons.get("duplicate", 0) + 1
            for hv in v["hard_violations"][:1]:
                reject_reasons[hv["rule_id"]] = reject_reasons.get(hv["rule_id"], 0) + 1
            if v["schema_errors"]:
                reject_reasons["schema"] = reject_reasons.get("schema", 0) + 1
            continue

        seen_hashes.add(v["event_hash"])
        subject_id = req["subject"]["subject_id"]
        rec = {
            # §16.5 required metadata
            "sample_id": sample_id,
            "request_id": req["request_id"],
            "seed": req["seed"],
            "ruleset_hash": rhash,
            "rule_profile": pname,
            "generator_version": cfg["generator_version"],
            "validator_version": cfg["validator_version"],
            "solver_status": solved["solver_status"],
            "solve_time_ms": solved["solve_time_ms"],
            "hard_valid": True,
            "soft_scores": v["soft_scores"],
            "source_type": "synthetic_valid",
            "parent_subject_id": subject_id,
            "split": split_for_subject(subject_id, seed),
            "event_hash": v["event_hash"],
            "coverage": req["_coverage"],
            "request": {k: x for k, x in req.items() if not k.startswith("_")},
            "score": doc,
        }
        shard_buf.append(rec)
        for axis, val in req["_coverage"].items():
            coverage.setdefault(axis, {})
            coverage[axis][str(val)] = coverage[axis].get(str(val), 0) + 1
        accepted_count += 1

        if len(shard_buf) >= 100:
            flush()
        if accepted_count % progress_every == 0:
            flush()
            ckpt_path.write_text(json.dumps({
                "requests_consumed": i, "accepted": accepted_count,
                "event_hashes": sorted(seen_hashes), "stats": stats,
                "coverage": coverage, "reject_reasons": reject_reasons,
            }, sort_keys=True))
            print(f"[generate_valid] {accepted_count}/{target} accepted "
                  f"({time.time()-t0:.0f}s)", flush=True)

    flush()
    summary = {
        "stage": "generate_valid",
        "generation_version": cfg["generation_version"],
        "rule_profile": pname,
        "ruleset_hash": rhash,
        "seed": seed,
        "target": target,
        "accepted": accepted_count,
        "requests_consumed": i,
        "stats": stats,
        "reject_reasons": dict(sorted(reject_reasons.items(), key=lambda kv: -kv[1])),
        "coverage": {k: dict(sorted(v.items())) for k, v in sorted(coverage.items())},
        "split_counts": {},
        "elapsed_s": round(time.time() - t0, 1),
        "failures": ([f"only {accepted_count} accepted, target {target} (§16.6)"]
                     if accepted_count < target else []),
    }
    counts: dict[str, int] = {}
    for shard in sorted(out_dir.glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                s = json.loads(line)["split"]
                counts[s] = counts.get(s, 0) + 1
    summary["split_counts"] = dict(sorted(counts.items()))
    ckpt_path.write_text(json.dumps({
        "requests_consumed": i, "accepted": accepted_count,
        "event_hashes": sorted(seen_hashes), "stats": stats,
        "coverage": coverage, "reject_reasons": reject_reasons}, sort_keys=True))
    (root / "manifests" / "generation_valid_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
