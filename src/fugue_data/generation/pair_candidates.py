"""Generate several hard-valid alternatives per prompt (Spec §18.3).

§18.2 requires the two candidates in a pair to be identical in subject, prefix,
formal target, length budget, rule profile, range profile and every other
generation condition -- differing only in the stochastic seed. That cannot be
harvested from the §16 set, where every request carries its own subject: it needs
a dedicated pass that solves *one* prompt repeatedly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from ..constraints.engine import load_profile, ruleset_hash
from .canonicalize import build_fragment_score
from .requests import build_requests
from .solver import solve_fragment
from .valid_runner import split_for_subject
from .validate import validate_candidate


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def generate_pair_candidates(root: Path | None = None, *, prompts: int = 600,
                             per_prompt: int | None = None,
                             progress_every: int = 100,
                             append: bool = False,
                             use_prior_motifs: bool = False,
                             motif_seed: int = 2024,
                             max_solve_time_s: float | None = None) -> dict:
    """Solve each prompt repeatedly to produce comparable alternatives.

    `append` keeps the existing candidate pool instead of replacing it. That
    matters once any pair has been labelled: `pair_id` is a hash of the two
    candidates' event hashes, so replacing the candidates changes every pair id
    and every label made against them stops matching anything. Appending lets an
    improved solver add better material without discarding human judgment, at the
    cost of a pool spanning two generator versions -- which `generator_revision`
    records, and which widens the quality range the reward model sees rather than
    narrowing it.
    """
    root = root or _repo_root()
    cfg = yaml.safe_load((root / "config" / "generation.yaml").read_text())
    pcfg = cfg["preferences"]
    per_prompt = per_prompt or pcfg["candidates_per_prompt"]
    profile = load_profile(cfg["valid"]["rule_profile"])
    rhash = ruleset_hash(profile)

    out_dir = root / "data" / "generated" / "preference_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    store = out_dir / "candidates.jsonl"

    # Reuse the §16 request grid for coverage, but each prompt is now solved
    # `per_prompt` times with different solver seeds.
    existing_rows: list[dict] = []
    existing_ids: set[str] = set()
    existing_subjects: set[str] = set()
    if append and store.exists():
        existing_rows = [json.loads(l) for l in store.read_text().splitlines()
                         if l.strip()]
        existing_ids = {r["sample_id"] for r in existing_rows}
        existing_subjects = {r["parent_subject_id"] for r in existing_rows}

    base = build_requests(prompts * 4, cfg, seed=pcfg["seed"])

    # Replace each random subject with a motif sampled from the Bach style prior.
    # Measured on 50 requests, this raised the solve rate from 42% to 62%: prior
    # motifs are stepwise and singable, and a stepwise subject leaves the solver
    # more room to harmonise than a random walk over the scale does. Fugue- and
    # chorale-derived motifs are mixed in one pool -- a listening comparison found
    # no audible difference between them.
    motif_stats = {}
    if use_prior_motifs:
        from ..modeling.best_of_n import PriorScorer
        from ..modeling.motifs import apply_motif, extract_motifs
        import random as _random
        pool = extract_motifs(PriorScorer(root), count=max(80, prompts // 4),
                              n_notes=5, seed=motif_seed)
        if not pool:
            raise RuntimeError("no motifs extracted from the prior")
        _random.Random(motif_seed).shuffle(pool)
        motif_stats = {
            "motifs": len(pool),
            "motifs_fugue_derived": sum(1 for m in pool if "wtc" in m["source"]),
        }
        base = [apply_motif(r, pool[i % len(pool)]) for i, r in enumerate(base)]
    # Both candidates in a pair share the prompt, so an open ending is identical
    # on both sides and cannot bias the comparison. It does dominate a listener's
    # impression though, so closed prompts come first: reviewer attention is
    # better spent on the counterpoint than on how the window was cut. Open
    # prompts still follow, so the labelled pool is not confined to cadences.
    base.sort(key=lambda r: 0 if r.get("cadence_target") else 1)
    rows: list[dict] = []
    stats = {"prompts_attempted": 0, "prompts_with_alternatives": 0,
             "solves": 0, "accepted": 0, "identical_rejected": 0}
    t0 = time.time()

    used = 0
    for req in base:
        if stats["prompts_with_alternatives"] >= prompts:
            break
        # Skip prompts already represented, so appending adds new material
        # instead of re-solving the same subjects.
        if append and req["subject"]["subject_id"] in existing_subjects:
            continue
        stats["prompts_attempted"] += 1
        used += 1
        seen_local: set[str] = set()
        accepted_local: list[dict] = []
        for k in range(per_prompt * 2):
            if len(accepted_local) >= per_prompt:
                break
            variant = {**req, "_solver_seed": (req["seed"] + k * 7919) % (2 ** 31)}
            solved = solve_fragment(variant,
                                    max_time_s=(max_solve_time_s
                                                or cfg["valid"]["max_solve_time_s"]),
                                    workers=cfg["valid"]["solver_workers"])
            stats["solves"] += 1
            if not solved:
                continue
            sample_id = f"synp:{stats['accepted']:06d}"
            while sample_id in existing_ids:
                stats["accepted"] += 1
                sample_id = f"synp:{stats['accepted']:06d}"
            doc = build_fragment_score(variant, solved, sample_id)
            v = validate_candidate(doc, variant, profile, seen_local)
            if not v["accepted"]:
                if v["duplicate_status"] != "unique":
                    stats["identical_rejected"] += 1
                continue
            seen_local.add(v["event_hash"])
            subject_id = req["subject"]["subject_id"]
            accepted_local.append({
                "sample_id": sample_id,
                "request_id": req["request_id"],
                "seed": variant["_solver_seed"],
                "ruleset_hash": rhash,
                "rule_profile": profile["profile"],
                "generator_version": cfg["generator_version"],
                "validator_version": cfg["validator_version"],
                "solver_status": solved["solver_status"],
                "solve_time_ms": solved["solve_time_ms"],
                "hard_valid": True,
                "soft_scores": v["soft_scores"],
                "source_type": "synthetic_preference_candidate",
                "parent_subject_id": subject_id,
                "split": split_for_subject(subject_id, cfg["valid"]["seed"]),
                "event_hash": v["event_hash"],
                "coverage": req["_coverage"],
                "generator_revision": cfg["generator_version"],
                "request": {k2: x for k2, x in req.items() if not k2.startswith("_")},
                "score": doc,
            })
            stats["accepted"] += 1
        if len(accepted_local) >= 2:
            stats["prompts_with_alternatives"] += 1
            rows.extend(accepted_local)
            if stats["prompts_with_alternatives"] % progress_every == 0:
                print(f"[pair_candidates] {stats['prompts_with_alternatives']}/{prompts} "
                      f"prompts ({time.time()-t0:.0f}s)", flush=True)

    out_rows = existing_rows + rows if append else rows
    store.write_text("".join(
        json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in out_rows))
    summary = {
        "stage": "generate_pair_candidates",
        "prompts_target": prompts,
        "candidates_per_prompt": per_prompt,
        **stats,
        "candidates_written": len(rows),
        "candidates_preserved": len(existing_rows),
        "candidates_total": len(existing_rows) + len(rows),
        "append": append,
        "use_prior_motifs": use_prior_motifs,
        **motif_stats,
        "elapsed_s": round(time.time() - t0, 1),
        "failures": ([f"only {stats['prompts_with_alternatives']} prompts with "
                      f"alternatives, wanted {prompts}"]
                     if stats["prompts_with_alternatives"] < prompts else []),
    }
    (root / "manifests" / "pair_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
