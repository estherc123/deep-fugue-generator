"""Deterministic complete-work split assignment (Spec §13).

The unit of assignment is never a piece: it is a *split unit* -- the connected
component of works fused by the blocking deduplication levels. That is what makes
§13.1 hold, "split complete works before window extraction, transposition,
augmentation, synthetic derivation, and preference-candidate generation": every
derivative later inherits its parent's split, and no duplicate group can straddle
a boundary.

Determinism: given the same corpus, config and seed, the assignment is identical.
Ordering comes from a seeded SHA-256 over the unit id, never from Python's hash
or set iteration order.
"""

from __future__ import annotations

import hashlib
import json

SPLITS = ("train", "validation", "test")


def _seeded_rank(seed: int, unit_id: str) -> str:
    return hashlib.sha256(f"{seed}:{unit_id}".encode("utf-8")).hexdigest()


def assignment_hash(seed: int, unit_id: str, members: list[str], split: str) -> str:
    """Verifiable fingerprint of one assignment decision (§13.4)."""
    payload = json.dumps(
        {"seed": seed, "unit": unit_id, "members": sorted(members), "split": split},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_split_units(features: list[dict], group_of_piece: dict[str, str]) -> list[dict]:
    """Fuse pieces into split units by duplicate group, else by work group.

    A unit carries the set of works it covers; its `weight` is how many unique
    works it consumes from the corpus target, so a duplicate group spanning two
    BWV ids counts as two works in whichever split it lands in.
    """
    units: dict[str, dict] = {}
    for f in features:
        pid = f["piece_id"]
        key = group_of_piece.get(pid) or f"work:{f['work_group_id'] or pid}"
        u = units.setdefault(key, {
            "split_unit_id": key, "corpus": f["corpus"],
            "piece_ids": [], "work_group_ids": set(), "features": [],
        })
        u["piece_ids"].append(pid)
        if f["work_group_id"]:
            u["work_group_ids"].add(f["work_group_id"])
        u["features"].append(f)

    out = []
    for u in units.values():
        u["piece_ids"] = sorted(u["piece_ids"])
        u["work_group_ids"] = sorted(u["work_group_ids"])
        u["weight"] = max(1, len(u["work_group_ids"]))
        # A unit's stratification fields come from its representative piece, the
        # lexicographically first -- deterministic, and members are near-identical
        # by construction.
        u["features"].sort(key=lambda f: f["piece_id"])
        u["representative"] = u["features"][0]["piece_id"]
        out.append(u)
    out.sort(key=lambda u: u["split_unit_id"])
    return out


def stratum_key(features: dict, fields: list[str]) -> str:
    return "|".join(f"{k}={features.get(k)}" for k in fields)


def _interleave(strata: dict[str, list[dict]]) -> list[dict]:
    """Round-robin across strata so each stratum spreads over all splits."""
    ordered_keys = sorted(strata)
    ordered = []
    depth = max((len(v) for v in strata.values()), default=0)
    for i in range(depth):
        for k in ordered_keys:
            if i < len(strata[k]):
                ordered.append(strata[k][i])
    return ordered


def assign_splits(units: list[dict], targets: dict[str, int], stratify_by: list[str],
                  seed: int) -> list[dict]:
    """Assign units to splits, hitting `targets` exactly by work weight.

    Heavier units go first: a unit covering two works has less room to fit near
    the end, so placing it while every split still has slack is what keeps the
    totals exact rather than off by one.
    """
    strata: dict[str, list[dict]] = {}
    for u in units:
        rep = next(f for f in u["features"] if f["piece_id"] == u["representative"])
        u["stratum"] = stratum_key(rep, stratify_by)
        u["_rank"] = _seeded_rank(seed, u["split_unit_id"])
        strata.setdefault(u["stratum"], []).append(u)
    for k in strata:
        strata[k].sort(key=lambda u: u["_rank"])

    # Stable position from the interleave, then heavier units first. The position
    # is captured up front: sorting on `list.index` would compare unit dicts by
    # value, which is both quadratic and wrong when two units look alike.
    ordered = [(i, u) for i, u in enumerate(_interleave(strata))]
    ordered.sort(key=lambda iu: (-iu[1]["weight"], iu[0]))

    def deficit(split: str) -> float:
        target = targets.get(split, 0)
        if target <= 0:
            return float("-inf")
        return (target - assigned[split]) / target

    assigned = {s: 0 for s in SPLITS}
    rows = []
    for _, u in ordered:
        # Largest *proportional* deficit wins, not the largest absolute one.
        # Absolute deficit would hand the first 97 units straight to train (111 vs
        # 14 vs 14) before validation or test received anything, which would sort
        # the interleaved stratum ordering into the splits and destroy the
        # stratification it exists to provide. Filling proportionally keeps all
        # three splits advancing together, so each stratum spreads across them.
        # Ties break in a fixed split order, never on dict ordering.
        best = max(SPLITS, key=lambda s: (deficit(s), -SPLITS.index(s)))
        assigned[best] += u["weight"]
        u["split"] = best
        rows.append(u)
    return sorted(rows, key=lambda u: u["split_unit_id"])


def split_manifest_rows(units: list[dict], seed: int, corpus: str) -> list[dict]:
    """One manifest row per *piece* (§13.4), carrying its unit's assignment."""
    rows = []
    for u in units:
        ah = assignment_hash(seed, u["split_unit_id"], u["piece_ids"], u["split"])
        for f in u["features"]:
            rows.append({
                "piece_id": f["piece_id"],
                "work_group_id": f["work_group_id"],
                "corpus": corpus,
                "split": u["split"],
                "split_seed": seed,
                "split_unit_id": u["split_unit_id"],
                "split_unit_weight": u["weight"],
                "stratification_fields": {
                    k: f.get(k) for k in
                    ("book", "mode", "voice_count", "meter", "length_bucket")
                    if f.get(k) is not None
                },
                "mode_source": f.get("mode_source"),
                "assignment_hash": ah,
            })
    return sorted(rows, key=lambda r: r["piece_id"])
