"""Validate split assignment and prove split isolation (Spec §13.5, §21).

These are the leakage gates. They are written to be checkable against the split
manifest alone plus the duplicate groups, so the proof does not depend on trusting
the code that produced the assignment.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import jsonschema

from ..splitting.split import assignment_hash


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _schema() -> dict:
    return json.loads((_repo_root() / "schemas" / "split_manifest.schema.json").read_text())


def validate_splits(rows: list[dict], duplicate_groups: list[dict],
                    targets_by_corpus: dict[str, dict]) -> list[str]:
    """Return a list of leakage/consistency errors; empty means isolation holds."""
    errors: list[str] = []

    validator = jsonschema.Draft7Validator(_schema())
    for r in rows:
        for e in sorted(validator.iter_errors(r), key=lambda e: list(e.path)):
            loc = "/".join(str(p) for p in e.path)
            errors.append(f"schema: {r.get('piece_id')}: {loc}: {e.message}")

    by_piece = {r["piece_id"]: r for r in rows}
    if len(by_piece) != len(rows):
        errors.append(f"duplicate piece rows: {len(rows)} rows, {len(by_piece)} pieces")

    # §13.5: no work_group_id appears in multiple splits.
    work_splits: dict[str, set] = {}
    for r in rows:
        if r.get("work_group_id"):
            work_splits.setdefault(r["work_group_id"], set()).add(r["split"])
    for wg, splits in sorted(work_splits.items()):
        if len(splits) > 1:
            errors.append(f"work_group_id in multiple splits: {wg} -> {sorted(splits)}")

    # §13.5: no exact or transposition-normalized duplicate crosses splits.
    for g in duplicate_groups:
        splits = {by_piece[m]["split"] for m in g["members"] if m in by_piece}
        if len(splits) > 1:
            errors.append(
                f"duplicate group crosses splits: {g['duplicate_group_id']} "
                f"{g['members']} -> {sorted(splits)}")

    # A split unit is the atom of assignment: all its members share one split.
    unit_splits: dict[str, set] = {}
    for r in rows:
        unit_splits.setdefault(r["split_unit_id"], set()).add(r["split"])
    for unit, splits in sorted(unit_splits.items()):
        if len(splits) > 1:
            errors.append(f"split unit crosses splits: {unit} -> {sorted(splits)}")

    # §13.4: the assignment hash must actually verify, not merely be present.
    unit_members: dict[str, list[str]] = {}
    for r in rows:
        unit_members.setdefault(r["split_unit_id"], []).append(r["piece_id"])
    for r in rows:
        expected = assignment_hash(
            r["split_seed"], r["split_unit_id"],
            unit_members[r["split_unit_id"]], r["split"])
        if r["assignment_hash"] != expected:
            errors.append(f"assignment_hash mismatch: {r['piece_id']}")

    # Target counts, measured in unique works so variants never inflate a split.
    for corpus, targets in sorted(targets_by_corpus.items()):
        works: dict[str, set] = {}
        for r in rows:
            if r["corpus"] == corpus and r.get("work_group_id"):
                works.setdefault(r["split"], set()).add(r["work_group_id"])
        for split, want in sorted(targets.items()):
            got = len(works.get(split, ()))
            if got != want:
                errors.append(f"{corpus} {split}: expected {want} works, got {got}")

    return errors
