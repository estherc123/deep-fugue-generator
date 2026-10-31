"""Validate an aligned annotation set (Spec §11.4, §21).

Acceptance criteria checked per piece:
  - JSON Schema conformance (schemas/annotation.schema.json)
  - every annotation onset lies within the score
  - every voice-specific annotation resolves to an existing voice
  - every duration is positive (point events carry null, not zero)
  - ambiguous boundaries are preserved, not collapsed
  - derived tonal function is never presented as source-annotated
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import jsonschema

from ..normalize.rational import parse_frac


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _schema() -> dict:
    return json.loads((_repo_root() / "schemas" / "annotation.schema.json").read_text())


def validate_annotations(doc: dict, score: dict) -> list[str]:
    """Return a list of validation error strings; empty means valid."""
    errors: list[str] = []

    validator = jsonschema.Draft7Validator(_schema())
    for e in sorted(validator.iter_errors(doc), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in e.path)
        errors.append(f"schema: {loc}: {e.message}")

    if doc.get("piece_id") != score.get("piece_id"):
        errors.append(
            f"piece_id mismatch: annotations {doc.get('piece_id')} vs score {score.get('piece_id')}")

    voice_ids = {v["voice_id"] for v in score.get("voices", [])}
    total = parse_frac(score.get("quarter_length_total", "0"))

    for a in doc.get("annotations", []):
        aid = a.get("annotation_id", "?")
        onset = parse_frac(a["onset_q"])

        # §11.4: every annotation onset lies within the score.
        if onset < 0 or onset > total:
            errors.append(f"onset out of bounds: {aid} at {a['onset_q']} (piece 0..{total})")

        # §11.4: every duration is positive.
        if a.get("duration_q") is not None:
            dur = parse_frac(a["duration_q"])
            if dur <= 0:
                errors.append(f"non-positive duration: {aid} = {a['duration_q']}")
            elif onset + dur > total:
                errors.append(f"span exceeds piece: {aid} ends at {onset + dur} > {total}")

        # §11.4: every voice-specific annotation resolves to an existing voice.
        if a.get("voice_id") is not None and a["voice_id"] not in voice_ids:
            errors.append(f"unresolved voice: {aid} -> {a['voice_id']}")

        # §27: derived tonal function must be labelled as derived.
        if a.get("entry_function") is not None and not a.get("entry_function_derivation"):
            errors.append(f"derived entry_function without derivation note: {aid}")

        # A rejected candidate must never be mistaken for an asserted label.
        if a.get("analytical_status") not in ("accepted", "rejected_candidate"):
            errors.append(f"unknown analytical_status: {aid} = {a.get('analytical_status')}")

    # §11.3: ambiguity is preserved -- a group id, once assigned, must have >1 member.
    groups: dict[str, int] = {}
    for a in doc.get("annotations", []):
        g = a.get("analytical_alternative_group")
        if g:
            groups[g] = groups.get(g, 0) + 1
    for g, n in sorted(groups.items()):
        if n < 2:
            errors.append(f"alternative group with a single member: {g}")

    return errors
