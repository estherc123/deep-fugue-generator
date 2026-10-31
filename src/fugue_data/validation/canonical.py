"""Validate a canonical score against the JSON Schema and the §10.4 criteria.

Acceptance criteria checked per piece (Spec §10.4, §21):
  - JSON Schema conformance (schemas/canonical_score.schema.json)
  - no negative structural durations
  - no zero-duration non-grace notes
  - all MIDI values within 0..127 (no notes outside score bounds)
  - onsets non-negative
  - any monophonic overlap within a voice is *explained* (recorded as a warning)
"""

from __future__ import annotations

import json
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import jsonschema

from ..normalize.rational import parse_frac


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _schema() -> dict:
    p = _repo_root() / "schemas" / "canonical_score.schema.json"
    return json.loads(p.read_text())


def validate_canonical(doc: dict) -> list[str]:
    """Return a list of validation error strings; empty means valid."""
    errors: list[str] = []

    # 1. Schema conformance.
    validator = jsonschema.Draft7Validator(_schema())
    for e in sorted(validator.iter_errors(doc), key=lambda e: e.path):
        loc = "/".join(str(p) for p in e.path)
        errors.append(f"schema: {loc}: {e.message}")

    # 2. Structural checks over notes.
    explained_overlaps = {
        (w.get("voice_id"), w.get("onset_q"))
        for w in doc.get("warnings", [])
        if w.get("code", "").startswith("voice_")
    }
    for v in doc.get("voices", []):
        notes = v["notes"]
        prev_end = None
        prev_onset = None
        for n in notes:
            onset = parse_frac(n["onset_q"])
            dur = parse_frac(n["duration_q"])
            if dur < 0:
                errors.append(f"negative duration: {n['note_id']}")
            if dur == 0 and not n.get("grace", False):
                errors.append(f"zero-duration non-grace note: {n['note_id']}")
            if onset < 0:
                errors.append(f"negative onset: {n['note_id']}")
            if not (0 <= n["midi"] <= 127):
                errors.append(f"midi out of range: {n['note_id']} = {n['midi']}")
            # Unexplained overlap = overlap not recorded as a warning.
            if (prev_end is not None and not n.get("grace", False)
                    and onset < prev_end and onset != prev_onset):
                key = (v["voice_id"], n["onset_q"])
                if key not in explained_overlaps:
                    errors.append(
                        f"unexplained monophonic overlap: {v['voice_id']} at {n['onset_q']}")
            end = onset + dur
            if prev_end is None or end > prev_end:
                prev_end = end
            prev_onset = onset

    return errors
