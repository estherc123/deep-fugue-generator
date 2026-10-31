"""Independent validation of a generated candidate (Spec §16.4).

Deliberately separate from the generator: it receives only the serialized
fragment and re-derives everything -- schema conformance, hard violations under
the requested profile, structural completeness, and duplicate status. It has no
access to the solver model, so a candidate cannot be accepted because the
generator believed it was correct.

Acceptance predicate (§16.4):
    hard_violation_count == 0
    and structure_complete
    and canonical_schema_valid
    and duplicate_status == "unique"
"""

from __future__ import annotations

from ..constraints.engine import evaluate, hard_violations
from ..constraints.fragment import Fragment
from ..validation.canonical import validate_canonical

_STRUCTURAL_RULES = {"subject_entry_identity", "answer_transformation",
                     "cadence_target_completion", "structural_target_reachable",
                     "section_length_limit"}


def validate_candidate(doc: dict, req: dict, profile: dict, seen_hashes: set) -> dict:
    schema_errors = validate_canonical(doc)

    plan = {
        "subject": req["subject"],
        "entry_plan": req["entry_plan"],
        "cadence_targets": [req["cadence_target"]] if req.get("cadence_target") else [],
        "length_bars": [req["length_bars"], req["length_bars"]],
    }
    frag = Fragment.from_score(doc, plan=plan)
    frag.tonic_pc = req["key"]["tonic_pc"]
    frag.mode = req["key"]["mode"]
    frag.range_profile = req.get("range_profile", "keyboard")

    violations = evaluate(frag, profile)
    hard = hard_violations(violations)
    structural_failures = [v for v in violations
                           if v["rule_id"] in _STRUCTURAL_RULES and v["severity"] == "hard"]

    event_hash = doc["normalization"]["core_event_hash"]
    duplicate_status = "duplicate" if event_hash in seen_hashes else "unique"

    accepted = (len(hard) == 0
                and not structural_failures
                and not schema_errors
                and duplicate_status == "unique")

    return {
        "accepted": accepted,
        "hard_violation_count": len(hard),
        "structure_complete": not structural_failures,
        "canonical_schema_valid": not schema_errors,
        "duplicate_status": duplicate_status,
        "violations": violations,
        "hard_violations": hard,
        "schema_errors": schema_errors[:5],
        "event_hash": event_hash,
        "soft_scores": _soft_scores(violations),
    }


def _soft_scores(violations: list[dict]) -> dict:
    out: dict[str, int] = {}
    for v in violations:
        if v["severity"] in ("soft", "diagnostic"):
            out[v["rule_id"]] = out.get(v["rule_id"], 0) + 1
    return dict(sorted(out.items()))
