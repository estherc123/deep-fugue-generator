"""Rule evaluation engine and profile loading (Spec §15.2-§15.4)."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import jsonschema
import yaml

from .rules import registry

PROFILES = ("species_strict", "high_baroque_conservative", "bach_empirical")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=8)
def load_profile(name: str, root: str | None = None) -> dict:
    r = Path(root) if root else _repo_root()
    if name not in PROFILES:
        raise ValueError(f"unknown rule profile {name!r}; expected one of {PROFILES}")
    prof = yaml.safe_load((r / "config" / "rules" / f"{name}.yaml").read_text())
    reg = yaml.safe_load((r / "config" / "rules" / "registry.yaml").read_text())
    prof["rules"] = {rr["rule_id"]: rr for rr in reg["rules"]}
    prof["ruleset_version"] = reg["ruleset_version"]
    prof["params"] = dict(prof.get("parameters", {}))
    prof["range_profiles"] = {
        name: {k: tuple(v) for k, v in rs.items()}
        for name, rs in (prof.get("range_profiles") or {}).items()}
    prof["default_range_profile"] = prof.get("default_range_profile", "keyboard")
    return prof


def ranges_for(profile: dict, range_profile: str | None) -> dict:
    """Ranges for a fragment's declared range profile, falling back to the default."""
    name = range_profile or profile["default_range_profile"]
    rps = profile["range_profiles"]
    return rps.get(name) or rps.get(profile["default_range_profile"]) or {}


def ruleset_hash(profile: dict) -> str:
    """Fingerprint of the exact rules and severities in force (§16.5)."""
    payload = json.dumps({
        "profile": profile["profile"],
        "ruleset_version": profile["ruleset_version"],
        "severities": profile["severities"],
        "parameters": {k: v for k, v in profile.get("parameters", {}).items()},
        "range_profiles": {n: {k: list(v) for k, v in rs.items()}
                           for n, rs in profile["range_profiles"].items()},
        "rules": {k: {"version": profile["ruleset_version"],
                      "category": v["category"], "implementation": v["implementation"]}
                  for k, v in profile["rules"].items()},
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_registry(root: Path | None = None) -> list[str]:
    """Check every rule against the rule schema and the citation requirement (§3.4)."""
    r = root or _repo_root()
    schema = json.loads((r / "schemas" / "rule.schema.json").read_text())
    reg = yaml.safe_load((r / "config" / "rules" / "registry.yaml").read_text())
    profs = {p: yaml.safe_load((r / "config" / "rules" / f"{p}.yaml").read_text())
             for p in PROFILES}
    impls = registry()
    errors: list[str] = []
    for rule in reg["rules"]:
        doc = {**rule,
               "version": reg["ruleset_version"],
               "profile_severity": {p: profs[p]["severities"].get(rule["rule_id"])
                                    for p in PROFILES}}
        for e in jsonschema.Draft7Validator(schema).iter_errors(doc):
            errors.append(f"{rule['rule_id']}: {e.message}")
        if rule["implementation"] not in impls:
            errors.append(f"{rule['rule_id']}: no implementation "
                          f"{rule['implementation']!r}")
        if not rule.get("sources"):
            errors.append(f"{rule['rule_id']}: no source cited (§3.4)")
    return errors


def evaluate(fragment, profile: dict, *, only: set[str] | None = None) -> list[dict]:
    """All violations of `fragment` under `profile`, severity attached (§15.4).

    Rules set to `off` are skipped entirely; everything else runs and is reported
    at its configured severity, so a diagnostic detection is still visible without
    blocking acceptance.
    """
    impls = registry()
    sev = profile["severities"]
    params = {**profile["params"],
              "ranges": ranges_for(profile, getattr(fragment, "range_profile", None))}
    out: list[dict] = []
    for rule_id, rule in sorted(profile["rules"].items()):
        if only and rule_id not in only:
            continue
        severity = sev.get(rule_id, "off")
        if severity == "off":
            continue
        fn = impls[rule["implementation"]]
        for v in fn(fragment, params):
            out.append({**v, "severity": severity})
    out.sort(key=lambda v: (v["onset_q"], v["rule_id"]))
    return out


def hard_violations(violations: list[dict]) -> list[dict]:
    return [v for v in violations if v["severity"] == "hard"]
