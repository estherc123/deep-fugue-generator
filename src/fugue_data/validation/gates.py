"""Pipeline-wide validation gates (Spec §21).

Each gate in §21's list is a named check that returns failures. They are written
to read the artefacts on disk, not the in-memory state of the stage that produced
them, so a gate can catch a stage that reported success but wrote something else.

The pipeline must fail on any of these.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import jsonschema
import yaml

from ..normalize.rational import parse_frac


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _jsonl(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def _shards(d: Path):
    for shard in sorted(d.glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def run_gates(root: Path | None = None) -> dict:
    root = root or _repo_root()
    results: list[dict] = []

    def gate(name: str, spec: str, failures: list[str], checked: int = 0):
        results.append({"gate": name, "spec": spec, "checked": checked,
                        "failures": failures[:10],
                        "failure_count": len(failures),
                        "passed": not failures})

    # ---- provenance, revisions, checksums ---------------------------------- #
    src_rows = list(_jsonl(root / "manifests" / "sources.jsonl"))
    f = [f"{r.get('relative_source_path')}: missing provenance"
         for r in src_rows
         if not (r.get("repository_commit") and r.get("sha256")
                 and r.get("retrieved_at_utc"))]
    gate("source_provenance", "§21 missing source provenance", f, len(src_rows))

    cfg_sources = yaml.safe_load((root / "config" / "sources.yaml").read_text())
    pinned = {s["source_name"]: s["repository_commit"] for s in cfg_sources["sources"]}
    f = [f"{r['source_name']}: manifest {r['repository_commit'][:10]} != pinned"
         for r in src_rows if pinned.get(r["source_name"]) != r["repository_commit"]]
    gate("source_revision", "§21 unexpected source revision", f, len(src_rows))

    # Checksums are re-verified against the bytes on disk, not trusted.
    import hashlib
    mismatches = []
    raw = root / "data" / "raw"
    sample = src_rows[::37]          # every 37th file: a spread across all sources
    for r in sample:
        p = raw / r["source_name"] / r["relative_source_path"]
        if not p.exists():
            mismatches.append(f"{r['relative_source_path']}: file missing")
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]:
            mismatches.append(f"{r['relative_source_path']}: checksum mismatch")
    gate("checksums", "§21 checksum mismatch", mismatches, len(sample))

    # ---- canonical schema + structural invariants --------------------------- #
    canon_schema = json.loads(
        (root / "schemas" / "canonical_score.schema.json").read_text())
    cv = jsonschema.Draft7Validator(canon_schema)
    schema_failures, duration_failures, hash_failures = [], [], []
    canonical = root / "data" / "canonical"
    files = sorted(p for p in canonical.glob("*.json")
                   if not p.name.endswith(".invalid.json"))
    from ..normalize.hashing import notes_from_dicts, voice_event_hash, _h
    for p in files:
        d = json.loads(p.read_text())
        if next(cv.iter_errors(d), None):
            schema_failures.append(f"{d['piece_id']}: schema")
        for v in d["voices"]:
            for n in v["notes"]:
                if parse_frac(n["duration_q"]) <= 0 and not n.get("grace"):
                    duration_failures.append(f"{d['piece_id']}:{n['note_id']}")
        per = {v["voice_id"]: voice_event_hash(notes_from_dicts(v["notes"]))
               for v in d["voices"]}
        if _h([f"{k}:{per[k]}" for k in sorted(per)]) != d["normalization"]["core_event_hash"]:
            hash_failures.append(d["piece_id"])
    gate("canonical_schema", "§21 schema failure", schema_failures, len(files))
    gate("structural_duration", "§21 invalid or zero structural duration",
         duration_failures, len(files))
    gate("canonical_event_hash_stable", "§21 canonical event hash changing unexpectedly",
         hash_failures, len(files))

    # ---- annotations -------------------------------------------------------- #
    ann_failures, voice_failures = [], []
    ann_dir = root / "data" / "annotations"
    n_ann = 0
    for p in sorted(ann_dir.glob("*.json")):
        if p.name.endswith(".invalid.json"):
            continue
        d = json.loads(p.read_text())
        score = json.loads((canonical / f"{d['piece_id']}.json").read_text())
        total = parse_frac(score["quarter_length_total"])
        vids = {v["voice_id"] for v in score["voices"]}
        for a in d["annotations"]:
            n_ann += 1
            o = parse_frac(a["onset_q"])
            end = o + (parse_frac(a["duration_q"]) if a["duration_q"] else Fraction(0))
            if o < 0 or end > total:
                ann_failures.append(f"{a['annotation_id']}: outside piece")
            if a["voice_id"] is not None and a["voice_id"] not in vids:
                voice_failures.append(f"{a['annotation_id']}: unresolved voice")
    gate("annotation_bounds", "§21 annotation outside piece bounds", ann_failures, n_ann)
    gate("annotation_voice", "§21 unresolved annotation voice", voice_failures, n_ann)

    # ---- splits, duplicates, derivatives ------------------------------------ #
    split_rows = list(_jsonl(root / "manifests" / "splits.jsonl"))
    split_of = {r["piece_id"]: r["split"] for r in split_rows}
    dup_path = root / "manifests" / "duplicate_groups.json"
    groups = json.loads(dup_path.read_text())["groups"] if dup_path.exists() else []
    f = [f"{g['duplicate_group_id']} spans {sorted({split_of[m] for m in g['members'] if m in split_of})}"
         for g in groups
         if len({split_of[m] for m in g["members"] if m in split_of}) > 1]
    gate("duplicate_group_split", "§21 duplicate group crossing splits", f, len(groups))

    # Every derivative inherits its parent's split.
    deriv_failures = []
    valid_split = {}
    n_valid = 0
    for rec in _shards(root / "data" / "generated" / "valid"):
        n_valid += 1
        valid_split[rec["sample_id"]] = rec["split"]
    n_invalid = 0
    for rec in _shards(root / "data" / "generated" / "invalid"):
        n_invalid += 1
        parent = rec["parent_valid_sample_id"]
        if parent in valid_split and valid_split[parent] != rec["split"]:
            deriv_failures.append(
                f"{rec['sample_id']}: {rec['split']} != parent {valid_split[parent]}")
    gate("derivative_split_inheritance", "§21 derivative crossing parent split",
         deriv_failures, n_invalid)

    # Excerpts inherit their parent piece's split.
    ex_failures = []
    n_ex = 0
    for r in _jsonl(root / "data" / "proposed" / "excerpts.jsonl"):
        if r.get("auto_rejected"):
            continue
        n_ex += 1
        parent = split_of.get(r["piece_id"])
        if parent and r.get("split") and r["split"] != parent:
            ex_failures.append(f"{r['excerpt_id']}: {r['split']} != parent {parent}")
    gate("excerpt_split_inheritance", "§13.5 no excerpt crosses its parent split",
         ex_failures, n_ex)

    # ---- generated material ------------------------------------------------- #
    f = [f"{rec['sample_id']}: hard_valid false"
         for rec in _shards(root / "data" / "generated" / "valid")
         if not rec.get("hard_valid")]
    gate("synthetic_positive_valid", "§21 synthetic positive with a hard violation",
         f, n_valid)

    neg_failures = []
    for rec in _shards(root / "data" / "generated" / "invalid"):
        vs = rec.get("violations") or []
        if len(vs) != 1:
            neg_failures.append(f"{rec['sample_id']}: {len(vs)} violations, expected 1")
        elif vs[0]["rule_id"] != rec["target_rule_id"]:
            neg_failures.append(f"{rec['sample_id']}: violation is not the target rule")
    gate("controlled_negative_single_violation",
         "§21 controlled negative with zero or multiple unintended hard violations",
         neg_failures, n_invalid)

    # ---- preference pairs ---------------------------------------------------- #
    from ..generation.preferences import shared_conditions_hash
    cand = {}
    cpath = root / "data" / "generated" / "preference_candidates" / "candidates.jsonl"
    for r in _jsonl(cpath):
        cand[r["sample_id"]] = r
    pair_failures, pending_as_reviewed = [], []
    n_pairs = 0
    for p in _jsonl(root / "data" / "generated" / "preference_candidates" / "pairs.jsonl"):
        n_pairs += 1
        a, b = cand.get(p["candidate_a_id"]), cand.get(p["candidate_b_id"])
        if a and b:
            ha = shared_conditions_hash(a["request"])
            hb = shared_conditions_hash(b["request"])
            if ha != hb or ha != p["shared_conditions_hash"]:
                pair_failures.append(f"{p['pair_id']}: prompt conditions differ")
        if p["review_status"] == "reviewed" and p["label"] is None:
            pending_as_reviewed.append(f"{p['pair_id']}: reviewed with no label")
    gate("preference_prompt_identity", "§21 preference pair with mismatched prompt conditions",
         pair_failures, n_pairs)

    # ---- pending review never presented as reviewed -------------------------- #
    exp = root / "data" / "exports"
    pending_failures = list(pending_as_reviewed)
    pref_dir = exp / "preferences"
    if pref_dir.exists():
        for f_ in sorted(pref_dir.glob("*_pairs.jsonl")):
            for r in _jsonl(f_):
                if r.get("review_status") != "reviewed" or r.get("label") is None:
                    pending_failures.append(f"{f_.name}: unreviewed pair exported")
    style_dir = exp / "style"
    if style_dir.exists():
        import zstandard as zstd
        approved_ids = {r["excerpt_id"] for r in
                        _jsonl(root / "data" / "proposed" / "excerpts.jsonl")
                        if r.get("review_status") == "approved"}
        for f_ in sorted(style_dir.glob("*.jsonl.zst")):
            raw_bytes = zstd.ZstdDecompressor().decompress(f_.read_bytes())
            for line in raw_bytes.decode().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("source_type", "").endswith("human_approved") \
                        and r["id"] not in approved_ids:
                    pending_failures.append(f"{f_.name}: {r['id']} not human approved")
    gate("pending_review_not_exported", "§21 `pending` review included as reviewed",
         pending_failures, n_pairs + n_ex)

    # ---- tokenizer and ruleset versions -------------------------------------- #
    from ..tokenization.vocab import build_vocabulary, vocabulary_hash
    tcfg = yaml.safe_load((root / "config" / "tokenizer.yaml").read_text())
    vhash = vocabulary_hash(build_vocabulary())
    tok_failures = []
    vocab_file = root / "data" / "tokenized" / "vocabulary.json"
    if vocab_file.exists():
        v = json.loads(vocab_file.read_text())
        if v["vocabulary_hash"] != vhash:
            tok_failures.append("stored vocabulary hash != rebuilt vocabulary hash")
        if v["tokenizer_version"] != tcfg["tokenizer_version"]:
            tok_failures.append("tokenizer version mismatch")
    else:
        tok_failures.append("no tokenizer vocabulary written")
    gate("tokenizer_version", "§21 unknown tokenizer version", tok_failures, 1)

    # Every emitted token must be in the declared vocabulary. The hash check above
    # only proves the vocabulary file matches the builder -- it says nothing about
    # what the tokenizer actually wrote. Five pieces were emitting BEAT_17 and
    # beyond, tokens no model could ever embed, and every existing gate passed.
    oov: list[str] = []
    checked_tokens = 0
    tokdir = root / "data" / "tokenized"
    declared = set()
    if vocab_file.exists():
        declared = set(json.loads(vocab_file.read_text())["tokens"])
    if declared:
        for f in sorted(tokdir.glob("*.json")):
            if f.name == "vocabulary.json":
                continue
            d = json.loads(f.read_text())
            toks = (d.get("grid") or {}).get("tokens") or []
            checked_tokens += len(toks)
            missing = sorted({x for x in toks if x not in declared})
            if missing:
                oov.append(f"{d['piece_id']}: {missing[:4]}")
    gate("tokens_in_vocabulary", "§19.4 emitted token outside the declared vocabulary",
         oov, checked_tokens)

    from ..constraints.engine import PROFILES, load_profile, ruleset_hash
    known = {ruleset_hash(load_profile(p)) for p in PROFILES}
    rule_failures = [f"{rec['sample_id']}: unknown ruleset hash"
                     for rec in _shards(root / "data" / "generated" / "valid")
                     if rec["ruleset_hash"] not in known]
    gate("ruleset_profile_hash", "§21 unknown rule-profile hash", rule_failures, n_valid)

    summary = {
        "stage": "validate",
        "gates": results,
        "gates_total": len(results),
        "gates_passed": sum(1 for r in results if r["passed"]),
        "gates_failed": sum(1 for r in results if not r["passed"]),
        "failures": [f"{r['gate']}: {r['failure_count']} failures"
                     for r in results if not r["passed"]],
    }
    (root / "manifests" / "validation_gates.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
