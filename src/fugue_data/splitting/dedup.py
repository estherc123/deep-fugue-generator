"""Deduplication at six levels (Spec §12).

Levels, in the order the spec lists them:

  1. exact source-file checksum
  2. exact canonical event hash
  3. transposition-normalized interval hash
  4. per-voice melody hash
  5. rhythm-only hash
  6. work and movement identifiers

Near duplicates are never deleted. Matches are grouped, one member is chosen as
the preferred canonical representative, and every member keeps its provenance.
Group membership is what the split stage consumes: no duplicate group may cross
a train/validation/test boundary (§12, §13.5).

Levels 3-5 are *similarity* signals, not identity: two different pieces can share
a rhythm hash. So the level at which a group formed is recorded on the group, and
only the levels a profile marks as blocking are allowed to fuse split units.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..normalize.hashing import (_h, notes_from_dicts, voice_melodic_hash,
                                 voice_rhythm_hash)

# Level names in spec order; `blocking` levels force co-assignment to one split.
LEVELS = [
    "source_checksum",
    "canonical_event_hash",
    "transposition_melodic_hash",
    "per_voice_melody_hash",
    "rhythm_hash",
    "work_identity",
]


def piece_signatures(score: dict) -> dict:
    """The six dedup keys for one canonical score.

    Per-voice melodic hashes are recomputed here with the normalizer's own hash
    functions (see `notes_from_dicts`) because normalization folds them into the
    global melodic hash without storing them individually.
    """
    norm = score.get("normalization", {})
    per_voice_melody = {
        v["voice_id"]: voice_melodic_hash(notes_from_dicts(v["notes"]))
        for v in score["voices"]
    }
    per_voice_rhythm = {
        v["voice_id"]: voice_rhythm_hash(notes_from_dicts(v["notes"]))
        for v in score["voices"]
    }
    return {
        "piece_id": score["piece_id"],
        "work_group_id": score.get("work_group_id"),
        "movement_type": score.get("movement_type"),
        "source_checksums": sorted(
            s["sha256"] for s in score.get("_source", []) if s.get("sha256")),
        "canonical_event_hash": norm.get("core_event_hash"),
        "transposition_melodic_hash": norm.get("transposition_melodic_hash"),
        "rhythm_hash": norm.get("rhythm_hash"),
        "per_voice_melody_hashes": per_voice_melody,
        "per_voice_rhythm_hashes": per_voice_rhythm,
        # A whole-piece key folding the per-voice melody hashes in voice order:
        # catches the same music re-encoded with different voice *labels* only if
        # the set matches, so it is compared as a multiset below.
        "voice_melody_set_hash": _h(sorted(per_voice_melody.values())),
    }


def _keys_for_level(sig: dict, level: str) -> list[str]:
    if level == "source_checksum":
        return list(sig["source_checksums"])
    if level == "canonical_event_hash":
        return [sig["canonical_event_hash"]] if sig["canonical_event_hash"] else []
    if level == "transposition_melodic_hash":
        return [sig["transposition_melodic_hash"]] if sig["transposition_melodic_hash"] else []
    if level == "per_voice_melody_hash":
        # Whole-piece voice-melody identity, not a single shared line: one voice
        # in common is a motivic echo, not a duplicate piece.
        return [sig["voice_melody_set_hash"]]
    if level == "rhythm_hash":
        return [sig["rhythm_hash"]] if sig["rhythm_hash"] else []
    if level == "work_identity":
        wg = sig["work_group_id"]
        return [f"{wg}:{sig['movement_type']}"] if wg else []
    raise ValueError(f"unknown dedup level: {level}")


class _Union:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_duplicate_groups(signatures: list[dict], blocking_levels: list[str]) -> dict:
    """Group pieces that match at any *blocking* level; report all levels.

    Returns the groups plus a per-level census, so a level that is reported but
    not blocking (a shared rhythm, say) is still visible in the build report.
    """
    by_id = {s["piece_id"]: s for s in signatures}
    ids = sorted(by_id)

    level_matches: dict[str, list[list[str]]] = {}
    for level in LEVELS:
        buckets: dict[str, list[str]] = {}
        for pid in ids:
            for key in _keys_for_level(by_id[pid], level):
                buckets.setdefault(key, []).append(pid)
        level_matches[level] = [sorted(v) for v in buckets.values() if len(v) > 1]

    uf = _Union(ids)
    for level in blocking_levels:
        for members in level_matches[level]:
            for other in members[1:]:
                uf.union(members[0], other)

    clusters: dict[str, list[str]] = {}
    for pid in ids:
        clusters.setdefault(uf.find(pid), []).append(pid)

    groups = []
    for n, (root, members) in enumerate(sorted(clusters.items()), start=1):
        members = sorted(members)
        if len(members) == 1:
            continue
        matched_levels = sorted(
            level for level in LEVELS
            if any(set(members) & set(m) and len(set(members) & set(m)) > 1
                   for m in level_matches[level])
        )
        groups.append({
            "duplicate_group_id": f"dup:{n:04d}",
            "members": members,
            # Preferred representative: deterministic and explainable -- the
            # lexicographically first piece_id, which for BCFB a/b variants is
            # variant "a". Nothing is deleted; every member keeps provenance.
            "preferred_piece_id": members[0],
            "matched_levels": matched_levels,
            "work_group_ids": sorted({by_id[m]["work_group_id"] for m in members}),
        })

    return {
        "groups": groups,
        "level_census": {
            level: {
                "match_groups": len(level_matches[level]),
                "pieces_involved": len({p for g in level_matches[level] for p in g}),
                "blocking": level in blocking_levels,
            }
            for level in LEVELS
        },
        "group_of_piece": {
            m: g["duplicate_group_id"] for g in groups for m in g["members"]},
    }


def load_signatures(canonical_dir: Path) -> list[dict]:
    sigs = []
    for p in sorted(canonical_dir.glob("*.json")):
        if p.name.endswith(".invalid.json"):
            continue
        sigs.append(piece_signatures(json.loads(p.read_text())))
    return sigs
