"""Agent review of proposed Bach excerpts (Spec §14.5), stored separately.

An agent verdict is NOT a human approval. §27 forbids labelling automatically
selected excerpts as human approved, and an agent is another automatic selector,
however well it reasons. So agent verdicts are written to their own file with
`reviewer_kind: "agent"` and never touch `review_status` in `excerpts.jsonl` --
exactly as agent preference labels are kept apart from human ones.

What agents are actually good for here is the half a non-specialist cannot check:
each window was proposed because a detector claimed it contains a subject entry,
a stretto, a sequence, a cadence arrival. Verifying that claim is analysis and
needs terminology. Judging whether the window stands on its own as a coherent
musical unit is an ear judgment and needs none.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from ..normalize.rational import parse_frac
from .agent_review import _bar_beat


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _excerpts_path(root: Path) -> Path:
    return root / "data" / "proposed" / "excerpts.jsonl"


def _verdicts_path(root: Path) -> Path:
    return root / "data" / "proposed" / "agent_excerpt_verdicts.jsonl"


def render_excerpt(root: Path, rec: dict) -> dict:
    """The excerpt's notes as text, with bar/beat positions an agent can reason about."""
    piece = root / "data" / "canonical" / f"{rec['piece_id']}.json"
    if not piece.exists():
        return {}
    score = json.loads(piece.read_text())
    mm = (score.get("meter_map") or [{}])[0]
    num = int(mm.get("numerator", 4)); den = int(mm.get("denominator", 4))
    lo = parse_frac(rec["start_onset_q"]); hi = parse_frac(rec["end_onset_q"])
    voices = []
    for v in score["voices"]:
        # Tie continuations are MERGED into the note that begins them, not
        # skipped. Skipping them (copied from the pairs renderer, where notes
        # arrive pre-merged) removed the sustained part of every tied note: a note
        # held across a barline showed its first segment and then an apparent
        # rest. That corrupted the rhythm an agent reads in 159 of 200 windows.
        merged = []
        for n in sorted(v["notes"], key=lambda x: parse_frac(x["onset_q"])):
            if n.get("tie") in ("continue", "stop") and merged \
                    and merged[-1]["midi"] == n["midi"]:
                merged[-1]["dur"] += parse_frac(n["duration_q"])
                continue
            merged.append({"onset": parse_frac(n["onset_q"]),
                           "dur": parse_frac(n["duration_q"]),
                           "midi": n["midi"]})
        notes = []
        for n in merged:
            o = n["onset"]
            if not (lo <= o < hi):
                continue
            notes.append({
                "at": _bar_beat(o - lo, num, den),
                "midi": n["midi"],
                "dur_q": str(n["dur"]),
                # Flag a note the window cuts through, so the reviewer knows the
                # sound continues past the boundary rather than stopping.
                **({"continues_past_end": True} if o + n["dur"] > hi else {}),
            })
        if notes:
            voices.append({"voice_id": v["voice_id"], "notes": notes})
    return {"meter": f"{num}/{den}", "voices": voices,
            "total_q": str(hi - lo)}


def export_batch(root: Path | None = None, *, offset: int = 0, limit: int = 50,
                 out_path: Path | None = None,
                 include_reviewed: bool = False) -> dict:
    root = root or _repo_root()
    recs = [json.loads(l) for l in _excerpts_path(root).read_text().splitlines()
            if l.strip()]
    # Only windows the automatic pass believed were clean are worth reviewing;
    # the rest were already rejected for tied boundaries, cuts through entries,
    # or voice-identity failures (§14.3).
    recs = [r for r in recs if not r.get("auto_rejected")]
    done = set()
    vp = _verdicts_path(root)
    if vp.exists() and not include_reviewed:
        done = {json.loads(l)["excerpt_id"] for l in vp.read_text().splitlines()
                if l.strip()}
    todo = [r for r in recs if r["excerpt_id"] not in done][offset:offset + limit]

    items = []
    for r in todo:
        items.append({
            "excerpt_id": r["excerpt_id"],
            "piece_id": r["piece_id"],
            "bars": r.get("bars"),
            "measures": f"{r.get('start_measure')}-{r.get('end_measure')}",
            # The claim under test. The agent is asked whether it holds.
            "claimed_reasons": r.get("proposal_reasons") or [],
            "clipped_boundary_notes": r.get("clipped_boundary_notes"),
            "music": render_excerpt(root, r),
        })
    out = {
        "batch_generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(items), "offset": offset,
        "remaining_unreviewed": max(0, len(recs) - len(done) - offset - len(items)),
        "items": items,
    }
    out_path = out_path or (root / "data" / "proposed" / "agent_excerpt_batch.json")
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return {"written": str(out_path), "count": len(items),
            "remaining_unreviewed": out["remaining_unreviewed"]}


_DECISIONS = {"approved", "rejected", "correction_required"}


def ingest(root: Path | None = None, *, verdicts_path: Path,
           reviewer: str = "agent") -> dict:
    """Record agent verdicts. Never writes `review_status` in excerpts.jsonl."""
    root = root or _repo_root()
    raw = json.loads(Path(verdicts_path).read_text())
    rows = raw if isinstance(raw, list) else raw.get("verdicts") or raw.get("items") or []
    valid_ids = {json.loads(l)["excerpt_id"]
                 for l in _excerpts_path(root).read_text().splitlines() if l.strip()}

    out, skipped = [], {"unknown_excerpt": 0, "bad_decision": 0}
    for r in rows:
        eid = r.get("excerpt_id", "")
        if eid not in valid_ids:
            skipped["unknown_excerpt"] += 1
            continue
        dec = (r.get("decision") or r.get("status") or "").strip()
        if dec not in _DECISIONS:
            skipped["bad_decision"] += 1
            continue
        out.append({
            "excerpt_id": eid,
            "decision": dec,
            "reason_claims_hold": r.get("reason_claims_hold"),
            "tags": r.get("tags") or [],
            "comment": (r.get("comment") or "")[:500],
            "reviewer": r.get("reviewer") or reviewer,
            "reviewer_kind": "agent",
            "reviewed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    vp = _verdicts_path(root)
    # Keyed on (excerpt_id, reviewer): two agents may review the same excerpt, and
    # collapsing them would destroy the agreement signal that comparison exists for.
    existing: dict[tuple, dict] = {}
    if vp.exists():
        for l in vp.read_text().splitlines():
            if l.strip():
                e = json.loads(l)
                existing[(e["excerpt_id"], e.get("reviewer") or "agent")] = e
    for r in out:
        existing[(r["excerpt_id"], r["reviewer"])] = r
    vp.write_text("".join(json.dumps(v, sort_keys=True, separators=(",", ":")) + "\n"
                          for v in existing.values()))

    by_rev: dict[str, int] = {}
    for v in existing.values():
        by_rev[v["reviewer"]] = by_rev.get(v["reviewer"], 0) + 1
    approved = sum(1 for v in existing.values() if v["decision"] == "approved")
    return {
        "ingested": len(out), "skipped": skipped,
        "verdicts_total": len(existing),
        "by_reviewer": dict(sorted(by_rev.items())),
        "agent_approved": approved,
        "note": ("Agent verdicts do not satisfy the §14.6 human-approval "
                 "criterion; they are advisory and stored separately."),
    }


def split_batches(root: Path | None = None, *, n_batches: int = 4,
                  per_batch: int = 50, out_dir: Path | None = None) -> dict:
    root = root or _repo_root()
    out_dir = Path(out_dir) if out_dir else (root / "data" / "proposed" / "agent_batches")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(n_batches):
        p = out_dir / f"agent_excerpt_batch_{i+1}.json"
        r = export_batch(root, offset=i * per_batch, limit=per_batch, out_path=p)
        written.append({"batch": i + 1, **r})
    return {"batches": written, "dir": str(out_dir)}
