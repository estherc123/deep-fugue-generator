"""Batch export and ingest for agent-assisted pair review.

An agent cannot hear the Web Audio playback, so pairs are rendered as symbolic
text: per-voice note lists in bar:beat positions plus a bar-by-bar vertical
reduction showing what sounds together.

Two deliberate choices:

  Diagnostics are withheld. The automated diagnostic vector was already used to
  *select* these pairs, and a preference label exists to supply the signal those
  metrics cannot. An agent shown the metrics tends to restate them, which would
  make the labels worthless for training a reward model against them.

  Sides are blinded and shuffled per pair from `presentation_seed`, the same way
  the browser does it (§18.5), so a positional habit cannot leak into the labels.

Agent verdicts are written to `agent_labels.jsonl`, never to the `label` field on
the pair. That field is the human channel and stays empty until a person fills
it, because §18.6 and §27 turn on the difference between the two.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from ..normalize.rational import frac_str, parse_frac

_ALTER = {"-2": "bb", "-1": "b", "0": "", "1": "#", "2": "x"}
VALID_LABELS = ("A", "B", "tie", "both_poor", "cant_tell")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _pc_dir(root: Path) -> Path:
    return root / "data" / "generated" / "preference_candidates"


def _name(n: dict) -> str:
    return f"{n['step']}{_ALTER.get(str(n.get('alter', '0')), '')}{n['octave']}"


def _bar_beat(onset: Fraction, num: int, den: int) -> str:
    bar_len = Fraction(num * 4, den)
    beat_len = Fraction(4, den)
    bar = int(onset // bar_len) + 1
    beat = (onset % bar_len) / beat_len + 1
    return f"{bar}:{beat if beat.denominator != 1 else beat.numerator}"


def render_candidate(score: dict, meter: tuple[int, int]) -> dict:
    """One candidate as readable lines plus a vertical reduction."""
    num, den = meter
    lines = []
    for v in score["voices"]:
        notes = sorted(v["notes"], key=lambda n: parse_frac(n["onset_q"]))
        lines.append({
            "voice_id": v["voice_id"],
            "notes": [
                {"pitch": _name(n), "midi": n["midi"],
                 "at": _bar_beat(parse_frac(n["onset_q"]), num, den),
                 "dur_q": n["duration_q"]}
                for n in notes],
            "text": " ".join(
                f"{_name(n)}({_bar_beat(parse_frac(n['onset_q']), num, den)},"
                f"{n['duration_q']}q)" for n in notes),
        })

    onsets = sorted({parse_frac(n["onset_q"])
                     for v in score["voices"] for n in v["notes"]})
    vertical = []
    for t in onsets:
        sounding = []
        for v in score["voices"]:
            for n in v["notes"]:
                o = parse_frac(n["onset_q"])
                if o <= t < o + parse_frac(n["duration_q"]):
                    sounding.append((n["midi"], _name(n), v["voice_id"]))
                    break
        sounding.sort(reverse=True)
        if not sounding:
            continue
        bass = sounding[-1][0]
        vertical.append({
            "at": _bar_beat(t, num, den),
            "pitches_high_to_low": [s[1] for s in sounding],
            "intervals_above_bass": [s[0] - bass for s in sounding[:-1]],
        })
    return {"voices": lines, "vertical": vertical,
            "total_notes": sum(len(v["notes"]) for v in score["voices"])}


def export_batch(root: Path | None = None, *, offset: int = 0, limit: int = 50,
                 out_path: Path | None = None, include_labelled: bool = False) -> dict:
    root = root or _repo_root()
    d = _pc_dir(root)
    pairs = [json.loads(l) for l in (d / "pairs.jsonl").read_text().splitlines()
             if l.strip()]
    cands = {r["sample_id"]: r for r in
             (json.loads(l) for l in (d / "candidates.jsonl").read_text().splitlines()
              if l.strip())}
    done = set()
    ap = d / "agent_labels.jsonl"
    if ap.exists() and not include_labelled:
        done = {json.loads(l)["pair_id"] for l in ap.read_text().splitlines()
                if l.strip()}

    todo = [p for p in pairs if p["pair_id"] not in done][offset:offset + limit]
    items = []
    for p in todo:
        a, b = cands.get(p["candidate_a_id"]), cands.get(p["candidate_b_id"])
        if a is None or b is None:
            continue
        req = a["request"]
        meter = (req["meter"]["numerator"], req["meter"]["denominator"])
        # Blind and shuffle exactly as the browser does.
        rng = random.Random(p.get("presentation_seed") or 0)
        flip = rng.random() < 0.5
        left_key, right_key = ("B", "A") if flip else ("A", "B")
        docs = {"A": a["score"], "B": b["score"]}
        items.append({
            "pair_id": p["pair_id"],
            "prompt": {
                "voice_count": req["voice_count"],
                "meter": f"{meter[0]}/{meter[1]}",
                "key": {"tonic_pitch_class": req["key"]["tonic_pc"],
                        "mode": req["key"]["mode"]},
                "length_bars": req["length_bars"],
                "texture_profile": req["texture_profile"],
                "subject_intervals_semitones": req["subject"]["intervals"],
                "subject_rhythm_q": req["subject"]["rhythm_q"],
                "entry_plan": [{"voice_id": e["voice_id"], "onset_q": e["onset_q"],
                                "role": e["role"]} for e in req["entry_plan"]],
                "cadence_target": req.get("cadence_target"),
            },
            "candidate_1": render_candidate(docs[left_key], meter),
            "candidate_2": render_candidate(docs[right_key], meter),
            # Kept out of the agent's view; used only to un-blind on ingest.
            "_left_is": left_key, "_right_is": right_key,
        })

    out = {
        "batch_generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(items),
        "offset": offset,
        "remaining_unlabelled": max(0, len(pairs) - len(done) - offset - len(items)),
        "items": items,
    }
    out_path = out_path or (d / "agent_batch.json")
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return {"written": str(out_path), "count": len(items),
            "remaining_unlabelled": out["remaining_unlabelled"]}


def ingest(root: Path | None = None, *, verdicts_path: Path,
           batch_path: Path | None = None, reviewer: str = "agent") -> dict:
    """Ingest agent verdicts, translating screen side back to candidate identity."""
    root = root or _repo_root()
    d = _pc_dir(root)
    batch = json.loads((batch_path or (d / "agent_batch.json")).read_text())
    sides = {i["pair_id"]: (i["_left_is"], i["_right_is"]) for i in batch["items"]}

    raw = json.loads(Path(verdicts_path).read_text())
    verdicts = raw["verdicts"] if isinstance(raw, dict) else raw

    rows, errors = [], []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for v in verdicts:
        pid = v.get("pair_id")
        choice = str(v.get("choice", "")).strip()
        if pid not in sides:
            errors.append(f"{pid}: not in this batch")
            continue
        left_is, right_is = sides[pid]
        if choice in ("1", "candidate_1", "left"):
            label = left_is
        elif choice in ("2", "candidate_2", "right"):
            label = right_is
        elif choice in ("tie", "both_poor", "cant_tell"):
            label = choice
        else:
            errors.append(f"{pid}: unusable choice {choice!r}")
            continue
        conf = v.get("confidence")
        rows.append({
            "pair_id": pid,
            "label": label,
            "reviewer": reviewer,
            # The provenance that keeps this out of the human channel.
            "reviewer_kind": "agent",
            "confidence": conf if isinstance(conf, (int, float)) else None,
            "rationale": (v.get("rationale") or "")[:600] or None,
            "shown_as": choice,
            "labelled_at_utc": now,
        })

    # Keyed on (pair_id, reviewer), not pair_id alone. A shared calibration set is
    # labelled by every agent on purpose, and keying on the pair alone meant each
    # ingest overwrote the last: after four agents only the fourth's judgments
    # survived on those pairs, destroying the very cross-agent comparison the
    # shared set exists to enable. One pair can carry one label per reviewer.
    ap = d / "agent_labels.jsonl"
    existing: dict[tuple[str, str], dict] = {}
    if ap.exists():
        for line in ap.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[(r["pair_id"], r.get("reviewer") or "agent")] = r
    for r in rows:
        existing[(r["pair_id"], r["reviewer"])] = r
    ap.write_text("".join(
        json.dumps(existing[k], sort_keys=True, separators=(",", ":")) + "\n"
        for k in sorted(existing)))

    dist: dict[str, int] = {}
    for r in existing.values():
        dist[r["label"]] = dist.get(r["label"], 0) + 1
    by_reviewer: dict[str, int] = {}
    for (_, rv) in existing:
        by_reviewer[rv] = by_reviewer.get(rv, 0) + 1
    summary = {
        "stage": "ingest_agent_labels",
        "ingested_now": len(rows),
        "total_agent_labels": len(existing),
        "distinct_pairs_labelled": len({pid for (pid, _) in existing}),
        "labels_by_reviewer": dict(sorted(by_reviewer.items())),
        "label_distribution": dict(sorted(dist.items())),
        "errors": errors[:20],
        "note": ("Agent labels are stored separately from the human `label` field. "
                 "§18.6's human-preference export stays gated on human review."),
    }
    (root / "manifests" / "agent_label_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def split_batches(root: Path | None = None, *, n_batches: int = 4,
                  per_batch: int = 625, shared: int = 50,
                  out_dir: Path | None = None, seed: int = 20260728) -> dict:
    """Carve the unlabelled pool into disjoint agent batches plus a shared set.

    Three properties the split has to hold:

      Human-labelled pairs are excluded. Those already carry a judgment, and
      re-labelling them by machine would put two labels of different provenance
      on one pair.

      Batches are disjoint, so no pair is labelled twice by different agents and
      counted as two independent judgments.

      A small `shared` set goes to *every* agent. Without it there is no way to
      tell whether the four agents agree with each other, and therefore no way to
      tell whether their labels reflect the music or the reviewer. Consistency
      repeats already in the pool measure an agent against itself; this measures
      agents against each other.
    """
    root = root or _repo_root()
    d = _pc_dir(root)
    out_dir = Path(out_dir) if out_dir else (root / "data" / "generated" / "agent_batches")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [json.loads(l) for l in (d / "pairs.jsonl").read_text().splitlines()
             if l.strip()]
    human = [p for p in pairs if p.get("review_status") == "reviewed" and p.get("label")]
    done = {h["pair_id"] for h in human}
    # Also exclude pairs an agent has already judged. Without this, a re-export
    # after the pair pool changes hands agents work they have already done, and
    # the second judgment is indistinguishable from an independent one.
    ap = d / "agent_labels.jsonl"
    if ap.exists():
        done |= {json.loads(l)["pair_id"] for l in ap.read_text().splitlines()
                 if l.strip()}
    pool = [p for p in pairs if p["pair_id"] not in done]

    rng = random.Random(seed)
    order = sorted(pool, key=lambda p: p["pair_id"])
    rng.shuffle(order)

    need = n_batches * per_batch + shared
    if len(order) < need:
        raise RuntimeError(
            f"pool has {len(order)} unlabelled pairs; need {need} "
            f"({n_batches}x{per_batch} + {shared} shared)")

    shared_set = order[:shared]
    rest = order[shared:]
    written = []
    for b in range(n_batches):
        chunk = rest[b * per_batch:(b + 1) * per_batch]
        ids = {p["pair_id"] for p in chunk + shared_set}
        path = out_dir / f"agent_batch_{b + 1}.json"
        res = export_batch(root, offset=0, limit=len(pairs), out_path=path,
                           include_labelled=True)
        # export_batch renders everything; keep only this batch's pairs.
        doc = json.loads(path.read_text())
        doc["items"] = [i for i in doc["items"] if i["pair_id"] in ids]
        doc["batch"] = b + 1
        doc["count"] = len(doc["items"])
        doc["shared_calibration_pairs"] = len(shared_set)
        doc.pop("offset", None)
        doc.pop("remaining_unlabelled", None)
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        written.append({"batch": b + 1, "file": str(path), "items": len(doc["items"])})

    reserved = rest[n_batches * per_batch:]
    summary = {
        "stage": "split_agent_batches",
        "total_pairs": len(pairs),
        "human_labelled_excluded": len(human),
        "already_labelled_excluded": len(done),
        "unlabelled_pool": len(order),
        "batches": written,
        "shared_calibration_pairs": len(shared_set),
        "shared_pair_ids": [p["pair_id"] for p in shared_set],
        "reserved_for_human_review": len(reserved),
        "distinct_pairs_to_agents": n_batches * per_batch + shared,
    }
    (root / "manifests" / "agent_batch_split.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
