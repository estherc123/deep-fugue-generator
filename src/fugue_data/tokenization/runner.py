"""Tokenize the canonical corpus into model views (Spec §19).

Every emitted sequence records tokenizer name, version, vocabulary hash,
quantization policy and the source canonical event hash (§19.4), so a training
set can always be traced back to the exact canonical content it came from.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import yaml

from ..normalize.rational import parse_frac
from .vocab import build_vocabulary, vocabulary_hash
from .views import (GridUnsupported, choose_denominator, event_records,
                    grid_tokens, merge_tied)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def decode_grid(tokens: list[str], voice_count: int, steps_per_quarter: int) -> dict:
    """Reconstruct (voice -> [(onset_q, midi)]) from a grid sequence.

    Used to prove the grid view is lossless for the pieces it claims to support:
    what comes back must match the canonical attacks exactly (§19.2).
    """
    step = Fraction(1, steps_per_quarter)
    out: dict[str, list[tuple[Fraction, int]]] = {
        f"voice_{i}": [] for i in range(1, voice_count + 1)}
    s = -1
    voice = None
    seen_voice_this_step = 0
    for t in tokens:
        if t.startswith("VOICE_"):
            voice = t
            seen_voice_this_step += 1
            if seen_voice_this_step == 1:
                s += 1
            if seen_voice_this_step == voice_count:
                seen_voice_this_step = 0
        elif t.startswith("NOTE_ON_") and voice is not None:
            vid = f"voice_{int(voice.split('_')[1])}"
            out[vid].append((s * step, int(t.split("_")[-1])))
    return out


def tokenize_corpus(root: Path | None = None, *, limit: int | None = None) -> dict:
    root = root or _repo_root()
    cfg = yaml.safe_load((root / "config" / "tokenizer.yaml").read_text())
    pipeline = yaml.safe_load((root / "config" / "pipeline.yaml").read_text())
    canonical_dir = root / pipeline["paths"]["canonical"]
    ann_dir = root / pipeline["paths"]["annotations"]
    out_dir = root / "data" / "tokenized"
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = build_vocabulary()
    vhash = vocabulary_hash(vocab)
    (out_dir / "vocabulary.json").write_text(json.dumps(
        {"tokenizer_name": cfg["tokenizer_name"],
         "tokenizer_version": cfg["tokenizer_version"],
         "vocabulary_hash": vhash, "size": len(vocab), "tokens": vocab},
        indent=2) + "\n")

    files = sorted(p for p in canonical_dir.glob("*.json")
                   if not p.name.endswith(".invalid.json"))
    if limit:
        files = files[:limit]

    results = {"tokenized": [], "grid_unsupported": [], "roundtrip_failures": []}
    denom_census: dict[int, int] = {}
    collision_total = 0

    for path in files:
        score = json.loads(path.read_text())
        pid = score["piece_id"]
        ann_path = ann_dir / f"{pid}.json"
        annotations = (json.loads(ann_path.read_text())["annotations"]
                       if ann_path.exists() else [])

        doc = {
            "piece_id": pid,
            "work_group_id": score.get("work_group_id"),
            "voice_count": score["voice_count"],
            # §19.4: every sequence carries its provenance.
            "tokenizer_name": cfg["tokenizer_name"],
            "tokenizer_version": cfg["tokenizer_version"],
            "vocabulary_hash": vhash,
            "source_canonical_event_hash": score["normalization"]["core_event_hash"],
            "annotation_tokens_from": pid if annotations else None,
        }

        try:
            toks, gpolicy = grid_tokens(score, cfg, annotations)
            doc["grid"] = {"tokens": toks, "policy": gpolicy}
            denom_census[gpolicy["steps_per_quarter"]] = \
                denom_census.get(gpolicy["steps_per_quarter"], 0) + 1

            # Round-trip: decoding the grid must return exactly the attacks the
            # grid claims to carry -- every tie-merged event except the intra-voice
            # simultaneities the policy already declares it dropped. Comparing
            # against the raw note list instead would silently excuse those.
            decoded = decode_grid(toks, score["voice_count"], gpolicy["steps_per_quarter"])
            dropped = {(c["voice_id"], parse_frac(c["onset_q"]), c["dropped"])
                       for c in gpolicy.get("voice_collision_detail_all", [])}
            expected: dict[str, set] = {}
            for v in score["voices"]:
                acc = {(onset, midi) for onset, _, midi in merge_tied(v["notes"])}
                acc -= {(o, m) for (vid, o, m) in dropped if vid == v["voice_id"]}
                expected[v["voice_id"]] = acc
            got = {k: set(v) for k, v in decoded.items()}
            if got != expected:
                diff = {}
                for k in expected:
                    miss = sorted(expected[k] - got.get(k, set()))[:3]
                    extra = sorted(got.get(k, set()) - expected[k])[:3]
                    if miss or extra:
                        diff[k] = {"missing": str(miss), "extra": str(extra)}
                results["roundtrip_failures"].append(
                    {"piece_id": pid, "diff": str(diff)[:400]})
            collision_total += gpolicy.get("voice_collisions", 0)
        except GridUnsupported as e:
            doc["grid"] = {"tokens": None, "policy": {
                "view": "grid", "exact": False, "unsupported_reason": str(e)}}
            results["grid_unsupported"].append({"piece_id": pid, "reason": str(e)})

        events, epolicy = event_records(score, cfg, annotations)
        doc["event"] = {"records": events, "policy": epolicy}

        (out_dir / f"{pid}.json").write_text(
            json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
        results["tokenized"].append({
            "piece_id": pid,
            "grid_tokens": len(doc["grid"]["tokens"]) if doc["grid"]["tokens"] else 0,
            "event_records": len(events),
            "steps_per_quarter": doc["grid"]["policy"].get("steps_per_quarter"),
        })

    summary = {
        "stage": "tokenize",
        "tokenizer_name": cfg["tokenizer_name"],
        "tokenizer_version": cfg["tokenizer_version"],
        "vocabulary_hash": vhash,
        "vocabulary_size": len(vocab),
        "pieces_tokenized": len(results["tokenized"]),
        "grid_unsupported": len(results["grid_unsupported"]),
        "roundtrip_failures": len(results["roundtrip_failures"]),
        "denominator_census": dict(sorted(denom_census.items())),
        "grid_voice_collisions": collision_total,
        "total_grid_tokens": sum(r["grid_tokens"] for r in results["tokenized"]),
        "total_event_records": sum(r["event_records"] for r in results["tokenized"]),
        "failures": ([f"{len(results['roundtrip_failures'])} grid round-trip failures"]
                     if results["roundtrip_failures"] else []),
        "results": results,
    }
    (root / "manifests" / "tokenization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
