"""Generate corrupted-Bach pairs (original wins by construction).

Acceptance is decided here, not in the operators:

1. **Hard gate.** Under the configured profile (default `bach_empirical`, which
   fires zero hard violations on held-out Bach), the corrupted window may not
   carry any hard violation the original does not. Windows are compared by
   (rule, voices) counts rather than demanding zero outright, because a window
   cut from an exposition legitimately has a voice that has not entered yet,
   and that is the same in both halves of the pair.
2. **Real change.** At least `min_events_changed` events must differ.
3. **Least detectable.** Of the candidates that pass, keep the one whose
   soft + diagnostic violation counts rise least over the original. Otherwise
   the judge could learn "more rule noise = corrupted", which the rule engine
   already knows and which is not the point.

Splits are inherited from the source piece and never mixed: a pair cut from a
test piece is a test pair, and its filler comes from test-split fragments.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import yaml

from ..constraints.engine import evaluate, load_profile
from ..normalize.rational import frac_str, parse_frac
from .ops import registry
from .windows import (Window, bar_length_q, estimate_key, is_compound,
                      voice_events, windows_for)

GENERATOR_VERSION = "1.0.0"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16)


def _jsonable(x):
    if isinstance(x, Fraction):
        return frac_str(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_config(root: Path) -> dict:
    return yaml.safe_load((root / "config" / "corruption.yaml").read_text())


def _splits(root: Path) -> dict[str, dict]:
    out = {}
    for line in (root / "manifests" / "splits.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["piece_id"]] = r
    return out


def _entries(root: Path, piece_id: str):
    """Labelled subject/answer entries, absolute, in onset order with an index."""
    p = root / "data" / "annotations" / f"{piece_id}.json"
    if not p.exists():
        return []
    doc = json.loads(p.read_text())
    rows = []
    for a in doc["annotations"]:
        if a["type"] != "subject" or not a.get("voice_id") or not a.get("duration_q"):
            continue
        if a.get("analytical_status") not in (None, "accepted"):
            continue
        s = parse_frac(a["onset_q"])
        rows.append((a["voice_id"], s, s + parse_frac(a["duration_q"])))
    rows.sort(key=lambda r: r[1])
    return [(v, s, e, i) for i, (v, s, e) in enumerate(rows)]


def build_filler_pool(root: Path) -> dict:
    """Hard-valid CP-SAT fragments keyed by (split, voices, bar_q, compound)."""
    pool: dict = defaultdict(list)
    for f in sorted((root / "data" / "generated" / "valid").glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("hard_valid"):
                continue
            req, score = r["request"], r["score"]
            meter = (req["meter"]["numerator"], req["meter"]["denominator"])
            ev = voice_events(score)
            pool[(r["split"], req["voice_count"], str(bar_length_q(meter)),
                  is_compound(meter))].append({
                "sample_id": r["sample_id"], "tonic_pc": req["key"]["tonic_pc"],
                "mode": req["key"]["mode"], "n_bars": req["length_bars"],
                "voice_order": list(ev), "voices": ev})
    return dict(pool)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _profile_counts(win: Window, profile: dict):
    viol = evaluate(win.to_fragment(), profile)
    hard, other = Counter(), Counter()
    for v in viol:
        if v["severity"] == "hard":
            hard[(v["rule_id"], tuple(v["voice_ids"]))] += 1
        else:
            other[v["rule_id"]] += 1
    return hard, other


def _events_changed(a: Window, b: Window) -> int:
    diff = 0
    for vid in a.voices:
        ca, cb = Counter(a.voices[vid]), Counter(b.voices.get(vid, []))
        diff += sum(((ca - cb) + (cb - ca)).values())
    return diff


def _hard_summary(c: Counter) -> dict:
    out = Counter()
    for (rule, _), n in c.items():
        out[rule] += n
    return dict(sorted(out.items()))


def corrupt_window(win: Window, op_name: str, ctx: dict, cfg: dict, profile: dict,
                   seed: int, base=None):
    """Best accepted corruption of `win` by `op_name`, or (None, reason)."""
    rng = random.Random(_seed(seed, win.piece_id, frac_str(win.start_q), op_name))
    op_cfg = cfg["operators"][op_name]
    cands = registry()[op_name](win, ctx, rng, op_cfg)
    if not cands:
        return None, "no_candidates"
    cap = cfg.get("max_candidates", 16)
    if len(cands) > cap:
        cands = rng.sample(cands, cap)
    hard0, other0 = base or _profile_counts(win, profile)
    best, reasons = None, Counter()
    for voices, params in cands:
        cw = win.copy_with(voices)
        changed = _events_changed(win, cw)
        if changed < cfg.get("min_events_changed", 4):
            reasons["no_change"] += 1
            continue
        hard1, other1 = _profile_counts(cw, profile)
        if any(n > hard0.get(k, 0) for k, n in hard1.items()):
            reasons["new_hard_violation"] += 1
            continue
        deltas = {r: other1.get(r, 0) - other0.get(r, 0)
                  for r in sorted(set(other0) | set(other1))}
        penalty = sum(max(0, d) for d in deltas.values())
        if best is None or penalty < best[0]:
            best = (penalty, cw, params, changed, deltas, hard0, hard1)
    if best is None:
        return None, max(reasons, key=reasons.get) if reasons else "no_candidates"
    penalty, cw, params, changed, deltas, h0, h1 = best
    rec = {
        "corruption_id": op_name, "params": _jsonable(params),
        "corrupted": cw.compact(), "events_changed": changed,
        "rule_penalty": penalty, "rule_deltas": deltas,
        "hard_original": _hard_summary(h0), "hard_corrupted": _hard_summary(h1),
        "candidates_tried": len(cands),
        "candidates_rejected": dict(reasons),
    }
    return rec, None


# --------------------------------------------------------------------------- #
# Per-piece worker
# --------------------------------------------------------------------------- #
_WORKER: dict = {}


def _init_worker(root: str, cfg: dict, pool: dict):
    _WORKER.update(root=Path(root), cfg=cfg, pool=pool,
                   profile=load_profile(cfg["rule_profile"]))


def _process_piece(job):
    path, split_row = job
    root, cfg, profile = _WORKER["root"], _WORKER["cfg"], _WORKER["profile"]
    score = json.loads(Path(path).read_text())
    pid = score["piece_id"]
    split = split_row["split"]
    abs_entries = _entries(root, pid)
    piece_events = voice_events(score)
    wcfg = cfg["windows"]
    meter = tuple((score.get("meter_map") or [{"numerator": 4, "denominator": 4}])[0]
                  [k] for k in ("numerator", "denominator"))
    vc = len(piece_events)
    filler = _WORKER["pool"].get((split, vc, str(bar_length_q(meter)),
                                  is_compound(meter)), [])
    records, fails = [], Counter()
    for win in windows_for(score, wcfg["bars"], wcfg["min_q"], wcfg["stride_bars"]):
        w0, w1 = win.start_q, win.start_q + win.length_q
        rel = [(v, s - w0, e - w0, i) for v, s, e, i in abs_entries
               if s < w1 and e > w0]
        key = estimate_key(win.voices)
        ctx = {"entries": rel, "abs_entries": abs_entries,
               "piece_events": piece_events, "voice_count": vc,
               "piece_length_q": parse_frac(score["quarter_length_total"]),
               "local_key": key, "filler_pool": filler}
        base = _profile_counts(win, profile)
        for op_name in cfg["operators"]:
            if not cfg["operators"][op_name].get("enabled", True):
                continue
            rec, why = corrupt_window(win, op_name, ctx, cfg, profile, cfg["seed"], base)
            if rec is None:
                fails[(op_name, why)] += 1
                continue
            pair_id = "corr:" + hashlib.sha256(json.dumps(
                [pid, frac_str(win.start_q), op_name, rec["params"]],
                sort_keys=True).encode()).hexdigest()[:16]
            records.append({
                "pair_id": pair_id, "generator_version": GENERATOR_VERSION,
                "piece_id": pid, "work_group_id": score.get("work_group_id"),
                "corpus": score.get("movement_type"), "split": split,
                "voice_count": vc, "meter": f"{meter[0]}/{meter[1]}",
                "window": {"start_q": frac_str(win.start_q),
                           "length_q": frac_str(win.length_q),
                           "first_bar_index": win.first_bar, "n_bars": win.n_bars,
                           "start_measure": win.meta.get("start_measure")},
                "local_key": {"tonic_pc": key[0], "mode": key[1],
                              "fit": round(key[2], 3)},
                "rule_profile": cfg["rule_profile"],
                "preferred": "original", "label_source": "by_construction",
                "original": win.compact(), **rec})
    return pid, records, {f"{k[0]}:{k[1]}": n for k, n in fails.items()}


# --------------------------------------------------------------------------- #
def generate_corrupted(root: Path | None = None, *, workers: int | None = None,
                       limit: int | None = None, ops: list[str] | None = None) -> dict:
    root = root or _repo_root()
    cfg = load_config(root)
    if ops:
        unknown = set(ops) - set(cfg["operators"])
        if unknown:
            raise ValueError(f"unknown operators: {sorted(unknown)}")
        for name in cfg["operators"]:
            cfg["operators"][name]["enabled"] = name in ops
    splits = _splits(root)
    corpora = set(cfg["corpora"])
    jobs = []
    for p in sorted((root / "data" / "canonical").glob("*.json")):
        if p.name.endswith(".invalid.json"):
            continue
        pid = p.stem
        row = splits.get(pid)
        if row is None or row["corpus"] not in corpora:
            continue
        jobs.append((str(p), row))
    if limit:
        jobs = jobs[:limit]
    use_filler = cfg["operators"].get("solver_filler", {}).get("enabled", True)
    pool = build_filler_pool(root) if use_filler else {}

    out_dir = root / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "pairs.jsonl.tmp"
    t0 = time.time()
    by = Counter()
    fails: Counter = Counter()
    pieces_with_pairs = set()
    penalties = defaultdict(list)
    seen = set()
    duplicates = 0
    with Pool(workers, initializer=_init_worker,
              initargs=(str(root), cfg, pool)) as pp, tmp.open("w") as fh:
        for pid, recs, f in pp.imap(_process_piece, jobs):
            fails.update(f)
            for r in recs:
                if r["pair_id"] in seen:
                    duplicates += 1
                    continue
                seen.add(r["pair_id"])
                fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
                by[(r["corruption_id"], r["split"], r["corpus"])] += 1
                penalties[r["corruption_id"]].append(r["rule_penalty"])
                pieces_with_pairs.add(pid)
    tmp.replace(out_dir / "pairs.jsonl")

    by_op = Counter()
    by_split = Counter()
    for (op, split, _), n in by.items():
        by_op[op] += n
        by_split[split] += n
    summary = {
        "stage": "generate_corrupted_pairs",
        "generator_version": GENERATOR_VERSION,
        "rule_profile": cfg["rule_profile"], "seed": cfg["seed"],
        "pieces": len(jobs), "pieces_with_pairs": len(pieces_with_pairs),
        "pairs": sum(by.values()), "duplicates_dropped": duplicates,
        "by_corruption": dict(sorted(by_op.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_corruption_split_corpus": {f"{o}|{s}|{c}": n
                                       for (o, s, c), n in sorted(by.items())},
        "rejections": dict(sorted(fails.items())),
        "rule_penalty_mean": {op: round(sum(v) / len(v), 3)
                              for op, v in sorted(penalties.items())},
        "rule_penalty_zero_share": {op: round(sum(1 for x in v if x == 0) / len(v), 3)
                                    for op, v in sorted(penalties.items())},
        "elapsed_s": round(time.time() - t0, 1),
        "output": str((out_dir / "pairs.jsonl").relative_to(root)),
    }
    (root / "manifests" / "corruption_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
