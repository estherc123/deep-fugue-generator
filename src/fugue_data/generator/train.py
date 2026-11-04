"""Train the v1 generator.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m fugue_data.generator.train

Sampling: a random crop of a random piece (weighted by length), a random
transposition (-5..+6 semitones, refused if it pushes a note off the keyboard),
and plan dropout. Loss is over music tokens only -- the prefix is given.

Evaluation is deterministic: every validation piece, untransposed, with its plan
where it has one, cut into consecutive context-sized chunks. Early stopping
watches validation perplexity. The checkpoint records what the old style prior
did not: step count, losses, whether early stopping fired, the data hash and
the git SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .encode import (FEATURES, MIDI_HI, MIDI_LO, NOTE0, SEG, STOI, VOCAB,
                     load_corpus, prefix_tokens)
from .model import FugueGenerator

N_NOTES = MIDI_HI - MIDI_LO + 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Building one training/eval sequence
# --------------------------------------------------------------------------- #
def _music_arrays(p: dict, plan: bool):
    """Flatten a piece time-major into tokens + per-position features."""
    n, nv = p["n_steps"], p["nv"]
    tok = p["grid"].reshape(-1).astype(np.int64)
    f = {k: np.zeros(n * nv, dtype=np.int64) for k in FEATURES}
    f["voice"][:] = np.tile(np.arange(1, nv + 1), n)
    for k in ("sub", "beat", "bar", "prog"):
        f[k][:] = np.repeat(p["feats"][k], nv)
    f["seg"][:] = SEG["music"]
    if plan:
        for k in ("key", "next_key", "to_key"):
            f[k][:] = np.repeat(p[k], nv)
        f["seq"][:] = p["seq"].reshape(-1)
    if plan and p["has_plan"]:
        f["role"][:] = p["role"].reshape(-1)
        f["pnote"][:] = p["pnote"].reshape(-1)
        f["epc"][:] = p["epc"].reshape(-1)
        for k in ("to_entry", "to_cad", "cad_key", "pedal"):
            f[k][:] = np.repeat(p[k], nv)
    return tok, f


def build_sequence(p: dict, start_step: int, ctx: int, shift: int, plan: bool):
    """(tokens, features, loss_mask) of length <= ctx + 1, starting at a step."""
    tonic = (p["tonic"] + shift) % 12
    use_plan = plan and p["has_plan"]
    pre = prefix_tokens(p, tonic, use_plan)
    pre_seg = [SEG["prefix"]] * len(pre)
    if use_plan and p["subject"] is not None:
        subj = [int(t) for t in p["subject"]]
        pre += [STOI["SUBJ"]] + subj + [STOI["/SUBJ"]]
        pre_seg += [SEG["subject"]] * (len(subj) + 2)
    pre += [STOI["<SEP>"]]
    pre_seg += [SEG["prefix"]]

    tok, f = _music_arrays(p, plan)
    nv = p["nv"]
    room = ((ctx + 1 - len(pre)) // nv) * nv
    a = start_step * nv
    m_tok = tok[a:a + room]
    seq = np.concatenate([np.array(pre, dtype=np.int64), m_tok])
    feats = {}
    for k in FEATURES:
        pre_f = np.zeros(len(pre), dtype=np.int64)
        if k == "seg":
            pre_f[:] = pre_seg
        feats[k] = np.concatenate([pre_f, f[k][a:a + room]])
    if shift:
        notes = (seq >= NOTE0) & (seq < NOTE0 + N_NOTES)
        seq = np.where(notes, seq + shift, seq)
        feats["pnote"] = np.where(feats["pnote"] >= 2, feats["pnote"] + shift,
                                  feats["pnote"])
    loss = np.zeros(len(seq), dtype=bool)
    loss[len(pre):] = True
    return seq, feats, loss


def _shift_ok(p: dict, shift: int) -> bool:
    notes = p["grid"][(p["grid"] >= NOTE0) & (p["grid"] < NOTE0 + N_NOTES)]
    if p["subject"] is not None:
        notes = np.concatenate([notes, p["subject"][p["subject"] >= NOTE0]])
    if not len(notes):
        return True
    lo, hi = int(notes.min()) + shift, int(notes.max()) + shift
    return lo >= NOTE0 and hi < NOTE0 + N_NOTES


def collate(items, ctx: int, dev):
    """Inputs at i carry the FEATURES of position i+1 (the predicted token)."""
    b = len(items)
    x = torch.full((b, ctx), STOI["<PAD>"], dtype=torch.long)
    y = torch.full((b, ctx), -100, dtype=torch.long)
    fx = {k: torch.zeros((b, ctx), dtype=torch.long) for k in FEATURES}
    for i, (seq, feats, loss) in enumerate(items):
        n = min(len(seq) - 1, ctx)
        x[i, :n] = torch.from_numpy(seq[:n])
        tgt = torch.from_numpy(seq[1:n + 1]).clone()
        tgt[~torch.from_numpy(loss[1:n + 1])] = -100
        y[i, :n] = tgt
        for k in FEATURES:
            fx[k][i, :n] = torch.from_numpy(feats[k][1:n + 1])
    return (x.to(dev, non_blocking=True), {k: v.to(dev, non_blocking=True)
                                           for k, v in fx.items()},
            y.to(dev, non_blocking=True))


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, pieces, ctx, dev, plan=True, bs=16, free_only=False):
    """Perplexity over music tokens. free_only scores only notes outside planned
    entries (identical positions with or without the plan), i.e. the
    counterpoint the model has to invent rather than copy from the plan."""
    model.eval()
    items = []
    for p in pieces:
        step = 0
        while step < p["n_steps"]:
            seq, feats, loss = build_sequence(p, step, ctx, 0, plan)
            used_tok = int(loss.sum())
            if free_only:
                a = step * p["nv"]
                pn = p["pnote"].reshape(-1)[a:a + used_tok]
                loss[len(seq) - used_tok:] &= pn == 0
            items.append((seq, feats, loss))
            used = used_tok // p["nv"]
            step += max(1, used)
    nll = n = 0.0
    for i in range(0, len(items), bs):
        x, fx, y = collate(items[i:i + bs], ctx, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            logits = model(x, fx)
        l = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                            ignore_index=-100, reduction="sum")
        nll += l.item()
        n += int((y != -100).sum())
    model.train()
    return math.exp(nll / max(1, n)), int(n)


def _git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _data_hash(pieces) -> str:
    h = hashlib.sha256()
    for p in sorted(pieces, key=lambda p: p["piece_id"]):
        h.update(p["piece_id"].encode())
        h.update(p["grid"].tobytes())
    return h.hexdigest()[:16]


def train(cfg: dict, root: Path | None = None) -> dict:
    root = root or _repo_root()
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    pieces = load_corpus(root)
    by = {s: [p for p in pieces if p["split"] == s] for s in ("train", "validation", "test")}
    tr = by["train"]
    weights = np.array([p["n_steps"] * p["nv"] for p in tr], dtype=float)
    weights /= weights.sum()

    m = cfg["model"]
    ctx = m["ctx"]
    model = FugueGenerator(len(VOCAB), m["d"], m["layers"], m["heads"], ctx,
                           m["dropout"]).to(dev)
    params = sum(p.numel() for p in model.parameters())
    o = cfg["optim"]
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], betas=(0.9, 0.95),
                            weight_decay=o["weight_decay"])

    def lr_at(step):
        if step < o["warmup"]:
            return o["lr"] * step / o["warmup"]
        frac = (step - o["warmup"]) / max(1, o["max_steps"] - o["warmup"])
        return o["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))

    out = root / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".log.jsonl")
    log = log_path.open("w")

    def emit(rec):
        rec["t"] = round(time.time() - t0, 1)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    t0 = time.time()
    emit({"event": "start", "params": params, "device": dev,
          "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
          "train_tokens": int(sum(p["n_steps"] * p["nv"] for p in tr)),
          "pieces": {k: len(v) for k, v in by.items()}, "config": cfg})

    best, best_state, bad, stopped_early = float("inf"), None, 0, False
    step = 0
    for step in range(1, o["max_steps"] + 1):
        items = []
        for _ in range(o["batch"]):
            p = tr[rng.choice(len(tr), p=weights)]
            room_steps = max(1, (ctx - 140) // p["nv"])
            start = int(rng.integers(0, max(1, p["n_steps"] - room_steps // 2)))
            shift = int(rng.integers(-5, 7))
            if not _shift_ok(p, shift):
                shift = 0
            plan = rng.random() >= cfg["plan_dropout"]
            items.append(build_sequence(p, start, ctx, shift, plan))
        x, fx, y = collate(items, ctx, dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(x, fx)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg["log_every"] == 0:
            emit({"event": "train", "step": step, "loss": round(loss.item(), 4),
                  "lr": round(lr_at(step), 6)})
        if step % cfg["eval_every"] == 0 or step == o["max_steps"]:
            vppl, vn = evaluate(model, by["validation"], ctx, dev)
            emit({"event": "eval", "step": step, "val_ppl": round(vppl, 4),
                  "val_tokens": vn, "train_loss": round(loss.item(), 4)})
            if vppl < best - 1e-4:
                best, bad = vppl, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    stopped_early = True
                    emit({"event": "early_stop", "step": step, "best_val_ppl": round(best, 4)})
                    break

    model.load_state_dict(best_state)
    final = {}
    for name, ps in (("validation", by["validation"]), ("test", by["test"])):
        ppl, n = evaluate(model, ps, ctx, dev)
        final[f"{name}_ppl"] = round(ppl, 4)
        for corpus in ("fugue", "chorale"):
            sub = [p for p in ps if p["corpus"] == corpus]
            if sub:
                final[f"{name}_ppl_{corpus}"] = round(evaluate(model, sub, ctx, dev)[0], 4)
        planned = [p for p in ps if p["has_plan"]]
        if planned:
            final[f"{name}_ppl_planned_with_plan"] = round(evaluate(model, planned, ctx, dev, True)[0], 4)
            final[f"{name}_ppl_planned_plan_dropped"] = round(evaluate(model, planned, ctx, dev, False)[0], 4)
            final[f"{name}_ppl_planned_free_with_plan"] = round(evaluate(model, planned, ctx, dev, True, free_only=True)[0], 4)
            final[f"{name}_ppl_planned_free_plan_dropped"] = round(evaluate(model, planned, ctx, dev, False, free_only=True)[0], 4)

    torch.save({"state": model.state_dict(), "config": cfg, "vocab": VOCAB,
                "features": FEATURES, "params": params, "steps_run": step,
                "stopped_early": stopped_early, "best_val_ppl": best,
                "final": final, "data_hash": _data_hash(pieces),
                "git_sha": _git_sha(root)}, out)
    emit({"event": "done", "steps_run": step, "stopped_early": stopped_early,
          "best_val_ppl": round(best, 4), **final, "checkpoint": str(out)})
    log.close()
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/generator.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = _repo_root()
    cfg = yaml.safe_load((root / args.config).read_text())
    if args.max_steps:
        cfg["optim"]["max_steps"] = args.max_steps
    if args.out:
        cfg["out"] = args.out
    train(cfg, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
