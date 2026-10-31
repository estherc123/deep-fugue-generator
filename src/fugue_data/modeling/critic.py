"""Constraint critic (Spec §20.3): can a learned model detect rule violations?

The question this answers is prior to any RL work. Every audible gain this project
has made came from measuring Bach and adding a hard constraint; nothing learned
has yet produced one. Before building a loop that trains against a learned signal,
it is worth knowing whether a learned model can detect violations the rules define
*exactly* -- a task where the ground truth is perfect and the labels are free.

The data is unusually clean for this: each of the 10,000 negatives is a mutation of
a specific positive, carrying exactly one intended violation of one named rule. So
the critic is not asked "does this look like Bach" but "was something broken here",
with the parent available as a control.

Reported honestly:
  * accuracy on held-out fragments, split-respecting
  * accuracy on MATCHED PAIRS -- parent positive vs its own mutation. This is the
    number that matters: it removes every confound except the mutation itself, and
    a critic that scores well overall but at chance on pairs has learned the
    surface statistics of the generator, not the rules.
  * per-rule recall, since a critic useless on parallel fifths but good on timing
    is not a constraint critic in any useful sense.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ..tokenization.views import GridUnsupported, grid_tokens
from .style_prior import Block, device


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_fragments(root: Path) -> dict:
    """sample_id -> canonical score, for both the valid and invalid pools."""
    out = {}
    for sub in ("valid", "invalid"):
        for shard in sorted((root / "data" / "generated" / sub).glob("shard_*.jsonl")):
            for line in shard.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if "score" in r:
                    out[r["sample_id"]] = r["score"]
    return out


class Critic(nn.Module):
    """Encoder + mean-pool + head. Bidirectional: judging a finished fragment is
    not a next-token problem, so there is no reason to mask the future."""

    def __init__(self, vocab: int, d=192, layers=4, heads=4, ctx=512, drop=0.1,
                 n_rules: int = 0):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.rule_head = nn.Linear(d, n_rules) if n_rules else None

    def forward(self, idx, pad_mask):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        # No causal mask: the whole fragment is visible, which is the task.
        for b in self.blocks:
            x = b(x, None)
        x = self.norm(x)
        m = pad_mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        logit = self.head(pooled).squeeze(-1)
        rules = self.rule_head(pooled) if self.rule_head is not None else None
        return logit, rules


def build_dataset(root: Path | None = None, cache: Path | None = None) -> dict:
    """Tokenize every positive and negative once, keyed by split."""
    root = root or _repo_root()
    cache = cache or (root / "data" / "models" / "critic_data.pt")
    if cache.exists():
        return torch.load(cache, map_location="cpu")

    import pandas as pd
    pos = pd.read_parquet(root / "data/exports/constraint_critic/positives.parquet")
    neg = pd.read_parquet(root / "data/exports/constraint_critic/controlled_negatives.parquet")
    frags = load_fragments(root)
    vocab = json.loads((root / "data/tokenized/vocabulary.json").read_text())
    stoi = {t: i for i, t in enumerate(vocab["tokens"])}
    cfg = yaml.safe_load((root / "config" / "tokenizer.yaml").read_text())

    rules = sorted(set(neg["target_rule_id"].dropna()))
    rtoi = {r: i for i, r in enumerate(rules)}

    def enc(sid):
        sc = frags.get(sid)
        if sc is None:
            return None
        try:
            toks, _ = grid_tokens(sc, cfg)
        except GridUnsupported:
            return None
        ids = [stoi[t] for t in toks if t in stoi]
        return ids or None

    rows = []
    for df, lab in ((pos, 0), (neg, 1)):
        for r in df.itertuples():
            ids = enc(r.sample_id)
            if ids is None:
                continue
            rows.append({
                "ids": ids, "y": lab, "split": r.split,
                "sample_id": r.sample_id,
                "parent": getattr(r, "parent_valid_sample_id", None),
                "rule": rtoi.get(getattr(r, "target_rule_id", None), -1),
            })
    data = {"rows": rows, "rules": rules, "vocab": len(vocab["tokens"])}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache)
    return data


def _batch(rows, idxs, ctx, dev):
    ids = [rows[i]["ids"][:ctx] for i in idxs]
    n = max(len(x) for x in ids)
    x = torch.zeros(len(ids), n, dtype=torch.long)
    m = torch.zeros(len(ids), n, dtype=torch.bool)
    for k, s in enumerate(ids):
        x[k, :len(s)] = torch.tensor(s, dtype=torch.long)
        m[k, :len(s)] = True
    y = torch.tensor([float(rows[i]["y"]) for i in idxs])
    r = torch.tensor([rows[i]["rule"] for i in idxs])
    return x.to(dev), m.to(dev), y.to(dev), r.to(dev)


@torch.no_grad()
def evaluate(model, rows, idxs, ctx, dev, bs=64) -> dict:
    model.eval()
    correct = tot = 0
    probs = {}
    per_rule = {}
    for i in range(0, len(idxs), bs):
        chunk = idxs[i:i + bs]
        x, m, y, r = _batch(rows, chunk, ctx, dev)
        logit, _ = model(x, m)
        p = torch.sigmoid(logit)
        pred = (p > 0.5).float()
        correct += (pred == y).sum().item(); tot += len(chunk)
        for k, j in enumerate(chunk):
            probs[rows[j]["sample_id"]] = p[k].item()
            if rows[j]["y"] == 1:
                rid = rows[j]["rule"]
                a, b = per_rule.get(rid, (0, 0))
                per_rule[rid] = (a + int(pred[k].item() == 1), b + 1)
    model.train()
    return {"acc": correct / max(1, tot), "n": tot, "probs": probs,
            "per_rule": per_rule}


def paired_accuracy(rows, idxs, probs) -> tuple[int, int]:
    """On matched (parent, mutation) pairs: is the mutation scored as worse?

    The number that matters. A critic can reach high overall accuracy by learning
    which generator run produced a fragment; it can only win here by detecting the
    mutation itself.
    """
    by_id = {rows[i]["sample_id"]: rows[i] for i in idxs}
    wins = tot = 0
    for r in by_id.values():
        if r["y"] != 1 or not r["parent"]:
            continue
        pp = probs.get(r["parent"])
        pn = probs.get(r["sample_id"])
        if pp is None or pn is None:
            continue
        tot += 1
        wins += int(pn > pp)
    return wins, tot


def train(root: Path | None = None, *, d=192, layers=4, heads=4, ctx=544,
          drop=0.1, steps=2000, bs=32, lr=3e-4, eval_every=200, patience=5,
          seed=0, rule_weight=0.3, out: Path | None = None) -> dict:
    root = root or _repo_root()
    data = build_dataset(root)
    rows, rules = data["rows"], data["rules"]
    dev = device()
    torch.manual_seed(seed)
    rng = random.Random(seed)

    idx = {s: [i for i, r in enumerate(rows) if r["split"] == s]
           for s in ("train", "validation", "test")}
    model = Critic(data["vocab"], d, layers, heads, ctx, drop, n_rules=len(rules)).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    best = -1.0; best_state = None; bad = 0; hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        pick = [rng.choice(idx["train"]) for _ in range(bs)]
        x, m, y, r = _batch(rows, pick, ctx, dev)
        logit, rl = model(x, m)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        # Naming WHICH rule broke is a harder task than noticing that one did;
        # as an auxiliary it pushes the encoder to represent the violation rather
        # than a generic "something is off".
        if rl is not None:
            mask = r >= 0
            if mask.any():
                loss = loss + rule_weight * F.cross_entropy(rl[mask], r[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % eval_every == 0 or step == steps:
            ev = evaluate(model, rows, idx["validation"], ctx, dev)
            w, t = paired_accuracy(rows, idx["validation"], ev["probs"])
            pair = w / max(1, t)
            hist.append({"step": step, "loss": round(loss.item(), 4),
                         "val_acc": round(ev["acc"], 4),
                         "val_paired": round(pair, 4)})
            print(f"  step {step:5d}  loss {loss.item():.4f}  "
                  f"val acc {ev['acc']:.3f}  paired {pair:.3f}", flush=True)
            if pair > best:
                best = pair
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print("  early stop", flush=True)
                    break

    if best_state:
        model.load_state_dict(best_state)
    te = evaluate(model, rows, idx["test"], ctx, dev)
    w, t = paired_accuracy(rows, idx["test"], te["probs"])
    per_rule = {rules[k]: (a / b if b else None, b)
                for k, (a, b) in sorted(te["per_rule"].items()) if k >= 0}

    out = out or (root / "data" / "models" / "constraint_critic.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "rules": rules,
                "config": {"vocab": data["vocab"], "d": d, "layers": layers,
                           "heads": heads, "ctx": ctx, "drop": drop}}, out)
    # Wilson lower bound, so a small matched-pair sample cannot look significant.
    z = 1.96; p = w / max(1, t)
    den = 1 + z * z / max(1, t)
    lo = ((p + z * z / (2 * max(1, t))) / den) - (z / den) * math.sqrt(
        p * (1 - p) / max(1, t) + z * z / (4 * max(1, t) ** 2))
    return {
        "params": params, "steps_run": hist[-1]["step"] if hist else 0,
        "elapsed_s": round(time.time() - t0, 1),
        "test_accuracy": round(te["acc"], 4), "test_n": te["n"],
        "test_paired_accuracy": round(p, 4), "test_paired_n": t,
        "paired_ci95_low": round(max(0.0, lo), 4),
        "paired_beats_chance": bool(lo > 0.5),
        "per_rule_recall": {k: (round(v[0], 3) if v[0] is not None else None, v[1])
                            for k, v in per_rule.items()},
        "history": hist, "model": str(out),
    }


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
