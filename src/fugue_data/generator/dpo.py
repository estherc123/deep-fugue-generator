"""DPO against drift: prefer Bach's own continuation to the model's.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m fugue_data.generator.dpo \
        --base data/models/generator_v7.pt --out data/models/generator_v7_dpo.pt

Pairs come from data/generated/critic_negatives: for one opening (Bach up to
a point, the piece's plan), the chosen continuation is Bach's and the rejected
one is the generator's (two per opening). No critic is involved: the critic
failed the owner's blind test (9/16), but "Bach's continuation beats the
model's" is a judgement the owner shares.

Loss (DPO, Rafailov et al. 2023, plus a small likelihood term on the chosen
side as in RPO, which keeps the model from lowering both):

    -log sigmoid(beta * [(logp(y_w) - logp_ref(y_w)) - (logp(y_l) - logp_ref(y_l))])
    + alpha * NLL(y_w) / |y_w|

Log-probabilities are summed over the continuation's tokens the model chooses
at inference: notes written in by the plan (subject entries) are left out.
The reference is the frozen base model. Early stopping on held-out pairs
(validation split): preference accuracy and the chosen side's NLL.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .encode import load_corpus
from .sample import load_model, plan_tokens
from .train import build_sequence, collate

ROOT = Path(__file__).resolve().parents[3]
KEYS = ("role", "to_entry", "to_cad", "cad_key", "pedal", "pnote", "epc", "key", "next_key",
        "to_key", "seq", "chord", "next_chord", "bass", "active")


def _cut(p, end):
    return {**p, "n_steps": end, "grid": p["grid"][:end],
            "feats": {k: v[:end] for k, v in p["feats"].items()},
            **{k: p[k][:end] for k in KEYS}}


def load_pairs(ctx: int) -> list[dict]:
    pieces = {p["piece_id"]: p for p in load_corpus(ROOT, baroque=True, labels=True)}
    out = []
    for f in sorted((ROOT / "data" / "generated" / "critic_negatives").glob("*.npz")):
        pid = f.stem.split("@")[0]
        if pid not in pieces:
            continue
        z = np.load(f)
        real, gens, pr = z["real"], z["generated"], int(z["prompt_steps"])
        n = len(real)
        q = _cut(pieces[pid], n)
        nv = q["nv"]
        room = (ctx - 200) // nv                       # leave room for the prefix
        start = max(0, n - room)
        forced = plan_tokens(q) >= 0 if q["has_plan"] else np.zeros(real.shape, dtype=bool)
        resp = np.zeros(real.shape, dtype=bool)
        resp[pr:] = True
        resp &= ~forced
        for g in gens:
            if (g[pr:] == real[pr:]).all():
                continue
            out.append({"q": q, "start": start, "chosen": real, "rejected": g,
                        "mask": resp, "split": str(z["split"]), "piece": pid})
    return out


def _seq(item, grid, ctx):
    q = {**item["q"], "grid": grid}
    seq, feats, loss = build_sequence(q, item["start"], ctx, 0, True)
    nv = q["nv"]
    pre = int((~loss).sum())
    m = item["mask"][item["start"]:].reshape(-1)[:len(seq) - pre]
    loss = loss.copy()
    loss[pre:pre + len(m)] &= m
    loss[pre + len(m):] = False
    return seq, feats, loss


def seq_logp(model, batch, ctx, dev):
    """Sum and count of log p over masked target tokens, per sequence."""
    x, fx, y = collate(batch, ctx, dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x, fx)
    lp = -F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-100, reduction="none")
    m = (y != -100).float()
    return (lp * m).sum(1), m.sum(1)


@torch.no_grad()
def evaluate(policy, ref, items, ctx, dev, beta, bs=8):
    policy.eval()
    acc = margin = nll = n_tok = 0.0
    for i in range(0, len(items), bs):
        it = items[i:i + bs]
        w = [_seq(x, x["chosen"], ctx) for x in it]
        l = [_seq(x, x["rejected"], ctx) for x in it]
        pw, cw = seq_logp(policy, w, ctx, dev)
        pl, _ = seq_logp(policy, l, ctx, dev)
        rw, _ = seq_logp(ref, w, ctx, dev)
        rl, _ = seq_logp(ref, l, ctx, dev)
        d = beta * ((pw - rw) - (pl - rl))
        acc += float((d > 0).sum())
        margin += float(d.sum())
        nll += float(-pw.sum())
        n_tok += float(cw.sum())
    policy.train()
    return {"pref_acc": round(acc / len(items), 4), "margin": round(margin / len(items), 3),
            "chosen_nll": round(nll / max(1, n_tok), 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/models/generator_v7.pt")
    ap.add_argument("--out", default="data/models/generator_v7_dpo.pt")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=50)
    args = ap.parse_args(argv)
    dev = "cuda"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    policy, ck = load_model(ROOT / args.base, dev)
    ref, _ = load_model(ROOT / args.base, dev)
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    policy.train()
    ctx = policy.ctx
    pairs = load_pairs(ctx)
    tr = [p for p in pairs if p["split"] == "train"]
    va = [p for p in pairs if p["split"] == "validation"]
    print(json.dumps({"pairs_train": len(tr), "pairs_val": len(va)}), flush=True)
    opt = torch.optim.AdamW([p_ for p_ in policy.parameters() if p_.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    base_eval = evaluate(policy, ref, va, ctx, dev, args.beta)
    print(json.dumps({"step": 0, "val": base_eval}), flush=True)
    best = (-1.0, None)
    step = 0
    t0 = time.time()
    steps_per_epoch = math.ceil(len(tr) / args.batch)
    for ep in range(args.epochs):
        order = rng.permutation(len(tr))
        for b in range(steps_per_epoch):
            it = [tr[i] for i in order[b * args.batch:(b + 1) * args.batch]]
            w = [_seq(x, x["chosen"], ctx) for x in it]
            l = [_seq(x, x["rejected"], ctx) for x in it]
            pw, cw = seq_logp(policy, w, ctx, dev)
            pl, _ = seq_logp(policy, l, ctx, dev)
            with torch.no_grad():
                rw, _ = seq_logp(ref, w, ctx, dev)
                rl, _ = seq_logp(ref, l, ctx, dev)
            d = args.beta * ((pw - rw) - (pl - rl))
            loss = -F.logsigmoid(d).mean() + args.alpha * (-pw / cw.clamp(min=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.eval_every == 0 or b == steps_per_epoch - 1:
                ev = evaluate(policy, ref, va, ctx, dev, args.beta)
                print(json.dumps({"step": step, "epoch": ep, "loss": round(loss.item(), 4),
                                  "val": ev, "t": round(time.time() - t0)}), flush=True)
                # keep the best held-out preference accuracy whose chosen-side NLL
                # has not risen more than 2% over the base model's
                if ev["chosen_nll"] <= base_eval["chosen_nll"] * 1.02 and ev["pref_acc"] > best[0]:
                    best = (ev["pref_acc"], {k: v.detach().cpu().clone()
                                             for k, v in policy.state_dict().items()}, ev, step)
    if best[1] is None:
        print("no checkpoint met the NLL guard; nothing saved", flush=True)
        return 1
    torch.save({**ck, "state": best[1], "dpo": {"base": args.base, "beta": args.beta,
                                                "alpha": args.alpha, "lr": args.lr,
                                                "step": best[3], "val": best[2],
                                                "val_base": base_eval}}, ROOT / args.out)
    print(json.dumps({"saved": args.out, "step": best[3], "val": best[2], "val_base": base_eval}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
