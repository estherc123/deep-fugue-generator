"""A small causal transformer style prior over the grid token stream (Spec §20.1).

Sized for the corpus, not for ambition. There are 615k training tokens across 150
pieces; an n-gram baseline already degrades past order 4, which is a direct
measurement of how little context this data can support. A large model here would
memorise the training split and tell us nothing, so the defaults are small and
regularised, and early stopping watches held-out content perplexity.

Perplexity is reported over *content* tokens -- the NOTE_ON/HOLD/REST decisions --
separately from the whole stream. The grid interleaves VOICE, BAR and BEAT markers
in a fixed pattern that any model predicts almost for free, and they are over half
the stream, so whole-stream perplexity flatters the model without saying anything
about the music.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .style_data import is_content, load


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d), nn.Dropout(drop))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class StylePrior(nn.Module):
    def __init__(self, vocab, d=192, layers=4, heads=4, ctx=256, drop=0.2):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight          # tied, saves parameters

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        mask = torch.triu(torch.ones(t, t, device=idx.device, dtype=torch.bool), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln(x))


def _stream(seqs, sep_id):
    ids = []
    for s in seqs:
        ids.extend(s["ids"])
        ids.append(sep_id)
    return torch.tensor(ids, dtype=torch.long)


def _batches(data, ctx, bs, gen):
    n = len(data) - ctx - 1
    ix = torch.randint(0, n, (bs,), generator=gen)
    x = torch.stack([data[i:i + ctx] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + ctx] for i in ix])
    return x, y


@torch.no_grad()
def perplexity(model, seqs, content_mask, dev, ctx):
    """Exact perplexity over each piece, split into all-token and content-token."""
    model.eval()
    tot_lp = tot_n = c_lp = c_n = 0.0
    for s in seqs:
        ids = torch.tensor(s["ids"], dtype=torch.long)
        cm = torch.tensor([content_mask[i] for i in s["ids"]], dtype=torch.bool)
        for start in range(0, len(ids) - 1, ctx):
            chunk = ids[start:start + ctx + 1]
            if len(chunk) < 2:
                continue
            x = chunk[:-1].unsqueeze(0).to(dev)
            y = chunk[1:].to(dev)
            logits = model(x)[0]
            lp = -F.cross_entropy(logits, y, reduction="none").cpu()
            tot_lp += lp.sum().item(); tot_n += len(lp)
            m = cm[start + 1:start + 1 + len(lp)]
            c_lp += lp[m].sum().item(); c_n += int(m.sum())
    model.train()
    return (math.exp(-tot_lp / max(1, tot_n)),
            math.exp(-c_lp / max(1, c_n)), int(tot_n), int(c_n))


def train(root: Path | None = None, *, d=192, layers=4, heads=4, ctx=256,
          drop=0.2, steps=3000, bs=32, lr=3e-4, eval_every=250, patience=4,
          seed=0, out: Path | None = None) -> dict:
    root = root or _repo_root()
    data, tokens, stoi = load(root)
    V = len(tokens)
    content_mask = [is_content(t) for t in tokens]
    dev = device()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    sep = stoi["<EOS>"]
    tr = _stream(data["train"], sep)
    model = StylePrior(V, d, layers, heads, ctx, drop).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    best = float("inf"); best_state = None; bad = 0; hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = _batches(tr, ctx, bs, gen)
        logits = model(x.to(dev))
        loss = F.cross_entropy(logits.reshape(-1, V), y.to(dev).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % eval_every == 0 or step == steps:
            _, vc, _, _ = perplexity(model, data["validation"], content_mask, dev, ctx)
            hist.append({"step": step, "train_loss": round(loss.item(), 4),
                         "val_ppl_content": round(vc, 3)})
            print(f"  step {step:5}  loss {loss.item():.3f}  val content ppl {vc:.3f}",
                  flush=True)
            if vc < best - 1e-3:
                best = vc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"  early stop at step {step}: {patience} evals without "
                          f"improvement", flush=True)
                    break

    if best_state:
        model.load_state_dict(best_state)
    ta, tc, tn, cn = perplexity(model, data["test"], content_mask, dev, ctx)
    va, vc, _, _ = perplexity(model, data["validation"], content_mask, dev, ctx)

    out = Path(out) if out else (root / "data" / "models" / "style_prior.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(),
                "config": {"vocab": V, "d": d, "layers": layers, "heads": heads,
                           "ctx": ctx, "drop": drop},
                "tokens": tokens}, out)

    return {
        "stage": "style_prior",
        "device": dev, "params": int(params),
        "train_tokens": int(len(tr)),
        "steps_run": hist[-1]["step"] if hist else 0,
        "val_ppl_all": round(va, 3), "val_ppl_content": round(vc, 3),
        "test_ppl_all": round(ta, 3), "test_ppl_content": round(tc, 3),
        "test_tokens": tn, "test_content_tokens": cn,
        "ngram_baseline_test_ppl_content": 4.72,
        "beats_baseline": bool(tc < 4.72),
        "elapsed_s": round(time.time() - t0, 1),
        "history": hist,
        "model_path": str(out),
    }
