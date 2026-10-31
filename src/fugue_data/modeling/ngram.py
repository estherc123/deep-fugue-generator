"""Interpolated n-gram baseline for the style prior.

A neural model that cannot beat this is not learning anything a count of the
previous few tokens does not already give you, which on a corpus of 615k tokens
is a real possibility worth ruling out before drawing conclusions from it.
"""

from __future__ import annotations

import collections
import math

from .style_data import is_content, load


def train(seqs, order=5):
    counts = [collections.defaultdict(collections.Counter) for _ in range(order + 1)]
    for s in seqs:
        ids = s["ids"]
        for i, t in enumerate(ids):
            for k in range(order + 1):
                if i - k < 0:
                    break
                counts[k][tuple(ids[i - k:i])][t] += 1
    return counts


def logprob(counts, ctx, tok, vocab_size, order, lam=0.75):
    # Simple backoff interpolation: longer contexts dominate where they have been
    # seen, shorter ones carry the rest, uniform at the bottom.
    p = 1.0 / vocab_size
    for k in range(0, order + 1):
        c = counts[k].get(tuple(ctx[len(ctx) - k:]) if k else ())
        if not c:
            continue
        tot = sum(c.values())
        if tot == 0:
            continue
        p = (1 - lam) * p + lam * (c.get(tok, 0) / tot)
    return math.log(max(p, 1e-12))


def evaluate(root=None, order=5):
    data, tokens, stoi = load(root)
    V = len(tokens)
    counts = train(data["train"], order=order)
    out = {}
    for split in ("validation", "test"):
        tot_lp = tot_n = 0.0
        c_lp = c_n = 0.0
        for s in data[split]:
            ids, toks = s["ids"], s["tokens"]
            for i, t in enumerate(ids):
                ctx = ids[max(0, i - order):i]
                lp = logprob(counts, ctx, t, V, order)
                tot_lp += lp; tot_n += 1
                if is_content(toks[i]):
                    c_lp += lp; c_n += 1
        out[split] = {
            "tokens": int(tot_n),
            "perplexity_all": round(math.exp(-tot_lp / max(1, tot_n)), 3),
            "content_tokens": int(c_n),
            "perplexity_content": round(math.exp(-c_lp / max(1, c_n)), 3),
        }
    out["order"] = order
    out["vocab"] = V
    return out
