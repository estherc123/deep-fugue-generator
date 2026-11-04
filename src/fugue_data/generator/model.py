"""Decoder-only transformer with per-position time/plan feature embeddings.

The features fed at input position i are those of the token being PREDICTED
(position i + 1): which voice it belongs to, where it falls in the bar and the
piece, and what the plan says that voice is doing then. The plan is given ahead
of time, so this is not leakage -- it is the conditioning. It means the model
always knows "the next decision is voice 3, beat 2, during an answer entry,
four beats before a cadence in the dominant".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import FEATURES


class Block(nn.Module):
    def __init__(self, d: int, heads: int, drop: float):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d), nn.Dropout(drop))
        self.drop = drop
        self.resid_drop = nn.Dropout(drop)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for z in (q, k, v))
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        a = a.transpose(1, 2).reshape(b, t, d)
        x = x + self.resid_drop(self.proj(a))
        return x + self.mlp(self.ln2(x))


class FugueGenerator(nn.Module):
    def __init__(self, vocab: int, d: int = 384, layers: int = 8, heads: int = 6,
                 ctx: int = 2048, drop: float = 0.15, features: dict | None = None):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        # A checkpoint carries its own feature table, so older models keep
        # loading after the encoder gains features; extra inputs are ignored.
        self.feat = nn.ModuleDict({k: nn.Embedding(n, d)
                                   for k, n in (features or FEATURES).items()})
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, feats: dict[str, torch.Tensor]):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for k, emb in self.feat.items():
            x = x + emb(feats[k])
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))
