"""Load the tokenized corpus for style modelling, split-respecting (Spec §13.5).

The test pieces are whole works held out before any window was cut, so a model
evaluated here has never seen a note of them -- not a transposition, not an
overlapping window. That is what the complete-work split exists to guarantee.
"""

from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Tokens that carry musical choice. The grid interleaves VOICE_n markers and
# BAR/BEAT tokens in a fixed pattern, so a model predicts those almost for free;
# reporting perplexity over the whole stream flatters it. Content perplexity --
# over the pitch and articulation decisions only -- is the honest number.
def is_content(tok: str) -> bool:
    return tok.startswith("NOTE_ON_") or tok in ("HOLD", "REST")


def load(root: Path | None = None):
    root = root or _repo_root()
    splits = {}
    for line in (root / "manifests" / "splits.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            splits[r["piece_id"]] = r["split"]

    vocab = json.loads((root / "data" / "tokenized" / "vocabulary.json").read_text())
    stoi = {t: i for i, t in enumerate(vocab["tokens"])}

    out = {"train": [], "validation": [], "test": []}
    for f in sorted((root / "data" / "tokenized").glob("*.json")):
        if f.name == "vocabulary.json":
            continue
        d = json.loads(f.read_text())
        s = splits.get(d["piece_id"])
        toks = (d.get("grid") or {}).get("tokens")
        if not s or not toks:
            continue
        out[s].append({"piece_id": d["piece_id"], "tokens": toks,
                       "ids": [stoi[t] for t in toks]})
    return out, vocab["tokens"], stoi
