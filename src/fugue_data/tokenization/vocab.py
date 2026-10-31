"""Token vocabulary and its hash (Spec §19.4).

The vocabulary is built from an explicit, closed set of token families rather
than harvested from whatever happens to appear in the corpus, so the same
tokenizer version always produces the same vocabulary hash regardless of which
pieces were tokenized.
"""

from __future__ import annotations

import hashlib

# Structural control tokens.
PAD, BOS, EOS, END_OF_SECTION = "<PAD>", "<BOS>", "<EOS>", "<END_OF_SECTION>"
REST, HOLD = "REST", "HOLD"

MIDI_LOW, MIDI_HIGH = 21, 108          # A0..C8, comfortably beyond the corpus
MAX_VOICES = 8
MAX_BEATS = 16                          # beats addressable within one bar
FIFTHS = range(-7, 8)
METERS = [(n, d) for d in (2, 4, 8, 16) for n in range(1, 13)]
ANNOTATION_TYPES = ["subject", "answer", "countersubject", "cadence", "pedal",
                    "episode", "other"]


def build_vocabulary() -> list[str]:
    """The closed token vocabulary, in a fixed order."""
    toks: list[str] = [PAD, BOS, EOS, END_OF_SECTION, REST, HOLD, "BAR"]
    toks += [f"NOTE_ON_{m}" for m in range(MIDI_LOW, MIDI_HIGH + 1)]
    toks += [f"VOICE_{i}" for i in range(1, MAX_VOICES + 1)]
    toks += [f"BEAT_{b}" for b in range(1, MAX_BEATS + 1)]
    toks += [f"KEY_{f}" for f in FIFTHS]
    toks += [f"METER_{n}_{d}" for n, d in METERS]
    toks += [f"ANN_{t.upper()}_START" for t in ANNOTATION_TYPES]
    toks += [f"ANN_{t.upper()}_END" for t in ANNOTATION_TYPES]
    return toks


def vocabulary_hash(vocab: list[str]) -> str:
    m = hashlib.sha256()
    for t in vocab:
        m.update(t.encode("utf-8"))
        m.update(b"\x1f")
    return m.hexdigest()


def vocabulary_index(vocab: list[str]) -> dict[str, int]:
    return {t: i for i, t in enumerate(vocab)}
