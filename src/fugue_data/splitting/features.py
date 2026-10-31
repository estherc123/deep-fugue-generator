"""Piece-level features used for split stratification (Spec §13.2).

Everything here is derived from the canonical score alone, so a piece's
stratification fields are reproducible from `data/canonical/<piece_id>.json`
without re-parsing any source.
"""

from __future__ import annotations

import collections
from fractions import Fraction

from ..normalize.rational import parse_frac

CORPUS_WTC = "wtc"
CORPUS_BCFB = "bcfb"


def corpus_of(piece_id: str) -> str:
    return CORPUS_WTC if "_fugue_" in piece_id else CORPUS_BCFB


def book_of(piece_id: str) -> int | None:
    if piece_id.startswith("bach_wtc1"):
        return 1
    if piece_id.startswith("bach_wtc2"):
        return 2
    return None


def meter_of(score: dict) -> str:
    mm = score.get("meter_map") or []
    if not mm:
        return "unknown"
    return f"{mm[0]['numerator']}/{mm[0]['denominator']}"


def derive_mode(score: dict) -> str:
    """Major or minor, from a whole-piece pitch-class histogram around the tonic.

    The tonic is the final note of the lowest voice -- reliable for both fugues
    and chorales, whose final sonority is the tonic. Mode then comes from whether
    the minor or major third above that tonic is more common across the *whole*
    piece, which is why a Picardy third (a major third in the final chord of a
    minor-mode piece) does not flip the result.

    `mode_if_known` is null throughout the corpus -- neither the Humdrum WTC nor
    the BCFB MusicXML records mode -- so this is derived, and labelled as such in
    the split manifest. It is validated against ground truth: the WTC's own
    ordering fixes the mode of all 48 fugues (fugue 01 C major, 02 C minor, 03
    C-sharp major, ... so odd indices are major, even are minor), and this rule
    reproduces that for 48 of 48. See tests/unit/test_splitting.py.
    """
    bass = score["voices"][-1]                      # voices ordered highest -> lowest
    if not bass["notes"]:
        return "unknown"
    final = max(bass["notes"], key=lambda n: parse_frac(n["onset_q"]))
    tonic = final["pitch_class"]
    pcs = collections.Counter(
        n["pitch_class"] for v in score["voices"] for n in v["notes"])
    return "minor" if pcs[(tonic + 3) % 12] > pcs[(tonic + 4) % 12] else "major"


def length_bucket(measure_count: int | None, thresholds: tuple[int, int]) -> str:
    if measure_count is None:
        return "unknown"
    lo, hi = thresholds
    if measure_count <= lo:
        return "short"
    if measure_count <= hi:
        return "medium"
    return "long"


def piece_features(score: dict) -> dict:
    """Stratification fields for one canonical score (length bucket added later,
    since its thresholds are corpus-relative)."""
    return {
        "piece_id": score["piece_id"],
        "work_group_id": score.get("work_group_id"),
        "corpus": corpus_of(score["piece_id"]),
        "book": book_of(score["piece_id"]),
        "mode": derive_mode(score),
        "mode_source": "derived_pitch_class_histogram",
        "voice_count": score["voice_count"],
        "meter": meter_of(score),
        "measure_count": score.get("measure_count"),
        "quarter_length_total": score.get("quarter_length_total"),
    }


def length_thresholds(measure_counts: list[int]) -> tuple[int, int]:
    """Corpus terciles, so 'short/medium/long' means something within each corpus."""
    xs = sorted(m for m in measure_counts if m is not None)
    if not xs:
        return (0, 0)
    return xs[len(xs) // 3], xs[(2 * len(xs)) // 3]
