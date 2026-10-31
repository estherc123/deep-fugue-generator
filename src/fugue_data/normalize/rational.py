"""Exact rational time in quarter-note units (Spec §8.1).

Canonical musical time is never stored as floating point. music21 represents
durations either as exact ``Fraction`` (tuplets) or as floats that are exact in
binary (dyadic values: 1/2, 1/4, 3/4, ...). We route every value through
``music21.common.opFrac`` and snap to a bounded-denominator ``Fraction`` so that
both dyadic and tuplet rhythms are represented exactly.
"""

from __future__ import annotations

from fractions import Fraction

try:
    from music21.common import opFrac
except Exception:  # pragma: no cover - music21 optional at import time
    def opFrac(x):  # type: ignore
        return x

# Denominator ceiling: covers 2^n subdivisions and 3,5,7,9 tuplets and products
# thereof, comfortably beyond anything in the WTC / chorale corpus.
_MAX_DENOM = 2 ** 10 * 3 * 3 * 5 * 7


def to_fraction(value) -> Fraction:
    """Convert a music21 quarterLength/offset to an exact ``Fraction``."""
    if isinstance(value, Fraction):
        return value
    f = opFrac(value)
    if isinstance(f, Fraction):
        return f
    # Dyadic floats convert exactly; snap anything else to a musical denominator.
    frac = Fraction(f).limit_denominator(_MAX_DENOM)
    return frac


def parse_frac(text: str) -> Fraction:
    """Parse a canonical rational string (``"3/8"``, ``"2"``, ``"0"``) to Fraction."""
    return Fraction(text)


def frac_str(value) -> str:
    """Serialize a rational as a canonical string: ``"3/8"``, ``"2"``, ``"0"``."""
    f = value if isinstance(value, Fraction) else to_fraction(value)
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"
