"""The unit every rule is evaluated over (Spec §15).

A fragment is the same shape as a canonical score -- `voices` of note dicts in
exact rational time -- so one engine serves real Bach, proposed excerpts and
generated material alike. `plan` carries the structural targets a generated
fragment was asked to hit (scheduled entries, cadence target); real Bach simply
has none, and the structural-target rules stay silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from ..normalize.rational import parse_frac
from ..tokenization.views import merge_tied


@dataclass
class Fragment:
    piece_id: str
    voices: list[dict]
    quarter_length_total: Fraction
    meter: tuple[int, int] = (4, 4)
    key_fifths: int = 0
    mode: str = "major"
    tonic_pc: int = 0
    measure_count: int | None = None
    # Which range profile applies: keyboard for the WTC, satb for the chorales.
    range_profile: str = "keyboard"
    # True when parts present in the source were deliberately not imported, so the
    # remaining voices legitimately fall silent while the excluded ones play.
    is_extract: bool = False
    plan: dict = field(default_factory=dict)

    @classmethod
    def from_score(cls, score: dict, plan: dict | None = None) -> "Fragment":
        mm = (score.get("meter_map") or [{"numerator": 4, "denominator": 4}])[0]
        km = (score.get("key_signature_map") or [{"fifths": 0}])[0]
        return cls(
            piece_id=score["piece_id"],
            voices=score["voices"],
            quarter_length_total=parse_frac(score.get("quarter_length_total", "0")),
            meter=(mm["numerator"], mm["denominator"]),
            key_fifths=km["fifths"],
            measure_count=score.get("measure_count"),
            range_profile=("satb" if score.get("movement_type") == "chorale"
                           else "keyboard"),
            is_extract=bool((score.get("normalization") or {}).get("excluded_parts")),
            plan=plan or {},
        )

    def voice_ids(self) -> list[str]:
        return [v["voice_id"] for v in self.voices]

    def merged(self, voice: dict) -> list[tuple[Fraction, Fraction, int]]:
        """Tie-merged (onset, duration, midi) events for one voice."""
        return merge_tied(voice["notes"])

    def notes_sorted(self, voice: dict) -> list[dict]:
        return sorted(voice["notes"], key=lambda n: (parse_frac(n["onset_q"]), n["midi"]))

    # ---- cached indices -------------------------------------------------- #
    # Rules query "what is sounding at time t" across every onset for every voice
    # pair, so a naive scan is quadratic in notes and made evaluation ~7s per
    # fugue. The slice table is built once per fragment and shared by all rules.
    _slices: dict | None = field(default=None, repr=False, compare=False)
    _onsets: list | None = field(default=None, repr=False, compare=False)

    def _build_index(self) -> None:
        spans = []
        for v in self.voices:
            for n in v["notes"]:
                o = parse_frac(n["onset_q"])
                spans.append((o, o + parse_frac(n["duration_q"]), v["voice_id"], n))
        onsets = sorted({s[0] for s in spans})
        spans.sort(key=lambda s: s[0])
        table: dict = {}
        # Sweep: a note is a candidate from its onset until its end.
        active: list = []
        i = 0
        for t in onsets:
            while i < len(spans) and spans[i][0] <= t:
                active.append(spans[i])
                i += 1
            active = [a for a in active if a[1] > t]
            slice_: dict = {}
            for _, _, vid, n in active:
                # Highest sounding pitch represents a voice that briefly divides.
                if vid not in slice_ or n["midi"] > slice_[vid]["midi"]:
                    slice_[vid] = n
            table[t] = slice_
        object.__setattr__(self, "_slices", table)
        object.__setattr__(self, "_onsets", onsets)

    def sounding_at(self, t: Fraction) -> list[tuple[str, dict]]:
        if self._slices is None:
            self._build_index()
        hit = self._slices.get(t)
        if hit is not None:
            return list(hit.items())
        out = []
        for v in self.voices:
            for n in v["notes"]:
                o = parse_frac(n["onset_q"])
                if o <= t < o + parse_frac(n["duration_q"]):
                    out.append((v["voice_id"], n))
                    break
        return out

    def onsets(self) -> list[Fraction]:
        if self._onsets is None:
            self._build_index()
        return list(self._onsets)
