"""Canonical score data model (Spec §8). Serializes deterministically to JSON."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from fractions import Fraction

from .rational import frac_str


@dataclass
class Note:
    note_id: str
    voice_id: str
    onset_q: Fraction
    duration_q: Fraction
    measure_number: str | None
    measure_index_unfolded: int | None
    offset_in_measure_q: Fraction
    midi: int
    step: str
    alter: Fraction
    octave: int
    pitch_class: int
    diatonic_step: int
    tie: str | None = None
    grace: bool = False
    articulations: list[str] = field(default_factory=list)
    ornaments: list[str] = field(default_factory=list)
    chord_group: str | None = None
    source_refs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "note_id": self.note_id,
            "voice_id": self.voice_id,
            "onset_q": frac_str(self.onset_q),
            "duration_q": frac_str(self.duration_q),
            "measure_number": self.measure_number,
            "measure_index_unfolded": self.measure_index_unfolded,
            "offset_in_measure_q": frac_str(self.offset_in_measure_q),
            "midi": self.midi,
            "step": self.step,
            "alter": frac_str(self.alter),
            "octave": self.octave,
            "pitch_class": self.pitch_class,
            "diatonic_step": self.diatonic_step,
            "tie": self.tie,
            "grace": self.grace,
            "articulations": self.articulations,
            "ornaments": self.ornaments,
            "chord_group": self.chord_group,
            "source_refs": self.source_refs,
        }


@dataclass
class Voice:
    voice_id: str
    staff: str | int | None
    notes: list[Note] = field(default_factory=list)
    mean_midi: float | None = None

    def to_dict(self) -> dict:
        return {
            "voice_id": self.voice_id,
            "staff": self.staff,
            "mean_midi": self.mean_midi,
            "notes": [n.to_dict() for n in self.notes],
        }


@dataclass
class CanonicalScore:
    schema_version: str
    piece_id: str
    work_group_id: str | None
    composer_id: str
    movement_type: str
    source_file_ids: list[str]
    voices: list[Voice] = field(default_factory=list)
    meter_map: list[dict] = field(default_factory=list)
    key_signature_map: list[dict] = field(default_factory=list)
    tempo_map: list[dict] = field(default_factory=list)
    annotations: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    staff_count: int | None = None
    quarter_length_total: Fraction = Fraction(0)
    measure_count: int | None = None
    normalization: dict = field(default_factory=dict)

    @property
    def voice_count(self) -> int:
        return len(self.voices)

    def all_notes(self) -> list[Note]:
        return [n for v in self.voices for n in v.notes]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "piece_id": self.piece_id,
            "work_group_id": self.work_group_id,
            "composer_id": self.composer_id,
            "movement_type": self.movement_type,
            "source_file_ids": self.source_file_ids,
            "voice_count": self.voice_count,
            "staff_count": self.staff_count,
            "voices": [v.to_dict() for v in self.voices],
            "meter_map": self.meter_map,
            "key_signature_map": self.key_signature_map,
            "tempo_map": self.tempo_map,
            "annotations": self.annotations,
            "quarter_length_total": frac_str(self.quarter_length_total),
            "measure_count": self.measure_count,
            "warnings": self.warnings,
            "normalization": self.normalization,
        }
