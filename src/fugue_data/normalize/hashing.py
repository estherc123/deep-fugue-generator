"""Deterministic content hashes over the canonical model (Spec §10.1 step 8, §12).

  core_event_hash          exact pitch+time+voice content (the canonical identity)
  event_hash               alias of core_event_hash for this notation profile
  rhythm_hash              onset/duration structure, pitch-independent
  transposition_melodic    per-voice interval sequences, transposition-invariant
  per_voice_hashes         one exact hash per voice (for dedup / leakage checks)

Tied-note handling: for melodic/rhythm hashes we merge tie continuations into the
initiating note so that a note split across a barline hashes the same as one that
is not (Spec §25.1 "tie merging").
"""

from __future__ import annotations

import hashlib
from fractions import Fraction

from .rational import frac_str
from .model import CanonicalScore, Note


def _h(parts) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update(str(p).encode("utf-8"))
        m.update(b"\x1f")
    return m.hexdigest()


def _merge_ties(notes: list[Note]) -> list[tuple[Fraction, Fraction, int]]:
    """Return (onset, effective_duration, midi) with tie-continuations merged."""
    merged: list[tuple[Fraction, Fraction, int]] = []
    ordered = sorted(notes, key=lambda n: (n.onset_q, n.midi))
    for n in ordered:
        if n.tie in ("continue", "stop") and merged and merged[-1][2] == n.midi:
            o, d, mid = merged[-1]
            merged[-1] = (o, d + n.duration_q, mid)
        else:
            merged.append((n.onset_q, n.duration_q, n.midi))
    return merged


def voice_event_hash(notes: list[Note]) -> str:
    rows = [
        (frac_str(n.onset_q), frac_str(n.duration_q), n.midi, n.tie or "")
        for n in sorted(notes, key=lambda n: (n.onset_q, n.midi))
    ]
    return _h(rows)


def voice_melodic_hash(notes: list[Note]) -> str:
    """Transposition-invariant: sequence of MIDI intervals between successive notes."""
    merged = _merge_ties(notes)
    seq = [m[2] for m in merged]
    intervals = [b - a for a, b in zip(seq, seq[1:])]
    return _h(intervals)


def voice_rhythm_hash(notes: list[Note]) -> str:
    merged = _merge_ties(notes)
    rows = [(frac_str(o), frac_str(d)) for o, d, _ in merged]
    return _h(rows)


def compute_hashes(score: CanonicalScore) -> dict:
    per_voice = {}
    per_voice_mel = {}
    per_voice_rhy = {}
    for v in score.voices:
        per_voice[v.voice_id] = voice_event_hash(v.notes)
        per_voice_mel[v.voice_id] = voice_melodic_hash(v.notes)
        per_voice_rhy[v.voice_id] = voice_rhythm_hash(v.notes)

    # Global hashes fold per-voice hashes in a stable voice order.
    order = sorted(per_voice)
    core = _h([f"{vid}:{per_voice[vid]}" for vid in order])
    rhythm = _h([f"{vid}:{per_voice_rhy[vid]}" for vid in order])
    melodic = _h([f"{vid}:{per_voice_mel[vid]}" for vid in order])
    return {
        "event_hash": core,
        "core_event_hash": core,
        "rhythm_hash": rhythm,
        "transposition_melodic_hash": melodic,
        "per_voice_hashes": per_voice,
    }


def notes_from_dicts(notes: list[dict]) -> list[Note]:
    """Rehydrate canonical-JSON note dicts into ``Note`` objects.

    Lets later stages (deduplication, splitting) recompute any hash directly from
    `data/canonical/*.json` using *these* functions rather than a reimplementation,
    so the two can never drift apart. Only the fields the hashes read are
    populated; the rest carry harmless defaults.
    """
    out: list[Note] = []
    for n in notes:
        out.append(Note(
            note_id=n["note_id"], voice_id=n["voice_id"],
            onset_q=Fraction(n["onset_q"]), duration_q=Fraction(n["duration_q"]),
            measure_number=n.get("measure_number"),
            measure_index_unfolded=n.get("measure_index_unfolded"),
            offset_in_measure_q=Fraction(n.get("offset_in_measure_q", "0")),
            midi=n["midi"], step=n["step"], alter=Fraction(n.get("alter", "0")),
            octave=n["octave"], pitch_class=n.get("pitch_class", n["midi"] % 12),
            diatonic_step=n.get("diatonic_step", 0), tie=n.get("tie"),
            grace=n.get("grace", False),
        ))
    return out
