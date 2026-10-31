"""Parse a native score and build a canonical rational-time model (Spec §10).

Voice policy (Spec §8.4, and the §30 risk list): the true contrapuntal voices are
the encoded parts/spines. We do NOT infer voices or equate staves with voices.
Each part becomes one canonical voice; brief intra-part divisi splits and chords
are preserved as note events and any resulting partial overlap is recorded as an
explained warning rather than dropped or silently merged.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import warnings as _warnings
from fractions import Fraction
from pathlib import Path

import music21
from music21 import converter

from .. import SCHEMA_VERSION, __version__
from .model import CanonicalScore, Voice, Note
from .rational import to_fraction, frac_str
from .hashing import compute_hashes


class NormalizationError(RuntimeError):
    pass


def _config_hash() -> str:
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "software_version": __version__,
            "music21_version": music21.__version__,
            "voice_policy": "part_as_voice.v1",
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _measure_index(part) -> list[tuple[Fraction, Fraction, str, int]]:
    """(start_offset, end_offset, measure_number_str, unfolded_index) for a part."""
    out = []
    measures = list(part.getElementsByClass("Measure"))
    for idx, m in enumerate(measures):
        start = to_fraction(m.offset)
        dur = to_fraction(m.duration.quarterLength)
        num = m.measureNumberWithSuffix() if hasattr(m, "measureNumberWithSuffix") else str(m.measureNumber)
        out.append((start, start + dur, str(num), idx))
    return out


def _locate_measure(onset: Fraction, mindex) -> tuple[str | None, int | None, Fraction]:
    for start, end, num, idx in mindex:
        if start <= onset < end:
            return num, idx, onset - start
    if mindex:  # onset at/after final barline: attribute to last measure
        start, end, num, idx = mindex[-1]
        return num, idx, onset - start
    return None, None, onset


def _pitch_fields(p) -> dict:
    return {
        "midi": int(p.midi),
        "step": p.step,
        "alter": to_fraction(p.alter or 0),
        "octave": int(p.octave) if p.octave is not None else 4,
        "pitch_class": int(p.pitchClass),
        "diatonic_step": int(p.diatonicNoteNum),
    }


def _tie_type(n) -> str | None:
    if n.tie is None:
        return None
    return n.tie.type  # 'start' | 'continue' | 'stop'


def _extract_maps(score) -> tuple[list, list, list]:
    flat = score.flatten()
    meter_map = []
    for ts in flat.getElementsByClass("TimeSignature"):
        meter_map.append({
            "onset_q": frac_str(to_fraction(ts.offset)),
            "numerator": int(ts.numerator),
            "denominator": int(ts.denominator),
        })
    key_map = []
    for ks in flat.getElementsByClass("KeySignature"):
        mode = getattr(ks, "mode", None)
        key_map.append({
            "onset_q": frac_str(to_fraction(ks.offset)),
            "fifths": int(ks.sharps),
            "mode_if_known": mode,
        })
    tempo_map = []
    for mm in flat.getElementsByClass("MetronomeMark"):
        tempo_map.append({
            "onset_q": frac_str(to_fraction(mm.offset)),
            "bpm_if_known": float(mm.number) if mm.number is not None else None,
        })
    return meter_map, key_map, tempo_map


def _build_voice_notes(part, piece_id: str, part_key: str, mindex) -> list[Note]:
    """Collect canonical notes for one part; expand chords into note events."""
    notes: list[Note] = []
    flat = part.flatten()
    chord_counter = 0
    for el in flat.notes:  # Notes and Chords, in offset order
        onset = to_fraction(el.offset)
        dur = to_fraction(el.duration.quarterLength)
        is_grace = bool(el.duration.isGrace) or dur == 0
        num, idx, off_in_m = _locate_measure(onset, mindex)
        artics = [type(a).__name__ for a in getattr(el, "articulations", [])]
        orns = [type(o).__name__ for o in getattr(el, "expressions", [])]
        pitches = el.pitches  # 1 for Note, N for Chord
        chord_group = None
        if len(pitches) > 1:
            chord_group = f"{part_key}_ch{chord_counter}"
            chord_counter += 1
        for pi, p in enumerate(pitches):
            pf = _pitch_fields(p)
            nid = f"{piece_id}:{part_key}:{frac_str(onset)}:{pf['midi']}:{pi}"
            notes.append(Note(
                note_id=nid,
                voice_id=part_key,  # renamed to canonical voice id later
                onset_q=onset,
                duration_q=dur,
                measure_number=num,
                measure_index_unfolded=idx,
                offset_in_measure_q=off_in_m,
                tie=_tie_type(el),
                grace=is_grace,
                articulations=artics,
                ornaments=orns,
                chord_group=chord_group,
                source_refs=[{"part": part_key}],
                **pf,
            ))
    return notes


def _detect_overlaps(voice_id: str, notes: list[Note]) -> list[dict]:
    """Record partial monophonic overlaps as explained warnings (Spec §10.4).

    Chord members share an onset (and duration) and are not flagged; genuine
    divisi/suspension moments where a note starts before the previous one ends
    are preserved and recorded so no overlap is *unexplained*.
    """
    warns: list[dict] = []
    ordered = sorted(notes, key=lambda n: (n.onset_q, n.midi))
    prev_end = None
    prev_onset = None
    for n in ordered:
        if n.grace:
            continue
        if prev_end is not None and n.onset_q < prev_end and n.onset_q != prev_onset:
            warns.append({
                "code": "voice_partial_overlap",
                "message": f"partial overlap in {voice_id} at onset {frac_str(n.onset_q)} "
                           f"(previous note ends {frac_str(prev_end)}); divisi/suspension preserved",
                "voice_id": voice_id,
                "onset_q": frac_str(n.onset_q),
            })
        end = n.onset_q + n.duration_q
        if prev_end is None or end > prev_end:
            prev_end = end
        prev_onset = n.onset_q
    return warns


def normalize_score(
    path: str | Path,
    *,
    piece_id: str,
    work_group_id: str | None,
    movement_type: str,
    source_file_ids: list[str],
    composer_id: str = "bach_js",
    part_filter=None,
) -> CanonicalScore:
    """Parse ``path`` and build a canonical score.

    ``part_filter`` (optional) is a callable ``(index, part) -> bool`` selecting
    which parts become canonical voices. Used for chorales, whose files are full
    cantata movements: only the 4 vocal parts are the chorale; instruments and
    continuo are excluded (and recorded as ``excluded_parts``). When ``None``,
    every part becomes a voice (the WTC-fugue case).
    """
    path = Path(path)
    warnings_out: list[dict] = []

    # Capture every parser warning (Spec §10.4). music21's humdrum spine parser
    # emits some via the Python ``warnings`` module and some straight to stderr,
    # so we record both channels around parse *and* repeat expansion.
    stderr_buf = io.StringIO()
    repeats_expanded = False
    with _warnings.catch_warnings(record=True) as wlist, \
            contextlib.redirect_stderr(stderr_buf):
        _warnings.simplefilter("always")
        try:
            parsed = converter.parse(str(path))
        except Exception as e:  # pragma: no cover
            raise NormalizationError(f"{piece_id}: parse failed: {e}") from e

        # Expand repeats into performance order while retaining measure identities.
        score = parsed
        try:
            expanded = parsed.expandRepeats()
            if expanded is not None:
                score = expanded
                repeats_expanded = True
        except Exception:
            score = parsed  # no repeat structure, or unsupported: use as-is

    for w in wlist:
        warnings_out.append({
            "code": "parser_warning",
            "message": f"{w.category.__name__}: {w.message}",
            "voice_id": None, "onset_q": None,
        })
    for line in stderr_buf.getvalue().splitlines():
        line = line.strip()
        if "WARNING" in line or "Error in parsing" in line:
            warnings_out.append({
                "code": "parser_warning", "message": line,
                "voice_id": None, "onset_q": None,
            })

    all_parts = list(score.parts)
    if not all_parts:
        raise NormalizationError(f"{piece_id}: no parts found")

    excluded_parts: list[str] = []
    if part_filter is not None:
        parts = []
        for i, p in enumerate(all_parts):
            if part_filter(i, p):
                parts.append(p)
            else:
                excluded_parts.append(p.partName or p.id)
        if not parts:
            raise NormalizationError(f"{piece_id}: part_filter selected no parts")
    else:
        parts = all_parts

    meter_map, key_map, tempo_map = _extract_maps(score)

    raw_voices: list[Voice] = []
    for i, part in enumerate(parts):
        part_key = f"part{i}_{part.id}"
        mindex = _measure_index(part)
        notes = _build_voice_notes(part, piece_id, part_key, mindex)
        midis = [n.midi for n in notes if not n.grace]
        mean_midi = round(sum(midis) / len(midis), 3) if midis else None
        raw_voices.append(Voice(voice_id=part_key, staff=part.id, notes=notes, mean_midi=mean_midi))

    # Canonical voice ordering: highest mean pitch = voice_1 (soprano-down),
    # stable tiebreak on original part order.
    order = sorted(
        range(len(raw_voices)),
        key=lambda k: (-(raw_voices[k].mean_midi if raw_voices[k].mean_midi is not None else -1), k),
    )
    voices: list[Voice] = []
    for new_idx, k in enumerate(order, start=1):
        v = raw_voices[k]
        vid = f"voice_{new_idx}"
        for n in v.notes:
            n.voice_id = vid
        ordered_notes = sorted(v.notes, key=lambda n: (n.onset_q, n.midi))
        voices.append(Voice(voice_id=vid, staff=v.staff, notes=ordered_notes, mean_midi=v.mean_midi))
        warnings_out.extend(_detect_overlaps(vid, ordered_notes))

    all_notes = [n for v in voices for n in v.notes]
    total_ql = max((n.onset_q + n.duration_q for n in all_notes), default=Fraction(0))
    measure_count = max((len(_measure_index(p)) for p in parts), default=0)

    score_obj = CanonicalScore(
        schema_version=SCHEMA_VERSION,
        piece_id=piece_id,
        work_group_id=work_group_id,
        composer_id=composer_id,
        movement_type=movement_type,
        source_file_ids=source_file_ids,
        voices=voices,
        meter_map=meter_map,
        key_signature_map=key_map,
        tempo_map=tempo_map,
        warnings=warnings_out,
        staff_count=len(parts),
        quarter_length_total=total_ql,
        measure_count=measure_count,
    )
    if excluded_parts:
        warnings_out.append({
            "code": "parts_excluded",
            "message": f"non-chorale parts excluded from canonical voices: {excluded_parts}",
            "voice_id": None, "onset_q": None,
        })

    hashes = compute_hashes(score_obj)
    score_obj.normalization = {
        "software_version": __version__,
        "music21_version": music21.__version__,
        "config_hash": _config_hash(),
        "repeats_expanded": repeats_expanded,
        "excluded_parts": excluded_parts,
        **hashes,
    }
    return score_obj
