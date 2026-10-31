"""Align Algomus structural labels onto canonical scores (Spec §11).

Alignment is by musical position, never by filename alone (§11.1): the source
file's fugue index and BWV both have to agree with the canonical piece, and the
resulting onsets and voices are then checked against the actual notes of that
score. The measured evidence for each voice-attributed annotation is
`voice_note_coverage` -- the fraction of the annotation's span in which the
resolved voice actually sounds. A wrong staff-to-voice mapping collapses that
number, so it is both reported and gated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from ..normalize.rational import frac_str, parse_frac, to_fraction
from .dezrann import DezFile, load_config, map_label


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


ANNOTATION_SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Canonical-score helpers
# --------------------------------------------------------------------------- #
@dataclass
class MeasureEntry:
    index: int
    number: str
    start_q: Fraction


def measure_table(score: dict) -> list[MeasureEntry]:
    """Exact start onset of every unfolded measure, from the notes themselves.

    `onset_q - offset_in_measure_q` is the measure's downbeat for any note in it,
    so the table is exact and needs no re-derivation from the meter map.
    """
    starts: dict[int, tuple[str, Fraction]] = {}
    for v in score["voices"]:
        for n in v["notes"]:
            idx = n.get("measure_index_unfolded")
            if idx is None:
                continue
            start = parse_frac(n["onset_q"]) - parse_frac(n["offset_in_measure_q"])
            prev = starts.get(idx)
            if prev is None or start < prev[1]:
                starts[idx] = (str(n.get("measure_number")), start)
    return [MeasureEntry(i, num, st) for i, (num, st) in sorted(starts.items())]


def locate(onset: Fraction, table: list[MeasureEntry]) -> tuple[str | None, Fraction | None]:
    """Map an absolute onset to (notated measure number, offset within measure)."""
    found = None
    for m in table:
        if m.start_q <= onset:
            found = m
        else:
            break
    if found is None:
        return None, None
    return found.number, onset - found.start_q


def piece_quarter_length(score: dict) -> Fraction:
    total = score.get("quarter_length_total")
    if total:
        return parse_frac(total)
    end = Fraction(0)
    for v in score["voices"]:
        for n in v["notes"]:
            end = max(end, parse_frac(n["onset_q"]) + parse_frac(n["duration_q"]))
    return end


def voice_note_coverage(voice: dict, start: Fraction, end: Fraction) -> float | None:
    """Fraction of [start, end) in which this voice actually sounds notes.

    Overlapping notes (divisi, cadential chords) are merged rather than summed,
    so coverage stays a true proportion of elapsed time in 0..1.
    """
    if end <= start:
        return None
    covered = Fraction(0)
    cursor = start
    for n in voice["notes"]:
        onset = parse_frac(n["onset_q"])
        if onset >= end:
            break
        note_end = onset + parse_frac(n["duration_q"])
        lo, hi = max(onset, cursor), min(note_end, end)
        if hi > lo:
            covered += hi - lo
            cursor = max(cursor, hi)
        if cursor >= end:
            break
    return float(covered / (end - start))


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
class AlignmentError(RuntimeError):
    pass


def _sorted_labels(dez: DezFile) -> list[dict]:
    """Deterministic label order, so annotation ids are stable across runs."""
    return sorted(
        dez.labels,
        key=lambda l: (
            to_fraction(l.get("start", 0)),
            str(l.get("staff") or ""),
            str(l.get("type") or ""),
            str(l.get("tag") or ""),
        ),
    )


def align_dez(dez: DezFile, score: dict, cfg: dict) -> tuple[list[dict], list[dict]]:
    """Align one `.dez` analysis onto one canonical score.

    Returns (annotations, warnings). Raises AlignmentError when the source file
    and the canonical score are not the same movement (Spec §11.1).
    """
    # §11.1: agree on fugue index *and* BWV before trusting any position.
    if dez.piece_id != score["piece_id"]:
        raise AlignmentError(
            f"piece mismatch: {dez.path.name} -> {dez.piece_id}, score {score['piece_id']}")
    if dez.work_group_id != score.get("work_group_id"):
        raise AlignmentError(
            f"work mismatch for {dez.piece_id}: annotation {dez.work_group_id} "
            f"vs score {score.get('work_group_id')}")

    annotator = cfg["source"]["annotators"][dez.annotator_kind]
    min_cov = float(cfg["voice_mapping"]["min_voice_note_coverage"])
    voices = {v["voice_id"]: v for v in score["voices"]}
    voice_count = len(score["voices"])
    table = measure_table(score)
    total_q = piece_quarter_length(score)

    annotations: list[dict] = []
    warnings: list[dict] = []

    for i, raw in enumerate(_sorted_labels(dez)):
        mapped = map_label(raw, cfg)
        ann_id = f"{dez.piece_id}:{dez.annotator_kind}:{i:03d}"

        if mapped.unmapped:
            warnings.append({
                "code": "annotation_unmapped_label",
                "annotation_id": ann_id,
                "message": (f"raw type {raw.get('type')!r} tag {raw.get('tag')!r} has no "
                            f"entry in config/annotations.yaml; emitted as 'other'"),
            })

        onset = to_fraction(raw.get("start", 0))
        raw_dur = raw.get("duration")
        duration = to_fraction(raw_dur) if raw_dur is not None else None

        # --- §11.4: every annotation onset lies within the score -------------
        if onset < 0 or onset > total_q:
            warnings.append({
                "code": "annotation_onset_out_of_bounds",
                "annotation_id": ann_id,
                "message": f"onset {frac_str(onset)} outside piece [0, {frac_str(total_q)}]",
            })
            continue
        # A zero duration on a point-event type means "arrival", not "empty span".
        if duration == 0 and mapped.type in cfg.get("point_event_types", []):
            warnings.append({
                "code": "annotation_zero_duration_point_event",
                "annotation_id": ann_id,
                "message": (f"{mapped.type} label carries duration 0 where the other "
                            f"{mapped.type} labels omit it; read as a point event"),
            })
            duration = None

        # --- §11.4: every duration is positive -------------------------------
        if duration is not None and duration <= 0:
            warnings.append({
                "code": "annotation_non_positive_duration",
                "annotation_id": ann_id,
                "message": f"duration {frac_str(duration)} is not positive; dropped",
            })
            continue
        if duration is not None and onset + duration > total_q:
            # Clip rather than drop: a label may run to the final barline while
            # the last voice releases earlier. Recorded, never silent.
            warnings.append({
                "code": "annotation_span_clipped",
                "annotation_id": ann_id,
                "message": (f"span end {frac_str(onset + duration)} exceeds piece length "
                            f"{frac_str(total_q)}; clipped"),
            })
            duration = total_q - onset
            if duration <= 0:
                continue

        # --- §11.4: every voice-specific annotation resolves to a voice ------
        voice_id = None
        coverage = None
        staff = raw.get("staff")
        if staff is not None and str(staff).strip() != "":
            try:
                staff_idx = int(str(staff).strip())
            except ValueError:
                staff_idx = -1
            if 1 <= staff_idx <= voice_count:
                voice_id = f"voice_{staff_idx}"          # staff N -> voice_N, top-down
            else:
                warnings.append({
                    "code": "annotation_voice_unresolved",
                    "annotation_id": ann_id,
                    "message": (f"staff {staff!r} does not resolve to a voice in a "
                                f"{voice_count}-voice score"),
                })
                continue
            if duration is not None:
                coverage = voice_note_coverage(voices[voice_id], onset, onset + duration)
                if coverage is not None and coverage < min_cov:
                    warnings.append({
                        "code": "annotation_low_voice_coverage",
                        "annotation_id": ann_id,
                        "message": (f"{voice_id} sounds for only {coverage:.2f} of the "
                                    f"labelled span; staff-to-voice mapping suspect"),
                    })

        measure_number, position = locate(onset, table)

        annotations.append({
            "annotation_id": ann_id,
            "piece_id": dez.piece_id,
            "type": mapped.type,
            "voice_id": voice_id,
            "onset_q": frac_str(onset),
            "duration_q": frac_str(duration) if duration is not None else None,
            "measure_number_source": measure_number,
            "position_source": frac_str(position) if position is not None else None,
            "variant": mapped.variant,
            "confidence": mapped.confidence,
            "analytical_alternative_group": None,     # assigned below
            "source_name": cfg["source"]["name"],
            "source_reference": f"{dez.path.name}#{i}",
            "annotator": annotator,
            "analytical_status": mapped.analytical_status,
            "transformations": mapped.transformations,
            "subject_index": mapped.subject_index,
            "countersubject_index": mapped.countersubject_index,
            "segment": mapped.segment,
            "cadence_key_area": mapped.cadence_key_area,
            "cadence_type": mapped.cadence_type,
            "pedal_degree": mapped.pedal_degree,
            "entry_function": None,                   # derived later, never sourced
            "entry_function_derivation": None,
            "entry_head_diatonic_interval": None,
            "voice_note_coverage": coverage,
            "raw": {
                "type": raw.get("type"),
                "tag": raw.get("tag"),
                "staff": raw.get("staff"),
                "start": raw.get("start"),
                "duration": raw.get("duration"),
                "layers": raw.get("layers"),
            },
        })

    return annotations, warnings


# --------------------------------------------------------------------------- #
# Analytical ambiguity (Spec §11.3)
# --------------------------------------------------------------------------- #
def assign_alternative_groups(annotations: list[dict], piece_id: str) -> int:
    """Link competing readings of the same span into alternative groups.

    Two annotations compete when they share a canonical type and voice, their
    spans overlap, and they disagree analytically -- either one analyst accepted
    the span while another rejected it (`ignore`), or two annotators labelled it
    differently. Disagreements are never resolved by discarding one side (§11.3).
    """
    spans = [
        (a, parse_frac(a["onset_q"]),
         parse_frac(a["onset_q"]) + (parse_frac(a["duration_q"]) if a["duration_q"] else Fraction(0)))
        for a in annotations
    ]
    parent = {a["annotation_id"]: a["annotation_id"] for a in annotations}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    linked = set()
    for i, (a, a0, a1) in enumerate(spans):
        for b, b0, b1 in spans[i + 1:]:
            if a["type"] != b["type"] or a["voice_id"] != b["voice_id"]:
                continue
            if a1 <= b0 or b1 <= a0:                      # no temporal overlap
                continue
            disagrees = (a["analytical_status"] != b["analytical_status"]
                         or a["annotator"] != b["annotator"])
            if not disagrees:
                continue
            union(a["annotation_id"], b["annotation_id"])
            linked.add(a["annotation_id"])
            linked.add(b["annotation_id"])

    roots = sorted({find(x) for x in linked})
    group_of = {r: f"{piece_id}:alt:{n:02d}" for n, r in enumerate(roots, start=1)}
    for a in annotations:
        if a["annotation_id"] in linked:
            a["analytical_alternative_group"] = group_of[find(a["annotation_id"])]
    return len(roots)
