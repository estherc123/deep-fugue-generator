"""Turn a solver proposal into a schema-valid canonical fragment (Spec §16.4).

The fragment is written in exactly the canonical score format so the *same*
validator, hashes and rule engine that judge real Bach judge it too -- there is no
separate, friendlier path for generated material.

§27: synthetic output is never labelled as Bach. `composer_id` is `synthetic` and
`source_type` says what it is on every record.
"""

from __future__ import annotations

from fractions import Fraction

from ..normalize.hashing import compute_hashes, notes_from_dicts
from ..normalize.model import CanonicalScore, Note, Voice
from ..normalize.rational import frac_str, parse_frac

_SHARP_SPELLING = [("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
                   ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0)]
_FLAT_SPELLING = [("C", 0), ("D", -1), ("D", 0), ("E", -1), ("E", 0), ("F", 0),
                  ("G", -1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0)]
_STEP_INDEX = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}


def spell(midi: int, prefer_flats: bool) -> tuple[str, int, int]:
    table = _FLAT_SPELLING if prefer_flats else _SHARP_SPELLING
    step, alter = table[midi % 12]
    octave = midi // 12 - 1
    # A B-sharp or C-flat spelling would shift the octave; this table avoids both.
    return step, alter, octave


def build_fragment_score(req: dict, solved: dict, piece_id: str) -> dict:
    num, den = req["meter"]["numerator"], req["meter"]["denominator"]
    beats_per_bar = Fraction(num * 4, den)
    prefer_flats = req["key"]["tonic_pc"] in (1, 3, 5, 8, 10)

    voices: list[Voice] = []
    for v in solved["voices"]:
        notes: list[Note] = []
        for n in v["notes"]:
            onset = parse_frac(n["onset_q"])
            bar_index = int(onset / beats_per_bar)
            step, alter, octave = spell(n["midi"], prefer_flats)
            notes.append(Note(
                note_id=n["note_id"], voice_id=n["voice_id"],
                onset_q=onset, duration_q=parse_frac(n["duration_q"]),
                measure_number=str(bar_index + 1),
                measure_index_unfolded=bar_index,
                offset_in_measure_q=onset - bar_index * beats_per_bar,
                midi=n["midi"], step=step, alter=Fraction(alter), octave=octave,
                pitch_class=n["midi"] % 12,
                diatonic_step=octave * 7 + _STEP_INDEX[step],
            ))
        voices.append(Voice(voice_id=v["voice_id"], staff=None, notes=notes,
                            mean_midi=(round(sum(x.midi for x in notes) / len(notes), 2)
                                       if notes else None)))

    total = parse_frac(solved["quarter_length_total"])
    score = CanonicalScore(
        schema_version="1.0.0",
        piece_id=piece_id,
        work_group_id=None,
        composer_id="synthetic",          # §27: never labelled as Bach
        movement_type="other",
        source_file_ids=[],
        voices=voices,
        meter_map=[{"onset_q": "0", "numerator": num, "denominator": den}],
        key_signature_map=[{"onset_q": "0", "fifths": req["key"].get("fifths", 0),
                            "mode_if_known": req["key"]["mode"]}],
        tempo_map=[],
        quarter_length_total=total,
        measure_count=int(total / beats_per_bar),
    )
    doc = score.to_dict()
    doc["normalization"] = {
        "software_version": "0.1.0",
        "config_hash": "synthetic",
        "repeats_expanded": False,
        **compute_hashes(score),
    }
    return doc
