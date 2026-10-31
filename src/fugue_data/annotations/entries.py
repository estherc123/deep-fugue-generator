"""Derived subject / answer classification for fugal entries.

Algomus labels every entry `S` and does not distinguish a subject (tonic-level)
statement from an answer (dominant-level). Spec §11.2 has an `answer` type and
§15.5 needs "permitted real/tonal answer transformation", so tonal function is
*computed* here from the transposition of each entry's head against the first
entry of the piece.

This is inference, not source data, so it never touches the source-given `type`
field: it lands in `entry_function` alongside `entry_function_derivation`, which
names the method. Nothing downstream may present it as annotated by Algomus
(Spec §27).
"""

from __future__ import annotations

from fractions import Fraction

from ..normalize.rational import parse_frac

# The label head may sit a fraction before the first note of the entry (the
# analysts round to the notated attack); accept a note starting within an
# eighth-note of the labelled onset as the entry's head.
_HEAD_TOLERANCE = Fraction(1, 2)


def _head_notes(voice: dict, onset: Fraction, end: Fraction, count: int) -> list[dict]:
    out = []
    for n in voice["notes"]:
        o = parse_frac(n["onset_q"])
        if o < onset - _HEAD_TOLERANCE:
            continue
        if o >= end or len(out) >= count:
            break
        if n.get("grace"):
            continue
        # Skip the tail of a note tied in from before the entry.
        if n.get("tie") in ("continue", "stop"):
            continue
        out.append(n)
    return out


def classify_entries(annotations: list[dict], score: dict, cfg: dict) -> dict:
    """Populate `entry_function` on subject-family annotations. Returns a summary."""
    ec = cfg.get("entry_function", {})
    if not ec.get("enabled", False):
        return {"enabled": False}

    count = int(ec.get("compare_head_notes", 4))
    answer_set = {i % 7 for i in ec.get("answer_diatonic_intervals", [4])}
    subject_set = {i % 7 for i in ec.get("subject_diatonic_intervals", [0])}
    voices = {v["voice_id"]: v for v in score["voices"]}

    entries = [
        a for a in annotations
        if a["type"] == "subject"
        and a["analytical_status"] == "accepted"
        and a["voice_id"] in voices
        and a["duration_q"] is not None
    ]
    entries.sort(key=lambda a: (parse_frac(a["onset_q"]), a["voice_id"]))
    if not entries:
        return {"enabled": True, "entries": 0, "subject": 0, "answer": 0, "undetermined": 0}

    def head(a):
        onset = parse_frac(a["onset_q"])
        end = onset + parse_frac(a["duration_q"])
        return _head_notes(voices[a["voice_id"]], onset, end, count)

    ref_head = head(entries[0])
    if not ref_head:
        return {"enabled": True, "entries": len(entries), "subject": 0, "answer": 0,
                "undetermined": len(entries), "note": "reference entry has no head notes"}
    ref_step = ref_head[0]["diatonic_step"]

    tally = {"subject": 0, "answer": 0, "undetermined": 0}
    for a in entries:
        h = head(a)
        if not h:
            tally["undetermined"] += 1
            continue
        interval = h[0]["diatonic_step"] - ref_step
        a["entry_head_diatonic_interval"] = interval
        mod = interval % 7
        if mod in subject_set:
            a["entry_function"] = "subject"
        elif mod in answer_set:
            a["entry_function"] = "answer"
        else:
            a["entry_function"] = None
        a["entry_function_derivation"] = (
            "computed: head diatonic interval vs first entry, mod 7"
        )
        tally[a["entry_function"] or "undetermined"] += 1

    return {"enabled": True, "entries": len(entries), **tally}
