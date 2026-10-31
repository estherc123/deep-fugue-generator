"""Human-readable per-fugue alignment report (Spec §11.4).

One Markdown file per aligned piece, showing every annotation in musical terms
-- bar and beat, voice, type -- next to the measured evidence that the alignment
landed on real notes. This is the artefact a musician reads to check the
alignment, so it is written for reading, not for parsing.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from ..normalize.rational import parse_frac


def _beat(position_q: str | None, meter: dict | None) -> str:
    """Offset-in-measure (quarters) rendered as a beat number in the local meter."""
    if position_q is None:
        return ""
    pos = parse_frac(position_q)
    denom = (meter or {}).get("denominator", 4)
    beat = pos * Fraction(denom, 4) + 1
    return str(beat) if beat.denominator == 1 else f"{float(beat):.3g}"


def _meter_at(onset: Fraction, meter_map: list[dict]) -> dict | None:
    current = None
    for m in meter_map:
        if parse_frac(m["onset_q"]) <= onset:
            current = m
        else:
            break
    return current


def _where(a: dict, meter_map: list[dict]) -> str:
    onset = parse_frac(a["onset_q"])
    meter = _meter_at(onset, meter_map)
    bar = a.get("measure_number_source") or "?"
    beat = _beat(a.get("position_source"), meter)
    return f"m{bar} b{beat}" if beat else f"m{bar}"


def _label(a: dict) -> str:
    parts = [a["type"]]
    if a.get("subject_index"):
        parts[0] = f"subject {a['subject_index']}"
    if a.get("countersubject_index"):
        parts[0] = f"countersubject {a['countersubject_index']}"
    if a.get("segment"):
        parts[0] += f" ({a['segment']})"
    # `other` covers source categories the canonical enum has no slot for
    # (codetta, and anything unmapped); name them from the source tag.
    if a["type"] == "other" and a.get("variant"):
        parts[0] = f"other ({a['variant']})"
    if a.get("cadence_type"):
        parts.append(f"{a.get('cadence_key_area')}:{a['cadence_type']}")
    if a.get("pedal_degree"):
        parts.append(f"on {a['pedal_degree']}")
    if a.get("transformations"):
        parts.append(", ".join(a["transformations"]))
    return " — ".join(parts)


def write_alignment_report(doc: dict, score: dict, path: Path) -> None:
    al = doc["alignment"]
    meter_map = score.get("meter_map", [])
    accepted = [a for a in doc["annotations"] if a["analytical_status"] == "accepted"]
    rejected = [a for a in doc["annotations"] if a["analytical_status"] == "rejected_candidate"]

    L: list[str] = []
    L.append(f"# Annotation alignment — {doc['piece_id']} ({doc.get('work_group_id')})")
    L.append("")
    L.append(f"**Source:** {doc['source_name']} · "
             f"**map version:** {doc['annotation_map_version']} · "
             f"**schema:** {doc['schema_version']}")
    L.append(f"**Score:** {al['voice_count']} voices, "
             f"{al['piece_quarter_length']} quarter-notes, "
             f"{score.get('measure_count')} bars")
    L.append(f"**Voice mapping:** `{al['voice_mapping_strategy']}` "
             f"(Dezrann staff *N* → canonical `voice_N`, both ordered highest-first)")
    cov = al.get("mean_voice_note_coverage")
    mn = al.get("min_voice_note_coverage")
    if cov is not None:
        L.append(f"**Alignment evidence:** voice-attributed labels land on sounding notes "
                 f"{cov:.1%} of their span on average (worst {mn:.1%}). "
                 f"A wrong staff→voice mapping would collapse this figure.")
    L.append("")

    L.append("## Accepted annotations")
    L.append("")
    counts = ", ".join(f"{k} {v}" for k, v in al["counts_by_type"].items())
    L.append(f"{len(accepted)} labels — {counts}.")
    L.append("")
    L.append("| where | span | type | voice | entry function | coverage |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for a in accepted:
        dur = a["duration_q"] or "—"
        voice = a["voice_id"] or "—"
        ef = a.get("entry_function") or ("—" if a["type"] != "subject" else "undetermined")
        if a["type"] != "subject":
            ef = "—"
        c = a.get("voice_note_coverage")
        L.append(f"| {_where(a, meter_map)} | {dur} | {_label(a)} | {voice} | {ef} | "
                 f"{'' if c is None else f'{c:.0%}'} |")
    L.append("")

    ef = al.get("entry_function", {})
    if ef.get("enabled"):
        L.append("### Derived subject / answer")
        L.append("")
        L.append(f"Algomus labels every entry `S`; tonal function below is **computed**, not "
                 f"annotated — the head note of each entry compared against the first entry "
                 f"(mod 7). Of {ef.get('entries', 0)} entries: "
                 f"**{ef.get('subject', 0)} at subject (tonic) level**, "
                 f"**{ef.get('answer', 0)} at answer (dominant) level**, "
                 f"{ef.get('undetermined', 0)} at other transposition levels "
                 f"(middle entries in related keys), which this method leaves undetermined "
                 f"rather than guessing.")
        L.append("")

    if rejected:
        L.append("## Rejected candidates (Dezrann `ignore`)")
        L.append("")
        L.append(f"{len(rejected)} spans the analyst explicitly considered and declined to "
                 f"count as entries. Retained as near-miss evidence, excluded from accepted "
                 f"totals and from default exports.")
        L.append("")
        L.append("| where | span | would-be type | voice |")
        L.append("| --- | --- | --- | --- |")
        for a in rejected:
            L.append(f"| {_where(a, meter_map)} | {a['duration_q'] or '—'} | {_label(a)} | "
                     f"{a['voice_id'] or '—'} |")
        L.append("")

    if al.get("alternative_group_count"):
        L.append("## Analytical alternatives (Spec §11.3)")
        L.append("")
        L.append(f"{al['alternative_group_count']} group(s) of competing readings of the same "
                 f"span, preserved rather than resolved.")
        L.append("")
        groups: dict[str, list[dict]] = {}
        for a in doc["annotations"]:
            g = a.get("analytical_alternative_group")
            if g:
                groups.setdefault(g, []).append(a)
        for g, members in sorted(groups.items()):
            L.append(f"- **{g}** — " + "; ".join(
                f"{_where(m, meter_map)} {_label(m)} ({m['analytical_status']})"
                for m in members))
        L.append("")

    if doc.get("warnings"):
        L.append("## Warnings")
        L.append("")
        for w in doc["warnings"]:
            L.append(f"- `{w['code']}` {w.get('annotation_id') or ''}: {w['message']}")
        L.append("")
    else:
        L.append("## Warnings")
        L.append("")
        L.append("None.")
        L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))
