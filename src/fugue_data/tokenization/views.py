"""Grid and event model views over a canonical score (Spec §19.2, §19.3).

Neither view replaces the canonical score: both are derived from it and both
record the canonical event hash they came from, so any sequence can be traced
back and re-derived (§19.1, §19.4).
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..normalize.rational import frac_str, parse_frac
from .vocab import END_OF_SECTION, HOLD, REST


class GridUnsupported(RuntimeError):
    """Raised when no configured denominator represents the piece exactly."""


def merge_tied(notes: list[dict]) -> list[tuple[Fraction, Fraction, int]]:
    """(onset, effective_duration, midi) with tie continuations folded in.

    Mirrors `normalize.hashing._merge_ties` for the canonical-JSON note form: a
    note split across a barline becomes one event, so the grid never re-attacks it.
    """
    merged: list[tuple[Fraction, Fraction, int]] = []
    for n in sorted(notes, key=lambda n: (parse_frac(n["onset_q"]), n["midi"])):
        onset = parse_frac(n["onset_q"])
        dur = parse_frac(n["duration_q"])
        if n.get("tie") in ("continue", "stop") and merged and merged[-1][2] == n["midi"]:
            o, d, m = merged[-1]
            merged[-1] = (o, d + dur, m)
        else:
            merged.append((onset, dur, n["midi"]))
    return merged


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def choose_denominator(score: dict, ladder: list[int]) -> int:
    """Smallest steps-per-quarter in `ladder` that represents the piece exactly.

    Never rounds: a piece whose rhythms do not divide evenly at any configured
    resolution is reported as unsupported rather than quantized (§19.2).
    """
    need = 1
    for v in score["voices"]:
        for n in v["notes"]:
            for f in (parse_frac(n["onset_q"]), parse_frac(n["duration_q"])):
                need = need * f.denominator // math.gcd(need, f.denominator)
    for d in ladder:
        if d % need == 0:
            return d
    raise GridUnsupported(
        f"no denominator in {ladder} represents 1/{need}-quarter rhythms exactly")


def meter_at(onset: Fraction, meter_map: list[dict]) -> dict | None:
    cur = None
    for m in meter_map:
        if parse_frac(m["onset_q"]) <= onset:
            cur = m
        else:
            break
    return cur


def _measure_starts(score: dict) -> list[tuple[int, str, Fraction]]:
    starts: dict[int, tuple[str, Fraction]] = {}
    for v in score["voices"]:
        for n in v["notes"]:
            idx = n.get("measure_index_unfolded")
            if idx is None:
                continue
            st = parse_frac(n["onset_q"]) - parse_frac(n["offset_in_measure_q"])
            prev = starts.get(idx)
            if prev is None or st < prev[1]:
                starts[idx] = (str(n.get("measure_number")), st)
    return [(i, num, st) for i, (num, st) in sorted(starts.items())]


def _annotation_events(annotations: list[dict], types: set[str],
                       accepted_only: bool) -> dict[Fraction, list[str]]:
    """Onset -> structural tokens opening or closing there."""
    out: dict[Fraction, list[str]] = {}
    for a in annotations or []:
        if a.get("type") not in types:
            continue
        if accepted_only and a.get("analytical_status") not in (None, "accepted"):
            continue
        start = parse_frac(a["onset_q"])
        out.setdefault(start, []).append(f"ANN_{a['type'].upper()}_START")
        if a.get("duration_q"):
            end = start + parse_frac(a["duration_q"])
            out.setdefault(end, []).append(f"ANN_{a['type'].upper()}_END")
    return {k: sorted(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Grid view (§19.2)
# --------------------------------------------------------------------------- #
def grid_tokens(score: dict, cfg: dict, annotations: list[dict] | None = None
                ) -> tuple[list[str], dict]:
    """Per-voice NOTE_ON / HOLD / REST on a fixed grid, with bar and beat tokens.

    Emitted time-major and fully explicit -- every step names its voice -- because
    §19.2 asks for a view suitable for debugging, not a compact one.
    """
    gcfg = cfg["grid"]
    denom = choose_denominator(score, list(gcfg["denominator_ladder"]))
    step = Fraction(1, denom)

    voices = score["voices"]
    total = parse_frac(score.get("quarter_length_total", "0"))
    n_steps = int(total / step)

    # Per voice: which step each note starts on, and which steps it sustains.
    # Ties are merged first -- a note written across a barline is one attack, not
    # two -- using the same rule the normalizer's content hashes use, so the grid
    # and the hashes agree about what counts as an event.
    onset_at: list[dict[int, int]] = [{} for _ in voices]
    sustain: list[set[int]] = [set() for _ in voices]
    collisions: list[dict] = []
    for vi, v in enumerate(voices):
        for onset, dur, midi in merge_tied(v["notes"]):
            s = int(onset / step)
            d = max(1, int(dur / step))
            if s in onset_at[vi] and onset_at[vi][s] != midi:
                # Two pitches attacking in one voice at one step: a divisi or a
                # cadential double-stop. The grid holds one pitch per voice per
                # step, so this is recorded, not silently dropped.
                collisions.append({"voice_id": v["voice_id"], "onset_q": frac_str(onset),
                                   "kept": max(onset_at[vi][s], midi),
                                   "dropped": min(onset_at[vi][s], midi)})
                onset_at[vi][s] = max(onset_at[vi][s], midi)
            else:
                onset_at[vi][s] = midi
            for k in range(s + 1, s + d):
                sustain[vi].add(k)

    # Bar positions come from the meter, not from where notes happen to be.
    # Deriving them from notes meant a bar in which every voice rests produced no
    # BAR token, so the running bar-start never advanced and the beat counter
    # carried on across bars -- emitting BEAT_17 and beyond, which are not in the
    # declared vocabulary. The BCFB chorales are cantata movements whose vocal
    # parts rest through instrumental bars, so this hit five of them.
    mm = score.get("meter_map") or []
    notated = {int(st / step): num for _, num, st in _measure_starts(score)}
    bar_at = {}
    cursor = Fraction(0)
    while cursor < total:
        m = meter_at(cursor, mm)
        bar_len = (Fraction(m["numerator"] * 4, m["denominator"]) if m
                   else Fraction(4))
        s_idx = int(cursor / step)
        bar_at[s_idx] = notated.get(s_idx, "")
        cursor += bar_len
        if bar_len <= 0:
            break
    ann_types = set(cfg.get("annotations", {}).get("token_types", []))
    accepted_only = bool(cfg.get("annotations", {}).get("accepted_status_only", True))
    ann_at = {}
    if gcfg.get("emit_structural_annotation_tokens") and annotations:
        for onset, toks in _annotation_events(annotations, ann_types, accepted_only).items():
            ann_at.setdefault(int(onset / step), []).extend(toks)

    toks: list[str] = []
    km = score.get("key_signature_map") or []
    if km:
        toks.append(f"KEY_{km[0]['fifths']}")
    if mm:
        toks.append(f"METER_{mm[0]['numerator']}_{mm[0]['denominator']}")

    cur_bar_start = Fraction(0)
    for s in range(n_steps):
        onset = s * step
        if s in bar_at and gcfg.get("emit_bar_tokens", True):
            toks.append("BAR")
            cur_bar_start = onset
        if s in ann_at:
            toks.extend(sorted(ann_at[s]))
        if gcfg.get("emit_beat_tokens", True):
            m = meter_at(onset, mm)
            beat_len = Fraction(4, m["denominator"]) if m else Fraction(1)
            off = onset - cur_bar_start
            if off % beat_len == 0:
                beat = int(off / beat_len) + 1
                toks.append(f"BEAT_{beat}")
        for vi, _ in enumerate(voices):
            toks.append(f"VOICE_{vi + 1}")
            if s in onset_at[vi]:
                toks.append(f"NOTE_ON_{onset_at[vi][s]}")
            elif s in sustain[vi]:
                toks.append(HOLD)
            else:
                toks.append(REST)
    toks.append(END_OF_SECTION)

    policy = {
        "view": "grid",
        "steps_per_quarter": denom,
        # Exact in time. `voice_collisions` counts intra-voice simultaneities the
        # one-pitch-per-voice-per-step grid cannot hold; the event view keeps them
        # all, which is why §19.1 forbids discarding the canonical layer.
        "exact": True,
        "quantization": "none; denominator chosen to represent the piece exactly",
        "steps": n_steps,
        "voice_collisions": len(collisions),
        "voice_collision_detail": collisions[:20],
        "voice_collision_detail_all": collisions,
    }
    return toks, policy


# --------------------------------------------------------------------------- #
# Event view (§19.3)
# --------------------------------------------------------------------------- #
def event_records(score: dict, cfg: dict, annotations: list[dict] | None = None
                  ) -> tuple[list[dict], dict]:
    """One record per note: voice, spelling, relative pitch, delta, duration, bar/beat."""
    ecfg = cfg.get("event", {})
    mm = score.get("meter_map") or []
    bars = _measure_starts(score)

    def bar_beat(onset: Fraction) -> tuple[str | None, str | None]:
        cur = None
        for _, num, st in bars:
            if st <= onset:
                cur = (num, st)
            else:
                break
        if cur is None:
            return None, None
        m = meter_at(onset, mm)
        beat_len = Fraction(4, m["denominator"]) if m else Fraction(1)
        return cur[0], frac_str((onset - cur[1]) / beat_len + 1)

    ann_spans = []
    for a in annotations or []:
        if cfg.get("annotations", {}).get("accepted_status_only", True) and \
                a.get("analytical_status") not in (None, "accepted"):
            continue
        start = parse_frac(a["onset_q"])
        end = start + parse_frac(a["duration_q"]) if a.get("duration_q") else start
        ann_spans.append((start, end, a.get("type"), a.get("voice_id")))

    out: list[dict] = []
    for v in score["voices"]:
        prev_onset = None
        prev = None
        for n in sorted(v["notes"], key=lambda n: (parse_frac(n["onset_q"]), n["midi"])):
            onset = parse_frac(n["onset_q"])
            bar, beat = bar_beat(onset)
            rec = {
                "voice_id": n["voice_id"],
                "onset_delta_q": frac_str(onset - prev_onset) if prev_onset is not None else None,
                "onset_q": frac_str(onset),
                "duration_q": n["duration_q"],
                "bar": bar,
                "beat": beat,
                "tie": n.get("tie"),
                "midi": n["midi"],
            }
            if ecfg.get("include_pitch_spelling", True):
                rec.update({"step": n["step"], "alter": n.get("alter", "0"),
                            "octave": n["octave"]})
            if ecfg.get("include_relative_pitch", True):
                rec["semitone_delta"] = None if prev is None else n["midi"] - prev["midi"]
                rec["diatonic_delta"] = (
                    None if prev is None
                    else n.get("diatonic_step", 0) - prev.get("diatonic_step", 0))
            if ecfg.get("include_section_metadata", True):
                rec["sections"] = sorted({
                    t for (s, e, t, vid) in ann_spans
                    if t and s <= onset < max(e, s + Fraction(1, 1000))
                    and (vid is None or vid == n["voice_id"])})
            out.append(rec)
            prev_onset, prev = onset, n

    out.sort(key=lambda r: (parse_frac(r["onset_q"]), r["voice_id"]))
    policy = {"view": "event", "quantization": "none; exact rational time preserved",
              "records": len(out)}
    return out, policy
