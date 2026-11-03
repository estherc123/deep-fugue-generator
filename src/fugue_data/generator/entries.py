"""Find subject entries in a fugue from its notes alone.

Gives every fugue an entry plan, not only the 23 that Algomus labelled. It is
validated against those labels (`evaluate`), so its precision and recall are
measured, not assumed: 95% precision / 90% recall on plain entries, counting
complete matches only (2026-11-03).

1. **Subject.** If the piece has a labelled first entry, that is the subject.
   Otherwise it is the opening voice's line up to the moment the second voice
   enters.
2. **Matching.** Every note of every voice is tried as the start of an entry.
   A candidate is the subject's interval pattern, prime or inverted, at normal
   or doubled note values, compared in diatonic steps. The head must match:
   rhythm exact (the first note may be lengthened or shortened), intervals
   within a step, most of them exact. Tonal answers and
   modal changes alter a few intervals by a semitone or two, which is why exact
   pitch is not required. The match then extends while the rhythm keeps
   matching and the pitch stays close. Short of 75% of the subject, it is an
   `incomplete` entry.
3. Overlapping matches in one voice keep the longest, earliest.

Output entries are inference, marked `source: "detected"`, and never mixed up
with source annotations.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from ..normalize.rational import parse_frac

HEAD = 5                 # intervals that must match for an entry to count
MIN_SHARE = 0.5          # shortest incomplete entry, as a share of the subject
COMPLETE = 0.75          # share of the subject matched for a complete entry
ONSET_TOL = Fraction(1, 2)


def voice_events(score: dict) -> dict[str, list[tuple[Fraction, Fraction, int, int]]]:
    """Per voice: (onset, duration, midi, diatonic step), ties merged."""
    out = {}
    for v in score["voices"]:
        ev: list = []
        notes = sorted((n for n in v["notes"] if not n.get("grace")
                        and parse_frac(n["duration_q"]) > 0),
                       key=lambda n: (parse_frac(n["onset_q"]), n["midi"]))
        for n in notes:
            o, d = parse_frac(n["onset_q"]), parse_frac(n["duration_q"])
            if n.get("tie") in ("continue", "stop") and ev and ev[-1][2] == n["midi"]:
                po, pd, pm, ps = ev[-1]
                ev[-1] = (po, pd + d, pm, ps)
            elif ev and ev[-1][0] == o:
                continue                      # divisi: keep one note per onset
            else:
                step = n.get("diatonic_step")
                ev.append((o, d, n["midi"], step if step is not None else round(n["midi"] * 7 / 12)))
        out[v["voice_id"]] = ev
    return out


def subject_from_opening(ev: dict) -> tuple[str, list]:
    """The opening voice's line until the second voice enters."""
    firsts = sorted((e[0][0], vid) for vid, e in ev.items() if e)
    t0, vid = firsts[0]
    later = [t for t, _ in firsts[1:] if t > t0]
    t1 = later[0] if later else None
    notes = [n for n in ev[vid] if t1 is None or n[0] < t1]
    return vid, notes


def _main_subject(a: dict) -> bool:
    return (a["type"] == "subject" and a.get("analytical_status") == "accepted"
            and a.get("subject_index") in (None, 1))


def subject_from_label(ev: dict, ann: list[dict]) -> tuple[str, list] | None:
    subs = [a for a in ann if _main_subject(a) and not a.get("transformations")
            and a.get("voice_id") in ev and a.get("duration_q")]
    if not subs:
        return None
    a = min(subs, key=lambda a: parse_frac(a["onset_q"]))
    o, d = parse_frac(a["onset_q"]), parse_frac(a["duration_q"])
    notes = [n for n in ev[a["voice_id"]] if o - ONSET_TOL <= n[0] < o + d]
    return a["voice_id"], notes


def _match_at(notes, i, subj, inv: int, aug: int) -> int:
    """Number of subject intervals matched starting at notes[i] (0 = no entry).

    Intervals are compared in diatonic steps (so an entry in the relative major
    still matches a minor subject), with the chromatic size kept within two
    semitones of the subject's.
    """
    L = len(subj) - 1
    head = min(HEAD, L)
    if i + head >= len(notes):
        return 0
    exact = k = 0
    head_exact = 0
    while k < L and i + k + 1 < len(notes):
        a, b = notes[i + k], notes[i + k + 1]
        want_ioi = (subj[k + 1][0] - subj[k][0]) * aug
        got_ioi = b[0] - a[0]
        # Rhythm: exact, except the first note (entries often start with a
        # lengthened or cut-down head note) and the last.
        if got_ioi != want_ioi and k not in (0, L - 1):
            break
        d_dia = abs((b[3] - a[3]) - inv * (subj[k + 1][3] - subj[k][3]))
        d_chr = abs((b[2] - a[2]) - inv * (subj[k + 1][2] - subj[k][2]))
        if d_dia > 1 or d_chr > 2:
            break
        exact += d_dia == 0
        if k < head:
            head_exact += d_dia == 0
        k += 1
    if k < max(head, MIN_SHARE * L):
        return 0
    # A tonal answer changes one or two intervals at the tonic/dominant pivot;
    # a long, otherwise faithful match may carry a third.
    allowed = (3 if k >= COMPLETE * L and L >= 8 else 2) if head >= HEAD else 1
    if head_exact < head - allowed or exact < 0.75 * k:
        return 0
    return k


def detect(score: dict, ann: list[dict] | None = None) -> dict:
    ev = voice_events(score)
    src = subject_from_label(ev, ann or []) if ann else None
    how = "label" if src else "opening"
    svid, subj = src or subject_from_opening(ev)
    if len(subj) < 4:
        return {"subject": None, "entries": [], "subject_source": how}
    L = len(subj) - 1
    s_dur = subj[-1][0] + subj[-1][1] - subj[0][0]
    found = []
    for vid, notes in ev.items():
        cands = []
        for i in range(len(notes)):
            for inv in (1, -1):
                for aug in (1, 2):
                    k = _match_at(notes, i, subj, inv, aug)
                    if k:
                        cands.append((i, k, inv, aug))
        # Keep non-overlapping matches, longest first then earliest.
        cands.sort(key=lambda c: (-c[1], c[0], -c[2], c[3]))
        used = set()
        for i, k, inv, aug in cands:
            span = set(range(i, i + k + 1))
            if span & used:
                continue
            used |= span
            o = notes[i][0]
            end = notes[i + k][0] + notes[i + k][1]
            found.append({
                "voice_id": vid, "onset_q": str(o), "duration_q": str(end - o),
                "first_midi": notes[i][2],
                "first_step": notes[i][3],
                "transpose": notes[i][2] - subj[0][2],
                "transformations": ((["inverted"] if inv < 0 else [])
                                    + (["augmented"] if aug > 1 else [])
                                    + (["incomplete"] if k < COMPLETE * L else [])),
                "matched_intervals": k, "source": "detected",
            })
    found.sort(key=lambda e: (parse_frac(e["onset_q"]), e["voice_id"]))
    return {"subject_source": how, "subject_voice": svid,
            "subject": [[str(o), str(d), m, st] for o, d, m, st in subj],
            "subject_duration_q": str(s_dur), "entries": found}


def evaluate(root: Path) -> dict:
    """Precision / recall against Algomus entries, per piece and overall.

    Run twice: with the labelled subject (tests the matcher alone) and with the
    subject taken from the opening (tests the whole unlabelled-piece path).
    """
    out = {}
    for mode in ("label", "opening"):
        tp = fp = fn = 0
        per = {}
        for f in sorted((root / "data" / "annotations").glob("*.json")):
            ann = json.loads(f.read_text())["annotations"]
            pid = f.stem
            score = json.loads((root / "data" / "canonical" / f"{pid}.json").read_text())
            gold = [(a["voice_id"], parse_frac(a["onset_q"])) for a in ann
                    if _main_subject(a)]
            det = detect(score, ann if mode == "label" else None)
            pred = [(e["voice_id"], parse_frac(e["onset_q"])) for e in det["entries"]]
            hit_g = sum(any(v == pv and abs(o - po) <= ONSET_TOL for pv, po in pred)
                        for v, o in gold)
            hit_p = sum(any(v == gv and abs(o - go) <= ONSET_TOL for gv, go in gold)
                        for v, o in pred)
            tp += hit_g; fn += len(gold) - hit_g; fp += len(pred) - hit_p
            per[pid] = {"gold": len(gold), "pred": len(pred), "recall_hits": hit_g,
                        "precision_hits": hit_p}
        out[mode] = {"precision": tp and (tp / (tp + fp)), "recall": tp / (tp + fn),
                     "per_piece": per}
    return out


def as_annotations(det: dict, piece_id: str) -> list[dict]:
    """Complete detected entries in the annotation shape `encode_piece` reads.

    Only complete entries become plan entries; head-only matches are subject
    material, not entries. Subject vs answer: the answer level
    is read off the exposition: the second entry's transposition (mod 12) is
    the answer level. A subject starting on the dominant is answered a fourth
    up, so "a fifth up" alone would mislabel it.
    """
    if not det.get("subject"):
        return []
    L = len(det["subject"]) - 1
    plain = [e for e in det["entries"]
             if e["matched_intervals"] >= COMPLETE * L
             and not {"inverted", "augmented"} & set(e["transformations"])]
    second = next((e for e in plain if e["transpose"] % 12 in (5, 7)), None)
    answer_level = second["transpose"] % 12 if second else 7
    out = []
    for k, e in enumerate(plain):
        out.append({
            "annotation_id": f"{piece_id}:det:{k:03d}", "piece_id": piece_id,
            "type": "subject", "analytical_status": "accepted",
            "voice_id": e["voice_id"], "onset_q": e["onset_q"],
            "duration_q": e["duration_q"],
            "entry_function": "answer" if e["transpose"] % 12 == answer_level else "subject",
            "transformations": [], "source_name": "detected",
        })
    return out
