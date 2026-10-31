"""Take the motif from the prior; let the solver build the counterpoint.

The prior degrades over four bars -- its openings are coherent and its endings
drift, which is what listening to unconstrained samples makes obvious. A motif is
one or two bars long, which is exactly the span where it still holds together. So
use it only there, and hand the long-range work (entries, spacing, harmony,
cadence) to CP-SAT, which enforces every hard rule by construction.

This needs no new solver machinery. `solve_fragment` already takes a subject as
(intervals, rhythm) and adds the realized subject pitches to the voice domain, so
a chromatic motif widens the domain rather than making the model infeasible. All
this module does is produce better (intervals, rhythm) pairs than a random walk
over the scale.
"""

from __future__ import annotations

import hashlib
import json
from itertools import zip_longest
from fractions import Fraction
from pathlib import Path

from .best_of_n import PriorScorer
from .sample import decode_to_score, sample_scaffolded
from .style_data import load


def _quantize(dur: Fraction, allowed: list[Fraction]) -> Fraction:
    return min(allowed, key=lambda a: (abs(a - dur), a))


def usable_primes(root=None, *, slots: int = 64) -> list[dict]:
    """Bach openings that actually sound.

    BWV 129.05 opens with 64 rest slots -- the voices wait through an
    instrumental introduction. Priming on that and asking the prior to continue
    produced a four-bar silence, correctly. Openings are therefore screened for
    attacks before use.
    """
    data, _, _ = load(root)
    out = []
    for p in data["test"] + data["validation"]:
        content = [t for t in p["tokens"]
                   if t in ("HOLD", "REST") or t.startswith("NOTE_ON_")]
        head = content[:slots]
        if sum(1 for t in head if t.startswith("NOTE_ON_")) >= 8:
            # The voice count travels with the prime. The grid interleaves voices
            # in a fixed cycle, so priming a 3-voice piece into a 4-voice scaffold
            # shifts every token one slot further out of phase than the last --
            # voice 1's second note lands in voice 4. Eight of the held-out WTC
            # fugues are 3-voice, so this silently corrupted exactly the fugal
            # material the motifs are most wanted from.
            nv = _voice_count(root, p["piece_id"])
            if nv:
                out.append({"piece_id": p["piece_id"], "content": head,
                            "voice_count": nv})
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _voice_count(root, piece_id: str) -> int | None:
    r = Path(root) if root else _repo_root()
    f = r / "data" / "canonical" / f"{piece_id}.json"
    if not f.exists():
        return None
    return len(json.loads(f.read_text()).get("voices") or [])


def extract_motifs(scorer: PriorScorer, *, count: int = 40,
                   rhythm_values: list[str] | None = None,
                   n_notes: int = 5, voice_count: int | None = None, bars: int = 4,
                   temperature: float = 1.0, seed: int = 0,
                   max_span: int = 16, max_leap: int = 9) -> list[dict]:
    """Sample textures from the prior and lift singable motifs out of them.

    Filters exist because a motif has to be usable as a fugue subject, not merely
    present: it must fit inside one voice's range band, move by singable
    intervals, and actually go somewhere rather than repeat one pitch.
    """
    allowed = sorted({Fraction(x) for x in (rhythm_values or ["1/2", "1", "2"])})
    primes = usable_primes()
    if not primes:
        return []
    # Round-robin between fugues and chorales. Walking the list in corpus order
    # and stopping at `count` never reached the fugues at all: each prime yields
    # up to one motif per voice, so 24 motifs exhausted only the first six primes,
    # every one a chorale. The target genre is fugue -- its material cannot be the
    # part that falls off the end.
    fug = [x for x in primes if "wtc" in x["piece_id"]]
    cho = [x for x in primes if "wtc" not in x["piece_id"]]
    primes = [x for pair in zip_longest(fug, cho) for x in pair if x is not None]
    out: list[dict] = []
    tried = 0
    while len(out) < count and tried < count * 8:
        pr = primes[tried % len(primes)]
        # Sample at the prime's own voice count, not a fixed 4, or the primed
        # content lands in the wrong voices.
        nv = voice_count or pr["voice_count"]
        toks = sample_scaffolded(
            scorer, voice_count=nv, bars=bars, temperature=temperature,
            seed=seed + tried, prime_content=pr["content"])
        tried += 1
        score = decode_to_score(toks, nv)
        for vi, v in enumerate(score["voices"]):
            notes = v["notes"]
            if len(notes) < n_notes:
                continue
            # The opening of a voice, after the primed bar: the prior's own
            # material, taken where it is still coherent.
            for start in (0, 1):
                win = notes[start:start + n_notes]
                if len(win) < n_notes:
                    continue
                midis = [n["midi"] for n in win]
                intervals = [b - a for a, b in zip(midis, midis[1:])]
                if not any(intervals):
                    continue                                  # a repeated pitch
                if max(abs(i) for i in intervals) > max_leap:
                    continue
                if max(midis) - min(midis) > max_span:
                    continue
                durs = [_quantize(Fraction(n["duration_q"]), allowed) for n in win]
                sig = hashlib.sha256(
                    f"{intervals}:{[str(d) for d in durs]}".encode()).hexdigest()[:12]
                if any(m["motif_id"] == sig for m in out):
                    continue
                out.append({
                    "motif_id": sig,
                    "intervals": intervals,
                    "rhythm_q": [str(d) for d in durs],
                    "length_q": str(sum(durs)),
                    "source": f"prior/{pr['piece_id']}/v{vi+1}",
                })
                break
            if len(out) >= count:
                break
    return out


def apply_motif(req: dict, motif: dict) -> dict:
    """Return a copy of ``req`` whose subject is ``motif``.

    Entry spacing, the entry plan and the subject id are all derived from subject
    length in `requests.build_requests`, so substituting a motif of a different
    length means recomputing them -- otherwise entries would overlap or run past
    the end of the fragment.
    """
    r = dict(req)
    subject_len = sum(Fraction(d) for d in motif["rhythm_q"])
    total_q = Fraction(req["_total_q"])
    r["subject"] = {
        "subject_id": f"subj:{motif['motif_id']}",
        "intervals": list(motif["intervals"]),
        "rhythm_q": list(motif["rhythm_q"]),
        "length_q": str(subject_len),
        "parent_piece_id": None,
        "source": motif.get("source", "prior"),
    }
    vids = [e["voice_id"] for e in req["entry_plan"]] or ["voice_1"]
    order = list(dict.fromkeys(vids))
    plan = []
    for k, vid in enumerate(order):
        onset = subject_len * k
        if onset + subject_len > total_q:
            break
        plan.append({"voice_id": vid, "onset_q": str(onset),
                     "role": "subject" if k % 2 == 0 else "answer",
                     "transpose": 0 if k % 2 == 0 else 7})
    r["entry_plan"] = plan or [{"voice_id": order[0], "onset_q": "0",
                                "role": "subject", "transpose": 0}]
    # The motif's own values must be representable on the grid the request
    # declares, or the solver refuses the request rather than snapping onsets.
    r["allowed_rhythmic_values"] = sorted(
        {*req.get("allowed_rhythmic_values", []), *motif["rhythm_q"]},
        key=lambda x: Fraction(x))
    return r
