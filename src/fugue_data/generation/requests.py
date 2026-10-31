"""Build balanced generation requests (Spec §16.2, §16.3).

Coverage is achieved by construction: the request builder walks the configured
coverage grid deterministically from the seed, so the accepted set is balanced
across voice counts, modes, keys, meters, subject placements and cadence targets
rather than being balanced by luck and checked afterwards.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from fractions import Fraction
from pathlib import Path

MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 11]          # harmonic minor: leading tone available

# Weighted so the character of each density comes from how often a value is
# picked, not from excluding short values outright. Excluding them set the solver
# grid -- which is the finest allowed value -- to a whole quarter for "sparse",
# leaving six slots in a two-bar fragment and no room for the voices to attack at
# different moments. Rhythmic independence needs somewhere to put the offsets.
_DENSITY_RHYTHM = {
    "sparse": ["1/2", "1", "1", "2", "2", "2"],
    "medium": ["1/2", "1/2", "1", "1", "2"],
    "dense": ["1/4", "1/4", "1/2", "1/2", "1"],
}
_REGISTER_SHIFT = {"low": -7, "middle": 0, "high": 7}


def scale_for(mode: str) -> list[int]:
    """Melodic material for subjects: harmonic minor, so a leading tone exists."""
    return MAJOR if mode == "major" else MINOR


def harmonic_scale_for(mode: str) -> list[int]:
    """Pitch classes the free voices may harmonise with.

    Minor gets BOTH the natural and the raised seventh. With harmonic minor
    alone the only available chords on those degrees are III+, vii-dim and
    ii-dim, so the solver had no consonant option and 23% of minor-key
    sonorities came out augmented and 18% diminished. Real minor-key writing
    borrows the raised seventh for cadential purposes and uses the natural one
    elsewhere; the cross-relation constraint already stops the two colliding.
    """
    if mode == "major":
        return MAJOR
    return sorted(set(MINOR) | {10})        # natural 7th alongside the raised one


# Melodic intervals a subject must not contain. Each is awkward to sing and, in
# a fugue, is heard repeatedly: the subject returns in every entry, so one bad
# interval is stated again in the answer and in every later entry. These were
# never checked -- the free voices were constrained against tritones while the
# subject generator happily produced them -- and about a fifth of all subjects
# came out containing one.
_FORBIDDEN_MELODIC = {
    6: "tritone",
    10: "minor seventh",
    11: "major seventh",
}
_MAX_SUBJECT_LEAP = 9          # a major sixth; the octave is handled separately


def _awkward_interval(a: int, b: int, scale: list[int]) -> str | None:
    """Name the problem with a melodic interval, or None if it is idiomatic.

    Semitone distance alone is not enough. G#->C is four semitones, which sounds
    like a major third, but it spans four letter names and is a diminished
    fourth; F->G# is three semitones but spans two and is an augmented second.
    Both are awkward precisely because the spelling and the sound disagree, so
    the scale-degree distance has to be consulted, not just the pitch distance.
    """
    iv = b - a
    size = abs(iv)
    if size in _FORBIDDEN_MELODIC:
        return _FORBIDDEN_MELODIC[size]
    if size > _MAX_SUBJECT_LEAP and size != 12:
        return "leap wider than a sixth"
    pcs = sorted(scale)
    try:
        da, db = pcs.index(a % 12), pcs.index(b % 12)
    except ValueError:
        return None
    degrees = abs(da - db)
    degrees = min(degrees, len(pcs) - degrees)
    if size == 3 and degrees == 1:
        return "augmented second"
    if size == 4 and degrees == 3:
        return "diminished fourth"
    if size == 8 and degrees == 4:
        return "augmented fifth"
    return None


def _subject_intervals(rng: random.Random, n_notes: int, scale: list[int]) -> list[int]:
    """A singable subject head: mostly steps, at most one leap, ending settled.

    Retries until every interval is idiomatic. The walk is over scale degrees, so
    in harmonic minor an innocuous-looking one-degree step can land on an
    augmented second; only checking the realised pitches catches that.
    """
    def pitch(d):
        octave, idx = divmod(d, 7)
        return scale[idx] + 12 * octave

    for _ in range(200):
        degrees = [0]
        for _ in range(n_notes - 1):
            step = rng.choice([1, 1, 1, -1, -1, 2, -2, 3, -3, 4])
            nxt = max(-7, min(9, degrees[-1] + step))
            degrees.append(nxt)
        pitches = [pitch(d) for d in degrees]
        if any(_awkward_interval(a, b, scale) for a, b in zip(pitches, pitches[1:])):
            continue
        return [b - a for a, b in zip(pitches, pitches[1:])]

    # Fall back to a plain stepwise subject rather than emit an awkward one.
    pitches = [pitch(d) for d in range(n_notes)]
    return [b - a for a, b in zip(pitches, pitches[1:])]


def build_requests(count: int, cfg: dict, *, seed: int | None = None) -> list[dict]:
    vcfg = cfg["valid"]
    cov = vcfg["coverage"]
    seed = vcfg["seed"] if seed is None else seed

    axes = ["voice_count", "mode", "tonic_pc", "meter", "length_bars",
            "subject_length_notes", "entry_order", "subject_voice",
            "rhythmic_density", "cadence_target", "countersubject", "register"]
    grid = list(itertools.product(*(cov[a] for a in axes)))
    rng = random.Random(seed)
    rng.shuffle(grid)

    out: list[dict] = []
    for i in range(count):
        combo = dict(zip(axes, grid[i % len(grid)]))
        # A distinct sub-seed per request keeps each one independently reproducible.
        rseed = int(hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()[:8], 16)
        r = random.Random(rseed)
        scale = scale_for(combo["mode"])
        num, den = (int(x) for x in combo["meter"].split("/"))
        beats_per_bar = Fraction(num * 4, den)
        total_q = beats_per_bar * combo["length_bars"]

        n_notes = combo["subject_length_notes"]
        intervals = _subject_intervals(r, n_notes, scale)
        durations = [r.choice(_DENSITY_RHYTHM[combo["rhythmic_density"]])
                     for _ in range(n_notes)]
        subject_len = sum(Fraction(d) for d in durations)
        # A subject must fit, with room for at least one more entry after it.
        # Dropping notes to make it fit produced two-note "subjects" -- a single
        # interval, with no identity to recognise when it returns in the answer --
        # in 14% of requests, and under four notes in 31%. The fix is to make the
        # notes shorter rather than fewer: a subject is defined by its shape, and
        # the coverage axis asked for 4 to 6 notes, not for however many happen
        # to fit.
        budget = total_q / 2
        allowed = sorted({Fraction(x) for x in _DENSITY_RHYTHM[combo["rhythmic_density"]]})
        while subject_len > budget:
            longest = max(durations, key=lambda d: Fraction(d))
            shorter = [a for a in allowed if a < Fraction(longest)]
            if not shorter:
                break
            durations[durations.index(longest)] = str(max(shorter))
            subject_len = sum(Fraction(d) for d in durations)
        # Only if even the shortest values will not fit does the subject lose
        # notes, and never below three.
        while subject_len > budget and len(durations) > 3:
            durations = durations[:-1]
            intervals = intervals[:len(durations) - 1]
            subject_len = sum(Fraction(d) for d in durations)

        vc = combo["voice_count"]
        vids = [f"voice_{k}" for k in range(1, vc + 1)]
        order = {"top_down": vids,
                 "bottom_up": list(reversed(vids)),
                 "middle_out": sorted(vids, key=lambda v: abs(
                     int(v.split("_")[1]) - (vc + 1) / 2))}[combo["entry_order"]]

        # Entries are spaced a subject-length apart, as far as the fragment allows.
        spacing = subject_len
        entry_plan = []
        for k, vid in enumerate(order):
            onset = spacing * k
            if onset + subject_len > total_q:
                break
            entry_plan.append({
                "voice_id": vid,
                "onset_q": str(onset),
                "role": "subject" if k % 2 == 0 else "answer",
                "transpose": 0 if k % 2 == 0 else 7,
            })
        if not entry_plan:
            entry_plan = [{"voice_id": order[0], "onset_q": "0",
                           "role": "subject", "transpose": 0}]

        cadence = None
        if combo["cadence_target"] == "tonic":
            cadence = {"onset_q": str(total_q), "tonic_pc": combo["tonic_pc"],
                       "type": "PAC"}

        req = {
            "request_id": f"req:{i:06d}",
            "seed": rseed,
            "rule_profile": vcfg["rule_profile"],
            "voice_count": vc,
            "meter": {"numerator": num, "denominator": den},
            "key": {"tonic_pc": combo["tonic_pc"], "mode": combo["mode"],
                    "fifths": 0},
            "length_bars": combo["length_bars"],
            "subject": {
                "subject_id": f"subj:{rseed:08x}",
                "intervals": intervals,
                "rhythm_q": durations,
                "length_q": str(subject_len),
                "parent_piece_id": None,
            },
            "entry_plan": entry_plan,
            "cadence_target": cadence,
            "allowed_rhythmic_values": sorted(
                set(_DENSITY_RHYTHM[combo["rhythmic_density"]]),
                key=lambda x: Fraction(x)),
            "range_profile": "keyboard",
            "texture_profile": ("with_countersubject" if combo["countersubject"]
                                else "free"),
            "_coverage": combo,
            "_register_shift": _REGISTER_SHIFT[combo["register"]],
            "_total_q": str(total_q),
        }
        out.append(req)
    return out
