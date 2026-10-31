"""CP-SAT generator for hard-valid fragments (Spec §16).

This module only *proposes*. §16.4 requires that a logically separate validation
call confirm a candidate, so nothing here decides acceptance: the rule engine does
that, from the serialized fragment, with no access to the solver's internal model.
Constraints encoded here are a tractable subset chosen to make solutions likely;
the validator is what makes them true.
"""

from __future__ import annotations

from fractions import Fraction

from ortools.sat.python import cp_model

from ..generation.canonicalize import spell
from ..normalize.rational import frac_str
from .requests import harmonic_scale_for, scale_for

# Comfortable keyboard tessitura per voice index (1 = highest), from the
# `keyboard` range profile in config/rules/*.yaml.
_VOICE_BAND = {1: (60, 84), 2: (53, 77), 3: (48, 72), 4: (40, 64), 5: (36, 60)}

# Ablation switches. Every constraint here narrows the feasible set, and stacking
# them can make a request unsolvable for reasons that are invisible from the
# outside -- the solver just returns nothing. Being able to turn one off is how
# you find which one is responsible instead of guessing.
FEATURES = {
    "harmony": True,          # functional chord progression under the notes
    "cadence_prep": True,     # dominant before the final tonic
    "stagnation": True,       # not every voice may go long at once
    "melodic_arc": True,      # a line reaches its high point once
    "subject_clarity": True,  # other voices stay out of an entry's way
    "stepwise": True,         # floor on stepwise motion, per Bach's measured rate
    "fifth_motion": True,     # floor on root motion by fifth, likewise
    "diss_density": True,     # ceiling on simultaneous dissonance
}

# Measured on 60 held-out Bach chorales, not taken from a textbook (§15.6).
#
#   stepwise   75.2% of melodic moves are <= 2 semitones (unisons excluded, since
#              the solver counts a move as pitch[s+1] != pitch[s])
#   fifths     37.2% of beat-to-beat root transitions move by a fifth
#
# The floors sit below the measured rates deliberately. A floor AT the corpus
# mean would force every fragment to be at least as stepwise as the average Bach
# chorale -- which half of Bach itself would fail. Four constraints added at once
# two rounds ago collapsed the solve rate from 63% to 7%, so these are calibrated
# by measuring the solve rate and keeping the strictest value that stays viable.
# The floors apply to FREE voices only -- entry slots are exempt, because the
# subject leaps as written. So a floor above the corpus mean is not a floor above
# Bach: 80% on free motion lands the fragment as a whole at ~75%, which is Bach's
# measured rate. Swept at 60/70/75/80 against 25/35/37/40 and chosen by measuring;
# the solve rate was identical at every setting (the requests that fail, fail for
# unrelated reasons), so the tightest viable pair was taken. Cost is solve time:
# ~12s per 60 requests against ~6s.
# Every individual dissonance can be a textbook passing tone and the texture can
# still be saturated: the fragment a listener called "a lot of dissonance in the
# middle" had 16 dissonances, 100% of them proper passing or neighbour tones, at
# 1.78 per quarter against Bach's 1.00. Correct treatment of each is not a limit
# on how many may sound at once, and nothing in the model was counting them.
# Swept at 26/22/18/14/12/10/8 against held-out requests. 12% is where all three
# distribution metrics land on Bach at once (diss/q 0.95 vs 1.00, stepwise 72.4%
# vs 75.2%, fifth motion 35.7% vs 37.2%). It costs solve rate -- 30% -> 23% -- and
# doubles solve time, which is the price of the texture being right.
DISSONANCE_MAX_PCT = 12       # share of (voice-pair, slot) cells allowed to clash
STEPWISE_FLOOR_PCT = 80       # -> 74.9% realized, vs 75.2% Bach, 40.9% before
FIFTH_MOTION_FLOOR_PCT = 40   # -> 29.9% realized, vs 37.2% Bach, 14.6% before


def _allowed_pitches(tonic_pc: int, mode: str, lo: int, hi: int, shift: int) -> list[int]:
    scale = {(tonic_pc + d) % 12 for d in harmonic_scale_for(mode)}
    lo, hi = lo + shift, hi + shift
    return [m for m in range(max(21, lo), min(108, hi) + 1) if m % 12 in scale]


def _subject_pitches(start_midi: int, intervals: list[int]) -> list[int]:
    out = [start_midi]
    for iv in intervals:
        out.append(out[-1] + iv)
    return out


def solve_fragment(req: dict, *, max_time_s: float = 5.0, workers: int = 8) -> dict | None:
    """Return a fragment dict (canonical `voices` shape) or None if infeasible."""
    total_q = Fraction(req["_total_q"])
    # The grid is the request's own finest rhythmic value, so every duration and
    # every entry onset lands exactly on a slot. A fixed grid would have to round
    # them, and rounding an entry onset silently moves the subject -- which is how
    # a scheduled answer came out shifted by one note and failed validation.
    slot = min(Fraction(x) for x in req["allowed_rhythmic_values"])
    n_slots = int(total_q / slot)
    if n_slots <= 0 or n_slots > 128:
        return None
    for entry in req["entry_plan"]:
        if Fraction(entry["onset_q"]) % slot != 0:
            return None                         # refuse rather than snap
    if any(Fraction(d) % slot != 0 for d in req["subject"]["rhythm_q"]):
        return None

    vc = req["voice_count"]
    vids = [f"voice_{k}" for k in range(1, vc + 1)]
    shift = req.get("_register_shift", 0)
    tonic, mode = req["key"]["tonic_pc"], req["key"]["mode"]
    domains = {v: _allowed_pitches(tonic, mode, *_VOICE_BAND[i + 1], shift)
               for i, v in enumerate(vids)}
    if any(len(d) < 4 for d in domains.values()):
        return None

    # --- realize the scheduled entries before building the model ------------ #
    # A *real* answer transposes the subject chromatically by a fifth, which is
    # idiomatic and deliberately introduces accidentals outside the tonic scale.
    # So the entry pitches are computed first and then admitted into the voice's
    # domain, rather than the entry being rejected for leaving the diatonic set.
    # The starting octave is chosen so the whole entry fits the voice's band.
    subj = req["subject"]
    rhythm = [Fraction(x) for x in subj["rhythm_q"]]
    realized: list[tuple[str, list[int], Fraction]] = []
    for entry in req["entry_plan"]:
        v = entry["voice_id"]
        if v not in domains:
            continue
        base_lo, base_hi = _VOICE_BAND[int(v.split("_")[1])]
        lo, hi = base_lo + shift, base_hi + shift
        want_pc = (tonic + entry.get("transpose", 0)) % 12
        best = None
        for start in range(max(21, lo - 12), min(108, hi + 12) + 1):
            if start % 12 != want_pc:
                continue
            ps = _subject_pitches(start, subj["intervals"])
            if min(ps) < lo or max(ps) > hi:
                continue
            centre = (lo + hi) / 2
            score = abs(sum(ps) / len(ps) - centre)
            if best is None or score < best[0]:
                best = (score, ps)
        if best is None:
            return None
        realized.append((v, best[1], Fraction(entry["onset_q"])))
        domains[v] = sorted(set(domains[v]) | set(best[1]))

    m = cp_model.CpModel()
    pitch = {}
    active = {}
    for v in vids:
        for s in range(n_slots):
            p = m.NewIntVarFromDomain(
                cp_model.Domain.FromValues(domains[v]), f"p_{v}_{s}")
            a = m.NewBoolVar(f"a_{v}_{s}")
            pitch[v, s] = p
            active[v, s] = a

    entry_slots: dict[tuple[str, int], int] = {}
    for v, pitches, at in realized:
        cursor = at
        for pm, dur in zip(pitches, rhythm):
            s = int(cursor / slot)
            if s >= n_slots:
                return None
            m.Add(pitch[v, s] == pm)
            m.Add(active[v, s] == 1)
            entry_slots[v, s] = pm
            for held in range(s + 1, min(n_slots, s + max(1, int(dur / slot)))):
                m.Add(pitch[v, held] == pm)
                m.Add(active[v, held] == 0)     # sustained, not re-attacked
                entry_slots[v, held] = pm
            cursor += dur

    # --- melodic motion ------------------------------------------------------ #
    # No melodic tritone. Measured against real Bach the unconstrained model wrote
    # them at 23.1 per 1000 notes against Bach's 9.7 -- and Bach's are mostly
    # diminished-fifth outlines inside a diminished-seventh harmony, which this
    # model has no way to express, so the safe form is to forbid the leap.
    max_leap = 12
    for v in vids:
        for s in range(n_slots - 1):
            d = m.NewIntVar(-max_leap, max_leap, f"d_{v}_{s}")
            m.Add(d == pitch[v, s + 1] - pitch[v, s])
            ad = m.NewIntVar(0, max_leap, f"ad_{v}_{s}")
            m.AddAbsEquality(ad, d)
            m.Add(ad != 6)

    # Free voices move by step or small leap only. A leap-recovery rule stated
    # over consecutive *slots* silently stopped working when the grid got finer:
    # a sustained note makes pitch[s+1] == pitch[s], so the real leap happens at a
    # note boundary the constraint never looks at, and 56% of leaps went
    # unrecovered. Bounding the interval itself cannot be defeated that way,
    # because it holds at every slot pair regardless of where notes begin. Entry
    # slots are exempt: the subject is fixed and may leap as written.
    free_leap_max = 5                       # a perfect fourth
    # A cap permits every leap up to a fourth and prefers nothing, so the solver
    # took them freely: 40.9% stepwise against Bach's 75.2%. The lines broke no
    # rule and were stricter than Bach on every prohibition -- they simply were
    # not melodies. A floor on the *share* of moves that are steps is what a cap
    # cannot express.
    step_terms: list = []
    for v in vids:
        for s in range(n_slots - 1):
            if (v, s) in entry_slots or (v, s + 1) in entry_slots:
                continue
            d = m.NewIntVar(-127, 127, f"fl_{v}_{s}")
            m.Add(d == pitch[v, s + 1] - pitch[v, s])
            ad = m.NewIntVar(0, 127, f"fla_{v}_{s}")
            m.AddAbsEquality(ad, d)
            m.Add(ad <= free_leap_max)
            step_terms.append((ad, v, s))

    # Of the slot pairs where a free voice actually changes pitch, at least
    # STEPWISE_FLOOR_PCT must move by a step. Expressed as a ratio rather than a
    # count because the number of moves is itself a variable: a sparse fragment
    # would otherwise satisfy a fixed count trivially, and a dense one could not
    # satisfy it at all.
    if FEATURES["stepwise"] and step_terms:
        moved_bs, step_bs = [], []
        for ad, v, s in step_terms:
            mv = m.NewBoolVar(f"mv_{v}_{s}")
            m.Add(ad >= 1).OnlyEnforceIf(mv)
            m.Add(ad == 0).OnlyEnforceIf(mv.Not())
            # st <=> (moved AND interval <= 2). Reifying it as `ad <= 2` alone
            # would be wrong twice over: a sustained note (ad == 0) satisfies
            # `ad <= 2` and would count as a step, and the negative half
            # (`st.Not() => ad >= 3`) makes ad == 0 infeasible, which would force
            # every voice to move on every slot. So it is built from two separate
            # reified comparisons and an AND.
            le2 = m.NewBoolVar(f"le2_{v}_{s}")
            m.Add(ad <= 2).OnlyEnforceIf(le2)
            m.Add(ad >= 3).OnlyEnforceIf(le2.Not())
            st = m.NewBoolVar(f"st_{v}_{s}")
            m.AddBoolAnd([mv, le2]).OnlyEnforceIf(st)
            m.AddBoolOr([mv.Not(), le2.Not()]).OnlyEnforceIf(st.Not())
            moved_bs.append(mv)
            step_bs.append(st)
        m.Add(100 * sum(step_bs) >= STEPWISE_FLOOR_PCT * sum(moved_bs))

    # --- no direct (hidden) fifths or octaves into the outer voices ---------- #
    # The outer pair reaching a perfect consonance by similar motion with a leap
    # on top: 9.8 per 1000 notes unconstrained against Bach's 1.9. Bach does it,
    # sparingly and with the top voice moving by step; a solver with no sense of
    # "sparingly" should simply not.
    if len(vids) >= 2:
        top, low = vids[0], vids[-1]
        for s in range(n_slots - 1):
            ivn = m.NewIntVar(0, 127, f"df_{s}")
            m.Add(ivn == pitch[top, s + 1] - pitch[low, s + 1])
            icn = m.NewIntVar(0, 11, f"dfc_{s}")
            m.AddModuloEquality(icn, ivn, 12)
            perfect = m.NewBoolVar(f"dfp_{s}")
            m.AddAllowedAssignments([icn], [(0,), (7,)]).OnlyEnforceIf(perfect)
            m.AddForbiddenAssignments([icn], [(0,), (7,)]).OnlyEnforceIf(perfect.Not())

            dt = m.NewIntVar(-127, 127, f"dft_{s}")
            db = m.NewIntVar(-127, 127, f"dfb_{s}")
            m.Add(dt == pitch[top, s + 1] - pitch[top, s])
            m.Add(db == pitch[low, s + 1] - pitch[low, s])
            top_up = m.NewBoolVar(f"dfu_{s}")
            low_up = m.NewBoolVar(f"dflu_{s}")
            m.Add(dt > 0).OnlyEnforceIf(top_up)
            m.Add(dt <= 0).OnlyEnforceIf(top_up.Not())
            m.Add(db > 0).OnlyEnforceIf(low_up)
            m.Add(db <= 0).OnlyEnforceIf(low_up.Not())
            same_dir = m.NewBoolVar(f"dfs_{s}")
            m.Add(top_up == low_up).OnlyEnforceIf(same_dir)
            m.Add(top_up != low_up).OnlyEnforceIf(same_dir.Not())
            moving = m.NewBoolVar(f"dfm_{s}")
            m.Add(dt != 0).OnlyEnforceIf(moving)
            m.Add(dt == 0).OnlyEnforceIf(moving.Not())
            leaps = m.NewBoolVar(f"dfl_{s}")
            adt = m.NewIntVar(0, 127, f"dfat_{s}")
            m.AddAbsEquality(adt, dt)
            m.Add(adt > 2).OnlyEnforceIf(leaps)
            m.Add(adt <= 2).OnlyEnforceIf(leaps.Not())
            # Forbid: perfect consonance AND similar motion AND a leap on top.
            m.AddBoolOr([perfect.Not(), same_dir.Not(), moving.Not(), leaps.Not()])

    # --- ceiling on simultaneous dissonance ---------------------------------- #
    # Counts dissonant vertical interval classes across every voice pair at every
    # slot and caps the total. A ceiling, not a prohibition: passing dissonance is
    # what makes counterpoint move, and forbidding it outright would give block
    # chords. The cap is calibrated against Bach's measured density.
    if FEATURES["diss_density"] and len(vids) >= 2 and n_slots:
        clash = []
        for i, a in enumerate(vids):
            for b in vids[i + 1:]:
                for s in range(n_slots):
                    ivd = m.NewIntVar(-127, 127, f"dv_{a}{b}_{s}")
                    m.Add(ivd == pitch[a, s] - pitch[b, s])
                    mdc = m.NewIntVar(0, 11, f"dc_{a}{b}_{s}")
                    m.AddModuloEquality(mdc, ivd, 12)
                    z = m.NewBoolVar(f"dz_{a}{b}_{s}")
                    m.AddAllowedAssignments(
                        [mdc], [(1,), (2,), (6,), (10,), (11,)]).OnlyEnforceIf(z)
                    m.AddForbiddenAssignments(
                        [mdc], [(1,), (2,), (6,), (10,), (11,)]).OnlyEnforceIf(z.Not())
                    clash.append(z)
        if clash:
            m.Add(100 * sum(clash) <= DISSONANCE_MAX_PCT * len(clash))

    # --- no parallel perfect fifths or octaves between any voice pair -------- #
    for i, a in enumerate(vids):
        for b in vids[i + 1:]:
            for s in range(n_slots - 1):
                iv0 = m.NewIntVar(-60, 60, f"iv0_{a}{b}_{s}")
                iv1 = m.NewIntVar(-60, 60, f"iv1_{a}{b}_{s}")
                m.Add(iv0 == pitch[a, s] - pitch[b, s])
                m.Add(iv1 == pitch[a, s + 1] - pitch[b, s + 1])
                mod0 = m.NewIntVar(0, 11, f"m0_{a}{b}_{s}")
                mod1 = m.NewIntVar(0, 11, f"m1_{a}{b}_{s}")
                m.AddModuloEquality(mod0, iv0, 12)
                m.AddModuloEquality(mod1, iv1, 12)
                for interval in (7, 0):
                    f0 = m.NewBoolVar(f"f0_{a}{b}_{s}_{interval}")
                    f1 = m.NewBoolVar(f"f1_{a}{b}_{s}_{interval}")
                    m.Add(mod0 == interval).OnlyEnforceIf(f0)
                    m.Add(mod0 != interval).OnlyEnforceIf(f0.Not())
                    m.Add(mod1 == interval).OnlyEnforceIf(f1)
                    m.Add(mod1 != interval).OnlyEnforceIf(f1.Not())
                    moved_a = m.NewBoolVar(f"ma_{a}{b}_{s}_{interval}")
                    moved_b = m.NewBoolVar(f"mb_{a}{b}_{s}_{interval}")
                    m.Add(pitch[a, s + 1] != pitch[a, s]).OnlyEnforceIf(moved_a)
                    m.Add(pitch[a, s + 1] == pitch[a, s]).OnlyEnforceIf(moved_a.Not())
                    m.Add(pitch[b, s + 1] != pitch[b, s]).OnlyEnforceIf(moved_b)
                    m.Add(pitch[b, s + 1] == pitch[b, s]).OnlyEnforceIf(moved_b.Not())
                    # Forbid: same interval class before and after, both voices moving.
                    m.AddBoolOr([f0.Not(), f1.Not(), moved_a.Not(), moved_b.Not()])

    # --- voices stay ordered (no crossing) and reasonably spaced ------------- #
    for s in range(n_slots):
        for a, b in zip(vids, vids[1:]):
            m.Add(pitch[a, s] >= pitch[b, s] + 1)

    # --- vertical harmony ---------------------------------------------------- #
    # Without this the model happily stacks adjacent scale degrees: measured over
    # the first release, 50.6% of generated simultaneities contained a semitone
    # clash against 9.8% in real Bach, and only 17% were triadic against 52%.
    # Parallels, ranges and voice order say nothing about whether a chord is a
    # chord, so harmony has to be stated.
    #
    # Two constraints, split by metrical weight, which is how the style works:
    #   - nowhere may two voices sound a semitone or major seventh apart;
    #   - on a beat, every upper voice is consonant against the bass.
    # Off-beat slots stay free, which is where passing dissonance belongs.
    beat = Fraction(4, req["meter"]["denominator"])
    strong = {s for s in range(n_slots) if (s * slot) % beat == 0}
    consonant_over_bass = (0, 3, 4, 7, 8, 9)
    bass = vids[-1]
    for s in range(n_slots):
        for i, a in enumerate(vids):
            for b in vids[i + 1:]:
                iv = m.NewIntVar(0, 127, f"h_{a}{b}_{s}")
                m.Add(iv == pitch[a, s] - pitch[b, s])     # >= 1 by voice order
                ic = m.NewIntVar(0, 11, f"hc_{a}{b}_{s}")
                m.AddModuloEquality(ic, iv, 12)
                m.Add(ic != 1)
                m.Add(ic != 11)
                if s in strong and b == bass:
                    m.AddAllowedAssignments([ic], [(c,) for c in consonant_over_bass])

    # --- no cross-relation between voices ------------------------------------ #
    # A cross-relation is one letter name appearing with two different
    # accidentals in *different* voices a beat apart -- F natural against F sharp.
    # It is not "any two pitches a semitone apart": E in one voice followed by F
    # in another is ordinary. And chromatic motion *within* one line is idiomatic,
    # especially in a real answer, so only cross-voice pairs are forbidden.
    prefer_flats = tonic in (1, 3, 5, 8, 10)
    letter_of = {pc: spell(60 + pc, prefer_flats)[0] for pc in range(12)}
    forbidden_pcs = [(a, b) for a in range(12) for b in range(12)
                     if a != b and letter_of[a] == letter_of[b]]
    if forbidden_pcs:
        for s in range(n_slots - 1):
            for a in vids:
                for b in vids:
                    if a == b:
                        continue
                    pa = m.NewIntVar(0, 11, f"xr_a_{a}{b}_{s}")
                    pb = m.NewIntVar(0, 11, f"xr_b_{a}{b}_{s}")
                    m.AddModuloEquality(pa, pitch[a, s], 12)
                    m.AddModuloEquality(pb, pitch[b, s + 1], 12)
                    m.AddForbiddenAssignments([pa, pb], forbidden_pcs)

    # --- closure, but only when the request asked for it --------------------- #
    # A cadence must actually be a chord: the original constraint pinned only the
    # *bass* to the tonic pitch class and left the upper voices free, so
    # `cadence_target: tonic` produced no audible cadence and 58% of fragments
    # stopped on a non-triad.
    #
    # But closure is required ONLY where the request asks for it. Measured over
    # 1,440 arbitrary 2-4 bar windows of real WTC fugues, just 25% end on a
    # major or minor triad and 39% end on no triad at all -- because a window is
    # a view into longer music, not a piece. Forcing every fragment to resolve
    # taught a false periodicity (a cadence every 2-4 bars) and collapsed the
    # `cadence_target: [tonic, none]` coverage axis of §16.3, since both halves
    # then ended identically. The open half is left open on purpose.
    last = n_slots - 1
    if vc >= 3 and req.get("cadence_target"):
        present_last = []
        for pc in range(12):
            b = m.NewBoolVar(f"lastpc_{pc}")
            lits = []
            for v in vids:
                eq = m.NewBoolVar(f"lasteq_{v}_{pc}")
                pcv = m.NewIntVar(0, 11, f"lastv_{v}_{pc}")
                m.AddModuloEquality(pcv, pitch[v, last], 12)
                m.Add(pcv == pc).OnlyEnforceIf(eq)
                m.Add(pcv != pc).OnlyEnforceIf(eq.Not())
                lits.append(eq)
            m.AddMaxEquality(b, lits)
            present_last.append(b)

        roots = [tonic]
        options = []
        for r in roots:
            for third in (3, 4):
                members = {r, (r + third) % 12, (r + 7) % 12}
                opt = m.NewBoolVar(f"lasttriad_{r}_{third}")
                for pc in range(12):
                    if pc in members:
                        m.AddImplication(opt, present_last[pc])
                    else:
                        m.AddImplication(opt, present_last[pc].Not())
                options.append(opt)
        m.AddBoolOr(options)

    if req.get("cadence_target"):
        bass = vids[-1]
        cands = [p for p in domains[bass] if p % 12 == tonic]
        if cands:
            m.AddAllowedAssignments([pitch[bass, last]], [(c,) for c in cands])
        m.Add(active[vids[0], last] == 1)

    # --- participation: every voice keeps moving ----------------------------- #
    # A total attack count is not enough. Tying attack to pitch-change (above)
    # removed stuttering, but the solver then satisfied participation by holding
    # one pitch for the whole fragment -- a voice sounding C for eleven quarters
    # scores 100% participation, because the rule measures sounding duration and
    # a drone sounds continuously. A sliding window forces each voice to attack at
    # least once per bar, which is what "participating" actually means.
    bar_len = Fraction(req['meter']['numerator'] * 4,
                       req['meter']['denominator'])
    per_bar = max(2, int(bar_len / slot))
    for v in vids:
        m.Add(sum(active[v, s] for s in range(n_slots)) >= max(2, n_slots // 6))
        for s in range(0, n_slots - per_bar + 1):
            m.Add(sum(active[v, s + k] for k in range(per_bar)) >= 1)

    # --- rhythmic independence ----------------------------------------------- #
    # The defining feature of counterpoint, and the one thing the model never
    # asked for. Measured against the corpus, 39.8% of generated onsets had every
    # voice attacking at once; real WTC fugues sit at 2.0% and Bach chorales --
    # which are homophonic by nature -- at 54.3%. The fragments were rhythmically
    # chorales wearing a fugue's entry plan. Capping simultaneous attacks forces
    # the voices to move at different moments, which is what makes lines audible
    # as separate lines.
    max_together = max(1, (vc + 1) // 2)
    for s in range(1, n_slots):
        m.Add(sum(active[v, s] for v in vids) <= max_together)

    # --- functional harmony -------------------------------------------------- #
    # The previous version asked only that a beat carry three distinct pitch
    # classes and not be augmented. That produces chords but not a progression:
    # a reviewer described the output as "locally plausible but directionless --
    # recognizable chords do not automatically create a progression", and they
    # were right, because nothing connected one beat to the next.
    #
    # So harmony is now chosen first, as a sequence, and the notes are made to
    # fit it. One chord per beat, drawn from the diatonic triads, with
    # transitions restricted to the functional grammar: tonic may go anywhere,
    # predominant may not follow dominant (the retrogression V->IV), and a
    # requested cadence must be approached by a dominant rather than merely
    # arriving on a tonic.
    beat_slots = sorted(strong)
    scale_pcs = [(tonic + d) % 12 for d in scale_for(mode)]

    # Diatonic triads by scale degree, rejecting the augmented one that harmonic
    # minor produces on III.
    TONIC, PREDOM, DOMINANT = "T", "P", "D"
    FUNCTION = {0: TONIC, 1: PREDOM, 2: TONIC, 3: PREDOM, 4: DOMINANT,
                5: TONIC, 6: DOMINANT}
    chords = []
    for deg in range(7):
        members = tuple(sorted({scale_pcs[deg], scale_pcs[(deg + 2) % 7],
                                scale_pcs[(deg + 4) % 7]}))
        if len(members) < 3:
            continue
        rel = sorted((x - scale_pcs[deg]) % 12 for x in members)
        if rel == [0, 4, 8]:          # augmented: never a usable harmony here
            continue
        chords.append({"degree": deg, "root": scale_pcs[deg], "pcs": set(members),
                       "function": FUNCTION[deg]})
    if mode == "minor":
        # The natural seventh gives a major VII, the chord the descending melodic
        # form implies. Without it the only harmonies on that degree are the
        # diminished vii and the solver is pushed back onto it constantly.
        sub = (tonic + 10) % 12
        vii = tuple(sorted({sub, (sub + 4) % 12, (sub + 7) % 12}))
        chords.append({"degree": 6, "root": sub, "pcs": set(vii), "function": PREDOM})
    if len(chords) < 3 or not beat_slots or not FEATURES['harmony']:
        chords = []

    if chords:
        n_ch = len(chords)
        sel = []
        for bi, s in enumerate(beat_slots):
            picks = [m.NewBoolVar(f"ch_{bi}_{ci}") for ci in range(n_ch)]
            m.AddExactlyOne(picks)
            sel.append(picks)
            # Every voice sounding on this beat must be a tone of the chord.
            # Entry voices are exempt: their pitches are fixed by the subject,
            # so requiring them to be chord tones asks the harmony to bend a note
            # it cannot move, and the request is simply unsolvable. The subject is
            # given; the harmony accompanies it.
            free_here = [v for v in vids if (v, s) not in entry_slots]
            if not free_here:
                continue
            # One pitch-class variable per voice per beat, reused across all the
            # chord options. Building it inside the chord loop created the same
            # modulo constraint once per candidate chord -- eight times the work
            # for identical information, and modulo is among the more expensive
            # things to ask CP-SAT for.
            # Pitch class of every voice on this beat, computed once and reused
            # both for chord membership and for the distinct-class floor below.
            pcv = {}
            for v in vids:
                pcv[v] = m.NewIntVar(0, 11, f"hpc_{v}_{bi}")
                m.AddModuloEquality(pcv[v], pitch[v, s], 12)
            for ci, ch in enumerate(chords):
                allowed = [(a,) for a in sorted(ch["pcs"])]
                for v in free_here:
                    m.AddAllowedAssignments([pcv[v]], allowed).OnlyEnforceIf(picks[ci])

            # Requiring chord *membership* does not require the chord to be
            # complete: three voices may all land on the root and fifth, giving a
            # hollow open fifth with the third missing. That floor lived in the
            # block this harmony layer replaced, and removing it dropped complete
            # sonorities from 100% to 66%.
            if vc >= 3:
                seen = []
                for pc in range(12):
                    b = m.NewBoolVar(f"hseen_{bi}_{pc}")
                    eqs = []
                    for v in vids:
                        e = m.NewBoolVar(f"hseq_{v}_{bi}_{pc}")
                        m.Add(pcv[v] == pc).OnlyEnforceIf(e)
                        m.Add(pcv[v] != pc).OnlyEnforceIf(e.Not())
                        eqs.append(e)
                    m.AddMaxEquality(b, eqs)
                    seen.append(b)
                m.Add(sum(seen) >= min(3, vc))

        # Functional grammar: a dominant never falls back to a predominant, and a
        # chord does not simply repeat forever.
        for bi in range(len(beat_slots) - 1):
            for ci, ch in enumerate(chords):
                for cj, nxt in enumerate(chords):
                    if ch["function"] == DOMINANT and nxt["function"] == PREDOM:
                        m.AddBoolOr([sel[bi][ci].Not(), sel[bi + 1][cj].Not()])
        # No more than two beats on the same chord in a row.
        for bi in range(len(beat_slots) - 2):
            for ci in range(n_ch):
                m.AddBoolOr([sel[bi][ci].Not(), sel[bi + 1][ci].Not(),
                             sel[bi + 2][ci].Not()])

        # Root motion by fifth, as a share of transitions.
        #
        # The grammar above is entirely negative -- it forbids V->IV and endless
        # repetition, and prefers nothing. Measured against Bach that left the
        # progression inverted: fifth motion 14.6% against Bach's 37.2%, third
        # motion 29.4% against 14.4%. Chords a third apart share two common
        # tones, so the harmony changed without going anywhere, which is what
        # every listener reported as directionless. The circle of fifths is
        # Bach's engine and nothing here asked for it.
        if FEATURES["fifth_motion"] and len(beat_slots) >= 2:
            fifth_bs = []
            for bi in range(len(beat_slots) - 1):
                z = m.NewBoolVar(f"fifth_{bi}")
                for ci, ch in enumerate(chords):
                    for cj, nxt in enumerate(chords):
                        iv = (nxt["root"] - ch["root"]) % 12
                        lit = z if iv in (5, 7) else z.Not()
                        # Selecting this pair pins z, in both directions, so the
                        # count cannot be inflated by leaving z free.
                        m.AddBoolOr([sel[bi][ci].Not(), sel[bi + 1][cj].Not(), lit])
                fifth_bs.append(z)
            m.Add(100 * sum(fifth_bs)
                  >= FIFTH_MOTION_FLOOR_PCT * len(fifth_bs))

        if (req.get("cadence_target") and len(beat_slots) >= 2
                and FEATURES['cadence_prep']):
            tonic_ci = [ci for ci, c in enumerate(chords) if c["degree"] == 0]
            dom_ci = [ci for ci, c in enumerate(chords)
                      if c["function"] == DOMINANT and c["degree"] == 4]
            if tonic_ci:
                m.AddBoolOr([sel[-1][ci] for ci in tonic_ci])
            if dom_ci:
                # The cadence is *prepared*: a dominant immediately before the
                # tonic. Placing a tonic chord last without one is what made
                # requested cadences "sound like stopping points" rather than
                # arrivals.
                m.AddBoolOr([sel[-2][ci] for ci in dom_ci])

    # --- simultaneous stagnation --------------------------------------------- #
    # Long notes clarify an entry, but when every voice goes long at once the
    # music stops developing and the subject's rhythmic identity disappears.
    # Per-voice activity cannot see this: each voice individually satisfies its
    # one-attack-per-bar floor while all of them freeze on the same bar.
    half_bar = max(2, per_bar // 2)
    for s in (range(0, n_slots - half_bar + 1) if FEATURES['stagnation'] else []):
        m.Add(sum(active[v, s + k] for v in vids for k in range(half_bar)) >= 2)

    # --- melodic arc ---------------------------------------------------------- #
    # A line should reach its high point once rather than batting at it. Voices
    # were "repeatedly leaping between chord tones without building toward a high
    # point", which is what makes activity read as motion without shape.
    for v in (vids if FEATURES['melodic_arc'] else []):
        hi = m.NewIntVar(0, 127, f"hi_{v}")
        m.AddMaxEquality(hi, [pitch[v, s] for s in range(n_slots)])
        peaks = []
        for s in range(n_slots):
            at_peak = m.NewBoolVar(f"pk_{v}_{s}")
            m.Add(pitch[v, s] == hi).OnlyEnforceIf(at_peak)
            m.Add(pitch[v, s] != hi).OnlyEnforceIf(at_peak.Not())
            struck = m.NewBoolVar(f"pks_{v}_{s}")
            m.AddBoolAnd([at_peak, active[v, s]]).OnlyEnforceIf(struck)
            m.AddBoolOr([at_peak.Not(), active[v, s].Not()]).OnlyEnforceIf(struck.Not())
            peaks.append(struck)
        m.Add(sum(peaks) <= 2)

    # --- subject clarity ------------------------------------------------------ #
    # While a scheduled entry is sounding, the other voices stay out of its way.
    # Rapid register changes elsewhere "compete with an entry's identity" and make
    # the subject hard to follow.
    entry_voice_at: dict[int, set[str]] = {}
    for (v, s) in entry_slots:
        entry_voice_at.setdefault(s, set()).add(v)
    # Capping how often the other voices attack was the wrong lever: it halved the
    # solve rate while `max_together` was already limiting simultaneous attacks,
    # so it mostly starved the texture. The complaint was about *register* --
    # "rapid register changes in several voices compete with an entry's identity"
    # -- so the accompaniment is asked to stay put rather than to fall silent.
    for s, owners in (sorted(entry_voice_at.items())
                      if FEATURES['subject_clarity'] else []):
        if s + 1 >= n_slots:
            continue
        for v in vids:
            if v in owners or (v, s + 1) in entry_slots:
                continue
            d = m.NewIntVar(-127, 127, f"sc_{v}_{s}")
            m.Add(d == pitch[v, s + 1] - pitch[v, s])
            ad = m.NewIntVar(0, 127, f"sca_{v}_{s}")
            m.AddAbsEquality(ad, d)
            m.Add(ad <= 4)          # a third: step or small leap, no register jumps

    # Attacks outside entries are free but bounded, so textures stay plausible.
    for v in vids:
        for s in range(n_slots):
            if (v, s) not in entry_slots:
                if s == 0:
                    m.Add(active[v, s] == 1)
                else:
                    # Attack exactly when the pitch changes. As an implication
                    # only, the solver was free to re-strike the same pitch, and
                    # 62% of fragments ended up with a voice hammering one note
                    # three or more times -- the single largest source of the
                    # mechanical quality in the output. A held note is expressed
                    # by sustaining, not by repeating.
                    changed = m.NewBoolVar(f"ch_{v}_{s}")
                    m.Add(pitch[v, s] != pitch[v, s - 1]).OnlyEnforceIf(changed)
                    m.Add(pitch[v, s] == pitch[v, s - 1]).OnlyEnforceIf(changed.Not())
                    m.Add(active[v, s] == changed)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_s
    solver.parameters.num_workers = workers
    # `_solver_seed` lets one prompt be solved repeatedly for different answers,
    # which is what §18.2 needs: candidates identical in every condition except
    # the stochastic seed. Without randomize_search CP-SAT returns the same
    # solution every time and a "pair" would be one fragment twice.
    solver.parameters.random_seed = int(req.get("_solver_seed", req["seed"])) % (2 ** 31)
    if req.get("_solver_seed") is not None:
        solver.parameters.randomize_search = True
        solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    voices = []
    for v in vids:
        notes = []
        s = 0
        while s < n_slots:
            if solver.Value(active[v, s]):
                p = solver.Value(pitch[v, s])
                e = s + 1
                while e < n_slots and not solver.Value(active[v, e]) \
                        and solver.Value(pitch[v, e]) == p:
                    e += 1
                onset = s * slot
                dur = (e - s) * slot
                notes.append({
                    "note_id": f"{req['request_id']}:{v}:{frac_str(onset)}",
                    "voice_id": v, "onset_q": frac_str(onset),
                    "duration_q": frac_str(dur), "measure_number": None,
                    "measure_index_unfolded": None,
                    "offset_in_measure_q": "0",
                    "midi": p, "step": "C", "alter": "0", "octave": p // 12 - 1,
                    "pitch_class": p % 12, "diatonic_step": 0, "tie": None,
                    "grace": False, "articulations": [], "ornaments": [],
                    "chord_group": None, "source_refs": [],
                })
                s = e
            else:
                s += 1
        voices.append({"voice_id": v, "staff": None, "mean_midi": None, "notes": notes})

    return {
        "voices": voices,
        "quarter_length_total": frac_str(total_q),
        "solver_status": solver.StatusName(status),
        "solve_time_ms": int(solver.WallTime() * 1000),
    }
