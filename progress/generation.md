# Tokenization, rules, generation and exports — Progress

**Last updated:** 2026-09-18 (UTC)
**Pipeline stages:** §19 tokenization · §14 excerpts · §15 rules · §16/§17 generation · §18 preferences · §20 exports · §21 gates · §24 report
**Machine-readable:** [`../reports/build_report.json`](../reports/build_report.json) · [`../manifests/validation_gates.json`](../manifests/validation_gates.json)

## Headline

| Metric | Value |
| --- | ---: |
| Hard-valid synthetic fragments (§16) | **10,000 / 10,000** ✅ |
| Controlled single-violation fragments (§17) | **10,000 / 10,000** ✅ |
| Preference-pair candidates (§18) | **3,150** from 4,193 candidates |
| Excerpt candidates proposed (§14) | **300** (≥200 required) ✅ |
| Rules implemented (§15) | **24** across 3 profiles |
| Validation gates passing (§21) | **17 / 17** ✅ |
| Visual-QA artefacts recorded (§21.1) | **294** |
| §28 Definition of Done | **17 / 19** — the 2 outstanding both need a human |

## 1. Tokenization (§19)

191 pieces in both required views. The grid resolution is chosen per piece as the
finest value that represents it **exactly** — never by rounding:

| Steps per quarter | Pieces | Meaning |
| ---: | ---: | --- |
| 4 | 167 | sixteenth-note grid, the spec default |
| 8 | 22 | thirty-second notes |
| 12 | 2 | triplets against sixteenths |

`grid_unsupported: 0`, `roundtrip_failures: 0` — decoding every grid sequence
returns exactly the attacks it claims to carry. Vocabulary is a closed 196-token
set with hash `af1e20ac…`, so the hash does not drift with corpus content.

**One honest limitation, declared rather than hidden:** a grid holds one pitch per
voice per step, so 185 intra-voice double-stops across the corpus cannot be
represented in the grid view. Each is recorded in that piece's
`voice_collision_detail`, and the event view keeps them all — which is exactly why
§19.1 forbids discarding the canonical layer after tokenization.

## 2. Rule framework (§15) — and what the corpus said about it

24 rules: 4 notational invariants, 5 hard constraints, 8 conditional, 3 soft
preferences, 4 structural targets. Every rule cites a source or is explicitly
marked `project_policy` / `empirical_rule` (§3.4).

§15.6 says no rule becomes hard merely because a textbook phrases it
categorically. Running every rule against 41 **held-out** Bach pieces produced
four corrections, each driven by evidence rather than taste:

| Finding | Evidence | Resolution |
| --- | --- | --- |
| `voice_range` fired **76×** on one fugue | SATB tessitura applied to keyboard music | Added range *profiles*; WTC uses `keyboard`, chorales `satb` |
| `max_dead_air` fired on chorales | `bwv129.05` has 23½ quarters of silence at the start — the *instrumental introduction* | Rule stands down for extracts, where excluded parts are playing |
| `parallel_fifths` fired on real Bach | Two cases: one across a phrase rest (not consecutive motion at all), one in BWV 848 m39 between ornamental **offbeat sixteenths** | Detector now requires genuinely adjacent simultaneities; offbeat parallels excluded except in `species_strict` |
| `suspension_resolution` fired on **93%** of pieces | Any note held while the bass moves is dissonant against some passing harmony | Required preparation-as-consonance and metric strength (→22%), then **demoted hard→soft** because 22% is still too high to call a hard law |

Final calibration:

| Profile | Hard rules firing on held-out Bach | Status |
| --- | --- | --- |
| `bach_empirical` | **none** | ✅ calibrated |
| `high_baroque_conservative` | `voice_monophonic` (fp 0.12) | deliberate: generated voices are strictly monophonic; Bach's keyboard divisi is a notational feature we do not generate |
| `species_strict` | 8 rules | by design — this is Fux, explicitly *not* a description of Bach |

## 3. Hard-valid generation (§16)

10,000 accepted fragments in 59 minutes. CP-SAT proposes; a **logically separate**
validator re-derives schema conformance, hard violations, structural completeness
and duplicate status from the serialized fragment alone, with no access to the
solver model (§16.4). Of 10,007 solved candidates, 10,000 were accepted, 7
rejected as duplicates — no candidate was ever accepted because the generator
believed it was correct.

Coverage is balanced **by construction** across all 12 configured axes — voice
count, mode, seven keys, three meters, length, subject length, entry order,
subject voice, rhythmic density, cadence target, countersubject, register — by
walking the coverage grid rather than sampling and hoping.

Splits: 8,042 train / 1,000 validation / 958 test, keyed on the parent subject so
everything derived from one subject lands in one split (§16.6).

**A bug the round-trip caught:** entry onsets like 13/4 did not align to a fixed
eighth-note grid, and the solver was silently snapping them — moving the subject
and producing answers that failed validation. The grid is now derived from each
request's own rhythmic values, and an unaligned onset is **refused**, not rounded.

## 3b. Generator revision: what a reviewer's critique changed

The first release satisfied every hard rule and still did not sound like music.
Seven rounds of listening feedback and one detailed written critique from a
reviewing agent drove a sequence of solver changes, each found by measuring the
generated corpus against real Bach rather than against a textbook.

| defect | measured before | after | how it was found |
| --- | ---: | ---: | --- |
| semitone clusters | 50.6% of sonorities | **0%** | listener said "terrible"; Bach sits at 9.8% |
| augmented triads (minor keys) | 23% | **1%** | one prompt kept sounding alien; major keys were 0% |
| voices re-striking one pitch | 71% of fragments | **0%** | attack was implied by pitch-change, not equivalent to it |
| drone voices (a note held all fragment) | common | **0%** | introduced *by* the previous fix; participation counted duration, not motion |
| all voices attacking together | 39.8% of onsets | **5.9%** | Bach fugues 2.0%, Bach chorales 54.3% — the output was rhythmically a chorale |
| melodic tritones | 23.1 /1k notes | **0** | Bach 9.7 |
| direct fifths into outer voices | 9.8 /1k notes | **0** | Bach 1.9 |
| awkward subject intervals | 22% of subjects | **0%** | a diminished fourth in a subject is restated at every entry |
| unprepared cadences | 47% | **20%** | dominant now required before the final tonic |
| frozen half-bars | 8% of fragments | **0%** | per-voice activity floors cannot see collective stagnation |
| hammered melodic peaks | 32% of fragments | **0%** | a line should reach its high point once |

Two lessons are recorded here because they recurred:

**A constraint solver routes around whatever is not constrained.** Fixing clusters
exposed tritones; fixing repeated notes produced drones; banning tritones alone
pushed unrecovered leaps from 97 to 160 per 1000 notes. Each fix revealed the next
defect rather than finishing the job.

**A metric stated crisply enough to test is usually stricter than Bach.** Three
times a check was written that real Bach fails: a leap-recovery test demanding
under 35% when Bach leaves 63% of his leaps unrecovered; a "nameable diatonic
triad" harmony metric on which Bach scores 27% and the generator 35%; and a
per-piece false-positive rate that demoted rules Bach observes almost always.
Every threshold here is now calibrated against the corpus, and where a metric
turned out to be unable to measure the thing it named, that is said rather than
the number being reported as a result.

Ablation switches (`solver.FEATURES`) exist because four constraints added
together dropped the solve rate from 63% to 7% and there was no way to tell which
was responsible without turning them off one at a time.

## 4. Controlled-invalid generation (§17)

10,000 accepted, each carrying **exactly one** intended hard violation and no
unintended one, each naming the valid parent it came from, each in its parent's
split.

Per-rule distribution — 8 rules at exactly 12.5%, under §17.4's 15% cap:

| Rule | Count | Share |
| --- | ---: | ---: |
| answer_transformation | 1,250 | 12.5% |
| cadence_target_completion | 1,250 | 12.5% |
| parallel_fifths | 1,250 | 12.5% |
| parallel_octaves | 1,250 | 12.5% |
| subject_entry_identity | 1,250 | 12.5% |
| timing_positive_duration | 1,250 | 12.5% |
| voice_participation | 1,250 | 12.5% |
| voice_range | 1,250 | 12.5% |

Four of §17.3's listed mutators are **not used here and this is recorded, not
quietly dropped**: `break_leading_tone_resolution`,
`create_illegal_voice_overlap`, `create_prohibited_melodic_leap` and
`create_unresolved_suspension` target rules that are *soft* in the generation
profile — precisely because §15.6's measurement showed Bach breaks them — so they
cannot produce the **hard** violation §17.2 demands. They remain implemented and
usable under `species_strict`.

**Two bugs found here.** The mutator selector picked the strict minimum count, so
one mutator that never succeeded consumed 3,591 of 3,600 attempts; it now rotates
and retires a persistently failing mutator. And the rule classifier labelled an
*unresolved* suspension as `unprepared`, meaning the one rule meant to catch it
could never fire.

## 5. Preference pairs (§18)

4,193 candidates across 700 prompts, then 3,150 pairs (3,000 + 150 consistency
repeats) across 601 prompts. Every pair shares subject, prefix, formal target,
length budget, rule profile and range profile, differing **only in the stochastic
seed** (§18.2) — verified by a gate that recomputes `shared_conditions_hash` from
both candidates.

This needed a dedicated generation pass. The §16 set gives every request its own
subject, so no two fragments there share a prompt and §18.2 can never be satisfied
from it; pairs are built from prompts solved repeatedly with different solver
seeds instead.

All 3,150 pairs were selected because the diagnostics **disagree** — the top
priority in §18.3 — rather than at random.

## 6. Exports (§20) and gates (§21)

| Export | Contents |
| --- | --- |
| `style/` | 191 real Bach movements, split 150/20/21 |
| `feasibility/` | 8,192 train + 1,020 validation hard-valid records |
| `constraint_critic/` | 10,000 positives + 10,000 matched negatives |
| `structure/` | 804 annotations + 566 section windows |
| `preferences/` | **empty — by design** (see §7) |
| `metadata/` | 1,029 sources, 191 pieces, 191 splits, 20,000 derivatives |

All **17 validation gates pass**, including the ones that bite hardest: 10,000
derivative split-inheritance checks, 10,000 single-violation checks, 3,150 prompt
identity checks, and a re-verification of source checksums against the bytes on
disk.

## 7. ⚠️ The two outstanding criteria both require a person

The pipeline cannot complete these, and §27 forbids it from pretending otherwise.

| §28 criterion | State | To finish |
| --- | --- | --- |
| 50–100 excerpts manually approved (§14.6) | 300 proposed, **0 approved** | `python -m fugue_data.review.app` → approve 50–100 |
| ≥500 reviewed preference pairs (§18.6) | 3,150 candidates, **0 reviewed** | `python -m fugue_data.review.pairs_app` → label ≥500 |

Both review applications are built and working: playback via Web Audio, piano-roll
notation, approve/reject/correction, tags and comments for excerpts; blinded
candidates with randomized left/right presentation, ties and `both_poor` for
pairs. Both persist atomically and never overwrite a decision already made.

The `style` export therefore contains no excerpts and the `preferences` export is
empty. Those are truthful artefacts: an empty preference export means labelling
has not happened, not that no candidates exist.

## 8. Visual QA (§21.1)

294 artefacts rendered to `reports/visual_qa/` and recorded in
`manifests/visual_qa.json`: 20 WTC fugues, 20 chorales, all 154 anomalous
normalizations, 50 synthetic-valid fragments spread across coverage categories,
and 50 controlled negatives with the violated notes ringed in red.

Every artefact carries `inspected: false`. The pipeline renders them; it does not
claim anyone has looked at them.

## 9. How to reproduce

```bash
PYTHONPATH=src .venv/bin/python -m fugue_data.cli tokenize
PYTHONPATH=src .venv/bin/python -m fugue_data.cli propose-excerpts
PYTHONPATH=src .venv/bin/python -m fugue_data.cli generate-valid    # ~59 min
PYTHONPATH=src .venv/bin/python -m fugue_data.cli generate-invalid
PYTHONPATH=src .venv/bin/python -m fugue_data.cli generate-pairs
PYTHONPATH=src .venv/bin/python -m fugue_data.cli export --all
PYTHONPATH=src .venv/bin/python -m fugue_data.cli validate --all
PYTHONPATH=src .venv/bin/python -m fugue_data.cli report
```

Or `snakemake all --cores 16`, which stops at the two human checkpoints.
Generation is checkpointed and resumable; every stage returns nonzero on a failed
acceptance criterion.

## Best-of-N ranking under the style prior (2026-09-27)

The prior separates real from synthetic cleanly — held-out Bach scores 2.83
content perplexity against 5.23 for generated fragments, and the generated pool
spans 0.65–3.99 NLL, so there is real headroom to rank within it.

**But raw NLL is not a usable ranking signal.** Measured over 200 generated
fragments, `corr(NLL, HOLD-fraction) = -0.83`. The prior's 20 favourite fragments
average 85% held notes and 23.6 notes; its 20 least favourite average 47% held
and 61.0 notes. Long held notes are trivially predictable, so best-of-N by raw
NLL would have systematically selected the most static music available —
reintroducing precisely the simultaneous stagnation the solver was revised to
remove two rounds earlier.

Ranking is therefore done on the residual against a density baseline (quadratic
in HOLD-fraction, fit over a 300-fragment pool): *how Bach-like is this fragment
compared with others of the same note density*. After adjustment
`corr(score, HOLD-fraction) = +0.000`, and the best- and worst-scoring groups
have matching density (0.587 vs 0.616). Density follows from the request's
`allowed_rhythmic_values` rather than from quality, so removing it costs no real
signal.

This is the same lesson as the rule calibration in §15.6, in a new place: **a
learned score optimises whatever correlates with its objective, and "predictable"
is not "good".** A likelihood model used as a ranker needs its nuisance
correlations measured before it is trusted, not after.

The A/B that tests whether the ranking helps a listener compares prior-best
against a random pick **from the same N solver outputs**, not against a single
solve — otherwise the extra solver attempts, not the ranker, would be under test.

### Result: best-of-N ranking does not demonstrably help a listener

40 blind comparisons, prior-best vs a random pick from the same 12 solver outputs.

| | |
| --- | --- |
| decisive verdicts | 23 (17 hedged, 42%) |
| prior wins | 13 |
| control wins | 10 |
| win rate | 56.5%, Wilson CI **36.8–74.4%** |
| dose-response (large vs small score gap) | 67% vs 45%, Fisher p = 0.41 |

Not distinguishable from chance. The dose-response gradient points the right way
but is noise at 11–12 items per bucket. Establishing a 56.5% effect would need
~400 decisive verdicts, i.e. ~690 more comparisons at the measured hedge rate —
far too expensive for an effect that small even if it is real.

The 42% hedge rate independently reproduces the 37% measured in the September
self-consistency study, which is the one clean confirmation here.

**Reading.** Ranking can only reorder what the solver already produces. The faults
the review agent named — directionless harmony, obscured subject — are absent
from the solver's output space entirely, so no re-ranking of that output can
introduce them. The style prior's value is as a training signal and as an
eventual *generative* component, not as a re-ranker of CP-SAT output.

## What is actually enforced, and what only looks enforced (2026-09-27)

Prompted by "does the solver avoid parallel fifths, illegal jumps, and other
illegal moves?" — verified rather than asserted.

**Genuinely enforced (hard; blocks acceptance).** `parallel_fifths` and
`parallel_octaves` are hard in `high_baroque_conservative`, checked over every
voice pair, mod 12 so compound fifths count, with rests correctly breaking
adjacency. An **independent** detector — written from the definition and
deliberately not importing the rule engine, because a gate cannot verify itself —
found **zero parallel fifths or octaves in 300 shipped fragments**, on-beat and
off-beat. The same detector fires 11 times on 60 real Bach chorales, which proves
it can.

One deliberate exemption: parallels between two off-beat notes are not counted by
default (`parallel_requires_metric_onset`), because BWV 848 m39 has textbook
parallel fifths between offbeat sixteenths. Calibration, not an oversight — and
the independent scan found none off-beat either.

**Solver-side, tighter than the validator.** Free voices are capped at a perfect
fourth (`free_leap_max = 5`), melodic tritones are forbidden, and cross-relations
are forbidden by letter name. Measured: **0.0% melodic tritones, 0.0% beyond an
octave, largest leap 12 semitones against Bach's 17.** Stricter than Bach.

**Soft only — recorded, never blocking:** `leading_tone_resolution`,
`suspension_resolution`, `dissonance_treatment`, `voice_spacing`, `voice_overlap`,
`melodic_leap_limit`, `augmented_interval_melodic`, `leap_recovery`. Diagnostic
only: `direct_fifths_outer` (hidden fifths), `voice_crossing`.

### The finding that matters: the melodic profile is inverted

| melodic interval | Bach | generated |
| --- | ---: | ---: |
| unison/step | **78.7%** | **40.9%** |
| third | 7.1% | 26.7% |
| fourth | 7.4% | 17.7% |
| fifth | 4.1% | 12.7% |

Nothing here is illegal. The lines break no rule and are *stricter* than Bach on
every prohibition. They simply are not melodies: Bach's counterpoint is
overwhelmingly stepwise, and the generator leaps in nearly 60% of its moves.

This is the same shape of defect as the root-motion finding recorded above —
**a legality cap with no preference for the distribution Bach actually writes.**
`free_leap_max = 5` permits every leap up to a fourth and prefers nothing, so the
solver takes them freely. Harmony: fifth:third ratio 2.49 in Bach, 0.50 generated.
Melody: 79% steps in Bach, 41% generated. Both inverted, for the same reason.

Together they explain the listening reports: leapy lines over third-drifting
harmony give neither a memorable line nor a sense of direction.

### Fix: distribution floors, calibrated by measurement (2026-09-27)

Two constraints added behind ablation switches (`stepwise`, `fifth_motion`),
targeting the measured Bach rates rather than a textbook rule:

- **stepwise floor** — of the slot pairs where a free voice changes pitch, at
  least 80% must move by ≤2 semitones. Free voices only: entry slots are exempt
  because the subject leaps as written, which is why a floor above Bach's
  whole-fragment rate is not a floor above Bach.
- **fifth-motion floor** — at least 40% of beat-to-beat root transitions in the
  solver's own chord sequence must move by a fifth.

Swept at 60/70/75/80 against 25/35/37/40. Results on a **held-out request seed**
(77777, not the seed the floors were tuned on):

| | before | after | Bach |
| --- | ---: | ---: | ---: |
| stepwise motion | 45.6% | **75.2%** | 75.2% |
| fifth motion | 22.5% | 33.3% | 37.2% |
| third motion | 31.5% | 27.8% | 14.4% |
| solve rate | 33% | 31% | — |
| solve time / 70 requests | 10s | 23s | — |

Stepwise motion now matches Bach exactly and generalizes across seeds. Fifth
motion improved by half but remains short.

**Two things this did not fix, stated plainly.** Third motion barely moved on the
held-out seed (31.5% → 27.8%) although it fell to 15.3% on the tuning seed — that
gain did not generalize. And the fifth floor does not track its own target: the
realized rate sat near 30% whether the floor was 25 or 40. The constraint binds on
the solver's chord *variable*, while the metric template-matches the realized
notes, and the two diverge because entry voices are exempt from chord membership.
The constraint is satisfied; the audible harmony only partly follows it.

**On solve rate.** It was identical (16/60) across all seven floor settings, which
looked like the constraints being inert. They are not: 56 of 60 requests reach
CP-SAT, and the 40 that fail do so for reasons unrelated to these floors. The
floors shape the solutions without shrinking the solvable set.

### Dissonance density ceiling (2026-09-27)

A listener flagged one AFTER fragment as "a lot of dissonance in the middle".
Checking it: all 16 of its dissonances fell off the beat (the on-beat consonance
constraint held), and **100% of them were proper passing or neighbour tones**.
Correct treatment of each dissonance individually, and it still sounded wrong.

The measurement that explained it was density, not treatment:

| | dissonances / quarter |
| --- | ---: |
| the flagged fragment | 1.78 |
| all AFTER fragments | 1.62 |
| all BEFORE fragments | 2.43 |
| **real Bach** | **1.00** |

Nothing in the model counted how many dissonances sounded at once. Added
`diss_density`: a ceiling on the share of (voice-pair, slot) cells carrying a
dissonant interval class. Swept at 26/22/18/14/12/10/8 on held-out requests; 12%
is where all three distribution metrics land on Bach simultaneously.

| | before | cap 12% | Bach |
| --- | ---: | ---: | ---: |
| dissonances/quarter | 2.08 | 0.95 | 1.00 |
| stepwise motion | 74.8% | 72.4% | 75.2% |
| fifth motion | 33.3% | 35.7% | 37.2% |
| solve rate | 30% | **23%** | — |
| solve time / 70 requests | 27s | **54s** | — |

The solve-rate and solve-time costs are real and will matter when a full pool is
regenerated.

**The general lesson, third time now.** Rules were calibrated against Bach and
each dissonance obeys them; root motion and stepwise motion were calibrated and
now match. Each time the defect was in a quantity nothing was counting —
*how often*, not *whether*. A constraint set that is correct rule-by-rule can
still produce a distribution nothing in the corpus resembles.
