# Fugue Composition — Training Spec (draft)

**Status:** proposal, 2026-10-31. Companion to
`fugue_data_collection_pipeline_spec.md`, which is complete (19/19 on its §28
Definition of Done). That spec defines the *data*; this one defines what to train
on it and how we will know whether it worked.

**Target hardware:** RTX 5090 (32 GB). Everything below is sized for a single
GPU. The data-collection work ran entirely on an M-series Mac; nothing here
requires a cluster.

---

## 1. What we already know, and why it shapes this

These are measured results from the data-collection phase, not assumptions. They
are the reason this spec is shaped the way it is.

### 1.1 Every quality gain so far came from a measured constraint

| change | before | after | Bach |
| --- | ---: | ---: | ---: |
| stepwise melodic motion | 40.9% | 75.2% | 75.2% |
| root motion by fifth | 14.6% | ~33% | 37.2% |
| dissonance density /quarter | 2.08 | 0.95 | 1.00 |

Each was found by measuring real Bach, discovering the generator was inverted
relative to it, and adding a floor or ceiling. None involved learning.

### 1.2 Nothing learned has yet produced an audible improvement

- **Best-of-N under the style prior**: prior-ranked vs a random pick from the same
  12 solver outputs, blind. 13–10 over 23 decisive verdicts, 56.5%, Wilson CI
  36.8–74.4%. Not distinguishable from chance.
- **Reward model on 4,049 agent preference labels**: predicted the human's choice
  at 57.4%, CI 49.7–65.1%. Not significant.
- **Style prior as a ranker** is actively misleading without correction:
  `corr(NLL, HOLD-fraction) = -0.83`. Its favourite fragments are the most static
  ones, because held notes are easy to predict.

### 1.3 The one learned component that helped was generative, not evaluative

Sampling motifs from the style prior and handing them to CP-SAT as subjects raised
the solve rate from **42% to 62%**. The prior proposes; the solver disposes.

### 1.4 The constraint critic result (§20.3), and what it bounds

A 1.9M-parameter bidirectional encoder, trained on 10,000 hard-valid against
10,000 single-violation fragments with perfect labels:

| metric | value |
| --- | ---: |
| test accuracy | 0.676 |
| matched-pair accuracy (mutation vs its own parent) | 0.688, CI low 0.659 |

Per-rule recall is the real finding:

| rule | recall |
| --- | ---: |
| voice_range | 100% |
| voice_participation | 100% |
| timing_positive_duration | 80% |
| **parallel_fifths** | **0%** |
| **parallel_octaves** | **0%** |
| **subject_entry_identity** | **0%** |
| **answer_transformation** | **0%** |
| **cadence_target_completion** | **0%** |

It learned every rule checkable from a single local fact, and none that requires
relating two voices to each other or a passage to an earlier one. Counterpoint is
entirely the second kind.

**This bounds one architecture at one scale, not all critics.** Re-running it at
5090 scale is the first experiment in §3, precisely because the negative result is
the main thing standing between us and an RL plan.

### 1.5 Human label reliability is measured, and must be modelled

The human reviewer re-judged 60 randomly sampled pairs blind:

- repeated their own earlier pick: **47%**
- moved to tie / both-poor: **37%**
- flipped outright: 17%

When they committed to a side, they repeated themselves **74%** of the time
(28/38, CI 60–88%). So the labels are not noisy — they carry a large, stable
abstain region. The 37% hedge rate reproduced on two later samples (42%, and the
excerpt round). **Treat hedges as "no preference", never as half a win.** Folding
them in manufactures signal.

Current totals: **561 human labels (356 decisive)**, 4,049 agent labels. Agent and
human agree at κ≈0.07 — the agents agree with each other (κ 0.48–0.58) and with
the human barely at all.

---

## 2. What the system should do

**Open question for the owner — this spec cannot be finished without it.** The
answers change the architecture, not just hyperparameters:

1. **Output scope.** A complete fugue (exposition → episodes → final entry), or
   N-bar continuations of a given opening?
2. **Conditioning.** Given a subject, or inventing one?
3. **Success criterion.** "Passes the hard rules and sounds Bach-like to the
   owner", or something stricter and externally checkable?
4. **Length.** Current fragments are 4 bars. Real development — sequence,
   modulation, stretto — is structurally impossible at that length, no matter
   what is trained. Any claim about "development" requires longer windows, which
   changes generation cost, review cost, and the splits.

The experiments in §3 are chosen to be useful under *any* of these answers.

**Owner's answers (2026-11-03), recorded verbatim in intent:**

1. **Output scope:** a complete fugue, not a continuation.
2. **Conditioning:** given a subject. From whatever subject is passed in, the
   generator writes the rest: the subject entering in the other voices
   (exposition), then returning from time to time through the piece.
3. **Required techniques:** modulation (middle entries in related keys) and
   sequence (episodes built from sequenced material) must be present, along
   with the other standard devices (countersubject, stretto, pedal, final
   entry in the tonic).
4. **Length:** BWV 543 scale, so windows are whole pieces.

What this implies for the generator is tracked in `progress/generator.md`.

---

## 3. Experiments, in dependency order

Each has a stated kill criterion. The point is to fail cheaply and early.

### E1 — Constraint critic at scale *(blocking)*

Re-run §1.4 with the capacity a 5090 allows: 50–150M parameters, longer context
(full fragment, no truncation), relational inductive bias (pairwise voice
attention, or explicit voice-pair features), and 10× the synthetic negatives if
needed (generation is cheap; it is already automated).

- **Metric:** per-rule recall on held-out matched pairs. The aggregate number is
  not the result; the breakdown is.
- **Pass:** `parallel_fifths` and `parallel_octaves` recall > 0.8. These are the
  simplest *relational* rules — two voices, two time steps. A model that cannot
  learn them will not learn voice independence.
- **Kill:** if scale does not move them off 0%, abandon the learned-critic branch
  entirely and keep the rule engine as the only validator. This is a legitimate
  outcome, not a failure of the project.

### E2 — Style prior at scale

Current: 1.87M params, test content perplexity 2.99 vs 4.72 n-gram baseline.

- Scale to 50–150M, context ≥ 2048 (whole pieces, not windows).
- **Metric:** test content perplexity (content tokens only — the grid interleaves
  structural tokens that any model predicts for free; whole-stream perplexity
  flatters). Split-respecting: the 21 test pieces are held out whole.
- **Also report** the HOLD-fraction correlation from §1.2. If it stays near −0.83,
  the prior remains unusable as a ranker without density adjustment at any scale.
- **Pass:** content ppl < 2.0 **and** coherent 8-bar unconditioned samples
  (currently the prior drifts after ~2 bars, and cold samples collapse into either
  an all-rest or an all-hold attractor).

### E3 — Prior-as-generator, solver-as-validator

The one direction with evidence behind it (§1.3). Sample candidate material from
the scaled prior; accept only what passes the rule engine.

- **Metric:** acceptance rate of prior samples under `high_baroque_conservative`,
  and a blind A/B against current CP-SAT output.
- **Pass:** prior-proposed material wins a blind A/B at a Wilson CI lower bound
  above 0.5. Budget ~60 comparisons; at the measured 37% hedge rate expect ~38
  decisive, which needs roughly a 68% win rate to clear.
- **Note:** the A/B must compare like with like. Best-of-N was tested against a
  random pick *from the same N solver outputs* precisely so the extra solver
  attempts were not what was under test.

### E4 — Reward model on human labels only

Only after E1 or E2 passes. 356 decisive human labels is thin for Bradley-Terry;
the agent labels do not substitute (κ≈0.07).

- **Metric:** held-out accuracy on the human's *decisive* labels, with hedges
  excluded, reported with a Wilson interval.
- **Pass:** CI lower bound > 0.5. Nothing below that is a signal.
- **Prerequisite:** more decisive human labels. Collecting them is cheap in
  machine terms and expensive in the owner's attention, so do not start until a
  generator exists whose output is worth ranking.

### E5 — RL *(not yet justified)*

Gated on E1 **and** E3. Optimising against a critic blind to parallel fifths
would teach the policy to write them. Do not start this until a learned signal is
demonstrably better than the rule engine at something.

---

## 4. Rules that carry over from the data phase

These were learned the hard way and should not be relearned.

1. **Measure against Bach before asserting a rule.** Four textbook rules were
   demoted during calibration because the corpus contradicted them. A metric
   stated crisply enough to test is usually stricter than Bach.
2. **A constraint solver routes around whatever is not constrained.** Expect any
   unconstrained quantity to drift to an extreme.
3. **Correct rule-by-rule is not correct in distribution.** Three separate bugs
   were of the form "every instance obeys the rule, but *how often* it happens
   resembles nothing in the corpus". Always check the rate, not just the legality.
4. **A likelihood model optimises whatever correlates with its objective.**
   "Predictable" is not "good". Measure nuisance correlations before trusting a
   learned score as a ranker.
5. **Never merge agent judgements into human ones.** Separate files, separate
   reviewer kinds, `κ` reported between them. §27 forbids presenting automatic
   selection as human approval.
6. **Report the breakdown, not the aggregate.** The critic's 68.8% looked like a
   working model and was 0% on five of eight rules.

---

## 5. Known defects to fix before or during training

| defect | impact | effort |
| --- | --- | --- |
| Parser drops rest-chords in split Humdrum spines | 3 of 191 pieces have voices that do not end together; WTC I fugue 6 is 132.5q in 3/4 | small |
| No annotations for WTC II or the chorales | 23 of 191 pieces annotated; the excerpt boundary rule only runs where annotations exist | large (annotation work) |
| `texture_change` trigger slices through the entry causing the change | 21 excerpt windows | medium |
| Fragments are 4 bars | no development is representable at this length | design decision, see §2.4 |

---

## 6. Reproducibility

Unchanged from the data spec: sources pinned to full commit SHAs, raw trees
immutable and checksummed, splits by complete work with leakage gates, every
stage restartable under a fixed seed. Training must preserve the split manifest —
the 21 test pieces are held out whole, before any window was cut, and no
derivative of a held-out work may appear in training.

Model checkpoints must record what the current style prior does not: step count,
final train/validation loss, whether early stopping fired, the data hash, and the
git SHA of the code that produced them.
