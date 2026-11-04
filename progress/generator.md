# Generator track

## Requirements (owner, 2026-11-03)

Given only a subject, the generator writes a complete fugue in which:

- **R1 — Exposition.** The subject enters in every other voice in turn
  (answers at the dominant, tonal where needed), each entry sounding as the
  subject, against counterpoint in the voices already in.
- **R2 — Returns.** The subject keeps coming back through the piece, not just
  in the exposition: middle entries, a final entry in the tonic, possibly
  stretto.
- **R3 — Modulation.** Middle entries and cadences in related keys (relative,
  dominant, subdominant, ...), then a return to the tonic.
- **R4 — Sequence.** Episodes between entries built from sequenced material,
  usually derived from the subject or countersubject.
- **R5 — Not a copy.** Sounds like Bach without reproducing training pieces.

## Why v1 misses entries (diagnosis)

v1 is told *that* a voice is in a subject entry (the `role` feature) but never
*what* to play there, and only ~19 training fugues carry those labels. It has
to learn "role = subject, so reproduce the line in the prefix, transposed to
this entry's level" from a few hundred examples. It didn't. Nothing in v1
forces an entry either: the plan is a hint, not a constraint.

## Plan of record

The subject is known, so entries need not be learned at all. Split the job:

1. **Entry detector** (`generator/entries.py`). Finds subject entries
   (voice, onset, transposition, real/tonal) in all 48 fugues by matching the
   subject's interval pattern, validated against the 23 Algomus-labelled
   fugues. Gives every fugue a plan, roughly 2.5x the planned training data.
2. **Plan as notes, not just roles.** At entries the plan carries the actual
   pitches. Decoding writes them in as a hard constraint; the model writes the
   counterpoint around them. R1 and R2 then hold by construction, and a
   feature tells the other voices what the entering voice will play.
3. **Rule masking in decoding.** Tokens that would create a hard-rule
   violation (parallel 5ths/8ves, voice crossing, ...) are masked.
4. **Memorisation metric.** Longest shared interval run with the training
   set, compared against held-out Bach's own overlap.
5. **Plan generator** (R2, R3). For a new subject there is no Bach plan to
   borrow. Build one from the corpus' plan statistics: exposition order,
   middle-entry key areas and their order, episode lengths, final entry.
   Key areas set the cadence/key features, which drive R3.
6. **Sequences** (R4). Detect sequences in the corpus (a melodic/harmonic
   unit repeated at a transposition, 2+ times) and add them as a plan feature
   for episodes, so the model learns to write them when asked. Later, a
   reward term for them.
7. **Reward/steering** from the corrupted-pairs judge, once 1–6 exist.

## Status

- **v1** (2026-11-03): val ppl 2.37, test 2.36. Plan doesn't help (19 planned
  fugues). Samples: no copying, entries missing, 5–8 parallels each.
- **Steps 1–4 done** (2026-11-03):
  - Entry detector: 95% precision / 90% recall on plain entries vs Algomus;
    all 48 fugues now planned (36 train, was 19).
  - v2 adds entry-note features (`pnote`, `epc`) and a local-key feature
    (`key`, Krumhansl–Schmuckler per bar, relative to the tonic). Test fugue
    ppl 3.32 -> 2.82, but on notes outside entries the plan still doesn't help
    (3.24 with vs 3.09 without): the gain is the entries.
  - Decoding: entries forced from the plan, parallel 5ths/8ves masked.
    Checked by the rule engine: 0 parallels in all v2 samples (v1: 5–8).
  - Memorisation: held-out Bach shares runs of median 10 / max 14 intervals
    with training; v2 samples 5–9. No copying.
- **Steps 5–6 started** (2026-11-03):
  - v3 adds key look-ahead features (next key, distance to it); v4 adds
    per-voice sequence roles (`sequences.py`: unit of 1–4 beats stated 3+
    times at a consistent step; Bach's fugues: median 15% of the time).
  - Decoding: soft key bias (`--key-bias`; 3 is best so far) and forced
    sequence restatements (`--force-sequences`: the model's *own* unit,
    transposed diatonically). Plan-only CFG did not help.
  - On Bach's plans (v4, key bias 3, forced sequences): bars the plan puts
    away from home match 33–67% (v2: 0–33%); sequences cover 9–24% (v2: 0%;
    Bach 24–25%). Still 0 parallels; longest shared run 7–9.
  - `plan.py` + `compose.py`: a fugue from a subject alone. The form comes
    from a training fugue with similar subject length, rescaled; entries are
    the subject moved degree-for-degree into each entry's key. Tested on the
    held-out subjects of I/2 and II/19.
- **Open:**
  - Answers are real, not tonal; no countersubject planning.
  - Compositions take one template's form: limited variety, and few planned
    sequences when the template has few (II/19 composition: 1).
  - Density runs ~15–25% above Bach with key bias 3.
  - Older open items below.
  - R3 is weak: samples follow the planned key path in only ~30–50% of bars
    and drift home (II/17: Bach visits vi, iii, ii; samples mostly stay in I).
    Next: classifier-free guidance on the plan (plan dropout already trains
    the unconditional model), then a key-agreement steering term.
  - R4 not started (sequence detection + feature).
  - R1/R2 for a *new* subject need the plan generator (step 5); today's
    entries come from a Bach piece's plan.
