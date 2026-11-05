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
- **Hierarchical plan, steps 1–2 started** (2026-11-03). The owner judged
  the output still very bad and asked for a decision layer that chooses the
  big events (entries, sequences, modulations, other devices). Diagnosis: v2/v4
  samples generated under *Bach's own* plan also sound bad, so the bar-to-bar
  writing is the main problem, not only the form. Plan: three levels, each
  deciding for the one below: form planner (sections, entries, keys, cadences,
  texture) -> harmony planner (chord + bass per beat) -> note model.
  With ~130 fugues there is not enough data to learn the form level with a
  network; it will be fugue-form rules with choice probabilities counted from
  the corpus.
  - **Corpus expansion** (`src/fugue_data/baroque/`, owner's request): 15
    public repositories pinned by commit. 947 new pieces parse, 747 have voices
    clean enough for the note model, including **86 new fugues** (51 organ
    fugues by J.S. Bach, W.F. Bach, Krebs, Kellner, van Noordt; 16 Art of
    Fugue; 13 Frescobaldi; 2 Musical Offering; 4 Couperin): 134 fugues in all.
    Also Inventions and Sinfonias, 289 more chorales, Corelli, Vivaldi, Bach
    suites, Couperin. MuseData Corelli left out (restrictive licence);
    kern.humdrum.org was down. Training tokens 237k -> 1.42M. BWV 543 is test.
  - **Step 1: labels** (`labels.py`, data/labels/): chords per beat, cadences,
    pedals, texture, entries with stretto, countersubjects, sections. Checked
    against expert labels:
    | detector | reference | result |
    | --- | --- | --- |
    | chord root / bass | DCML Corelli (20.8k beats) | 74% / 88% |
    | chord root / bass | DCML Frescobaldi | 70% / 91% |
    | chord root / bass | DCML all (incl. solo violin/cello, 2-voice suites) | 59% / 72% |
    | cadences | Algomus WTC I (102) | precision 48%, recall 65% |
    | pedal points | Algomus WTC I (19) | precision 75%, recall 68% |
    | countersubjects | Algomus WTC I (174) | precision 52%, recall 59% |
    Weak where harmony is implied by one line (solo violin/cello: 39%).
  - **Step 2: chord layer in the note model** (v5): per-step chord, next chord,
    bass (relative to the tonic) and per-voice bar texture as plan features;
    detected cadences/pedals where no expert labels exist; chord dropout 0.3
    so the same model runs without the chord layer.
  - **v5 results.** Held-out WTC fugues, perplexity on free notes (outside
    entries): v4 3.22 -> v5 1.79 with chords, 2.01 without. Sampling under
    Bach's plan (I/2, II/17, BWV 543 bars 1-40; 3 samples each):
    | | chord root / bass match | key match (away bars) | notes/bar vs Bach |
    | --- | --- | --- | --- |
    | v4, key pull 3 | 0.22-0.31 / 0.29-0.37 | 0.33-0.67 | +0-25% |
    | v5 no chords, key pull 3 | 0.19-0.35 / 0.34-0.40 | 0.44-0.68 | -25 to +32% |
    | v5 chords, no key pull | 0.45-0.58 / 0.53-0.72 | 0.30-0.89 | -42 to +2% |
    | v5 chords, key pull 3 | 0.47-0.63 / 0.62-0.75 | 0.56-0.89 | -26 to +23% |
    The chord layer is followed: root agreement roughly doubles. Still 0
    parallels. Longest shared runs 7-17; one of 24 is a repeated pedal note
    (the memorisation metric should skip repeated-note runs). Density: too
    sparse on BWV 543 (11-17 vs 19.6), still too busy on II/17 with key pull.
  - Note: GPU 1's fan was unplugged during v5 training (93 C peak). After
    reconnecting: VRAM pattern test, compute check and load test clean (peak
    60 C), but one Xid 8 hang at 23:13 on 2026-11-03 during the first load test;
    9 later sampling jobs ran without errors. Watch for recurrence.
- **Voice crossing / repeated notes** (owner: "voice crossing ... a lot of
  repeated notes at once"). Not caused by the parallel mask (without it all
  three get worse). Causes: Corelli (violins cross 13.6%; WTC fugues 1.1%),
  Vivaldi (27% unisons, 23% repeats), and the key pull (repeats 5.6% -> 10.3%).
  Fix at decoding: `--no-crossing` (a free note stays between its sounding
  neighbours) and `--repeat-penalty 2`: crossings 3.8% -> 0.9%, repeats
  10.3% -> 2.9% (Bach 1.1% / 3.9%), key and chord agreement kept, density
  at Bach's level. Unisons still 4-5% (Bach 1.5%).
- **Models.** Held-out WTC free-note perplexity: v5 1.79; v6 (d512, without
  Corelli/Vivaldi, fugues x3) 1.90; v6b (v5 size, same data) 2.01; **v7
  (v5 data, d512) 1.76**. Dropping data hurt; the larger model helped.
- **Critic / reward model** (`critic.py`). Pairwise-trained encoder, real vs
  worse version of the same bars: 3,982 corrupted pairs + the generator's own
  continuations (`critic_negatives.py`, incl. continuations from the middle of
  each fugue). critic_v2: v5 fugue output vs Bach 74% (validation, ~200
  pairs; v1 with fewer fugue examples: 53%); corrupted test pairs 75%.
  Plan agreed with the owner: (1) critic held-out check, (2) blind A/B of the
  critic's best vs worst of 8 (`critic_ab.py`, 16 cases on test fugues), (3)
  DPO with critic-chosen pairs + Bach continuations, KL to the reference,
  (4) refresh the critic on the new model's output and repeat; GRPO only
  after the critic survives a refresh.
- **Blind test of the critic** (owner, 16 cases): the critic's pick was
  preferred 9/16 (p = 0.40). The critic does not match the owner's ear and is
  not used for training. Owner's notes: 16B "really good", clear improvement
  over earlier versions, but even the better side often has awkward places;
  continuations start close to Bach and get progressively worse.
- **Drift** (`drift.py`, 10 test fugues x 4 samples, 24 bars from mid-piece):
  chord agreement with the plan falls from ~75% to ~55% over 20 bars and the
  model's own surprise rises above its level on Bach; lower temperature does
  not stop the harmonic drift. Fixes: `--chord-bias` (notes of the planned
  chord get a bonus) holds agreement at 66–86% (1.5) / 72–95% (3) through 24
  bars, with fewer harsh dissonances than Bach at 3; and DPO (`dpo.py`) with
  Bach's continuation preferred over the model's (no critic): small further
  gain. Pending owner blind test: data/review/ab_v7_vs_new (v7 T1.0 vs
  v7-DPO T0.85 chord bias 1.5), `python -m fugue_data.generator.ab serve`.
- **Autonomous session, 2026-11-04** (owner away ~20 h; goals: Bach's rules
  forbidden except ~once per five fugues; Bach's "icing" baked into a
  decision model; keep iterating; nothing destructive, nothing posted).
  - Research (two subagents): research/bach_fugue_rules.md (operational
    rules, severities, Bach exceptions, the 15 most audible errors) and
    research/bach_fugue_techniques.md (form proportions, key plans, stretto
    statistics from McDonald 2019, decision rules for a planner).
  - `bach_rules.py`: the research rules as a grid checker, calibrated on 134
    fugues (research/bach_rule_rates.json). FORBIDDEN: parallel 5ths/8ves in
    any motion, outer hidden 5ths/8ves, unresolved leading tone in outer
    voices at V-I, augmented 2nd, doubled leading tone, leaps of a major 7th or
    over an octave, bass crossing, range, final cadence. Others must stay at
    or under Bach's rate (strong-beat unresolved dissonance, chordal 7ths,
    minor-7th leaps, same-direction leaps, long crossings, spacing, cross
    relations).
  - The old parallel mask was leaky: contrary-motion and off-beat parallels
    passed (81 / 19 per 100 bars of octaves / fifths). Now strict.
  - Sampler (`generate`): masks for the forbidden rules, including leaps into
    written-in notes and parallels into them; step-resolution obligation for
    beat dissonances; bass consonance with written-in notes that cannot
    resolve; bar-by-bar selection over K=8 candidates by the weighted rule
    cost, with a density target (fewer notes otherwise always wins).
  - `harmony.py`: chord HMM (root rel. local key, quality, inversion) fitted on
    958 pieces; constrained by cadence formulas, sequence root patterns, pedals.
  - `form_planner.py`: exposition with a tonal answer; middle entries in related
    keys with voice rotation; episodes with sequences (descending fifths,
    rising 5-6, falling 6-3) and the next entry's voice resting; stretto only
    when the subject passes the checker against itself; inversion; dominant
    pedal; final entry in an outer voice; tonic-pedal coda; deceptive cadence,
    Neapolitan, vii°7 into the final entry; Picardy third; written-in final
    chord; countersubject (the line against the answer) restated against later
    entries, alternating above/below; soprano climax at 0.80-0.90.
  - Copies (sequences, countersubject) are soft (+8 logits), not written in:
    literal copies landed in other harmonies and caused most unresolved
    dissonances (26 of 41 on one piece); soft copies keep ~75% of countersubject
    notes and ~64% of sequence notes with dissonance at Bach's rate.
  - `compose2.py`: subject -> plan -> generation -> audit; `review_compositions.py`
    checks cadence realisation, chord agreement, copy adherence, density,
    texture and climax position.
  - Fixes found by reviewing output: the sampler drifted toward sparse music
    (fewer notes = fewer rule breaks; held notes = more probable) -> density
    target from the subject's rhythm; a fixed final-chord voicing caused
    octave leaps and parallels -> pitch-class constraint that overrides other
    bans; free notes leapt into written-in entries -> look-ahead; unisons 4.8%
    (Bach 1.5%) -> neighbour-unison penalty (2.1%).
  - `repair.py`: proofreading pass; tries nearby pitches for the free notes
    involved in each remaining forbidden break (7 notes changed in 20 fugues).
  - **Result** (10 held-out subjects x 2 samples, compositions_v3): 1
    forbidden break in 20 fugues (0.05 per fugue; target 0.2). Strong-beat
    unresolved dissonance 14.7 / 100 bars (Bach 30.6), cross relations 9.9
    (10.2), same-direction leaps 2.3 (11.8). Planned cadences realised 82%,
    countersubject notes kept 66%, sequence notes 74%, density 24.0 vs Bach
    21.2 notes/bar. Model surprise is flat over the piece (no late drift).
  - Long form (`--groups 7`, counter-exposition, two strettos when
    admissible): 107-bar fugue on the BWV 543 subject, 0 forbidden breaks in
    2 samples, cadences 78-89%, climax at 0.83; mid-piece surprise ~0.8 vs
    Bach ~0.65, recovering in the final section; density too high (29 vs 19).
  - Tried and dropped: DPO on rule pairs (clean vs free continuations) did not
    change the model's unmasked rule-break rate (71 -> 74 / 100 bars); the
    masks remove only 4.5% of probability mass per decision, so distilling the
    rules into the model has little to gain.
  - Owner blind tests ready: http://127.0.0.1:8780 (old v7 T1.0 vs current
    pipeline, `ab serve --name old_vs_rules --port 8780`); listening page
    filters "composed v3 (planner, latest)".
- **v4 (latest)**: climax note written into the plan (in window 27/40, was
  17/40), density target capped (22.1 vs Bach 21.2 notes/bar); 39/40 fugues
  free of forbidden breaks. No copying: longest run shared with training
  (model-written notes) median 12 / max 16, held-out Bach median 14 / max 32.
  Original subjects (compositions_original, 3 subjects x 2): 0 forbidden.

- **v5: keys after the first entry** (owner: "the keys are a bit off right
  after the intro subject"). Two bugs in the chord plans: (1) sequential
  episodes wrote their chain in the target key's scale but applied it relative
  to the previous key, shifting every chord by the key distance (I/2: B major
  and A-flat minor planned in E-flat major); (2) the chord HMM, fitted on ~75%
  accurate labels, sampled out-of-key chords (E minor in G minor, minor v as
  the dominant). Fixes: chain roots converted to the local key frame and
  sequential episodes pivot at their start; a key vocabulary prior (diatonic +
  secondary dominants / leading-tone chords / Neapolitan; V not v in minor);
  the tonal answer's head harmonised in the tonic. Music chords in the
  planned key 86.5% -> 90.0% (Bach 84.6%), in the 6 bars after the first entry
  83.9% -> 87.5%; 40/40 fugues free of forbidden breaks; cross relations 8.8
  -> 6.2 per 100 bars.

### How to compose

    # from your own subject (pitch:quarters), key given as tonic pitch class
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m fugue_data.generator.compose2 \
        --subject "D4:1 A4:1 F4:1/2 E4:1/4 D4:1/4 C#4:1/2 D4:1/4 E4:1/4 F4:1/2 G4:1/2 A4:1 D4:1" \
        --meter 4/4 --mode minor --tonic 2 --voices 4 --samples 2 --name my_fugue
    # longer form (BWV 543 scale): add --groups 7
    # from the subject of a corpus fugue: --subject-from bach_wtc1_fugue_02
    # output: data/review/compositions_v2/<name>/ (wav, mid, plan.txt, meta.json with the audit)

- **Open:**
  - (Done 2026-11-04: tonal answers and countersubjects in form_planner.)
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

## v6: the answer entry (2026-11-04)

Owner: "right after the subject entry, when the second voice joined, it becomes very dissonant".
Causes found in the I/2 plan: (1) the answer's opening was kept in the tonic for a fixed quarter
of the subject even when the answer is real, so the free voice used B natural against the
answer's F#; (2) chord planning only checked entry notes struck on the beat (B-flat and E-flat
chords under G-F#-G); (3) the raised-6th, Neapolitan and major-tonic chords in the vocabulary.
Fixes: `tonal_head` (the tonic covers only the notes the tonal answer changed), off-beat entry
notes scored (`EMIT_WEAK_NCT`, `AUG2_CLASH` for the leading tone over the 6th degree),
`EMIT_NCT` 0.15 -> 0.05, a narrower minor/major vocabulary, and a bar-selection cost
(`EXPOSED_W` 1.5) for dissonant beats when exactly two voices sound.

| 40 fugues | v5 | v6 (plan only) | v6x (+ exposed cost) | Bach |
|---|---|---|---|---|
| answer region, strong beats dissonant | 0.34 | 0.30 | 0.21 | 0.20 |
| two-voice beat onsets dissonant (whole piece) | 0.21 | 0.20 | 0.06 | 0.18 |
| chords in planned key | 90.0% | 89.6% | 90.9% | 84.6% |
| forbidden breaks | 0 | 1 | 1 | |

Two-voice consonance now overshoots Bach (0.06 vs 0.18); lower `EXPOSED_W` if it sounds bland.

## v7: the owner's marked spots (2026-11-04)

The listening page's play line is now draggable, with a readout (set, piece/seed, version,
bar, beat, time). The owner marked 7 spots that sounded off-key or dissonant, mostly in minor fugues:

| Spot | Cause | Fix |
|---|---|---|
| O08C bar 29 | a middle-section answer is written in the dominant of the group key, but the key plan said the group key (A minor entry against D minor voices) | local answers' spans planned in the dominant |
| I/2 bar 29 | stretto follower transposed chromatically (B-flat minor against F minor) | `diatonic_shift`: the follower moves by scale steps in the leader's key (also in `stretto_options`) |
| I/2 bar 3, II/22 bar 17 | stray chromatic notes (B natural in G minor, G natural in D-flat): the Krumhansl key pull barely penalises them in minor | `scale_penalty` 3.0: notes outside the key's scale (all three minor forms), the planned chord and the key a quarter ahead |
| I/2 bar 5 | bass ignoring the planned chord under the subject (F under B natural) | `beat_chord_bias` 0.75 on quarter onsets |
| I/2 bar 4.9 | entry on a pickup with a B natural under a planned B-flat chord | `CHROM_CLASH`: chords a semitone from a foreign entry note |

| 40 fugues | v5 | v6x | v7b | Bach |
|---|---|---|---|---|
| beat onsets with a semitone clash | 7.4% | 7.9% | 6.3% | 7.0% |
| free notes outside the planned key | — | 0.9% | 0.4% | 5.9% (vs labelled keys) |
| music chords in planned key | 90.0% | 90.9% | 92.8% | 84.6% |
| in-key, 6 bars after first entry | 87.5% | 88.6% | 92.8% | |
| chord agreement with plan | 0.57 | — | 0.65 | |
| cross relations / 100 bars | 6.2 | 4.8 | 2.6 | 10.2 |
| forbidden breaks | 0 | 1 | 1 | |
