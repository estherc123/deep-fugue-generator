# Handoff: fugue generator (2026-11-05)

Read this first, then `progress/generator.md` (the full history with every
table), then the code paths below. Branch: `corrupted-bach-pairs`.

## 1. The goal and the owner's rules

**Goal.** A Bach-style 4-voice fugue generator that composes a whole fugue
(up to BWV 543 scale, ~100+ bars) from a subject alone:

| | Requirement |
|---|---|
| R1 | Exposition: the subject enters in every voice in turn (tonal or real answer) |
| R2 | Returns: middle entries, a final tonic entry, stretto where possible |
| R3 | Modulation: entries and cadences in related keys, then back home |
| R4 | Sequences: episodes built from sequenced, subject-derived material |
| R5 | Not a copy of any training piece |

**Standing instructions (verbatim where quoted):**

- "do NOT do anything destructive, or post anything online on my behalf, do
  NOT do anything destructive on this PC".
- "when running commands, do not wait for my approvals, just use your best
  judgements".
- Bach's forbidden rules may break at most about once per 5 fugues (≤ 0.2 per
  fugue, as counted by `review_compositions`).
- Keep iterating on quality. **The owner's ear is the final judge, not the
  metrics** (see section 4).
- GPUs: two RTX 5090s, capped at ~90% power (540 W). Never run them at 100%.
- Commits are authored by the owner, with **no Claude co-author trailer**:
  `git -c user.name=yuchencai -c user.email=yuchencai286@gmail.com commit -m "..."`
- There is no bare `python`. Use `.venv/bin/python` with `PYTHONPATH=src`.
- Don't edit code while a composition batch is running: workers import the
  modules lazily.

## 2. Generation architecture

A subject goes through three levels. Each level decides for the one below it.

```
subject ──► form planner ──► harmony planner ──► note model + constrained sampler ──► repair ──► audit
            (form_planner.py) (harmony.py)        (model.py, sample.py, rules_select.py)  (repair.py) (bach_rules.py)
```

Entry point: `src/fugue_data/generator/compose2.py`. Defaults live in
`DEFAULTS` there. The model is `data/models/generator_v7_dpo_b.pt`.

### 2.1 Form planner (`form_planner.py`)

Fugue-form rules, with choice probabilities counted from the corpus. With
~134 fugues there isn't enough data to learn this level.

- **Exposition:** tonal answer via `tonal_answer`. `tonal_head` keeps in the
  tonic only the leading notes the tonal answer actually changes. A
  countersubject (the line against the answer) is restated later as a soft
  copy.
- **Middle groups:** entries in related keys, with voice rotation.
  - Local answers in a middle group sound in the dominant of the group key.
    Their span's `key` is planned in that dominant.
- **Episodes:** sequences (descending fifths, rising 5-6, falling 6-3). The
  next entry's voice rests during the episode.
  - Sequential episodes **pivot to the new key at their first bar** (an open
    problem, see section 5).
- **Stretto:** only when the subject passes the rule checker against itself.
  The follower moves by scale steps in the leader's key (`diatonic_shift`,
  also used in `stretto_options`).
- **Other devices:** inversion, dominant pedal, final entry in an outer voice,
  tonic-pedal coda, deceptive cadence, Neapolitan, vii°7 into the final entry,
  Picardy third, a written-in final chord, and a soprano climax at 0.80–0.90
  of the piece.
- `plan_piece(subject, meter, mode, tonic, nv, seed, piece_id, groups)`
  returns the plan dict. It holds the per-step `key`, `chord`, forced entry
  notes, copies and sequences.
- `--groups 7` gives the long form (counter-exposition, two strettos).
- `subject_from_piece(pid)` extracts a corpus fugue's subject.

### 2.2 Harmony planner (`harmony.py`)

An HMM over chords relative to the local key: root, quality and inversion.

- **Fitting:** fitted on 958 pieces from labels that are about 75% accurate.
  Forward filtering then backward sampling.
- **Constraints:** cadence formulas, sequence root patterns, pedals.
- **Emission penalties:**

  | Constant | Value | Meaning |
  |---|---|---|
  | `EMIT_NCT` | 0.05 | an entry note on a unit start that is not a chord tone |
  | `EMIT_WEAK_NCT` | 0.5 | the same for an entry note inside a unit |
  | `CHROM_CLASH` | −3 | an entry note foreign to the key, a semitone from a chord tone |
  | `AUG2_CLASH` | −3 | the leading tone over a chord holding the 6th degree |

- **`VOCAB` key prior** (`OUT_OF_KEY` −6): diatonic chords, secondary
  dominants and leading-tone chords. Neapolitan, Picardy and raised-6th chords
  appear only when a constraint asks for them. In minor the dominant is V, not
  v (`MINOR_V` −1.5).

### 2.3 Note model (`model.py`, `encode.py`, `train.py`)

- A decoder-only transformer over a 16th-note grid (`STEPS_PER_Q` = 4), with
  voices interleaved per step.
- Per-position plan features describe the token being predicted: voice, bar
  position, role (subject / answer / free), the entry's notes, local key,
  next key, sequence role, chord, next chord, bass, and per-voice texture.
- **v7:** d512, trained on 1.42M tokens (WTC, the Baroque expansion, chorales,
  Corelli, Vivaldi). Held-out WTC free-note perplexity 1.76.
- **`_dpo_b`:** DPO with Bach's continuation preferred over the model's own.
  It gave a small gain against harmonic drift.

### 2.4 Constrained sampler (`sample.py: generate`, `ParallelMask`)

**Forced notes.** Entry notes are written in as hard constraints, and the
final chord as a pitch-class constraint.

**Logit adjustments** (defaults in `compose2.DEFAULTS`):

| Setting | Value | Effect |
|---|---|---|
| `temperature` | 0.85 | |
| `key_bias` | 1.5 | Krumhansl key pull |
| `chord_bias` | 0.75 | bonus for notes of the planned chord |
| `beat_chord_bias` | 0.75 | extra chord pull on quarter-note onsets |
| `scale_penalty` | 3.0 | `out_of_key_table`: penalises notes outside the extended scale (minor = natural + harmonic + melodic), the planned chord, and the key one quarter ahead |
| `repeat_penalty` | 2.0 | |
| `copy_bias` | 8.0 | soft copies of the countersubject and sequence units |

**Masks.** Every forbidden rule is masked: parallel 5ths/8ves in any motion,
hidden outer 5ths/8ves, augmented 2nd, doubled leading tone, leaps of a major
7th or over an octave, bass crossing, range. Free voices may not cross their
neighbours (`no_crossing`). Leaps and parallels into written-in notes are
blocked by look-ahead. Beat dissonances must resolve by step.

**Melodic rules.** These are soft penalties. Among them is the
neighbouring-voice unison penalty `UNISON_PEN`:
- **1.5 is the v7 value**, which the owner prefers. It is back in the code now.
- 3.5 was v7c (see section 4).

**Bar-by-bar selection** (`bar_select` = 8). Each bar is sampled K = 8 times,
and the candidate with the lowest cost is kept. The cost (`bar_score_v2`) is
the weighted rule cost, plus a density term, plus `EXPOSED_W` 1.5 × the
dissonant beat onsets where exactly two voices sound.

### 2.5 Repair and audit

- **`repair.py`:** tries nearby pitches for free notes involved in any
  forbidden break that remains.
- **`bach_rules.py`:** the rule checker, calibrated on 134 fugues
  (`research/bach_rule_rates.json`). `FORBIDDEN` holds the hard rules.
- **`review_compositions.py DIR`:** forbidden breaks, Bach-rate rules,
  cadence hits, chord agreement, copy adherence, density, climax. It writes
  `DIR/review.json`.

### 2.6 Output

Each composition goes to `data/review/compositions_*/<pid>_s<seed>/`:
- `generated_{1,2}.npy`, `.wav`, `.mid`;
- `forced.npy`, `plan.json`, `plan_key.npy`, `plan.txt`;
- `meta.json`, which holds every sampler setting.

## 3. Tools and workflow

**Listening page:** http://127.0.0.1:8777/generator.html
- Served by `python3 -m http.server 8777` from `data/review/corruption_site`;
  already running.
- Rebuild it with `samples_site` (below).
- The play line is draggable. The readout shows set · piece_s<seed> · version
  · bar · beat · time.
- **The owner gives feedback as screenshots of this readout.** Paths look like
  `~/Pictures/Screenshots/Screenshot from ....png`; read them with Read.
- Current sets:
  - "composed v7 (preferred)" = `compositions_v7b`
  - "composed v7c (stronger unison penalty)"
  - "composed v6"
  - "original subjects"
  - "composed long (107 bars)"

```bash
PYTHONPATH=src .venv/bin/python -m fugue_data.generator.samples_site \
  "composed v7 (preferred)=data/review/compositions_v7b" \
  "composed v7c (stronger unison penalty)=data/review/compositions_v7c" \
  "composed v6=data/review/compositions_v6x" \
  "original subjects=data/review/compositions_original" \
  "composed long (107 bars)=data/review/compositions_long"
```

**Standard test set:** 10 held-out subjects × seeds 1, 2 × 2 samples = 40
fugues. Generation takes about 40 minutes on both GPUs:

```bash
bash tools/metrics/compose_set.sh data/review/compositions_vX     # run in background
bash tools/metrics/measure.sh data/review/compositions_vX         # all metrics
```

**Metric scripts** (`tools/metrics/`, copied from the last session's
scratchpad). Each takes a compositions directory; `bach` as the argument
gives the Bach reference where supported.

| Script | Measures | Bach | v7 (preferred) |
|---|---|---|---|
| `review_compositions` | forbidden breaks per fugue | target ≤ 0.2 | 0.03 |
| `inkey.py` | music chords in the planned key | 84.6% | 92.8% |
| `semi.py` | beat onsets with a semitone clash | 7.0% | 6.3% |
| `exposed.py` | dissonant two-voice beat onsets | 0.18 | ~0.06 (v6x) |
| `ook.py` | free notes outside the planned key | 5.9% | 0.4% |
| `texture_stats.py` | neighbouring-voice unisons (also crossings, repeats) | 1.55% | 3.1% |
| — | cross relations per 100 bars | 10.2 | 2.6 |
| `spot.py DIR GEN BAR0 BAR1` | per-beat dump: planned key and chord, heard chord, notes, entry (`*`) and out-of-key (`!`) flags, clashes | — | — |

`diss2.py` measures dissonance in the answer region.

`spot.py` is the main tool for diagnosing a spot the owner marked. Map the
readout's set to its directory, `piece_sN` to the folder, and `vK` to
`generated_K`.

**Blind A/B tool** (`ab.py`): `ab serve --name old_vs_rules --port 8780` is
running and waiting for the owner. Score it with
`PYTHONPATH=src .venv/bin/python -m fugue_data.generator.ab score --name old_vs_rules`.

**The loop that has worked:**
1. The owner marks spots.
2. `spot.py` on each spot.
3. Find the cause in the plan or the sampler, and fix it at that level.
4. Regenerate the 40 fugues and measure.
5. Put the new set on the page **next to the previous one** so the owner can
   compare by ear.
6. Write up the results in `progress/generator.md` and commit.

## 4. What went wrong most recently: v7c, and the lesson

- **The complaint.** II/22 seed 1 v1, bars 29–32 (1:32–1:43), "sounds really
  weird". Three causes:
  1. The sequential episode pivots to VII (A♭ major) at its first bar.
  2. The episode is crowded and zig-zags.
  3. At bar 32 the alto's entry (A♭3–D♭4) sits in the tenor's register. The
     tenor isn't allowed to cross, so it shadowed the alto in unison (17.5%
     of neighbouring pairs in bars 32–33) while two voices rested.
- **v7c's fix.** `UNISON_PEN` 1.5 → 3.5, applied globally. Unisons fell from
  3.1% to 2.0% (Bach 1.55%) and the other metrics held steady.
- **The owner's verdict.** "i actually like v7 better than v7c... it sounds
  much more pleasant and musical." The penalty is back at 1.5 (uncommitted
  until this handoff's commit), so the code now reproduces v7.
- **The lesson.** A global penalty that moves a metric toward Bach's value can
  make the music worse to the ear. Most likely it forces awkward leaps and
  wide spacing to avoid a shared pitch everywhere, not just where the shadowing
  happens. From here on:
  - Prefer **local, cause-specific fixes** over global weight changes.
  - Before you change a global weight, compare the two sets' melodic
    statistics, not only the target metric: leap sizes, step share, spacing,
    note density.
  - Keep the preferred set on the page as the baseline, and ask the owner to
    compare.
  - **A good first task:** measure v7b against v7c (leap histogram, share of
    stepwise motion, average spacing, notes per bar, rests) to learn what the
    owner's ear is responding to.
- **Possible overshoot.** v6x's two-voice consonance (0.06 vs Bach 0.18) and
  v7's very high in-key rate (92.8% vs 84.6%) overshoot Bach. So far that has
  sounded pleasant to the owner, but if they ever call the music bland, these
  are the knobs: `EXPOSED_W`, `scale_penalty`.

## 5. Plan: what to work on next, in order

1. **Fix the bar-32 shadowing locally, without v7c's global penalty.**
   Options, best first:
   - (a) In the form planner, place middle entries in a register that fits
     their voice, so they don't sit inside a neighbour's range.
   - (b) When an entry sits in a neighbour's register, let that neighbour
     rest, or let it cross briefly.
   - (c) Apply the stronger unison penalty only against forced (entry)
     notes, keeping 1.5 elsewhere.

   Check II/22 s1 bars 29–33 with `spot.py`, and make sure the overall
   character stays v7-like.
2. **The abrupt pivot in sequential episodes.** Episodes jump to the new key
   (VII in II/22) at their first bar. Try pivoting mid-episode, or let the
   sequence carry the modulation, as Bach does: the chain starts in the old
   key and cadences in the new one. The code is in `form_planner.py`, where
   episodes and sequence chains are realised (`form_planner.py:421`, the comment on pivots
   in `realise_plan`).
3. **Texture density and inner-voice rests.**
   - Generated music averages about 22.5 notes per bar (II/22) against Bach's
     17.3. Inner voices rest 9–14% of the time against Bach's 20–32%.
   - The long form is worse: 29 vs 19 notes per bar.
   - Levers: the density target in bar selection, and texture features in the
     plan (have voices drop out in episodes).
   - This is expected to help the crowded episodes and long-form drift.
4. **Regenerate the original-subject and long (107-bar) sets** with the current
   code (they are from v4/v5) and put them on the page.
5. **The owner's preferences as training signal.** Score the `old_vs_rules`
   A/B once the owner has done it. Collect v7 vs v7c and future A/B choices,
   then DPO on the owner's preferences.
   - The critic (critic_v2) matched the owner only 9/16 times, so don't train
     on it.
   - GRPO only after a reward model survives a refresh.
6. **Long-form drift:** mid-piece surprise is about 0.8, against Bach's about
   0.65.

Ask the owner to listen after each step. Report results in plain language,
and say which set on the page to compare.

## 6. Pitfalls

- The plan isn't fully saved. To re-create it, call `plan_piece` with the
  saved `seed`, `groups` and `piece_id` (`meta["name"]`) after
  `subject_from_piece`. `plan.json` has `key` and `chord` but not `form`.
- Entries have no `length` key. Use the difference between start steps.
- Commands that run past 600 s must run in the background. Use until-loops or
  Monitor to wait, not sleep.
- Don't open headless Firefox: it clashes with the owner's running Firefox.
  Check page JS by parsing it (esprima) instead.
- Chord labels in the training data are only about 75% accurate. Don't trust
  the raw HMM table for chords outside the key; that is why `VOCAB` exists.
- `data/` is git-ignored. Composition sets live only on disk, so never delete
  them.
