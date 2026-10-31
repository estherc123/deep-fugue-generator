# Fugue Data Collection Pipeline Specification

**Document status:** Implementation handoff  
**Project type:** Personal, non-commercial research  
**Primary goal:** Build a reproducible, provenance-tracked, training-ready symbolic dataset for a hybrid hard-constraint / learned-prior / reinforcement-learning system that composes fugues  
**Target style:** High-Baroque counterpoint, with J. S. Bach as the principal style reference  
**Initial long-term target:** Four-voice fugues at the scale of BWV 543  

---

## 1. Agent Mandate

Implement an automated data pipeline that:

1. Downloads the specified public research repositories without requiring browser interaction.
2. Pins every source to an immutable revision and records provenance.
3. Extracts the 48 fugues from both books of the Well-Tempered Clavier.
4. Aligns the 24 WTC I fugues with the available Algomus structural annotations.
5. Imports the 139 unique BCFB Bach chorales.
6. Converts all accepted scores into one loss-minimizing canonical score representation.
7. Validates notation, timing, voices, metadata, annotations, provenance, and split isolation.
8. Proposes 2–8-bar real Bach excerpts for human approval.
9. Provides interfaces for generating:
   - 10,000 hard-valid synthetic fragments;
   - 10,000 controlled single-violation fragments;
   - 2,000–5,000 preference-pair records.
10. Splits data by complete work before creating windows or transpositions.
11. Exports separate training datasets for style modeling, feasibility learning, constraint classification, structural supervision, and preference learning.
12. Produces an auditable report proving that all acceptance criteria were met.

The agent must not represent automatically generated or automatically selected material as human verified.

---

## 2. Scope

### 2.1 Required real-music corpus

| Corpus | Required content | Unique movements |
| --- | --- | ---: |
| Humdrum WTC I | 24 Bach fugues | 24 |
| Humdrum WTC II | 24 Bach fugues | 24 |
| BCFB v2 | Bach chorales | 139 |
| Algomus | Structural annotations for WTC I fugues | 0 additional movements |
| **Total** | Unique real Bach movements | **187** |

The WTC repository also contains 48 preludes. Preserve their raw files and make their inclusion configurable, but exclude them from the default fugue dataset.

BCFB contains 143 encoded score variants representing 139 unique chorales. Preserve variants but group them under the same canonical work identifier.

### 2.2 Required derived and generated records

| Dataset | Target |
| --- | ---: |
| Human-approved Bach excerpts | 50–100 |
| Hard-valid synthetic fragments | 10,000 |
| Controlled single-violation fragments | 10,000 |
| Preference-pair records | 2,000–5,000 |

### 2.3 Non-goals for the first implementation

- Full scraping of every known Bach work.
- Audio collection or audio-model training.
- Optical music recognition from PDFs or manuscript scans.
- Automatic replacement of expert musical judgment.
- A complete BWV 543 generator.
- Treating Fux species-counterpoint rules as a complete description of Bach’s practice.
- Flattening all music into MIDI as the sole archival representation.

---

## 3. Authoritative Sources

### 3.1 Humdrum Well-Tempered Clavier

- Repository: `https://github.com/humdrum-tools/bach-wtc.git`
- Expected content: both books of the WTC, preludes and fugues, in Humdrum `**kern`.
- Required default selection:
  - `wtc1f01` through `wtc1f24`;
  - `wtc2f01` through `wtc2f24`.
- Optional selection:
  - WTC preludes for supplementary keyboard-style learning.

### 3.2 Algomus fugue annotations

- Repository: `https://gitlab.com/algomus.fr/algomus-data.git`
- Reference page: `https://www.algomus.fr/fugues/`
- Expected content for Bach WTC I:
  - subject entries;
  - countersubject entries;
  - cadences;
  - pedal points;
  - variant or alternative analytical boundaries where supplied.
- Expected coverage: 24 WTC I fugues.

### 3.3 Bach Chorales Figured Bass dataset

- Repository: `https://github.com/juyaolongpaul/Bach_chorale_FB.git`
- Required version: BCFB v2 or later compatible release.
- Expected content:
  - 139 unique Bach chorales;
  - 143 encoded variants;
  - MusicXML;
  - Humdrum `**kern`;
  - MEI;
  - figured-bass information;
  - reference metadata.
- Declared repository license: CC BY 4.0.
- Default primary source format: MusicXML.
- Secondary formats are retained for cross-validation, not counted as separate works.

### 3.4 Reference theory sources

These inform rule metadata and documentation but are not automatically converted into hard rules:

- Open Music Theory, species counterpoint.
- Open Music Theory, High Baroque Fugal Exposition.
- Music Theory for the 21st-Century Classroom, counterpoint and fugue analysis.
- Fux, *Gradus ad Parnassum*, as a pedagogical curriculum source.
- FuxCP4 formal constraint implementation.
- “Expressing Musical Ideas with Constraint Programming Using a Model of Tonal Harmony,” IJCAI 2024.

Every implemented rule must cite at least one source or be explicitly marked `project_policy` or `empirical_rule`.

---

## 4. Reproducibility Requirements

### 4.1 Source pinning

For every Git source:

1. Clone or fetch the repository.
2. Resolve the selected revision to a full commit SHA.
3. Check out the detached commit.
4. Record:
   - repository URL;
   - full commit SHA;
   - retrieval timestamp in UTC;
   - repository license when known;
   - source directory;
   - source file checksum.

Do not rely on a moving `main` or `master` branch after acquisition.

### 4.2 Environment pinning

The implementation must include:

- Python version specification.
- Lockfile containing exact dependency versions.
- Optional container definition.
- OR-Tools version.
- `music21` version.
- Schema version.
- Rule-profile version.
- Tokenizer version.

### 4.3 Determinism

All stochastic stages must accept explicit seeds. Store the seed with every generated artifact.

Given:

- identical source commits;
- identical configuration;
- identical dependency lock;
- identical seed;

the pipeline must reproduce identical normalized event hashes, dataset membership, and splits.

---

## 5. Recommended Technology Stack

### 5.1 Required or equivalent

- Python 3.11 or a deliberately pinned compatible release.
- Snakemake for orchestration.
- `music21` for symbolic parsing and analysis.
- `lxml` for MusicXML operations.
- `pandas` and `pyarrow` for tabular datasets.
- `orjson` or equivalent deterministic JSON serialization.
- `jsonschema` or Pydantic for schema validation.
- OR-Tools CP-SAT for constraint generation.
- `zstandard` for compressed JSONL output.
- `pytest` for unit and integration tests.
- A notation renderer such as Verovio or MuseScore CLI for visual QA.
- Streamlit or an equivalent lightweight local review application.

### 5.2 Repository layout

```text
fugue-data/
  README.md
  pyproject.toml
  requirements.lock
  Snakefile
  config/
    sources.yaml
    pipeline.yaml
    rules/
      species_strict.yaml
      high_baroque_conservative.yaml
      bach_empirical.yaml
    splits.yaml
    tokenizer.yaml
  schemas/
    source_manifest.schema.json
    canonical_score.schema.json
    annotation.schema.json
    rule.schema.json
    review.schema.json
    preference_pair.schema.json
  data/
    raw/
    canonical/
    annotations/
    proposed/
    reviewed/
    generated/
      valid/
      invalid/
      preference_candidates/
    exports/
  manifests/
    sources.parquet
    pieces.parquet
    derivatives.parquet
    validation.parquet
    reviews.parquet
  src/fugue_data/
    acquire/
    normalize/
    annotations/
    constraints/
    generation/
    review/
    splitting/
    tokenization/
    validation/
    export/
  tests/
    unit/
    integration/
    fixtures/
  reports/
```

Raw files must be immutable after download. Corrections belong in canonical or correction-overlay layers.

---

## 6. Source Manifest

Every acquired source file must produce one manifest row.

### 6.1 Required fields

```yaml
source_file_id: string
source_name: string
repository_url: string
repository_commit: full_sha
retrieved_at_utc: ISO_8601
relative_source_path: string
sha256: hex_string
source_format: kern | musicxml | mei | csv | json | other
composer_claimed: string | null
work_id_claimed: string | null
movement_claimed: string | null
license_name: string | null
license_url: string | null
license_scope: composition | encoding | annotation | repository | unknown
commercial_use_known: boolean | null
attribution_required: boolean | null
quarantine_reason: string | null
ingest_status: accepted | quarantined | rejected
```

### 6.2 Quarantine behavior

Quarantined files:

- remain stored and checksummed;
- are not normalized into the active training corpus;
- do not enter any training export;
- appear in the final report with a reason.

---

## 7. Canonical Piece and Work Identity

### 7.1 Identity hierarchy

Use:

```text
composer → canonical work → movement → source edition → encoded variant
```

Examples:

```text
bach / BWV846 / fugue / humdrum-wtc / default
bach / BWV10.07 / chorale / BCFB / version-a
bach / BWV10.07 / chorale / BCFB / version-b
```

### 7.2 Required identifiers

```yaml
composer_id: bach_js
work_group_id: canonical BWV-level identity
movement_id: stable movement identity
piece_id: stable source-independent movement identity
variant_id: source/edition/variant identity
source_file_id: reference into source manifest
```

All alternate encodings of the same musical content must share `piece_id` or `work_group_id` as appropriate.

---

## 8. Canonical Score Schema

The canonical representation is the archival normalized format. Model-specific tokens are derived from it.

### 8.1 Timing

- Represent onset and duration as exact rational strings, such as `"3/8"`.
- Do not store canonical musical time as floating point.
- Use quarter-note length as the base unit.
- Preserve:
  - notated measure number;
  - unfolded measure index;
  - beat position;
  - absolute onset.

### 8.2 Pitch

Store both sounding pitch and spelling:

```yaml
midi: integer
step: A-G
alter: rational or integer semitone alteration
octave: integer
pitch_class: 0-11
diatonic_step: integer
```

### 8.3 Notes

```yaml
note_id: stable string
voice_id: stable string
onset_q: rational_string
duration_q: rational_string
measure_number: string
measure_index_unfolded: integer
offset_in_measure_q: rational_string
midi: integer
step: string
alter: rational_string
octave: integer
tie: null | start | continue | stop
grace: boolean
articulations: array
ornaments: array
source_refs: array
```

### 8.4 Voices

- Preserve actual independent voices.
- Do not infer four voices for a three-voice fugue.
- Do not treat staves or hands as equivalent to contrapuntal voices.
- Allow voices to enter late and rest for extended periods.
- Store staff assignment separately from voice identity.

### 8.5 Global maps

```yaml
meter_map:
  - onset_q
  - numerator
  - denominator

key_signature_map:
  - onset_q
  - fifths
  - mode_if_known

tempo_map:
  - onset_q
  - bpm_if_known
```

### 8.6 Example top-level record

```json
{
  "schema_version": "1.0.0",
  "piece_id": "bach_wtc1_fugue_01",
  "work_group_id": "BWV846",
  "composer_id": "bach_js",
  "movement_type": "fugue",
  "source_file_ids": ["..."],
  "voice_count": 4,
  "voices": [],
  "meter_map": [],
  "key_signature_map": [],
  "annotations": [],
  "normalization": {
    "software_version": "...",
    "config_hash": "...",
    "event_hash": "..."
  }
}
```

---

## 9. Acquisition Stage

### 9.1 Required commands

Expose:

```bash
fugue-data acquire --source wtc
fugue-data acquire --source algomus
fugue-data acquire --source bcfb
fugue-data acquire --all
```

### 9.2 Acquisition acceptance criteria

- Repository is pinned to a full SHA.
- All raw files are checksummed.
- Expected WTC I fugue identifiers: 24.
- Expected WTC II fugue identifiers: 24.
- Expected BCFB unique chorales: 139.
- Expected BCFB encoded variants: 143, subject to documented upstream changes.
- Algomus WTC I coverage: 24 fugues.
- Any mismatch stops the pipeline unless explicitly accepted through a versioned override.

---

## 10. Normalization Stage

### 10.1 Parsing

For each score:

1. Parse the native format.
2. Identify parts, staves, voices, measures, meter, key signatures, notes, rests, ties, tuplets, grace notes, and ornaments.
3. Expand repeats into a performance-order view while retaining original measure identities.
4. Build independent voice streams.
5. Generate canonical rational-time events.
6. Validate timing and pitch.
7. Serialize deterministically.
8. Compute:
   - exact event hash;
   - transposition-normalized melodic hash;
   - rhythm hash;
   - per-voice hashes.

### 10.2 Cross-format validation for BCFB

Where MusicXML, Kern, and MEI versions exist:

- compare note counts;
- compare onset/duration multisets;
- compare pitch sequences by voice where identities align;
- report discrepancies;
- prefer MusicXML as the primary canonical source unless a documented issue makes another encoding superior.

### 10.3 Round-trip validation

Canonical score → MusicXML → canonical score must preserve the canonical event hash for supported notation.

If the round trip is intentionally lossy for ornaments or layout, define separate:

- `core_event_hash`;
- `extended_notation_hash`.

### 10.4 Normalization acceptance criteria

- 48 WTC fugues normalized.
- 139 BCFB unique chorales normalized.
- Alternate BCFB variants retained and grouped.
- No negative durations.
- No notes outside score bounds.
- No unexplained monophonic overlaps within a voice.
- All measure-duration discrepancies are resolved or quarantined.
- All parser warnings are recorded.

---

## 11. Annotation Alignment

### 11.1 Algomus mapping

Map Algomus labels to canonical WTC I pieces by:

1. WTC book number.
2. Fugue index.
3. BWV identifier when available.
4. Measure number.
5. Position within measure.

Never align only by filename.

### 11.2 Canonical annotation schema

```yaml
annotation_id: string
piece_id: string
type: subject | answer | countersubject | cadence | pedal | episode | other
voice_id: string | null
onset_q: rational_string
duration_q: rational_string | null
measure_number_source: string | null
position_source: string | null
variant: string | null
confidence: number | null
analytical_alternative_group: string | null
source_name: algomus
source_reference: string
```

### 11.3 Analytical ambiguity

Do not discard disagreements in subject endings. Store alternatives as separate annotations linked by `analytical_alternative_group`.

### 11.4 Annotation acceptance criteria

- 24 WTC I fugues have annotation records.
- Every annotation onset lies within the score.
- Every voice-specific annotation resolves to an existing voice.
- Every duration is positive.
- Ambiguous boundaries are preserved.
- A human-readable alignment report is produced per fugue.

---

## 12. Deduplication

Deduplicate at several levels:

1. Exact file checksum.
2. Exact canonical event hash.
3. Transposition-normalized interval hash.
4. Per-voice melody hash.
5. Rhythm-only hash.
6. Work and movement identifiers.

Do not automatically delete near duplicates. Group them and select a preferred canonical representative while retaining provenance.

No duplicate group may cross training, validation, or test boundaries.

---

## 13. Dataset Splitting

### 13.1 Required order

Split complete works before:

- window extraction;
- transposition;
- augmentation;
- synthetic derivation from real subjects;
- preference-candidate generation.

Every derivative inherits its parent’s split.

### 13.2 WTC default split

Target:

| Split | Fugue count |
| --- | ---: |
| Train | 36 |
| Validation | 6 |
| Test | 6 |

Stratify approximately by:

- WTC book;
- major/minor mode;
- voice count;
- meter;
- piece length.

### 13.3 BCFB default split

Target:

| Split | Unique chorales |
| --- | ---: |
| Train | 111 |
| Validation | 14 |
| Test | 14 |

Group alternate versions together.

### 13.4 Split manifest

```yaml
piece_id: string
work_group_id: string
split: train | validation | test
split_seed: integer
stratification_fields: object
assignment_hash: string
```

### 13.5 Leakage acceptance criteria

- No `work_group_id` appears in multiple splits.
- No exact or transposition-normalized duplicate crosses splits.
- No excerpt crosses its parent split.
- No transposition crosses its parent split.
- No synthetic sample derived from held-out subjects appears in training.

---

## 14. Real Bach Excerpt Proposal

### 14.1 Purpose

Create 2–8-bar candidates for:

- style modeling;
- structural supervision;
- qualitative evaluation;
- preference comparisons;
- rule-engine testing.

### 14.2 Candidate triggers

Propose windows around:

- complete subject/answer entries;
- countersubject combinations;
- exposition boundaries;
- cadence arrivals;
- pedal points;
- short episodes;
- sequences;
- stretto;
- clean changes in texture;
- complete dissonance-resolution events.

### 14.3 Automatic rejection

Reject windows that:

- begin or end inside a tied note unless context is preserved;
- cut through an annotated subject entry;
- end on an unresolved suspension;
- contain parser errors;
- contain missing measures;
- cannot preserve voice identity.

### 14.4 Review record

```yaml
excerpt_id: string
piece_id: string
start_onset_q: rational_string
end_onset_q: rational_string
start_measure: string
end_measure: string
proposal_reasons: array
render_path: string
midi_preview_path: string
review_status: pending | approved | rejected | correction_required
review_tags: array
review_comment: string | null
reviewer: string | null
reviewed_at_utc: string | null
```

### 14.5 Human review UI

Required actions:

- play;
- view notation;
- approve;
- reject;
- request correction;
- assign tags;
- add a short comment.

### 14.6 Acceptance criteria

- At least 200 candidates proposed.
- 50–100 manually approved.
- No `pending` excerpt enters an export marked human-reviewed.
- Review provenance is retained.

---

## 15. Hard-Constraint Framework

### 15.1 Rule categories

Every rule belongs to one category:

1. `notational_invariant`;
2. `hard_constraint`;
3. `conditional_constraint`;
4. `soft_preference`;
5. `structural_target`.

### 15.2 Rule profiles

Support:

- `species_strict`;
- `high_baroque_conservative`;
- `bach_empirical`.

The same rule may have different severity by profile.

### 15.3 Rule schema

```yaml
rule_id: string
name: string
version: string
category: string
scope: note | voice | voice_pair | chord | transition | phrase | section | piece
description: string
applies_when: object
violation_when: object
exceptions: array
profile_severity:
  species_strict: off | diagnostic | soft | hard
  high_baroque_conservative: off | diagnostic | soft | hard
  bach_empirical: off | diagnostic | soft | hard
sources: array
implementation: string
test_fixtures:
  valid: array
  invalid: array
```

### 15.4 Violation schema

```yaml
rule_id: string
severity: diagnostic | soft | hard
piece_id: string
voice_ids: array
note_ids: array
onset_q: rational_string
measure_number: string | null
evidence: object
message: string
```

### 15.5 Initial rule set

Implement 15–25 reliable rules first, covering:

- valid timing;
- no negative or zero structural duration;
- instrument/voice ranges under a configured profile;
- monophonic consistency within each voice;
- scheduled subject-entry identity;
- permitted real/tonal answer transformation;
- perfect-parallel detection;
- basic leap constraints;
- tendency-tone handling;
- cadence target completion;
- basic suspension/passing/neighbor classification;
- voice participation;
- maximum configured dead-air duration;
- section-length limits;
- no search state from which required structural targets are impossible.

Do not make voice crossing, direct fifths, every large leap, every dissonance, chord completeness, or repeated notes universally hard without contextual conditions.

### 15.6 Empirical validation

For each proposed hard or conditional rule:

1. Run it on held-out Bach.
2. Collect all detections.
3. Review a representative sample.
4. Distinguish:
   - implementation bug;
   - missing context;
   - genuine historical exception;
   - source-encoding issue.
5. Revise the rule.
6. Record false-positive and false-negative estimates.

No rule becomes hard merely because a textbook phrases it categorically.

---

## 16. Hard-Valid Synthetic Generation

### 16.1 Required target

Generate exactly 10,000 accepted hard-valid fragments for the initial release.

### 16.2 Generation request schema

```yaml
request_id: string
seed: integer
rule_profile: string
voice_count: 2 | 3 | 4
meter: object
key: object
length_bars: integer
subject: object
entry_plan: array
cadence_target: object | null
allowed_rhythmic_values: array
range_profile: string
texture_profile: string
```

### 16.3 Required coverage

Balance across:

- two, three, and four voices;
- major and minor;
- several keys;
- 4/4, 3/4, and 6/8 where supported;
- subject lengths;
- entry orders;
- subject in upper, middle, and bass voices;
- rhythmic density;
- cadence targets;
- presence/absence of a reusable countersubject;
- permitted dissonance types;
- different registers.

### 16.4 Generator and validator separation

The generator proposes a candidate. A logically separate validation call must confirm it.

Required acceptance predicate:

```text
hard_violation_count == 0
and structure_complete == true
and canonical_schema_valid == true
and duplicate_status == unique
```

### 16.5 Generated record metadata

```yaml
sample_id: string
request_id: string
seed: integer
ruleset_hash: string
generator_version: string
validator_version: string
solver_status: string
solve_time_ms: integer
hard_valid: true
soft_scores: object
source_type: synthetic_valid
parent_subject_id: string
split: string
```

### 16.6 Acceptance criteria

- 10,000 accepted unique fragments.
- 100% independently validated hard-valid.
- Coverage report produced.
- No split leakage through parent subjects.
- Generation seeds and rule hashes present.
- Re-running a request reproduces the same accepted output under the pinned environment.

---

## 17. Controlled Invalid Generation

### 17.1 Required target

Generate exactly 10,000 controlled single-violation fragments.

### 17.2 Method

1. Select a hard-valid source fragment.
2. Apply one rule-aware mutation.
3. Validate the result.
4. Accept only if:
   - the source had zero hard violations;
   - the result has exactly one intended hard violation;
   - no unintended hard violations were introduced.

### 17.3 Initial mutators

- create parallel fifth;
- create parallel octave;
- break leading-tone resolution;
- create unresolved suspension;
- alter required subject note;
- exceed configured voice range;
- create illegal voice overlap;
- create prohibited melodic leap;
- break cadence target;
- create invalid duration or tie in a notation-only test subset.

### 17.4 Balance

Set per-rule quotas. No single rule should dominate more than 15% of the initial invalid set unless documented.

### 17.5 Invalid record

```yaml
sample_id: string
parent_valid_sample_id: string
mutation_id: string
target_rule_id: string
mutation_seed: integer
pre_validation_hash: string
post_validation_hash: string
violations: array
source_type: synthetic_controlled_invalid
split: string
```

### 17.6 Acceptance criteria

- 10,000 accepted unique invalid fragments.
- Each has exactly one intended hard violation.
- Each points to a valid parent.
- Per-rule distribution report produced.
- Positives and negatives derived from the same parent remain in the same split.

---

## 18. Preference-Pair Candidate Generation

### 18.1 Required target

Generate candidate material sufficient for 2,000–5,000 labelled preference pairs.

### 18.2 Prompt identity

Candidates in a pair must share:

- subject;
- prefix;
- formal target;
- length budget;
- rule profile;
- instrumentation/range profile;
- generation conditions, except stochastic seed.

### 18.3 Candidate generation

For each prompt:

1. Generate 4–8 hard-valid alternatives.
2. Reject duplicates.
3. Compute separate diagnostic scores:
   - style-prior likelihood;
   - structural completion;
   - voice participation;
   - rhythmic diversity;
   - melodic diversity;
   - repetition;
   - novelty;
   - soft-rule vector.
4. Create pairs preferentially where:
   - evaluators disagree;
   - scores are close but musical character differs;
   - one candidate exposes a suspected reward-model weakness;
   - uncertainty is high.

### 18.4 Pair schema

```yaml
pair_id: string
prompt_id: string
candidate_a_id: string
candidate_b_id: string
shared_conditions_hash: string
diagnostics_a: object
diagnostics_b: object
label: A | B | tie | both_poor | null
review_status: pending | reviewed | rejected
reviewer: string | null
review_comment: string | null
reviewed_at_utc: string | null
split: train | validation
```

### 18.5 Human review

The review interface must:

- blind candidate source where possible;
- randomize left/right presentation;
- support playback and notation;
- store ties and `both_poor`;
- avoid forcing a winner;
- periodically repeat some pairs to estimate reviewer consistency.

### 18.6 Acceptance criteria

- Candidate generation is automated.
- At least 500 pairs can be labelled for the first iteration.
- Target final labelled pool: 2,000–5,000.
- No unreviewed pair enters a human-preference export.
- Duplicate or near-identical pairs are excluded.

---

## 19. Tokenization and Model Views

### 19.1 Principle

Canonical score data and model tokens are different layers. Do not discard canonical notation after tokenization.

### 19.2 Required initial model view

Provide a grid-based view suitable for debugging:

- sixteenth-note resolution where exactly representable;
- per-voice `NOTE_ON`, `HOLD`, and `REST`;
- bar and beat tokens;
- key and meter tokens;
- voice identifier;
- optional structural annotation tokens;
- explicit end-of-section token.

Do not silently quantize tuplets or finer rhythms. Either:

- represent them exactly in an event-based view;
- mark the grid view as unsupported;
- use a configured finer denominator.

### 19.3 Required event-based view

Also provide an event-based representation with:

- voice;
- pitch spelling and/or relative pitch;
- onset delta;
- duration;
- bar/beat;
- tie;
- section metadata.

### 19.4 Tokenizer versioning

Every exported sequence must record:

- tokenizer name;
- tokenizer version;
- vocabulary hash;
- quantization policy;
- source canonical event hash.

---

## 20. Training Exports

Produce separate datasets.

### 20.1 Style prior

```text
exports/style/
  train.jsonl.zst
  validation.jsonl.zst
  test.jsonl.zst
```

Contains:

- real Bach movements and approved excerpts;
- optionally a configurable low-weight subset of valid synthetic fragments;
- source-type labels.

### 20.2 Feasibility / constrained policy

```text
exports/feasibility/
  valid_train.jsonl.zst
  valid_validation.jsonl.zst
```

Contains hard-valid real and synthetic material with rule-profile metadata.

### 20.3 Constraint critic

```text
exports/constraint_critic/
  positives.parquet
  controlled_negatives.parquet
```

Contains validity labels, violations, evidence, and matched parent relationships.

### 20.4 Structural supervision

```text
exports/structure/
  wtc1_annotations.parquet
  section_windows.jsonl.zst
```

Contains subject/answer/countersubject/cadence/pedal labels and canonical positions.

### 20.5 Preferences

```text
exports/preferences/
  train_pairs.jsonl
  validation_pairs.jsonl
```

Contains only reviewed pairs.

### 20.6 Metadata

```text
exports/metadata/
  sources.parquet
  pieces.parquet
  derivatives.parquet
  splits.parquet
  ruleset.json
  vocabulary.json
  build_report.json
```

---

## 21. Validation Gates

The pipeline must fail on:

- missing source provenance;
- unexpected source revision;
- checksum mismatch;
- schema failure;
- invalid or zero structural duration;
- annotation outside piece bounds;
- unresolved annotation voice;
- duplicate group crossing splits;
- derivative crossing parent split;
- synthetic positive with a hard violation;
- controlled negative with zero or multiple unintended hard violations;
- preference pair with mismatched prompt conditions;
- `pending` review included as reviewed;
- unknown tokenizer version;
- unknown rule-profile hash;
- canonical event hash changing unexpectedly.

### 21.1 Visual QA

Render and inspect:

- at least 20 random WTC fugues or excerpts;
- at least 20 random BCFB chorales;
- every anomalous normalization;
- at least 50 synthetic valid fragments across coverage categories;
- at least 50 controlled negatives with violation highlighting.

Visual QA results must be recorded, not merely performed informally.

---

## 22. Workflow Orchestration

### 22.1 Required targets

```bash
snakemake acquire --cores 4
snakemake normalize --cores 8
snakemake annotations --cores 8
snakemake split --cores 4
snakemake propose_excerpts --cores 8
snakemake generate_valid --cores 16
snakemake generate_invalid --cores 16
snakemake generate_preference_candidates --cores 8
snakemake validate --cores 8
snakemake export --cores 8
snakemake report --cores 4
```

The aggregate target:

```bash
snakemake all --cores 16
```

must stop at human-review checkpoints when approved or labelled records do not yet exist.

### 22.2 Human checkpoints

```text
propose_excerpts
    → pause
    → human review
    → reviewed_excerpts

generate_preference_candidates
    → pause
    → human labels
    → reviewed_preference_pairs
```

Provide explicit commands to resume after review.

---

## 23. Command-Line Interface

Expose an equivalent CLI:

```bash
fugue-data acquire --all
fugue-data normalize --all
fugue-data annotate --source algomus
fugue-data split --seed 20260728
fugue-data propose-excerpts --count 300
fugue-data review-excerpts
fugue-data generate-valid --count 10000 --profile high_baroque_conservative
fugue-data generate-invalid --count 10000
fugue-data generate-pairs --target 5000
fugue-data review-pairs
fugue-data validate --all
fugue-data export --all
fugue-data report
```

Every command must:

- be restartable;
- avoid duplicating successful work;
- write structured logs;
- return nonzero on failed acceptance criteria;
- support `--dry-run` when mutation is involved.

---

## 24. Logging and Reports

### 24.1 Structured logs

Log:

```yaml
timestamp
stage
piece_id
source_file_id
operation
status
warning_code
error_code
duration_ms
details
```

### 24.2 Final build report

The report must include:

- source repositories and commits;
- source-file counts;
- unique-piece counts;
- normalization success/quarantine/failure counts;
- parser warning counts;
- annotation counts by type;
- split counts;
- duplicate-group statistics;
- excerpt proposal/review counts;
- synthetic-valid counts and coverage;
- controlled-invalid counts by rule;
- preference-pair counts and label distribution;
- validation-gate results;
- visual-QA results;
- exported record counts;
- schema, tokenizer, rule, and pipeline versions.

---

## 25. Testing Requirements

### 25.1 Unit tests

Cover:

- rational-time conversion;
- tie merging;
- repeat expansion;
- voice separation;
- interval classification;
- perfect-parallel detection;
- subject transposition matching;
- annotation onset conversion;
- hashing;
- split inheritance;
- each mutator;
- schema validation.

### 25.2 Integration tests

Include:

- one WTC I fugue with Algomus annotations;
- one WTC II fugue;
- one BCFB chorale with alternate version behavior;
- one deliberately malformed score;
- one valid synthetic fragment;
- one controlled invalid fragment;
- one reviewed excerpt;
- one reviewed preference pair.

### 25.3 Regression fixtures

Pin representative canonical outputs and hashes. Any change requires:

- documented migration;
- schema or normalizer version change;
- review of affected exports.

---

## 26. Performance Expectations

The pipeline should:

- download and normalize the historical MVP corpus on an ordinary workstation;
- parallelize piece-level normalization;
- checkpoint synthetic generation;
- support resuming interrupted solver batches;
- avoid loading the entire corpus into memory;
- write compressed shards incrementally.

Do not sacrifice determinism or validation for throughput.

---

## 27. Safety and Data Integrity

- Never overwrite raw files.
- Never silently repair a source encoding.
- Store corrections as overlays with author, timestamp, reason, and before/after hashes.
- Never label synthetic output as Bach.
- Never label automatically selected excerpts as human approved.
- Never add unreviewed preference labels.
- Never mix train and held-out work derivatives.
- Never promote a soft preference to a hard rule without versioning and empirical validation.

---

## 28. Definition of Done

The initial pipeline is complete when all of the following are true:

- [ ] Source repositories are pinned and checksummed.
- [ ] 48 WTC fugues are normalized.
- [ ] 139 unique BCFB chorales are normalized.
- [ ] BCFB variants are grouped correctly.
- [ ] 24 WTC I fugues have aligned Algomus annotations.
- [ ] Canonical score schema is documented and validated.
- [ ] Complete-work splits are deterministic and leakage-free.
- [ ] At least 200 real excerpt candidates are proposed.
- [ ] 50–100 excerpts can be manually approved through the review tool.
- [ ] Rule profiles and rule schemas are implemented.
- [ ] 10,000 unique hard-valid fragments are independently validated.
- [ ] 10,000 unique controlled single-violation fragments are validated.
- [ ] Preference candidates can be generated and reviewed.
- [ ] At least 500 reviewed pairs are supported for the first training iteration.
- [ ] All training exports are produced separately.
- [ ] Automated validation gates pass.
- [ ] Required visual QA is recorded.
- [ ] Final build report is generated.
- [ ] A clean checkout can reproduce the build using documented commands.

---

## 29. Recommended Implementation Order

1. Repository structure and configuration.
2. Source acquisition and provenance manifests.
3. Canonical score schema.
4. WTC parsing and normalization.
5. BCFB parsing, identity grouping, and cross-format checks.
6. Canonical validation and hashing.
7. Algomus annotation alignment.
8. Complete-work deterministic splitting.
9. Model-view tokenization.
10. Excerpt proposal and review UI.
11. Rule schema and rule diagnostics.
12. Small valid solver prototype.
13. Small controlled-invalid generator.
14. Scale valid/invalid generation to required counts.
15. Preference candidate generation and review UI.
16. Exporters, full validation, and reports.

Do not begin large-scale synthetic generation before canonical validation, split isolation, and rule versioning are operational.

---

## 30. Agent Handoff Instruction

The implementing agent should begin by returning:

1. A proposed repository tree.
2. Dependency choices and exact versions.
3. A source-acquisition plan with resolved repository revisions.
4. The proposed canonical score JSON schema.
5. A risk list, especially for voice separation and annotation alignment.
6. A milestone plan corresponding to Section 29.

The agent should then implement one end-to-end vertical slice:

```text
one WTC I fugue
→ acquire
→ normalize
→ align annotations
→ validate
→ assign split
→ tokenize
→ export
→ render QA artifact
```

Only after that vertical slice passes should the agent scale acquisition and normalization to the full 187-piece corpus.

