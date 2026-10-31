# fugue-data

Reproducible, provenance-tracked pipeline that builds a training-ready symbolic
dataset of High-Baroque counterpoint (principally J. S. Bach) for a hybrid
hard-constraint / learned-prior / RL fugue-composition system.

Full specification: [`fugue_data_collection_pipeline_spec.md`](fugue_data_collection_pipeline_spec.md).
Collected-dataset record: [`progress/`](progress/README.md).

## Status

| Stage (Spec §29) | State |
| --- | --- |
| 1. Repo structure & config | ✅ done |
| 2. Source acquisition & provenance manifests | ✅ done — 187 real Bach movements, all gates pass |
| 3. Canonical score schema | ✅ done — `schemas/canonical_score.schema.json` |
| 4. WTC parsing & normalization | ✅ done — 48/48 fugues, 0 validation failures |
| 5. BCFB parsing, grouping & cross-format checks | ✅ done — 143 variants→139 unique, 0 discrepancies |
| 6. Canonical validation & hashing | ✅ done — exact/melodic/rhythm/per-voice hashes |
| 7. Algomus annotation alignment | ✅ done — 804 labels over 23 fugues, 0 validation failures |
| 8. Deduplication & complete-work splitting | ✅ done — 6 duplicate groups, 0 leakage failures |
| 9. Model-view tokenization | ✅ done — 191 pieces, 0 round-trip failures |
| 10. Excerpt proposal & review UI | ✅ built — 300 candidates; **approval pending a human** |
| 11. Rule schema & diagnostics | ✅ done — 24 rules, 3 profiles, empirically calibrated |
| 12–14. Valid & invalid generation | ✅ done — 10,000 + 10,000, all gates pass |
| 15. Preference candidates & review UI | ✅ built — 3,150 pairs; **labels pending a human** |
| 16. Exporters, validation, reports | ✅ done — 17/17 gates, build report written |

**Full 187-piece corpus normalized:** 48 WTC fugues + 139 BCFB chorales (143
variants) in the canonical rational-time model, schema-validated (0 failures),
5,843 bars / 93,584 note events.

**WTC I structurally annotated:** all 804 Algomus reference labels over 23
fugues aligned onto canonical scores — 368 subject entries, 174 countersubject
entries, 102 cadences, 19 pedal points — with the staff→voice mapping verified
empirically (0.969 mean note coverage vs 0.786 reversed) rather than assumed.
**Deduplicated and split:** 191 pieces in 185 split units, WTC 36/6/6 and BCFB
111/14/14 by unique work, all §13.5 leakage gates passing. Deduplication caught
two duplicate pairs that work identity alone would have missed, including one
byte-identical file shipped upstream under two names.

**Generated and exported:** 10,000 hard-valid synthetic fragments and 10,000
controlled single-violation fragments (each with exactly one intended violation,
8 rules at 12.5% apiece), 3,150 preference-pair candidates, 300 excerpt
candidates, 5 training exports, **17/17 validation gates passing**, and a build
report scoring 17/19 on §28's Definition of Done.

**The 2 outstanding criteria both require a person** and the pipeline will not
fake them (§27): 50–100 excerpts must be manually approved (§14.6) and ≥500
preference pairs manually labelled (§18.6). Both review apps are built and
working — see [`progress/generation.md`](progress/generation.md) §7.

See [`progress/README.md`](progress/README.md),
[`progress/normalization.md`](progress/normalization.md),
[`progress/annotations.md`](progress/annotations.md),
[`progress/splits.md`](progress/splits.md) and
[`progress/generation.md`](progress/generation.md).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt              # music21, jsonschema, pyyaml, ...

# 1. Acquire all sources (clone → pin to SHA → checksum → manifest → verify counts)
PYTHONPATH=src .venv/bin/python -m fugue_data.cli acquire --all
# or a single source:  ... acquire --source bcfb

# 2. Normalize WTC fugues → canonical rational-time scores + validate
PYTHONPATH=src .venv/bin/python -m fugue_data.cli normalize --all

# 3. Align Algomus structural annotations onto the canonical WTC I fugues
PYTHONPATH=src .venv/bin/python -m fugue_data.cli annotate --source algomus

# 4. Deduplicate and assign deterministic complete-work splits
PYTHONPATH=src .venv/bin/python -m fugue_data.cli split          # or --seed N

# 5. Model views, excerpts, generation, exports, gates, report
PYTHONPATH=src .venv/bin/python -m fugue_data.cli tokenize
PYTHONPATH=src .venv/bin/python -m fugue_data.cli propose-excerpts
PYTHONPATH=src .venv/bin/python -m fugue_data.cli generate-valid     # ~59 min
PYTHONPATH=src .venv/bin/python -m fugue_data.cli generate-invalid
PYTHONPATH=src .venv/bin/python -m fugue_data.cli generate-pairs
PYTHONPATH=src .venv/bin/python -m fugue_data.cli export --all
PYTHONPATH=src .venv/bin/python -m fugue_data.cli validate --all
PYTHONPATH=src .venv/bin/python -m fugue_data.cli report
# ...or the whole thing, stopping at the two human checkpoints:
snakemake all --cores 16

# Human review (the two steps the pipeline cannot do for you)
PYTHONPATH=src .venv/bin/python -m fugue_data.review.app         # approve 50-100 excerpts
PYTHONPATH=src .venv/bin/python -m fugue_data.review.pairs_app   # label 500+ pairs

# 6. Regenerate the progress/ dataset record
PYTHONPATH=src .venv/bin/python -m fugue_data.acquire.report
PYTHONPATH=src .venv/bin/python -m fugue_data.annotations.progress
PYTHONPATH=src .venv/bin/python -m fugue_data.splitting.progress

# Tests
.venv/bin/python -m pytest tests -q
```

## Layout

```
config/           sources.yaml (pinned SHAs), pipeline.yaml, annotations.yaml,
                  splits.yaml, tokenizer.yaml, excerpts.yaml, generation.yaml,
                  rules/ (registry + 3 severity profiles)
schemas/          source_manifest, canonical_score, annotation, split_manifest,
                  rule, violation, review, preference_pair, generation_request
src/fugue_data/   acquire/ normalize/ annotations/ splitting/ tokenization/
                  review/ constraints/ generation/ export/ validation/ report/
data/raw/         immutable pinned checkouts (humdrum-wtc, algomus, bcfb)
data/canonical/   191 canonical scores — 48 WTC fugues + 143 chorale variants
data/annotations/ 23 aligned WTC I annotation sets (<piece_id>.json)
data/tokenized/   grid + event model views, vocabulary.json
data/proposed/    300 excerpt candidates with SVG/MIDI renders
data/generated/   valid/ (10,000) invalid/ (10,000) preference_candidates/ (3,150)
data/exports/     style, feasibility, constraint_critic, structure, preferences,
                  metadata
manifests/        sources.jsonl/csv, splits.jsonl/csv, duplicate_groups.json +
                  acquire / normalization / annotation / split summaries
reports/          per-fugue alignment reports (§11.4), splits.md (§12/§13),
                  visual_qa/ (294 artefacts), build_report.json, pipeline.log.jsonl
Snakefile         §22 orchestration with both human checkpoints
progress/         human-readable record of the collected dataset
```

## Design notes

- **Determinism & provenance first.** Sources are pinned to full commit SHAs and
  checked out detached; raw trees are immutable (`.git` stripped, per-file sha256).
- **Acquisition is dependency-light** (stdlib + PyYAML) so it runs before the heavy
  symbolic-music stack (`music21`, OR-Tools) is installed. Those land with §10
  normalization.
- **No faked coverage.** The one upstream gap (Algomus fugue 11 / BWV 856 has no
  structural `.dez`) is recorded as a documented override, not papered over.
- **Mappings are data, and are verified.** The Dezrann→canonical label mapping is
  a versioned config file, not code. The staff→voice direction it depends on was
  established by measuring how much of each labelled span the resolved voice
  actually sounds, and that measurement is kept as a per-annotation gate.
- **Inference is never dressed as source data.** Algomus does not distinguish
  subject from answer; the pipeline derives it and records it in a separate
  `entry_function` field with its derivation named, leaving the source-given
  `type` untouched (§27). Split `mode` is likewise derived and labelled
  `mode_source`, and validated against the WTC's own major/minor ordering (48/48).
- **Splits are proved, not asserted.** The leakage gates verify every
  `assignment_hash` and are themselves tested by tampering with a manifest and
  requiring rejection.
- **Generator and validator are separate on purpose** (§16.4). CP-SAT proposes;
  acceptance is decided by re-deriving everything from the serialized fragment,
  with no access to the solver's model.
- **Rules are calibrated against Bach, not against textbooks** (§15.6). Every
  hard rule was measured on 41 held-out pieces; four were corrected or demoted
  because the corpus said they were wrong. `bach_empirical` now fires zero hard
  violations on real Bach.
- **Human review is never simulated.** Excerpt approvals and preference labels
  stay empty until a person supplies them, and the exports, gates and build
  report all say so rather than filling the gap.
