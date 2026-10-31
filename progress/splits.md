# Deduplication and dataset splits — Progress

**Last updated:** 2026-08-30 (UTC)
**Pipeline stages:** Deduplication (Spec §12) and complete-work splitting (Spec §13) — **complete, all leakage gates pass**
**Machine-readable status:** [`split_status.json`](split_status.json)
**Inventory:** [`splits.csv`](splits.csv) · **Report:** [`../reports/splits.md`](../reports/splits.md)
**Manifests:** [`../manifests/splits.jsonl`](../manifests/splits.jsonl) · [`../manifests/duplicate_groups.json`](../manifests/duplicate_groups.json)

Every count below is derived from the split manifest on disk by
`python -m fugue_data.splitting.progress` — nothing is hand-typed.

## Headline

| Metric | Value |
| --- | ---: |
| Pieces assigned | **191** |
| Split units (atoms of assignment) | **185** |
| Duplicate groups found | **6** |
| Leakage gate failures (§13.5) | **0** ✅ |
| Split seed | `20260728` |

## 1. Deduplication found two duplicates that work identity alone would have missed

All six §12 levels are computed. Four groups are the known BCFB a/b encoded
variants. **Two are not** — and both would have leaked across splits:

| Pair | How they match | What it is |
| --- | --- | --- |
| `BWV 117.04` / `BWV 117.09` | **identical source bytes** (same sha256) | Upstream ships the same MusicXML file under two names |
| `BWV 125.06` / `BWV 72.06` | different files, **identical canonical music** (215 notes, 13 bars, same event hash) | The same chorale setting published under two BWV numbers |

So BCFB's *nominal* 139 unique chorales contain **137 distinct musical works**.
The nominal count is still what the split targets are measured against (§13.3),
but the two content-duplicate pairs are now fused into single split units, so
neither can put the same music in both train and test.

### Level census

| Level | Match groups | Pieces | Blocking |
| --- | ---: | ---: | :---: |
| source_checksum | 1 | 2 | ✅ |
| canonical_event_hash | 4 | 8 | ✅ |
| transposition_melodic_hash | 4 | 8 | ✅ |
| per_voice_melody_hash | 4 | 8 | — |
| rhythm_hash | 5 | 10 | — |
| work_identity | 4 | 8 | ✅ |

*Blocking* levels fuse their members into one split unit. §13.5 requires that
neither exact nor transposition-normalized duplicates cross splits, so those
block. Per-voice-melody and rhythm-only matches are **similarity signals, not
identity** — two different chorales in the same meter can share a rhythm hash —
so they are reported for the build report but never fuse works on their own.

Nothing is deleted (§12). Every member keeps its provenance and each group names
one preferred representative, chosen deterministically as the lexicographically
first `piece_id` (which for a/b variants is variant `a`).

## 2. Split assignment (§13.2, §13.3)

Targets count **unique works**, not encoded variant files, so the four a/b
variant pairs never inflate a split.

| Corpus | Split | Target works | Assigned works | Pieces |
| --- | --- | ---: | ---: | ---: |
| WTC | train | 36 | **36** ✅ | 36 |
| WTC | validation | 6 | **6** ✅ | 6 |
| WTC | test | 6 | **6** ✅ | 6 |
| BCFB | train | 111 | **111** ✅ | 114 |
| BCFB | validation | 14 | **14** ✅ | 14 |
| BCFB | test | 14 | **14** ✅ | 15 |

The BCFB piece counts exceed the work counts by exactly the four a/b variants.

### Stratification balance (train / validation / test)

WTC is stratified by book, mode, voice count, meter and length bucket (§13.2):

| Field | Distribution |
| --- | --- |
| book | I 19/3/2 · II 17/3/4 |
| mode | major 17/3/4 · minor 19/3/2 |
| voice count | 2 → 1/0/0 · 3 → 18/4/4 · 4 → 15/2/2 · 5 → 2/0/0 |
| length | short 14/1/2 · medium 12/2/2 · long 10/3/2 |

The 2-voice fugue (BWV 855) and the two 5-voice fugues (BWV 849, BWV 867) are
too few to spread across three splits; they sit in train, which is the honest
outcome rather than a manufactured balance.

## 3. Mode is derived, and validated against ground truth

Neither the Humdrum WTC nor the BCFB MusicXML records mode — `mode_if_known` is
null throughout the corpus — but §13.2 requires stratifying by it. Mode is
therefore **derived** from a whole-piece pitch-class histogram around the tonic
(the final note of the lowest voice), and every manifest row carries
`mode_source: derived_pitch_class_histogram` so it is never mistaken for source
data.

The derivation is validated against real ground truth: the WTC's own ordering
fixes the mode of all 48 fugues (fugue 01 C major, 02 C minor, 03 C♯ major, …,
so odd indices are major and even are minor), and the rule **reproduces that for
48 of 48**. Because the histogram spans the whole piece, a Picardy third does
not flip a minor-mode piece — which is covered by a unit test.

## 4. A defect found and fixed during this stage

The first allocator assigned each unit to whichever split had the largest
**absolute** remaining deficit. With BCFB targets of 111/14/14 that hands the
first ~97 units straight to train before validation or test receive anything,
which sorts the stratified ordering into the splits and destroys the
stratification it exists to provide. The symptom was stark: **all 37 long
chorales and all 16 chorales in 3/4 landed in train, with zero validation or
test representation.**

Allocating on **proportional** deficit instead keeps all three splits advancing
together. Those same fields are now long 30/1/6 and 3/4 → 12/1/3, and every
stratum value appears in every split. A regression test pins the behaviour.

## 5. Leakage gates (§13.5)

| Gate | Status |
| --- | --- |
| No `work_group_id` appears in multiple splits | ✅ |
| No exact or transposition-normalized duplicate crosses splits | ✅ |
| Every split unit lies wholly within one split | ✅ |
| Every `assignment_hash` verifies against seed, unit and members | ✅ |
| Per-corpus work counts match configured targets | ✅ |

The gates are proved to bite, not merely to pass: an integration test tampers
with a split assignment and asserts the validator rejects it.

Excerpt, transposition and synthetic-derivative isolation (§13.1) applies to
stages not yet built. The mechanism is in place — every derivative inherits
`split` from its parent piece through `split_unit_id`.

## 6. Determinism (§4.3)

Given the same corpus, config and seed the assignment is byte-identical; a
forced re-run reproduces `manifests/splits.jsonl` exactly. Ordering comes from a
seeded SHA-256 over the split-unit id, never from Python's hash or set iteration
order. Changing the seed to 99999 reassigns 48 of 191 pieces while still hitting
every target exactly.

## 7. How to reproduce

```bash
PYTHONPATH=src .venv/bin/python -m fugue_data.cli split          # or --seed N
PYTHONPATH=src .venv/bin/python -m fugue_data.splitting.progress  # regenerate this record
```

A leakage-gate failure returns nonzero and names the failing criterion.
