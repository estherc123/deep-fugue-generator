# Annotation alignment — Progress

**Last updated:** 2026-08-30 (UTC)
**Pipeline stage:** Algomus annotation alignment (Spec §11) — **complete, 23/23 available fugues**
**Machine-readable status:** [`annotation_status.json`](annotation_status.json)
**Inventory:** [`algomus_alignment.csv`](algomus_alignment.csv) · **Per-fugue reports:** [`../reports/annotations/`](../reports/annotations/)

Every WTC I fugue that ships a human structural analysis at the pinned Algomus
revision is now aligned onto its canonical score and written to
`data/annotations/<piece_id>.json`, schema-validated against
`schemas/annotation.schema.json`. Every count below is derived from those files
by `python -m fugue_data.annotations.progress` — nothing is hand-typed.

## Headline

| Metric | Value |
| --- | ---: |
| Fugues aligned | **23 / 23 available** (24 nominal — see §4 below) |
| Source labels ingested | **804 / 804** (nothing dropped) |
| Accepted annotations | **668** |
| Rejected candidates retained | **136** |
| Analytical alternative groups (§11.3) | **3** |
| Validation failures | **0** ✅ |
| Alignment warnings | **2** (both explained below) |

### Accepted annotations by canonical type (§11.2)

| Type | Count |
| --- | ---: |
| subject | 368 |
| countersubject | 174 |
| cadence | 102 |
| pedal | 19 |
| episode | 4 |
| other (codetta) | 1 |

Cadences carry a parsed key area and type: 66 PAC, 24 rIAC, 7 HC, 4 DC, 1 IAC.
Entries carry parsed transformations: 46 incomplete, 43 inverted, 28 varied,
4 fragmented, 3 augmented, plus in-episode and segment markers.

## 1. Time base and voice mapping — both verified, not assumed

Dezrann `start`/`duration` are quarter-note units from the start of the piece,
the same time base as canonical `onset_q`. Confirmed: for all 23 files the
maximum label extent lies within the canonical piece length.

Dezrann `staff` is **not** a keyboard staff — it is a voice index numbered
top-down, running 1..`voice_count` and reaching 5 in exactly the two 5-voice
fugues (BWV 849, BWV 867). It therefore maps to canonical `voice_N`, which is
also ordered highest-first.

That direction was established empirically rather than assumed. For every
voice-attributed annotation the pipeline measures `voice_note_coverage` — the
fraction of the labelled span in which the resolved voice actually sounds:

| Mapping | Corpus mean coverage |
| --- | ---: |
| `staff N → voice_N` (adopted) | **0.969** |
| reversed (`staff N → voice_{count+1-N}`) | 0.786 |

The adopted mapping wins in **all 23 fugues** individually. Across the 542
accepted voice-attributed annotations the minimum is 0.533 and 394 sit at 100%.
A wrong mapping would collapse this figure, so the pipeline gates on it
(`min_voice_note_coverage: 0.5`) and records it per annotation and per piece.

## 2. Vocabulary mapping (§11.2)

The Algomus reference files use 22 raw label types and 90+ distinct tags against
the seven canonical types. The mapping is data, not code — it lives in
[`../config/annotations.yaml`](../config/annotations.yaml) and is versioned
(`annotation_map_version: 1.0.0`) because changing it changes downstream
semantics. **All 804 reference labels map; zero fall through to a guess.** The
raw `type`, `tag`, `staff`, `start` and `duration` are preserved verbatim on
every record, so nothing in the source vocabulary is lost in translation.

Analytical qualifiers encoded in the raw type (`S-inv`, `S-invinc`, `S-aug`,
`CS1-inc`, …) become a structured `transformations` list rather than being
flattened away, and `S2`/`S3`/`CS1`/`CS2`/`CS3` keep their subject and
countersubject indices — which is what makes BWV 849's triple-fugue structure
survive the mapping.

### `ignore` labels are kept, not discarded

136 source labels are typed `ignore` — spans the analysts explicitly considered
and declined to count as entries (fragmentary, varied past recognition, or a
subject shape drifting through an episode). They are retained with
`analytical_status: rejected_candidate`, keeping the musical family their tag
implies, because they record where the analyst drew the boundary of the concept
— exactly the near-miss evidence rule validation needs (§15.6). They never count
toward accepted totals and are excluded from default exports.

## 3. Derived subject / answer — clearly marked as derived

Algomus labels every entry `S` and never distinguishes subject from answer, but
§11.2 has an `answer` type and §15.5 needs answer transformations. Tonal
function is therefore **computed** — each entry's head note compared against the
first entry of the piece, mod 7 — and stored in `entry_function` alongside
`entry_function_derivation`, never in the source-given `type` field (§27).

Of 368 accepted entries: **130 at subject (tonic) level, 86 at answer (dominant)
level, 152 undetermined.** The undetermined ones are middle entries at other
transposition levels (mediant and related keys), which the method leaves
undetermined rather than guessing. Validated against published analyses: BWV 847
reads subject (alto, m1) → answer (soprano, m3) → subject (bass, m7), and BWV
846 reads subject (m1) → answer (m2), both matching the textbook expositions.

## 4. ⚠️ Documented deviation: fugue 11 (BWV 856)

Unchanged from acquisition: at the pinned Algomus revision, fugue 11 ships only
an audio-synchro JSON, no `-ref.dez` structural analysis. Coverage is **23/24**,
recorded as a versioned override in
[`../config/annotations.yaml`](../config/annotations.yaml)
(`acceptance.overrides.ref_fugue_coverage`) with its reason, so the gate passes
*with the deviation explicitly accepted* rather than papered over. Fugue 11
carries full canonical score data and no structural labels.

## 5. The two alignment warnings

Both are real properties of the source data, surfaced rather than silently fixed:

- **`bach_wtc1_fugue_08` — cadence with `duration: 0`.** The mistyped
  `cad:PAC ?` label writes an explicit zero where all 101 other cadence labels
  omit `duration` entirely. A cadence is an arrival, not a span, so a zero
  duration on a point-event type is read as `null` and the annotation is kept.
  Dropping it would have silently lost a real cadence from BWV 853; the source
  value stays verbatim in `raw.duration`.
- **`bach_wtc1_fugue_14` — one label at 0% voice coverage.** `CS1-ext-IGN` at
  m8 points at `voice_1`, which does not enter until onset 85. It is the single
  outlier in the corpus and it is *already rejected material* — every one of the
  542 accepted voice-attributed annotations clears the coverage gate.

## 6. Analytical ambiguity preserved (§11.3)

3 alternative groups, **all in BWV 849** — the C♯ minor triple fugue, the one
WTC I fugue where competing readings of overlapping entries are expected. Each
group links an accepted `S2`/`S3` entry with a rejected candidate covering the
same span in the same voice. Disagreements are linked, never resolved by
discarding one side.

## 7. What each output contains

| Path | Contents |
| --- | --- |
| `data/annotations/<piece_id>.json` | Aligned annotation set + per-piece alignment provenance |
| `reports/annotations/<piece_id>.md` | Human-readable alignment report (§11.4) — every label in bars and beats |
| `manifests/annotation_summary.json` | Machine-readable stage summary and gate results |
| `progress/algomus_alignment.csv` | Per-fugue inventory: counts by type, coverage, warnings |

## 8. Acceptance criteria (§11.4)

| Criterion | Status |
| --- | --- |
| 24 WTC I fugues have annotation records | ⚠️ 23/24, documented upstream gap (§4) |
| Every annotation onset lies within the score | ✅ enforced, 0 violations |
| Every voice-specific annotation resolves to an existing voice | ✅ enforced, 0 violations |
| Every duration is positive | ✅ enforced (point events carry `null`, not `0`) |
| Ambiguous boundaries are preserved | ✅ 136 rejected candidates + 3 alternative groups |
| A human-readable alignment report per fugue | ✅ `reports/annotations/*.md` |

## 9. How to reproduce

```bash
PYTHONPATH=src .venv/bin/python -m fugue_data.cli annotate --source algomus
PYTHONPATH=src .venv/bin/python -m fugue_data.annotations.progress   # regenerate this record
```

Alignment is idempotent and restartable: a piece whose source `.dez` checksums
and annotation-map version already match on disk is skipped. A forced re-run is
byte-identical (§4.3). A count mismatch or a failed gate returns nonzero and
names the failing criterion.
