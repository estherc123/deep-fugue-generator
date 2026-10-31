# Fugue Dataset — Collection Progress

**Last updated:** 2026-08-30 (UTC)
**Pipeline stages reached:**
- Agent counterpoint preferences — **in progress** → see [`agent_review/README.md`](agent_review/README.md)
- Acquisition (Spec §9) — complete, all acceptance gates pass — this file
- Normalization (Spec §10) — **full 187-piece corpus complete** → see [`normalization.md`](normalization.md)
- Annotation alignment (Spec §11) — **complete, 23/23 available fugues** → see [`annotations.md`](annotations.md)
- Deduplication + splitting (Spec §12, §13) — **complete, 0 leakage failures** → see [`splits.md`](splits.md)
- Tokenization, rules, generation, exports (Spec §14-§21, §24) — **complete except the
  two human-review checkpoints** → see [`generation.md`](generation.md)

**Machine-readable status:** [`status.json`](status.json) · [`normalization_status.json`](normalization_status.json) · [`annotation_status.json`](annotation_status.json) · [`split_status.json`](split_status.json)

This folder records the **real Bach corpus actually collected** by the acquisition
stage. Every count here is derived from files on disk (checksummed, pinned to an
immutable commit) by `python -m fugue_data.acquire.report` — nothing is hand-typed.

---

## 1. Headline: 187 unique real Bach movements collected ✅

| Corpus | Collected | Spec target | Status |
| --- | ---: | ---: | :--- |
| WTC Book I fugues | 24 | 24 | ✅ |
| WTC Book II fugues | 24 | 24 | ✅ |
| **WTC fugues total** | **48** | **48** | ✅ |
| BCFB unique chorales | 139 | 139 | ✅ |
| BCFB encoded variants | 143 | 143 | ✅ |
| Algomus WTC I reference annotations | 23 | 24 | ⚠️ documented gap (see §4) |
| **Unique real Bach movements** | **187** | **187** | ✅ |

Also collected and preserved (not in the default fugue corpus):
- **48 WTC preludes** (raw, retained, excluded by default per Spec §2.1)
- **143 kern** + **143 mei** secondary BCFB encodings (cross-format validation, Spec §10.2)
- **60 Algomus `.dez`** analysis files (23 reference + 5 computed + variants) and audio-sync JSON

---

## 2. Pinned sources & provenance (Spec §4.1)

All three repositories are checked out in detached-HEAD state at a full commit SHA;
`.git` is stripped so the raw trees under `data/raw/` are immutable. Full per-file
provenance (sha256, license, format, claimed identity) is in
[`../manifests/sources.jsonl`](../manifests/sources.jsonl) — **1,029 manifest rows**.

| Source | Repository | Pinned commit | License |
| --- | --- | --- | --- |
| `humdrum-wtc` | github.com/humdrum-tools/bach-wtc | `0b4f4d84…9611cb6` | unknown (encoding) |
| `algomus` | gitlab.com/algomus.fr/algomus-data | `a1801b5b…07cd5ef` | annotation (attrib. req.) |
| `bcfb` | github.com/juyaolongpaul/Bach_chorale_FB | `431c5c01…64c2d786` | CC BY 4.0 |

---

## 3. What each inventory file contains

| File | Rows | Description |
| --- | ---: | --- |
| [`wtc_fugues.csv`](wtc_fugues.csv) | 48 | Every WTC fugue: `piece_id`, book, BWV, source path, sha256 |
| [`bcfb_chorales.csv`](bcfb_chorales.csv) | 143 | Every canonical MusicXML chorale variant, grouped by BWV work |
| [`algomus_annotations.csv`](algomus_annotations.csv) | 24 | Per-fugue annotation coverage (reference + computed) |
| [`algomus_alignment.csv`](algomus_alignment.csv) | 23 | Per-fugue *aligned* annotation counts, coverage, warnings |
| [`splits.csv`](splits.csv) | 191 | Per-piece split assignment + stratification fields |
| [`status.json`](status.json) | — | Machine-readable totals + pinned commits |

### WTC fugues (BWV mapping)
- **Book I:** `bach_wtc1_fugue_01`…`_24` → BWV 846–869
- **Book II:** `bach_wtc2_fugue_01`…`_24` → BWV 870–893

### BCFB chorales
- 143 MusicXML variants → 139 unique works *by BWV identity*; deduplication later
  found two further content-duplicate pairs, so there are 137 distinct musical
  works (see [`splits.md`](splits.md)). Four chorales carry two variants
  (a/b), correctly grouped under one `work_group_id`:
  **BWV 10.07, 38.06, 161.06, 177.05**.
- 11 standalone chorales use dot-less BWV numbers (BWV 250–252, 447, 461, 470,
  481, 488, 495, 499, 506) and are handled.

---

## 4. ⚠️ Documented deviation: Algomus fugue 11 (BWV 856)

The spec expects structural annotations for all 24 WTC I fugues. At the pinned
Algomus revision, **fugue 11 (BWV 856) has no `-ref.dez` structural analysis** —
only an audio-synchro JSON, which is not a structural label set. So reference
coverage is **23/24**, not 24/24.

This is a real upstream gap, **not fabricated coverage**. It is recorded as a
versioned override in [`../config/pipeline.yaml`](../config/pipeline.yaml)
(`overrides.algomus_wtc1_coverage`), so the acceptance gate passes *with the
deviation explicitly accepted and reasoned*, per Spec §9.2. The missing fugue is
listed in `status.json` under `summaries.algomus.wtc1_fugues_missing_ref`.

Annotation alignment (Spec §11) accordingly aligned 23 fugues; fugue 11 carries
full WTC score data but no Algomus structural labels until upstream provides
them or another annotation source is added. See [`annotations.md`](annotations.md).

---

## 5. How to reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install pyyaml==6.0.2
PYTHONPATH=src .venv/bin/python -m fugue_data.cli acquire --all   # pin + checksum + manifest
PYTHONPATH=src .venv/bin/python -m fugue_data.acquire.report      # regenerate this folder
```

Acquisition is idempotent: re-running reuses the pinned checkouts and re-verifies
counts. A count mismatch returns a nonzero exit and names the failing criterion.

---

## 6. Next stages (not yet built)

Done: acquisition (§9), canonical score schema (§8), **full-corpus
normalization** — 48 WTC fugues + 139 BCFB chorales (143 variants) with
cross-format checks, validation, and hashing (§10 / §10.2, see
[`normalization.md`](normalization.md)) — and **Algomus annotation alignment**
(§11, 804 labels over 23 fugues, see [`annotations.md`](annotations.md)), and
**deduplication + complete-work splitting** (§12/§13, see [`splits.md`](splits.md)).
Also done: tokenization (§19), excerpt proposal + review UI (§14),
the rule framework (§15),
synthetic valid/invalid generation (§16–17) → preference pairs (§18) → exports,
validation gates, and the final build report. See the repo `README.md` for the
milestone plan.
