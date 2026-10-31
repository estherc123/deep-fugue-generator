# Annotation alignment — bach_wtc1_fugue_08 (BWV853)

**Source:** algomus · **map version:** 1.0.0 · **schema:** 1.0.0
**Score:** 3 voices, 348 quarter-notes, 87 bars
**Voice mapping:** `staff_index_top_down` (Dezrann staff *N* → canonical `voice_N`, both ordered highest-first)
**Alignment evidence:** voice-attributed labels land on sounding notes 99.6% of their span on average (worst 95.0%). A wrong staff→voice mapping would collapse this figure.

## Accepted annotations

32 labels — cadence 6, subject 26.

| where | span | type | voice | entry function | coverage |
| --- | --- | --- | --- | --- | --- |
| m1 b1 | 10 | subject | voice_2 | subject | 100% |
| m3 b3 | 10 | subject | voice_1 | answer | 100% |
| m8 b1 | 10 | subject | voice_3 | subject | 100% |
| m12 b1.5 | 19/2 | subject | voice_3 | answer | 100% |
| m19 b3 | — | cadence — v:PAC | — | — |  |
| m19 b3 | 10 | subject | voice_2 | answer | 95% |
| m20 b1 | 10 | subject | voice_1 | answer | 95% |
| m24 b1 | 10 | subject | voice_1 | undetermined | 100% |
| m26 b4 | 11 | subject — varied | voice_1 | subject | 100% |
| m27 b3 | 10 | subject | voice_2 | undetermined | 100% |
| m30 b1 | — | cadence — III:PAC | — | — |  |
| m30 b1 | 10 | subject — inverted | voice_1 | undetermined | 100% |
| m36 b1 | 10 | subject — inverted | voice_2 | undetermined | 100% |
| m39 b1 | — | cadence — I:PAC | — | — |  |
| m39 b1 | 10 | subject — fragmented, inverted, varied | voice_3 | answer | 100% |
| m44 b3 | 10 | subject — inverted | voice_3 | answer | 100% |
| m45 b1 | 10 | subject — inverted | voice_1 | undetermined | 100% |
| m47 b3 | 10 | subject — inverted | voice_2 | undetermined | 100% |
| m52 b1 | — | cadence — I:HC | — | — |  |
| m57 b3.5 | 19/2 | subject | voice_1 | undetermined | 100% |
| m61 b3 | — | cadence — I:HC | — | — |  |
| m61 b3.5 | 19/2 | subject — varied | voice_2 | undetermined | 100% |
| m62 b1 | 20 | subject — augmented | voice_3 | answer | 100% |
| m64 b3 | 10 | subject — inverted | voice_1 | answer | 100% |
| m67 b1 | 10 | subject | voice_3 | undetermined | 100% |
| m67 b3 | 20 | subject — augmented | voice_2 | undetermined | 100% |
| m69 b3.5 | 19/2 | subject | voice_1 | undetermined | 100% |
| m72 b3 | 10 | subject | voice_2 | undetermined | 100% |
| m77 b1.5 | 19/2 | subject | voice_3 | undetermined | 100% |
| m77 b3 | 20 | subject — augmented | voice_1 | undetermined | 100% |
| m80 b3.5 | 19/2 | subject | voice_2 | undetermined | 100% |
| m87 b3 | — | cadence — I:PAC | — | — |  |

### Derived subject / answer

Algomus labels every entry `S`; tonal function below is **computed**, not annotated — the head note of each entry compared against the first entry (mod 7). Of 26 entries: **3 at subject (tonic) level**, **8 at answer (dominant) level**, 15 at other transposition levels (middle entries in related keys), which this method leaves undetermined rather than guessing.

## Rejected candidates (Dezrann `ignore`)

9 spans the analyst explicitly considered and declined to count as entries. Retained as near-miss evidence, excluded from accepted totals and from default exports.

| where | span | would-be type | voice |
| --- | --- | --- | --- |
| m24 b2 | 12 | subject — augmented, varied | voice_2 |
| m47 b4 | 12 | subject — inverted, varied | voice_1 |
| m52 b1 | 6 | subject — incomplete | voice_3 |
| m52 b2 | 6 | subject — incomplete | voice_2 |
| m52 b3 | 6 | subject — incomplete | voice_1 |
| m54 b1 | 6 | subject — incomplete, inverted | voice_3 |
| m54 b2 | 6 | subject — incomplete, inverted | voice_2 |
| m54 b3 | 6 | subject — incomplete, inverted | voice_1 |
| m77 b2.5 | 23/2 | subject — varied | voice_2 |

## Warnings

- `annotation_zero_duration_point_event` bach_wtc1_fugue_08:ref:014: cadence label carries duration 0 where the other cadence labels omit it; read as a point event
