# Annotation alignment — bach_wtc1_fugue_14 (BWV859)

**Source:** algomus · **map version:** 1.0.0 · **schema:** 1.0.0
**Score:** 4 voices, 240 quarter-notes, 40 bars
**Voice mapping:** `staff_index_top_down` (Dezrann staff *N* → canonical `voice_N`, both ordered highest-first)
**Alignment evidence:** voice-attributed labels land on sounding notes 100.0% of their span on average (worst 100.0%). A wrong staff→voice mapping would collapse this figure.

## Accepted annotations

20 labels — cadence 3, countersubject 7, pedal 1, subject 9.

| where | span | type | voice | entry function | coverage |
| --- | --- | --- | --- | --- | --- |
| m1 b2 | 17 | subject | voice_3 | subject | 100% |
| m4 b2 | 17 | subject | voice_2 | answer | 100% |
| m4 b4.5 | 29/2 | countersubject 1 | voice_3 | — | 100% |
| m8 b2 | 17 | subject | voice_4 | subject | 100% |
| m8 b4.5 | 29/2 | countersubject 1 | voice_2 | — | 100% |
| m15 b2 | 17 | subject | voice_1 | subject | 100% |
| m15 b4.5 | 29/2 | countersubject 1 | voice_4 | — | 100% |
| m20 b4 | — | cadence — v:PAC | — | — |  |
| m20 b5 | 17 | subject — inverted | voice_2 | answer | 100% |
| m25 b2.5 | 33/2 | subject — varied | voice_1 | undetermined | 100% |
| m25 b4.5 | 29/2 | countersubject 1 | voice_2 | — | 100% |
| m28 b1 | — | cadence — v:PAC | — | — |  |
| m29 b2 | 17 | subject | voice_3 | subject | 100% |
| m29 b4.5 | 29/2 | countersubject 1 | voice_1 | — | 100% |
| m32 b2 | 17 | subject — inverted | voice_4 | subject | 100% |
| m37 b2 | 17 | subject | voice_1 | undetermined | 100% |
| m37 b4 | 9 | pedal — on V | — | — |  |
| m37 b4.5 | 29/2 | countersubject 1 — varied | voice_2 | — | 100% |
| m37 b4.5 | 17/2 | countersubject 1 — incomplete | voice_3 | — | 100% |
| m40 b1 | — | cadence — I:PAC | — | — |  |

### Derived subject / answer

Algomus labels every entry `S`; tonal function below is **computed**, not annotated — the head note of each entry compared against the first entry (mod 7). Of 9 entries: **5 at subject (tonic) level**, **2 at answer (dominant) level**, 2 at other transposition levels (middle entries in related keys), which this method leaves undetermined rather than guessing.

## Rejected candidates (Dezrann `ignore`)

6 spans the analyst explicitly considered and declined to count as entries. Retained as near-miss evidence, excluded from accepted totals and from default exports.

| where | span | would-be type | voice |
| --- | --- | --- | --- |
| m8 b1.25 | 11/4 | countersubject 1 — extended | voice_1 |
| m8 b2 | 17 | countersubject 2 | voice_3 |
| m15 b1.25 | 11/4 | countersubject 1 — extended | voice_4 |
| m15 b1.5 | 35/2 | countersubject 2 — varied | voice_3 |
| m29 b1.25 | 11/4 | countersubject 1 — extended | voice_1 |
| m29 b2 | 17 | countersubject 2 — varied | voice_2 |

## Warnings

- `annotation_low_voice_coverage` bach_wtc1_fugue_14:ref:003: voice_1 sounds for only 0.00 of the labelled span; staff-to-voice mapping suspect
