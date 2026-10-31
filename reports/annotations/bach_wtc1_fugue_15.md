# Annotation alignment — bach_wtc1_fugue_15 (BWV860)

**Source:** algomus · **map version:** 1.0.0 · **schema:** 1.0.0
**Score:** 3 voices, 258 quarter-notes, 86 bars
**Voice mapping:** `staff_index_top_down` (Dezrann staff *N* → canonical `voice_N`, both ordered highest-first)
**Alignment evidence:** voice-attributed labels land on sounding notes 100.0% of their span on average (worst 100.0%). A wrong staff→voice mapping would collapse this figure.

## Accepted annotations

13 labels — cadence 2, pedal 4, subject 7.

| where | span | type | voice | entry function | coverage |
| --- | --- | --- | --- | --- | --- |
| m1 b1 | 12 | subject | voice_1 | subject | 100% |
| m5 b1 | 12 | subject | voice_2 | answer | 100% |
| m11 b1 | 12 | subject | voice_3 | subject | 100% |
| m20 b1 | 12 | subject — inverted | voice_2 | answer | 100% |
| m24 b1 | 12 | subject — inverted | voice_1 | undetermined | 100% |
| m38 b1 | 12 | subject | voice_1 | undetermined | 100% |
| m56 b1 | 6 | pedal — on V/vi | — | — |  |
| m62 b1 | 9 | pedal — on ii | — | — |  |
| m69 b4 | — | cadence — V:PAC | — | — |  |
| m69 b4 | 9/2 | pedal — on V | — | — |  |
| m69 b4 | 12 | subject — inverted, varied | voice_3 | undetermined | 100% |
| m83 b1 | — | cadence — I:PAC | — | — |  |
| m83 b1 | 12 | pedal — on I | — | — |  |

### Derived subject / answer

Algomus labels every entry `S`; tonal function below is **computed**, not annotated — the head note of each entry compared against the first entry (mod 7). Of 7 entries: **2 at subject (tonic) level**, **2 at answer (dominant) level**, 3 at other transposition levels (middle entries in related keys), which this method leaves undetermined rather than guessing.

## Rejected candidates (Dezrann `ignore`)

19 spans the analyst explicitly considered and declined to count as entries. Retained as near-miss evidence, excluded from accepted totals and from default exports.

| where | span | would-be type | voice |
| --- | --- | --- | --- |
| m5 b1 | 12 | countersubject 1 | voice_1 |
| m12 b5 | 11/2 | countersubject 1 — incomplete | voice_2 |
| m20 b1 | 9 | countersubject 1 — inverted | voice_3 |
| m24 b1 | 9 | countersubject 1 — inverted | voice_2 |
| m28 b1 | 6 | subject — incomplete, inverted | voice_3 |
| m29 b2 | 11/2 | countersubject 1 — inverted | voice_1 |
| m40 b1.5 | 17/4 | countersubject 1 — incomplete | voice_2 |
| m43 b1 | 9 | countersubject 1 — inverted | voice_1 |
| m43 b1 | 6 | subject — incomplete, inverted | voice_2 |
| m51 b1.5 | 35/4 | subject — incomplete | voice_1 |
| m52 b1 | 6 | subject — incomplete | voice_3 |
| m60 b4 | 9 | subject — incomplete | voice_2 |
| m61 b4 | 15/2 | subject — incomplete | voice_1 |
| m70 b4.5 | 23/4 | countersubject 1 — inverted | voice_2 |
| m77 b1 | 6 | subject — incomplete, inverted | voice_2 |
| m78 b4 | 3/2 | subject — incomplete, inverted | voice_3 |
| m79 b1 | 9 | subject — incomplete | voice_1 |
| m79 b1 | 9/2 | subject — incomplete | voice_2 |
| m79 b1 | 3 | subject — incomplete, inverted | voice_3 |

## Warnings

None.
