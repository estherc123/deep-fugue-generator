# Annotation alignment — bach_wtc1_fugue_23 (BWV868)

**Source:** algomus · **map version:** 1.0.0 · **schema:** 1.0.0
**Score:** 4 voices, 136 quarter-notes, 34 bars
**Voice mapping:** `staff_index_top_down` (Dezrann staff *N* → canonical `voice_N`, both ordered highest-first)
**Alignment evidence:** voice-attributed labels land on sounding notes 100.0% of their span on average (worst 100.0%). A wrong staff→voice mapping would collapse this figure.

## Accepted annotations

21 labels — cadence 4, countersubject 4, pedal 1, subject 12.

| where | span | type | voice | entry function | coverage |
| --- | --- | --- | --- | --- | --- |
| m1 b1.5 | 15/2 | subject | voice_3 | subject | 100% |
| m3 b1.5 | 15/2 | subject | voice_2 | answer | 100% |
| m3 b2.5 | 6 | countersubject 1 | voice_3 | — | 100% |
| m5 b1.5 | 15/2 | subject | voice_1 | subject | 100% |
| m5 b2.5 | 13/2 | countersubject 1 | voice_2 | — | 100% |
| m7 b1.5 | 15/2 | subject | voice_4 | answer | 100% |
| m7 b2.5 | 13/2 | countersubject 1 | voice_1 | — | 100% |
| m11 b3.5 | 15/2 | subject | voice_3 | subject | 100% |
| m13 b3 | — | cadence — I:rIAC | — | — |  |
| m16 b1.5 | 15/2 | subject — fragmented, varied | voice_2 | answer | 100% |
| m18 b1 | — | cadence — V:PAC | — | — |  |
| m18 b1.5 | 15/2 | subject — inverted | voice_1 | answer | 100% |
| m20 b1.5 | 15/2 | subject — inverted | voice_2 | subject | 100% |
| m21 b3.5 | 15/2 | subject | voice_4 | subject | 100% |
| m24 b1.5 | 15/2 | subject — fragmented, varied | voice_3 | undetermined | 100% |
| m26 b1 | — | cadence — ii:PAC | — | — |  |
| m29 b1.5 | 15/2 | subject | voice_2 | subject | 100% |
| m31 b1.5 | 15/2 | subject | voice_1 | answer | 100% |
| m31 b2.5 | 6 | countersubject 1 | voice_2 | — | 100% |
| m31 b3 | 3 | pedal — on V | — | — |  |
| m34 b1 | — | cadence — I:rIAC | — | — |  |

### Derived subject / answer

Algomus labels every entry `S`; tonal function below is **computed**, not annotated — the head note of each entry compared against the first entry (mod 7). Of 12 entries: **6 at subject (tonic) level**, **5 at answer (dominant) level**, 1 at other transposition levels (middle entries in related keys), which this method leaves undetermined rather than guessing.

## Rejected candidates (Dezrann `ignore`)

4 spans the analyst explicitly considered and declined to count as entries. Retained as near-miss evidence, excluded from accepted totals and from default exports.

| where | span | would-be type | voice |
| --- | --- | --- | --- |
| m11 b2.5 | 9/2 | countersubject 1 — incomplete | voice_1 |
| m12 b2.5 | 9/2 | countersubject 1 — incomplete | voice_2 |
| m16 b4.5 | 9/2 | countersubject 1 — incomplete, varied | voice_1 |
| m29 b4.5 | 9/2 | countersubject 1 — incomplete, varied | voice_3 |

## Warnings

None.
