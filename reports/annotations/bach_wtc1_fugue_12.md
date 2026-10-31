# Annotation alignment — bach_wtc1_fugue_12 (BWV857)

**Source:** algomus · **map version:** 1.0.0 · **schema:** 1.0.0
**Score:** 4 voices, 232 quarter-notes, 58 bars
**Voice mapping:** `staff_index_top_down` (Dezrann staff *N* → canonical `voice_N`, both ordered highest-first)
**Alignment evidence:** voice-attributed labels land on sounding notes 95.1% of their span on average (worst 75.0%). A wrong staff→voice mapping would collapse this figure.

## Accepted annotations

30 labels — cadence 3, countersubject 17, subject 10.

| where | span | type | voice | entry function | coverage |
| --- | --- | --- | --- | --- | --- |
| m1 b2 | 11 | subject | voice_3 | subject | 100% |
| m4 b1.5 | 23/2 | countersubject 1 | voice_3 | — | 93% |
| m4 b2 | 11 | subject | voice_2 | undetermined | 100% |
| m7 b1.5 | 23/2 | countersubject 1 | voice_2 | — | 93% |
| m7 b1.5 | 23/2 | countersubject 2 | voice_3 | — | 96% |
| m7 b2 | 11 | subject | voice_4 | subject | 100% |
| m13 b1.5 | 3/2 | countersubject 2 | voice_3 | — | 100% |
| m13 b1.5 | 23/2 | countersubject 1 | voice_4 | — | 93% |
| m13 b2 | 11 | subject | voice_1 | subject | 100% |
| m13 b3 | 10 | countersubject 2 | voice_2 | — | 85% |
| m13 b4.25 | 11/4 | countersubject 3 | voice_3 | — | 82% |
| m14 b3 | 6 | countersubject 3 | voice_2 | — | 75% |
| m19 b1.5 | 3/2 | countersubject 1 | voice_2 | — | 100% |
| m19 b2 | 11 | subject | voice_3 | answer | 100% |
| m19 b3 | 10 | countersubject 1 | voice_1 | — | 92% |
| m27 b3.5 | 3/2 | countersubject 2 | voice_2 | — | 100% |
| m27 b3.5 | 23/2 | countersubject 1 | voice_3 | — | 93% |
| m27 b4 | 11 | subject | voice_4 | subject | 100% |
| m28 b1 | 10 | countersubject 2 | voice_1 | — | 95% |
| m28 b2.25 | 35/4 | countersubject 3 | voice_2 | — | 83% |
| m34 b1.5 | 3/2 | countersubject 1 — inverted, segment_a | voice_3 | — | 100% |
| m34 b2 | 11 | subject | voice_2 | undetermined | 100% |
| m34 b3 | 10 | countersubject 1 — inverted, segment_b | voice_1 | — | 92% |
| m40 b4 | 11 | subject | voice_3 | undetermined | 100% |
| m43 b3 | — | cadence — VII:rIAC | — | — |  |
| m47 b1 | — | cadence — v:HC | — | — |  |
| m47 b1.5 | 23/2 | countersubject 1 — fragmented, varied | voice_2 | — | 93% |
| m47 b2 | 11 | subject | voice_1 | answer | 100% |
| m53 b2 | 11 | subject | voice_4 | subject | 100% |
| m58 b1 | — | cadence — I:PAC | — | — |  |

### Derived subject / answer

Algomus labels every entry `S`; tonal function below is **computed**, not annotated — the head note of each entry compared against the first entry (mod 7). Of 10 entries: **5 at subject (tonic) level**, **2 at answer (dominant) level**, 3 at other transposition levels (middle entries in related keys), which this method leaves undetermined rather than guessing.

## Rejected candidates (Dezrann `ignore`)

3 spans the analyst explicitly considered and declined to count as entries. Retained as near-miss evidence, excluded from accepted totals and from default exports.

| where | span | would-be type | voice |
| --- | --- | --- | --- |
| m19 b3 | 10 | countersubject 2 — incomplete | voice_2 |
| m47 b3.5 | 19/2 | countersubject 2 — incomplete, varied | voice_3 |
| m53 b3.25 | 39/4 | countersubject 1 — tail | voice_1 |

## Warnings

None.
