# Annotation alignment — bach_wtc1_fugue_09 (BWV854)

**Source:** algomus · **map version:** 1.0.0 · **schema:** 1.0.0
**Score:** 3 voices, 116 quarter-notes, 29 bars
**Voice mapping:** `staff_index_top_down` (Dezrann staff *N* → canonical `voice_N`, both ordered highest-first)
**Alignment evidence:** voice-attributed labels land on sounding notes 92.9% of their span on average (worst 88.9%). A wrong staff→voice mapping would collapse this figure.

## Accepted annotations

20 labels — cadence 3, countersubject 5, subject 12.

| where | span | type | voice | entry function | coverage |
| --- | --- | --- | --- | --- | --- |
| m1 b2.5 | 5/2 | subject | voice_2 | subject | 90% |
| m2 b1.25 | 15/4 | countersubject 1 | voice_2 | — | 100% |
| m2 b2.5 | 5/2 | subject | voice_1 | answer | 90% |
| m3 b4.5 | 5/2 | subject | voice_3 | subject | 90% |
| m6 b4.5 | 5/2 | subject | voice_1 | undetermined | 90% |
| m7 b3.25 | 15/4 | countersubject 1 | voice_1 | — | 100% |
| m7 b4.5 | 5/2 | subject | voice_2 | answer | 90% |
| m8 b3 | — | cadence — V:rIAC | — | — |  |
| m9 b4.5 | 5/2 | subject | voice_3 | answer | 90% |
| m16 b2.5 | 5/2 | subject | voice_2 | subject | 90% |
| m17 b1 | — | cadence — vi:rIAC | — | — |  |
| m19 b2.5 | 5/2 | subject | voice_3 | subject | 90% |
| m20 b1.25 | 15/4 | countersubject 1 | voice_3 | — | 100% |
| m20 b2.75 | 9/4 | subject | voice_1 | undetermined | 89% |
| m21 b1.25 | 15/4 | countersubject 1 — varied | voice_1 | — | 100% |
| m21 b2.5 | 5/2 | subject | voice_2 | subject | 90% |
| m25 b1.25 | 15/4 | countersubject 1 — varied | voice_2 | — | 100% |
| m25 b2.5 | 5/2 | subject | voice_1 | undetermined | 90% |
| m28 b1.5 | 5/2 | subject | voice_3 | undetermined | 90% |
| m29 b1 | — | cadence — I:PAC | — | — |  |

### Derived subject / answer

Algomus labels every entry `S`; tonal function below is **computed**, not annotated — the head note of each entry compared against the first entry (mod 7). Of 12 entries: **5 at subject (tonic) level**, **3 at answer (dominant) level**, 4 at other transposition levels (middle entries in related keys), which this method leaves undetermined rather than guessing.

## Rejected candidates (Dezrann `ignore`)

18 spans the analyst explicitly considered and declined to count as entries. Retained as near-miss evidence, excluded from accepted totals and from default exports.

| where | span | would-be type | voice |
| --- | --- | --- | --- |
| m3 b1.25 | 15/4 | countersubject 1 — in_episode | voice_1 |
| m4 b3.25 | 15/4 | countersubject 1 — in_episode | voice_3 |
| m5 b3.25 | 15/4 | countersubject 1 — in_episode, varied | voice_3 |
| m6 b3.25 | 15/4 | countersubject 1 — in_episode, varied | voice_3 |
| m8 b3.25 | 15/4 | countersubject 1 — in_episode | voice_2 |
| m10 b1.25 | 15/4 | countersubject 1 — incomplete | voice_1 |
| m11 b1.25 | 15/4 | countersubject 1 — in_episode | voice_3 |
| m12 b1.25 | 15/4 | countersubject 1 — in_episode | voice_3 |
| m13 b1.25 | 15/4 | countersubject 1 — in_episode | voice_1 |
| m14 b1.25 | 15/4 | countersubject 1 — in_episode | voice_1 |
| m15 b1.25 | 15/4 | countersubject 1 — in_episode, varied | voice_1 |
| m17 b1.25 | 15/4 | countersubject 1 — in_episode | voice_2 |
| m18 b1.25 | 15/4 | countersubject 1 — in_episode | voice_2 |
| m22 b1.25 | 15/4 | countersubject 1 — in_episode | voice_2 |
| m23 b1.25 | 15/4 | countersubject 1 — in_episode | voice_2 |
| m24 b1.25 | 15/4 | countersubject 1 — in_episode | voice_2 |
| m26 b1.25 | 15/4 | countersubject 1 — in_episode | voice_1 |
| m27 b1.25 | 15/4 | countersubject 1 — in_episode, varied | voice_1 |

## Warnings

None.
