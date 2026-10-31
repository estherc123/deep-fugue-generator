# Deduplication and dataset splits

**Splits version:** 1.0.0 · **seed:** 20260728 · **pieces:** 191 · **split units:** 185

**Leakage gates (§13.5):** ✅ all pass

## Deduplication (§12)

All six levels are computed. *Blocking* levels fuse their members into one split unit so the group cannot straddle a split; the others are similarity signals, reported but not fused.

| Level | Match groups | Pieces | Blocking |
| --- | ---: | ---: | :---: |
| source_checksum | 1 | 2 | ✅ |
| canonical_event_hash | 4 | 8 | ✅ |
| transposition_melodic_hash | 4 | 8 | ✅ |
| per_voice_melody_hash | 4 | 8 | — |
| rhythm_hash | 5 | 10 | — |
| work_identity | 4 | 8 | ✅ |

### 6 duplicate groups

Nothing is deleted: every member keeps its provenance, and one preferred representative is named per group.

| Group | Members | Preferred | Matched at |
| --- | --- | --- | --- |
| `dup:0001` | bach_chorale_bwv10.07a, bach_chorale_bwv10.07b | `bach_chorale_bwv10.07a` | rhythm_hash, work_identity |
| `dup:0013` | bach_chorale_bwv117.04, bach_chorale_bwv117.09 | `bach_chorale_bwv117.04` | canonical_event_hash, per_voice_melody_hash, rhythm_hash, source_checksum, transposition_melodic_hash |
| `dup:0016` | bach_chorale_bwv125.06, bach_chorale_bwv72.06 | `bach_chorale_bwv125.06` | canonical_event_hash, per_voice_melody_hash, rhythm_hash, transposition_melodic_hash |
| `dup:0037` | bach_chorale_bwv161.06a, bach_chorale_bwv161.06b | `bach_chorale_bwv161.06a` | work_identity |
| `dup:0044` | bach_chorale_bwv177.05a, bach_chorale_bwv177.05b | `bach_chorale_bwv177.05a` | canonical_event_hash, per_voice_melody_hash, rhythm_hash, transposition_melodic_hash, work_identity |
| `dup:0100` | bach_chorale_bwv38.06a, bach_chorale_bwv38.06b | `bach_chorale_bwv38.06a` | canonical_event_hash, per_voice_melody_hash, rhythm_hash, transposition_melodic_hash, work_identity |

## Splits (§13)

### `bcfb` — 137 split units, 13 strata

| Split | Target works | Assigned works | Pieces |
| --- | ---: | ---: | ---: |
| train | 111 | 111 | 114 |
| validation | 14 | 14 | 14 |
| test | 14 | 14 | 15 |

Stratified by: `mode`, `meter`, `length_bucket`.

<details><summary><code>mode</code> balance</summary>

| value | train | validation | test |
| --- | ---: | ---: | ---: |
| major | 62 | 4 | 9 |
| minor | 52 | 10 | 6 |

</details>

<details><summary><code>meter</code> balance</summary>

| value | train | validation | test |
| --- | ---: | ---: | ---: |
| 12/8 | 0 | 0 | 1 |
| 2/2 | 1 | 0 | 0 |
| 3/4 | 12 | 1 | 3 |
| 4/4 | 101 | 13 | 11 |

</details>

<details><summary><code>length_bucket</code> balance</summary>

| value | train | validation | test |
| --- | ---: | ---: | ---: |
| long | 30 | 1 | 6 |
| medium | 43 | 2 | 4 |
| short | 41 | 11 | 5 |

</details>

### `wtc` — 48 split units, 38 strata

| Split | Target works | Assigned works | Pieces |
| --- | ---: | ---: | ---: |
| train | 36 | 36 | 36 |
| validation | 6 | 6 | 6 |
| test | 6 | 6 | 6 |

Stratified by: `book`, `mode`, `voice_count`, `meter`, `length_bucket`.

<details><summary><code>book</code> balance</summary>

| value | train | validation | test |
| --- | ---: | ---: | ---: |
| 1 | 19 | 3 | 2 |
| 2 | 17 | 3 | 4 |

</details>

<details><summary><code>mode</code> balance</summary>

| value | train | validation | test |
| --- | ---: | ---: | ---: |
| major | 17 | 3 | 4 |
| minor | 19 | 3 | 2 |

</details>

<details><summary><code>voice_count</code> balance</summary>

| value | train | validation | test |
| --- | ---: | ---: | ---: |
| 2 | 1 | 0 | 0 |
| 3 | 18 | 4 | 4 |
| 4 | 15 | 2 | 2 |
| 5 | 2 | 0 | 0 |

</details>

<details><summary><code>meter</code> balance</summary>

| value | train | validation | test |
| --- | ---: | ---: | ---: |
| 12/16 | 1 | 0 | 0 |
| 2/2 | 6 | 1 | 0 |
| 2/4 | 1 | 0 | 1 |
| 3/2 | 0 | 0 | 1 |
| 3/4 | 5 | 0 | 0 |
| 3/8 | 2 | 1 | 0 |
| 4/2 | 0 | 1 | 0 |
| 4/4 | 17 | 2 | 4 |
| 6/16 | 1 | 0 | 0 |
| 6/4 | 1 | 0 | 0 |
| 6/8 | 1 | 1 | 0 |
| 9/8 | 1 | 0 | 0 |

</details>

<details><summary><code>length_bucket</code> balance</summary>

| value | train | validation | test |
| --- | ---: | ---: | ---: |
| long | 10 | 3 | 2 |
| medium | 12 | 2 | 2 |
| short | 14 | 1 | 2 |

</details>

## Leakage gates (§13.5)

- ✅ No `work_group_id` appears in multiple splits
- ✅ No exact or transposition-normalized duplicate crosses splits
- ✅ Every split unit is wholly within one split
- ✅ Every `assignment_hash` verifies against seed, unit and members
- ✅ Per-corpus work counts match the configured targets

Excerpt, transposition and synthetic-derivative gates (§13.1) apply to stages not yet built; every derivative inherits `split` from its parent piece through `split_unit_id`.
