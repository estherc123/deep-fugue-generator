# Normalization — Progress

**Last updated:** 2026-08-03 (UTC)
**Pipeline stage:** Normalization (Spec §10) — **full 187-piece corpus complete**
**Machine-readable status:** [`normalization_status.json`](normalization_status.json)
**Inventories:** [`canonical_fugues.csv`](canonical_fugues.csv) · [`canonical_chorales.csv`](canonical_chorales.csv)

The **entire real-Bach corpus** is now parsed to the canonical rational-time score
model (`schemas/canonical_score.schema.json`), schema-validated, and written to
`data/canonical/<piece_id>.json`: 48 WTC fugues + 143 BCFB chorale variants (139
unique chorales). Every count below is derived from those files.

## Headline — whole corpus

| Metric | Fugues | Chorales | Total |
| --- | ---: | ---: | ---: |
| Canonical files | 48 | 143 | **191** |
| Unique works | 48 | 139 | **187** |
| Bars (measures) | 2,955 | 2,888 | **5,843** |
| Note events | 54,240 | 39,344 | **93,584** |
| Validation / parse failures | 0 ✅ | 0 ✅ | **0** ✅ |
| Cross-format discrepancies (§10.2) | — | 0 ✅ | **0** ✅ |

191 canonical files cover 187 unique movements (the 4 extra are the a/b encoded
variants of BWV 10.07, 38.06, 161.06, 177.05, grouped under one `work_group_id`).

## Fugues — detail (48 files, 54,240 notes)

## Voice-count distribution (musicologically verified)

| Voices | Fugues | Sanity check |
| ---: | ---: | --- |
| 2 | 1 | BWV 855 (WTC I/10) — the *only* 2-voice fugue in the WTC ✅ |
| 3 | 26 | |
| 4 | 19 | |
| 5 | 2 | BWV 849 (WTC I/4) and BWV 867 (WTC I/22) — the two 5-voice fugues ✅ |

The voice count comes from the encoded contrapuntal parts, never inferred. Spot
checks confirm correctness: C♯-major fugue BWV 848 carries 7 sharps; BWV 873 is
in 12/16; BWV 855's incomplete final bar yields an exact fractional total length
(`247/2` quarter-notes), which the rational-time model represents precisely.

## Chorales — detail (143 variant files → 139 unique, 39,344 notes)

The BCFB MusicXML files are **full cantata movements**, not bare 4-part settings:
each has 4 vocal parts (all named "Voice") plus, in most, instrumental doublings
and a continuo. All **143 files carry exactly 4 vocal parts**, so the canonical
chorale is those 4 voices (SATB); instruments/continuo are excluded and recorded
in `normalization.excluded_parts` (262 non-vocal parts excluded across the set).

| Voices | Chorale variants |
| ---: | ---: |
| 4 (SATB) | 143 ✅ |

- **139 unique chorales** confirmed by grouping `work_group_id`; the 4 works with
  a/b variants (BWV 10.07, 38.06, 161.06, 177.05) each produce two variant files
  under one work id.
- **Cross-format validation (§10.2):** for every chorale, the MusicXML per-part
  note-count vector and total were compared against the translated **kern** and
  **MEI** encodings. **0 discrepancies** — the three encodings agree note-for-note
  in structure. MusicXML remains the canonical source.

## What the canonical model captures (Spec §8)

- **Exact rational time** — onset/duration as `"3/8"`-style strings in quarter
  units; never floats. Repeats expanded to performance order (none in WTC), with
  notated + unfolded measure indices and offset-in-measure preserved.
- **Full pitch spelling** — midi, step, alter, octave, pitch-class, diatonic step.
- **True voices** — one canonical voice per contrapuntal part, ordered highest→
  lowest mean pitch (`voice_1` = top). Staff kept separate from voice identity.
- **Divisi & chords preserved** — brief intra-voice splits (double-stops, cadential
  chords) are kept as note events; any resulting partial overlap is recorded as an
  *explained* `voice_partial_overlap` warning, so no overlap is unexplained (§10.4).
- **Global maps** — meter, key-signature (fifths), and tempo maps.
- **Content hashes** — exact `event_hash`/`core_event_hash`, transposition-invariant
  `transposition_melodic_hash`, `rhythm_hash`, and per-voice hashes (for later
  deduplication and split-leakage checks, §12). Hashing is deterministic and
  tie-aware: a note split across a barline hashes identically to an unsplit note.

## Validation gates enforced per piece (Spec §10.4, §21)

JSON-Schema conformance · no negative durations · no zero-duration non-grace notes
· MIDI in 0–127 (no out-of-bounds notes) · non-negative onsets · every monophonic
overlap explained. A piece failing any gate is written as `<piece>.invalid.json`
for inspection and excluded from the corpus (currently: none).

## Known limitations / not yet done

- **Round-trip validation** (canonical → MusicXML → canonical hash preservation,
  §10.3) is specified but not yet implemented; needs a canonical→MusicXML writer.
- **Figured bass** — BCFB's figured-bass annotations and the continuo line are
  preserved in the raw files but not yet attached to the canonical chorales;
  the continuo is currently excluded with the other non-vocal parts.
- **Parser-warning capture** is wired for both the Python `warnings` channel and
  music21 stderr; steady-state parses emit none. The fugue warnings are all
  explained divisi overlaps, not parser errors.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -m fugue_data.cli normalize --all             # both corpora
PYTHONPATH=src .venv/bin/python -m fugue_data.cli normalize --corpus chorales # just chorales
PYTHONPATH=src .venv/bin/python -m fugue_data.acquire.report                  # regenerate this folder
```

Normalization is deterministic (identical source + config ⇒ identical event
hashes) and restartable (unchanged pieces are skipped).
