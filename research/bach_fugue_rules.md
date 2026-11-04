# Operational rules of Bach's four-voice fugue style

A rule catalogue that a hard checker can implement. Input: a note grid with voices ordered highest first (V0 = soprano, V1 = alto, V2 = tenor, V3 = bass), onsets and durations in quarter notes, MIDI pitches, known meter and local key.

Each rule has an ID, a **definition** you can code, a **severity**, the **exceptions** (with Bach's own practice where it is documented), and the source.

Severity levels:

- **F (forbidden):** a hard reject. Bach's rate is about 0 per fugue, or the only exceptions are narrow and codable.
- **A (avoid):** count it and penalize it, and reject above a threshold. Bach does it occasionally in specific contexts.
- **P (preference):** a stylistic statistic. Use it for scoring, never for rejection.

> Calibration note. The project target is at most one violation per five fugues. Luke Dahn's census of the chorales found about **one "syntactical" consecutive 5th/8ve per 224 measures**. Excluding the cadential-anticipation type, it is about one per 350 measures (5,600+ measures in total; see §1.1). A 70-measure four-voice fugue written at Bach's chorale rate would therefore contain about 0.2–0.3 such consecutives. So the parallels rule must be a hard F rule with only the narrowly coded exceptions below. A checker that adds the exceptions back generously would let through more than Bach himself wrote.

---

## 0. Preliminaries: shared definitions used by every rule

**Slices.** Collect the sorted set of every onset time in every voice. A *slice* is the interval [t_k, t_{k+1}). In each slice, every voice is either sounding one pitch or resting. A note *attacks* in slice k if its onset is t_k. Otherwise it is *held* (tied over) in that slice.

**Motion between consecutive slices (k → k+1) for a voice pair (i, j), both sounding in both slices:**

- *Parallel/similar*: both pitches change in the same direction.
- *Contrary*: both change, in opposite directions.
- *Oblique*: exactly one of them changes.
- *Static*: neither changes.

"Changes" means the MIDI pitch differs. A re-attacked note at the same pitch does not count as motion for parallel detection.

**Harmonic interval.** `h = p_upper − p_lower` in semitones, normally ≥ 0 when voices are not crossed. Interval class is `h mod 12`. Perfect unison/octave means `h mod 12 == 0`. Perfect fifth means `h mod 12 == 7`. Tritone means `h mod 12 == 6`.

**Diatonic (generic) interval.** Several rules need spelling: A2 vs m3, d4 vs M3, d5 vs A4, d7 vs M6, and so on. Spell each pitch against the local key:

1. Map each pitch class to a scale degree using the local key's diatonic collection. In minor, use the union of natural, harmonic and melodic minor, so ♭6/♮6 and ♭7/♯7 both have a degree.
2. Spell chromatic pitches by the direction of resolution: an ascending chromatic note is spelled as a raised degree, a descending one as a lowered degree. If that fails, use the harmonic-minor / secondary-leading-tone preference: a sharp is the leading tone of the following chord root.
3. The generic interval is the difference in letter steps. The quality is the semitone count compared with the generic interval.

Without spelling, use these fallback semitone tests:

- A2 = 3 semitones between scale degrees ♭6 and ♯7 in minor.
- A4/d5 = 6 semitones.
- d7 = 9 semitones between ♯7 and ♭6 (upper) in minor.
- d4 = 4 semitones between ♯7 and ♭3 in minor.

**Lowest sounding voice (LSV).** The lowest pitch sounding in the slice. Normally V3, but V2 when the bass rests. All "against the bass" rules use the LSV.

**Metric strength** (use the project's meter grid). Let *b* be the beat unit.

| Meter | Strongest | Strong | Weak | Weaker |
|---|---|---|---|---|
| 4/4 (C) | beat 1 | beat 3 | beats 2, 4 | half-beat offbeats, then sixteenth subdivisions |
| alla breve (2/2) | beat 1 | beat 2 (half notes) | quarter offbeats | |
| 3/4 | beat 1 | | beats 2, 3 | |
| 6/8, 12/8, 9/8 | dotted-quarter beats | | eighth subdivisions | |

In 6/8 and 12/8, beat 1 is strongest and the middle beat is strong.

Define `accented(t, level)`: an onset is accented *at level L* if it falls on a grid point of level L and is stronger than its immediate neighbours at the next finer level. In Bach's sixteenth-note textures, apply most dissonance rules at the **beat level and the half-beat level**. Apply them only loosely at the sixteenth level, where passing dissonance is free (see §3.0).

**Consonance (two-voice, against the LSV).**

- Perfect: P1, P8, P5 (ic 0, 7).
- Imperfect: m3, M3, m6, M6 (ic 3, 4, 8, 9), when spelled as 3rds/6ths.
- Dissonant: m2, M2, P4, A4/d5, m7, M7 (ic 1, 2, 5, 6, 10, 11), plus every augmented or diminished interval (A2, d4, d7, A5, A6).

**Between two upper voices**, P4 (ic 5) is consonant, as are A4 and d5 when both notes belong to a V7 or vii°(7) chord over a consonant bass, and 6/4 sonorities are allowed as discussed in §3.

**Harmonic reduction.** Many rules need "the chord" at each beat. Use the project's harmony labels if they exist. Otherwise:

1. For each beat (or half-beat in fast harmonic rhythm), choose the triad or seventh chord (with root) that covers the most note-duration, weighting accented and bass notes.
2. A note is a *chord tone* (CT) if its pitch class belongs to that chord. Otherwise it is a *non-chord tone* (NCT).

Every NCT must then be classifiable under §3.

Sources for these conventions: Aldwell & Schachter, *Harmony and Voice Leading* (4th ed., 2011); Gauldin, *A Practical Approach to 18th-Century Counterpoint* (1988, Waveland 1995/2013), chs. 1–8; Kennan, *Counterpoint* (4th ed., 1999), chs. 2–5.

---

## 1. Voice-leading rules between voice pairs

Unless stated otherwise, check every pair (i, j), i < j, of the 6 pairs.

### VL-1 Parallel perfect octaves / unisons, **F**

- **Definition.** For consecutive slices k → k+1 where both voices sound and **both pitches change**: `h_k mod 12 == 0` and `h_{k+1} mod 12 == 0`. This includes P1→P1, P8→P8, P8→P15 and P1→P8. It covers similar **and** contrary motion; contrary-motion octaves (P1→P8, P8→P15) are "consecutive octaves" too.
- **Rests.** If either voice rests in slice k+1 but the next attacked notes still form the pattern within ≤ 1 beat with no intervening note in either voice, count it too. This catches octaves across a short rest.
- **Exceptions.**
  1. A voice that deliberately doubles another for longer than one beat (keyboard octave doubling of a bass line, the organ-pedal convention) is not independent. It should not occur in a strict 4-voice grid, so treat it as a violation here.
  2. Across a phrase boundary: the previous slice is the final chord of a cadence (with a fermata or a rest of ≥ 1 beat in some voice). Dahn counts 29 chorale consecutives of this kind, most of them by contrary motion. Fugues have no fermatas, so allow this only right after a cadence onto a strong beat followed by a section boundary.
  3. Dahn finds octaves rarer and more objectionable than fifths, and finds no syntactical octaves between the outer voices. Keep this rule F with no exceptions for V0–V3.
- **Source.** Fux, *Gradus ad Parnassum* (1725; trans. Mann, *The Study of Counterpoint*, 1965), first species. Dahn, "Consecutive Fifths & Octaves in the Bach Chorales", https://bach-chorales.com/ConsecutivesInChorales.htm. Aldwell & Schachter ch. 7.

### VL-2 Parallel perfect fifths (incl. contrary / "consecutive" fifths, P5→P12), **F**

- **Definition.** As VL-1 with `h mod 12 == 7` in both slices, both voices changing pitch, in any direction (similar or contrary).
- **Exceptions, coded narrowly** (each must be matched exactly):
  - **E-cad, the cadential anticipation fifths** (Dahn's 9 "cadential" cases). In V(7)→I at a cadence, the soprano anticipates the tonic (re–do–do with the anticipation on a weak subdivision) while an inner voice moves sol–fa (the delayed 7th of V7). Allow only when the upper of the two notes is an anticipation (§3.6), the lower is a passing 7th, and the pair is not V0–V3.
  - **E-ornament, non-structural fifths.** The fifth exists only because one of its four notes is an NCT (passing, neighbour, escape) on a sub-beat-level weak position, and the structural (beat-level) intervals are not P5→P5. Dahn found 14 non-structural cases in more than 400 chorales. Even so, the Breitkopf editors "corrected" many of them, which suggests they sound worse than structural fifths. **Recommendation: do not allow this for the generator.** Allow it only E-cad-style, at most one per piece, and only between inner voices.
  - **Contrary fifths across a phrase boundary**, as in VL-1 exception 2.
- **Bach figures.** Dahn (Riemenschneider set, more than 400 chorales, more than 5,600 measures):
  - 54 consecutive 5ths/8ves in total: 29 at fermatas, 14 non-structural, 11 structural.
  - Of the 11 structural ones: 6 contrary, 4 parallel.
  - Only **one** instance (BWV 306) of stepwise chordal consecutives in parallel motion.
  - **No syntactical consecutives between soprano and bass.** Every one of them involves an inner voice.
  - The wtfbach blog (https://wtfbach.substack.com/p/parallel-fifths-in-bach) mentions "more than 40" chorale fifths and "dozens upon dozens" in the instrumental works, but publishes no systematic count for the WTC. There is no published census of parallels in the WTC fugues; the project could compute one from its corpus.
- **Source.** Dahn (above). Fux (above). Fang, Liu, Cuthbert et al., "Bach or Mock? A Grading Function for Chorales in the Style of J.S. Bach" (arXiv:2006.13329, 2020). That paper counts P1/P5/P8 "in similar and contrary motion" relative to Bach's error-to-note ratio, and finds that parallelisms are the weakest feature of neural chorale generators.

### VL-3 Accented (off-beat-separated) parallels, **A**

- **Definition.** Take two voices on **consecutive accented beats** (beat level, e.g. beats 1 and 2 in 4/4, or 1 and 3 in a half-note pulse) that form P5→P5 or P8→P8 with both voices moving. Count it when the notes between them are only NCTs, or are notes lasting < one beat that do not form a different consonance with the other voice. A suspension placed between them does **not** cancel the parallel. Classic examples are 6–5 chains and "5ths hidden behind a 9–8".
- **Exception.** One intervening chord tone lasting ≥ ½ beat at a different interval (for example a 5–6 motion) breaks the parallel. This is the standard 5–6 technique.
- **Source.** Kennan ch. 3–4 (second species: "fifths on successive downbeats"). Fux, second and fourth species ("a suspension does not remove consecutive fifths"). Gauldin ch. 4.

### VL-4 Hidden / direct 5ths and 8ves, **A for outer voices, P otherwise**

- **Definition.** Both voices move in the **same direction** (k→k+1) into `h_{k+1} mod 12 ∈ {0, 7}`, and `h_k mod 12 ∉ {0, 7}` (otherwise it is VL-1/VL-2).
  - **Flag** only when the pair is **V0–V3** (outer voices) **and the upper voice moves by leap** (> 2 semitones).
- **Allowed.** The soprano moves by step. This covers the standard cadence: bass 5→1 against soprano 2→1 or 7→1. Hidden intervals between any pair involving an inner voice are allowed. Hidden intervals in contrary motion are not hidden intervals at all.
- **Severity nuance.** Direct octaves between outer voices with a soprano leap are much more audible than direct fifths. Make **direct 8ve (outer, soprano leaps) = A-strong**, effectively F for the generator, and **direct 5th (outer, soprano leaps) = A**.
  - Exception: the bass and soprano both leap within the same chord (an arpeggiation change of position, with no harmony change). This is common in Bach's keyboard textures and can be allowed.
- **Source.** Aldwell & Schachter ch. 7. Chorale guide, https://choraleguide.com/vl-parallels.php ("applies only to the outer parts … acceptable when the soprano moves by step"). Wikipedia, "Consecutive fifths" (hidden fifths allowed "when the interval doesn't involve both outer voices … when the upper part moves by step"). David Huron, *Voice Leading: The Science behind a Musical Art* (MIT Press, 2016), ch. "Direct Intervals Revisited", explains the outer-voice restriction by the perceptual salience of the outer voices.

### VL-5 Unequal fifths, **A / F by case**

- **d5→P5.** Detect `h_k mod 12 == 6` (spelled as a d5, i.e. upper = fa, lower = ti, or a secondary equivalent), then `h_{k+1} mod 12 == 7`, with both voices moving in similar motion.
  - **F when the bass (LSV) is involved.**
  - **A between upper voices.** The textbook allowance (Aldwell & Schachter) is d5→P5 between upper voices in **I – V4/3 – I6**: the soprano rises fa→sol in parallel tenths with the bass re→mi while an inner voice goes ti→do.
- **P5→d5.** Allowed (P), provided the d5 resolves normally afterwards: the upper note down by step, the lower up by step. Some theorists forbid it against the bass. Bach uses it freely, e.g. a passing V6/5 or vii°6.
- **Source.** Aldwell & Schachter ch. 7. Wikipedia "Consecutive fifths" ("some theorists prohibit only one direction, or only when the bass is involved"). Chorale guide (I–V4/3–I6 exception).

### VL-6 Voice crossing, **A** (F for the bass and for long crossings)

- **Definition.** In any slice, `p(V_i) < p(V_j)` for i < j, with both sounding.
- **Severity by case.**
  - F if it involves V3 (the bass goes above the tenor) for any duration more than an eighth note, or if any crossing lasts > 1 beat.
  - A if it is a brief crossing (≤ 1 beat) between inner voices or between soprano and alto.
  - In stretto and in the exposition, a crossing at the **entry** of a subject is F, because it obscures the entry.
- **Bach figures.** Huron, "The Avoidance of Part-Crossing in Polyphonic Music: Perceptual Evidence and Musical Practice", *Music Perception* 9/1 (1991): 93–104. Across 105 polyphonic works by Bach, "a marked reluctance to have parts cross — even when the effects of pitch distribution are controlled". Crossings do occur in the keyboard fugues, mostly inner voices for a few notes. A per-piece rate should come from the project's own corpus; from experience, crossings appear in a minority of WTC fugues and are almost always short.
- **Also.** Keyboard-unison collisions, where two voices strike the same pitch, are allowed (P) as a single sounding note. Parallel unisons are still VL-1.

### VL-7 Voice overlap, **A**

- **Definition.** Moving k→k+1, a lower voice moves **above** the pitch the upper adjacent voice had in slice k (`p_{j}(k+1) > p_{i}(k)` for adjacent i < j). The mirror case also counts: an upper voice moves below the previous pitch of the voice beneath it (`p_i(k+1) < p_j(k)`).
- **Exception.** Overlap with a rest in between (the voice re-enters) is fine. Overlap within an arpeggiated keyboard figure is P.
- **Source.** OMT / Iowa State intmus course, https://intmus.github.io/inttheory/10-intro-part-writing/a2-voiceleadingerrors.html. Aldwell & Schachter ch. 4.

### VL-8 Spacing, **A** (choral norm) / **F** (keyboard playability)

- **Choral norm (A).**
  - |V0 − V1| ≤ 12 semitones.
  - |V1 − V2| ≤ 12 semitones.
  - |V2 − V3| ≤ 19 semitones (a 12th). Larger gaps are tolerated in the bass.
  - Apply this on beat-level onsets. Brief excursions inside a running figure are allowed.
- **Keyboard playability (F for WTC-style output).** Every slice's sounding pitches must be splittable into two hands, each spanning ≤ 15 semitones (a minor/major 10th is about the practical upper limit; use 16 as the absolute maximum) with ≤ 5 notes per hand. Also prefer to keep the soprano within ~1 octave of the alto for more than 90% of beats.
- **Bach figures.** Aarden & von Hippel, "Rules for Chord Doubling (and Spacing): Which Ones Do We Need?", *Music Theory Online* 10/2 (2004). In 2,643 chorale triads compared with random ones, Bach's chords put the **largest gap between the two lowest parts** significantly more often than chance.
- **Source.** UVM part-writing handout, https://www.uvm.edu/~dfeurzei/110/materials/partwriting.pdf ("S/A, A/T within an 8ve; B/T normally within a 12th"). intmus page above ("upper voices should never span more than an octave" in strict SATB).

### VL-9 Cross relation (false relation), **A**

- **Definition.** Slice k contains pitch class x in voice i. Slice k+1 (or any slice within ½ beat) contains the **chromatic alteration of the same letter** (x ± 1 semitone, same scale degree spelled differently, e.g. F♮ vs F♯) in a **different** voice j ≠ i, and voice i does not itself move x → x±1.
- **Allowed.**
  1. The chromatic step happens within one voice (chromatic line).
  2. In minor, ♮6/♭6 and ♮7/♭7 alternating between voices when one is a descending melodic-minor line (♭7–♭6) and the other ascending (♯6–♯7). Bach does this constantly in minor.
  3. The "Phrygian/augmented-6th" approach to V.
  4. Descending chromatic lament basses.
- **Simultaneous cross relation** (both forms sounding at once): F, except for the augmented-6th/diminished-3rd chord and a passing-tone clash lasting ≤ 1 sixteenth.
- **Source.** Aldwell & Schachter ch. 26 (mixture/chromatic). Gauldin ch. 15 (chromaticism). Kennan ch. 5.

### VL-10 Simultaneous leaps, **P/A**

- **Definition.**
  - (a) All sounding voices leap (> 2 semitones) in the same direction at once. **A** unless the harmony is unchanged (arpeggiation of one chord).
  - (b) Two voices leap in similar motion into a perfect interval. This is VL-4. When neither is the outer pair, still A if both leaps are ≥ P4.
  - (c) Two voices leap simultaneously in opposite directions by ≥ an octave in total. **P**.
- **Source.** Kennan ch. 2 ("avoid simultaneous leaps in the same direction, especially to perfect intervals"). Huron 2016 (onset synchrony and similar motion promote fusion of the voices).

### VL-11 Avoid fusion: perfect consonances with synchronous onsets, **P**

Huron (*Music Perception* 19/1, 2001, "Tone and Voice", and the 2016 book) shows that Bach avoids **synchronous onsets of unisons, octaves and fifths** more than other intervals. This promotes the independence of the voices.

Use it as a score: the proportion of beat-level slices in which two voices attack together into P1/P8/P5. It is not a rule.

---

## 2. Melodic rules per voice

Check every voice's successive attacked pitches. Skip across rests ≤ 1 beat; a longer rest resets the melodic context.

### ML-1 Forbidden intervals, **F**

| Interval | Semitones / spelling | Severity | Bach exceptions |
|---|---|---|---|
| Augmented 2nd | 3 st, spelled as a 2nd (♭6→♯7 in minor, either direction) | F | Very rare in Bach, and then in expressive chromatic subjects or recitative-like writing. Forbid for the generator. |
| Augmented 4th | 6 st, spelled as a 4th (fa→ti up, ti→fa down) | A (F if not recovered) | Appears in bass lines when the line continues stepwise in the resolving direction (fa→ti→do). |
| Augmented 5th, augmented intervals in general | | F | |
| Diminished 5th | 6 st, spelled as a 5th | A, allowed if the next note moves by step **inside** the interval (ti↓fa→mi, or fa↑ti→do) | Common in Bach's bass and subjects. |
| Diminished 4th | 4 st (♯7→♭3 in minor) | P (allowed) | Characteristic of Bach's minor subjects (e.g. WTC I C♯ minor fugue subject, B♯→E). Must resolve: ♯7→1. |
| Diminished 7th | 9 st, ♭6↔♯7 in minor | P (allowed) | A hallmark of minor subjects (e.g. WTC II A minor, *Musical Offering*). Must change direction by step: ♯7→1 or ♭6→5. |
| Minor 7th | 10 st | A | Allowed in the bass when the line arpeggiates V7, or when an octave-leap substitute appears in compound melody. In upper voices only within arpeggiation of one chord, and it must turn back. |
| Major 7th | 11 st | F | |
| Leaps > octave | > 12 st | F upper voices; A bass (≤ a 10th, octave+step, within one harmony) | Bach's keyboard bass sometimes leaps a 10th. |
| Octave | 12 st | P (allowed) | Must change direction afterwards (ML-2). |

Source: Fux (first species melodic rules). Kennan ch. 2 ("Melodic line": avoid augmented intervals and 7ths; diminished intervals acceptable if followed by a step inward). Gauldin ch. 2 (allows d7/d4 in minor as Baroque idioms). Aldwell & Schachter ch. 4.

### ML-2 Leap recovery, **A** (P for small leaps)

- **Definition.** After a leap ≥ 5 semitones (P4 or more) the next melodic interval should be in the **opposite** direction. After a leap ≥ 8 semitones (m6, octave) the opposite motion should be **by step** or a smaller leap.
- **Flag.**
  - A leap ≥ m6 followed by motion in the same direction: A.
  - A leap ≥ m6 followed by a further leap in the same direction: F.
- **Exceptions.**
  - Arpeggiation of one chord (all notes chord tones of the same harmony): P.
  - Compound melody / implied polyphony, where a voice alternates between two registers. This is very common in Bach's solo and keyboard writing. Detect it with alternating-register patterns and exempt it.
- **Empirical nuance.** von Hippel & Huron, "Why Do Skips Precede Reversals? The Effect of Tessitura on Melodic Structure", *Music Perception* 18/1 (2000). Post-skip reversal is largely regression toward the middle of the range: a leap from the middle of the range is not reversed more than chance. The rule is therefore strongest for leaps that end near the edge of the voice's tessitura. Weight the penalty by distance from the median pitch.

### ML-3 Consecutive leaps in one direction, **A**

- **Definition.** Two or more successive leaps (> 2 st) in the same direction.
  - Flag if their combined span is a **dissonant** interval (ic 1, 2, 6, 10, 11, or > 12 st), unless every note belongs to one chord. Bach arpeggiates V7 and vii°7, which gives a 7th outline legitimately.
  - Flag if the combined span is > 12 st, unless the notes form a chord arpeggio of ≤ 15 st.
- **Source.** Kennan ch. 2 ("successive leaps in the same direction should outline a triad"). Fux.

### ML-4 Dissonant outline across a turning point, **P**

Gauldin and Kennan advise against outlining a tritone between a melodic high point and the following low point (fa…ti) when the line doesn't resolve. Use this as a score only.

### ML-5 Ranges, **F** (absolute) / **A** (per-voice)

**Absolute keyboard compass (F).** The WTC was written for a harpsichord/clavichord compass of roughly **C2–C6** (MIDI 36–84). Allow C2–D6 at most (MIDI 36–86); Book II reaches slightly higher in a few places. Organ fugues: manual C2–C6, pedal C2–D4 (MIDI 36–62).

**Per-voice ranges (A when exceeded by more than 2 semitones for more than one beat).** Recommended values come in two sets.

Choral SATB, for vocal-style fugues (Aldwell & Schachter ranges):

| Voice | Range | MIDI |
|---|---|---|
| S | C4–G5 | 60–79 |
| A | G3–D5 | 55–74 |
| T | C3–G4 | 48–67 |
| B | E2–C4 | 40–60 |

Keyboard four-voice fugues are typically wider. Approximate envelope (from WTC practice; **verify against the corpus**):

| Voice | Range | MIDI |
|---|---|---|
| S | C4–C6 | 60–84 |
| A | F3–G5 | 53–79 |
| T | C3–C5 | 48–72 |
| B | C2–E4 | 36–64 |

- **Implementation advice.** These sources give only textbook ranges. Compute each voice's 1st and 99th percentile pitch from the project's own Bach four-voice keyboard fugue corpus, and use those as the A thresholds. Use the corpus minimum and maximum (± 1 semitone) as the F thresholds.
- **Tessitura (P).** The median pitch of each voice over the fugue should fall within the central 50% of that voice's corpus distribution. A voice that sits for more than 4 measures in the top or bottom 10% of its range is P-penalized.

### ML-6 Repeated notes, **P**

- Repeated pitches are idiomatic in Bach subjects (repeated-note heads, e.g. WTC I D major or the *Art of Fugue* rhythmic variants). Do not forbid them.
- Flag **A** for more than 4 consecutive re-attacks of the same pitch in an inner voice outside a motivic (subject/countersubject) statement.
- Flag **A** for a re-attack across a strong beat that forms a dissonance with no suspension treatment (see §3.4).
- The "Bach or Mock" grader flagged an "abnormally long sequence of repeated quarter notes" as a typical generator flaw.

### ML-7 Melodic chromaticism, **A**

- A chromatic step (1 st, same letter) must continue in the same direction: up for ♯, down for ♭.
- Chromatic lines of ≥ 3 semitones in one voice are allowed (lament bass, chromatic subjects).
- Avoid a chromatic semitone immediately followed by a leap in the same direction.

---

## 3. Dissonance treatment

### 3.0 Procedure

For every note that is dissonant (§0) against the LSV or against any other voice in any slice where it sounds, determine its role:

- a chord tone of a seventh chord (chordal 7th, 9th),
- or one of the NCT types below.

**Unclassifiable dissonances are F.** This is the single most important hard rule after parallels.

The check should run at **two metric levels**:

1. **Beat level.** On every accented beat onset, every dissonance must be a suspension, an appoggiatura, an accented passing tone, a chordal 7th (prepared or properly approached), or a pedal-point dissonance.
2. **Subdivision level** (eighths and sixteenths). Unaccented dissonances must be passing, neighbour, anticipation, escape, cambiata, or a member of an arpeggiated chord (implied harmony).

In Bach's figuration a sixteenth note may leap to a dissonance that is a chord tone of the *implied* harmony (compound melody). Classify by the beat's harmonic reduction, not only by the instantaneous slice.

Source: Gauldin 1988, chs. 3–8 (his "modified species" for 18th-century counterpoint). Gauldin, "Fux to Bach" (contrasts Fux's strict stepwise control with Baroque accented dissonance and appoggiaturas). Kennan chs. 3–5.

### 3.1 Passing tone (PT), **allowed**

- Approached by step and left by step **in the same direction**, filling a 3rd (diatonic or chromatic) between two consonances or chord tones.
- *Unaccented PT*: on a weaker position than the note it resolves to.
- *Accented PT*: on a beat or strong subdivision. Allowed in Bach, mostly descending, typically at the eighth or sixteenth level. On the strongest beat of the bar, treat it as **A** if it lasts ≥ one beat.
- Two consecutive PTs filling a 4th (e.g. 5–4–3–2–1 over one harmony) are allowed.

### 3.2 Neighbour tone (NT), **allowed**

- Step away and step back to the **same pitch**.
- Unaccented in normal use. Accented (on the beat) counts as an appoggiatura (§3.5).
- *Double neighbour* (upper and lower in succession: 1–2–7–1) is allowed (P).
- *Incomplete neighbour* = escape tone or appoggiatura.

### 3.3 Suspension (SUS), **allowed only in this exact form**

Three stages:

1. **Preparation (P).** The same voice has the **same pitch**, consonant with the other voice, in the preceding slice. It is tied over or re-struck. Re-striking is common in Bach's keyboard writing; treat both as valid.
   - The preparation should last ≥ the duration of the dissonance in strict style. **A** if shorter; Bach relaxes this in fast figuration.
2. **Suspension (S).** It falls on an **accented** position: beat or strong half-beat. The other voice has moved, making a dissonance.
3. **Resolution (R).** **Down by step** (1 or 2 st) to a consonance on a weaker position. Resolution timing: it occurs within the duration of one beat or the next weaker metric unit.

**Permitted suspension types** (generic interval against the voice it is dissonant with, usually the bass):

| Type | Where | Status | Notes |
|---|---|---|---|
| **4–3** | upper voice over bass | P (allowed) | The note of resolution (the 3rd) must not sound in another voice simultaneously with the 4th. The bass may hold it as a root/octave. |
| **7–6** | upper over bass | P (allowed) | Chains of 7–6 are idiomatic. Do not sound the 6th elsewhere during the 7th. |
| **9–8** | upper over bass | P (allowed) | Compound only. A 2–1 into a unison is A. The octave of resolution is not doubled elsewhere. |
| **2–3** | bass suspended below an upper voice | P (allowed) | The **lower** voice is held and resolves **down** to make a 3rd. Equivalently, the inversion of 7–6. |
| 4–5 bass suspension | lower voice | A (rare) | |
| **6–5** | consonant "suspension" | P | Not a dissonance unless over a 6/4. |
| 9–10, 7–8 (rising, "retardation") | upper voice | A | Allowed when the suspended note is the leading tone rising to the tonic (7–8 retardation). Otherwise forbidden. |

Further rules:

- **Ornamented resolution (P).** Between S and R, the voice may insert a note: an anticipation of R, a lower neighbour (escape below), or a leap to a consonance and back to R. R must arrive at the expected time or within one subdivision of it.
- **Change of bass at resolution (P).** The bass may move as the suspension resolves (e.g. 7–6 becomes 7–3 over a new bass, or 9–6). This is allowed if the resolution note is consonant with the new bass.
- **Forbidden.**
  - Resolution by leap: F.
  - Upward resolution of a non-leading tone: F.
  - Resolution to a dissonance: F, except in a chain where it becomes the preparation of the next suspension and the resolution note itself is consonant.
  - Suspension that hides parallels (VL-3): A.

Sources: Fux fourth species. Kennan ch. 4. Gauldin chs. 5–6 (7–6, 4–3, 9–8, 2–3 as the standard 18th-century suspensions). Ithaca College double counterpoint handout, https://musictech.ithaca.edu/MusicTech/Form/Handouts/DblCpt.html.

### 3.4 Pedal point, **allowed**

- A sustained (or repeated) note, almost always in the bass, on degree 1 or 5.
- It begins as a consonance and ends as a consonance with the harmony above.
- Duration ≥ 1 measure.
- While it is held, dissonances *between the pedal and the upper voices* are exempt. The upper voices must still obey all rules among themselves, treating the voice just above the pedal as the "bass" for 4th/6-4 rules.
- In fugues: a dominant pedal before the final entries and a tonic pedal under the coda are idiomatic. WTC I C minor and many organ fugues use these.

### 3.5 Appoggiatura, **allowed (A on the downbeat)**

- **Leap** to an **accented** dissonance, resolved **by step** in the **opposite** direction (usually down) to a chord tone.
- Bach's figuration has many sixteenth-level accented appoggiaturas. At beat level, they mostly appear as the unprepared 4th or 7th in a cadential gesture.
- Allow at beat level if resolved within that beat. Penalize A if it lasts > one beat.
- Gauldin explicitly lists "freer use of accented melodic dissonance (appoggiaturas)" as a difference from Fux.

### 3.6 Anticipation, **allowed**

- An unaccented note that moves by step to the pitch of the **next** chord before that chord arrives, then is repeated or held into it.
- Typical at cadences: soprano re–do–(do).
- Never on a strong beat.
- This is the source of Bach's cadential fifths (E-cad in VL-2).

### 3.7 Escape tone (échappée), **allowed**

- Step to an **unaccented** dissonance, then **leap** (usually a 3rd) in the opposite direction to a consonance.
- Commonly an upper escape before descending.

### 3.8 Cambiata (Fux nota cambiata), **allowed**

- Four notes, starting on a consonance on a strong position:
  1. Step down to an **unaccented dissonance** (second note).
  2. Leap down a **3rd** to a consonance.
  3. Step up.
- Bach also uses the inverted and "double-passing" variants. Treat them as P.

### 3.9 Chordal 7th (and 9th), **F if mistreated**

- The 7th of any seventh chord must be **resolved down by step** when the harmony changes. The resolution may be in the same voice after an ornamental figure, or transferred to another voice within the same beat; transfer is A.
- **Approach.** Preferably prepared (common tone) or by step (a passing 7th). Leaping into the 7th is A, except in V7 and vii°7 in Bach's figuration, where it is common: treat as P when the resolution is correct.
- **Exceptions.**
  - (a) V4/3 → I6 with the 7th rising in parallel 10ths with the bass (Aldwell & Schachter).
  - (b) The 7th held over as a common tone into the next chord, when that chord contains the same pitch (e.g. V7 → V4/2 → I6, where it resolves later).
  - (c) The 7th that moves to another chord member of the *same* harmony (arpeggiation) before resolving.

### 3.10 Leading tone (LT) resolution, **F in outer voices, A in inner voices**

- At a **V(7) → I** or **vii°(7) → I** progression (local key, including tonicizations), the LT (degree 7, or the raised 7 in minor):
  - in V0 (soprano) or V3 (bass), must move **up a semitone** to the tonic (or be held as part of a suspension that resolves up);
  - in an inner voice, may **fall to degree 5** (the "frustrated" LT), especially to complete the tonic triad at cadences. Bach does this routinely in chorales.
  - LT descending stepwise in a descending scale line (8–7–6–5) is allowed in any voice.
- Source: intmus (ti→do, with the exceptions "inner voice may fall to sol"; "in a stepwise descent re–do–ti–la–sol"). Aldwell & Schachter ch. 6.

### 3.11 Diminished 7th chord (vii°7, and secondary vii°7/x), **F if mistreated**

- Both tritones resolve inward or outward:
  - The root (LT) goes up a semitone.
  - The 7th (♭6 of the key, the "le") goes down a semitone to 5.
  - The chord's 5th (degree 4, fa) goes down by step to 3.
  - The chord's 3rd (degree 2, re) is free: up to 3 or down to 1.
- The diminished-7th may be approached by leap and need not be prepared; it is often the head of a minor subject.
- Melodic A2 inside it (♭6→♯7) is forbidden (ML-1). The d7 leap is allowed.
- **Unequal fifths in vii°7 → i.** When re lies below le (a d5) and they move re→do, le→sol, the result is d5→P5. This is acceptable between upper voices but F when the bass is involved (VL-5). When re moves up to 3 instead, the problem is avoided.

### 3.12 The perfect 4th against the bass; 6/4 chords, **F if not one of these types**

A P4 (ic 5, spelled as a 4th) between the LSV and any upper voice is a dissonance. It is allowed only as:

1. A **suspension** 4–3 (§3.3).
2. A PT or NT (§3.1, 3.2).
3. A **cadential 6/4**:
   - On a **stronger** beat than its resolution.
   - Bass on degree 5.
   - The 6th and 4th resolve **down by step** (6→5, 4→3) over the **same** bass, to V or V7.
   - The bass is approached by step or leap from a predominant (IV, ii6, ii6/5) or I6, not from V.
4. A **passing 6/4**:
   - Weak beat, between two positions of the same harmony (or I–V6/4–I6).
   - The bass moves by step in one direction through the 6/4.
   - One upper voice moves stepwise in contrary motion, and the 4th above the bass is held or moves by step.
5. A **neighbour (pedal) 6/4**:
   - Bass stationary (e.g. I – IV6/4 – I).
   - The 6th and 4th enter and leave by step as upper neighbours.
   - Weak or medium beat.
6. An **arpeggiated 6/4**: the bass arpeggiates within a sustained harmony.

**Any other 6/4 is F.** That includes a root-position-like 6/4 on a downbeat that doesn't resolve as cadential, and a 6/4 approached and left by leap in the bass. The same applies to unprepared 4ths over the bass on the beat.

Between upper voices the 4th is consonant: parallel 4ths in upper voices are allowed (P), provided no voice forms parallel 5ths with the bass. When a countersubject is involved, see §5.6.

Source: Aldwell & Schachter ch. 9 (6/4 types). Gauldin ch. 6. Kennan ch. 3 (4th as a dissonance against the lowest voice).

### 3.13 Consecutive dissonances / chained NCTs, **A**

- Two consecutive dissonant notes in one voice against the same held note are allowed if the pair is a recognized figure: double PT, PT + NT, cambiata, or ornamented suspension.
- Otherwise A.
- More than two consecutive dissonant attacks in one voice against a stationary harmony, all on accented subdivisions: F.

---

## 4. Harmonic and cadential rules

### HC-1 Doubling, **A** (F for LT doubling over dominant function)

Apply at accented beat onsets where ≥ 3 voices sound.

- **F:** doubling the leading tone (pc of degree 7 in major / raised 7 in minor, or a secondary LT) when the harmony is V, V7, vii°, vii°7 or a secondary dominant of the same.
- **F:** doubling a chordal 7th. **A:** doubling a suspended dissonance or its note of resolution while the suspension sounds.
- **A:** doubling any chromatically altered tone, e.g. ♯4 as a secondary LT, the ♭6 in an augmented 6th.
- **P:**
  - Root-position triads: double the root (most common), then the 5th, then the 3rd.
  - 6/3 chords: double a tonal degree (1, 4, 5, 2) rather than a tendency degree.
  - 6/4: double the bass.
- **Incomplete chords (P).** Omitting the 5th of a root-position triad or V7 is fine; the final tonic is often root ×3 + 3rd. Omitting the 3rd is A, except the final chord of a section where an open 5th/octave is used. Bach occasionally does this in archaic or organ contexts. Treat it as A.

Bach figures (Aarden & von Hippel 2004, MTO 10/2): in 2,643 chorale triads compared with random voicings, Bach doubled the **root of major root-position triads more** and the **leading tone and chromatic tones less** than chance. They reduce all doubling rules to "avoid doubling tendency tones" and favour a voice-leading explanation (doubled tendency tones force parallel octaves). Huron (1993; 50 chorales) found that doubled scale degrees match the Krumhansl–Kessler key profile. Source: https://www.mtosmt.org/issues/mto.04.10.2/mto.04.10.2.aarden_hippel.html.

### HC-2 Cadence formulas, **F for phrase-ending authentic cadences**

- **Perfect authentic cadence (PAC).**
  - Bass 5 → 1: leap up a 4th, down a 5th, or by octave displacement.
  - Harmony V(7) → I (or i).
  - Tonic arrival on a **stronger** beat than the dominant.
  - Soprano ends 2→1 or 7→1 (3 for an imperfect AC).
  - Typically prepared by a predominant (ii6, ii6/5, IV, iv6) and very often by a **cadential 6/4** or a **4–3 suspension** over 5. The latter is Bach's most frequent ornament at cadences, together with the anticipation of the final tonic in the top voice.
- **Half cadence.** Ends on V on a strong beat. The Phrygian HC (iv6 → V in minor, bass ♭6→5 by semitone) is idiomatic.
- **Deceptive cadence.** V → vi (VI in minor). The LT goes up; in minor double the 3rd of VI to avoid A2 and parallels.
- **Evaded cadences** in fugues are common (V → I6, V → vi). Treat them as P.
- **Internal cadences (P).** Internal sections of Bach fugues cadence in the key of the entry just completed. Allowed keys are as in §5.5.
- **F.**
  - A root-position V → I where the bass moves 5→1 but V lacks the LT anywhere.
  - A "cadence" arriving on a weak beat with a dissonant unresolved tone.
  - An LT in an outer voice not resolving (HC-3/§3.10).

### HC-3 Final cadence of the fugue, **F**

1. The last sounding sonority starts on a **downbeat** (beat 1). It lasts until the end of the bar, at least one full beat, and normally the whole measure.
2. It is **tonic** harmony of the home key in **root position**: LSV on degree 1, **no** other bass degree.
3. The **penultimate** harmony is **V or V7** (possibly over a tonic pedal; see 6). The bass moves 5→1 unless a tonic pedal is in effect, or it is a plagal extension after a PAC.
4. The final chord **contains the 3rd**. In minor, it is either the **Picardy (major) 3rd**, which is Bach's norm in the WTC, or the minor 3rd (A). The bare 5th/octave is A. Do not allow a final chord with the 5th in the bass, or one whose root is missing.
5. The soprano ends on 1 (P), or on 3 (A in Bach's fugues; it occurs).
6. The coda is often over a **tonic pedal** with a plagal (iv/IV → I) motion or a subdominant colouring (♭7 to IV) before the last chord. That is allowed (P).
7. All voices end together, or the final chord adds voices (keyboard thickening, P). A **voice rests through the final cadence** only if its last note completed a PAC.

Note for implementation: with voices ordered highest first, check that V3 (or the LSV) at the final onset has `pc == tonic`.

### HC-4 Second-inversion chords

See §3.12. All 6/4 sonorities on accented beats must be **cadential** or **pedal/neighbour** types. A **passing** 6/4 must be weak. Severity F.

### HC-5 Harmonic rhythm and bass motion, **P**

- Avoid the same harmony repeated across a bar line from weak beat to strong beat (a "syncopated harmony"), except for suspensions and pedal points.
- Avoid root progressions by descending 3rd → ascending 2nd → … that weaken function. Bach's normal motion is by descending 5th, ascending 2nd, and 6/3 sequences.
- Avoid **V → IV** (retrogression) except as V–IV6 passing in the bass (5–6–7–8 is fine). Use these as scores.

### HC-6 Augmented-sixth and Neapolitan, **F if mistreated**

- **A6.** The two notes forming the A6 (♭6 in the bass, ♯4 above) resolve outward by semitone to an octave on 5. Avoid the parallel 5ths in the German A6 → V; Bach normally goes to a cadential 6/4 or uses the Italian/French form.
- **N6 (♭II6).** ♭2 resolves down (to ♯7 or via 1). Avoid ♭2→♮2 cross relation between voices. Bass on 4.

---

## 5. Fugue-specific conventions

### FG-1 Subject, **P/A**

- The subject begins on **degree 1 or 5** of the tonic. Starting on 3 is rare (A); on other degrees, F.
- It ends on degree 1 or 3 of the tonic, or on degree 5 / 1 of the dominant key if it modulates.
- It stays within ≤ an octave + 4th (Prout/Mann norm; most WTC subjects are within an octave or a 10th).
- It is stated **alone** in the first voice: monophonic until the answer enters.
- Its melodic intervals obey ML-1, with the d7/d4 allowed in minor subjects.

### FG-2 Answer: real vs tonal, and the exact mutation, **F if wrong**

Definitions (Hilse, "The Fugal Answer", *The American Organist*, April 2006; https://wp.agohq.org/wp-content/uploads/2019/11/The-Fugal-Answer.pdf):

- *Real answer.* Every note of the subject transposed **up a P5 or down a P4** (+7 or −5 semitones, adjusted by octave). In the dominant key, spelled diatonically there.
- *Tonal answer.* Each note is taken from either **Transposition A** (+P5 = the real answer) or **Transposition B** (+P4 = a 5th lower), with occasional chromatic inflections for harmony.

**Algorithm (Hilse; Prout, *Fugue*, 1891, ch. 3–4; Grove, "Fugue" / "Tonal Fugue"):**

1. If the subject has **no prominent degree 5 at its head** and does **not modulate** to the dominant, use the **real answer** (all A). Example: WTC I C major has a real answer.
2. **Initial (head) mutation.**
   - Scope: a degree 5 of the tonic among the first notes of the subject, typically the first 1–4 notes or before the first strong-beat arrival after the head. That includes a subject beginning on 5, or a head motion 1→5 or 5→1.
   - Mutation: that 5 is answered by **degree 1 of the tonic**, taken from B, rather than by 2, which A would give.
   - Neighbouring notes in the head may also be taken from B to preserve contour.
   - The rest of the answer comes from A.
   - Consequence: the head interval 1↑5 (P5) becomes 5↑8 (P4), and 5↓1 becomes 1↓5.
   - Concrete pitches: in C, a subject C–G… is answered G–C…; a subject G–C… is answered C–G….
3. **Terminal mutation (modulating subjects).**
   - Applies when the subject modulates to V and cadences there.
   - The answer must cadence back in **I**, not in II. So from some "splice point" (a caesura) onwards, take the notes from B.
   - Prout: "the answer must return to the tonic at the same point where the subject left it for the dominant".
   - Example: WTC I E♭ major. WTC I G♯ minor is a case where the splice comes as early as the second note.
4. **Minor mode.**
   - A: ♯7 of the tonic (LT) maps to ♯4 (the LT of the dominant) in the answer. This is correct.
   - Grove: "a diminished 7th in the subject is left unaltered in the answer".
   - The answer is in the **minor dominant** (v), with its own raised 7th (♯2 of the tonic).
5. **Validation rule for the checker.**
   - Given subject S and an entry E in the dominant region, E must equal A, or be a note-wise mix of A and B.
   - B notes may appear only (a) within the head window (first ≤ 4 notes / first beat-group), corresponding to subject degrees 5 (and immediate neighbours), or (b) from a single splice point to the end, in modulating subjects.
   - Every A/B difference is exactly ±1 diatonic step; the rhythm is identical.
   - Chromatic inflection of ±1 semitone on at most 1–2 notes is A (Bach does it, e.g. WTC II B♭ minor's A♮ answering D♭).
6. **Bach's exceptions.** Hilse: "Herr Bach broke every would-be 'rule' at one time or another". For example, the "Little" G minor organ fugue BWV 578 starts on the triad without the expected head mutation. So a real answer where a tonal one is "expected" is **A**, not F. A mutation that is **not** a 1↔5 exchange, or that changes rhythm, is F.
7. Other sources: Wikipedia, "Fugue" (https://en.wikipedia.org/wiki/Fugue), e.g. BWV 861 G minor: the subject's opening D is answered by G. Gedalge, *Traité de la fugue* (1901; trans. Levin, *Treatise on the Fugue*, 1965), Part I chs. on the answer, with the same tonic↔dominant exchange principle.

### FG-3 Exposition: order, entry intervals, timing, **F (structure) / P (order)**

- **Number of entries.** Exactly one entry per voice, with all 4 voices in the exposition. Occasionally there is an extra (redundant) entry: P.
- **Alternation.** Subject (tonic) – Answer (dominant) – Subject – Answer: S A S A. Bach occasionally uses S A A S, as in WTC I C major, BWV 846 (Wikipedia). Treat any other order as A.
- **Pitch levels.**
  - The two subject entries are an **octave (or two) apart**, both at tonic level.
  - The two answer entries are an octave apart at dominant level.
  - Each answer enters **a 5th above or a 4th below** the preceding subject, or the corresponding inversion.
- **Voice order.** Typically each new entry is in a voice **adjacent** to an existing one, e.g. A–S–T–B, T–A–S–B, B–T–A–S, S–A–T–B. Any order is allowed (P). Bach's WTC four-voice fugues show many orders.
- **Timing.**
  - Each new entry begins **when the previous statement ends**, i.e. on its last note or the beat after.
  - Or after a short **codetta**, about 1–2 bars, which may modulate back to the tonic for the next subject entry. The codetta between the answer and the third entry is common.
  - The answer normally enters without a codetta (Wikipedia, "Fugue").
  - An entry starting before the previous one ends is a stretto in the exposition: A in a normal exposition.
- **Continuity.** Once a voice has entered, it continues with countersubject or free counterpoint until the exposition ends (P; a short rest is allowed). It must not double another voice.
- **Entry clarity.**
  - A voice that enters with the subject (or re-enters later) should be preceded by a **rest** of ≥ ½ beat in that voice (P/A). Prout and Gedalge recommend this so the entry is audible.
  - Huron & Fantini, "The Avoidance of Inner-Voice Entries", *Music Perception* 7/1 (1989), show that listeners detect inner-voice entries poorly, and that polyphonic composers thin the texture to compensate.
  - Do not let other voices cross the entering voice during its first 2 beats (VL-6).

### FG-4 Countersubject, **F for inversion errors**

- A countersubject (CS) is a counterpoint that returns with (most) later subject entries. It must work **above and below** the subject: **invertible counterpoint at the octave/15th**.
- **Interval mapping** at the octave: 1↔8, 2↔7, 3↔6, 4↔5 (Ithaca handout; Maves, https://mavesd.people.charleston.edu/invertiblecpt.htm).
- **Rules (F).** Check each CS/subject combination in **both** vertical orders: the original, and with the CS transposed by octaves to the other side.
  1. **P5 between CS and subject** becomes a P4 when inverted. It is allowed only on a weak position as a PT or NT, or where the inverted P4 is acceptable: an upper-voice pair not involving the bass, or a 4–3 suspension. Otherwise it is F in the **inverted** version.
  2. **Parallel 4ths** in either version invert to parallel 5ths, so they are forbidden.
  3. **Suspensions.**
     - **7–6** (suspended voice above) inverts to **2–3** (suspended voice below): valid both ways. This is the preferred invertible suspension.
     - **4–3** (above) inverts to **5–6** (below). It stays legal, because the 5th is consonant and the voice still falls by step, but the suspension effect is lost. It is P.
     - **9–8** (above) inverts to **7–8** (suspended voice below, resolving down into an octave). That is a dissonance resolving to a perfect consonance, with weak effect and exposed octaves. Treat it as **A** (Maves's "p"-marked case).
     - **2–3 bass suspension** inverts to 7–6 above: valid.
     - Hence: prefer 7–6 / 2–3 in the countersubject, and avoid suspensions that resolve to a perfect consonance in either version.
     - Maves: "approach that suspension by an interval other than the one it resolves to"; "a fourth that resolves to a fourth is not allowed".
  4. **Distance.** Keep the CS–subject distance ≤ an octave (≤ 12 st) for clean octave inversion. With wider spacing, check the compound intervals mod 12 as above.
- **Other inversions Bach uses.**
  - At the **12th**: 6ths become 7ths, so the 6th must be treated as a dissonance. Mapping 1↔12, 3↔10, 5↔8.
  - At the **10th**: parallel 3rds and 6ths become 8ves and 5ths, so no parallel motion is allowed.
  - Source: Ithaca handout. Maves. Gauldin chs. 11–12 (invertible counterpoint). Kennan ch. 8.
- **Style (P).** The CS contrasts rhythmically with the subject: it moves while the subject holds, and vice versa. It starts after the subject's first note or on an offbeat. It does not double the subject at the 3rd or 6th for more than 2 beats.

### FG-5 Keys of entries (middle entries), **A / F**

- **Allowed keys** (closely related; Prout, Mann): keys whose tonic triad is a major or minor diatonic triad of the home key.
  - Major home key: I, ii, iii, IV, V, vi.
  - Minor home key: i, III, iv, v, VI, VII. In minor, also the major dominant region via the harmonic minor; the dominant minor is the usual answer key.
  - Bach's first middle entry is often in the **relative** key (Wikipedia, "Fugue"). Entries in ii, iii, IV, vi are standard for the later middle entries.
- **A.** An entry in a key one step outside these, e.g. the minor subdominant in a major fugue. Bach does this rarely, in long or chromatic fugues.
- **F.** An entry in any more distant key, such as a tritone or chromatic-mediant relation.
- **Modulation between entries.** Episodes modulate by sequence, typically by descending 5ths or by step (Wikipedia, "Fugue": the episode in BWV 847 modulates to the relative major). The key change happens before or at the entry. An entry must not begin before its key is established, meaning a dominant or a pivot of the new key within the previous 2 beats (P).

### FG-6 Final entry / return, **F**

- The fugue has **at least one complete subject entry in the tonic after the last middle entry in another key**. This is the final section or recapitulation. It is often in the **bass or soprano**, and is often a stretto or over a dominant/tonic pedal.
- The last complete entry is in the tonic, or its answer form leads back to it.
- After it there is only cadential or coda material. **No** new non-tonic entry after the final tonic entry: F.

### FG-7 Stretto, **P (when used, all VL rules apply)**

- Overlapping entries at intervals that allow consonant counterpoint, commonly at the octave, 5th or 4th, and at a time distance the subject permits.
- All VL and dissonance rules apply. No exemptions for "subject integrity".
- Stretto *maestrale* means all voices complete the subject (Wikipedia, "Fugue").

### FG-8 What a "rule-breaking" entry looks like (checker targets)

Classify an entry as **broken** (F) if any of these hold:

1. **Wrong transposition.** Interval content differs from S, A, or a valid tonal mix (FG-2) other than by ≤ 2 chromatic inflections. This includes a "real" answer where a tonal one is expected with extra non-1↔5 changes, and degree-wise drift in the middle.
2. **Rhythm altered**, other than augmentation, diminution, or a lengthened/shortened first or last note.
3. **Remote key** (FG-5).
4. **Truncated** in the exposition or the final entry. Fragmentary entries are allowed only in episodes, as motif work.
5. **Obscured.**
   - Entering voice crosses or overlaps others (VL-6/7).
   - No rest beforehand **and** starting inside another voice's sustained note at the same pitch.
   - Starting on an unprepared dissonance against the bass that is not a suspension.
6. **Harmonized against its implied harmony.** The subject's head (typically 1 or 5 of the entry key) sounds over a dissonant or non-tonic bass that undermines the entry key, e.g. the tonic note of the entry as a 4th above the bass on a downbeat (§3.12 violation).
7. **Mis-voiced exposition.** Two subject entries at the same octave, or a voice entering twice before all voices have entered (unless it is a deliberate redundant entry).

### FG-9 Episodes, **P**

- Built from subject, countersubject or codetta motifs.
- Sequential, by descending 5ths, by step, or as a canon.
- Modulating to the next entry key.
- Usually lighter texture (fewer voices).
- No new thematic material. All VL rules apply. Sequences may repeat a pattern **at most 3 times** (two repetitions after the model is the textbook norm; Bach often does 2–3 statements). A 4th or later repetition is A.

---

## 6. Rule table with severity, exceptions and Bach rates

| ID | Rule (short) | Code test | Severity | Exceptions (coded) | Bach rate / evidence |
|---|---|---|---|---|---|
| VL-1 | Parallel/contrary 8ves & unisons | consecutive slices, both move, ic 0→0 | F | phrase boundary only | octaves rarer than fifths; none between S/B in chorales (Dahn) |
| VL-2 | Parallel/contrary 5ths | both move, ic 7→7 | F | E-cad anticipation (inner voice, ≤1/piece) | 25 syntactical consecutives in 5,600+ mm (~1/224 mm), 11 structural, 1 stepwise parallel (Dahn) |
| VL-3 | Accented (beat-to-beat) parallels | beat-level reduction | A | 5–6 interpolation | Fux/Kennan |
| VL-4 | Hidden 8ve/5th outer voices, soprano leaps | similar motion into ic 0/7, V0–V3, Δsoprano > 2 | A (8ve: near-F) | step in soprano; same-chord arpeggiation | |
| VL-5 | Unequal 5ths d5→P5 | ic 6→7, similar | F with bass, A upper | I–V4/3–I6 | |
| VL-6 | Voice crossing | p_i < p_j, i<j | A; F for bass, > 1 beat, at entries | inner voices ≤ 1 beat | Huron 1991: 105 Bach works avoid crossing beyond chance |
| VL-7 | Overlap | next pitch beyond neighbour's previous | A | across rests | |
| VL-8 | Spacing / hand span | S–A, A–T ≤ 12; T–B ≤ 19; hand span ≤ 15–16 | A / F | | largest gap usually lowest pair (Aarden & von Hippel 2004) |
| VL-9 | Cross relation | same letter, different alteration, different voices, adjacent slices | A (simultaneous: F) | minor 6/7 melodic-minor exchange; chromatic line; Phrygian; A6 | frequent in minor |
| VL-10 | Simultaneous similar leaps | all voices leap same direction | A | same-chord arpeggiation | |
| ML-1 | A2, A4, M7, > 8ve leaps | spelled intervals | F (A4: A) | d7, d4 allowed in minor; d5 if recovered | |
| ML-2 | Leap recovery | leap ≥ m6 → opposite motion | A | arpeggio, compound melody | weaker in mid-tessitura (von Hippel & Huron 2000) |
| ML-3 | Same-direction leaps outlining dissonance | combined span ic ∈ {1,2,6,10,11} or > 12 | A | V7/vii°7 arpeggio | |
| ML-5 | Range | per-voice corpus percentiles | F (compass) / A | | WTC compass ≈ C2–C6 |
| D-0 | Unclassified dissonance | any dissonant note not PT/NT/SUS/APP/ANT/ET/CAMB/7th/pedal | F | implied-harmony arpeggio at 16th level | |
| D-SUS | Suspension form | prep same pitch consonant; accented; resolve down by step | F if malformed | LT retardation up; ornamented resolution | |
| D-7 | Chordal 7th resolution | down by step at harmony change | F | V4/3–I6 rising; held as common tone | |
| D-LT | LT resolution | outer voices: 7→8 at V→I | F outer, A inner | descending scale; inner-voice frustrated LT | Bach drops inner LT often at cadences |
| D-6/4 | 4th vs bass / 6/4 | ic 5 vs LSV not in allowed types | F | cadential, passing, neighbour, arpeggio 6/4, 4–3 sus, PT/NT | |
| HC-1 | Double LT / 7th | pc count ≥ 2 | F | | Bach doubles LT less than chance (Aarden & von Hippel) |
| HC-3 | Final cadence | root-position tonic on downbeat, V→I, 3rd present | F | Picardy or minor 3rd | WTC minor fugues usually end major |
| FG-2 | Answer form | A/B mix rule | F / A (real instead of tonal) | | BWV 578 real where tonal expected (Hilse) |
| FG-3 | Exposition | one entry per voice, S A S A, octave pairing | F structure / P order | SAAS (BWV 846) | |
| FG-4 | CS invertibility | check both orders | F | | |
| FG-5 | Entry keys | closely related | F remote, A borderline | | first middle entry usually relative key |
| FG-6 | Final tonic entry | ≥ 1 complete tonic entry after last non-tonic entry | F | | |

---

## 7. Implementation notes

1. **Two levels of check.**
   - *Surface*: every slice transition. Use it for VL-1, VL-2 (hard), crossing, ranges and melodic rules.
   - *Reduced*: beat-level chord-tone skeleton. Use it for VL-3, suspension parsing, 6/4 typing, doubling, cadences and keys.
   - Many false positives disappear if the reduced-level checks ignore NCTs. Do **not** relax VL-1/VL-2 on the surface level, except for the coded E-cad case.
2. **Spelling matters** for A2/d7, d5/A4, cross relations and unequal 5ths. Spell from the local key (§0) before running ML-1, VL-5 and VL-9.
3. **Rests and re-entries.** Treat a voice that rests ≥ 1 beat as absent. Do not compute parallels across that rest. Do compute the "entry clarity" rule (FG-3) when it returns.
4. **Thresholds.** For A-rules, measure Bach's rate per 100 measures on the project's own corpus with the same code (precision first), and set the reject threshold at about the 95th percentile of Bach fugues. This is the only defensible way to calibrate "avoid" rules: the textbook literature gives counts only for the chorales.

---

## 8. Prioritized list: the ~15 most audible / most clearly "wrong" violations

1. **Unclassifiable dissonance on a strong beat** (D-0 at beat level): a clash that neither resolves by step nor was prepared. This is the single most "wrong-note" sounding error.
2. **Parallel octaves/unisons** (VL-1), especially involving an outer voice. Two voices collapse into one.
3. **Parallel fifths** (VL-2), especially outer voices or exposed two-voice texture.
4. **Final cadence not on a root-position tonic** (HC-3): ending on an inverted, dissonant, or non-tonic chord, or a weak-beat ending.
5. **Unresolved or wrongly resolved leading tone in the soprano or bass** at a cadence (D-LT).
6. **Wrong answer / distorted subject entry** (FG-2, FG-8): a subject statement with a wrong interval, which a listener who knows the subject hears at once. This includes an answer that drifts to the supertonic key (real answer to a modulating subject).
7. **Malformed suspension**: a dissonance resolving upward or by leap (D-SUS), or a 7th of a chord not resolving down (D-7).
8. **Unprepared 4th / bad 6/4 against the bass on a downbeat** (D-6/4).
9. **Augmented 2nd or 4th melodic leaps, or leaps of a 7th or larger than an octave** (ML-1) in an upper voice.
10. **Simultaneous cross relation, or an immediate false relation between outer voices** (VL-9).
11. **Direct (hidden) octave between outer voices with a soprano leap** (VL-4).
12. **Entries in remote keys / no final tonic entry** (FG-5, FG-6): tonal incoherence over the large-scale form.
13. **Voice crossing that obscures an entry, or the bass rising above the tenor** (VL-6).
14. **Doubled leading tone or doubled chordal 7th** (HC-1). It usually also produces parallel octaves on resolution.
15. **Unrecovered large leaps / same-direction leaps outlining a 7th or tritone** (ML-2, ML-3), and **range or keyboard-span violations** (ML-5, VL-8). These sound "un-Bach-like" or are unplayable.

---

## Sources

**Treatises and textbooks** (book references; not available online in full):

- J. J. Fux, *Gradus ad Parnassum* (1725). Trans. Alfred Mann, *The Study of Counterpoint* (Norton, 1943/1965).
- Alfred Mann, *The Study of Fugue* (Rutgers, 1958; Dover 1987).
- Kent Kennan, *Counterpoint: Based on Eighteenth-Century Practice*, 4th ed. (Prentice Hall, 1999).
- Robert Gauldin, *A Practical Approach to 18th-Century Counterpoint* (Prentice Hall 1988; Waveland 1995/2013). Also Gauldin, "Fux to Bach" (cited via a core.ac.uk search result; not read in full).
- Ebenezer Prout, *Fugue* (Augener, 1891) and *Fugal Analysis* (1892).
- André Gedalge, *Traité de la fugue* (1901). Trans. A. Levin, *Treatise on the Fugue* (Univ. of Oklahoma Press, 1965).
- Edward Aldwell & Carl Schachter, *Harmony and Voice Leading*, 4th ed. (Schirmer, 2011).
- David Huron, *Voice Leading: The Science behind a Musical Art* (MIT Press, 2016).

**Articles:**

- Huron, "The Avoidance of Part-Crossing in Polyphonic Music: Perceptual Evidence and Musical Practice", *Music Perception* 9/1 (1991).
- Huron, "Tone and Voice: A Derivation of the Rules of Voice-Leading from Perceptual Principles", *Music Perception* 19/1 (2001): 1–64.
- Huron & Fantini, "The Avoidance of Inner-Voice Entries", *Music Perception* 7/1 (1989).
- von Hippel & Huron, "Why Do Skips Precede Reversals?", *Music Perception* 18/1 (2000).
- Aarden & von Hippel, "Rules for Chord Doubling (and Spacing): Which Ones Do We Need?", *MTO* 10/2 (2004). https://www.mtosmt.org/issues/mto.04.10.2/mto.04.10.2.aarden_hippel.html
- Fang et al., "Bach or Mock? A Grading Function for Chorales in the Style of J.S. Bach" (2020). https://arxiv.org/abs/2006.13329
- Walter Hilse, "The Fugal Answer", *The American Organist* (April 2006). https://wp.agohq.org/wp-content/uploads/2019/11/The-Fugal-Answer.pdf

**Web pages:**

- Luke Dahn, "Consecutive Fifths & Octaves in the Bach Chorales": https://bach-chorales.com/ConsecutivesInChorales.htm
- "Parallel Fifths in Bach" (wtfbach): https://wtfbach.substack.com/p/parallel-fifths-in-bach
- Chorale guide, parallels: https://choraleguide.com/vl-parallels.php
- Iowa State / intmus, voice-leading errors: https://intmus.github.io/inttheory/10-intro-part-writing/a2-voiceleadingerrors.html
- UVM part-writing handout: https://www.uvm.edu/~dfeurzei/110/materials/partwriting.pdf
- David Maves, "Invertible Counterpoint": https://mavesd.people.charleston.edu/invertiblecpt.htm
- Ithaca College, "Double (Invertible) Counterpoint": https://musictech.ithaca.edu/MusicTech/Form/Handouts/DblCpt.html
- Wikipedia, "Fugue": https://en.wikipedia.org/wiki/Fugue
- Wikipedia, "Consecutive fifths": https://en.wikipedia.org/wiki/Consecutive_fifths
- Grove, *Dictionary of Music and Musicians*, "Tonal Fugue" (Wikisource): https://en.wikisource.org/wiki/A_Dictionary_of_Music_and_Musicians/Tonal_Fugue

**Caveats:**

- Fetched and verified: Dahn, Hilse, Bach-or-Mock, Aarden & von Hippel (summary sections), Maves, Ithaca, intmus, chorale guide and Wikipedia.
- Not consulted directly (no web access to full text): Fux, Kennan, Gauldin, Prout, Gedalge, Mann, Aldwell & Schachter, and Huron's articles. Rules attributed to them reflect their standard, widely taught content.
- The keyboard per-voice ranges and the crossing frequency in the WTC are approximate. They should be measured on the project's corpus.
