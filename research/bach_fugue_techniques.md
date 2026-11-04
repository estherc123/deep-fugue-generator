# Bach fugue techniques for a form planner

This report collects what the treatise and analysis literature says about how Bach's fugues are built, and turns it into decision rules for a hierarchical generator: form planner, then harmony planner, then note model.

## How the sources are marked

- **[V]** means I read the claim in the source during this research (web text or PDF), with the URL given.
- **[B]** means the claim is standard teaching from a book I could not open online: Mann, Kennan, Gauldin, Gedalge, Ledbetter, Tovey, Bruhn or Renwick. These are textbook generalizations, not numbers I checked against the source text. Treat them as priors, not facts.
- Counts from analyses depend on definitions: what counts as a stretto, an entry or an episode. Analysts differ by a few entries per fugue; Altschuler and Bruhn, for example, differ by 2 on WTC I/20.
- WTC numbering follows Prout: 1–24 is Book I and 25–48 is Book II. For example, "Fugue 31" is II/7 in E-flat major.

Two corpus papers are useful but I could not reach their text: the Algomus "Computational Fugue Analysis" (Giraud et al., *CMJ* 39/2, 2015, ground truth for 36 fugues by Bach and Shostakovich) and the Algomus episode-detection paper. Their ground-truth annotations are the best source for computing exact frequencies yourself; see the open questions at the end.

---

## 1. Large-scale form

### 1.1 Section sequence

Prout describes every fugue as **ternary in outline** [V, Prout *Fugue* ch. IX §293]:

1. **First section**: the exposition (S–A–S–A in four voices), plus an optional **counter-exposition**. It ends with the last entry in the original keys (tonic or dominant).
2. **Middle section**: it begins with the first episode that **modulates**. It alternates entries in related keys with episodes. Entries can be single or grouped.
3. **Final section**: return to the tonic, with "all the voices engaged" [V, §327]. It often holds the closest stretto, a pedal and a coda.

Supporting details:

- **Counter-exposition.** It is optional: only **13 of the 48** WTC fugues have one [V, Prout ch. VI §211]. It stays in tonic and dominant with the voices' roles swapped, often answer first. Its entries are often closer together than in the exposition (Fugue 1: one crotchet; Fugue 31: a canon at one bar). It can also be in inversion (Fugues 15 and 46) [V, §§207–211].
- **Exposition.** The last entry falls in an outer voice in **42 of 48** WTC fugues [V, Prout §182]. The preferred four-voice orders are S-A-T-B, B-T-A-S, A-T-B-S and T-A-S-B [V, §199].
- **Codettas.** These are bridges inside the exposition. They are needed when the subject ends on the tonic, so that the next entry is not static, or to avoid a tonic–dominant clash [V, Prout §§188–193]. Belkin adds that codettas break "the squareness of equidistant successive entries" [V, Belkin].
- **Number of middle-entry groups.** Prout says there is "no fixed number": from "one or two to five or six, or even more" [V, Prout §328]. The usual four-voice WTC fugue has roughly 3–5 middle entries or groups. WTC I/20 in A minor has 39 entries, the most in the set together with II/3 [V, earsense summary of Bruhn and Altschuler].
- **Episodes.** "Most episodes are two or three bars long, though some run much longer" [V, Prout ch. VII]. Some fugues have no real episodes, such as I/22 in B-flat minor, which has only canonic bridges [V, Wikipedia BWV 867; Belkin].

### 1.2 Proportions (fraction of the total length)

Worked examples from the sources:

| Fugue | Exposition ends | Middle | Final section starts |
|---|---|---|---|
| WTC I/21 (B-flat major) | bar 17/48 (35%) | 17–41 | bar 41/48 (85%) [V, Prout §296] |
| WTC II/7 (E-flat major, "Fugue 31") | bar 44/70 (63%), a long first section | short | subdominant entry at 53–58 (76%), tonic return at 59 (84%) [V, Prout §294] |
| WTC II/10 (E minor, "Fugue 34") | – | 5 episodes, all built from the subject's tail | coda 77–86 of 86 (last 12%), over an ornamented dominant pedal [V, Prout §§304–305] |
| WTC I/2 (C minor), 31 bars | about bar 9 (29%) | E-flat major, then G minor, then C minor (soprano) | final bass entry at about bar 26 (84%); tonic-pedal coda in bars 29–31 ending on a Picardy third [B for bar numbers; Picardy third V, Wikipedia BWV 847] |

Planner priors drawn from these examples:

- Exposition, plus any counter-exposition: about 20–35% of the piece.
- Middle section: about 35–60%.
- Tonic return: about 65–85%.
- Coda: the last 5–12%.

Stretto is the best-measured device. Across all WTC stretto passages the mean start position is **57% (SD 23%)**. The most common bin is **51–63%**, and the third quarter has more stretto than the fourth [V, McDonald 2019, Fig. 5]. See section 2.1.

### 1.3 Key plan

- **Doctrine.** Middle entries stay in the closely related keys: the keys of the diatonic triads of the home key. In major these are V, vi, iii, IV and ii. In minor they are III, v, iv, VI and VII (the Baroque "ambitus" of Heinichen and Mattheson) [B, Ledbetter; Mann]. The minor supertonic is "a remote key, seldom figuring as the site for a fugal entry" [V, Baker 2017 citing Bach's tonal system].
- **First middle entry.** In minor fugues it usually goes to the **relative major** (III). In major fugues it goes to the **dominant**, or to the relative minor (vi) when the exposition has already worked the dominant hard [V, Prout via search snippet: "probably the dominant if the movement be in a major key, and the relative major if it be in a minor key"; V, Wikipedia *Fugue*: "the first is often in the relative major or minor, followed by an entry in the dominant of that key"]. Examples: I/2 C minor goes to E-flat major; I/16 G minor has its first middle entry at bar 12 in B-flat major [V, textbook analysis via search].
- **Pairing within the middle section.** An entry in a new key is often followed by an entry in **that key's dominant**, a local subject–answer pair [V, Wikipedia *Fugue*]. Prout: grouped entries are best a 4th, 5th or octave apart. Successive groups should not repeat the same voice order, the same key, or the same voice twice in a row [V, Prout §325].
- **Cherubini's school order is not Bach's.** Cherubini's order for major keys is V, vi, IV, ... back to V; minor keys begin with III. Prout says **no fugue in the 48 follows it** [V, §§322–323]. Bach occasionally uses keys outside the ambitus when they are "naturally introduced", such as D minor in the E minor fugue and E-flat minor in the A-flat fugue [V, §324].
- **Typical shape** [B, Kennan/Gauldin; consistent with the Prout examples]:
  - Major: I/V (exposition) → V or vi → (iii or ii) → **IV** → I.
  - Minor: i/v (exposition) → **III** → v (or VII) → **iv** → (VI) → i.
  - The flat side (IV/iv) comes late, just before the tonic return. This is the "subdominant shift" of the final third. Prout counts the subdominant entry as part of the *middle* section, placed immediately before the tonic return [V, §294].
- **Modulation does not need an episode.** It can happen on the subject's own notes; in Fugue 29 the middle section starts at bar 10 that way [V, Prout §312]. A middle section can also contain a tonic entry before its final modulation (Fugue 34) [V, §302].

### 1.4 Climax and final section

- **Return to the tonic.** It is always present, with all voices engaged [V, Prout §327].
- **Pedal points.** When Bach uses them, they are "almost always" in the final section, dominant then tonic. Earlier pedals are rare; Fugue 35 is an exception [V, Prout §329]. Close stretti are often built over the pedal [V, Prout §272]. Belkin warns that the pedal is "an obligatory part of the school fugue" but "not a norm in Bach". Tonic pedals often end Bach's fugues because they stabilize the even harmonic rhythm [V, Belkin]. A coda over a tonic pedal, sometimes entered after a **deceptive cadence**, is a recognized type [V, search summary of general sources; B, Kennan].
- **Last entry.** It usually falls in an **outer voice**: the bass for weight (I/2 has its last real entry in the bass and then a soprano statement over the tonic pedal) or the soprano for brilliance. In I/22, a five-voice fugue, the single stretto at 90% enters voice by voice from top to bottom at half-note distance [V, McDonald p. 58].
- **Added voices.** Extra voices or thickening near the end are common: Prout counts this in **16 of the 48** fugues, "mostly near the cadence" [V, Prout §306].
- **Picardy third.** Minor fugues usually end on a major tonic (I/2, I/22 and many others) [V, Wikipedia BWV 847 and BWV 867]. Treat this as a near-default.
- **Climax.** There are two kinds. A *climax of weight* is the full-voice stretto near the end; Kerman's model is I/22 at 90%. A *climax of agitation* is a dense, high and fast stretto, as in I/4 (C-sharp minor), which comes earlier, with "the climax of weight" later [V, McDonald quoting Kerman *The Art of Fugue* pp. 26–27]. Belkin defines a climax as "a high point, followed by a resolution" and says fugue builds "by accumulation, rather than through strong contrasts" [V].
- **Ending.** "The difficulty of ending a fugue lies in stopping its momentum." Three tools stop it: harmonic stasis (a pedal), dissolving imitation into a simpler texture, and climax followed by resolution. These are often combined [V, Belkin].
- **Late dissonance.** I/4 places a striking dissonance four bars before the end, on the downbeat of bar 112 of 115 [V, Wikipedia BWV 849].

---

## 2. Techniques

### 2.1 Stretto (and stretto maestrale)

- **What it is.** An entry starts before the previous one has finished. In a *stretto maestrale* or grand stretto, every voice completes the subject [V, Wikipedia *Fugue*]. Prout's rule: "the last entering voice in a stretto should have the subject complete" [V, §252].
- **Effect.** Intensification and compression. It is the "most natural tension available in a fugue" (Bruhn on I/1, via earsense).
- **Frequency** [V, McDonald 2019 corpus study of all 48 fugues]:
  - **21 of 48 fugues (44%)** contain stretto: 12 of 24 in Book I and about 38% of Book II. Prout: "more than half of the 48 have no stretto at all", and some of the finest (Nos. 2, 12, 21, 40) have none [V, Prout §§245–246].
  - The corpus has 88 stretto passages and 234 stretto entries.
  - In fugues that use stretto, the mean is **4.2 passages per fugue (median 3, mode 2)**. Two outliers, I/20 and I/8, have 11 and 10.
  - Entries per passage: **2 entries in 45 of 88 passages**, 3 in 20, 4 in 10, and at most 4 in 75 of 88. The maximum is 10 entries, in I/4.
- **Placement** [V, McDonald]:
  - Passages start anywhere from 2% to 95% of the piece. The mean is **57%**, with significantly more after the midpoint (t(87) = 3.01, p = .003). The peak is at **51–63%**, and the last quarter has *fewer* passages than the third.
  - The first stretto falls at a mean of **41% (SD 23%)**. An early first stretto predicts many stretti, a "stretto fugue" (r = −0.61).
  - Only three fugues have stretto in the exposition (I/19 A major BWV 864, II/3 BWV 872, II/5 BWV 874). Only II/5 in D major is a clear stretto fugue: stretti at 10, 28, 42, 56, 67 and 89%, with the time interval shrinking from one bar to two beats to one beat.
  - Counter-examples to the textbook narrowing rule: overall, time intervals **do not shrink** toward the end; they grow slightly (r = 0.17). Entries per passage do not increase toward the end either (r = 0.06). So "ever-closer stretti" is a teaching convention that Bach follows only sometimes. Belkin agrees: Bach uses stretto "here and there, in no particular order of closeness" [V]. Prout gives the school version: later stretti use more voices or closer distances, and the final one is the closest with all voices [V, §§251, 327].
- **Pitch intervals between stretto entries** [V, McDonald Fig. 11]. Perfect intervals dominate:
  - Descending 5th: 29
  - Ascending octave: 25
  - Descending 4th: 21
  - Ascending 5th: 17
  - Descending octave: 16
  - Ascending 4th: 13
  - All imperfect intervals are rare, and they become slightly more frequent near the end.
  - Prout says the same: "fourths, fifths, and octaves work best" [V, §248].
  - More than 75% of stretto passages begin on local degree 1 or 5.
- **Time intervals** [V, McDonald Fig. 8]. Typical distances are 1–4 beats, often half a bar or one bar: I/20 uses 2 beats, I/7 uses 2 bars, I/22 uses a half note in cut time.
- **Preconditions.** The subject must make good counterpoint with itself at the chosen interval and delay. "A subject which opens with a sequence automatically allows a few canons" [V, Belkin]. Altered, truncated or inverted entries are allowed in stretto:
  - 8 of 48 fugues put **inverted** entries in stretto.
  - I/8 puts **augmented** entries in its last three stretti, with the augmentation in a different voice each time (B, A, S) and the soprano augmentation an octave or more above the alto.
  - I/20 shortens its late stretto entries by 2–6 of 12 beats [V, McDonald pp. 36–38].
- **Planner implication.** Precompute admissible (interval, delay) pairs for the subject: interval in {P4, P5, P8} up or down, delay from 1 beat to 2 bars. Allow stretto only where a pair exists. Place the first stretto at about 40% ± 20% and the strongest one at 55–90%.

### 2.2 Inversion

- **What it is.** Melodic mirror of the subject (the tonal form is adjusted).
- **Effect.** Variety while keeping identity. It is often used as a "development" step after the plain entries.
- **Placement.**
  - It can drive a whole counter-exposition: Fugue 15 (I/15), bars 20–31; Fugue 46 (II/22), from bar 42 with subject and countersubject inverted [V, Prout §210].
  - Where entries are scarce, a composer "may develop the subject by altering it, often by inversion" [V, Wikipedia *Fugue*].
  - I/20 in A minor states 50% of its entries inverted [V, earsense].
  - I/6 in D minor puts inverted entries at bars 22–23 [V, Marlowe 2020].
  - Inverted forms also appear in late stretti: I/8, I/20 and II/3 [V, McDonald].
- **Frequency.** I could not verify a single count. Inversion is used in a substantial minority, roughly 10–15 of the 48: among others I/6, I/8, I/15, I/20, II/2, II/3, II/22 [B/partial V]. It is used much more often than augmentation [V, earsense vocabulary: "augmentation is used far less often than inversion"].
- **Preconditions.** The inverted subject must still imply sensible harmony. Diatonic inversion around degree 3 or 5 preserves the tonic–dominant frame [B, Mann; V, teoria.com on the *Art of Fugue* mirror: the dominant becomes the subdominant under inversion].

### 2.3 Augmentation and diminution

- **What it is.** The subject with note values doubled (augmentation) or halved (diminution).
- **Effect.** Augmentation gives gravity and a cantus-firmus feeling. It works best in the bass or soprano under or over normal-speed voices.
- **Placement.** Late. In I/8 (D-sharp minor) it is the final three stretti. In II/2 (C minor) it is the bass augmentation near the end. II/3 (C-sharp major) uses augmentation and diminution, with inverted and augmented entries in one stretto at 63% [V, McDonald; earsense]. II/9 (E major) uses diminution together with stretto [V, earsense; Wikipedia *Stretto*: the counter-exposition with the subject in stretto]. Augmentation can also bring in a late voice: in II/2 the fourth voice enters at bar 19 in augmentation [V, Prout §203].
- **Frequency.** Rare: about 3–5 of the 48 [V, earsense lists II/2 and II/3 for augmentation and II/9 for diminution, with I/8 from McDonald].
- **Planner implication.** Treat augmentation as a rare, late, climactic device. It needs a subject whose augmented form fits a slow harmonic rhythm.

### 2.4 Double and triple counterpoint; countersubjects

- **Countersubject (CS).** "A counterpoint which accompanies answer or subject systematically, though not of necessity invariably." It must be **invertible**, usually at the octave and sometimes at the 10th or 12th [V, Prout ch. V].
  - It should contrast with the subject in rhythm and character. Bach often uses longer notes against a busy subject, or the reverse.
  - It usually enters with the answer, in the answer's key.
  - A CS first heard with the answer is "usually kept for every later entry".
- **Frequency.** **17 of 48** WTC fugues have **no** countersubject, mostly the fugues that rely on stretto, inversion, canon or diminution instead [V, Prout ch. V].
  - Some fugues introduce the CS late: Fugue 4 (I/4), where it enters at bar 35.
  - Some use two countersubjects together (Fugue 21) or one after another (Fugue 47, the second at the 12th).
  - In I/4 the second and third themes enter at bars 36 and 49 and combine in invertible counterpoint from bar 49 to 88 [V, Wikipedia BWV 849].
- **Reuse in episodes.** In Fugue 12 (I/12), all six episodes are built on the first six notes of the CS [V, Prout ch. V].
- **Triple counterpoint.** With S, CS1 and CS2 all mutually invertible there are 6 vertical permutations. Bach rotates them so that **no voice keeps the same material in consecutive entries**, giving each entry a new registral layout. I/2 is the textbook case [B, Gauldin; Kennan]. Prout's rule that successive groups should not use the same voice order supports this [V, §325].
- **Planner implication.** Represent each entry as a permutation of (S, CS1, CS2, free) over the voices. Avoid repeating the previous permutation, and put the subject in an outer voice for structural entries.

### 2.5 Episodes from subject or countersubject fragments

- **Function.** An episode allows modulation without altering the subject. It provides relief ("lower the temperature"), and it makes the subject's return "more striking" [V, Prout ch. VII; Belkin].
- **Material.** Usually the head or tail of the subject, the countersubject or the codetta, worked by imitation. Rarely fresh material [V, Prout ch. VII]. In II/10 all five episodes recombine "the last part of the subject" [V, Prout §304]. The subject divides into head and tail motives [V, Harrison 2008, "Heads and Tails"; LibreTexts].
- **Construction.** "Construct your episodes sequentially" [V, Prout]. "Most episodes are harmonic sequences drawn from the theme and countersubject" [V, Belkin].
- **Variety.** Each episode should bring something new. A literal transposition of an earlier episode is weak, but repeating an episode **with its parts inverted** (invertible counterpoint) is good [V, Prout ch. VII]. Example: in I/6, the episode at bars 28–34 repeats the motives of the first episode in invertible counterpoint at the octave [V, Marlowe 2020].
- **Texture.** Episodes often use fewer voices: "three parts in a four-part fugue". The voice that rests can then re-enter with the subject as "a special event" [V, Prout; Belkin].
- **Length.** 2–3 bars is typical. Longer episodes are possible, mainly before major arrivals: the return to the tonic, the final section, or the pedal [V, Prout].

### 2.6 Sequences

Concrete WTC examples from I/6 in D minor [V, Marlowe 2020]:

- Bars 9–13: **descending fifths** (I–IV–vii–iii..., realized as parallel sixths over an inner pedal).
- Bars 15–17: **ascending 5–6** (parallel thirds over a pedal A).
- Bars 28–34: **parallel tenths** spanning a descending third-progression.

General norms [B, Gauldin/Kennan, consistent with Prout]:

- **Descending fifths** is the most common type, diatonic or with applied dominants (the chromatic form).
- **Ascending 5–6** is used to climb toward an arrival and raise tension.
- **Descending thirds or steps** (parallel 6/3 chains, "Pachelbel" type) are used for winding down.

Lengths and modulation:

- A sequence usually runs for **2–4 statements** of the model, then breaks off into a cadential approach to the next key.
- The chromatic (applied-dominant) form of a sequence modulates easily. The diatonic form usually returns to the key it started in.
- An episode is **modulating** when it links entries in different keys; this is the norm in the middle section. It is **non-modulating** when it prolongs a key: after the exposition, in the counter-exposition, or before the final entry while dominant tension builds.
- In Prout's analysis of II/10, the first episode (bars 18–23) modulates to G major, and that modulation is what starts the middle section [V, §299].

### 2.7 False entries and partial entries

- **What they are.** Incomplete statements, usually the head only, that anticipate a real entry and "heighten its impact" [V, Wikipedia *Fugue*]. I/2 in C minor has a false entry in the alto before the bass enters [V, Wikipedia *Fugue*].
- **Uses.**
  - Episode imitation of the head, so the listener expects an entry.
  - Truncated stretto entries late in a fugue (I/20, I/8) [V, McDonald].
  - Head-motive "teasing" before the final entry.
- **Planner implication.** Late in an episode, allow 1–2 head-only statements in other voices, then the real entry in a voice that was resting or in an outer voice.

### 2.8 Pedal points

- **Where.** Almost always in the final section: dominant first, then tonic (as in the five-voice C-sharp minor fugue). Close stretti are often built over the pedal [V, Prout §§272, 329].
  - Coda over a dominant pedal: II/10 [V, §305].
  - Pedal under a closing stretto, bars 83–87: Fugue 20 [V, §329].
  - Tonic pedal at the end of I/2, with I, IV and V triads above it [V, Britannica, *pedal point*].
  - Smaller inner or upper pedals also appear inside episodes (I/6, bars 9–17) [V, Marlowe].
- **Frequency.** I found no exact count. Belkin: "not a norm in Bach" [V]. A tonic-pedal coda in the last 2–4 bars is common, especially in minor. A dominant pedal before the final entry is the school (Gedalge/Paris Conservatoire) convention and occurs in a minority of WTC fugues [V, Belkin; B, Gedalge].
- **Length.** Codas over a tonic pedal usually last 2–4 bars. Dominant pedals usually last 2–6 bars, often with a stretto or the final entry above them.

### 2.9 Cadences and section marking

- **Where they go.** Authentic cadences close the exposition (in V or I), the middle-entry groups (in each new key) and the piece.
- **Overlap rule.** A cadence in the middle "should lead directly into a new entry so the music never stops." Prout cites Mozart's *Musical Joke*: a fugue "chopped up by the full cadences into lengths of four bars each" sounds ludicrous [V, Prout §330]. Bach regularly lets the **entry begin on the cadence beat**, in an overlapping voice [V, Marlowe: "a subject can overlap a cadence"].
- **Cadence types.**
  - **PAC (perfect authentic cadence)**: a structural arrival in a new key, usually at the end of a modulating episode, overlapped by the next entry.
  - **IAC or evaded cadence (imperfect authentic, or interrupted at the last moment)**: keeps momentum and lets the music continue past an arrival.
  - **Deceptive cadence (V–vi)**: Prout notes the *inganno* that lets the third voice enter [V, ch. V]. In the *Art of Fugue*, Bach approaches the submediant by V7–VI [V, teoria.com]. Its main late use is the deceptive cadence just before the final entry or coda, which delays the close [V, search summary; B, Kennan].
  - **Half cadence or dominant arrival**: marks the end of an episode that prepares a tonic entry, and the start of a dominant pedal. A fermata on V before the final-section stretto is a known Bach gesture [V, search snippet].
- **Harmony at entries.** Do not harmonize an entry with a root-position tonic. Bach builds tension *into* entries, with a rising line or a suspension elsewhere, "making the new entry seem inevitable" [V, Belkin].

### 2.10 Texture changes and voice re-entry

- **Exposition.** It builds a "textural crescendo". Voices rarely drop out there [V, Belkin].
- **Episodes.** These thin to 2–3 voices. "It would be bad for all the voices to be continually at work throughout" [V, Prout].
- **Re-entry.** A voice returns after a rest, preferably **with the subject** (the "special event"). It is prepared by a rest of at least a bar or two, an entry on a strong beat or immediately after another voice's cadence or suspension resolution, and often a leap into a fresh register [V, Belkin; B, Kennan].
- **Middle entries.** Do not restart each entry from a single voice ("formally monotonous"). Voices without the subject should keep imitating [V, Belkin].
- **End.** The texture thickens (extra voices or chords in 16 of 48 fugues) [V, Prout §306].

### 2.11 Chromaticism

Mostly [B]. Gauldin, Kennan and Ledbetter all describe the following devices:

- **Chromatic descending tetrachord** (lamento bass, 1–7–♭7–6–♭6–5) in the bass or as a countersubject. It is an affect marker in minor and works as a CS line or under a late entry. The WTC I/4 subject and the "cross" motive mark the chromatic and Passion affect [V, Wikipedia BWV 849].
- **Diminished seventh chords** at points of maximum tension: before the final entry, or just before the last cadence, often as vii°7/V or as a common-tone °7 over the tonic pedal. I/4 has a striking dissonance at bar 112 of 115 [V].
- **Neapolitan sixth** in the final cadential approach of minor fugues: it substitutes for iv before V.
- **Chromatic intensification at the end**: ♭7 over a tonic pedal (that is, V7/IV) to touch the subdominant in the coda; I/2 uses a tonic pedal with I, IV and V above it [V, Britannica].
- **Restraint.** Chromaticism is concentrated in the final 15% and in minor-key fugues. Middle episodes use mainly diatonic or applied-dominant sequences.

### 2.12 Suspension chains

These are 7–6 chains in descending sequences, 4–3 at cadences, and 2–3 chains in the bass. They are the standard way to keep episodes moving and to "time an entry to coincide with a suspension elsewhere" [V, Belkin; B, Kennan ch. on episodes]. Use them especially in slow (*stile antico*) fugues (I/4, I/22, II/9) and in parallel-sixths and tenths sequences [V, Marlowe on I/6].

### 2.13 Shaping the climax

- **Accumulation.** Thickening texture, closer or more complete entries, wider register, and the highest soprano note of the piece. Then resolution [V, Belkin].
- **Placement.** Agitation and stretto activity peak in the third quarter (McDonald). The climax of weight comes near the end: in I/22 it is a five-voice stretto at 90% [V, McDonald and Kerman].
- **Register.** The augmented soprano entry in I/8 sits an octave or more above the alto 89% of the time: a registral "spotlight" on the slow subject [V, McDonald].
- **Planner implication.** Plan two peaks: a density or agitation peak at about 55–75% and a register-and-weight peak at about 80–92%. Then relax into the coda.

### 2.14 Tonal answer

- **Rule.** If the subject begins on, or prominently emphasizes, degree 5, the answer is tonal: 5 is answered by 1. This keeps the tonic–dominant frame [V, Wikipedia *Fugue*; LibreTexts 30.8; B, Renwick].
- **Renwick's paradigms.** Renwick classifies subject and answer pairs by scale-degree paradigm. The most common in the WTC is **Paradigm 1 (5-4-3 / 8-8-7)**, used in **22 of 48** subjects, 12 of which have stretto [V, McDonald Fig. 16–17].
- **Planner implication.** The answer form for later dominant-key entries should follow the same paradigm. In middle-section key pairs, use the tonal-answer form when the entry acts as the "answer" of a local pair [V, Wikipedia: an entry "in the dominant of that key when a tonal answer is needed"].

### 2.15 Subdominant entries near the end

- **What it is.** An entry in IV or iv placed just before the tonic return: the classic "subdominant recapitulation" feeling. Prout's example: the episode at bars 44–53, a tenor entry in IV at bars 53–58, then subject and answer in the tonic from bar 59 [V, Prout §294]. A closing section "often... emphasizing the subdominant key" [V, search summary].
- **Why it works.** IV balances the earlier dominant side, and IV leads to V (half cadence or pedal) and then I very naturally.
- **Planner prior.** About 40–60% of fugues [B, estimate; not counted in a source] have an entry in IV/iv in the last third, or a strong tonicization of IV inside the tonic-pedal coda.

### 2.16 Dramatizing the final entry

Typical ways to dramatize the last entry [V, Prout; McDonald; Belkin; Wikipedia]:

- Put it in the **bass** (grounding, as in I/2) or the **soprano** (brilliance).
- Use the full texture, with added voices or chords.
- Precede it by a **rest in the entering voice**, a **dominant preparation** (half cadence, fermata or dominant pedal) or a **deceptive cadence**.
- Accompany it with the most complete invertible combination: all countersubjects together, or a stretto maestrale.
- Follow it with a coda over a tonic pedal and a Picardy third (in minor).

---

## 3. Decision rules for the form planner

The probabilities are priors. Where a number comes from a source, the source follows it in brackets. Otherwise, numbers in brackets are my estimates from the literature as a whole. Tune them against the project's own corpus annotations.

### A. Global skeleton

1. **Three-part layout.**
   - Exposition (all voices, S/A alternation; last entry in an outer voice, p ≈ 0.88 [42/48, Prout]).
   - Optional counter-exposition, p ≈ 0.27 [13/48, Prout]. When present, it stays in I/V with roles swapped, often answer first, and its entries are often closer or inverted.
   - Middle section with 2–5 entry groups [Prout: 1–6].
   - Final section in the tonic.
2. **Section boundaries as fractions of the length.**
   - Exposition and counter-exposition end at 0.20–0.35.
   - Middle section spans 0.30–0.80.
   - Tonic return at 0.65–0.85.
   - Coda covers the last 0.05–0.12.
3. **Episodes.**
   - Place one between entry groups. Default length is 2–3 bars [Prout], with a tail up to 6–8 bars before the tonic return or the final section.
   - Allow 0–1 missing episodes (a direct entry-to-entry link) in a mid-texture position.
   - Total episodic material is about 30–45% of the piece in an episodic fugue, and 0–15% in a stretto or canonic fugue [estimate].

### B. Key plan

4. Middle-entry keys come only from the closely related set:
   - Major: V, vi, iii, IV, ii.
   - Minor: III, v, iv, VI, VII; avoid ii° and ii in minor.
   - Allow one "unrelated" key at most, with probability ≤ 0.05, and only by smooth chromatic voice-leading [Prout §§324–325].
5. **First middle entry.**
   - Minor fugues: III with p ≈ 0.6, v ≈ 0.2, other ≈ 0.2.
   - Major fugues: vi with p ≈ 0.45, V ≈ 0.35, iii or ii ≈ 0.2. Do not use V if the exposition just ended on V.
6. Optionally follow a new-key entry with an entry in the **dominant of that key** (a local subject–answer pair) [Wikipedia *Fugue*], p ≈ 0.4.
7. **Order of keys:** sharp/dominant side early, relative key mid, flat/subdominant side late. **The entry immediately before the tonic return is in IV/iv with p ≈ 0.4–0.5** [Prout §294 example; B].
8. No two successive entry groups in the same key, with the same voice order, or with the same subject voice [Prout §325].

### C. Episodes and sequences

9. **Material.** The subject's head or tail, or a CS fragment, in imitation between 2–3 voices. Choose the fragment by cycling through them so successive episodes differ. Reusing an episode with its voices swapped in invertible counterpoint is allowed and good [Prout].
10. **Sequence type.**
    - Modulating episode: descending fifths (chromatic, with applied dominants) with p ≈ 0.5; descending steps or thirds via 7–6 or 6/3 chains ≈ 0.25; ascending 5–6 ≈ 0.25. Prefer ascending 5–6 for the last episode before a climax.
    - 2–4 statements of the model, then a 1–2 bar cadential break that lands on V of the next key or directly on the entry.
11. **Modulating or not.** Middle-section episodes modulate (target: the next entry's key). The post-exposition episode, the counter-exposition links and the episode before the final section are usually non-modulating or return-to-tonic, the latter often ending on a half cadence.
12. **Texture.** 2–3 active voices in 4-voice episodes, p ≈ 0.7. The voice that rests should be the one carrying the next entry [Prout; Belkin].

### D. Entries

13. **Voice for the entry.**
    - Avoid the voice that had the previous entry.
    - Prefer a voice that has rested for at least 1 bar.
    - Structural entries (first middle entry, tonic return, final entry) go in an outer voice with p ≈ 0.75.
14. **Harmony at entries.** Never start a middle entry on a static root-position tonic after a full stop. Overlap the entry with the cadence of the previous phrase, or enter against a suspension or an approach chord [Belkin; Prout §330].
15. **Countersubject usage.** If the subject has a CS, (a) keep it on at least 80% of later entries and (b) vary the vertical permutation every entry. With p ≈ 0.35 [17/48], plan *no* CS and rely instead on stretto, inversion or free counterpoint.
16. **False entries.** In about 30% of late episodes [estimate], place 1–2 head-only statements before the real entry.

### E. Devices: gating, timing, frequency

17. **Stretto.**
    - Enable only if the subject has an admissible (interval ∈ {P4, P5, P8}, delay of 1 beat to 2 bars) self-overlap.
    - Given that it is enabled, use it in about 44% of fugues overall [21/48, McDonald], and nearly always if the subject is highly stretto-capable.
    - Number of stretto passages: median 3 (mode 2). Most passages have 2 entries (51%), then 3 (23%), then 4 (11%) [McDonald].
    - First stretto at about 0.41 ± 0.23 of the length; the main stretto mass at 0.50–0.75; a final all-voice or maestrale stretto at 0.80–0.92 (weight climax), p ≈ 0.5 given stretto.
    - Do *not* force monotonically shrinking delays. Vary them, and give the final stretto more voices rather than necessarily a shorter delay [McDonald; Belkin].
18. **Inversion.**
    - About 25% of fugues [estimate].
    - First inverted entry after the exposition: in the counter-exposition or the first middle group (0.25–0.5).
    - Can combine with stretto later.
    - Requires the inverted subject to imply good harmony.
19. **Augmentation.**
    - About 8% of fugues [about 4/48].
    - Late only (> 0.7), in the bass or soprano, usually in stretto against normal-speed entries.
20. **Diminution.** About 4%, in the middle or late part, in stretto.
21. **Pedal.**
    - Tonic pedal in the coda with p ≈ 0.5–0.6 [estimate], lasting 2–4 bars, with the subject or its head above it (often soprano), and IV/iv and V7/IV colours over the pedal.
    - Dominant pedal before the final entry or final stretto with p ≈ 0.25–0.35 [estimate; "not a norm in Bach"], lasting 2–6 bars and ending in a half cadence or a direct dominant-to-tonic arrival with the final entry.
    - Pedals before 0.75 should be rare (< 0.1).
22. **Cadence placement.**
    - PAC in each new key at the end of the modulating episode, overlapped by the entry.
    - Evaded cadences inside long middle stretches.
    - One deceptive cadence before the final entry or coda, p ≈ 0.35.
    - Half cadence or fermata on V before the final section, p ≈ 0.3.
    - Final PAC. In minor, a Picardy third with p ≈ 0.85.

### F. Climax and ending

23. **Two-peak plan.**
    - Density or agitation peak at 0.55–0.75: the most stretto, the fastest surface, the thickest imitation.
    - Weight peak at 0.80–0.92: the highest soprano note (enforce a register maximum in this window), full voices, the final entry.
    - Then resolution: thinning or chordal texture, tonic pedal, slower harmonic rhythm.
24. **Final entry.**
    - Bass with p ≈ 0.45, soprano ≈ 0.45, inner ≈ 0.1.
    - Prepare it with a rest in that voice of at least 1 bar, plus either a dominant pedal, a half cadence or a deceptive cadence.
    - Accompany it with the fullest available invertible combination.
25. **Chromatic intensification window: 0.85–1.0 (minor especially).**
    - Neapolitan sixth before the final V, p ≈ 0.3 in minor.
    - One diminished seventh at the point of maximum tension, p ≈ 0.6.
    - Lamento (chromatic descending) bass or line, p ≈ 0.2, or as a planned CS.
26. **Voice count.** Allow an extra voice or a chordal thickening in the last 2–4 bars, p ≈ 0.33 [16/48, Prout].

### G. Planner-level variety constraints

- Never repeat the same (key, voice, CS permutation) triple.
- Never have two consecutive episodes from the same fragment with the same sequence type.
- No PAC with a full stop, meaning all voices resting or a static tonic, before the final cadence.

---

## 4. Mechanical vs. beautiful

| Sounds mechanical or amateurish | Sounds like Bach | Source |
|---|---|---|
| Rigid school road map: exposition, counter-exposition, episodes, stretti in ever-closer order, obligatory dominant pedal | Form grows from the subject's capabilities ("an outgrowth of the material, rather than a mold"); devices used only when the subject invites them | Belkin; McDonald (Bach doesn't narrow stretti systematically); Prout §246 (many great fugues without stretto) |
| Full cadences chopping the piece into 4-bar blocks; music stops before each entry | Cadences overlapped by entries; continuous momentum ("mastery of musical movement") | Prout §330 (Mozart's parody); Belkin |
| Entries harmonized with a static root-position tonic | Entries timed against suspensions or rising lines so they "seem inevitable" | Belkin |
| All voices playing all the time; no rests | Episodes thinned to 2–3 voices; a rested voice returns with the subject as an event | Prout ch. VII, §319; Belkin |
| Each episode a transposed copy of the last | Each episode with new material or a new combination; repeats only with voices swapped (invertible) | Prout ch. VII |
| Entries in the same voice, key or order each time | Rotating voices, keys and CS permutations | Prout §325 |
| Plain chordal accompaniment of the subject; too many parallel thirds and sixths | Independent lines; thirds and sixths used sparingly | Prout §332 |
| Random distant keys, "pulled in by the hair of its head" | Closely related keys, with distant ones "naturally introduced" | Prout §§324–325 |
| Subject cluttered with too many motives | A subject with a clear head and tail that fragment well into episodes | Belkin; Harrison 2008 |
| Even intensity throughout, or a climax by sudden contrast | Cumulative waves of intensity toward a high point, then resolution, with momentum stopped deliberately (pedal, simplification) | Belkin; Kerman via McDonald |
| Stretti or pedals used because the textbook says so | Used "by artistic need"; many of the finest fugues have neither | Belkin; Prout §246 |

---

## Sources

**Read this session [V]:**

- Ebenezer Prout, *Fugue* (1891), Wikisource:
  - ch. V (Countersubject): https://en.wikisource.org/wiki/Fugue_(Prout)/Chapter_5
  - ch. VI (Exposition and counter-exposition): https://en.wikisource.org/wiki/Fugue_(Prout)/Chapter_6
  - ch. VII (Episode): https://en.wikisource.org/wiki/Fugue_(Prout)/Chapter_7
  - ch. VIII (Stretto): https://en.wikisource.org/wiki/Fugue_(Prout)/Chapter_8
  - ch. IX (Middle and final sections): https://en.wikisource.org/wiki/Fugue_(Prout)/Chapter_9
- Kathryn McDonald, *Exploring Stretto: An Investigation into the Use of Stretto in J. S. Bach's Well-Tempered Clavier*, MA thesis, Western University, 2019. A corpus study of all 48 fugues: https://hdl.handle.net/20.500.14721/29801 (PDF: https://uwo.scholaris.ca/bitstreams/cbc60387-6315-4015-9d60-01a05478cc10/download). It includes Renwick's subject paradigms and quotes Kerman, *The Art of Fugue* (2005).
- Alan Belkin, *Principles of Counterpoint: Contrapuntal Forms* (Université de Montréal): https://alanbelkinmusic.com/counterpoint-contrapuntal-forms/
- Sarah Marlowe, analysis of WTC I/6 in D minor, *Music Theory Online* 26.3 (2020): https://www.mtosmt.org/issues/mto.20.26.3/mto.20.26.3.marlowe.html
- Wikipedia, "Fugue": https://en.wikipedia.org/wiki/Fugue ; "Stretto": https://en.wikipedia.org/wiki/Stretto ; BWV 847, 849 and 867 pages.
- Hutchinson, *Music Theory for the 21st-Century Classroom*, §30.8 Fugue Analysis: https://musictheory.pugetsound.edu/mt21c/FugueAnalysis.html
- teoria.com, *Art of Fugue* Contrapunctus XII analysis: https://teoria.com/en/articles/kdf/XII/
- earsense.org WTC pages (summaries of Bruhn's analyses, inversion, augmentation and diminution vocabulary), via search snippets: https://www.earsense.org/Earsense/WTC/
- Britannica, "pedal point" (tonic pedal at the end of WTC I/2).
- Michael Baker, Euromac 2017, on the Dorian middleground in Bach's minor-key fugues (minor supertonic seldom used for entries): https://creaa.unistra.fr/websites/gream/Activites/Euromac_2017_-_Postprint_-_Article_-_BAKER_Michael.pdf (snippet only)

**Book references used as priors [B] (not opened online):**

- Alfred Mann, *The Study of Fugue* (1958/1965)
- Kent Kennan, *Counterpoint* (4th ed., 1999)
- Robert Gauldin, *A Practical Approach to Eighteenth-Century Counterpoint* (1988)
- André Gedalge, *Traité de la fugue* (1901; English translation 1965)
- David Ledbetter, *Bach's Well-tempered Clavier: The 48 Preludes and Fugues* (2002)
- Siglind Bruhn, *J. S. Bach's Well-Tempered Clavier: In-depth Analysis and Interpretation* (1993)
- William Renwick, *Analyzing Fugue: A Schenkerian Approach* (1995)
- D. F. Tovey, notes to the Associated Board edition of the WTC
- Daniel Harrison, "Heads and Tails: Subject Play in Bach's Fugues", *Music Theory Spectrum* 30/1 (2008)

**Could not access** (bot-protected or paywalled):

- Giraud, Groult, Leguy and Levé, "Computational Fugue Analysis", *Computer Music Journal* 39/2 (2015), HAL hal-01113520, with ground truth for 36 fugues by Bach (WTC) and Shostakovich.
- Algomus episode and harmonic-sequence detection papers.
- Algomus fugue reference annotations (algomus.fr/data, gitlab.com/algomus.fr).

## Open questions and suggested follow-up

- Exact WTC counts for inversion, augmentation, pedal points (tonic and dominant), counter-exposition length, subdominant entries in the final third, and the number of middle entries. These could be computed from the Algomus ground truth (if downloadable) or from the project's own form and harmony labels on the WTC subset. That would replace the "[estimate]" priors in section 3 with measured frequencies.
- McDonald's appendix tables list each stretto passage (fugue, bar, voices, pitch interval, time interval). They could seed an admissible-stretto lookup for testing the planner.
