"""Entry detector, parallel mask and memorisation index (no GPU, no data)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fugue_data.generator.encode import MIDI_LO, NOTE0, STOI  # noqa: E402
from fugue_data.generator.entries import as_annotations, detect  # noqa: E402
from fugue_data.generator.memorization import Index, score_grid  # noqa: E402
from fugue_data.generator.sample import ParallelMask  # noqa: E402

# C major scale steps: midi -> diatonic step
_STEP = {60: 35, 62: 36, 64: 37, 65: 38, 67: 39, 69: 40, 71: 41, 72: 42, 74: 43,
         76: 44, 77: 45, 79: 46, 48: 28, 50: 29, 52: 30, 53: 31, 55: 32, 57: 33,
         59: 34, 43: 25, 45: 26, 47: 27}
SUBJ = [60, 62, 64, 65, 67, 64, 65, 62, 60]


def _voice(vid, notes):
    return {"voice_id": vid, "notes": [
        {"onset_q": str(o), "duration_q": "1", "midi": m, "diatonic_step": _STEP[m],
         "tie": None, "grace": False} for o, m in notes]}


def _fugue():
    # voice_1: subject at 0; voice_2: answer a fifth up at 9; voice_1 free after;
    # voice_2 enters with the subject again at 30 an octave lower.
    v1 = [(k, m) for k, m in enumerate(SUBJ)] + [(9 + k, m) for k, m in
                                                 enumerate([59, 60, 62, 59, 60, 64, 62, 60, 59])]
    ans = [m + 7 for m in SUBJ]
    v2 = [(9 + k, m) for k, m in enumerate(ans)] + [(30 + k, m - 12) for k, m in enumerate(SUBJ)]
    return {"piece_id": "toy", "voices": [_voice("voice_1", v1), _voice("voice_2", v2)]}


def test_detect_finds_exposition_and_return():
    det = detect(_fugue())
    got = {(e["voice_id"], e["onset_q"], e["transpose"]) for e in det["entries"]}
    assert ("voice_1", "0", 0) in got
    assert ("voice_2", "9", 7) in got
    assert ("voice_2", "30", -12) in got
    ann = as_annotations(det, "toy")
    funcs = {(a["voice_id"], a["onset_q"]): a["entry_function"] for a in ann}
    assert funcs[("voice_2", "9")] == "answer"
    assert funcs[("voice_2", "30")] == "subject"


def test_parallel_mask_bans_parallel_fifth_only():
    pm = ParallelMask(1, 2, beat_steps=4)
    prev = np.array([[NOTE0 + 67 - MIDI_LO, NOTE0 + 60 - MIDI_LO]])   # G over C: a fifth
    pm.start_step(prev, 4)
    pm.decide(0, 1, NOTE0 + 62 - MIDI_LO)                             # lower voice C -> D
    banned = pm.banned(0, 0)
    assert banned[69 - MIDI_LO]          # G -> A: parallel fifth with D
    assert not banned[65 - MIDI_LO]      # G -> F: a third, fine
    assert not banned[67 - MIDI_LO]      # G held/repeated: not moving


def test_memorisation_index_finds_copied_run():
    rng = np.random.default_rng(0)
    g = np.full((64, 1), STOI["REST"])
    g[:, 0] = NOTE0 + 30 + rng.integers(0, 20, 64)
    idx = Index([g])
    assert score_grid(idx, g, run=12)["longest_run"] >= 60
    other = np.full((64, 1), STOI["REST"])
    other[:, 0] = NOTE0 + 30 + rng.integers(0, 20, 64)
    assert score_grid(idx, other, run=12)["longest_run"] < 6


def _line_grid(pitches_per_step):
    g = np.full((len(pitches_per_step), 1), STOI["REST"])
    for s, m in enumerate(pitches_per_step):
        g[s, 0] = STOI["HOLD"] if m is None else NOTE0 + m - MIDI_LO
    return g


def test_sequence_detector_finds_descending_sequence_not_scale():
    from fugue_data.generator.sequences import coverage, find_sequences
    unit = [72, 74, 71, 72]                       # turn figure, one beat of 16ths
    seq = sum(([m - 2 * k for m in unit] for k in range(3)), []) + [60] * 4
    q = find_sequences(_line_grid(seq), beat_steps=4)
    assert q and q[0]["statements"] == 3 and q[0]["step"] == -2
    scale = list(range(72, 56, -1))               # a plain run is not a sequence
    assert coverage(_line_grid(scale), beat_steps=4) == 0.0


# --------------------------------------------------------------------------- #
# Form / harmony labels (labels.py)
# --------------------------------------------------------------------------- #
def test_chords_cadence_and_pedal():
    from fugue_data.generator.labels import cadences, chords, pedals
    S = 4                                        # steps per quarter
    bar = 16
    # Four voices, one chord per quarter: | C C F G7 | C C C C |, with the bass
    # holding a tonic pedal under bar 3's I-IV-V7-I.
    prog = [("C", 48, (60, 64, 67)), ("C", 48, (60, 64, 67)), ("F", 53, (60, 65, 69)),
            ("G7", 43, (59, 62, 65)), ("C", 48, (64, 67, 72)), ("C", 48, (55, 64, 72)),
            ("C", 48, (60, 64, 67)), ("C", 48, (60, 64, 67)),
            ("C", 36, (60, 64, 67)), ("F", 36, (60, 65, 69)), ("G7", 36, (59, 65, 67)),
            ("C", 36, (60, 64, 67)), ("F", 36, (60, 65, 69)), ("G7", 36, (59, 62, 65)),
            ("C", 36, (60, 64, 72)), ("C", 36, (60, 64, 72))]
    voices = [[], [], [], []]
    for i, (_, b, upper) in enumerate(prog):
        s = i * S
        for v, m in enumerate(sorted(upper, reverse=True)):
            voices[v].append((s, s + S, m))
        if i >= 8:                               # pedal: one long C2
            if i == 8:
                voices[3].append((s, s + 8 * S, b))
        else:
            voices[3].append((s, s + S, b))
    n = len(prog) * S
    ch = chords(voices, n, S, bar)
    roots = [c["root"] for c in ch[:8]]
    assert roots == [0, 0, 5, 7, 0, 0, 0, 0]
    assert ch[3]["qual"] == "dom7"
    cads = cadences(ch, voices, S, bar, (4, 4), 0)
    assert any(c["step"] == 4 * S and c["type"] == "PAC" and c["key_area"] == "I" for c in cads)
    peds = pedals(voices, ch, S, bar, 0)
    assert any(p["start"] == 8 * S and p["degree"] == "I" for p in peds)


def test_dcml_key_arithmetic():
    from fugue_data.generator.labels import _local_tonic
    assert _local_tonic("F", "I") == (5, False)
    assert _local_tonic("F", "V") == (0, False)
    assert _local_tonic("g", "III") == (10, False)      # B-flat major in G minor
    assert _local_tonic("C", "vi") == (9, True)
    assert _local_tonic("C", "V/V") == (2, False)


def test_crossing_mask_bounds_by_neighbours():
    pm = ParallelMask(1, 3, 4)
    row = np.array([[NOTE0 + 72 - MIDI_LO, NOTE0 + 64 - MIDI_LO, NOTE0 + 55 - MIDI_LO]])
    pm.start_step(row, 1)
    b = pm.crossing_banned(0, 1)                  # middle voice: between 55 and 72
    allowed = MIDI_LO + np.flatnonzero(~b)
    assert allowed.min() == 55 and allowed.max() == 72
    pm.decide(0, 0, NOTE0 + 67 - MIDI_LO)         # top voice moves down to 67 first
    assert (MIDI_LO + np.flatnonzero(~pm.crossing_banned(0, 1))).max() == 67


def test_critic_window_grid():
    from fugue_data.generator.critic import window_grid
    g = window_grid({"voice_1": [["0", "1", 72], ["1", "1/2", 74]],
                     "voice_2": [["0", "2", 60]]}, __import__("fractions").Fraction(2))
    assert g.shape == (8, 2)
    assert g[0, 0] == NOTE0 + 72 - MIDI_LO and g[4, 0] == NOTE0 + 74 - MIDI_LO
    assert g[6, 0] == STOI["REST"] and (g[1:8, 1] == STOI["HOLD"]).all()


# --------------------------------------------------------------------------- #
# Rules, tonal answer, harmony planner
# --------------------------------------------------------------------------- #
def _grid(lines, n):
    """lines: per voice [(start, end, midi)] -> token grid."""
    g = np.full((n, len(lines)), STOI["REST"], dtype=np.int16)
    for v, ev in enumerate(lines):
        for s, e, m in ev:
            g[s, v] = NOTE0 + m - MIDI_LO
            g[s + 1:e, v] = STOI["HOLD"]
    return g


def test_bach_rules_parallels_and_resolution():
    from fugue_data.generator.bach_rules import check
    # C5/F4 -> D5/G4: parallel fifths, similar motion
    g = _grid([[(0, 4, 72), (4, 8, 74)], [(0, 4, 65), (4, 8, 67)]], 8)
    assert any(r == "par5" for r, _, _ in check(g))
    # contrary octaves C5/C4 -> G5/G3 are consecutive octaves too
    g = _grid([[(0, 4, 72), (4, 8, 79)], [(0, 4, 60), (4, 8, 55)]], 8)
    assert any(r == "par8" for r, _, _ in check(g))
    # suspension: F held over a C bass into the beat, resolving down to E: fine
    g = _grid([[(0, 8, 65), (8, 12, 64)], [(0, 4, 57), (4, 12, 48)]], 12)
    assert not any(r == "strong_diss" for r, _, _ in check(g))
    # the same F leaping away instead of resolving: an unresolved dissonance
    g = _grid([[(0, 8, 65), (8, 12, 72)], [(0, 4, 57), (4, 12, 48)]], 12)
    assert any(r == "strong_diss" for r, _, _ in check(g))


def test_tonal_answer_swaps_head_dominant():
    from fractions import Fraction as Fr
    from fugue_data.generator.form_planner import tonal_answer
    # C major subject G4 C5 B4 C5 D5: the head's G (degree 5) is answered by C, not D
    subj = [(Fr(k), Fr(1), m) for k, m in enumerate([67, 72, 71, 72, 74])]
    ans = tonal_answer(subj, 0, "major", 74)
    assert ans[0] % 12 == 0          # G -> C (tonal), not G -> D (real)
    assert ans[1] % 12 == 7          # C -> G


def test_harmony_planner_honours_cadence():
    from fugue_data.generator import harmony
    if not harmony.MODEL.exists():
        pytest.skip("harmony model not fitted")
    rng = np.random.default_rng(0)
    U = 8
    allowed = [None] * U
    allowed[U - 1] = harmony.state_ids(0, ("maj",), (0,))
    allowed[U - 2] = harmony.state_ids(7, ("maj", "dom7"), (0,))
    st = harmony.plan_chords(U, [(0, False)] * U, [set()] * U, allowed, rng)
    assert st[-1][:2] == (0, "maj") and st[-2][0] == 7


def test_repair_fixes_parallel_fifths():
    from fugue_data.generator.bach_rules import check
    from fugue_data.generator.repair import repair
    # free parallel fifths C5/F4 -> D5/G4; the repair must move one of the notes
    g = _grid([[(0, 4, 72), (4, 8, 74), (8, 12, 72)], [(0, 4, 65), (4, 8, 67), (8, 12, 65)]], 12)
    fixed = np.zeros(g.shape, dtype=bool)
    kw = {"meter": (4, 4), "tonic": 5, "mode": "major", "keys": None}
    g2, changed = repair(g, fixed, kw, 16)
    assert changed >= 1
    assert not any(r in ("par5", "par8") for r, _, _ in check(g2))
