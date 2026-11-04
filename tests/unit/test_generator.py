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
