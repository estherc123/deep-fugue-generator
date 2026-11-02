"""Unit tests for corrupted-Bach pair generation (src/fugue_data/corruption)."""

import json
import random
import sys
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fugue_data.constraints.engine import evaluate, load_profile
from fugue_data.corruption import ops
from fugue_data.corruption.runner import (_profile_counts, corrupt_window,
                                          load_config)
from fugue_data.corruption.windows import (cut, estimate_key, rejoin,
                                           replace_span, voice_events,
                                           windows_for)

CFG = load_config(ROOT)
PROFILE = load_profile(CFG["rule_profile"])


# ---- a small synthetic 3-voice score, 4/4, 10 bars -------------------------- #
def _score(lines: dict[str, list[tuple]]) -> dict:
    voices = []
    for vid, notes in lines.items():
        out = []
        for k, (o, d, m) in enumerate(notes):
            o, d = F(o), F(d)
            out.append({"note_id": f"{vid}:{k}", "voice_id": vid, "onset_q": str(o),
                        "duration_q": str(d), "midi": m, "pitch_class": m % 12,
                        "octave": m // 12 - 1, "tie": None, "grace": False,
                        "measure_index_unfolded": int(o // 4),
                        "measure_number": str(int(o // 4) + 1),
                        "offset_in_measure_q": str(o % 4)})
        voices.append({"voice_id": vid, "notes": out,
                       "mean_midi": sum(n[2] for n in notes) / len(notes)})
    return {"piece_id": "synthetic_test", "movement_type": "fugue",
            "quarter_length_total": "40", "measure_count": 10,
            "meter_map": [{"numerator": 4, "denominator": 4, "onset_q": "0"}],
            "key_signature_map": [{"fifths": 0, "onset_q": "0"}],
            "voices": voices}


def _scale_line(base: int, step_q: str, pattern: list[int], total: int = 40):
    out, t, k = [], F(0), 0
    while t < total:
        d = F(step_q)
        out.append((t, d, base + pattern[k % len(pattern)]))
        t += d
        k += 1
    return out


# C major: a running top line, a middle line that holds notes across bar lines
# (onsets on beat 4, two beats long), and a bass in half notes.
SCORE = _score({
    "voice_1": _scale_line(72, "1/2", [0, 2, 4, 5, 7, 5, 4, 2, 0, -1, 0, 2]),
    "voice_2": [(0, 3, 64)] + [(3 + 2 * k, 2, [65, 64, 62, 64, 67][k % 5])
                                for k in range(18)] + [(39, 1, 60)],
    "voice_3": _scale_line(48, "2", [0, 5, 7, 0, 9, 5, 7, 0]),
})


def _window():
    return next(windows_for(SCORE, 8, 16, 4))


def _ctx(win, **kw):
    ctx = {"entries": [], "abs_entries": [], "voice_count": len(win.voices),
           "piece_events": voice_events(SCORE), "piece_length_q": F(40),
           "local_key": estimate_key(win.voices), "filler_pool": []}
    ctx.update(kw)
    return ctx


# ---- windows ------------------------------------------------------------------ #
def test_window_is_bar_aligned_and_keeps_sounding_time():
    win = _window()
    assert win.start_q % win.bar_q == 0 and win.length_q == 8 * win.bar_q
    for vid, ev in voice_events(SCORE).items():
        want = sum(min(o + d, win.start_q + win.length_q) - max(o, win.start_q)
                   for o, d, _ in ev
                   if o + d > win.start_q and o < win.start_q + win.length_q)
        assert sum(d for _, d, _ in win.voices[vid]) == want


def test_key_estimate_finds_c_major():
    tonic, mode, _ = estimate_key(_window().voices)
    assert (tonic, mode) == (0, "major")


def test_cut_and_rejoin_round_trip():
    ev = _window().voices["voice_2"]
    s, e = F(4), F(12)                      # both edges cut a held note
    _, inside, _ = cut(ev, s, e)
    split = replace_span(ev, s, e, inside)
    assert len(split) > len(ev)             # cutting divided the held notes...
    assert rejoin(split, ev, {s: s, e: e}) == sorted(ev)   # ...and rejoin restores them


def test_rejoin_keeps_genuine_repeated_notes():
    ev = [(F(0), F(1), 60), (F(1), F(1), 60)]       # Bach repeats the pitch
    assert rejoin(ev, ev, {F(1): F(1)}) == ev


# ---- operators ---------------------------------------------------------------- #
def test_shuffle_permutes_whole_bars():
    win = _window()
    cands = ops.registry()["segment_shuffle"](win, _ctx(win), random.Random(1),
                                              CFG["operators"]["segment_shuffle"])
    assert cands
    voices, p = cands[0]
    s, e = map(F, p["span"])
    perm = p["permutation"]
    assert sorted(perm) == list(range(len(perm))) and all(perm[i] != i for i in range(len(perm)))
    # Everything outside the region is untouched, and the region sounds as long.
    for vid, ev in win.voices.items():
        b0, i0, a0 = cut(ev, s, e)
        b1, i1, a1 = cut(voices[vid], s, e)
        assert (b0, a0) == (b1, a1)
        assert sum(d for _, d, _ in i0) == sum(d for _, d, _ in i1)


def test_foreign_key_area_shifts_only_the_span():
    win = _window()
    voices, p = ops.registry()["foreign_key_area"](
        win, _ctx(win), random.Random(1), CFG["operators"]["foreign_key_area"])[0]
    s, e, shift = F(p["span"][0]), F(p["span"][1]), p["shift"]
    for vid, ev in win.voices.items():
        b0, i0, a0 = cut(ev, s, e)
        b1, i1, a1 = cut(voices[vid], s, e)
        assert (b0, a0) == (b1, a1)
        assert [m + shift for _, _, m in i0] == [m for _, _, m in i1]


def test_rhythm_flatten_attacks_on_beats_and_keeps_held_notes():
    win = _window()
    cands = ops.registry()["rhythm_flatten"](win, _ctx(win), random.Random(1),
                                             CFG["operators"]["rhythm_flatten"])
    assert cands
    voices, p = cands[0]
    s, e = map(F, p["span"])
    for vid, ev in voices.items():
        for o, d, m in ev:
            if s < o < e:
                assert o % 1 == 0, (vid, o)          # new attacks only on beats
    # voice_2 holds each note across a bar line; flattening must not re-attack it.
    held = [x for x in win.voices["voice_2"] if x[0] < s < x[0] + x[1]]
    if held:
        o, d, m = held[0]
        assert any(x[0] == o and x[2] == m and x[0] + x[1] > s for x in voices["voice_2"])


def test_subject_return_removal_replaces_the_entry():
    win = _window()
    # Pretend voice_1 carries entry #3 (post-exposition for 3 voices) at bar 3.
    s, e = F(8), F(12)
    abs_s = win.start_q + s
    entries = [("voice_1", s, e, 3)]
    abs_entries = [("voice_1", abs_s, abs_s + 4, 3)]
    cands = ops.registry()["subject_return_removal"](
        win, _ctx(win, entries=entries, abs_entries=abs_entries), random.Random(1),
        CFG["operators"]["subject_return_removal"])
    assert cands
    for voices, p in cands:
        assert p["voice_id"] == "voice_1"
        donor = F(p["donor_start_q"])
        assert donor + 4 <= win.start_q or donor >= win.start_q + win.length_q
        for vid in ("voice_2", "voice_3"):
            assert voices[vid] == win.voices[vid]


def test_subject_return_removal_needs_labels():
    win = _window()
    assert ops.registry()["subject_return_removal"](
        win, _ctx(win), random.Random(1), CFG["operators"]["subject_return_removal"]) == []


# ---- acceptance -------------------------------------------------------------- #
@pytest.mark.parametrize("op", ["segment_shuffle", "foreign_key_area", "rhythm_flatten"])
def test_accepted_corruption_adds_no_hard_violation_and_is_deterministic(op):
    win = _window()
    rec, why = corrupt_window(win, op, _ctx(win), CFG, PROFILE, seed=7)
    assert rec is not None, why
    again, _ = corrupt_window(win, op, _ctx(win), CFG, PROFILE, seed=7)
    assert again == rec
    corrupted = win.copy_with({v: [(F(o), F(d), m) for o, d, m in ev]
                               for v, ev in rec["corrupted"].items()})
    h0 = Counter(v["rule_id"] for v in evaluate(win.to_fragment(), PROFILE)
                 if v["severity"] == "hard")
    h1 = Counter(v["rule_id"] for v in evaluate(corrupted.to_fragment(), PROFILE)
                 if v["severity"] == "hard")
    assert all(n <= h0[k] for k, n in h1.items())
    assert rec["events_changed"] >= CFG["min_events_changed"]


def test_gate_rejects_a_candidate_with_new_parallels(monkeypatch):
    win = _window()

    def bad(win, ctx, rng, cfg):
        # Force voice_1 a fifth above voice_3 throughout: parallel fifths.
        v = dict(win.voices)
        v["voice_1"] = [(o, d, m + 19) for o, d, m in win.voices["voice_3"]]
        return [(v, {"span": [0, 0]})]

    monkeypatch.setitem(ops._REGISTRY, "segment_shuffle", bad)
    rec, why = corrupt_window(win, "segment_shuffle", _ctx(win), CFG, PROFILE, seed=1)
    assert rec is None and why == "new_hard_violation"


# ---- the generated corpus, when present ------------------------------------- #
PAIRS = ROOT / "data" / "generated" / "corrupted" / "pairs.jsonl"


@pytest.mark.slow
@pytest.mark.skipif(not PAIRS.exists(), reason="corrupted pairs not generated")
def test_generated_pairs_inherit_splits_and_validate():
    import jsonschema
    schema = json.loads((ROOT / "schemas" / "corrupted_pair.schema.json").read_text())
    splits = {json.loads(l)["piece_id"]: json.loads(l)["split"]
              for l in (ROOT / "manifests" / "splits.jsonl").read_text().splitlines() if l}
    filler_split = {}
    for f in (ROOT / "data" / "generated" / "valid").glob("shard_*.jsonl"):
        for l in f.read_text().splitlines():
            if l:
                r = json.loads(l)
                filler_split[r["sample_id"]] = r["split"]
    ops_seen = set()
    for line in PAIRS.read_text().splitlines():
        r = json.loads(line)
        jsonschema.validate(r, schema)
        assert r["split"] == splits[r["piece_id"]]
        assert r["preferred"] == "original"
        if r["corruption_id"] == "solver_filler":
            assert filler_split[r["params"]["filler_sample_id"]] == r["split"]
        ops_seen.add(r["corruption_id"])
    assert ops_seen == set(CFG["operators"])
