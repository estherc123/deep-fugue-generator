"""Generate from the v1 generator, re-composing a held-out fugue from its plan.

The plan (entry schedule, roles, cadences, pedal points, subject) and the time
grid come from a validation/test piece the model never trained on; the model is
prompted with the opening of that piece (by default, through the end of the
first subject entry) and writes everything after it. The real piece is rendered
alongside, so the output can be compared with what Bach did under the same plan.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m fugue_data.generator.sample \
        --piece bach_wtc1_fugue_14 --samples 3

Decoding constraints: HOLD only where the voice is sounding; the plan's subject
entries written in note for note (--no-force-entries to let the model try);
no parallel fifths/octaves (--no-mask to switch off).
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from ..review.export_audio import render_wav
from ..review.render import write_midi
from .encode import MIDI_LO, NOTE0, STOI, STEPS_PER_Q, load_corpus
from .model import FugueGenerator
from .train import build_sequence

N_NOTES = 88


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_model(path: Path, dev: str):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = ck["config"]["model"]
    model = FugueGenerator(len(ck["vocab"]), m["d"], m["layers"], m["heads"],
                           m["ctx"], 0.0, ck.get("features")).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck


def grid_to_score(grid: np.ndarray, piece_id: str) -> dict:
    """(steps, voices) token grid -> canonical-like score for WAV/MIDI."""
    n, nv = grid.shape
    voices = []
    for v in range(nv):
        notes, cur = [], None
        for s in range(n + 1):
            t = int(grid[s, v]) if s < n else STOI["REST"]
            if cur is not None and t != STOI["HOLD"]:
                o, m = cur
                notes.append({"onset_q": str(Fraction(o, STEPS_PER_Q)),
                              "duration_q": str(Fraction(s - o, STEPS_PER_Q)),
                              "midi": m, "tie": None})
                cur = None
            if NOTE0 <= t < NOTE0 + N_NOTES:
                cur = (s, MIDI_LO + t - NOTE0)
        voices.append({"voice_id": f"voice_{v + 1}", "notes": notes})
    return {"piece_id": piece_id, "voices": voices,
            "quarter_length_total": str(Fraction(n, STEPS_PER_Q))}


def plan_tokens(p: dict) -> np.ndarray:
    """(n_steps, nv) forced tokens from the plan's entries; -1 where free.

    An entry that the plan starts mid-note (labels can sit a little after the
    attack) begins with an attack of that pitch rather than a HOLD, so the
    forced line is always playable whatever came before it.
    """
    n, nv = p["n_steps"], p["nv"]
    out = np.full((n, nv), -1, dtype=np.int64)
    pn = p["pnote"]
    for v in range(nv):
        cur = None
        for s in range(n):
            t = int(pn[s, v])
            if t >= 2:
                cur = NOTE0 + t - 2
                out[s, v] = cur
            elif t == 1:
                if cur is None or (s > 0 and pn[s - 1, v] == 0):
                    # entry begins on a held note: attack the sounding pitch
                    g = p["grid"][:s + 1, v]
                    hits = np.flatnonzero((g >= NOTE0) & (g < NOTE0 + N_NOTES))
                    cur = int(g[hits[-1]]) if len(hits) else None
                    out[s, v] = cur if cur is not None else -1
                else:
                    out[s, v] = STOI["HOLD"]
            else:
                cur = None
    return out


class ParallelMask:
    """Masks NOTE_ON choices that would make parallel fifths or octaves.

    Same definition as the hard rule (`constraints.rules._parallel` under the
    default profile): two voices sounding through consecutive simultaneities,
    both moving to a new pitch at the same step, in similar motion, from one
    perfect interval class (unison/octave or fifth) to the same one, with the
    earlier or later attack on a beat.
    """

    def __init__(self, n_samples: int, nv: int, beat_steps: int):
        self.pitch = np.full((n_samples, nv), -1)       # sounding before this step
        self.onset = np.zeros((n_samples, nv), dtype=int)
        self.new = np.full((n_samples, nv), -2)         # decided at this step (-2 = not yet)
        self.prev2 = np.full((n_samples, nv), -1)       # the pitch before the last attack
        # strict: consecutive fifths/octaves in any motion and at any metric
        # position (bach_rules par5/par8), not only similar motion on the beat
        self.strict = False
        # dissonance on a beat obliges the voice's next note to move by step
        # (1: either way, 2: down, the note having been held into the beat)
        self.oblig = np.zeros((n_samples, nv), dtype=np.int8)
        self.pedal = None
        self.ahead = None
        self.lastp = np.full((n_samples, nv), -1)       # pitch before a rest
        self.rest_from = np.full((n_samples, nv), -10 ** 6)
        self.below_next = None                          # (voice -> upcoming written-in pitch)
        self.beat = max(1, beat_steps)
        self.cands = MIDI_LO + np.arange(N_NOTES)

    def start_step(self, prev_tokens: np.ndarray, s: int):
        """Advance to step s; prev_tokens is the (samples, nv) row at s - 1."""
        for i in range(prev_tokens.shape[0]):
            for v in range(prev_tokens.shape[1]):
                t = int(prev_tokens[i, v])
                if NOTE0 <= t < NOTE0 + N_NOTES:
                    if self.onset[i, v] != s - 1 or self.pitch[i, v] != MIDI_LO + t - NOTE0:
                        self.prev2[i, v] = self.pitch[i, v]
                    self.pitch[i, v] = MIDI_LO + t - NOTE0
                    self.onset[i, v] = s - 1
                elif t == STOI["REST"]:
                    if self.pitch[i, v] >= 0:
                        self.lastp[i, v] = self.pitch[i, v]
                        self.rest_from[i, v] = s - 1
                    self.pitch[i, v] = -1
                    self.prev2[i, v] = -1
        self.new[:] = -2
        self.s = s
        for i in range(prev_tokens.shape[0]):
            for v in range(prev_tokens.shape[1]):
                if NOTE0 <= int(prev_tokens[i, v]) < NOTE0 + N_NOTES or int(prev_tokens[i, v]) == STOI["REST"]:
                    self.oblig[i, v] = 0
        b = s - 1
        if b >= 0 and b % self.beat == 0 and not (self.pedal is not None and self.pedal[min(b, len(self.pedal) - 1)] == 2):
            for i in range(prev_tokens.shape[0]):
                live = [v for v in range(prev_tokens.shape[1]) if self.pitch[i, v] >= 0]
                if len(live) < 2:
                    continue
                B = min(self.pitch[i, v] for v in live)
                pcs = {int(self.pitch[i, v]) % 12 for v in live}
                if len(pcs) == 4 and _is_seventh(pcs):
                    continue
                for v in live:
                    x = self.pitch[i, v]
                    if x != B and (x - B) % 12 in (1, 2, 5, 6, 10, 11):
                        self.oblig[i, v] = 2 if self.onset[i, v] < b else 1

    def decide(self, i: int, v: int, tok: int):
        self.new[i, v] = (MIDI_LO + tok - NOTE0 if NOTE0 <= tok < NOTE0 + N_NOTES
                          else self.pitch[i, v] if tok == STOI["HOLD"] else -1)

    def banned(self, i: int, v: int) -> np.ndarray:
        """Boolean over the 88 notes: True = would form a parallel."""
        out = np.zeros(N_NOTES, dtype=bool)
        pv = self.pitch[i, v]
        if pv < 0:
            return out
        for w in range(self.pitch.shape[1]):
            pw, nw = self.pitch[i, w], self.new[i, w]
            if w == v or pw < 0 or nw < 0 or nw == pw:
                continue                        # w not sounding, undecided, or not moving
            ic = abs(pv - pw) % 12
            if ic not in (0, 7):
                continue
            if not self.strict:
                if self.s % self.beat and max(self.onset[i, v], self.onset[i, w]) % self.beat:
                    continue                    # neither attack on a beat
            c = self.cands
            same_dir = (np.sign(c - pv) == np.sign(nw - pw)) if not self.strict else True
            out |= (np.abs(c - nw) % 12 == ic) & (c != pv) & same_dir
        return out


    def crossing_banned(self, i: int, v: int) -> np.ndarray:
        """Boolean over the 88 notes: True = above the nearest sounding voice
        over v, or below the nearest sounding voice under it (voices are
        ordered highest first). Voices already decided at this step count with
        their new pitch, the others with what they are sounding now."""
        out = np.zeros(N_NOTES, dtype=bool)
        nv = self.pitch.shape[1]
        now = lambda w: self.new[i, w] if self.new[i, w] != -2 else self.pitch[i, w]
        hi = next((now(u) for u in range(v - 1, -1, -1) if now(u) >= 0), None)
        lo = next((now(w) for w in range(v + 1, nv) if now(w) >= 0), None)
        if self.below_next is not None:
            # written-in notes about to start in the neighbours bound us too
            if v + 1 < nv and self.below_next.get(v + 1, -1) >= 0:
                lo = max(lo if lo is not None else -1, self.below_next[v + 1])
            if v - 1 >= 0 and self.below_next.get(v - 1, -1) >= 0:
                up = self.below_next[v - 1]
                hi = up if hi is None else min(hi, up)
        if hi is not None:
            out |= self.cands > hi
        if lo is not None:
            out |= self.cands < lo
        return out


    def copy_from(self, src: int, dst: np.ndarray):
        for a in (self.pitch, self.onset, self.new, self.prev2, self.oblig, self.lastp, self.rest_from):
            a[dst] = a[src]

    def melodic_rules(self, i: int, v: int, leap_penalty: float = 2.0,
                      tritone_penalty: float = 1.5, key: tuple | None = None,
                      cadence_to: int | None = None,
                      approach: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(banned, penalty) over the 88 notes for a new attack in voice v.

        Banned (Bach: ~2-3 per 100 bars, rule audit): a leap larger than an
        octave; similar motion into a fifth or octave with the top voice when v
        is the lowest sounding voice and the top voice leaps (direct / hidden
        fifths and octaves in the outer voices).
        Penalised (Bach bends these regularly, so not banned): after a leap of
        a fifth or more, anything but a step back the other way; a tritone leap.
        """
        ban = np.zeros(N_NOTES, dtype=bool)
        pen = np.zeros(N_NOTES, dtype=np.float32)
        c = self.cands
        ban |= (c < 36) | (c > 86)                      # keyboard compass (ML-5)
        nv = self.pitch.shape[1]
        now = lambda w: self.new[i, w] if self.new[i, w] != -2 else self.pitch[i, w]
        if key is not None:
            kt, minor = key
            lt = (kt + 11) % 12
            # no doubled leading tone on the beat (HC-1)
            if self.s % self.beat == 0 and any(now(w) >= 0 and now(w) % 12 == lt
                                               for w in range(nv) if w != v):
                ban |= (c % 12 == lt)
        if approach is not None:
            # a written-in note follows within a beat: the free note before it
            # must not make a forbidden leap into it either
            da = np.abs(approach - c)
            ban |= (da > 12) | (da == 11)
            pen += 1.5 * ((da == 10) | (da == 6))
        # unisons with a neighbouring voice: Bach 1.5% of sounding pairs,
        # generated 4.8% at weight 1.5, 3.1% at 1.5 with the later rules; two
        # voices on one pitch thin the texture (worst when a free voice
        # shadows an entry squeezed into its register: II/22 bars 32-33)
        for w in (v - 1, v + 1):
            if 0 <= w < nv and now(w) >= 0:
                pen += UNISON_PEN * (c == now(w))
        pv = self.pitch[i, v]
        if pv < 0:
            # after a rest of at most a beat the line continues (bach_rules
            # counts leaps across it)
            if self.lastp[i, v] >= 0 and self.s - self.rest_from[i, v] <= self.beat + 1:
                dl = np.abs(c - self.lastp[i, v])
                ban |= (dl > 12) | (dl == 11)
            return ban, pen
        d = c - pv
        ban |= (np.abs(d) > 12) | (np.abs(d) == 11)     # ML-1: > octave, major 7th
        if self.oblig[i, v]:                             # §3.0: resolve by step
            ok = (np.abs(d) >= 1) & (np.abs(d) <= 2)
            if self.oblig[i, v] == 2:
                ok &= d < 0
            pen += 3.0 * (~ok)
        if key is not None:
            kt, minor = key
            if minor and (pv - kt) % 12 in (8, 11):     # ML-1: augmented 2nd b6 <-> #7
                other = (kt + (11 if (pv - kt) % 12 == 8 else 8)) % 12
                ban |= (np.abs(d) == 3) & (c % 12 == other)
            lowest = not any(now(w) >= 0 for w in range(v + 1, nv))
            highest = not any(now(u) >= 0 for u in range(v))
            # §3.10: the leading tone in an outer voice rises to the tonic at V -> I
            if cadence_to is not None and (highest or lowest) and pv % 12 == (cadence_to + 11) % 12:
                ban |= (c != pv + 1)
        pen += tritone_penalty * (np.abs(d) == 6)
        p2 = self.prev2[i, v]
        if p2 >= 0 and abs(pv - p2) >= 7:
            back = (np.sign(d) == -np.sign(pv - p2)) & (np.abs(d) >= 1) & (np.abs(d) <= 2)
            pen += leap_penalty * (~back)
        if p2 >= 0 and abs(pv - p2) > 2:                # ML-3: two leaps one way
            d0 = pv - p2
            span = np.abs(d0 + d)
            bad = (np.sign(d) == np.sign(d0)) & (np.abs(d) > 2) & \
                ((span > 12) | np.isin(span % 12, (1, 2, 6, 10, 11)))
            pen += leap_penalty * bad
        pen += 0.75 * (np.abs(d) == 10)                 # ML-1: minor 7th (A)
        lowest = not any(now(w) >= 0 for w in range(v + 1, nv))
        top = next((u for u in range(v) if now(u) >= 0), None)
        if lowest and top is not None and self.new[i, top] >= 0 and self.pitch[i, top] >= 0:
            du = self.new[i, top] - self.pitch[i, top]
            if abs(du) > 2:
                ban |= (np.sign(d) == np.sign(du)) & np.isin(np.abs(self.new[i, top] - c) % 12, (0, 7))
        highest = not any(now(u) >= 0 for u in range(v))
        bot = next((w for w in range(nv - 1, v, -1) if now(w) >= 0), None)
        if highest and bot is not None and self.new[i, bot] >= 0 and self.pitch[i, bot] >= 0:
            db = self.new[i, bot] - self.pitch[i, bot]
            # the top voice leaps in the bass's direction into a fifth / octave
            ban |= (np.sign(d) == np.sign(db)) & (db != 0) & (np.abs(d) > 2) \
                & np.isin(np.abs(c - self.new[i, bot]) % 12, (0, 7))
        if self.ahead is not None:
            # written-in attacks at the next step in this voice and others:
            # no consecutive fifths / octaves into them
            fv, others = self.ahead
            if fv is not None:
                for w, (xw, fw) in others.items():
                    if w == v or xw < 0 or xw == fw:
                        continue
                    ic = abs(fv - fw) % 12
                    if ic in (0, 7):
                        ban |= (np.abs(c - xw) % 12 == ic) & (c != fv)
        return ban, pen


def _is_seventh(pcs) -> bool:
    for r in pcs:
        for tones in ((0, 4, 7, 10), (0, 3, 6, 9), (0, 3, 6, 10), (0, 3, 7, 10), (0, 4, 7, 11)):
            if {(r + t) % 12 for t in tones} == set(pcs):
                return True
    return False


def _in_scale(m: int, T: int, p: dict, s: int) -> int:
    """Snap m to the planned key's scale at step s (harmonic-minor 7th allowed),
    moving one semitone in the sequence's direction first."""
    k = int(p["key"][min(s, len(p["key"]) - 1)])
    if k <= 0:
        return m
    deg, minor = (k - 1) // 2, (k - 1) % 2
    t = (p["tonic"] + deg) % 12
    scale = {(t + x) % 12 for x in ([0, 2, 3, 5, 7, 8, 10, 11] if minor else [0, 2, 4, 5, 7, 9, 11])}
    if m % 12 in scale:
        return m
    d = -1 if T < 0 else 1
    return m + d if (m + d) % 12 in scale else m - d


def key_bias_table(p: dict) -> np.ndarray:
    """(n_steps, 88): log Krumhansl-Kessler weight of each note in the planned key."""
    from ..corruption.windows import _KK_MAJOR, _KK_MINOR
    prof = [np.log(np.array(x) / np.mean(x)) for x in (_KK_MAJOR, _KK_MINOR)]
    pcs = (MIDI_LO + np.arange(N_NOTES)) % 12
    out = np.zeros((p["n_steps"], N_NOTES), dtype=np.float32)
    for s, k in enumerate(p["key"][:p["n_steps"]]):
        if k > 0:
            deg, minor = (int(k) - 1) // 2, (int(k) - 1) % 2
            tpc = (p["tonic"] + deg) % 12
            out[s] = prof[minor][(pcs - tpc) % 12]
    return out


def chord_bias_table(p: dict) -> np.ndarray:
    """(n_steps, 88): 1 for notes of the planned chord at that step, else 0
    (from the `chord` plan feature; 0 everywhere where there is no chord)."""
    from .labels import QUALS, QUAL_NAMES
    pcs = (MIDI_LO + np.arange(N_NOTES)) % 12
    out = np.zeros((p["n_steps"], N_NOTES), dtype=np.float32)
    if "chord" not in p:
        return out
    for s, c in enumerate(p["chord"][:p["n_steps"]]):
        c = int(c)
        if c >= 2:
            q, r = divmod(c - 2, 12)
            root = (p["tonic"] + r) % 12
            tones = {(root + t) % 12 for t in QUALS[QUAL_NAMES[q]]}
            out[s] = np.isin(pcs, list(tones))
    return out


UNISON_PEN = 3.5
_EXT = {0: [0, 2, 4, 5, 7, 9, 11], 1: [0, 2, 3, 5, 7, 8, 9, 10, 11]}   # minor: all three forms


def out_of_key_table(p: dict) -> np.ndarray:
    """(n_steps, 88): 1 for notes outside the planned key's scale (minor: the
    natural, harmonic and melodic forms together), the planned chord, and the
    key one quarter ahead (so a modulation can be approached); else 0.

    The Krumhansl profile behind key_bias barely separates these in minor (a
    B natural in G minor cost 0.5 logits), which is where the music sounded
    off-key."""
    from .labels import QUALS, QUAL_NAMES
    pcs = (MIDI_LO + np.arange(N_NOTES)) % 12
    n = p["n_steps"]
    out = np.zeros((n, N_NOTES), dtype=np.float32)

    def scale(k):
        k = int(k)
        if k <= 0:
            return set(range(12))
        deg, minor = (k - 1) // 2, (k - 1) % 2
        t = (p["tonic"] + deg) % 12
        return {(t + x) % 12 for x in _EXT[minor]}
    for s in range(n):
        ok = scale(p["key"][s]) | scale(p["key"][min(n - 1, s + STEPS_PER_Q)])
        c = int(p["chord"][s]) if "chord" in p else 0
        if c >= 2:
            q, r = divmod(c - 2, 12)
            ok |= {(p["tonic"] + r + t) % 12 for t in QUALS[QUAL_NAMES[q]]}
        out[s] = ~np.isin(pcs, list(ok))
    return out


@torch.no_grad()
def generate(model, p: dict, prompt_steps: int, n_samples: int, dev: str,
             temperature: float = 1.0, top_p: float = 0.95, plan: bool = True,
             seed: int = 0, force_entries: bool = True,
             mask_parallels: bool = True, guidance: float = 1.0,
             key_bias: float = 0.0, seq_bias: float = 0.0,
             sequences: list[dict] | None = None,
             force_sequences: bool = False, chords: bool = True,
             no_crossing: bool = False, repeat_penalty: float = 0.0,
             chord_bias: float = 0.0, melodic_rules: bool = False,
             bar_select: int = 0, select_lambda: float = 0.5,
             copies: list | None = None, copy_bias: float = 0.0,
             density_target: float | None = None, density_weight: float = 0.5,
             climax: dict | None = None, final_chord: dict | None = None,
             scale_penalty: float = 0.0, beat_chord_bias: float = 0.0) -> tuple[np.ndarray, dict]:
    """(n_samples, n_steps, nv) grids; the first prompt_steps are Bach's.

    With force_entries the plan's subject entries are written in note for note
    and the model writes everything else; with mask_parallels no free note may
    form a parallel fifth or octave against a voice already decided at that
    step (entries included).

    guidance > 1 is classifier-free guidance on the plan: logits are pushed
    away from the plan-free prediction, toward what the plan asks for
    (u + w * (c - u) on log-probabilities). Plan dropout in training is what
    makes the plan-free prediction available from the same model.

    key_bias > 0 adds key_bias * log(Krumhansl profile weight) of the plan's
    local key to every NOTE_ON logit: a soft pull toward the planned key that
    leaves chromatic notes possible.

    seq_bias > 0: inside a planned sequence restatement (`sequences`, default
    the ones found in the plan's source piece), the token this voice wrote one
    unit earlier, transposed by the sequence step, gets +seq_bias (and +-1
    semitone half that, for diatonic adjustment). It steers toward
    restating the model's *own* unit, not Bach's.

    no_crossing: a free note may not rise above the nearest sounding voice
    over it or fall below the nearest one under it (forced entries may still
    cross; if every note is banned the ban is lifted for that choice).

    repeat_penalty > 0 is subtracted from the logit of striking again the
    pitch the voice is sounding now (or just sounded): Bach's fugues repeat 4%
    of their notes, the samples 6-18%.

    melodic_rules: ban leaps over an octave and hidden fifths/octaves in the
    outer voices; penalise unrecovered leaps and tritone leaps
    (`ParallelMask.melodic_rules`).

    bar_select = K > 1: sequential selection by bar. Each requested sample is
    K particles; at every barline the particle whose last bar breaks the
    fewest rules (weighted, `rules_select.bar_score`), with the model's own
    log-probability as tie-breaker (select_lambda per nat), is copied over
    its group, and generation continues from it. A bad bar is replaced as it
    happens instead of spoiling what follows. density_target (attacks per bar,
    all voices): a bar falling short pays density_weight per missing attack,
    since fewer notes otherwise always means fewer rule breaks and a more
    probable (held-note) bar, and selection drifts toward sparse music.

    copy_bias > 0 makes sequence restatements and countersubject copies soft:
    instead of writing the copied note in, its logit gets +copy_bias, so the
    model, the rule masks and bar selection can adjust it to the harmony (a
    literal copy lands in a different harmony and was the main source of
    unresolved dissonances).

    chord_bias > 0 is added to the logits of notes in the planned chord at that
    step: the chord plan re-anchors the harmony every unit. Without it, chord
    agreement with the plan falls from ~75% to ~55% over 20 bars (drift.py).

    force_sequences: the same, as a write-in. Each planned restatement copies
    the model's own previous unit, transposed by the sequence step and snapped
    to the planned key's scale (a diatonic sequence). Where the copy is not
    playable (HOLD after a rest) or would form a parallel, the model samples.
    """
    g = torch.Generator(device=dev).manual_seed(seed)
    n, nv = p["n_steps"], p["nv"]
    K = max(1, bar_select)
    n_out = n_samples
    n_samples = n_out * K
    lp_bar = np.zeros(n_samples)
    if K > 1:
        from .rules_select import bar_score_v2 as bar_score, piece_meta_v2
        meta = piece_meta_v2(p)
        bar_of = p["feats"]["bar"]
    ctx = model.ctx
    # Prefix and per-position features never depend on the notes, so build the
    # whole-piece sequence once; only the music tokens change while sampling.
    full_ctx = 10 ** 9
    seq0, feats0, loss0 = build_sequence(p, 0, full_ctx, 0, plan, chords)
    pre_len = int((~loss0).sum())
    toks = np.repeat(seq0[None], n_samples, axis=0)
    toks[:, pre_len + prompt_steps * nv:] = STOI["<PAD>"]
    room_steps = (ctx - pre_len) // nv - 1
    hold, rest = STOI["HOLD"], STOI["REST"]
    feats_t = {k: torch.tensor(v, device=dev) for k, v in feats0.items()}
    use_cfg = guidance != 1.0 and plan
    if use_cfg:
        seq_u, feats_u, loss_u = build_sequence(p, 0, full_ctx, 0, False)
        pre_u = int((~loss_u).sum())
        toks_u = np.repeat(seq_u[None], n_samples, axis=0)
        feats_ut = {k: torch.tensor(v, device=dev) for k, v in feats_u.items()}
    kb = key_bias_table(p) * key_bias if (key_bias and plan) else None
    if scale_penalty and plan:
        kb = (kb if kb is not None else 0) - scale_penalty * out_of_key_table(p)
    if beat_chord_bias and plan:
        # the planned chord pulls harder on quarter-note onsets, where a clash
        # with an entry is heard most
        on = (np.arange(p["n_steps"]) % STEPS_PER_Q == 0)[:, None]
        kb = (kb if kb is not None else 0) + beat_chord_bias * on * chord_bias_table(p)
    if kb is not None:
        kb = torch.tensor(kb, device=dev, dtype=torch.float32)
    cb = (torch.tensor(chord_bias_table(p) * chord_bias, device=dev, dtype=torch.float32)
          if (chord_bias and plan) else None)
    seq_at = {}
    if (seq_bias or force_sequences) and plan:
        from .sequences import find_sequences
        from .report import beat_steps_of
        for q in (sequences if sequences is not None
                  else find_sequences(p["grid"], beat_steps_of(p))):
            for t in range(q["start"] + q["unit"], q["start"] + q["statements"] * q["unit"]):
                seq_at[(t, q["voice"])] = (q["unit"], q["step"], q["voice"], q["start"] + q["unit"])
    # copies: restate earlier material of another voice (a countersubject),
    # transposed: {voice, start, length, src_voice, src_start, shift}
    for q in (copies or []):
        for t in range(q["start"], min(n, q["start"] + q["length"])):
            seq_at[(t, q["voice"])] = (q["start"] - q["src_start"], q["shift"], q["src_voice"], q["start"])
    forced = plan_tokens(p) if (force_entries and plan and p["has_plan"]) else None
    den = int(p["meter_tok"].split("_")[2]) if p["meter_tok"] != "METER_other" else 4
    pm = (ParallelMask(n_samples, nv, 16 // den)
          if (mask_parallels or no_crossing or repeat_penalty or melodic_rules) else None)
    if pm is not None and melodic_rules:
        pm.strict = True
        pm.pedal = p.get("pedal")
    key_abs = None
    next_forced = None
    fixed_res = None
    if melodic_rules and forced is not None:
        # (step, voice): a written-in note sounds here and its next written-in
        # attack is not a step away, so it cannot resolve a dissonance itself
        fixed_res = np.zeros((n, nv), dtype=bool)
        for v in range(nv):
            cur_m, cur_s = None, None
            atts = [(t, MIDI_LO + int(forced[t, v]) - NOTE0) for t in range(n)
                    if NOTE0 <= int(forced[t, v]) < NOTE0 + N_NOTES]
            for k, (t, m) in enumerate(atts):
                end = atts[k + 1][0] if k + 1 < len(atts) else n
                nxt = atts[k + 1][1] if k + 1 < len(atts) else None
                stuck = nxt is None or abs(nxt - m) > 2 or nxt == m
                if stuck:
                    span = slice(t, end)
                    fixed_res[span, v] = forced[span, v] >= 0
    if melodic_rules and forced is not None:
        # for every (step, voice): the next written-in attack within a beat, if any
        next_forced = np.full((n, nv), -1, dtype=np.int64)
        beat_steps = max(1, 16 // den) * (3 if den >= 8 and int(p["meter_tok"].split("_")[1]) % 3 == 0 else 1) \
            if p["meter_tok"] != "METER_other" else 4
        for v in range(nv):
            nxt_s, nxt_m = None, -1
            for t in range(n - 1, -1, -1):
                tok = int(forced[t, v])
                if NOTE0 <= tok < NOTE0 + N_NOTES:
                    nxt_s, nxt_m = t, MIDI_LO + tok - NOTE0
                if nxt_s is not None and 0 < nxt_s - t <= 4 * beat_steps:
                    next_forced[t, v] = nxt_m
    if melodic_rules and plan:
        kk = p["key"][:n]
        key_abs = [((p["tonic"] + (int(k) - 1) // 2) % 12, bool((int(k) - 1) % 2)) if k > 0
                   else (p["tonic"], p["mode"] == "minor") for k in kk]
        ch_root = [None if int(c) < 2 else (p["tonic"] + (int(c) - 2) % 12) % 12
                   for c in (p["chord"][:n] if "chord" in p else np.zeros(n))]
    stats = {"forced_tokens": 0, "masked_choices": 0}
    music = lambda s, v: pre_len + s * nv + v
    for s in range(prompt_steps, n):
        start = max(0, s - room_steps + 1)
        if pm is not None:
            pm.start_step(toks[:, music(s - 1, 0):music(s - 1, 0) + nv] if s > 0
                          else np.full((n_samples, nv), rest), s)
            if forced is not None:
                for v in range(nv):
                    if forced[s, v] >= 0:
                        for i in range(n_samples):
                            pm.decide(i, v, int(forced[s, v]))
        for v in range(nv):
            pos = music(s, v)                      # index of the token to predict
            if forced is not None and forced[s, v] >= 0:
                toks[:, pos] = forced[s, v]
                stats["forced_tokens"] += n_samples
                continue
            if (force_sequences or copies) and not copy_bias and (s, v) in seq_at and s - seq_at[(s, v)][0] >= 0:
                D, T, vs, c0 = seq_at[(s, v)]
                ok = np.zeros(n_samples, dtype=bool)
                for i in range(n_samples):
                    t0 = int(toks[i, music(s - D, vs)])
                    if t0 == hold and s == c0:
                        # the copy starts inside a held note: attack its pitch
                        for back in range(s - D - 1, -1, -1):
                            tb = int(toks[i, music(back, vs)])
                            if tb != hold:
                                t0 = tb
                                break
                    prev_t = int(toks[i, pos - nv]) if s > 0 else rest
                    if NOTE0 <= t0 < NOTE0 + N_NOTES:
                        m = _in_scale(MIDI_LO + t0 - NOTE0 + T, T, p, s)
                        cand = NOTE0 + m - MIDI_LO
                        if not (NOTE0 <= cand < NOTE0 + N_NOTES):
                            continue
                        if pm is not None and mask_parallels and pm.banned(i, v)[cand - NOTE0]:
                            continue
                        if pm is not None and no_crossing and pm.crossing_banned(i, v)[cand - NOTE0]:
                            continue
                        if pm is not None and melodic_rules:
                            mb, _ = pm.melodic_rules(i, v, key=key_abs[s] if key_abs is not None else None)
                            if mb[cand - NOTE0]:
                                continue
                    elif t0 == hold:
                        if prev_t == rest:
                            continue
                        cand = hold
                    else:
                        cand = rest
                    toks[i, pos] = cand
                    ok[i] = True
                    if pm is not None:
                        pm.decide(i, v, cand)
                stats["seq_forced"] = stats.get("seq_forced", 0) + int(ok.sum())
                if ok.all():
                    continue
                todo = ~ok
            else:
                todo = np.ones(n_samples, dtype=bool)
            a = pre_len + start * nv
            idx = np.concatenate([np.arange(pre_len), np.arange(a, pos)])
            x = torch.tensor(toks[:, idx], device=dev)
            fidx = torch.tensor(idx + 1, device=dev)
            fx = {k: t[fidx].unsqueeze(0).expand(n_samples, -1) for k, t in feats_t.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(x, fx)[:, -1].float()
            if use_cfg:
                pos_u = pre_u + s * nv + v
                toks_u[:, pre_u:pos_u] = toks[:, pre_len:pos]
                idx_u = np.concatenate([np.arange(pre_u), np.arange(pre_u + start * nv, pos_u)])
                xu = torch.tensor(toks_u[:, idx_u], device=dev)
                fiu = torch.tensor(idx_u + 1, device=dev)
                fxu = {k: t[fiu].unsqueeze(0).expand(n_samples, -1) for k, t in feats_ut.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    lu = model(xu, fxu)[:, -1].float()
                lc, lu = logits.log_softmax(-1), lu.log_softmax(-1)
                logits = lu + guidance * (lc - lu)
            # Grammar: only REST / HOLD / NOTE_ON are music; HOLD needs a sounding voice.
            mask = torch.full_like(logits, float("-inf"))
            mask[:, rest] = 0
            mask[:, NOTE0:NOTE0 + N_NOTES] = 0
            prev = toks[:, pos - nv] if s > 0 else np.full(n_samples, rest)
            sounding = torch.tensor(prev != rest, device=dev)
            mask[sounding, hold] = 0
            if pm is not None:
                for i in range(n_samples):
                    b = pm.banned(i, v) if mask_parallels else np.zeros(N_NOTES, dtype=bool)
                    if no_crossing:
                        pm.below_next = None
                        if next_forced is not None:
                            pm.below_next = {w: int(next_forced[s, w]) for w in (v - 1, v + 1)
                                             if 0 <= w < nv and next_forced[s, w] >= 0}
                        c = pm.crossing_banned(i, v)
                        if not (b | c).all():
                            stats["crossing_masked"] = stats.get("crossing_masked", 0) + int((c & ~b).sum())
                            b = b | c
                    if melodic_rules and forced is not None and s % pm.beat == 0:
                        lowest_now = not any(pm.new[i, w] >= 0 or (pm.new[i, w] == -2 and pm.pitch[i, w] >= 0)
                                             for w in range(v + 1, nv))
                        if lowest_now:
                            cands = pm.cands
                            for u in range(v):
                                x = pm.new[i, u] if pm.new[i, u] != -2 else pm.pitch[i, u]
                                if x < 0 or not fixed_res[s, u]:
                                    continue
                                diss = np.isin((x - cands) % 12, (1, 2, 5, 6, 10, 11))
                                logits[i, NOTE0:NOTE0 + N_NOTES] -= 2.5 * torch.tensor(diss, device=dev, dtype=torch.float32)
                    if melodic_rules:
                        kinfo = key_abs[s] if key_abs is not None else None
                        cad = None
                        if key_abs is not None and s > 0 and ch_root[s] is not None \
                                and ch_root[s - 1] is not None and ch_root[s] != ch_root[s - 1] \
                                and ch_root[s - 1] == (ch_root[s] + 7) % 12:
                            cad = ch_root[s]
                        appr = int(next_forced[s, v]) if next_forced is not None and next_forced[s, v] >= 0 else None
                        pm.ahead = None
                        if forced is not None and s + 1 < n:
                            fn = forced[s + 1]
                            if NOTE0 <= int(fn[v]) < NOTE0 + N_NOTES:
                                oth = {}
                                for w in range(nv):
                                    if w != v and NOTE0 <= int(fn[w]) < NOTE0 + N_NOTES:
                                        xw = pm.new[i, w] if pm.new[i, w] != -2 else pm.pitch[i, w]
                                        oth[w] = (int(xw), MIDI_LO + int(fn[w]) - NOTE0)
                                pm.ahead = (MIDI_LO + int(fn[v]) - NOTE0, oth)
                        mb, mp = pm.melodic_rules(i, v, key=kinfo, cadence_to=cad, approach=appr)
                        if not (b | mb).all():
                            b = b | mb
                        else:
                            stats["melodic_relaxed"] = stats.get("melodic_relaxed", 0) + 1
                        if mp.any():
                            logits[i, NOTE0:NOTE0 + N_NOTES] -= torch.tensor(mp, device=dev)
                    if final_chord is not None and s >= final_chord["from"]:
                        # the final chord's pitch classes take precedence over
                        # every other ban: relax among chord tones, never allow
                        # a note outside the chord
                        ok_pc = final_chord["bass"] if v == nv - 1 else final_chord["pcs"]
                        in_chord = np.array([((MIDI_LO + k) % 12) in ok_pc for k in range(N_NOTES)])
                        if not (in_chord & ~b).any():
                            b = ~in_chord
                            stats["final_relaxed"] = stats.get("final_relaxed", 0) + 1
                        else:
                            b = b | ~in_chord
                    if b.any():
                        stats["masked_choices"] += int(b.sum())
                        pr_ = torch.softmax(logits[i], -1)[NOTE0:NOTE0 + N_NOTES]
                        stats["banned_mass"] = stats.get("banned_mass", 0.0) + float(pr_[torch.tensor(b, device=dev)].sum())
                        mask[i, NOTE0:NOTE0 + N_NOTES][torch.tensor(b, device=dev)] = float("-inf")
                    stats["decisions"] = stats.get("decisions", 0) + 1
                    if repeat_penalty:
                        last = pm.pitch[i, v]
                        if last >= 0:
                            logits[i, NOTE0 + last - MIDI_LO] -= repeat_penalty
            if kb is not None:
                logits[:, NOTE0:NOTE0 + N_NOTES] += kb[s]
            if cb is not None:
                logits[:, NOTE0:NOTE0 + N_NOTES] += cb[s]
            if final_chord is not None and s >= final_chord["from"]:
                # the last bar: only notes of the final chord, the bass on its root
                pcs_ok = final_chord["bass"] if v == nv - 1 else final_chord["pcs"]
                bad_pc = torch.tensor([((MIDI_LO + k) % 12) not in pcs_ok for k in range(N_NOTES)], device=dev)
                logits[:, NOTE0:NOTE0 + N_NOTES][:, bad_pc] = -1e9
                if s == final_chord["from"]:
                    # every voice strikes the chord on the downbeat: no held-over
                    # note from outside the chord, no rest
                    logits[:, hold] = -1e9
                    logits[:, rest] = -1e9
                elif v == nv - 1:
                    logits[:, rest] = -1e9                # the bass keeps the root to the end
            if climax is not None and v == 0:
                # soprano register: 2 logits per semitone above the ceiling;
                # near the climax, a small pull toward the top
                c_ = MIDI_LO + torch.arange(N_NOTES, device=dev)
                ceil_ = int(climax["ceiling"][s])
                logits[:, NOTE0:NOTE0 + N_NOTES] -= 3.0 * (c_ - ceil_).clamp(min=0).float()
                if abs(s - climax["step"]) <= climax.get("window", 32):
                    logits[:, NOTE0:NOTE0 + N_NOTES] += 2.0 * ((c_ >= climax["peak"] - 3) & (c_ <= climax["peak"])).float()
            if copy_bias and (s, v) in seq_at and s - seq_at[(s, v)][0] >= 0:
                D, T, vs, c0 = seq_at[(s, v)]
                for i in range(n_samples):
                    t0 = int(toks[i, music(s - D, vs)])
                    if t0 == hold and s == c0:
                        for back in range(s - D - 1, -1, -1):
                            tb = int(toks[i, music(back, vs)])
                            if tb != hold:
                                t0 = tb
                                break
                    if NOTE0 <= t0 < NOTE0 + N_NOTES:
                        m = _in_scale(MIDI_LO + t0 - NOTE0 + T, T, p, s)
                        cand = NOTE0 + m - MIDI_LO
                        if NOTE0 <= cand < NOTE0 + N_NOTES:
                            logits[i, cand] += copy_bias
                    else:
                        logits[i, t0] += copy_bias
                stats["copy_steered"] = stats.get("copy_steered", 0) + n_samples
            if seq_bias and (s, v) in seq_at:
                D, T, _, _ = seq_at[(s, v)]
                if s - D >= 0:
                    for i in range(n_samples):
                        t0 = int(toks[i, music(s - D, v)])
                        if NOTE0 <= t0 < NOTE0 + N_NOTES:
                            tgt = t0 + T
                            for off, w in ((0, 1.0), (-1, 0.5), (1, 0.5)):
                                if NOTE0 <= tgt + off < NOTE0 + N_NOTES:
                                    logits[i, tgt + off] += w * seq_bias
                        else:
                            logits[i, t0] += seq_bias
                    stats["seq_steered"] = stats.get("seq_steered", 0) + n_samples
            raw_lp = torch.log_softmax(logits + mask, -1) if K > 1 else None
            logits = (logits + mask) / temperature
            probs = torch.softmax(logits, -1)
            sp, si = probs.sort(-1, descending=True)
            sp = sp * ((sp.cumsum(-1) - sp) < top_p)
            choice = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1,
                                                     generator=g))
            ch = choice[:, 0].cpu().numpy()
            if melodic_rules and pm is not None:
                bad = (mask[torch.arange(n_samples, device=dev), choice[:, 0]] == float("-inf")).cpu().numpy()
                if (bad & todo).any():
                    stats["banned_sampled"] = stats.get("banned_sampled", 0) + int((bad & todo).sum())
            toks[todo, pos] = ch[todo]
            if raw_lp is not None:
                lp_bar[todo] += raw_lp[torch.arange(n_samples, device=dev), choice[:, 0]].cpu().numpy()[todo]
            if pm is not None:
                for i in np.flatnonzero(todo):
                    pm.decide(i, v, int(toks[i, pos]))
        # ---- bar-level selection ------------------------------------------- #
        if K > 1 and (s + 1 == n or bar_of[s + 1] != bar_of[s]):
            b0 = int(np.searchsorted(bar_of, bar_of[s]))
            grid_now = toks[:, pre_len:pre_len + (s + 1) * nv].reshape(n_samples, s + 1, nv)
            for gi in range(n_out):
                idx = np.arange(gi * K, (gi + 1) * K)
                a0 = max(prompt_steps, b0)
                scale = (s + 1 - b0) / max(1, int(np.searchsorted(bar_of, bar_of[s] + 1)) - b0) if density_target else 1
                def dens(i):
                    if not density_target:
                        return 0.0
                    seg = grid_now[i][a0:s + 1]
                    att = int(((seg >= NOTE0) & (seg < NOTE0 + N_NOTES)).sum())
                    want = density_target * (s + 1 - a0) / max(1, s + 1 - b0) * scale
                    return density_weight * max(0.0, want - att)
                scores = [bar_score(grid_now[i], a0, s + 1, meta) - select_lambda * lp_bar[i] + dens(i)
                          for i in idx]
                best = idx[int(np.argmin(scores))]
                stats.setdefault("bar_rule_cost", []).append(round(float(min(scores) + select_lambda * lp_bar[best]), 2))
                others = idx[idx != best]
                toks[others, pre_len:pre_len + (s + 1) * nv] = toks[best, pre_len:pre_len + (s + 1) * nv]
                if pm is not None:
                    pm.copy_from(best, others)
            lp_bar[:] = 0
    out = toks[:, pre_len:pre_len + n * nv].reshape(n_samples, n, nv)
    if K > 1:
        out = out[::K]
        stats["bar_select"] = K
    return out, stats


def first_entry_end(p: dict) -> int:
    """Step at which the first labelled subject entry finishes (0 if none)."""
    roles = p["role"]
    hits = np.argwhere(roles == 2)
    if not len(hits):
        return 0
    s0, v0 = hits[0]
    after = np.flatnonzero(roles[s0:, v0] != 2)
    return int(s0 + after[0]) if len(after) else int(p["n_steps"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/models/generator_v2.pt")
    ap.add_argument("--piece", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--bars", type=int, default=None,
                    help="generate only this many bars in total (default: whole piece)")
    ap.add_argument("--prompt", choices=["first_entry", "none"], default="first_entry")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--no-plan", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-force-entries", action="store_true",
                    help="let the model write the entries itself (v1 behaviour)")
    ap.add_argument("--no-mask", action="store_true", help="no parallel-5th/8ve masking")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="classifier-free guidance weight on the plan (1 = off)")
    ap.add_argument("--key-bias", type=float, default=0.0,
                    help="soft pull toward the plan's local key (0 = off)")
    ap.add_argument("--seq-bias", type=float, default=0.0,
                    help="pull toward restating the model's own unit in planned sequences")
    ap.add_argument("--force-sequences", action="store_true",
                    help="write planned sequence restatements in from the model's own unit")
    ap.add_argument("--no-crossing", action="store_true",
                    help="free notes may not cross the neighbouring voices")
    ap.add_argument("--repeat-penalty", type=float, default=0.0,
                    help="logit penalty on re-striking the voice's current pitch")
    ap.add_argument("--n-best", type=int, default=0,
                    help="generate this many and keep the --samples best by the critic")
    ap.add_argument("--critic", default="data/models/critic_v1.pt")
    ap.add_argument("--melodic-rules", action="store_true",
                    help="ban leaps > octave and outer hidden 5ths/8ves; penalise unrecovered leaps")
    ap.add_argument("--bar-select", type=int, default=0,
                    help="K candidates per bar; keep the one breaking the fewest rules")
    ap.add_argument("--chord-bias", type=float, default=0.0,
                    help="logit bonus for notes of the planned chord")
    ap.add_argument("--no-chords", action="store_true",
                    help="plan without the chord / texture layer (v5+ models)")
    ap.add_argument("--out", default="data/review/generator_samples")
    args = ap.parse_args(argv)

    root = _repo_root()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(root / args.model, dev)
    p = next(x for x in load_corpus(root, baroque=True, labels=True)
             if x["piece_id"] == args.piece)
    if p["split"] == "train":
        print(f"warning: {args.piece} is a TRAINING piece; this is not a held-out test")
    if args.bars:
        steps = int(np.searchsorted(p["feats"]["bar"], p["feats"]["bar"][0] + args.bars))
        p = {**p, "n_steps": steps, "grid": p["grid"][:steps],
             "feats": {k: v[:steps] for k, v in p["feats"].items()},
             **{k: p[k][:steps] for k in ("role", "to_entry", "to_cad", "cad_key", "pedal",
                                          "pnote", "epc", "key", "next_key", "to_key", "seq",
                                          "chord", "next_chord", "bass", "active") if k in p}}

    prompt = first_entry_end(p) if args.prompt == "first_entry" and p["has_plan"] else 0
    n_gen = max(args.samples, args.n_best)
    grids, gstats = generate(model, p, prompt, n_gen, dev, args.temperature,
                             args.top_p, plan=not args.no_plan, seed=args.seed,
                             force_entries=not args.no_force_entries,
                             mask_parallels=not args.no_mask, guidance=args.guidance,
                             key_bias=args.key_bias, seq_bias=args.seq_bias,
                             force_sequences=args.force_sequences, chords=not args.no_chords,
                             no_crossing=args.no_crossing, repeat_penalty=args.repeat_penalty,
                             chord_bias=args.chord_bias, melodic_rules=args.melodic_rules,
                             bar_select=args.bar_select)
    critic_scores = None
    if args.n_best > args.samples:
        from .critic import load_critic, score_grid
        critic = load_critic(root / args.critic, dev)
        sc = [score_grid(critic, g, prompt, dev) for g in grids]
        order = np.argsort(sc)[::-1][:args.samples]
        critic_scores = {"all": [round(x, 4) for x in sc], "kept": [int(i) for i in order]}
        grids = grids[order]

    out = root / args.out / args.piece
    out.mkdir(parents=True, exist_ok=True)
    real = grid_to_score(p["grid"], args.piece)
    render_wav(real, out / "bach.wav")
    write_midi(real, Fraction(0), Fraction(p["n_steps"], STEPS_PER_Q), out / "bach.mid")
    for k in range(args.samples):
        sc = grid_to_score(grids[k], f"{args.piece}:gen{k}")
        render_wav(sc, out / f"generated_{k + 1}.wav")
        write_midi(sc, Fraction(0), Fraction(p["n_steps"], STEPS_PER_Q),
                   out / f"generated_{k + 1}.mid")
        np.save(out / f"generated_{k + 1}.npy", grids[k])
    np.save(out / "forced.npy", plan_tokens(p) if p["has_plan"] else np.full(p["grid"].shape, -1))
    meta = {"piece": args.piece, "split": p["split"], "model": args.model,
            "model_steps_run": ck["steps_run"], "model_best_val_ppl": round(ck["best_val_ppl"], 4), "prompt_steps": prompt,
            "prompt_quarters": prompt / STEPS_PER_Q, "n_steps": p["n_steps"],
            "plan": not args.no_plan, "temperature": args.temperature,
            "top_p": args.top_p, "seed": args.seed,
            "force_entries": not args.no_force_entries, "guidance": args.guidance, "key_bias": args.key_bias, "seq_bias": args.seq_bias, "force_sequences": args.force_sequences, "mask_parallels": not args.no_mask,
            "chords": not args.no_chords, "no_crossing": args.no_crossing,
            "repeat_penalty": args.repeat_penalty, "n_best": args.n_best,
            "chord_bias": args.chord_bias, "melodic_rules": args.melodic_rules,
            "bar_select": args.bar_select,
            "critic_scores": critic_scores,
            "plan_source": p.get("plan_source"), **gstats}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
