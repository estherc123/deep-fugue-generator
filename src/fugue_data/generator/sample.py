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
                           m["ctx"], 0.0).to(dev)
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
        self.beat = max(1, beat_steps)
        self.cands = MIDI_LO + np.arange(N_NOTES)

    def start_step(self, prev_tokens: np.ndarray, s: int):
        """Advance to step s; prev_tokens is the (samples, nv) row at s - 1."""
        for i in range(prev_tokens.shape[0]):
            for v in range(prev_tokens.shape[1]):
                t = int(prev_tokens[i, v])
                if NOTE0 <= t < NOTE0 + N_NOTES:
                    self.pitch[i, v] = MIDI_LO + t - NOTE0
                    self.onset[i, v] = s - 1
                elif t == STOI["REST"]:
                    self.pitch[i, v] = -1
        self.new[:] = -2
        self.s = s

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
            if self.s % self.beat and max(self.onset[i, v], self.onset[i, w]) % self.beat:
                continue                        # neither attack on a beat
            c = self.cands
            out |= ((np.abs(c - nw) % 12 == ic) & (c != pv)
                    & (np.sign(c - pv) == np.sign(nw - pw)))
        return out


@torch.no_grad()
def generate(model, p: dict, prompt_steps: int, n_samples: int, dev: str,
             temperature: float = 1.0, top_p: float = 0.95, plan: bool = True,
             seed: int = 0, force_entries: bool = True,
             mask_parallels: bool = True) -> tuple[np.ndarray, dict]:
    """(n_samples, n_steps, nv) grids; the first prompt_steps are Bach's.

    With force_entries the plan's subject entries are written in note for note
    and the model writes everything else; with mask_parallels no free note may
    form a parallel fifth or octave against a voice already decided at that
    step (entries included).
    """
    g = torch.Generator(device=dev).manual_seed(seed)
    n, nv = p["n_steps"], p["nv"]
    ctx = model.ctx
    # Prefix and per-position features never depend on the notes, so build the
    # whole-piece sequence once; only the music tokens change while sampling.
    full_ctx = 10 ** 9
    seq0, feats0, loss0 = build_sequence(p, 0, full_ctx, 0, plan)
    pre_len = int((~loss0).sum())
    toks = np.repeat(seq0[None], n_samples, axis=0)
    toks[:, pre_len + prompt_steps * nv:] = STOI["<PAD>"]
    room_steps = (ctx - pre_len) // nv - 1
    hold, rest = STOI["HOLD"], STOI["REST"]
    feats_t = {k: torch.tensor(v, device=dev) for k, v in feats0.items()}
    forced = plan_tokens(p) if (force_entries and plan and p["has_plan"]) else None
    den = int(p["meter_tok"].split("_")[2]) if p["meter_tok"] != "METER_other" else 4
    pm = ParallelMask(n_samples, nv, 16 // den) if mask_parallels else None
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
            a = pre_len + start * nv
            idx = np.concatenate([np.arange(pre_len), np.arange(a, pos)])
            x = torch.tensor(toks[:, idx], device=dev)
            fidx = torch.tensor(idx + 1, device=dev)
            fx = {k: t[fidx].unsqueeze(0).expand(n_samples, -1) for k, t in feats_t.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(x, fx)[:, -1].float()
            # Grammar: only REST / HOLD / NOTE_ON are music; HOLD needs a sounding voice.
            mask = torch.full_like(logits, float("-inf"))
            mask[:, rest] = 0
            mask[:, NOTE0:NOTE0 + N_NOTES] = 0
            prev = toks[:, pos - nv] if s > 0 else np.full(n_samples, rest)
            sounding = torch.tensor(prev != rest, device=dev)
            mask[sounding, hold] = 0
            if pm is not None:
                for i in range(n_samples):
                    b = pm.banned(i, v)
                    if b.any():
                        stats["masked_choices"] += int(b.sum())
                        mask[i, NOTE0:NOTE0 + N_NOTES][torch.tensor(b, device=dev)] = float("-inf")
            logits = (logits + mask) / temperature
            probs = torch.softmax(logits, -1)
            sp, si = probs.sort(-1, descending=True)
            sp = sp * ((sp.cumsum(-1) - sp) < top_p)
            choice = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1,
                                                     generator=g))
            toks[:, pos] = choice[:, 0].cpu().numpy()
            if pm is not None:
                for i in range(n_samples):
                    pm.decide(i, v, int(toks[i, pos]))
    return toks[:, pre_len:pre_len + n * nv].reshape(n_samples, n, nv), stats


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
    ap.add_argument("--out", default="data/review/generator_samples")
    args = ap.parse_args(argv)

    root = _repo_root()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(root / args.model, dev)
    p = next(x for x in load_corpus(root) if x["piece_id"] == args.piece)
    if p["split"] == "train":
        print(f"warning: {args.piece} is a TRAINING piece; this is not a held-out test")
    if args.bars:
        steps = int(np.searchsorted(p["feats"]["bar"], p["feats"]["bar"][0] + args.bars))
        p = {**p, "n_steps": steps, "grid": p["grid"][:steps],
             "feats": {k: v[:steps] for k, v in p["feats"].items()},
             **{k: p[k][:steps] for k in ("role", "to_entry", "to_cad", "cad_key", "pedal")}}

    prompt = first_entry_end(p) if args.prompt == "first_entry" and p["has_plan"] else 0
    grids, gstats = generate(model, p, prompt, args.samples, dev, args.temperature,
                             args.top_p, plan=not args.no_plan, seed=args.seed,
                             force_entries=not args.no_force_entries,
                             mask_parallels=not args.no_mask)

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
            "force_entries": not args.no_force_entries, "mask_parallels": not args.no_mask,
            "plan_source": p.get("plan_source"), **gstats}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
