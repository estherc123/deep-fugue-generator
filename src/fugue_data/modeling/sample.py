"""Sample new counterpoint directly from the Bach style prior.

This is the inverse of best-of-N: instead of the solver proposing and the prior
ranking, the prior proposes. Whether the result can then pass the hard rule gates
is the open question -- this module only generates and decodes; it makes no claim
about legality.

Two sampling modes, and the difference matters when reading the output:

  free        -- every token sampled, including BAR / BEAT_n / VOICE_n. Tests
                 whether the prior learned the grid format at all. Malformed
                 output here is a fact about the model.
  scaffolded  -- the grid's structural skeleton is emitted deterministically and
                 only the content slots (NOTE_ON / HOLD / REST) are sampled. The
                 skeleton is a fixed function of (voice count, meter, length), so
                 supplying it removes nothing musical and guarantees the sample
                 decodes. This is the honest mode for listening.

Decoding reconstructs durations, which `tokenization.runner.decode_grid` does not
(it recovers attacks only, being a round-trip proof rather than a player).
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import torch

from .best_of_n import PriorScorer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _content_ids(scorer: PriorScorer) -> list[int]:
    return [i for i, t in enumerate(scorer.tokens) if scorer.content[i]]


@torch.no_grad()
def _next(scorer: PriorScorer, ids: list[int], allowed: list[int] | None,
          temperature: float, top_p: float, gen: torch.Generator) -> int:
    ctx = ids[-scorer.ctx:]
    x = torch.tensor(ctx, dtype=torch.long).unsqueeze(0).to(scorer.dev)
    logits = scorer.model(x)[0, -1].float().cpu()
    if allowed is not None:
        mask = torch.full_like(logits, float("-inf"))
        mask[allowed] = 0.0
        logits = logits + mask
    logits = logits / max(1e-6, temperature)
    probs = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1:
        srt, idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(srt, dim=-1)
        keep = cum <= top_p
        keep[0] = True                      # never empty the candidate set
        srt = srt * keep
        srt = srt / srt.sum()
        pick = torch.multinomial(srt, 1, generator=gen).item()
        return int(idx[pick])
    return int(torch.multinomial(probs, 1, generator=gen).item())


def sample_scaffolded(scorer: PriorScorer, *, voice_count: int = 3, bars: int = 4,
                      meter: tuple[int, int] = (4, 4), fifths: int = 0,
                      spq: int = 4, temperature: float = 1.0, top_p: float = 0.92,
                      seed: int = 0, prime_content: list[str] | None = None,
                      ) -> list[str]:
    """Emit the grid skeleton deterministically; sample only the content slots.

    ``prime_content`` forces the first N content slots to given tokens -- used to
    hand the model a real Bach opening and ask it to continue, which is what a
    generative component would actually be asked to do. Sampling cold from the
    header alone leaves the model with no musical context, and it then tends to
    settle into an all-rest or all-hold attractor.
    """
    num, den = meter
    gen = torch.Generator().manual_seed(seed)
    allowed = _content_ids(scorer)
    stoi = scorer.stoi

    beat_len_steps = int(Fraction(4, den) * spq)
    bar_len_steps = num * beat_len_steps
    n_steps = bars * bar_len_steps

    toks = [f"KEY_{fifths}", f"METER_{num}_{den}"]
    for s in range(n_steps):
        if s % bar_len_steps == 0:
            toks.append("BAR")
        if s % beat_len_steps == 0:
            toks.append(f"BEAT_{s % bar_len_steps // beat_len_steps + 1}")
        for v in range(1, voice_count + 1):
            toks.append(f"VOICE_{v}")
            k = len([x for x in toks if x in ("HOLD", "REST")
                     or x.startswith("NOTE_ON_")])
            if prime_content and k < len(prime_content):
                toks.append(prime_content[k])
            else:
                ids = [stoi[t] for t in toks]
                toks.append(scorer.tokens[
                    _next(scorer, ids, allowed, temperature, top_p, gen)])
    toks.append("<END_OF_SECTION>")
    return toks


def sample_free(scorer: PriorScorer, *, max_tokens: int = 900, fifths: int = 0,
                meter: tuple[int, int] = (4, 4), temperature: float = 1.0,
                top_p: float = 0.92, seed: int = 0) -> list[str]:
    """Sample every token. Diagnostic: did the prior learn the grid format?"""
    gen = torch.Generator().manual_seed(seed)
    toks = [f"KEY_{fifths}", f"METER_{meter[0]}_{meter[1]}"]
    for _ in range(max_tokens):
        ids = [scorer.stoi[t] for t in toks if t in scorer.stoi]
        nxt = scorer.tokens[_next(scorer, ids, None, temperature, top_p, gen)]
        toks.append(nxt)
        if nxt == "<END_OF_SECTION>":
            break
    return toks


def grid_wellformed(toks: list[str], voice_count: int) -> dict:
    """How closely a token stream follows the grid's own rules.

    The grid is rigid: after each VOICE_n comes exactly one content token, and
    VOICE_1..VOICE_n cycle in order. Departures are counted rather than raised,
    because for a free sample they are the measurement.
    """
    expect_v = 1
    slots = bad_order = bad_content = 0
    i = 0
    body = [t for t in toks if not t.startswith(("KEY_", "METER_"))]
    while i < len(body):
        t = body[i]
        if t.startswith("VOICE_"):
            slots += 1
            if int(t.split("_")[1]) != expect_v:
                bad_order += 1
            expect_v = expect_v % voice_count + 1
            nxt = body[i + 1] if i + 1 < len(body) else None
            if nxt is None or not (nxt.startswith("NOTE_ON_") or nxt in ("HOLD", "REST")):
                bad_content += 1
            i += 2
            continue
        i += 1
    return {"voice_slots": slots, "out_of_order": bad_order,
            "slot_without_content": bad_content,
            "wellformed": bad_order == 0 and bad_content == 0 and slots > 0}


def decode_to_score(toks: list[str], voice_count: int, spq: int = 4,
                    meter: tuple[int, int] = (4, 4), fifths: int = 0) -> dict:
    """Grid tokens -> a canonical score dict with real durations.

    A HOLD in a voice with nothing sounding is treated as a rest, not as a note
    of unknown pitch: the prior can emit one, and inventing a pitch to cover it
    would put notes in the output that the model never chose.
    """
    step = Fraction(1, spq)
    open_note: list[tuple[int, int] | None] = [None] * voice_count   # (midi, start)
    notes: list[list[dict]] = [[] for _ in range(voice_count)]
    s = -1
    slot = 0
    vi = 0

    def close(v: int, end: int) -> None:
        if open_note[v] is None:
            return
        midi, start = open_note[v]
        notes[v].append({"midi": midi, "start": start, "end": end})
        open_note[v] = None

    i = 0
    body = [t for t in toks if not t.startswith(("KEY_", "METER_"))]
    while i < len(body):
        t = body[i]
        if t.startswith("VOICE_"):
            vi = int(t.split("_")[1]) - 1
            if slot % voice_count == 0:
                s += 1
            slot += 1
            nxt = body[i + 1] if i + 1 < len(body) else None
            if vi < voice_count and nxt is not None:
                if nxt.startswith("NOTE_ON_"):
                    close(vi, s)
                    open_note[vi] = (int(nxt.split("_")[-1]), s)
                elif nxt == "REST":
                    close(vi, s)
                # HOLD: leave the open note running; if none, it is silence.
            i += 2
            continue
        i += 1
    for v in range(voice_count):
        close(v, s + 1)

    total = Fraction(max(1, s + 1)) * step
    voices = []
    for v in range(voice_count):
        out = []
        for j, n in enumerate(notes[v]):
            onset = Fraction(n["start"]) * step
            dur = Fraction(max(1, n["end"] - n["start"])) * step
            out.append({
                "note_id": f"p{v}_{j}", "voice_id": f"voice_{v+1}",
                "onset_q": str(onset), "duration_q": str(dur),
                "midi": n["midi"], "step": "C", "alter": "0",
                "octave": n["midi"] // 12 - 1, "pitch_class": n["midi"] % 12,
                "tie": None, "grace": False, "measure_number": "1",
                "measure_index_unfolded": 0, "offset_in_measure_q": "0",
                "diatonic_step": 0, "articulations": [], "ornaments": [],
                "chord_group": None, "source_refs": [],
            })
        voices.append({"voice_id": f"voice_{v+1}", "staff": None, "notes": out})
    return {
        "piece_id": "prior_sample", "composer_id": "synthetic",
        "voices": voices, "quarter_length_total": str(total),
        "meter_map": [{"onset_q": "0", "numerator": meter[0],
                       "denominator": meter[1]}],
        "key_signature_map": [{"onset_q": "0", "fifths": fifths,
                               "mode_if_known": None}],
        "annotations": [],
    }
