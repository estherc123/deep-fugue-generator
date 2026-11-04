"""Bach fugue rules on the note grid, from research/bach_fugue_rules.md.

    PYTHONPATH=src python -m fugue_data.generator.bach_rules calibrate
    PYTHONPATH=src python -m fugue_data.generator.bach_rules audit DIR [DIR ...]

Each rule returns (rule_id, step, voice) hits. Rule ids carry the report's
section numbers. Following the report's §7.4, every rule is measured on Bach's
own fugues with this same code (`calibrate`), and its role is set by Bach's
rate: rules Bach (almost) never breaks are forbidden for the generator; the
others must not exceed Bach's rate.

Rules implemented (report section in brackets):
- par8, par5 [VL-1, VL-2]: consecutive octaves / fifths between any two
  voices, similar or contrary motion, both voices moving.
- hidden_outer [VL-4]: outer voices move in the same direction into a fifth
  or octave from another interval, with a leap in the top voice.
- strong_diss [§3.0 D-0, §3.3, §3.12]: on a beat onset, a note dissonant
  against the lowest voice (2nd, 4th, tritone, 7th) that is not a member of a
  seventh-chord sonority, not over a pedal, and does not leave by step (down
  by step if it was held into the beat: a suspension).
- chord7_res [§3.9]: the 7th of a seventh-chord sonority on a beat does not
  resolve down by step (or stay as a common tone) at its next attack.
- lt_outer [§3.10]: in the top or bottom voice, the leading tone sounding at
  the end of a V(7) unit followed by I does not rise a semitone to the tonic.
- leap_big [ML-1]: a leap larger than an octave; leap_maj7: a major 7th;
  leap_min7 [ML-1, A]: a minor 7th; aug2 [ML-1]: b6 <-> #7 in a minor key.
- leap_same_dir [ML-3]: two leaps in the same direction spanning a
  dissonant interval or more than an octave.
- crossing_bass [VL-6]: the lowest voice above the voice over it, or any
  crossing lasting more than a beat.
- spacing [VL-8]: adjacent upper voices more than an octave apart.
- cross_rel [VL-9]: within a beat, one voice has a pitch class and another
  voice its chromatic neighbour, one of them outside the local key.
- lt_double [HC-1]: on a beat, the leading tone doubled under a dominant
  sonority.
- range [ML-5]: outside MIDI 36-86 (keyboard compass).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .encode import MIDI_LO, NOTE0, STOI, STEPS_PER_Q, _beat_q

ROOT = Path(__file__).resolve().parents[3]
DISS = {1, 2, 5, 6, 10, 11}                 # against the lowest voice
SEVENTHS = {"dom7": (0, 4, 7, 10), "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10),
            "min7": (0, 3, 7, 10), "maj7": (0, 4, 7, 11)}
MAJOR = (0, 2, 4, 5, 7, 9, 11)
MINOR = (0, 2, 3, 5, 7, 8, 10, 11)          # natural + raised 7th (and 6th below)
MINOR_X = MINOR + (9,)                       # melodic minor's raised 6th


def sounding(grid):
    n, nv = grid.shape
    snd = np.full((n, nv), -1, dtype=np.int16)
    att = np.zeros((n, nv), dtype=bool)
    for v in range(nv):
        cur = -1
        for s in range(n):
            t = int(grid[s, v])
            if NOTE0 <= t < NOTE0 + 88:
                cur = MIDI_LO + t - NOTE0
                att[s, v] = True
            elif t == STOI["REST"]:
                cur = -1
            snd[s, v] = cur
    return snd, att


def notes(grid):
    """Per voice: [(start, end, midi)]."""
    snd, att = sounding(grid)
    out = []
    for v in range(grid.shape[1]):
        ev = []
        for s in np.flatnonzero(att[:, v]):
            e = s + 1
            while e < len(grid) and grid[e, v] == STOI["HOLD"]:
                e += 1
            ev.append((int(s), int(e), int(snd[s, v])))
        out.append(ev)
    return out


def _seventh_sonority(pcs):
    """(root, quality) if the pitch-class set is a seventh chord (all 4 tones)."""
    for r in pcs:
        for q, tones in SEVENTHS.items():
            if {(r + t) % 12 for t in tones} == set(pcs):
                return r, q
    return None


def check(grid: np.ndarray, meter=(4, 4), tonic: int = 0, mode: str = "major",
          keys: list | None = None, unit: int | None = None, start: int = 0,
          final: bool = False, fixed: np.ndarray | None = None) -> list[tuple[str, int, int]]:
    """All rule hits at steps >= start. keys: per-step (tonic_pc, minor) local
    key, or None for the home key throughout. fixed: (n, nv) bool, notes the
    plan wrote in (subject entries, pedals): melodic rules skip a leap between
    two fixed notes, and a dissonance between two fixed notes is not counted,
    since neither was the generator's choice."""
    n, nv = grid.shape
    snd, att = sounding(grid)
    ev = notes(grid)
    beat = max(1, int(_beat_q(meter) * STEPS_PER_Q))
    bar = int(4 * meter[0] / meter[1] * STEPS_PER_Q)
    hits = []
    key_at = (lambda s: keys[min(s, len(keys) - 1)]) if keys is not None else \
        (lambda s: (tonic, mode == "minor"))

    def add(rule, s, v, prev_s=None):
        if s < start:
            return
        if fixed is not None and prev_s is not None and fixed[s, v] and fixed[prev_s, v]:
            return
        hits.append((rule, int(s), int(v)))

    lowest = np.full(n, -1, dtype=np.int16)
    low_v = np.full(n, -1, dtype=np.int16)
    for s in range(n):
        live = np.flatnonzero(snd[s] >= 0)
        if len(live):
            low_v[s] = live[-1]
            lowest[s] = snd[s, live[-1]]

    # ---- parallels and hidden fifths/octaves ------------------------------ #
    for a in range(nv):
        for b in range(a + 1, nv):
            prev = None
            for s in range(n):
                if not (att[s, a] or att[s, b]):
                    continue
                if snd[s, a] < 0 or snd[s, b] < 0:
                    prev = None
                    continue
                cur = (int(snd[s, a]), int(snd[s, b]))
                if prev is not None and cur[0] != prev[0] and cur[1] != prev[1]:
                    i0, i1 = abs(prev[0] - prev[1]) % 12, abs(cur[0] - cur[1]) % 12
                    if i0 == i1 == 0:
                        add("par8", s, b)
                    elif i0 == i1 == 7:
                        add("par5", s, b)
                prev = cur
    for s in range(1, n):
        live = np.flatnonzero(snd[s] >= 0)
        if len(live) < 2:
            continue
        t, bvo = live[0], live[-1]
        if not (att[s, t] and att[s, bvo]) or snd[s - 1, t] < 0 or snd[s - 1, bvo] < 0:
            continue
        dt, db = int(snd[s, t] - snd[s - 1, t]), int(snd[s, bvo] - snd[s - 1, bvo])
        i0, i1 = abs(int(snd[s - 1, t] - snd[s - 1, bvo])) % 12, abs(int(snd[s, t] - snd[s, bvo])) % 12
        if dt * db > 0 and i1 in (0, 7) and i0 not in (0, 7) and abs(dt) > 2:
            add("hidden_outer", s, t)

    # ---- dissonance treatment on the beat ---------------------------------- #
    pedal = np.zeros(n, dtype=bool)
    for v in range(nv):
        for s0, e0, m in ev[v]:
            if e0 - s0 >= bar and np.all(low_v[s0:e0] == v):
                pedal[s0:e0] = True
    for s in range(0, n, beat):
        live = np.flatnonzero(snd[s] >= 0)
        if len(live) < 2:
            continue
        B = int(lowest[s])
        pcs = sorted({int(x) % 12 for x in snd[s, live]})
        sev = _seventh_sonority(pcs) if len(pcs) == 4 else None
        for v in live[:-1]:
            x = int(snd[s, v])
            if (x - B) % 12 not in DISS:
                continue
            # the note this voice is sounding now, and its next attack
            cur = next((e for e in ev[v] if e[0] <= s < e[1]), None)
            if cur is None:
                continue
            if fixed is not None and fixed[cur[0], v]:
                bv = int(low_v[s])
                bnote = next((e for e in ev[bv] if e[0] <= s < e[1]), None)
                if bnote is not None and fixed[bnote[0], bv]:
                    continue
            nxt = next((e for e in ev[v] if e[0] >= cur[1]), None)
            held = cur[0] < s
            if sev is not None:
                r, q = sev
                seventh = (r + SEVENTHS[q][3]) % 12
                if x % 12 == seventh and nxt is not None and nxt[0] - cur[1] <= beat:
                    d = nxt[2] - x
                    if not (d in (-1, -2) or d == 0):
                        add("chord7_res", s, v)
                continue
            if pedal[s]:
                continue
            if nxt is None or nxt[0] - cur[1] > beat:
                add("strong_diss", s, v)             # left by a rest: not resolved
                continue
            d = nxt[2] - x
            ok = d in (-1, -2) if held else 1 <= abs(d) <= 2
            if not ok:
                add("strong_diss", s, v)

    # ---- leading tone in the outer voices at V -> I ------------------------ #
    if unit and keys is not None:
        for s in range(unit, n, unit):
            kt, _ = key_at(s)
            lt = (kt + 11) % 12
            prev_pcs = {int(x) % 12 for x in snd[s - 1] if x >= 0}
            cur_pcs = {int(x) % 12 for x in snd[s] if x >= 0}
            if not ({(kt + 7) % 12, lt} <= prev_pcs and kt in cur_pcs and lowest[s] >= 0
                    and int(lowest[s]) % 12 == kt and (kt + 6) % 12 not in cur_pcs):
                continue
            live = np.flatnonzero(snd[s - 1] >= 0)
            for v in (live[0], live[-1]):
                if snd[s - 1, v] % 12 == lt and att[s, v] and snd[s, v] - snd[s - 1, v] != 1 \
                        and snd[s, v] >= 0:
                    add("lt_outer", s, v)

    # ---- melodic ----------------------------------------------------------- #
    for v in range(nv):
        e = ev[v]
        for k in range(1, len(e)):
            if e[k][0] - e[k - 1][1] > beat:          # a rest resets the line
                continue
            d = e[k][2] - e[k - 1][2]
            s = e[k][0]
            ps = e[k - 1][0]
            if abs(d) > 12:
                add("leap_big", s, v, ps)
            elif abs(d) == 11:
                add("leap_maj7", s, v, ps)
            elif abs(d) == 10:
                add("leap_min7", s, v, ps)
            kt, minor = key_at(s)
            if minor and abs(d) == 3:
                pa, pb = (e[k - 1][2] - kt) % 12, (e[k][2] - kt) % 12
                if {pa, pb} == {8, 11}:
                    add("aug2", s, v, ps)
            if k >= 2 and e[k - 1][0] - e[k - 2][1] <= beat:
                d0 = e[k - 1][2] - e[k - 2][2]
                if abs(d0) > 2 and abs(d) > 2 and d0 * d > 0:
                    span = abs(d0 + d)
                    if span > 12 or span % 12 in (1, 2, 6, 10, 11):
                        add("leap_same_dir", s, v, ps)
            if not (36 <= e[k][2] <= 86):
                add("range", s, v)

    # ---- crossing, spacing ------------------------------------------------- #
    for a in range(nv - 1):
        b = a + 1
        run = 0
        for s in range(n):
            if snd[s, a] >= 0 and snd[s, b] >= 0 and snd[s, b] > snd[s, a]:
                run += 1
                if run == beat + 1 or (b == nv - 1 and run == 1 and att[s, a] | att[s, b]):
                    add("crossing_bass" if b == nv - 1 else "crossing_long", s, b)
            else:
                run = 0
    for s in range(0, n, beat):
        live = np.flatnonzero(snd[s] >= 0)
        for a, b in zip(live[:-2], live[1:-1]):                 # upper voices only
            if snd[s, a] - snd[s, b] > 12:
                add("spacing", s, b)

    # ---- cross relations, doubled leading tone ----------------------------- #
    for s0 in range(0, n, beat):
        kt, minor = key_at(s0)
        scale = {(kt + x) % 12 for x in (MINOR_X if minor else MAJOR)}
        seen = {}
        for s in range(s0, min(n, s0 + beat)):
            for v in np.flatnonzero(att[s]):
                seen.setdefault(int(snd[s, v]) % 12, set()).add(int(v))
        for pc, vs in seen.items():
            q = (pc + 1) % 12
            if q in seen and (pc not in scale or q not in scale) and not (vs & seen[q]) \
                    and (pc not in scale) != (q not in scale):
                add("cross_rel", s0, min(vs | seen[q]))
        live = np.flatnonzero(snd[s0] >= 0)
        pcs = [int(snd[s0, v]) % 12 for v in live]
        lt = (kt + 11) % 12
        if pcs.count(lt) >= 2 and (kt + 7) % 12 in pcs:
            add("lt_double", s0, int(live[0]))

    # ---- final cadence ----------------------------------------------------- #
    if final:
        live = np.flatnonzero(snd[n - 1] >= 0)
        pcs = {int(x) % 12 for x in snd[n - 1, live]}
        if not len(live) or int(lowest[n - 1]) % 12 != tonic or not pcs <= {tonic, (tonic + 3) % 12,
                                                                                (tonic + 4) % 12, (tonic + 7) % 12}:
            add("final_cadence", n - 1, 0)
    return hits


# Forbidden for the generator: the owner's target is about one break per five
# fugues. Everything else in RULES must stay at or under Bach's own rate.
FORBIDDEN = ["par8", "par5", "hidden_outer", "lt_outer", "aug2", "lt_double", "range",
             "final_cadence", "leap_big", "leap_maj7", "crossing_bass"]

RULES = ["par8", "par5", "hidden_outer", "strong_diss", "chord7_res", "lt_outer", "leap_big",
         "leap_maj7", "leap_min7", "aug2", "leap_same_dir", "crossing_bass", "crossing_long",
         "spacing", "cross_rel", "lt_double", "range", "final_cadence"]


def piece_args(p, lab=None):
    """check() keyword arguments for an encoded piece (labels give local keys)."""
    if p["meter_tok"] == "METER_other":
        meter = (4, 4)
    else:
        meter = tuple(int(x) for x in p["meter_tok"].split("_")[1:3])
    kw = {"meter": meter, "tonic": p["tonic"], "mode": p["mode"]}
    if lab is not None:
        u = lab["unit_steps"]
        keys = [tuple(lab["chord_keys"][min(s // u, len(lab["chord_keys"]) - 1)]) for s in range(p["n_steps"])]
        kw.update(keys=keys, unit=u)
    return kw


def calibrate(out: Path | None = None) -> dict:
    """Per-100-bar rate of every rule on Bach's fugues (all splits), by corpus."""
    from .encode import load_corpus
    from .report import bar_steps
    pieces = [p for p in load_corpus(ROOT, baroque=True) if p["corpus"] == "fugue"]
    tot, bars = Counter(), 0.0
    per_piece = []
    for p in pieces:
        lp = ROOT / "data" / "labels" / f"{p['piece_id']}.json"
        lab = json.loads(lp.read_text()) if lp.exists() else None
        h = Counter(r for r, _, _ in check(p["grid"], final=True, **piece_args(p, lab)))
        b = p["n_steps"] / bar_steps(p)
        tot.update(h)
        bars += b
        per_piece.append({"piece": p["piece_id"], "bars": round(b, 1), **h})
    rates = {r: round(100 * tot[r] / bars, 2) for r in RULES}
    # share of fugues with at least one hit (for "once per five fugues")
    share = {r: round(sum(1 for x in per_piece if x.get(r)) / len(per_piece), 3) for r in RULES}
    res = {"fugues": len(pieces), "bars": round(bars), "per_100_bars": rates,
           "share_of_fugues_with_hit": share}
    (out or ROOT / "research" / "bach_rule_rates.json").write_text(json.dumps(res, indent=1))
    return res


def audit(dirs: list[str]) -> dict:
    """Rates in generated output (generated region only) vs Bach's rate table."""
    from .audit import collect
    from .encode import load_corpus
    from .report import bar_steps
    pieces = {p["piece_id"]: p for p in load_corpus(ROOT, baroque=True)}
    bach = json.loads((ROOT / "research" / "bach_rule_rates.json").read_text())["per_100_bars"]
    res = {}
    for d in dirs:
        tags = {}
        for pid, g, start, tag in collect(Path(d)):
            p = pieces[pid]
            lp = ROOT / "data" / "labels" / f"{pid}.json"
            lab = json.loads(lp.read_text()) if lp.exists() else None
            q = {**p, "n_steps": len(g)}
            c, b = tags.setdefault(tag, [Counter(), 0.0])
            c.update(r for r, _, _ in check(g, start=start, **piece_args(q, lab)))
            tags[tag][1] = b + (len(g) - start) / bar_steps(p)
        for tag, (c, b) in tags.items():
            res[f"{Path(d).name}:{tag}"] = ({r: round(100 * c[r] / b, 2) for r in RULES}, round(b))
    print(f"{'rule (per 100 bars)':18s}{'bach':>8s}" + "".join(f"{k[:20]:>22s}" for k in res))
    for r in RULES:
        mark = "F" if r in FORBIDDEN else " "
        print(f"{mark} {r:16s}{bach[r]:8.2f}" + "".join(f"{v[0][r]:22.2f}" for v in res.values()))
    print(f"{'  bars':18s}{'':8s}" + "".join(f"{v[1]:22d}" for v in res.values()))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "audit"])
    ap.add_argument("dirs", nargs="*")
    args = ap.parse_args(argv)
    if args.cmd == "audit":
        audit(args.dirs)
        return 0
    res = calibrate()
    print(f"{res['fugues']} fugues, {res['bars']} bars")
    for r in RULES:
        print(f"{r:16s} {res['per_100_bars'][r]:8.2f} / 100 bars   in {100 * res['share_of_fugues_with_hit'][r]:5.1f}% of fugues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
