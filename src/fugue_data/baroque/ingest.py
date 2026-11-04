"""Import the expanded Baroque corpus into canonical scores.

    PYTHONPATH=src python -m fugue_data.baroque.ingest [--jobs 16]

For every source in `sources.SOURCES`:
1. Clone at the pinned commit into data/raw/baroque/<name> (skipped if present
   at that commit).
2. Parse each file into the canonical score shape the rest of the project uses
   (data/canonical_baroque/<piece_id>.json). Kern goes through the existing
   music21 normaliser (one spine = one voice). DCML note tables are read
   directly: one MuseScore (staff, voice layer) = one voice; a layer holding
   under 10% of its staff's notes is folded into layer 1 as divisi; a voice
   doubling another (a continuo part copying the cello) is dropped.
3. Classify: fugue (fugal title), chorale, or other.
4. Voice check: a voice is clean when under 5% of its notes start while another
   of its notes sounds (chords, divisi); 12% for fugues, whose final bars
   are often written as divisi chords. Only pieces whose voices are all clean,
   with 2-6 voices, are used to train the note model (`note_model: true`);
   keyboard music written as two staves of chords is kept for harmony and form
   statistics only.
5. Duplicates: a piece sharing most of its voice openings (intervals and
   rhythm, transposition-free) with an earlier piece, including the original
   191, is marked `dup_of` and left out.
6. Split by work: about 8% validation, 8% test, by hash of the work id; the
   organ fugue BWV 543 is always test (it is the target scale).

Writes manifests/baroque_manifest.jsonl (one row per file, with sha256,
licence and every decision above) and manifests/baroque_splits.jsonl (rows in
the shape of manifests/splits.jsonl, for pieces that pass).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

from ..normalize.rational import frac_str, parse_frac
from ..splitting.features import derive_mode
from .sources import FUGAL, SOURCES

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "baroque"
OUT = ROOT / "data" / "canonical_baroque"
OVERLAP_MAX = 0.05
OVERLAP_MAX_FUGUE = 0.12      # organ fugues close on divisi chords
TEST_ALWAYS = {"kernscores-organ_bwv543"}
TONICS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def checkout(src: dict) -> Path:
    d = RAW / src["name"]
    if d.exists():
        head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        if head == src["commit"]:
            return d
        subprocess.run(["git", "-C", str(d), "fetch", "-q", "--depth", "1", "origin",
                        src["commit"]], check=True)
    else:
        RAW.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "remote", "add", "origin", src["repo"]], check=True)
        subprocess.run(["git", "-C", str(d), "fetch", "-q", "--depth", "1", "origin",
                        src["commit"]], check=True)
    subprocess.run(["git", "-C", str(d), "checkout", "-q", src["commit"]], check=True)
    return d


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", s).strip("_")


def kind_of(title: str, source: str) -> str:
    if source == "chorales-370":
        return "chorale"
    t = title.lower()
    if "canon" in t and "contrapunctus" not in t:
        return "other"
    return "fugue" if any(w in t for w in FUGAL) else "other"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _kern_meta(path: Path) -> dict:
    meta = {}
    for line in path.read_text(errors="replace").splitlines():
        for tag in ("COM", "OTL", "SCT"):
            if line.startswith(f"!!!{tag}:") and tag not in meta:
                meta[tag] = line.split(":", 1)[1].strip()
        m = re.match(r"^\*([A-Ga-g][#-]?):", line)
        if m and "key" not in meta:
            meta["key"] = m.group(1)
    return meta


def parse_kern(src: dict, path: Path, piece_id: str) -> tuple[dict, dict]:
    from ..normalize.normalize import normalize_score
    meta = _kern_meta(path)
    title = meta.get("OTL", path.stem)
    kind = kind_of(title, src["name"])
    score = normalize_score(path, piece_id=piece_id, work_group_id=_work_of(src, path.stem),
                            movement_type=kind, source_file_ids=[f"{src['name']}:{path.name}"],
                            composer_id=_slug(meta.get("COM", "unknown")).lower())
    d = score.to_dict()
    mode = None
    if "key" in meta:
        mode = "minor" if meta["key"][0].islower() else "major"
    return d, {"title": title, "composer": meta.get("COM"), "mode": mode, "kind": kind}


def _frac(x: str) -> Fraction:
    return Fraction(x) if x not in ("", None) else Fraction(0)


def parse_dcml(src: dict, path: Path, piece_id: str) -> tuple[dict, dict]:
    stem = path.name.replace(".notes.tsv", "")
    md = {}
    with (path.parent.parent / "metadata.tsv").open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["piece"] == stem:
                md = r
                break
    title = " ".join(x for x in (md.get("workTitle"), md.get("movementTitle"), stem) if x)
    kind = kind_of(title, src["name"])
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["quarterbeats"] != ""]
    by_layer = defaultdict(list)
    for r in rows:
        by_layer[(int(r["staff"]), int(r["voice"]))].append(r)
    staff_total = Counter(s for (s, _), rs in by_layer.items() for _ in rs)
    layers = defaultdict(list)
    for (s, v), rs in by_layer.items():
        key = (s, 1) if v > 1 and len(rs) < 0.1 * staff_total[s] else (s, v)
        layers[key].extend(rs)
    tie = {"1": "start", "0": "continue", "-1": "stop"}
    voices = []
    for (s, v), rs in sorted(layers.items()):
        notes = []
        for r in sorted(rs, key=lambda r: (_frac(r["quarterbeats"]), int(r["midi"]))):
            o = _frac(r["quarterbeats"])
            d = _frac(r["duration_qb"]) if "/" in r["duration_qb"] else Fraction(r["duration_qb"]).limit_denominator(96)
            name = r["name"]
            step = int(r["octave"]) * 7 + "CDEFGAB".index(name[0])
            notes.append({"onset_q": frac_str(o), "duration_q": frac_str(d), "midi": int(r["midi"]),
                          "pitch_class": int(r["midi"]) % 12, "diatonic_step": step,
                          "tie": tie.get(r.get("tied") or "", None), "grace": d == 0,
                          "measure_number": r["mn"], "measure_index_unfolded": int(r["mc"]),
                          "offset_in_measure_q": frac_str(_frac(r["mc_onset"]) * 4)})
        if notes:
            midis = [n["midi"] for n in notes]
            voices.append({"voice_id": f"s{s}v{v}", "staff": str(s), "notes": notes,
                           "mean_midi": round(sum(midis) / len(midis), 3)})
    voices = _drop_doublings(voices)
    voices.sort(key=lambda v: -v["mean_midi"])
    for i, v in enumerate(voices, 1):
        v["voice_id"] = f"voice_{i}"
    ts = (md.get("TimeSig") or "1: 4/4").split(",")[0].split(":")[-1].strip()
    num, den = (int(x) for x in ts.split("/"))
    fifths = int((md.get("KeySig") or "1: 0").split(",")[0].split(":")[-1].strip() or 0)
    ak = md.get("annotated_key") or ""
    mode = ("minor" if ak[:1].islower() else "major") if ak else None
    total = max((parse_frac(n["onset_q"]) + parse_frac(n["duration_q"])
                 for v in voices for n in v["notes"]), default=Fraction(0))
    score = {"piece_id": piece_id, "work_group_id": _work_of(src, stem), "movement_type": kind,
             "composer_id": _slug(md.get("composer") or "unknown").lower(),
             "voices": voices, "voice_count": len(voices),
             "meter_map": [{"numerator": num, "denominator": den, "onset_q": "0"}],
             "key_signature_map": [{"fifths": fifths, "mode_if_known": mode, "onset_q": "0"}],
             "quarter_length_total": frac_str(total),
             "measure_count": int(md.get("last_mn") or 0), "source_file_ids": [f"{src['name']}:{path.name}"]}
    return score, {"title": title, "composer": md.get("composer"), "mode": mode, "kind": kind,
                   "harmonies": str((path.parent.parent / "harmonies" / f"{stem}.harmonies.tsv")
                                    .relative_to(ROOT))
                   if (path.parent.parent / "harmonies" / f"{stem}.harmonies.tsv").exists() else None}


def _drop_doublings(voices: list[dict]) -> list[dict]:
    """Drop a voice whose attacks mostly coincide with a kept voice's, at the same
    pitch class (a continuo bass doubling the cello). Fuller voices are kept first."""
    sig = [{(n["onset_q"], n["midi"] % 12) for n in v["notes"]} for v in voices]
    kept: list[int] = []
    for i in sorted(range(len(voices)), key=lambda i: -len(sig[i])):
        if not any(len(sig[i] & sig[j]) >= 0.85 * len(sig[i]) for j in kept):
            kept.append(i)
    return [voices[i] for i in sorted(kept)]


def _work_of(src: dict, stem: str) -> str:
    s = re.sub(r"[a-d]$", "", stem) if src["name"] == "art-of-fugue" else stem
    return f"{src['name']}:{s}"


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def overlap_share(notes: list[dict]) -> float:
    """Share of notes attacked while another note of the same voice sounds."""
    ev = sorted((parse_frac(n["onset_q"]), parse_frac(n["duration_q"])) for n in notes
                if not n.get("grace") and n.get("tie") not in ("continue", "stop"))
    if not ev:
        return 0.0
    bad, end = 0, None
    for o, d in ev:
        if end is not None and o < end:
            bad += 1
        end = o + d if end is None else max(end, o + d)
    return bad / len(ev)


def voice_signatures(score: dict, n: int = 24) -> set[str]:
    out = set()
    for v in score["voices"]:
        ev = []
        for x in sorted(v["notes"], key=lambda x: (parse_frac(x["onset_q"]), -x["midi"])):
            if x.get("grace") or x.get("tie") in ("continue", "stop"):
                continue
            o = parse_frac(x["onset_q"])
            if ev and ev[-1][0] == o:
                continue
            ev.append((o, x["midi"]))
        if len(ev) < 8:
            continue
        ev = ev[:n + 1]
        sig = tuple((b[1] - a[1], str(b[0] - a[0])) for a, b in zip(ev, ev[1:]))
        out.add(hashlib.sha1(repr(sig).encode()).hexdigest()[:16])
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _one(job):
    src, path = job
    path = Path(path)
    stem = path.name.replace(".notes.tsv", "").replace(".krn", "")
    piece_id = f"{src['name']}_{_slug(stem)}"
    row = {"piece_id": piece_id, "source_name": src["name"], "repository_url": src["repo"],
           "repository_commit": src["commit"], "relative_source_path": str(path.relative_to(RAW / src["name"])),
           "sha256": sha256(path), "license": src["license"]}
    try:
        score, info = (parse_kern if src["format"] == "kern" else parse_dcml)(src, path, piece_id)
    except Exception as e:  # noqa: BLE001 - record and move on
        row.update(status="parse_failed", error=f"{type(e).__name__}: {e}"[:300])
        return row, None
    voices = [v for v in score["voices"] if v["notes"]]
    score["voices"] = voices
    score["voice_count"] = len(voices)
    ov = [round(overlap_share(v["notes"]), 4) for v in voices]
    mode = info["mode"] or (derive_mode(score) if voices else "unknown")
    row.update({k: info[k] for k in ("title", "composer", "kind")}, mode=mode,
               nv=len(voices), voice_overlap=ov,
               quarters=float(parse_frac(score["quarter_length_total"])),
               harmonies=info.get("harmonies"),
               note_model=bool(2 <= len(voices) <= 6 and ov and max(ov) <= (
                   OVERLAP_MAX_FUGUE if info["kind"] == "fugue" else OVERLAP_MAX)),
               status="ok", signatures=sorted(voice_signatures(score)))
    if row["quarters"] < 8:
        row.update(status="too_short", note_model=False)
    return row, score


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args(argv)
    jobs = []
    for src in SOURCES:
        d = checkout(src)
        jobs += [(src, str(p)) for p in sorted(d.glob(src["glob"]))]
    print(f"{len(jobs)} files from {len(SOURCES)} sources")
    with Pool(args.jobs) as pool:
        results = pool.map(_one, jobs, chunksize=1)

    # Duplicates: against the original corpus first, then in source order.
    seen: list[tuple[str, set]] = []
    for f in sorted((ROOT / "data" / "canonical").glob("*.json")):
        if not f.name.endswith(".invalid.json"):
            s = json.loads(f.read_text())
            seen.append((s["piece_id"], voice_signatures(s)))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, splits = [], []
    for row, score in results:
        sig = set(row.pop("signatures", []))
        if row["status"] == "ok" and sig:
            for pid, other in seen:
                if len(sig & other) >= max(1, (len(sig) + 1) // 2):
                    row.update(status="duplicate", dup_of=pid, note_model=False)
                    break
            else:
                seen.append((row["piece_id"], sig))
        manifest.append(row)
        if row["status"] != "ok":
            continue
        (OUT / f"{row['piece_id']}.json").write_text(json.dumps(score, sort_keys=True) + "\n")
        work = score["work_group_id"]
        h = int(hashlib.sha256(work.encode()).hexdigest(), 16) % 100
        split = ("test" if row["piece_id"] in TEST_ALWAYS
                 else "validation" if h < 8 else "test" if h < 16 else "train")
        splits.append({"piece_id": row["piece_id"], "split": split, "corpus": row["source_name"],
                       "kind": row["kind"], "note_model": row["note_model"], "work_group_id": work,
                       "stratification_fields": {"mode": row["mode"], "voice_count": row["nv"]}})
    (ROOT / "manifests" / "baroque_manifest.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in manifest))
    (ROOT / "manifests" / "baroque_splits.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in splits))

    c = Counter((r["source_name"], r["status"]) for r in manifest)
    print("\nsource                          ok  dup  fail  note-model  fugues(nm)")
    for src in SOURCES:
        rs = [r for r in manifest if r["source_name"] == src["name"]]
        nm = [r for r in rs if r.get("note_model")]
        print(f"{src['name']:30s} {c[(src['name'], 'ok')]:4d} {c[(src['name'], 'duplicate')]:4d} "
              f"{c[(src['name'], 'parse_failed')]:5d} {len(nm):11d} "
              f"{sum(r['kind'] == 'fugue' for r in nm):10d}")
    nm = [s for s in splits if s["note_model"]]
    print(f"\nnote-model pieces: {len(nm)} ({sum(s['kind'] == 'fugue' for s in nm)} fugues); "
          f"splits {Counter(s['split'] for s in nm)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
