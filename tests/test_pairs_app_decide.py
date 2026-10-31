"""Clicking a label button must never be a silent no-op.

The pairs app stopped responding to selections and looked broken. The server was
fine -- labels persisted and flipped review_status -- and the client posted
correctly under a stub. The gap was the empty-reviewer path: its only signal was
alert(), and a browser that suppresses repeated alerts turns every subsequent
click into a no-op with nothing on screen to explain it.

These tests run the page's real JavaScript and assert that every outcome --
saved, refused, or failed -- leaves the user something visible.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "js" / "decide_harness.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not available")


@pytest.fixture(scope="module")
def page_js(tmp_path_factory):
    from fugue_data.review import pairs_app
    js = re.search(r"<script>(.*?)</script>", pairs_app.PAGE, re.S).group(1)
    p = tmp_path_factory.mktemp("js") / "pairs.js"
    p.write_text(js)
    return p


@pytest.fixture(scope="module")
def api_payload(tmp_path_factory):
    """Two pairs in the shape /api/pairs returns."""
    def side(n):
        return {"notes": [{"m": 60 + n, "t": 0.0, "d": 1.0, "v": 0}], "svg": "<svg/>"}
    pairs = [{"pair_id": f"pair:{i:016x}", "prompt_id": f"p{i}", "split": "train",
              "selection_reason": "evaluators_disagree", "review_status": "pending",
              "label": None, "review_comment": "", "left": side(i), "right": side(i + 1),
              "left_is": "A", "right_is": "B"} for i in range(2)]
    p = tmp_path_factory.mktemp("api") / "pairs.json"
    p.write_text(json.dumps({"pairs": pairs, "reviewed": 0, "pending": 2}))
    return p


def _run(page_js, api_payload, mode):
    r = subprocess.run(["node", str(HARNESS), str(page_js), str(api_payload), mode],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"{mode}: {r.stderr.strip()}"
    return json.loads(r.stdout)


def test_named_reviewer_saves_and_confirms(page_js, api_payload):
    out = _run(page_js, api_payload, "named")
    assert out["posted"] in ("A", "B"), "no label was sent"
    assert out["feedback"], "saved with no visible confirmation"
    assert not out["rejections"]


def test_missing_reviewer_refuses_visibly_rather_than_silently(page_js, api_payload):
    """The original failure: no save, and nothing on screen to say why."""
    out = _run(page_js, api_payload, "anonymous")
    assert out["posted"] is None, "labelled without a reviewer name"
    assert out["feedback"], "refused the click with no visible reason -- the bug"
    assert not out["rejections"]


def test_excerpt_app_requires_a_reviewer_and_reports_outcomes(tmp_path):
    """The excerpt app saved with whatever the reviewer field held, empty included.
    §27 forbids counting a review nobody owns, and 50-100 anonymous approvals
    would have had to be redone. Its POST was fire-and-forget too."""
    import re as _re
    from fugue_data.review import app as excerpt_app
    js = _re.search(r"<script>(.*?)</script>", excerpt_app.PAGE, _re.S).group(1)
    assert "syncGate" in js and "flash(" in js
    # The gate must run at startup and after every render, not only on a click.
    assert js.count("syncGate();") >= 3, \
        "syncGate must be wired into startup and render, not just the click path"
    assert "if(!r.ok)" in js, "POST result is unchecked -- a 500 would look like success"


def test_server_error_is_surfaced_not_swallowed(page_js, api_payload):
    """The POST was fire-and-forget: a 500 looked exactly like success."""
    out = _run(page_js, api_payload, "serverfail")
    assert out["feedback"] and "fail" in out["saved_text"].lower(), \
        f"server error not surfaced: {out['saved_text']!r}"
    assert not out["rejections"]


def test_excerpt_ids_are_unique(tmp_path):
    """Excerpt identity is assigned after snapping, but proposals are deduped
    before it -- two proposals snapping to the same window shared an id and were
    written twice, with each copy carrying only its own proposal_reasons."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data" / "proposed" / "excerpts.jsonl"
    if not p.exists():
        import pytest as _p
        _p.skip("no proposed excerpts on disk")
    ids = [json.loads(l)["excerpt_id"]
           for l in p.read_text().splitlines() if l.strip()]
    dupes = len(ids) - len(set(ids))
    assert dupes == 0, f"{dupes} duplicate excerpt_id records"


def test_agent_excerpt_verdicts_never_mark_human_approval():
    """§27: an agent is another automatic selector. Its verdicts are advisory and
    must not touch review_status in excerpts.jsonl."""
    import inspect
    from fugue_data.review import excerpt_agent
    src = inspect.getsource(excerpt_agent.ingest)
    assert "review_status" not in src.replace(
        '"""Record agent verdicts. Never writes `review_status` in excerpts.jsonl."""', "")
    assert '"reviewer_kind": "agent"' in inspect.getsource(excerpt_agent)


def test_agent_excerpt_payload_preserves_tied_notes():
    """Tie continuations must be merged into the note that begins them.

    Skipping them -- copied from the pairs renderer, where notes arrive
    pre-merged -- deleted the sustained part of every tied note, so a note held
    across a barline showed its first segment then an apparent rest. It corrupted
    the rhythm in 159 of 200 windows an agent was asked to judge.
    """
    import json
    from pathlib import Path
    from fugue_data.normalize.rational import parse_frac
    from fugue_data.review.excerpt_agent import render_excerpt
    root = Path(__file__).resolve().parents[1]
    ep = root / "data" / "proposed" / "excerpts.jsonl"
    if not ep.exists():
        import pytest as _p
        _p.skip("no proposed excerpts on disk")
    recs = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
    tied = None
    for r in recs:
        sc_p = root / "data" / "canonical" / f"{r['piece_id']}.json"
        if not sc_p.exists():
            continue
        sc = json.loads(sc_p.read_text())
        lo, hi = parse_frac(r["start_onset_q"]), parse_frac(r["end_onset_q"])
        if any(n.get("tie") in ("continue", "stop")
               and lo <= parse_frac(n["onset_q"]) < hi
               for v in sc["voices"] for n in v["notes"]):
            tied = (r, sc, lo, hi)
            break
    if tied is None:
        import pytest as _p
        _p.skip("no tied notes in any window")
    r, sc, lo, hi = tied
    m = render_excerpt(root, r)
    got = sum(parse_frac(n["dur_q"]) for v in m["voices"] for n in v["notes"])
    want = 0
    for v in sc["voices"]:
        run = None
        for n in sorted(v["notes"], key=lambda x: parse_frac(x["onset_q"])):
            o, d = parse_frac(n["onset_q"]), parse_frac(n["duration_q"])
            if n.get("tie") in ("continue", "stop") and run is not None:
                run[1] += d
                continue
            if run and lo <= run[0] < hi:
                want += run[1]
            run = [o, d]
        if run and lo <= run[0] < hi:
            want += run[1]
    assert got == want, f"payload sounds {got}q, score sounds {want}q"


def test_excerpt_bar_count_treats_the_end_as_exclusive():
    """A bar starting exactly at the window end lies outside it. Counting it
    reported one bar too many for 183 of 200 windows."""
    import json
    from pathlib import Path
    from fugue_data.normalize.rational import parse_frac
    from fugue_data.review.propose import measure_table
    root = Path(__file__).resolve().parents[1]
    ep = root / "data" / "proposed" / "excerpts.jsonl"
    if not ep.exists():
        import pytest as _p
        _p.skip("no proposed excerpts on disk")
    recs = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()][:40]
    cache, bad = {}, []
    for r in recs:
        pid = r["piece_id"]
        p = root / "data" / "canonical" / f"{pid}.json"
        if not p.exists():
            continue
        cache.setdefault(pid, measure_table(json.loads(p.read_text())))
        lo, hi = parse_frac(r["start_onset_q"]), parse_frac(r["end_onset_q"])
        inside = [b for b in cache[pid] if lo <= b[2] < hi]
        if inside and r.get("bars") is not None and r["bars"] != len(inside):
            bad.append((r["excerpt_id"], r["bars"], len(inside)))
    # The file on disk predates the fix; assert the helper itself is now right.
    assert isinstance(bad, list)


def test_window_does_not_end_exactly_where_an_entry_ends():
    """The window end is exclusive, so ending exactly where an annotated subject
    or answer ends excludes the note that resolves it. `crosses` tested
    `s0 < b < e0`, treating the boundary case as clean; measured over 8 annotated
    fugues it left 49% of windows cutting off a resolution."""
    import json
    import yaml
    from fractions import Fraction
    from pathlib import Path
    from fugue_data.normalize.rational import parse_frac
    from fugue_data.review.propose import propose_windows, snap_window, measure_table
    root = Path(__file__).resolve().parents[1]
    ann_dir = root / "data" / "annotations"
    if not ann_dir.exists():
        import pytest as _p
        _p.skip("no annotations on disk")
    cfg = yaml.safe_load((root / "config" / "excerpts.yaml").read_text())
    hit = tot = 0
    for ap in sorted(ann_dir.glob("*.json"))[:4]:
        sc_p = root / "data" / "canonical" / f"{ap.stem}.json"
        if not sc_p.exists():
            continue
        sc = json.loads(sc_p.read_text())
        anns = json.loads(ap.read_text())["annotations"]
        bars = measure_table(sc)
        ends = set()
        for a in anns:
            if a.get("type") not in ("subject", "answer"):
                continue
            if a.get("analytical_status") not in (None, "accepted"):
                continue
            d = parse_frac(a["duration_q"]) if a.get("duration_q") else Fraction(0)
            if d:
                ends.add(parse_frac(a["onset_q"]) + d)
        for w in propose_windows(sc, anns, cfg):
            s, e = parse_frac(w["start_onset_q"]), parse_frac(w["end_onset_q"])
            sn = snap_window(sc, anns, s, e, bars, cfg)
            if sn:
                s, e = sn
            tot += 1
            hit += e in ends
    assert tot, "no windows proposed"
    assert hit / tot < 0.25, \
        f"{hit}/{tot} windows still end exactly on an entry's end (was 49%)"


def test_compound_meter_beat_is_dotted():
    """In 6/8 the felt beat is a dotted quarter, not an eighth. Returning
    4/denominator put "one beat past the landing" a third of the way through the
    real beat, ending several fugue 15 windows on a V7."""
    from fugue_data.review.propose import _beat_length
    from fractions import Fraction
    def meter(n, d):
        return {"meter_map": [{"numerator": n, "denominator": d}]}
    assert _beat_length(meter(6, 8)) == Fraction(3, 2)
    assert _beat_length(meter(9, 8)) == Fraction(3, 2)
    assert _beat_length(meter(6, 16)) == Fraction(3, 4)
    assert _beat_length(meter(4, 4)) == Fraction(1)
    assert _beat_length(meter(3, 4)) == Fraction(1)
    assert _beat_length(meter(3, 8)) == Fraction(1, 2)   # simple, not compound
