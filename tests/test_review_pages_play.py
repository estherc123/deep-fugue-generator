"""Every review page must actually produce sound, not merely parse.

The gallery page shipped unplayable twice over: play() referenced an AC that only
the *other* pages declared, and the caller passed BPM into a parameter measured in
seconds per quarter. Both pages parsed cleanly under `node --check` and rendered
fine; only clicking play revealed anything. These tests run the page's JavaScript
under stubbed DOM / fetch / Web Audio and call play() for real.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "js" / "harness.js"
PAGES = ["app", "arbitrate_app", "pairs_app", "gallery"]

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not available")


def _page_js(name: str, tmp: Path) -> Path:
    import importlib
    mod = importlib.import_module(f"fugue_data.review.{name}")
    js = re.search(r"<script>(.*?)</script>", mod.PAGE, re.S).group(1)
    p = tmp / f"{name}.js"
    p.write_text(js)
    return p


@pytest.fixture(scope="module")
def notes(tmp_path_factory):
    """Four voices over four bars -- enough to exercise every timbre branch."""
    d = tmp_path_factory.mktemp("notes") / "n.json"
    rows = [{"m": 60 - 12 * v + (i % 5), "t": i * 0.5, "d": 0.5, "v": v}
            for v in range(4) for i in range(8)]
    d.write_text(json.dumps(rows))
    return d


@pytest.mark.parametrize("page", PAGES)
def test_page_play_schedules_audio(page, notes, tmp_path):
    js = _page_js(page, tmp_path)
    r = subprocess.run(["node", str(HARNESS), str(js), str(notes)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"{page}: {r.stderr.strip()}"
    out = json.loads(r.stdout)
    # Four voices x 8 notes, with an octave partial on the lower two and a second
    # partial on the bass, so comfortably more oscillators than notes.
    assert out["oscillators"] > 32, f"{page} scheduled only {out['oscillators']}"
    # 8 notes x 0.5 quarters x 0.55 s = 2.2s of music; the release adds a little.
    # A page passing BPM here would report ~380s.
    assert 2.0 < out["span_seconds"] < 6.0, \
        f"{page} span {out['span_seconds']}s -- wrong time unit?"


@pytest.mark.parametrize("page", PAGES)
def test_page_declares_synth_state_exactly_once(page, tmp_path):
    """Two `let AC` in one scope is a SyntaxError; zero is the shipped bug."""
    js = _page_js(page, tmp_path).read_text()
    assert js.count("let AC") == 1, f"{page} declares AC {js.count('let AC')} times"


def test_harness_detects_an_undeclared_audio_context(notes, tmp_path):
    """The guard above is only worth having if it can fail -- prove it does."""
    js = _page_js("gallery", tmp_path)
    broken = tmp_path / "broken.js"
    broken.write_text(js.read_text().replace("let AC=null, CUR=null;", ""))
    r = subprocess.run(["node", str(HARNESS), str(broken), str(notes)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 4 and "AC is not defined" in r.stderr
