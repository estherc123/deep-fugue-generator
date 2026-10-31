"""Tests for the two local review applications (Spec §14.5, §18.5).

These exist because a bug slipped through without them: the pages referenced
hyphenated element ids (`id="c-rev"`) as bare JavaScript globals (`c_rev`), which
is a ReferenceError, so `load()` threw on its first line and both apps sat on
"loading…" forever. Nothing in the Python test suite could see that, because the
failure was entirely in the browser.

So the page markup and its script are now checked directly: every id the script
looks up must exist, no script identifier may be a hyphenated id read as an
underscore global, and the script must parse.
"""

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fugue_data.review import app as excerpt_app
from fugue_data.review import pairs_app

PAGES = {"excerpts": excerpt_app.PAGE, "pairs": pairs_app.PAGE}


def _script(page: str) -> str:
    m = re.search(r"<script>(.*?)</script>", page, re.S)
    assert m, "page has no inline script"
    return m.group(1)


def _element_ids(page: str) -> set[str]:
    return set(re.findall(r'id="([^"]+)"', page))


@pytest.mark.parametrize("name", sorted(PAGES))
def test_every_looked_up_element_id_exists(name):
    page = PAGES[name]
    ids = _element_ids(page)
    looked_up = set(re.findall(r"""\$\(\s*['"]([^'"]+)['"]\s*\)""", _script(page)))
    looked_up |= set(re.findall(
        r"""getElementById\(\s*['"]([^'"]+)['"]\s*\)""", _script(page)))
    missing = sorted(looked_up - ids)
    assert not missing, f"{name}: script looks up ids that do not exist: {missing}"


@pytest.mark.parametrize("name", sorted(PAGES))
def test_no_hyphenated_id_is_read_as_a_bare_global(name):
    """`id="c-rev"` cannot be reached as `c_rev`; that throws a ReferenceError."""
    page = PAGES[name]
    script = _script(page)
    declared = set(re.findall(r"\b(?:const|let|var|function)\s+([A-Za-z_$][\w$]*)",
                              script))
    declared |= {"window", "document", "fetch", "JSON", "String", "Math", "console",
                 "AudioContext", "webkitAudioContext", "Error"}
    offenders = []
    for ident in set(re.findall(r"\b([A-Za-z_$][\w$]*)\s*\.\s*textContent", script)):
        if ident in declared:
            continue
        candidate = ident.replace("_", "-")
        if candidate in _element_ids(page) or ident in _element_ids(page):
            offenders.append(ident)
    assert not offenders, (
        f"{name}: {offenders} read an element id as a bare global; use $('id')")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
@pytest.mark.parametrize("name", sorted(PAGES))
def test_page_script_parses(name):
    """A syntax error in the page is invisible to Python but fatal in the browser."""
    r = subprocess.run(["node", "--check", "-"], input=_script(PAGES[name]),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"{name} script does not parse:\n{r.stderr[:800]}"


@pytest.mark.parametrize("name", sorted(PAGES))
def test_load_failure_is_surfaced_not_left_spinning(name):
    script = _script(PAGES[name])
    assert "catch" in script and "Could not load" in script, (
        f"{name}: a failed fetch must show an error, not leave 'loading…' on screen")


# --------------------------------------------------------------------------- #
# Live server checks
# --------------------------------------------------------------------------- #
def _serve(handler_cls, store, port):
    handler_cls.store = store
    srv = HTTPServer(("127.0.0.1", port), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)
    return srv


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read()


needs_data = pytest.mark.skipif(
    not (ROOT / "data" / "proposed" / "excerpts.jsonl").exists(),
    reason="excerpt candidates not built")


@needs_data
def test_excerpt_app_serves_page_and_api():
    srv = _serve(excerpt_app.Handler, excerpt_app.Store(ROOT), 8811)
    try:
        status, page = _get("http://127.0.0.1:8811/")
        assert status == 200 and b"Excerpt review" in page
        status, body = _get("http://127.0.0.1:8811/api/excerpts")
        d = json.loads(body)
        assert status == 200 and d["records"]
        r = d["records"][0]
        for field in ("excerpt_id", "svg", "notes", "review_status",
                      "proposal_reasons"):
            assert field in r
    finally:
        srv.shutdown()


@pytest.mark.skipif(
    not (ROOT / "data" / "generated" / "preference_candidates" / "pairs.jsonl").exists(),
    reason="preference pairs not built")
def test_pairs_app_serves_page_and_api_with_blinded_sides():
    srv = _serve(pairs_app.Handler, pairs_app.PairStore(ROOT), 8812)
    try:
        status, page = _get("http://127.0.0.1:8812/")
        assert status == 200 and b"Preference review" in page
        status, body = _get("http://127.0.0.1:8812/api/pairs")
        d = json.loads(body)
        assert status == 200 and d["pairs"]
        p = d["pairs"][0]
        assert p["left"]["notes"] and p["right"]["notes"]
        # §18.5: the reviewer must not be told which side is which candidate.
        assert "sample_id" not in json.dumps(p["left"])
        assert "candidate_a_id" not in p and "candidate_b_id" not in p
    finally:
        srv.shutdown()


@needs_data
def test_unknown_path_returns_404_not_a_hang():
    srv = _serve(excerpt_app.Handler, excerpt_app.Store(ROOT), 8813)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            _get("http://127.0.0.1:8813/nope")
        assert e.value.code == 404
    finally:
        srv.shutdown()
