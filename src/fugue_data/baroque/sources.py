"""Pinned sources for the expanded Baroque corpus.

Each source is a public Git repository pinned to a commit. Raw files are not
redistributed (data/ is git-ignored); the importer re-clones at the pinned
commit and records a sha256 per file in manifests/baroque_manifest.jsonl.

Left out on purpose:
- musedata/humdrum-corelli: the CCARH MuseData licence restricts use to its
  licence agreement. Corelli comes from DCML (CC BY-NC-SA 4.0) instead.
- The KernScores website (kern.humdrum.org) was down (HTTP 503) on 2026-11-03;
  its organ fugues are reached through the pvankranenburg mirror.
"""

from __future__ import annotations

SOURCES = [
    # --- Humdrum **kern ----------------------------------------------------- #
    {"name": "kernscores-organ", "repo": "https://github.com/pvankranenburg/kernscores.git",
     "commit": "6a2e8409adc6629a79be670213b638a816537e1c", "format": "kern",
     "glob": "kernfiles/*.krn", "license": None,
     "note": "Organ fugues: J.S. Bach BWV 533-567, W.F. Bach, Krebs, Kellner, "
             "Schneider, van Noordt. No licence stated (as humdrum-wtc)."},
    {"name": "art-of-fugue", "repo": "https://github.com/craigsapp/art-of-the-fugue.git",
     "commit": "c970eb9e723a81832dc049986a05a1c7421dba1e", "format": "kern",
     "glob": "kern/*.krn", "license": None},
    {"name": "musical-offering", "repo": "https://github.com/craigsapp/bach-musical-offering.git",
     "commit": "2487a50d62be23d0bfc2ce48406f28c0550622bd", "format": "kern",
     "glob": "kern/*.krn", "license": None},
    {"name": "inventions", "repo": "https://github.com/humdrum-tools/inventions.git",
     "commit": "93183f6fdb9ab7aa1fa601e3aa200b40763469cf", "format": "kern",
     "glob": "kern/*.krn", "license": None},
    {"name": "chorales-370", "repo": "https://github.com/craigsapp/bach-370-chorales.git",
     "commit": "a95574f9e93b4d72a96cd390a059df4f8471ddb6", "format": "kern",
     "glob": "kern/*.krn", "license": "see LICENSE.txt (C. S. Sapp, 2009)"},
    {"name": "vivaldi-op6", "repo": "https://github.com/craigsapp/vivaldi-op6.git",
     "commit": "084329bf3cada3f6e2d042ff44962b34d3b55be4", "format": "kern",
     "glob": "kern/*.krn", "license": None},
    {"name": "scarlatti", "repo": "https://github.com/craigsapp/scarlatti-keyboard-sonatas.git",
     "commit": "567731b13020802a422bd5aa49347ee3dac3ea25", "format": "kern",
     "glob": "kern/*.krn", "license": "see LICENSE.txt"},
    # --- DCML (MuseScore + note/harmony TSVs, expert chord labels) ---------- #
    *[{"name": f"dcml-{c}", "repo": f"https://github.com/DCMLab/{c}.git", "commit": sha,
       "format": "dcml", "glob": "notes/*.notes.tsv", "license": "CC BY-NC-SA 4.0"}
      for c, sha in [
          ("corelli", "5d58a0f3982da87de3d1e74cf1808c807d72fafb"),
          ("bach_en_fr_suites", "9cd6d362ed8246ed8edfc425944862ed88ddf1a5"),
          ("bach_solo", "dce67f753ced46b43dbfb16779fe3a19233893da"),
          ("handel_keyboard", "d3b42765e0d2457b6abb1bdb3e8dd622db61b084"),
          ("couperin_clavecin", "3bd00fc56d3473b26f1bea5fe50d1a1f2458c462"),
          ("couperin_concerts", "49efcdd24c39d48bf009d60f8aa3bbc1a1d9713a"),
          ("frescobaldi_fiori_musicali", "e17de917111e900929db785bfa2d0313968ca5d4"),
          ("pergolesi_stabat_mater", "b24d5432884d641ba98cf49b107c47d199d38100"),
      ]],
]

# Titles that make a movement a fugue (or fugal: ricercar, canzona, fantasia
# in the Sweelinck/van Noordt sense). Canons stay "other".
FUGAL = ("fuga", "fugue", "fuguet", "contrapunctus", "ricercar", "ricarcare",
         "recercar", "canzon", "fantasia")
