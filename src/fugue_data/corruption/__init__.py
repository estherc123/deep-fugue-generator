"""Corrupted-Bach pairs: real passages against rule-valid, musically damaged copies.

Every pair has a winner by construction -- the original. The corruptions are
chosen to damage what the rule engine cannot see (succession, key plan, surface
rhythm, thematic return), while the hard-rule gate guarantees the damaged copy is
still legal. A judge trained on these pairs therefore cannot win by spotting rule
violations; it has to learn something about how Bach's music is put together.
"""
