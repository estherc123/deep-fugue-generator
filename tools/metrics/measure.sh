#!/bin/bash
# All metrics for one composition set. Usage: bash tools/metrics/measure.sh data/review/compositions_vX
cd "$(dirname "$0")/../.."
D=$1; P=".venv/bin/python"
PYTHONPATH=src $P -m fugue_data.generator.review_compositions $D | tail -1   # forbidden breaks
$P tools/metrics/inkey.py $D                 # chords in planned key (Bach 84.6%)
$P tools/metrics/semi.py $D                  # semitone clash per beat (Bach 7.0%)
$P tools/metrics/exposed.py $D               # 2-voice beat dissonance (Bach 0.18)
$P tools/metrics/ook.py $D                   # free notes out of key (Bach 5.9%)
$P tools/metrics/texture_stats.py $D 2>/dev/null | tail -1   # unison% (Bach 1.55), cross%, repeat%
