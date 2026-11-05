#!/bin/bash
# Compose the 10 held-out subjects x seeds 1,2 x 2 samples (= 40 fugues) into OUT,
# 4 workers over 2 GPUs (~40 min). Usage: bash tools/metrics/compose_set.sh data/review/compositions_vX
cd "$(dirname "$0")/../.."
OUT=$1
A="bach_wtc1_fugue_02 bach_wtc1_fugue_03 bach_wtc2_fugue_01 bach_wtc2_fugue_17 bach_wtc2_fugue_19"
B="bach_wtc2_fugue_22 kernscores-organ_O08C kernscores-organ_VNoordt_F1 kernscores-organ_bwv533 kernscores-organ_bwv543"
run() {  # gpu seed pids...
  local gpu=$1 seed=$2; shift 2
  for pid in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src timeout 3000 .venv/bin/python -m fugue_data.generator.compose2 \
      --subject-from $pid --seed $seed --samples 2 --out $OUT 2>&1 | grep -v -i warn | grep -E "^sample|Traceback|Error" | sed "s/^/$pid s$seed /"
  done
}
run 0 1 $A & run 1 1 $B & run 0 2 $B & run 1 2 $A & wait
echo ALLDONE
