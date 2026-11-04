# Checkpoints

Half-precision copies of the full checkpoints in `data/models/` (git-ignored),
so they stay under GitHub's 100 MB file limit. Weights load into fp32 models
unchanged (`sample.load_model`); sampling runs in bf16 anyway.

| file | what |
| --- | --- |
| generator_v7_dpo.fp16.pt | v7 after DPO against drift (Bach's continuation preferred over the model's); current best |
| generator_v7.fp16.pt | v7: 33M params, full expanded corpus, chord/texture plan layer |
| critic_v2.fp16.pt | pairwise "real vs worse" critic; failed the owner's blind test (9/16), not used for training |

Use, e.g.:

    PYTHONPATH=src python -m fugue_data.generator.sample --model checkpoints/generator_v7_dpo.fp16.pt \
        --piece bach_wtc1_fugue_02 --samples 3 --force-sequences --key-bias 3 --chord-bias 1.5 \
        --temperature 0.85 --no-crossing --repeat-penalty 2
