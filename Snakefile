"""Workflow orchestration (Spec §22).

Targets mirror §22.1. The two human checkpoints in §22.2 are real stops: the
`all` target depends on `report`, and `report` succeeds while leaving the two
review criteria outstanding, but `reviewed_excerpts` and `reviewed_preference_pairs`
fail loudly until a person has done the labelling. That is the intended shape --
the pipeline must not manufacture human approval (§27).

  snakemake acquire --cores 4
  snakemake normalize --cores 8
  snakemake annotations --cores 8
  snakemake split --cores 4
  snakemake propose_excerpts --cores 8
  snakemake generate_valid --cores 16
  snakemake generate_invalid --cores 16
  snakemake generate_preference_candidates --cores 8
  snakemake validate --cores 8
  snakemake export --cores 8
  snakemake report --cores 4
  snakemake all --cores 16

After review, resume with:
  snakemake reviewed_excerpts --cores 4
  snakemake reviewed_preference_pairs --cores 4
"""

import json
from pathlib import Path

CLI = "PYTHONPATH=src python -m fugue_data.cli"
M = "manifests"


rule all:
    input: "reports/build_report.json"


rule acquire:
    output: f"{M}/acquire_summary.json"
    shell: f"{CLI} acquire --all"


rule normalize:
    input: f"{M}/acquire_summary.json"
    output: f"{M}/normalization_summary.json"
    shell: f"{CLI} normalize --all"


rule annotations:
    input: f"{M}/normalization_summary.json"
    output: f"{M}/annotation_summary.json"
    shell: f"{CLI} annotate --source algomus"


# §13.1: complete works are split before any window, transposition or derivative.
rule split:
    input: f"{M}/annotation_summary.json"
    output: f"{M}/split_summary.json", f"{M}/splits.jsonl"
    shell: f"{CLI} split"


rule tokenize:
    input: f"{M}/split_summary.json"
    output: f"{M}/tokenization_summary.json"
    shell: f"{CLI} tokenize"


rule propose_excerpts:
    input: f"{M}/split_summary.json"
    output: f"{M}/excerpt_summary.json", "data/proposed/excerpts.jsonl"
    shell: f"{CLI} propose-excerpts"


rule generate_valid:
    input: f"{M}/split_summary.json"
    output: f"{M}/generation_valid_summary.json"
    shell: f"{CLI} generate-valid"


rule generate_invalid:
    input: f"{M}/generation_valid_summary.json"
    output: f"{M}/generation_invalid_summary.json"
    shell: f"{CLI} generate-invalid"


rule generate_preference_candidates:
    input: f"{M}/generation_valid_summary.json"
    output: f"{M}/preference_summary.json"
    shell: f"{CLI} generate-pairs"


# ---- §22.2 human checkpoints ---------------------------------------------- #
# These do not run a reviewer; they assert that review has happened and stop the
# build if it has not. Resume by running them again after using the review apps.
rule reviewed_excerpts:
    input: "data/proposed/excerpts.jsonl"
    output: touch("data/reviewed/excerpts.ok")
    run:
        rows = [json.loads(l) for l in Path(input[0]).read_text().splitlines()
                if l.strip()]
        approved = sum(1 for r in rows
                       if not r.get("auto_rejected") and r["review_status"] == "approved")
        if not (50 <= approved <= 100):
            raise SystemExit(
                f"§14.6 needs 50-100 approved excerpts; {approved} are approved.\n"
                f"Review them: PYTHONPATH=src python -m fugue_data.review.app\n"
                f"Then re-run: snakemake reviewed_excerpts --cores 4")
        Path("data/reviewed").mkdir(parents=True, exist_ok=True)


rule reviewed_preference_pairs:
    input: "data/generated/preference_candidates/pairs.jsonl"
    output: touch("data/reviewed/preference_pairs.ok")
    run:
        rows = [json.loads(l) for l in Path(input[0]).read_text().splitlines()
                if l.strip()]
        labelled = sum(1 for r in rows if r["review_status"] == "reviewed" and r["label"])
        if labelled < 500:
            raise SystemExit(
                f"§18.6 needs at least 500 reviewed pairs; {labelled} are labelled.\n"
                f"Review them: PYTHONPATH=src python -m fugue_data.review.pairs_app\n"
                f"Then re-run: snakemake reviewed_preference_pairs --cores 4")
        Path("data/reviewed").mkdir(parents=True, exist_ok=True)


rule export:
    input:
        f"{M}/tokenization_summary.json",
        f"{M}/generation_invalid_summary.json",
        f"{M}/preference_summary.json",
        f"{M}/excerpt_summary.json",
    output: f"{M}/export_summary.json"
    shell: f"{CLI} export --all"


rule validate:
    input: f"{M}/export_summary.json"
    output: f"{M}/validation_gates.json", f"{M}/visual_qa.json"
    shell: f"{CLI} validate --all"


rule report:
    input: f"{M}/validation_gates.json"
    output: "reports/build_report.json"
    shell: f"{CLI} report"
