"""fugue-data command-line interface (Spec Section 23).

Currently implemented subcommands:
  acquire   Clone + pin + checksum + manifest the authoritative sources.
  normalize Parse accepted scores into the canonical representation.
  annotate  Align structural annotations onto canonical scores.
  split     Deduplicate and assign deterministic complete-work splits.
  tokenize  Derive grid and event model views from canonical scores.
  propose-excerpts / review-excerpts   Real-Bach excerpt proposal and review.
  generate-valid / generate-invalid / generate-pairs / review-pairs
  generate-corrupted  Real Bach windows vs rule-valid corrupted copies.
  validate  Run the pipeline-wide validation gates and visual QA.
  export    Write the training exports.
  report    Assemble the final build report.

Every command is restartable, avoids duplicating successful work, writes a
structured summary, and returns nonzero when acceptance criteria fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .acquire.acquire import load_sources, acquire_source
from .acquire.manifest import write_manifest


def cmd_normalize(args) -> int:
    # Imported lazily so `acquire` works without the heavy music21 stack.
    rc = 0
    corpus = args.corpus or "all"
    if corpus in ("wtc", "all"):
        from .normalize.runner import normalize_wtc
        s = normalize_wtc(force=args.force, limit=args.limit)
        print("WTC fugues:", json.dumps({k: v for k, v in s.items() if k != "results"}))
        done = s["normalized_count"] + s["skipped_count"]
        if s["failed_count"] or s["validation_failure_count"] or (
                not args.limit and done != s["expected_wtc_fugues"]):
            print("[normalize] WTC acceptance failure", file=sys.stderr)
            rc = 1
    if corpus in ("chorales", "all"):
        from .normalize.chorales import normalize_chorales
        s = normalize_chorales(force=args.force, limit=args.limit)
        print("BCFB chorales:", json.dumps({k: v for k, v in s.items() if k != "results"}))
        done = s["normalized_count"] + s["skipped_count"]
        if s["failed_count"] or s["validation_failure_count"] or (
                not args.limit and done != s["expected_variants"]):
            print("[normalize] chorale acceptance failure", file=sys.stderr)
            rc = 1
    if rc == 0:
        print("\n[normalize] corpus normalized and validated.")
    return rc


def cmd_annotate(args) -> int:
    from .annotations.runner import annotate_algomus
    s = annotate_algomus(force=args.force, limit=args.limit)
    print("Algomus annotations:", json.dumps(
        {k: v for k, v in s.items() if k not in ("results", "overrides")}, indent=2))
    if s["failures"]:
        print("\n[annotate] ACCEPTANCE FAILURES:", file=sys.stderr)
        for f in s["failures"]:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"\n[annotate] {s['aligned_count'] + s['skipped_count']} fugues aligned and validated.")
    return 0


def cmd_split(args) -> int:
    from .splitting.runner import run_split
    s = run_split(seed=args.seed)
    print("Splits:", json.dumps(
        {k: v for k, v in s.items()
         if k not in ("results", "failures", "duplicate_level_census")}, indent=2))
    if s["leakage_errors"]:
        print("\n[split] LEAKAGE / ACCEPTANCE FAILURES:", file=sys.stderr)
        for f in s["leakage_errors"][:20]:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"\n[split] {s['total_pieces']} pieces in {s['total_split_units']} split units; "
          f"{s['duplicate_groups']} duplicate groups; all leakage gates pass.")
    return 0


def _emit(stage_name: str, summary: dict, ok_msg: str) -> int:
    """Print a stage summary, log it, and return the process exit code."""
    from .logging_util import log_event
    printable = {k: v for k, v in summary.items()
                 if k not in ("results", "artefacts", "gates", "coverage")}
    print(f"{stage_name}:", json.dumps(printable, indent=2, sort_keys=True)[:4000])
    failures = summary.get("failures") or []
    log_event(stage_name, status="failed" if failures else "ok",
              details={"failures": failures[:10]})
    if failures:
        print(f"\n[{stage_name}] ACCEPTANCE FAILURES:", file=sys.stderr)
        for f in failures[:20]:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"\n[{stage_name}] {ok_msg}")
    return 0


def cmd_tokenize(args) -> int:
    from .tokenization.runner import tokenize_corpus
    s = tokenize_corpus(limit=args.limit)
    return _emit("tokenize", s,
                 f"{s['pieces_tokenized']} pieces tokenized, "
                 f"{s['roundtrip_failures']} grid round-trip failures.")


def cmd_propose_excerpts(args) -> int:
    from .review.runner import propose_excerpts
    if args.dry_run:
        print("[propose-excerpts] dry run: would propose "
              f"{args.count or 'the configured target'} candidates and write "
              "data/proposed/excerpts.jsonl plus renders. No files written.")
        return 0
    s = propose_excerpts(count=args.count, render=not args.no_render)
    return _emit("propose_excerpts", s,
                 f"{s['candidates_proposed']} candidates pending review.")


def cmd_review_excerpts(args) -> int:
    from .review.app import serve
    serve(port=args.port)
    return 0


def cmd_generate_valid(args) -> int:
    from .generation.valid_runner import generate_valid
    if args.dry_run:
        print("[generate-valid] dry run: would generate "
              f"{args.count or 'the configured target'} fragments under profile "
              f"{args.profile or 'from config'}. No files written.")
        return 0
    s = generate_valid(count=args.count, profile_name=args.profile,
                       resume=not args.restart)
    return _emit("generate_valid", s, f"{s.get('accepted')} hard-valid fragments.")


def cmd_generate_invalid(args) -> int:
    from .generation.invalid_runner import generate_invalid
    if args.dry_run:
        print("[generate-invalid] dry run: would generate "
              f"{args.count or 'the configured target'} controlled negatives. "
              "No files written.")
        return 0
    s = generate_invalid(count=args.count)
    return _emit("generate_invalid", s, f"{s['accepted']} controlled negatives.")


def cmd_generate_pairs(args) -> int:
    from .generation.pair_candidates import generate_pair_candidates
    from .generation.pairs_runner import generate_pairs
    if args.dry_run:
        target = args.target or "the configured target"
        print(f"[generate-pairs] dry run: would solve {args.prompts} prompts and "
              f"build up to {target} pairs. No files written.")
        return 0
    if not args.pairs_only:
        c = generate_pair_candidates(prompts=args.prompts)
        print("candidates:", json.dumps(
            {k: v for k, v in c.items() if k != "failures"}, sort_keys=True))
    s = generate_pairs(target=args.target)
    return _emit("generate_pairs", s,
                 f"{s['pairs']} candidate pairs pending human labels.")


def cmd_generate_corrupted(args) -> int:
    from .corruption.runner import generate_corrupted
    if args.dry_run:
        print("[generate-corrupted] dry run: would write corrupted-Bach pairs for "
              f"{args.ops or 'every configured operator'}. No files written.")
        return 0
    s = generate_corrupted(workers=args.workers, limit=args.limit, ops=args.ops)
    return _emit("generate_corrupted", s,
                 f"{s['pairs']} corrupted-Bach pairs over {s['pieces_with_pairs']} pieces.")


def cmd_review_pairs(args) -> int:
    from .review.pairs_app import serve
    serve(port=args.port)
    return 0


def cmd_validate(args) -> int:
    from .validation.gates import run_gates
    from .validation.visual_qa import run_visual_qa
    g = run_gates()
    print("gates:", json.dumps(
        {k: v for k, v in g.items() if k != "gates"}, indent=2, sort_keys=True))
    for r in g["gates"]:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['gate']:38} "
              f"checked={r['checked']}")
    rc = 1 if g["gates_failed"] else 0
    if args.skip_visual:
        return rc
    v = run_visual_qa()
    print("\nvisual QA:", json.dumps(
        {k: x for k, x in v.items() if k != "artefacts"}, indent=2, sort_keys=True)[:1500])
    if v["failures"]:
        rc = 1
    return _emit("validate", {"failures": g["failures"] + v["failures"],
                              "gates_passed": g["gates_passed"],
                              "gates_total": g["gates_total"],
                              "visual_artefacts": v["total_rendered"]},
                 "all gates pass and visual QA recorded.") or rc


def cmd_export(args) -> int:
    from .export.exporters import export_all
    s = export_all()
    return _emit("export", s, "training exports written to data/exports/.")


def cmd_report(args) -> int:
    from .report.build_report import build
    r = build()
    s = r["definition_of_done_summary"]
    print(json.dumps({"versions": r["versions"],
                      "definition_of_done": s}, indent=2, sort_keys=True))
    print("\n=== §28 Definition of Done ===")
    for k, v in r["definition_of_done"].items():
        mark = "[x]" if v["met"] else ("[human]" if v.get("requires_human_review")
                                       else "[ ]")
        print(f"  {mark} {k}")
        if not v["met"]:
            print(f"        {v['detail']}")
    print(f"\nreports/build_report.json written ({s['met']}/{s['total']} met)")
    from .logging_util import log_event
    log_event("report", status="failed" if s["outstanding_other"] else "ok",
              details={"met": s["met"], "total": s["total"],
                       "outstanding_requiring_human_review":
                           s["outstanding_requiring_human_review"],
                       "outstanding_other": s["outstanding_other"]})
    # Outstanding human review is a legitimate state, not a pipeline failure; an
    # unmet criterion that does NOT need a person is a real failure.
    return 1 if s["outstanding_other"] else 0


def cmd_export_audio(args) -> int:
    from .review.export_audio import export
    s = export(count=args.count,
               out_dir=Path(args.out) if args.out else None,
               only_revision=args.revision)
    print(json.dumps(s, indent=2, sort_keys=True))
    print(f"\nOpen {s['index']} in a browser, or send the whole folder.")
    return 0


def cmd_reviewer_agreement(args) -> int:
    from .review.agreement import analyse
    s = analyse()
    print(json.dumps(s, indent=2, sort_keys=True))
    return 0


def cmd_merge_preview(args) -> int:
    from .generation.merge_preview import merge
    s = merge(preview_root=Path(args.preview_root),
              labelled_only=args.labelled_only)
    print(json.dumps(s, indent=2, sort_keys=True))
    return 0


def cmd_export_agent_batch(args) -> int:
    from .review.agent_review import export_batch
    r = export_batch(offset=args.offset, limit=args.limit,
                     out_path=Path(args.out) if args.out else None,
                     include_labelled=args.redo)
    print(json.dumps(r, indent=2))
    return 0


def cmd_ingest_agent_labels(args) -> int:
    from .review.agent_review import ingest
    s = ingest(verdicts_path=Path(args.verdicts),
               batch_path=Path(args.batch) if args.batch else None,
               reviewer=args.reviewer)
    print(json.dumps(s, indent=2, sort_keys=True))
    return 1 if s["errors"] else 0


def _repo_root() -> Path:
    # src/fugue_data/cli.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[2]


def _check_counts(summaries: dict, expected: dict, overrides: dict) -> list[str]:
    """Return a list of acceptance-criteria failures (Spec Section 9.2)."""
    failures: list[str] = []

    def check(key, actual, want):
        if key in overrides:
            return
        if actual is None:
            return
        if actual != want:
            failures.append(f"{key}: expected {want}, got {actual}")

    wtc = summaries.get("humdrum-wtc")
    if wtc:
        check("wtc1_fugues", wtc.get("wtc1_fugues"), expected["wtc1_fugues"])
        check("wtc2_fugues", wtc.get("wtc2_fugues"), expected["wtc2_fugues"])
    bcfb = summaries.get("bcfb")
    if bcfb:
        check("bcfb_encoded_variants", bcfb.get("canonical_variants"),
              expected["bcfb_encoded_variants"])
        check("bcfb_unique_chorales", bcfb.get("unique_chorales"),
              expected["bcfb_unique_chorales"])
    algo = summaries.get("algomus")
    if algo:
        check("algomus_wtc1_coverage", algo.get("ref_annotation_fugues"),
              expected["algomus_wtc1_coverage"])
    return failures


def cmd_acquire(args) -> int:
    root = _repo_root()
    sources_cfg = root / "config" / "sources.yaml"
    pipeline_cfg = yaml.safe_load((root / "config" / "pipeline.yaml").read_text())
    raw_root = root / pipeline_cfg["paths"]["raw"]
    manifests_dir = root / pipeline_cfg["paths"]["manifests"]
    expected = pipeline_cfg["acquisition"]["expected"]
    overrides = pipeline_cfg["acquisition"].get("overrides") or {}

    specs = load_sources(sources_cfg)
    if args.source and not args.all:
        specs = [s for s in specs if s.source_name == args.source]
        if not specs:
            print(f"error: unknown source {args.source!r}", file=sys.stderr)
            return 2

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for spec in specs:
        print(f"[acquire] pinning {spec.source_name} @ {spec.repository_commit[:10]} ...",
              flush=True)
        result = acquire_source(spec, raw_root, force=args.force)
        all_rows.extend(result.rows)
        summaries[spec.source_name] = result.counts()
        s = summaries[spec.source_name]
        print(f"[acquire]   {s['total_files']} files, formats={s['files_by_format']}",
              flush=True)

    # When acquiring a single source, merge with existing manifest rows for others.
    if args.source and not args.all and (manifests_dir / "sources.jsonl").exists():
        keep = [
            json.loads(l)
            for l in (manifests_dir / "sources.jsonl").read_text().splitlines()
            if l.strip() and json.loads(l)["source_name"] != args.source
        ]
        all_rows = keep + all_rows

    written = write_manifest(all_rows, manifests_dir)
    print(f"[acquire] manifest: {written['rows']} rows -> {written['jsonl']}")

    failures = _check_counts(summaries, expected, overrides)
    summary_path = manifests_dir / "acquire_summary.json"
    summary_path.write_text(json.dumps(
        {"summaries": summaries, "expected": expected, "failures": failures},
        indent=2, sort_keys=True) + "\n")

    print("\n=== Acquisition summary ===")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    if failures:
        print("\n[acquire] ACCEPTANCE FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\n[acquire] all checked acceptance criteria passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fugue-data")
    sub = p.add_subparsers(dest="command", required=True)

    acq = sub.add_parser("acquire", help="clone + pin + checksum + manifest sources")
    acq.add_argument("--source", choices=["humdrum-wtc", "algomus", "bcfb"],
                     help="acquire a single source")
    acq.add_argument("--all", action="store_true", help="acquire every source")
    acq.add_argument("--force", action="store_true",
                     help="re-acquire even if a valid pinned checkout exists")
    acq.set_defaults(func=cmd_acquire)

    nrm = sub.add_parser("normalize", help="normalize scores to canonical representation")
    nrm.add_argument("--all", action="store_true", help="normalize the full corpus")
    nrm.add_argument("--corpus", choices=["wtc", "chorales", "all"], default=None,
                     help="which corpus to normalize (default: all)")
    nrm.add_argument("--force", action="store_true", help="re-normalize existing canonical files")
    nrm.add_argument("--limit", type=int, default=None, help="normalize only the first N (debug)")
    nrm.set_defaults(func=cmd_normalize)

    ann = sub.add_parser("annotate", help="align structural annotations onto canonical scores")
    ann.add_argument("--source", choices=["algomus"], default="algomus",
                     help="annotation source (default: algomus)")
    ann.add_argument("--force", action="store_true", help="re-align existing annotation sets")
    ann.add_argument("--limit", type=int, default=None, help="align only the first N (debug)")
    ann.set_defaults(func=cmd_annotate)

    spl = sub.add_parser("split", help="deduplicate and assign complete-work splits")
    spl.add_argument("--seed", type=int, default=None,
                     help="split seed (default: config/splits.yaml split_seed)")
    spl.set_defaults(func=cmd_split)

    tok = sub.add_parser("tokenize", help="derive grid and event model views")
    tok.add_argument("--limit", type=int, default=None)
    tok.set_defaults(func=cmd_tokenize)

    pe = sub.add_parser("propose-excerpts", help="propose real-Bach excerpt candidates")
    pe.add_argument("--count", type=int, default=None)
    pe.add_argument("--no-render", action="store_true", help="skip SVG/MIDI renders")
    pe.add_argument("--dry-run", action="store_true")
    pe.set_defaults(func=cmd_propose_excerpts)

    re_ = sub.add_parser("review-excerpts", help="open the excerpt review app")
    re_.add_argument("--port", type=int, default=8765)
    re_.set_defaults(func=cmd_review_excerpts)

    gv = sub.add_parser("generate-valid", help="generate hard-valid fragments")
    gv.add_argument("--count", type=int, default=None)
    gv.add_argument("--profile", choices=["species_strict", "high_baroque_conservative",
                                          "bach_empirical"], default=None)
    gv.add_argument("--restart", action="store_true", help="ignore the checkpoint")
    gv.add_argument("--dry-run", action="store_true")
    gv.set_defaults(func=cmd_generate_valid)

    gi = sub.add_parser("generate-invalid", help="generate controlled single-violation "
                                                 "fragments")
    gi.add_argument("--count", type=int, default=None)
    gi.add_argument("--dry-run", action="store_true")
    gi.set_defaults(func=cmd_generate_invalid)

    gp = sub.add_parser("generate-pairs", help="generate preference-pair candidates")
    gp.add_argument("--target", type=int, default=None)
    gp.add_argument("--prompts", type=int, default=700)
    gp.add_argument("--pairs-only", action="store_true",
                    help="rebuild pairs from existing candidates")
    gp.add_argument("--dry-run", action="store_true")
    gp.set_defaults(func=cmd_generate_pairs)

    gc = sub.add_parser("generate-corrupted",
                        help="pair real Bach windows with rule-valid corrupted copies")
    gc.add_argument("--ops", nargs="+", default=None,
                    help="only these operators (default: all in config/corruption.yaml)")
    gc.add_argument("--workers", type=int, default=None)
    gc.add_argument("--limit", type=int, default=None, help="first N pieces only")
    gc.add_argument("--dry-run", action="store_true")
    gc.set_defaults(func=cmd_generate_corrupted)

    rp = sub.add_parser("review-pairs", help="open the preference review app")
    rp.add_argument("--port", type=int, default=8766)
    rp.set_defaults(func=cmd_review_pairs)

    va = sub.add_parser("validate", help="run validation gates and visual QA")
    va.add_argument("--all", action="store_true", help="accepted for symmetry")
    va.add_argument("--skip-visual", action="store_true")
    va.set_defaults(func=cmd_validate)

    ex = sub.add_parser("export", help="write the training exports")
    ex.add_argument("--all", action="store_true", help="accepted for symmetry")
    ex.set_defaults(func=cmd_export)

    ea = sub.add_parser("export-audio",
                        help="render generated fragments to WAV + MIDI for sharing")
    ea.add_argument("--count", type=int, default=25)
    ea.add_argument("--out", default=None)
    ea.add_argument("--revision", default=None,
                    help="only fragments from this generator revision")
    ea.set_defaults(func=cmd_export_audio)

    ra = sub.add_parser("reviewer-agreement",
                        help="measure agreement between reviewers of preference pairs")
    ra.set_defaults(func=cmd_reviewer_agreement)

    mp = sub.add_parser("merge-preview",
                        help="fold a reviewed preview batch into the real dataset")
    mp.add_argument("--preview-root", default="/tmp/preview")
    mp.add_argument("--labelled-only", action="store_true",
                    help="merge only pairs that carry a human label")
    mp.set_defaults(func=cmd_merge_preview)

    eab = sub.add_parser("export-agent-batch",
                         help="write a blinded batch of pairs for agent review")
    eab.add_argument("--offset", type=int, default=0)
    eab.add_argument("--limit", type=int, default=50)
    eab.add_argument("--out", default=None)
    eab.add_argument("--redo", action="store_true",
                     help="include pairs an agent has already labelled")
    eab.set_defaults(func=cmd_export_agent_batch)

    ial = sub.add_parser("ingest-agent-labels",
                         help="ingest agent verdicts into agent_labels.jsonl")
    ial.add_argument("--verdicts", required=True)
    ial.add_argument("--batch", default=None)
    ial.add_argument("--reviewer", default="agent")
    ial.set_defaults(func=cmd_ingest_agent_labels)

    rep = sub.add_parser("report", help="assemble the final build report")
    rep.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "acquire" and not args.all and not args.source:
        parser.error("acquire requires --all or --source")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
