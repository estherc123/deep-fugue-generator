"""Human-readable deduplication and split report (Spec §24.2).

Covers the build-report items this stage owns: duplicate-group statistics, split
counts, and the leakage-gate results.
"""

from __future__ import annotations

import collections
from pathlib import Path

SPLITS = ("train", "validation", "test")


def write_split_report(summary: dict, dedup: dict, rows: list[dict], path: Path) -> None:
    L: list[str] = []
    L.append("# Deduplication and dataset splits")
    L.append("")
    L.append(f"**Splits version:** {summary['splits_version']} · "
             f"**seed:** {summary['split_seed']} · "
             f"**pieces:** {summary['total_pieces']} · "
             f"**split units:** {summary['total_split_units']}")
    L.append("")
    errs = summary["leakage_errors"]
    L.append(f"**Leakage gates (§13.5):** {'✅ all pass' if not errs else f'❌ {len(errs)} failure(s)'}")
    L.append("")

    # ---- §12 ------------------------------------------------------------- #
    L.append("## Deduplication (§12)")
    L.append("")
    L.append("All six levels are computed. *Blocking* levels fuse their members into "
             "one split unit so the group cannot straddle a split; the others are "
             "similarity signals, reported but not fused.")
    L.append("")
    L.append("| Level | Match groups | Pieces | Blocking |")
    L.append("| --- | ---: | ---: | :---: |")
    for level, c in summary["duplicate_level_census"].items():
        L.append(f"| {level} | {c['match_groups']} | {c['pieces_involved']} | "
                 f"{'✅' if c['blocking'] else '—'} |")
    L.append("")
    L.append(f"### {len(dedup['groups'])} duplicate groups")
    L.append("")
    L.append("Nothing is deleted: every member keeps its provenance, and one "
             "preferred representative is named per group.")
    L.append("")
    L.append("| Group | Members | Preferred | Matched at |")
    L.append("| --- | --- | --- | --- |")
    for g in dedup["groups"]:
        L.append(f"| `{g['duplicate_group_id']}` | {', '.join(g['members'])} | "
                 f"`{g['preferred_piece_id']}` | {', '.join(g['matched_levels'])} |")
    L.append("")

    # ---- §13 ------------------------------------------------------------- #
    L.append("## Splits (§13)")
    L.append("")
    for corpus, c in summary["corpora"].items():
        L.append(f"### `{corpus}` — {c['split_units']} split units, {c['strata']} strata")
        L.append("")
        L.append("| Split | Target works | Assigned works | Pieces |")
        L.append("| --- | ---: | ---: | ---: |")
        for s in SPLITS:
            L.append(f"| {s} | {c['targets'].get(s, 0)} | "
                     f"{c['works_by_split'].get(s, 0)} | {c['pieces_by_split'].get(s, 0)} |")
        L.append("")
        L.append(f"Stratified by: {', '.join(f'`{f}`' for f in c['stratify_by'])}.")
        L.append("")
        # Stratification balance: how each field's values landed across splits.
        subset = [r for r in rows if r["corpus"] == corpus]
        for field in c["stratify_by"]:
            table: dict = collections.defaultdict(collections.Counter)
            for r in subset:
                v = r["stratification_fields"].get(field)
                if v is not None:
                    table[str(v)][r["split"]] += 1
            if not table:
                continue
            L.append(f"<details><summary><code>{field}</code> balance</summary>")
            L.append("")
            L.append("| value | " + " | ".join(SPLITS) + " |")
            L.append("| --- | " + " | ".join("---:" for _ in SPLITS) + " |")
            for v in sorted(table):
                L.append(f"| {v} | " + " | ".join(str(table[v][s]) for s in SPLITS) + " |")
            L.append("")
            L.append("</details>")
            L.append("")

    # ---- gates ------------------------------------------------------------ #
    L.append("## Leakage gates (§13.5)")
    L.append("")
    checks = [
        "No `work_group_id` appears in multiple splits",
        "No exact or transposition-normalized duplicate crosses splits",
        "Every split unit is wholly within one split",
        "Every `assignment_hash` verifies against seed, unit and members",
        "Per-corpus work counts match the configured targets",
    ]
    for c in checks:
        L.append(f"- {'✅' if not errs else '⚠️'} {c}")
    L.append("")
    if errs:
        L.append("### Failures")
        L.append("")
        for e in errs:
            L.append(f"- {e}")
        L.append("")
    L.append("Excerpt, transposition and synthetic-derivative gates (§13.1) apply to "
             "stages not yet built; every derivative inherits `split` from its parent "
             "piece through `split_unit_id`.")
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))
