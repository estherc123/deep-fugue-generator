"""Acquisition orchestration: pin sources, checksum files, build the source
manifest, classify pieces, and verify expected counts (Spec Sections 6, 7, 9).

Dependency-light on purpose (stdlib + PyYAML) so acquisition runs before the
heavy symbolic-music stack (music21, OR-Tools) is installed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

import yaml

from .provenance import pin_repository, Checkout

# --------------------------------------------------------------------------- #
# Format inference
# --------------------------------------------------------------------------- #
_EXT_FORMAT = {
    ".krn": "kern",
    ".kern": "kern",
    ".musicxml": "musicxml",
    ".xml": "musicxml",
    ".mxl": "musicxml",
    ".mei": "mei",
    ".csv": "csv",
    ".json": "json",
    ".dez": "json",  # Dezrann analysis files are JSON containers
}

# Files we never want as manifest rows (repo scaffolding / our own markers).
_SKIP_NAMES = {".pinned_commit", ".retrieved_at_utc", ".gitignore", ".gitattributes"}
_SKIP_DIR_PARTS = {".git", "__MACOSX"}


def infer_format(path: Path) -> str:
    return _EXT_FORMAT.get(path.suffix.lower(), "other")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Piece identity classification (Spec Section 7)
# --------------------------------------------------------------------------- #
# WTC I fugues → BWV 846-869; WTC II fugues → BWV 870-893.
_WTC_RE = re.compile(r"wtc([12])([fp])(\d{2})", re.IGNORECASE)


def wtc_bwv(book: int, kind: str, index: int) -> str:
    base = {1: 846, 2: 870}[book]
    return f"BWV{base + index - 1}"


@dataclass
class PieceIdentity:
    composer_claimed: str | None = None
    work_id_claimed: str | None = None      # canonical work group (e.g. BWV846)
    movement_claimed: str | None = None     # fugue | prelude | chorale | annotation
    piece_id: str | None = None             # source-independent movement identity
    variant_id: str | None = None
    included_in_default: bool = True
    quarantine_reason: str | None = None


def classify(source_name: str, rel_path: str, fmt: str) -> PieceIdentity:
    name = Path(rel_path).name
    stem = Path(rel_path).stem

    if source_name == "humdrum-wtc":
        m = _WTC_RE.search(name)
        if m and fmt == "kern":
            book, kind_ch, idx = int(m.group(1)), m.group(2).lower(), int(m.group(3))
            kind = "fugue" if kind_ch == "f" else "prelude"
            bwv = wtc_bwv(book, kind_ch, idx)
            return PieceIdentity(
                composer_claimed="bach_js",
                work_id_claimed=bwv,
                movement_claimed=kind,
                piece_id=f"bach_wtc{book}_{kind}_{idx:02d}",
                variant_id="humdrum-default",
                # Preludes are preserved but excluded from the default fugue set.
                included_in_default=(kind == "fugue"),
            )
        # Non-score files (docs, indices) are metadata, not default corpus.
        return PieceIdentity(included_in_default=False)

    if source_name == "bcfb":
        # Canonical encoded variants live directly under the *_master parallel dirs:
        #   FB_source/musicXML_master/BWV_10.07a_FB.musicxml   (143 variants)
        #   FB_source/translated_kern/BWV_10.07a_FB.krn        (secondary)
        #   FB_source/translated_mei/BWV_10.07a_FB.mei         (secondary)
        # The editorial_FB_only/ subfolder and the encoding experiment folder are
        # derived material: preserved as raw but excluded from the corpus.
        parent = Path(rel_path).parent.name
        is_master = parent in ("musicXML_master", "translated_kern", "translated_mei")
        chorale_id = _bcfb_chorale_id(stem)  # e.g. BWV_10.07a_FB -> ("BWV10.07", "a")
        if is_master and chorale_id and fmt in ("musicxml", "kern", "mei"):
            work, variant_letter = chorale_id
            return PieceIdentity(
                composer_claimed="bach_js",
                work_id_claimed=work,                       # groups a/b variants (139 unique)
                movement_claimed="chorale",
                piece_id=f"bach_chorale_{work}{variant_letter}".lower(),
                variant_id=f"bcfb-{fmt}-{stem}",
                # MusicXML is the primary canonical source (Spec 3.3); kern/mei are
                # retained for cross-format validation, not the default scoring corpus.
                included_in_default=(fmt == "musicxml"),
            )
        # editorial_FB_only, experiment folder, README/reference tables.
        return PieceIdentity(
            composer_claimed="bach_js" if chorale_id else None,
            work_id_claimed=chorale_id[0] if chorale_id else None,
            movement_claimed="chorale" if chorale_id else None,
            included_in_default=False,
        )

    if source_name == "algomus":
        # Structural annotations for WTC I fugues are the Dezrann *.dez files, named
        #   NN-bwvXXX-ref.dez  (human reference)  /  NN-bwvXXX-computed.dez (algorithmic)
        # The synchro/*.json files are audio-alignment maps, not structural labels.
        m = re.match(r"(\d{2})-bwv(\d+)-(ref|computed)", name, re.IGNORECASE)
        if rel_path.endswith(".dez") and m:
            idx, bwv_num, kind = int(m.group(1)), m.group(2), m.group(3).lower()
            return PieceIdentity(
                composer_claimed="bach_js",
                work_id_claimed=f"BWV{bwv_num}",
                movement_claimed="annotation",
                piece_id=f"bach_wtc1_fugue_{idx:02d}",  # links to the WTC canonical piece
                variant_id=f"algomus-{kind}",
                included_in_default=False,  # annotations attach to pieces, not corpus rows
            )
        return PieceIdentity(
            composer_claimed="bach_js",
            movement_claimed="annotation",
            included_in_default=False,
        )

    return PieceIdentity(included_in_default=False)


# BCFB filenames: BWV_10.07a_FB -> base "10.07", variant "a"; BWV_250_FB -> "250".
# The movement dot is optional: 11 standalone chorales (BWV 250-252, 447, ...) have
# no cantata-movement suffix.
_BCFB_RE = re.compile(r"BWV[_\-]?(\d+(?:\.\d+)?)([a-z])?(?=_|\.|$)", re.IGNORECASE)


def _bcfb_chorale_id(stem: str) -> tuple[str, str] | None:
    """Return (work_group_id, variant_letter) e.g. ('BWV10.07', 'a').

    The work_group_id strips the trailing a/b variant letter so alternate
    encodings of the same chorale share one canonical work identity (Spec 7).
    """
    m = _BCFB_RE.search(stem)
    if not m:
        return None
    return f"BWV{m.group(1)}", (m.group(2) or "").lower()


# --------------------------------------------------------------------------- #
# Manifest building
# --------------------------------------------------------------------------- #
@dataclass
class SourceSpec:
    source_name: str
    repository_url: str
    repository_commit: str
    license_name: str | None
    license_url: str | None
    license_scope: str
    commercial_use_known: bool | None
    attribution_required: bool | None


@dataclass
class AcquireResult:
    source_name: str
    checkout: Checkout
    rows: list[dict] = field(default_factory=list)

    def counts(self) -> dict:
        return summarize_rows(self.source_name, self.rows)


def load_sources(config_path: Path) -> list[SourceSpec]:
    data = yaml.safe_load(config_path.read_text())
    specs = []
    for s in data["sources"]:
        specs.append(
            SourceSpec(
                source_name=s["source_name"],
                repository_url=s["repository_url"],
                repository_commit=s["repository_commit"],
                license_name=s.get("license_name"),
                license_url=s.get("license_url"),
                license_scope=s.get("license_scope", "unknown"),
                commercial_use_known=s.get("commercial_use_known"),
                attribution_required=s.get("attribution_required"),
            )
        )
    return specs


def _iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name in _SKIP_NAMES:
            continue
        if any(part in _SKIP_DIR_PARTS for part in p.parts):
            continue
        yield p


def acquire_source(
    spec: SourceSpec, raw_root: Path, *, force: bool = False
) -> AcquireResult:
    checkout = pin_repository(
        spec.source_name,
        spec.repository_url,
        spec.repository_commit,
        raw_root,
        force=force,
    )
    rows: list[dict] = []
    for path in _iter_files(checkout.root):
        rel = path.relative_to(checkout.root).as_posix()
        fmt = infer_format(path)
        ident = classify(spec.source_name, rel, fmt)
        digest = sha256_file(path)
        source_file_id = f"{spec.source_name}:{digest[:12]}"
        rows.append(
            {
                "source_file_id": source_file_id,
                "source_name": spec.source_name,
                "repository_url": spec.repository_url,
                "repository_commit": spec.repository_commit,
                "retrieved_at_utc": checkout.retrieved_at_utc,
                "relative_source_path": rel,
                "sha256": digest,
                "source_format": fmt,
                "composer_claimed": ident.composer_claimed,
                "work_id_claimed": ident.work_id_claimed,
                "movement_claimed": ident.movement_claimed,
                "license_name": spec.license_name,
                "license_url": spec.license_url,
                "license_scope": spec.license_scope,
                "commercial_use_known": spec.commercial_use_known,
                "attribution_required": spec.attribution_required,
                "quarantine_reason": ident.quarantine_reason,
                "ingest_status": "quarantined" if ident.quarantine_reason else "accepted",
                # Non-schema helper fields (dropped from the strict manifest export):
                "_piece_id": ident.piece_id,
                "_variant_id": ident.variant_id,
                "_included_in_default": ident.included_in_default,
            }
        )
    return AcquireResult(spec.source_name, checkout, rows)


# --------------------------------------------------------------------------- #
# Summaries & counting (Spec Section 9.2 acceptance criteria)
# --------------------------------------------------------------------------- #
def summarize_rows(source_name: str, rows: list[dict]) -> dict:
    out: dict = {"source_name": source_name, "total_files": len(rows)}
    by_fmt: dict[str, int] = {}
    for r in rows:
        by_fmt[r["source_format"]] = by_fmt.get(r["source_format"], 0) + 1
    out["files_by_format"] = by_fmt

    if source_name == "humdrum-wtc":
        fugues = [r for r in rows if r["movement_claimed"] == "fugue"]
        preludes = [r for r in rows if r["movement_claimed"] == "prelude"]
        wtc1 = sorted({r["_piece_id"] for r in fugues if r["_piece_id"].startswith("bach_wtc1")})
        wtc2 = sorted({r["_piece_id"] for r in fugues if r["_piece_id"].startswith("bach_wtc2")})
        out["wtc1_fugues"] = len(wtc1)
        out["wtc2_fugues"] = len(wtc2)
        out["preludes"] = len(preludes)
    elif source_name == "bcfb":
        # Canonical variants = primary-format (musicxml) files in the master dir.
        variants = [r for r in rows if r["_included_in_default"]]
        unique_works = sorted({r["work_id_claimed"] for r in variants})
        out["canonical_variants"] = len(variants)            # target 143
        out["unique_chorales"] = len(unique_works)           # target 139
        out["musicxml_master_files"] = len(variants)
        out["kern_secondary_files"] = len(
            [r for r in rows if r["source_format"] == "kern"
             and Path(r["relative_source_path"]).parent.name == "translated_kern"])
        out["mei_secondary_files"] = len(
            [r for r in rows if r["source_format"] == "mei"
             and Path(r["relative_source_path"]).parent.name == "translated_mei"])
        out["retained_non_default_files"] = len(rows) - len(variants)
    elif source_name == "algomus":
        dez = [r for r in rows if r["relative_source_path"].endswith(".dez")]
        ref = [r for r in dez if r["_variant_id"] == "algomus-ref"]
        computed = [r for r in dez if r["_variant_id"] == "algomus-computed"]
        covered = sorted({r["_piece_id"] for r in dez if r["_piece_id"]})
        ref_covered = sorted({r["_piece_id"] for r in ref if r["_piece_id"]})
        out["dez_files"] = len(dez)
        out["ref_annotation_fugues"] = len(ref_covered)      # target 24 (upstream: 23)
        out["computed_annotation_fugues"] = len(
            {r["_piece_id"] for r in computed if r["_piece_id"]})
        out["wtc1_fugues_with_any_annotation"] = len(covered)
        # Which WTC I fugue indices lack a reference annotation, for the report.
        all_idx = {f"bach_wtc1_fugue_{i:02d}" for i in range(1, 25)}
        out["wtc1_fugues_missing_ref"] = sorted(all_idx - set(ref_covered))
    return out
