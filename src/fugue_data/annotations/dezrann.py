"""Parse Algomus Dezrann `.dez` analyses and map them onto the canonical
annotation vocabulary of Spec §11.2.

A `.dez` file is JSON with a `meta` block and a flat `labels` list. Each label
carries a raw analytical `type` (`S`, `CS1`, `S-inv`, `Cadence`, `ignore`, ...),
a free-text `tag`, a `start` and usually a `duration` in quarter-note units from
the start of the piece, and usually a `staff` -- which is a *voice* index
numbered top-down, not a keyboard staff (see config/annotations.yaml).

The mapping from that vocabulary to the seven canonical types is data, not code:
it lives in `config/annotations.yaml` and is versioned, because changing it
changes annotation semantics downstream (§25.3). Every raw label is preserved
verbatim on the mapped record, so nothing in the source vocabulary is lost.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def load_config(root: Path | None = None) -> dict:
    root = root or _repo_root()
    return yaml.safe_load((root / "config" / "annotations.yaml").read_text())


# NN-bwvXXX-{ref,computed}.dez
_DEZ_RE = re.compile(r"^(\d{2})-bwv(\d+)-(ref|computed)\.dez$", re.IGNORECASE)

# Cadence tags look like " v:PAC" or " I:rIAC" (leading space is in the source);
# a trailing "?" marks analyst uncertainty.
_CADENCE_RE = re.compile(r"^\s*([A-Za-z]+(?:/[A-Za-z]+)?)\s*:\s*([A-Za-z]+)\s*(\?)?\s*$")


@dataclass
class DezFile:
    path: Path
    piece_id: str
    work_group_id: str
    fugue_index: int
    annotator_kind: str          # "ref" | "computed"
    labels: list[dict]
    meta: dict = field(default_factory=dict)


def parse_dez_filename(name: str) -> tuple[int, str, str] | None:
    """`02-bwv847-ref.dez` -> (2, "BWV847", "ref"). None if not an analysis file."""
    m = _DEZ_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), f"BWV{m.group(2)}", m.group(3).lower()


def load_dez(path: Path) -> DezFile | None:
    parsed = parse_dez_filename(path.name)
    if parsed is None:
        return None
    idx, bwv, kind = parsed
    doc = json.loads(path.read_text())
    return DezFile(
        path=path,
        piece_id=f"bach_wtc1_fugue_{idx:02d}",
        work_group_id=bwv,
        fugue_index=idx,
        annotator_kind=kind,
        labels=list(doc.get("labels", [])),
        meta=doc.get("meta", {}),
    )


def discover_dez_files(root: Path | None = None, *, include_computed: bool | None = None
                       ) -> list[DezFile]:
    """All WTC I structural analyses at the pinned Algomus checkout, sorted."""
    root = root or _repo_root()
    cfg = load_config(root)
    if include_computed is None:
        include_computed = bool(cfg["source"].get("include_computed", False))
    base = root / "data" / "raw" / "algomus" / cfg["source"]["relative_dir"]
    out: list[DezFile] = []
    for p in sorted(base.glob("*.dez")):
        f = load_dez(p)
        if f is None:
            continue
        if f.annotator_kind == "computed" and not include_computed:
            continue
        out.append(f)
    out.sort(key=lambda f: (f.fugue_index, f.annotator_kind))
    return out


def _as_list(value) -> list[str]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _tag_words(tag: str | None) -> list[str]:
    """Whitespace-separated qualifier words after the leading label name.

    Tags look like `"CS1 varied-end"` or `"S-var augmented+rhythmicvar-IGN"`;
    the first word restates the type, the rest are qualifiers. Trailing `-IGN`
    markers are stripped -- rejection is carried by the label type, not the tag.
    """
    if not tag:
        return []
    words = tag.strip().split()
    out = []
    for w in words[1:]:
        w = re.sub(r"-IGN$", "", w)
        if w:
            out.append(w)
    return out


@dataclass
class MappedLabel:
    """One source label mapped to the canonical vocabulary, still untimed."""
    type: str
    transformations: list[str]
    analytical_status: str = "accepted"
    subject_index: int | None = None
    countersubject_index: int | None = None
    segment: str | None = None
    cadence_key_area: str | None = None
    cadence_type: str | None = None
    pedal_degree: str | None = None
    confidence: float | None = None
    variant: str | None = None
    unmapped: bool = False


def map_label(label: dict, cfg: dict) -> MappedLabel:
    """Map one raw Dezrann label onto the canonical §11.2 vocabulary."""
    raw_type = label.get("type") or ""
    tag = label.get("tag")
    spec = cfg["label_types"].get(raw_type)

    if spec is None:
        # Unknown vocabulary is surfaced as `other`, flagged, and reported --
        # never silently dropped, and never guessed into a musical category.
        return MappedLabel(type="other", transformations=[], unmapped=True,
                           variant=(tag or raw_type or None))

    canonical = spec["type"]
    transformations = list(spec.get("transformations", []))
    status = spec.get("analytical_status", "accepted")

    # `Structure` is disambiguated by its tag: episode vs codetta.
    if raw_type == "Structure":
        key = (tag or "").strip().lower()
        canonical = cfg["structure_tags"].get(key, "other")

    # A rejected candidate keeps the musical family implied by its tag so the
    # near-miss stays interpretable (e.g. `S-inc-IGN` is a rejected *subject*).
    if status == "rejected_candidate":
        family = (tag or "").strip().split()[0] if tag else ""
        family = re.sub(r"-IGN[ab]?$", "", family)
        base = re.match(r"^(S|CS)(\d?)", family)
        if base:
            canonical = "subject" if base.group(1) == "S" else "countersubject"
            if base.group(2):
                if base.group(1) == "S":
                    spec = {**spec, "subject_index": int(base.group(2))}
                else:
                    spec = {**spec, "countersubject_index": int(base.group(2))}
        if "inc" in family:
            transformations.append("incomplete")
        if "inv" in family:
            transformations.append("inverted")
        if "var" in family:
            transformations.append("varied")
        if "ext" in family:
            transformations.append("extended")
        if "tail" in family:
            transformations.append("tail")
        if "in-episode" in family:
            transformations.append("in_episode")

    # Tag qualifiers layer additional transformations on top of the type map.
    tag_map = cfg.get("tag_transformations", {})
    for word in _tag_words(tag):
        extra = tag_map.get(word)
        if extra is not None:
            transformations.extend(_as_list(extra))

    out = MappedLabel(
        type=canonical,
        transformations=sorted(dict.fromkeys(transformations)),
        analytical_status=status,
        subject_index=spec.get("subject_index"),
        countersubject_index=spec.get("countersubject_index"),
        segment=spec.get("segment"),
        variant=(tag.strip() if isinstance(tag, str) and tag.strip() else None),
    )

    if canonical == "cadence":
        m = _CADENCE_RE.match(tag or "")
        if m:
            out.cadence_key_area = m.group(1)
            out.cadence_type = m.group(2)
            if m.group(3) or spec.get("uncertain"):
                out.confidence = 0.5
        elif spec.get("uncertain"):
            out.confidence = 0.5
    elif canonical == "pedal":
        deg = (tag or "").strip()
        # Tags are roman-numeral degrees (" I", " V/vi") except one prose label.
        out.pedal_degree = "I" if deg.lower() == "tonic pedal" else (deg or None)

    return out
