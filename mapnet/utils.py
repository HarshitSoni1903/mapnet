"""Shared helpers used across every other module."""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from functools import cache
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse

import bioregistry
import curies
from curies import NamableReference

LOG_ROOT = Path("logs")

HEADER_LINES = 200

TABLES = {".tsv": "\t", ".csv": ","}

ONTOLOGY_LINE = re.compile(r"^ontology:\s*(\S+)", re.MULTILINE)

ONTOLOGY_IRI = re.compile(r'<owl:Ontology rdf:about="([^"]+)"')


def run_log(tool: str, source: Path, target: Path, root: Path = LOG_ROOT) -> Path:
    """Build the log path for one tool run, creating its directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"run_{tool}_{source.stem}_{target.stem}_{stamp}.log"


@cache
def converter() -> curies.Converter:
    """Build the bioregistry converter, once per process."""
    return bioregistry.get_converter()


def to_prefix(path: Path) -> str:
    """Read the ontology prefix the file declares in its header or its id column."""
    if path.suffix in TABLES:
        return _table_prefix(path)
    with path.open(encoding="utf-8", errors="replace") as handle:
        head = "".join(islice(handle, HEADER_LINES))
    declared = ONTOLOGY_LINE.search(head)
    if declared:
        return declared.group(1).strip().lower()
    iri = ONTOLOGY_IRI.search(head)
    if iri:
        return Path(urlparse(iri.group(1)).path).name.split(".")[0].lower()
    raise ValueError(f"{path.name} declares no ontology prefix in its header")


def _table_prefix(path: Path) -> str:
    """Take the dominant prefix from a concept table's id column."""
    seen: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        rows = csv.DictReader(handle, delimiter=TABLES[path.suffix])
        if not rows.fieldnames or "id" not in rows.fieldnames:
            raise ValueError(f"{path.name} has no id column to read a prefix from")
        for row in islice(rows, HEADER_LINES):
            value = (row["id"] or "").strip()
            if ":" in value:
                seen[value.split(":")[0].lower()] += 1
    if not seen:
        raise ValueError(f"{path.name} has no prefixed ids to read a prefix from")
    return seen.most_common(1)[0][0]


def check_prefixes(path: Path, prefix: str, nodes: Iterable[str]) -> None:
    """Warn when the file carries prefixes other than the one being mapped."""
    seen: Counter[str] = Counter()
    for node in nodes:
        name = _namespace(node)
        if name:
            seen[name] += 1
    others = Counter({name: n for name, n in seen.items() if name != prefix})
    if not others:
        return
    listed = ", ".join(f"{name} ({n})" for name, n in others.most_common(5))
    print(
        f"[prefix] {path.name}: mapping {prefix} ({seen[prefix]} terms), "
        f"ignoring {listed}. Pass --src-prefix or --tgt-prefix to choose another.",
        file=sys.stderr,
    )


def _namespace(node: str) -> str | None:
    """Name the prefix a node id belongs to."""
    head, sep, _ = node.partition(":")
    if sep and not node.startswith(("http://", "https://")):
        return _normalize(head)
    try:
        return to_curie(node).split(":")[0]
    except ValueError:
        return None


@cache
def _normalize(head: str) -> str | None:
    """Normalize one CURIE prefix, cached since a file reuses only a handful."""
    return bioregistry.normalize_prefix(head)


def to_curie(value: str) -> str:
    """Turn an IRI or a CURIE into a normalized CURIE."""
    text = value.strip()
    if text.startswith(("http://", "https://")):
        curie = converter().compress(text)
    else:
        curie = bioregistry.normalize_curie(text)
    if curie is None:
        raise ValueError(f"cannot normalize {value!r} to a known prefix")
    return curie


def to_reference(value: str, name: str | None = None) -> NamableReference:
    """Turn an IRI or a CURIE into a normalized reference."""
    return NamableReference.from_curie(to_curie(value), name=name)
