"""Read and write mapping predictions as SSSOM."""

from __future__ import annotations

import importlib.metadata as md
import os
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path

import bioregistry
import curies
import sssom_pydantic
from pydantic import AnyUrl
from sssom_pydantic import MappingSet, MappingTool, SemanticMapping

from mapnet.manifest import MAPPING_SET_BASE
from mapnet.utils import table, to_curie

MAPNET = MappingTool(name="mapnet", version=md.version("mapnet"))


def read(path: Path) -> list[SemanticMapping]:
    """Read every mapping from an SSSOM file."""
    mappings, _, _, errors = sssom_pydantic.read(path, return_errors=True)
    if errors:
        first = errors[0]
        raise ValueError(
            f"{path.name}: {len(errors)} unreadable rows, "
            f"line {first.line_number}: {first.exception}"
        )
    return list(mappings)


def stem(path: Path) -> str:
    """Take a file's name without the suffixes an SSSOM table carries."""
    return path.name.removesuffix(".tsv").removesuffix(".sssom")


def to_pairs(paths: Iterable[Path]) -> set[tuple[str, str]]:
    """Read the mappings SSSOM rows and OBO xrefs declare, each held both ways round."""
    return {
        held
        for path in paths
        for pair in (_xrefs(path) if path.suffix == ".obo" else _rows(path))
        for held in (pair, pair[::-1])
    }


def _rows(path: Path) -> Iterator[tuple[str, str]]:
    """Yield the normalized subject and object of every row in an SSSOM table."""
    for row in table(path):
        try:
            yield to_curie(row["subject_id"]), to_curie(row["object_id"])
        except ValueError:
            continue


def _xrefs(path: Path) -> Iterator[tuple[str, str]]:
    """Yield the cross references an OBO file declares on its own terms."""
    subject = ""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("id: "):
                subject = line[4:].strip()
            elif line.startswith("xref: ") and subject:
                try:
                    yield to_curie(subject), to_curie(line[6:].split()[0])
                except ValueError:
                    continue


def write(
    mappings: Iterable[SemanticMapping],
    out: Path,
    tool: MappingTool = MAPNET,
    source_version: str | None = None,
    target_version: str | None = None,
    mapping_set_id: str | None = None,
) -> int:
    """Write mappings as SSSOM at `out` and return the number written."""
    today = date.today()
    rows = [
        row.model_copy(
            update={
                "mapping_tool": row.mapping_tool or tool,
                "mapping_date": row.mapping_date or today,
                "subject_source_version": row.subject_source_version or source_version,
                "object_source_version": row.object_source_version or target_version,
            }
        )
        for row in mappings
    ]
    title = stem(out)
    set_id = AnyUrl(mapping_set_id or f"{MAPPING_SET_BASE}/{title}")
    metadata = MappingSet(id=set_id, title=title)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp")
    try:
        sssom_pydantic.write(rows, tmp, converter=_converter(rows), metadata=metadata)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    return len(rows)


def _converter(rows: list[SemanticMapping]) -> curies.Converter:
    """Build a curie map covering the prefixes the rows actually use."""
    prefixes = set()
    for row in rows:
        prefixes.update({row.subject.prefix, row.object.prefix})
        prefixes.update({row.predicate.prefix, row.justification.prefix})
        if row.mapping_tool and row.mapping_tool.reference:
            prefixes.add(row.mapping_tool.reference.prefix)
    prefix_map: dict[str, str] = {}
    unknown = []
    for prefix in prefixes:
        uri = bioregistry.get_uri_prefix(prefix)
        if uri is None:
            unknown.append(prefix)
        else:
            prefix_map[str(prefix)] = uri
    if unknown:
        raise ValueError(f"bioregistry cannot resolve prefixes {sorted(unknown)}")
    return curies.Converter.from_prefix_map(prefix_map)
