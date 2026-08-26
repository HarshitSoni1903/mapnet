"""Read and write mapping predictions as SSSOM."""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import bioregistry
import curies
import sssom_pydantic
from pydantic import AnyUrl
from sssom_pydantic import MappingSet, MappingTool, SemanticMapping

from mapnet.manifest import MAPPING_SET_BASE


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


def write(
    mappings: Iterable[SemanticMapping],
    out: Path,
    tool: str,
    version: str,
    tool_id: str = "",
    source_version: str | None = None,
    target_version: str | None = None,
    mapping_set_id: str | None = None,
) -> int:
    """Write mappings as SSSOM at `out` and return the number written."""
    identity = _tool(tool, version, tool_id)
    today = date.today()
    rows = [
        row.model_copy(
            update={
                "mapping_tool": row.mapping_tool or identity,
                "mapping_date": row.mapping_date or today,
                "subject_source_version": row.subject_source_version or source_version,
                "object_source_version": row.object_source_version or target_version,
            }
        )
        for row in mappings
    ]
    stem = out.name.removesuffix(".tsv").removesuffix(".sssom")
    set_id = AnyUrl(mapping_set_id or f"{MAPPING_SET_BASE}/{stem}")
    metadata = MappingSet(id=set_id, title=stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp")
    try:
        sssom_pydantic.write(rows, tmp, converter=_converter(rows), metadata=metadata)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    return len(rows)


def _tool(name: str, version: str, tool_id: str) -> MappingTool:
    """Build the tool identity stamped onto every row."""
    reference = curies.Reference.from_curie(tool_id) if tool_id else None
    return MappingTool(name=name, version=version, reference=reference)


def _converter(rows: list[SemanticMapping]) -> curies.Converter:
    """Build a curie map covering the prefixes the rows actually use."""
    prefixes = set()
    for row in rows:
        prefixes.update({row.subject.prefix, row.object.prefix})
        prefixes.update({row.predicate.prefix, row.justification.prefix})
        if row.mapping_tool and row.mapping_tool.reference:
            prefixes.add(row.mapping_tool.reference.prefix)
    resolved = {prefix: bioregistry.get_uri_prefix(prefix) for prefix in prefixes}
    unknown = sorted(p for p, uri in resolved.items() if uri is None)
    if unknown:
        raise ValueError(f"bioregistry cannot resolve prefixes {unknown}")
    prefix_map = {str(p): uri for p, uri in resolved.items() if uri is not None}
    return curies.Converter.from_prefix_map(prefix_map)
