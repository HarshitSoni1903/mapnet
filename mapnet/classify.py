"""Combine and classify predictions against curated evidence."""

from __future__ import annotations

import importlib.metadata as md
from collections.abc import Sequence
from pathlib import Path

from sssom_pydantic import SemanticMapping

from mapnet.sssom import read, write


def aggregate(paths: Sequence[Path], out: Path) -> int:
    """Write one mapping set from several prediction files, first pair winning."""
    return write(union(paths), out, tool="mapnet", version=md.version("mapnet"))


def union(paths: Sequence[Path]) -> list[SemanticMapping]:
    """Read every prediction file in order, keeping the first row for each pair."""
    seen: set[tuple[str, str]] = set()
    rows = []
    for path in paths:
        for row in read(path):
            key = (row.subject.curie, row.object.curie)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows
