"""Shared helpers used across every other module."""

from __future__ import annotations

from datetime import datetime
from functools import cache
from pathlib import Path

import bioregistry
import curies
from curies import NamableReference

LOG_ROOT = Path("logs")


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
    """Read the ontology prefix from a data filename."""
    return path.stem.split("_")[0].lower()


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
