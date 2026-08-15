"""Shared helpers used across every other module."""

from __future__ import annotations

from functools import cache

import bioregistry
import curies
from curies import NamableReference


@cache
def converter() -> curies.Converter:
    """Build the bioregistry converter, once per process."""
    return bioregistry.get_converter()


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
