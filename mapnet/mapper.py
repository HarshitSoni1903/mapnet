"""The contract every adapter subclasses."""

from __future__ import annotations

import argparse
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

import sssom_pydantic
from curies import Reference
from sssom_pydantic import MappingTool, SemanticMapping


class Mapper(ABC):
    """Base class for adapters."""

    name: ClassVar[str]
    version: ClassVar[str]
    tool_id: ClassVar[str] = ""

    @abstractmethod
    def match(
        self, source: Path, target: Path, config: Path | None
    ) -> Iterable[SemanticMapping]:
        """Yield predicted mappings between two ontology files."""

    @classmethod
    def tool(cls) -> MappingTool:
        """Build the tool identity stamped onto every row."""
        reference = Reference.from_curie(cls.tool_id) if cls.tool_id else None
        return MappingTool(name=cls.name, version=cls.version, reference=reference)

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        """Run the adapter from the command line."""
        args = _parse_args(cls.name, argv)
        tool = cls.tool()
        mappings = (
            mapping.model_copy(update={"mapping_tool": tool})
            for mapping in cls().match(args.source, args.target, args.config)
        )
        write(mappings, args.out)
        return 0


def write(mappings: Iterable[SemanticMapping], out: Path) -> None:
    """Write mappings as SSSOM, appearing at `out` only on success."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp")
    sssom_pydantic.write(mappings, tmp)
    os.replace(tmp, out)


def _parse_args(prog: str, argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the four standard adapter arguments."""
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    return parser.parse_args(argv)
