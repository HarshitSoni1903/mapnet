"""The base class an adapter subclasses."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

from curies import Reference
from sssom_pydantic import MappingTool, SemanticMapping

from mapnet.data import get_version
from mapnet.sssom import write
from mapnet.utils import LOG_ROOT, to_prefix


class Mapper(ABC):
    """Base class for adapters."""

    name: ClassVar[str]
    version: ClassVar[str]
    tool_id: ClassVar[str] = ""

    @abstractmethod
    def match(self, args: argparse.Namespace) -> Iterable[SemanticMapping]:
        """Yield predicted mappings for the requested run."""

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        """Run the adapter from the command line."""
        args = parse_args(cls.name, argv)
        write(
            cls().match(args),
            args.out,
            tool=cls.identity(),
            source_version=get_version(args.source),
            target_version=get_version(args.target),
        )
        return 0

    @staticmethod
    def prefixes(args: argparse.Namespace) -> tuple[str, str]:
        """Take the source and target prefixes, preferring the flags over the files."""
        return (
            args.src_prefix or to_prefix(args.source),
            args.tgt_prefix or to_prefix(args.target),
        )

    @classmethod
    def identity(cls) -> MappingTool:
        """Build the tool identity stamped onto every row the adapter writes."""
        reference = Reference.from_curie(cls.tool_id) if cls.tool_id else None
        return MappingTool(name=cls.name, version=cls.version, reference=reference)


def parse_args(prog: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the arguments MapNet passes to every adapter."""
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--logs", type=Path, default=LOG_ROOT)
    parser.add_argument("--src-prefix", help="override the source ontology prefix")
    parser.add_argument("--tgt-prefix", help="override the target ontology prefix")
    parser.add_argument("--config", type=Path)
    return parser.parse_args(argv)
