"""The base class an adapter subclasses."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

from sssom_pydantic import SemanticMapping

from mapnet.data import get_version
from mapnet.sssom import write


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
            tool=cls.name,
            version=cls.version,
            tool_id=cls.tool_id,
            source_version=get_version(args.source),
            target_version=get_version(args.target),
        )
        return 0


def parse_args(prog: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the arguments MapNet passes to every adapter."""
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--config", type=Path)
    return parser.parse_args(argv)
