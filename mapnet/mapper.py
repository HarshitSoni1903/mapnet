"""The base class an adapter subclasses."""

from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

from curies import Reference
from sssom_pydantic import MappingTool, SemanticMapping

from mapnet.data import get_version
from mapnet.eval import score
from mapnet.sssom import read, stem, to_pairs, write
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
        """Run the adapter from the command line, scoring it when given a gold set."""
        args = parse_args(cls.name, argv)
        mapper = cls()
        write(
            mapper.match(args),
            args.out,
            tool=cls.identity(),
            source_version=get_version(args.source),
            target_version=get_version(args.target),
        )
        if args.gold:
            mapper.report(args)
        return 0

    def report(self, args: argparse.Namespace) -> Path:
        """Write the metrics this adapter can measure beside its predictions."""
        rows = read(args.out)
        metrics = self.evaluate(rows, to_pairs([args.gold]), args)
        out = args.out.with_name(f"{stem(args.out)}.eval.json")
        payload = {"tool": self.name, "version": self.version, "metrics": metrics}
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[eval] {out}  {metrics}")
        return out

    def evaluate(
        self,
        rows: Sequence[SemanticMapping],
        gold: set[tuple[str, str]],
        args: argparse.Namespace,
    ) -> dict[str, float]:
        """Score the pairs the adapter wrote."""
        pairs = [(row.subject.curie, row.object.curie) for row in rows]
        return score(pairs, gold).as_dict()

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
    parser.add_argument("--gold", type=Path, help="gold standard to score against")
    return parser.parse_args(argv)
