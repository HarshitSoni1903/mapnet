"""The base class an adapter subclasses."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

from curies import Reference
from sssom_pydantic import MappingTool, SemanticMapping

from mapnet.data import Dataset, MapNet, get_version
from mapnet.manifest import RAW, TOOLS
from mapnet.matchers import Result, run_folder
from mapnet.sssom import write
from mapnet.utils import LOG_ROOT, to_prefix


class Mapper(ABC):
    """Base class for adapters, holding the dataset one run maps."""

    name: ClassVar[str]
    version: ClassVar[str]
    tool_id: ClassVar[str] = ""

    def __init__(self, dataset: Dataset, config: Path | None = None) -> None:
        self.dataset = dataset
        self.config = config

    @abstractmethod
    def match(self) -> Iterable[SemanticMapping]:
        """Yield predicted mappings for this run."""

    def run(self, out: Path | None = None) -> Result:
        """Match the pair and write the predictions as SSSOM."""
        source, target = self.ontologies()
        folder = (
            out.parent
            if out
            else run_folder(
                dataset=self.dataset,
                tool=self.name,
                pair=f"{to_prefix(source)}_{to_prefix(target)}",
            )
        )
        raw = out or folder / RAW
        write(
            mappings=self.match(),
            out=raw,
            tool=self.identity(),
            source_version=get_version(path=source),
            target_version=get_version(path=target),
        )
        return Result(dataset=self.dataset, directory=folder, raw=raw)

    def ontologies(self) -> tuple[Path, Path]:
        """Return local paths to both ontologies, downloading either when absent."""
        return self.dataset.ontologies(fmt=self.wants_format())

    def prefixes(self) -> tuple[str, str]:
        """Take the source and target prefixes from the two ontology files."""
        source, target = self.ontologies()
        return to_prefix(source), to_prefix(target)

    def work(self) -> Path:
        """Return this adapter's own directory in the workspace, created."""
        folder = (self.dataset.mapnet.workdir / self.name).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @classmethod
    def wants_format(cls) -> str:
        """Name the ontology format the manifest registers for this adapter."""
        return str(TOOLS.get(cls.name, {}).get("wants_format", "obo"))

    @classmethod
    def identity(cls) -> MappingTool:
        """Build the tool identity stamped onto every row the adapter writes."""
        reference = Reference.from_curie(cls.tool_id) if cls.tool_id else None
        return MappingTool(name=cls.name, version=cls.version, reference=reference)

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        """Run the adapter as the subprocess MapNet launches."""
        args = parse_args(prog=cls.name, argv=argv)
        dataset = Dataset(
            src=str(args.source),
            tgt=str(args.target),
            mapnet=MapNet(workdir=args.workdir),
        )
        cls(dataset=dataset, config=args.config).run(out=args.out)
        return 0


def parse_args(prog: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the arguments MapNet passes to every adapter."""
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--logs", type=Path, default=LOG_ROOT)
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path)
    return parser.parse_args(argv)
