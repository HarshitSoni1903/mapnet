"""Tool registry and isolated run orchestration."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sssom_pydantic import SemanticMapping

from mapnet.classify import Evidence, Split, classify
from mapnet.data import Dataset, MapNet
from mapnet.eval import Scores, evaluate
from mapnet.manifest import EVIDENCE, RAW, RUN_STAMP, TOOLS
from mapnet.sssom import prefixes, read, stem, to_pairs, union, write
from mapnet.utils import run_log, to_prefix

ADAPTERS = Path(__file__).parent.parent / "adapters"


@dataclass(frozen=True)
class Tool:
    """One registered matcher and how to run it."""

    name: str
    command: list[str]
    wants_format: str
    config: Path | None


@dataclass
class Config:
    tool: str = "gilda"
    reverse: bool = False
    extra: Sequence[str] = field(default_factory=list)


@dataclass
class Result:
    dataset: Dataset
    directory: Path
    raw: Path
    reverse: Path | None = None
    _rows: list[SemanticMapping] | None = field(default=None, repr=False, compare=False)

    @classmethod
    def load(
        cls,
        raw: Path,
        mapnet: MapNet | None = None,
        evidence: Iterable[str] = EVIDENCE,
        gold: str | None = None,
        reverse: Path | None = None,
        out: Path | None = None,
    ) -> Result:
        """Take an existing results file as a run, naming its pair from its own rows."""
        rows = read(raw)
        seen = prefixes(rows)
        if not seen:
            raise ValueError(f"{raw} carries no prefixed ids")
        dataset = Dataset(
            src=seen[0],
            tgt=seen[1] if len(seen) > 1 else seen[0],
            gold=gold,
            evidence=list(evidence),
            mapnet=mapnet or MapNet(),
        )
        return cls(
            dataset=dataset,
            directory=out or raw.parent,
            raw=raw,
            reverse=reverse,
            _rows=rows,
        )

    def rows(self) -> list[SemanticMapping]:
        """Read the raw mappings this run wrote, once."""
        if self._rows is None:
            self._rows = read(self.raw)
        return self._rows

    def classify(self) -> Split:
        """Read the raw mappings, split them, and write the four sets."""
        rows = self.rows()
        used = prefixes(rows)
        evidence = Evidence.load(
            names=self.dataset.evidence,
            prefixes=used,
            mapnet=self.dataset.mapnet,
        )
        split = classify(
            rows=rows,
            evidence=evidence,
            prefixes=used,
            reverse=read(self.reverse) if self.reverse else [],
        )
        for name, found in split.sets():
            write(mappings=found, out=self.directory / f"{name}.sssom.tsv")
        return split

    def evaluate(self) -> Scores:
        """Read the raw mappings, score them against the gold, and write the metrics."""
        if not self.dataset.gold:
            raise ValueError("scoring needs a gold standard, pass gold= to Dataset")
        gold = to_pairs([self.dataset.mapnet.fetch(name=self.dataset.gold)])
        scores = evaluate(rows=self.rows(), gold=gold)
        scores.write(path=self.directory / f"{stem(self.raw)}.eval.json")
        return scores


def load_tools() -> dict[str, Tool]:
    """Read every tool the manifest registers."""
    return {name: _tool(name, entry) for name, entry in TOOLS.items()}


def match(dataset: Dataset, config: Config | None = None) -> Result:
    """Run one tool over both ontologies and return where it wrote."""
    config = config or Config()
    tools = load_tools()
    tool = tools.get(config.tool)
    if tool is None:
        raise ValueError(f"unknown tool {config.tool!r}, have {sorted(tools)}")
    source, target = dataset.ontologies(fmt=tool.wants_format)
    stamp = datetime.now().strftime(RUN_STAMP)
    pair = f"{to_prefix(source)}_{to_prefix(target)}"
    folder = run_folder(dataset=dataset, tool=config.tool, pair=pair, stamp=stamp)
    launch(
        tool=tool,
        source=source,
        target=target,
        out=folder / RAW,
        stamp=stamp,
        mapnet=dataset.mapnet,
        extra=config.extra,
    )
    back = None
    if config.reverse:
        back = folder / f"reverse_{RAW}"
        launch(
            tool=tool,
            source=target,
            target=source,
            out=back,
            stamp=stamp,
            mapnet=dataset.mapnet,
            extra=config.extra,
        )
    return Result(dataset=dataset, directory=folder, raw=folder / RAW, reverse=back)


def aggregate(results: Sequence[Result], out: Path | None = None) -> Result:
    """Combine several runs into one set, the first row for each pair winning."""
    if not results:
        raise ValueError("aggregating needs at least one result")
    rows = union(result.raw for result in results)
    dataset = results[0].dataset
    pair = "_".join(prefixes(rows))
    folder = out or run_folder(dataset=dataset, tool="aggregate", pair=pair)
    raw = folder / RAW
    write(mappings=rows, out=raw)
    return Result(dataset=dataset, directory=folder, raw=raw)


def run_folder(
    dataset: Dataset, tool: str, pair: str, stamp: str | None = None
) -> Path:
    """The directory one run's files land in."""
    stamp = stamp or datetime.now().strftime(RUN_STAMP)
    return dataset.mapnet.outputs / tool / pair / stamp


def launch(
    tool: Tool,
    source: Path,
    target: Path,
    out: Path,
    stamp: str,
    mapnet: MapNet | None = None,
    extra: Sequence[str] = (),
) -> Path:
    """Run a tool over two ontologies and return the predictions it wrote."""
    space = mapnet or MapNet()
    log = run_log(
        tool=tool.name, source=source, target=target, stamp=stamp, root=space.logs
    )
    command = [*tool.command, "--source", str(source), "--target", str(target)]
    command += ["--out", str(out), "--logs", str(space.logs)]
    command += ["--workdir", str(space.workdir)]
    if tool.config:
        command += ["--config", str(tool.config)]
    command += extra
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, text=True
        )
    if result.returncode != 0:
        raise RuntimeError(f"{tool.name} failed, see {log}: {_tail(log)}")
    if not out.is_file():
        raise RuntimeError(f"{tool.name} wrote no predictions at {out}, see {log}")
    return out


def _tail(log: Path) -> str:
    """Read the last non-empty line of a log."""
    lines = [line.strip() for line in log.read_text(encoding="utf-8").splitlines()]
    return next((line for line in reversed(lines) if line), "no output")


def _tool(name: str, entry: dict) -> Tool:
    """Build one registry entry, resolving its paths against the adapters folder."""
    if "command" not in entry or "wants_format" not in entry:
        raise ValueError(f"{name!r} needs both command and wants_format")
    config = entry.get("config")
    return Tool(
        name=name,
        command=[_resolve(part) for part in entry["command"]],
        wants_format=entry["wants_format"],
        config=ADAPTERS / config if config else None,
    )


def _resolve(part: str) -> str:
    """Make a command part absolute when it names a file in the adapters folder."""
    candidate = ADAPTERS / part
    return str(candidate) if candidate.is_file() else part
