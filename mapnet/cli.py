"""Command line commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mapnet import __version__, get_evidence, get_ontology, get_version, list_versions
from mapnet.classify import BUCKETS, aggregate, classify, load_evidence
from mapnet.data import DATA_ROOT
from mapnet.matchers import load_tools, run
from mapnet.sssom import read, write
from mapnet.utils import LOG_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    """Run the mapnet command line."""
    parser = argparse.ArgumentParser(prog="mapnet")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_fetch(commands)
    _add_versions(commands)
    _add_tools(commands)
    _add_map(commands)
    _add_aggregate(commands)
    _add_classify(commands)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _add_fetch(commands: argparse._SubParsersAction) -> None:
    """Register the fetch command."""
    fetch = commands.add_parser("fetch", help="download an ontology")
    fetch.add_argument("source", help="a prefix such as mondo, or a download URL")
    fetch.add_argument("--format", default="obo", choices=sorted(("obo", "owl")))
    fetch.add_argument("--version", help="an OBO Foundry release, such as 2024-01-31")
    fetch.add_argument("--redownload", action="store_true")
    fetch.add_argument("--data", type=Path, default=DATA_ROOT)
    fetch.set_defaults(run=_fetch)


def _add_versions(commands: argparse._SubParsersAction) -> None:
    """Register the versions command."""
    versions = commands.add_parser("versions", help="list an ontology's releases")
    versions.add_argument("prefix")
    versions.add_argument("--refresh", action="store_true", help="re-query the source")
    versions.add_argument("--data", type=Path, default=DATA_ROOT)
    versions.set_defaults(run=_versions)


def _versions(args: argparse.Namespace) -> int:
    """Print every release published for an ontology."""
    for version in list_versions(args.prefix, refresh=args.refresh, root=args.data):
        print(version)
    return 0


def _add_tools(commands: argparse._SubParsersAction) -> None:
    """Register the tools command."""
    tools = commands.add_parser("tools", help="list registered matchers")
    tools.set_defaults(run=_tools)


def _add_map(commands: argparse._SubParsersAction) -> None:
    """Register the map command."""
    mapping = commands.add_parser("map", help="run a matcher over two ontologies")
    mapping.add_argument("--tool", required=True)
    mapping.add_argument("--src", required=True, help="source prefix or URL")
    mapping.add_argument("--tgt", required=True, help="target prefix or URL")
    mapping.add_argument("--out", type=Path, required=True)
    mapping.add_argument("--data", type=Path, default=DATA_ROOT)
    mapping.add_argument("--logs", type=Path, default=LOG_ROOT)
    mapping.add_argument("--src-prefix", help="override the source ontology prefix")
    mapping.add_argument("--tgt-prefix", help="override the target ontology prefix")
    mapping.set_defaults(run=_map)


def _add_aggregate(commands: argparse._SubParsersAction) -> None:
    """Register the aggregate command."""
    combine = commands.add_parser("aggregate", help="combine prediction files into one")
    combine.add_argument("predictions", type=Path, nargs="+")
    combine.add_argument("--out", type=Path, required=True)
    combine.set_defaults(run=_aggregate)


def _add_classify(commands: argparse._SubParsersAction) -> None:
    """Register the classify command."""
    split = commands.add_parser("classify", help="split predictions against evidence")
    split.add_argument("predictions", type=Path)
    split.add_argument(
        "--evidence", required=True, help="comma separated evidence names"
    )
    split.add_argument("--out", type=Path, required=True, help="directory for the sets")
    split.add_argument("--data", type=Path, default=DATA_ROOT)
    split.set_defaults(run=_classify)


def _aggregate(args: argparse.Namespace) -> int:
    """Combine several prediction files into one mapping set."""
    count = aggregate(args.predictions, args.out)
    print(f"{args.out}  ({count} mappings from {len(args.predictions)} files)")
    return 0


def _classify(args: argparse.Namespace) -> int:
    """Split one prediction file into right, wrong, novel and conflicts."""
    names = [name.strip() for name in args.evidence.split(",") if name.strip()]
    evidence = load_evidence([get_evidence(name, root=args.data) for name in names])
    print(
        f"evidence: {len(evidence.pairs) // 2} pairs, {len(evidence.prefixes)} entities"
    )
    buckets = classify(read(args.predictions), evidence)
    stem = args.predictions.name.removesuffix(".tsv").removesuffix(".sssom")
    for bucket in BUCKETS:
        rows = buckets[bucket]
        out = args.out / f"{stem}_{bucket}.sssom.tsv"
        write(rows, out, tool="mapnet", version=__version__)
        print(f"  {bucket:10} {len(rows):6}  {out}")
    return 0


def _tools(_: argparse.Namespace) -> int:
    """Print every registered matcher."""
    for name, tool in sorted(load_tools().items()):
        print(f"{name:12} {tool.wants_format:4} {' '.join(tool.command)}")
    return 0


def _map(args: argparse.Namespace) -> int:
    """Fetch both ontologies, run the tool, and report the predictions."""
    tools = load_tools()
    tool = tools.get(args.tool)
    if tool is None:
        raise ValueError(f"unknown tool {args.tool!r}, have {sorted(tools)}")
    source = get_ontology(args.src, fmt=tool.wants_format, root=args.data)
    target = get_ontology(args.tgt, fmt=tool.wants_format, root=args.data)
    out = run(
        tool,
        source,
        target,
        args.out,
        logs=args.logs,
        src_prefix=args.src_prefix,
        tgt_prefix=args.tgt_prefix,
    )
    with out.open(encoding="utf-8") as handle:
        rows = sum(1 for line in handle if not line.startswith("#")) - 1
    print(f"{out}  ({rows} mappings)")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    """Download one ontology and report where it landed."""
    path = get_ontology(
        args.source,
        fmt=args.format,
        version=args.version,
        redownload=args.redownload,
        root=args.data,
    )
    size = path.stat().st_size / 1e6
    print(f"{path}  ({size:.1f} MB, version {get_version(path) or 'unknown'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
