"""Command line commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mapnet.data import DATA_ROOT, get_ontology, get_version, list_versions
from mapnet.matchers import load_tools, run


def main(argv: Sequence[str] | None = None) -> int:
    """Run the mapnet command line."""
    parser = argparse.ArgumentParser(prog="mapnet")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_fetch(commands)
    _add_versions(commands)
    _add_tools(commands)
    _add_match(commands)
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


def _add_match(commands: argparse._SubParsersAction) -> None:
    """Register the match command."""
    match = commands.add_parser("match", help="run a matcher over two ontologies")
    match.add_argument("--tool", required=True)
    match.add_argument("--src", required=True, help="source prefix or URL")
    match.add_argument("--tgt", required=True, help="target prefix or URL")
    match.add_argument("--out", type=Path, required=True)
    match.add_argument("--data", type=Path, default=DATA_ROOT)
    match.set_defaults(run=_match)


def _tools(args: argparse.Namespace) -> int:
    """Print every registered matcher."""
    for name, tool in sorted(load_tools().items()):
        print(f"{name:12} {tool.wants_format:4} {' '.join(tool.command)}")
    return 0


def _match(args: argparse.Namespace) -> int:
    """Fetch both ontologies, run the tool, and report the predictions."""
    tools = load_tools()
    tool = tools.get(args.tool)
    if tool is None:
        raise ValueError(f"unknown tool {args.tool!r}, have {sorted(tools)}")
    source = get_ontology(args.src, fmt=tool.wants_format, root=args.data)
    target = get_ontology(args.tgt, fmt=tool.wants_format, root=args.data)
    out = run(tool, source, target, args.out)
    rows = sum(1 for line in out.open() if not line.startswith("#")) - 1
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
