"""Command line commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mapnet.classify import BUCKETS, aggregate, classify
from mapnet.data import DATA_ROOT, downloads, get_source, get_version, list_versions
from mapnet.manifest import EVIDENCE, REFRESH, SOURCES
from mapnet.matchers import load_tools, run
from mapnet.sssom import read, stem, write
from mapnet.utils import LOG_ROOT, table, to_prefix

_Commands = argparse._SubParsersAction


def _parser() -> argparse.ArgumentParser:
    """Build the parser by letting every command declare itself."""
    parser = argparse.ArgumentParser(prog="mapnet")
    commands = parser.add_subparsers(dest="command", required=True)
    shared, judged = _shared()
    _add_fetch(commands, shared)
    _add_versions(commands, shared)
    _add_tools(commands)
    _add_evidence(commands)
    _add_map(commands, shared, judged)
    _add_aggregate(commands)
    _add_classify(commands, shared, judged)
    return parser


def _shared() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Build the option groups more than one command takes."""
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--data", type=Path, default=DATA_ROOT, help="download root")
    evidence = argparse.ArgumentParser(add_help=False)
    evidence.add_argument(
        "--evidence",
        default=",".join(EVIDENCE),
        help="comma separated evidence names or file paths, each optionally "
        "tagged rejected: or predicted:",
    )
    return data, evidence


def _add_fetch(commands: _Commands, shared: argparse.ArgumentParser) -> None:
    """Declare the command that downloads an ontology or an evidence set."""
    fetch = commands.add_parser(
        "fetch", parents=[shared], help="download an ontology or an evidence set"
    )
    fetch.add_argument(
        "source",
        nargs="?",
        help="a prefix, an evidence name, or a URL; omit to refresh every "
        f"volatile source ({', '.join(REFRESH)})",
    )
    fetch.add_argument("--format", default="obo", choices=sorted(("obo", "owl")))
    fetch.add_argument("--version", help="an OBO Foundry release, such as 2024-01-31")
    fetch.add_argument("--redownload", action="store_true")
    fetch.set_defaults(run=_fetch)


def _add_versions(commands: _Commands, shared: argparse.ArgumentParser) -> None:
    """Declare the command that lists an ontology's releases."""
    versions = commands.add_parser("versions", parents=[shared], help="list releases")
    versions.add_argument("prefix")
    versions.add_argument("--refresh", action="store_true", help="re-query the source")
    versions.set_defaults(run=_versions)


def _add_tools(commands: _Commands) -> None:
    """Declare the command that lists registered matchers."""
    commands.add_parser("tools", help="list matchers").set_defaults(run=_tools)


def _add_evidence(commands: _Commands) -> None:
    """Declare the command that lists registered evidence sets."""
    commands.add_parser("evidence", help="list evidence sets").set_defaults(
        run=_evidence
    )


def _add_map(
    commands: _Commands,
    shared: argparse.ArgumentParser,
    judged: argparse.ArgumentParser,
) -> None:
    """Declare the command that runs a matcher over two ontologies."""
    mapping = commands.add_parser(
        "map",
        parents=[shared, judged],
        help="run a matcher",
        epilog="Any other flag is passed to the tool, which validates it.",
    )
    mapping.add_argument("--tool", required=True)
    mapping.add_argument(
        "--classify", action="store_true", help="split the predictions once written"
    )
    mapping.add_argument(
        "--reverse", action="store_true", help="also run with the sides swapped"
    )
    mapping.add_argument("--src", required=True, help="source prefix or URL")
    mapping.add_argument("--tgt", required=True, help="target prefix or URL")
    mapping.add_argument(
        "--out", type=Path, required=True, help="folder to write this run's files in"
    )
    mapping.add_argument("--logs", type=Path, default=LOG_ROOT)
    mapping.set_defaults(run=_map)


def _add_aggregate(commands: _Commands) -> None:
    """Declare the command that unions several prediction files."""
    combine = commands.add_parser("aggregate", help="combine prediction files into one")
    combine.add_argument("predictions", type=Path, nargs="+")
    combine.add_argument("--out", type=Path, required=True)
    combine.set_defaults(run=_aggregate)


def _add_classify(
    commands: _Commands,
    shared: argparse.ArgumentParser,
    judged: argparse.ArgumentParser,
) -> None:
    """Declare the command that splits predictions against evidence."""
    split = commands.add_parser(
        "classify", parents=[shared, judged], help="split predictions"
    )
    split.add_argument("predictions", type=Path)
    split.add_argument("--out", type=Path, required=True, help="directory for the sets")
    split.add_argument(
        "--reverse", type=Path, help="predictions from the run with the sides swapped"
    )
    split.set_defaults(run=_classify)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command asked for, reporting a failure as one line."""
    parser = _parser()
    args, extra = parser.parse_known_args(argv)
    if extra and args.command != "map":
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.extra = extra
    try:
        return args.run(args)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _versions(args: argparse.Namespace) -> int:
    """Print every release published for an ontology."""
    for version in list_versions(args.prefix, refresh=args.refresh, root=args.data):
        print(version)
    return 0


def _evidence(_: argparse.Namespace) -> int:
    """Print every registered evidence set, marking what is consulted and refreshed."""
    for name, (kind, source) in sorted(SOURCES.items()):
        marks = ("*" if name in EVIDENCE else " ") + ("^" if name in REFRESH else " ")
        print(f"{marks} {name:24} {kind:10} {source}")
    print("\n* consulted by default, override with --evidence")
    print("^ refetched by a bare `mapnet fetch`")
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    """Combine several prediction files into one mapping set."""
    count = aggregate(args.predictions, args.out)
    print(f"{args.out}  ({count} mappings from {len(args.predictions)} files)")
    return 0


def _classify(args: argparse.Namespace) -> int:
    """Split one prediction file into right, wrong, novel and conflicts."""
    _split(args.predictions, args.evidence, args.out, args.data, args.reverse)
    return 0


def _split(
    predictions: Path, evidence: str, out: Path, data: Path, reverse: Path | None = None
) -> None:
    """Classify one prediction file, write the four sets, and report what happened."""
    rows = read(predictions)
    back = read(reverse) if reverse else []
    names = [name.strip() for name in evidence.split(",") if name.strip()]
    split = classify(rows, names, root=data, reverse=back)
    print(f"candidates  {len(rows)} over {', '.join(split.prefixes)}")
    if reverse:
        print(f"reverse     {len(back)} pairs from {reverse}")
    for kind, entries in split.evidence.sources.items():
        for path, count in entries:
            print(f"{kind:11} {count:9} pairs  {path}")
    held = len(split.buckets["conflicts"])
    kept = len(split.buckets["novel"])
    print(
        f"rescued     {split.rescued} right on an uncurated prediction alone\n"
        f"reduced     {kept + held} novel candidates -> "
        f"{kept} one to one, {held} held as conflicts"
    )
    for bucket in BUCKETS:
        written = split.buckets[bucket]
        path = out / f"{stem(predictions)}_{bucket}.sssom.tsv"
        write(written, path)
        share = 100 * len(written) / len(rows) if rows else 0.0
        print(f"  {bucket:10} {len(written):6} ({share:4.1f}%)  {path}")


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
    source = get_source(args.src, fmt=tool.wants_format, root=args.data)
    target = get_source(args.tgt, fmt=tool.wants_format, root=args.data)
    src, tgt = to_prefix(source), to_prefix(target)
    out = args.out / f"{tool.name}_{src}_{tgt}.sssom.tsv"
    run(tool, source, target, out, logs=args.logs, extra=args.extra)
    print(f"{out}  ({_count(out)} mappings)")
    back = None
    if args.reverse:
        back = args.out / f"{tool.name}_{tgt}_{src}.sssom.tsv"
        run(tool, target, source, back, logs=args.logs, extra=args.extra)
        print(f"{back}  ({_count(back)} mappings)")
    if args.classify:
        _split(out, args.evidence, args.out, args.data, back)
    return 0


def _count(path: Path) -> int:
    """Count the mapping rows an SSSOM file holds."""
    return sum(1 for _ in table(path))


def _fetch(args: argparse.Namespace) -> int:
    """Download one source, or refresh every volatile one, and report what changed."""
    if args.source is None and args.version:
        raise ValueError("--version names one release, so it needs a source")
    names = [args.source] if args.source else REFRESH
    before = downloads(args.data)
    for name in names:
        path = get_source(
            name,
            fmt=args.format,
            version=args.version,
            redownload=args.redownload or args.source is None,
            root=args.data,
        )
        _report(name, path, before, args.data)
    return 0


def _report(name: str, path: Path, before: dict, root: Path) -> None:
    """Print where a fetched file landed, its version, and whether it changed."""
    was, now = before.get(str(path), {}), downloads(root).get(str(path), {})
    if not now:
        state = "cached"
    elif not was:
        state = "new"
    else:
        state = "unchanged" if was.get("sha256") == now.get("sha256") else "changed"
    version = path.parent.name if name in SOURCES else get_version(path) or "unknown"
    size = path.stat().st_size / 1e6
    print(f"{name:24} {state:10} {size:7.1f} MB  version {version}  {path}")


if __name__ == "__main__":
    raise SystemExit(main())
