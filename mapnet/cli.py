"""Command line commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mapnet import get_ontology, get_version, list_versions
from mapnet.classify import BUCKETS, aggregate, classify
from mapnet.data import DATA_ROOT
from mapnet.manifest import EVIDENCE, SOURCES
from mapnet.matchers import load_tools, run
from mapnet.sssom import read, stem, write
from mapnet.utils import LOG_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    """Register every command and run the one asked for."""
    parser = argparse.ArgumentParser(prog="mapnet")
    commands = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--data", type=Path, default=DATA_ROOT, help="download root")
    judged = argparse.ArgumentParser(add_help=False)
    judged.add_argument(
        "--evidence",
        default=",".join(EVIDENCE),
        help="comma separated evidence names or file paths",
    )

    fetch = commands.add_parser("fetch", parents=[shared], help="download an ontology")
    fetch.add_argument("source", help="a prefix such as mondo, or a download URL")
    fetch.add_argument("--format", default="obo", choices=sorted(("obo", "owl")))
    fetch.add_argument("--version", help="an OBO Foundry release, such as 2024-01-31")
    fetch.add_argument("--redownload", action="store_true")
    fetch.set_defaults(run=_fetch)

    versions = commands.add_parser("versions", parents=[shared], help="list releases")
    versions.add_argument("prefix")
    versions.add_argument("--refresh", action="store_true", help="re-query the source")
    versions.set_defaults(run=_versions)

    commands.add_parser("tools", help="list matchers").set_defaults(run=_tools)
    commands.add_parser("evidence", help="list evidence sets").set_defaults(
        run=_evidence
    )

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
    mapping.add_argument("--src", required=True, help="source prefix or URL")
    mapping.add_argument("--tgt", required=True, help="target prefix or URL")
    mapping.add_argument("--out", type=Path, required=True)
    mapping.add_argument("--logs", type=Path, default=LOG_ROOT)
    mapping.set_defaults(run=_map)

    combine = commands.add_parser("aggregate", help="combine prediction files into one")
    combine.add_argument("predictions", type=Path, nargs="+")
    combine.add_argument("--out", type=Path, required=True)
    combine.set_defaults(run=_aggregate)

    split = commands.add_parser(
        "classify", parents=[shared, judged], help="split predictions"
    )
    split.add_argument("predictions", type=Path)
    split.add_argument("--out", type=Path, required=True, help="directory for the sets")
    split.set_defaults(run=_classify)

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
    """Print every registered evidence set, marking the ones classify consults."""
    for name, (kind, source) in sorted(SOURCES.items()):
        mark = "*" if name in EVIDENCE else " "
        print(f"{mark} {name:24} {kind:10} {source}")
    print("\n* consulted by default, override with --evidence")
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    """Combine several prediction files into one mapping set."""
    count = aggregate(args.predictions, args.out)
    print(f"{args.out}  ({count} mappings from {len(args.predictions)} files)")
    return 0


def _classify(args: argparse.Namespace) -> int:
    """Split one prediction file into right, wrong, novel and conflicts."""
    _split(args.predictions, args.evidence, args.out, args.data)
    return 0


def _split(predictions: Path, evidence: str, out: Path, data: Path) -> None:
    """Classify one prediction file, write the four sets, and report what happened."""
    rows = read(predictions)
    names = [name.strip() for name in evidence.split(",") if name.strip()]
    split = classify(rows, names, root=data)
    print(f"candidates  {len(rows)} over {', '.join(split.prefixes)}")
    for kind, paths in split.evidence.sources.items():
        for path in paths:
            print(f"{kind:11} {path}")
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
    source = get_ontology(args.src, fmt=tool.wants_format, root=args.data)
    target = get_ontology(args.tgt, fmt=tool.wants_format, root=args.data)
    out = run(tool, source, target, args.out, logs=args.logs, extra=args.extra)
    with out.open(encoding="utf-8") as handle:
        rows = sum(1 for line in handle if not line.startswith("#")) - 1
    print(f"{out}  ({rows} mappings)")
    if args.classify:
        _split(out, args.evidence, out.parent, args.data)
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
