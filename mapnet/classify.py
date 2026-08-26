"""Combine and classify predictions against curated evidence."""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sssom_pydantic import SemanticMapping

from mapnet.data import get_evidence, get_ontology
from mapnet.manifest import EVIDENCE, SOURCES
from mapnet.sssom import read, to_pairs, write

BUCKETS = ("right", "wrong", "novel", "conflicts")

KINDS = ("pairs", "rejected", "predicted")


@dataclass(frozen=True)
class Evidence:
    """The pair sets a candidate is judged against, and the files behind them."""

    pairs: set[tuple[str, str]] = field(default_factory=set)
    rejected: set[tuple[str, str]] = field(default_factory=set)
    predicted: set[tuple[str, str]] = field(default_factory=set)
    prefixes: dict[str, set[str]] = field(default_factory=dict)
    sources: dict[str, list[Path]] = field(default_factory=dict)


@dataclass(frozen=True)
class Split:
    """The four sets, the evidence behind them, and what a prediction alone rescued."""

    buckets: dict[str, list[SemanticMapping]]
    evidence: Evidence
    prefixes: Sequence[str]
    rescued: int


def aggregate(paths: Sequence[Path], out: Path) -> int:
    """Write one mapping set from several prediction files, first pair winning."""
    return write(union(paths), out)


def union(paths: Sequence[Path]) -> list[SemanticMapping]:
    """Read every prediction file in order, keeping the first row for each pair."""
    seen: set[tuple[str, str]] = set()
    rows: list[SemanticMapping] = []
    for path in paths:
        for row in read(path):
            key = (row.subject.curie, row.object.curie)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def load_evidence(
    names: Iterable[str], prefixes: Iterable[str], root: Path | None = None
) -> Evidence:
    """Resolve evidence names to files and stream each kind into its own pair set."""
    files: dict[str, list[Path]] = {kind: [] for kind in KINDS}
    for name in [names] if isinstance(names, str) else names:
        kind, source = SOURCES.get(name, ("pairs", name))
        if kind not in files:
            raise ValueError(f"{name!r} has unknown kind {kind!r}, expected {KINDS}")
        if source == "obo":
            for prefix in prefixes:
                try:
                    files[kind].append(get_ontology(prefix, root=root))
                except ValueError as error:
                    print(f"[evidence] no xrefs for {prefix}: {error}", file=sys.stderr)
        elif Path(name).is_file():
            files[kind].append(Path(name))
        else:
            files[kind].append(get_evidence(name, root=root))
    pairs = to_pairs(files["pairs"])
    mapped: dict[str, set[str]] = defaultdict(set)
    for subject, obj in pairs:
        mapped[subject].add(obj.split(":")[0])
    return Evidence(
        pairs,
        to_pairs(files["rejected"]),
        to_pairs(files["predicted"]),
        dict(mapped),
        files,
    )


def classify(
    rows: Sequence[SemanticMapping],
    evidence: Iterable[str] | Evidence = EVIDENCE,
    root: Path | None = None,
) -> Split:
    """Split candidates against evidence, loading it first when given names."""
    prefixes = sorted(
        {p for row in rows for p in (row.subject.prefix, row.object.prefix)}
    )
    if not isinstance(evidence, Evidence):
        evidence = load_evidence(evidence, prefixes, root)
    buckets: dict[str, list[SemanticMapping]] = {name: [] for name in BUCKETS}
    for row in rows:
        buckets[_bucket(row, evidence)].append(row)
    buckets["novel"], buckets["conflicts"] = reduce(buckets["novel"])
    rescued = sum(
        (row.subject.curie, row.object.curie) not in evidence.pairs
        for row in buckets["right"]
    )
    return Split(buckets, evidence, prefixes, rescued)


def reduce(
    rows: Sequence[SemanticMapping],
) -> tuple[list[SemanticMapping], list[SemanticMapping]]:
    """Keep the single highest confidence candidate for each subject and object."""
    by_subject: dict[str, list[SemanticMapping]] = defaultdict(list)
    by_object: dict[str, list[SemanticMapping]] = defaultdict(list)
    for row in rows:
        by_subject[row.subject.curie].append(row)
        by_object[row.object.curie].append(row)
    kept: list[SemanticMapping] = []
    conflicts: list[SemanticMapping] = []
    for row in rows:
        subjects = by_subject[row.subject.curie]
        objects = by_object[row.object.curie]
        target = kept if _wins(row, subjects) and _wins(row, objects) else conflicts
        target.append(row)
    return kept, conflicts


def _bucket(row: SemanticMapping, evidence: Evidence) -> str:
    """Name the bucket one candidate belongs in, judged within its own prefix pair."""
    subject, obj = row.subject.curie, row.object.curie
    if (subject, obj) in evidence.rejected:
        return "wrong"
    if (subject, obj) in evidence.pairs:
        return "right"
    mapped_subject = row.object.prefix in evidence.prefixes.get(subject, ())
    mapped_object = row.subject.prefix in evidence.prefixes.get(obj, ())
    if mapped_subject or mapped_object:
        return "wrong"
    return "right" if (subject, obj) in evidence.predicted else "novel"


def _wins(row: SemanticMapping, group: list[SemanticMapping]) -> bool:
    """Whether the row is the only highest confidence candidate in its group."""
    if len(group) == 1:
        return True
    scores = [other.confidence or 0.0 for other in group]
    best = max(scores)
    return (row.confidence or 0.0) == best and scores.count(best) == 1
