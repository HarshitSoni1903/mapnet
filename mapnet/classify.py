"""Combine and classify predictions against curated evidence."""

from __future__ import annotations

import csv
import importlib.metadata as md
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sssom_pydantic import SemanticMapping

from mapnet.sssom import read, write

BUCKETS = ("right", "wrong", "novel", "conflicts")


@dataclass(frozen=True)
class Evidence:
    """Curated pairs, and the prefixes each entity is already mapped into."""

    pairs: set[tuple[str, str]] = field(default_factory=set)
    prefixes: dict[str, set[str]] = field(default_factory=dict)


def aggregate(paths: Sequence[Path], out: Path) -> int:
    """Write one mapping set from several prediction files, first pair winning."""
    return write(union(paths), out, tool="mapnet", version=md.version("mapnet"))


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


def load_evidence(paths: Sequence[Path]) -> Evidence:
    """Stream curated mapping files into a pair set and the entities they cover."""
    pairs: set[tuple[str, str]] = set()
    prefixes: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            body = (line for line in handle if not line.startswith("#"))
            for row in csv.DictReader(body, delimiter="\t"):
                subject, obj = row["subject_id"], row["object_id"]
                pairs.add((subject, obj))
                pairs.add((obj, subject))
                prefixes[subject].add(obj.split(":")[0])
                prefixes[obj].add(subject.split(":")[0])
    return Evidence(pairs, dict(prefixes))


def classify(
    rows: Sequence[SemanticMapping], evidence: Evidence
) -> dict[str, list[SemanticMapping]]:
    """Split candidates against evidence, then reduce the novel ones to one to one."""
    buckets: dict[str, list[SemanticMapping]] = {name: [] for name in BUCKETS}
    for row in rows:
        buckets[_bucket(row, evidence)].append(row)
    kept, conflicts = reduce(buckets["novel"])
    buckets["novel"], buckets["conflicts"] = kept, conflicts
    return buckets


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
    if (subject, obj) in evidence.pairs:
        return "right"
    mapped_subject = row.object.prefix in evidence.prefixes.get(subject, ())
    mapped_object = row.subject.prefix in evidence.prefixes.get(obj, ())
    if mapped_subject or mapped_object:
        return "wrong"
    return "novel"


def _wins(row: SemanticMapping, group: list[SemanticMapping]) -> bool:
    """Whether the row is the only highest confidence candidate in its group."""
    if len(group) == 1:
        return True
    scores = [other.confidence or 0.0 for other in group]
    best = max(scores)
    return (row.confidence or 0.0) == best and scores.count(best) == 1
