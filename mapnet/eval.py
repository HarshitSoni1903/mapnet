"""Score predicted mappings against a gold standard."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass

Pair = tuple[str, str]

Ranked = Mapping[str, Sequence[str]]


@dataclass(frozen=True)
class Scores:
    """One prediction set judged against a gold standard, with the counts behind it."""

    hits: int
    predicted: int
    expected: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        """Return the scores as plain numbers."""
        return asdict(self)


def score(predicted: Iterable[Pair], gold: Iterable[Pair]) -> Scores:
    """Score predicted pairs against a gold standard, ignoring which side is which."""
    found = {_unordered(pair) for pair in predicted}
    wanted = {_unordered(pair) for pair in gold}
    hits = len(found & wanted)
    precision = hits / len(found) if found else 0.0
    recall = hits / len(wanted) if wanted else 0.0
    total = precision + recall
    f1 = 2 * precision * recall / total if total else 0.0
    return Scores(hits, len(found), len(wanted), precision, recall, f1)


def mrr(ranked: Ranked, gold: Iterable[Pair]) -> float:
    """Take the mean reciprocal rank of the first correct object per subject."""
    ranks = list(_ranks(ranked, gold))
    if not ranks:
        return 0.0
    return sum(1 / rank for rank in ranks if rank) / len(ranks)


def hits_at_k(ranked: Ranked, gold: Iterable[Pair], k: int = 1) -> float:
    """Take the share of scored subjects whose correct object is in the top k."""
    ranks = list(_ranks(ranked, gold))
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank and rank <= k) / len(ranks)


def _ranks(ranked: Ranked, gold: Iterable[Pair]) -> Iterator[int]:
    """Yield each gold-covered subject's rank of its first correct object, 0 if none."""
    answers: dict[str, set[str]] = {}
    for subject, obj in gold:
        answers.setdefault(subject, set()).add(obj)
        answers.setdefault(obj, set()).add(subject)
    for subject, candidates in ranked.items():
        correct = answers.get(subject)
        if not correct:
            continue
        yield next((i for i, o in enumerate(candidates, 1) if o in correct), 0)


def _unordered(pair: Pair) -> Pair:
    """Sort a pair's two ids into a fixed order."""
    subject, obj = pair
    return (subject, obj) if subject <= obj else (obj, subject)
