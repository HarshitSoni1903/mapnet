# /// script
# requires-python = ">=3.11"
# dependencies = ["mapnet", "gilda", "obonet"]
#
# [tool.uv.sources]
# mapnet = { path = "..", editable = true }
# ///
"""Run the Gilda lexical matcher over two ontologies."""

import importlib.metadata as md

import gilda
import obonet
from curies import Reference
from sssom_pydantic import SemanticMapping

from mapnet import Mapper, to_reference

JUSTIFICATION = Reference(prefix="semapv", identifier="LexicalMatching")
PREDICATE = Reference(prefix="skos", identifier="exactMatch")


class GildaMapper(Mapper):
    name = "gilda"
    version = md.version("gilda")

    def match(self, args):
        """Ground every source label against the target namespace."""
        prefix = _prefix(args.target).lower()
        for curie, label in _terms(args.source):
            for scored in gilda.ground(label, namespaces=[prefix.upper()]):
                if scored.term.db.lower() != prefix:
                    continue
                yield _mapping(curie, label, scored)
                break


def _prefix(path):
    """Read the prefix of the first term in an OBO file."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("id: ") and ":" in line[4:]:
                return line[4:].split(":", 1)[0].strip()
    raise ValueError(f"no prefixed term ids found in {path}")


def _terms(path):
    """Yield the CURIE and label of every named term in an OBO file."""
    for node, data in obonet.read_obo(path).nodes(data=True):
        label = data.get("name")
        if label:
            yield node, label


def _mapping(curie, label, scored):
    """Turn one Gilda match into an SSSOM row."""
    term = scored.term
    return SemanticMapping(
        subject=to_reference(curie, name=label),
        predicate=PREDICATE,
        object=to_reference(f"{term.db}:{term.id}", name=term.entry_name),
        justification=JUSTIFICATION,
        confidence=scored.score,
    )


if __name__ == "__main__":
    raise SystemExit(GildaMapper.main())
