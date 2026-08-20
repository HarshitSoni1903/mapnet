# /// script
# requires-python = ">=3.11"
# dependencies = ["mapnet", "gilda", "obonet"]
#
# [tool.uv.sources]
# mapnet = { path = "..", editable = true }
# ///
"""Match two ontologies on Gilda-normalized labels."""

import importlib.metadata as md
from collections import defaultdict

import gilda
import obonet

from mapnet import Mapper, Reference, SemanticMapping, to_curie, to_prefix, to_reference

JUSTIFICATION = Reference(prefix="semapv", identifier="LexicalMatching")
PREDICATE = Reference(prefix="skos", identifier="exactMatch")


class GildaMapper(Mapper):
    name = "gilda"
    version = md.version("gilda")

    def match(self, args):
        """Match the two ontologies one to one, names before synonyms."""
        names, synonyms = _index(args.source, to_prefix(args.source))
        targets = _targets(to_prefix(args.target))
        used_subjects, used_objects = set(), set()
        for confidence, table in ((1.0, names), (0.95, synonyms)):
            for text in sorted(table.keys() & targets.keys()):
                subjects, objects = table[text], targets[text]
                if len(subjects) != 1 or len(objects) != 1:
                    continue
                subject, obj = next(iter(subjects)), next(iter(objects))
                if subject[0] in used_subjects or obj[0] in used_objects:
                    continue
                used_subjects.add(subject[0])
                used_objects.add(obj[0])
                yield _mapping(subject, obj, confidence)


def _index(path, prefix):
    """Index the ontology's own names and synonyms by normalized text."""
    names, synonyms = defaultdict(set), defaultdict(set)
    for node, data in obonet.read_obo(path).nodes(data=True):
        label = data.get("name")
        if not label or not node.lower().startswith(f"{prefix}:"):
            continue
        names[gilda.process.normalize(label)].add((node, label))
        for raw in data.get("synonym", []):
            text = raw.split('"')[1] if '"' in raw else ""
            if text:
                synonyms[gilda.process.normalize(text)].add((node, label))
    return names, synonyms


def _targets(prefix):
    """Index one namespace's own Gilda terms by normalized text."""
    index = defaultdict(set)
    for group in gilda.get_grounder().entries.values():
        for term in group:
            if (term.source or "").lower() != prefix:
                continue
            curie = to_curie(f"{term.db}:{term.id}")
            if curie.split(":")[0] == prefix:
                index[term.norm_text].add((curie, term.entry_name))
    return index


def _mapping(subject, obj, confidence):
    """Turn one matched pair into an SSSOM row."""
    return SemanticMapping(
        subject=to_reference(subject[0], name=subject[1]),
        predicate=PREDICATE,
        object=to_reference(obj[0], name=obj[1]),
        justification=JUSTIFICATION,
        confidence=confidence,
    )


if __name__ == "__main__":
    raise SystemExit(GildaMapper.main())
