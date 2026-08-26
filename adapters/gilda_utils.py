# /// script
# requires-python = ">=3.11"
# dependencies = ["mapnet", "gilda", "obonet", "indra"]
#
# [tool.uv.sources]
# mapnet = { path = "..", editable = true }
# ///
"""Match two ontologies on Gilda-normalized labels."""

import importlib.metadata as md
import inspect
import re
from collections import defaultdict

import gilda.generate_terms
import obonet

import mapnet

JUSTIFICATION = mapnet.Reference(prefix="semapv", identifier="LexicalMatching")
PREDICATE = mapnet.Reference(prefix="skos", identifier="exactMatch")
SYNONYM = re.compile(r'^"(.+)" (EXACT|RELATED|NARROW|BROAD|\[\])')
CONFIDENCE = {2: 1.0, 1: 0.95, 0: 0.9}


class GildaMapper(mapnet.Mapper):
    name = "gilda"
    version = md.version("gilda")

    def match(self, args):
        """Match the two ontologies on every unambiguous shared label."""
        source = args.src_prefix or mapnet.to_prefix(args.source)
        names, synonyms = _index(args.source, source)
        targets = _targets(args.tgt_prefix or mapnet.to_prefix(args.target))
        seen = set()
        for named, table in ((True, names), (False, synonyms)):
            for text in sorted(table.keys() & targets.keys()):
                subjects, objects = table[text], targets[text]
                if len(subjects) != 1 or len({obj[0] for obj in objects}) != 1:
                    continue
                subject = next(iter(subjects))
                obj = min(objects, key=lambda entry: entry[2] != "name")
                if (subject[0], obj[0]) in seen:
                    continue
                seen.add((subject[0], obj[0]))
                yield _mapping(subject, obj, CONFIDENCE[named + (obj[2] == "name")])


def _index(path, prefix):
    """Index the ontology's names and exact synonyms by normalized text."""
    names, synonyms = defaultdict(set), defaultdict(set)
    graph = obonet.read_obo(path)
    mapnet.check_prefixes(path, prefix, graph.nodes)
    for node, data in graph.nodes(data=True):
        label = data.get("name")
        if not label or not node.lower().startswith(f"{prefix}:"):
            continue
        names[gilda.process.normalize(label)].add((node, label))
        for raw in data.get("synonym", []):
            found = SYNONYM.match(raw)
            if found and found.group(2) == "EXACT":
                text = gilda.process.normalize(found.group(1))
                synonyms[text].add((node, label))
    return names, synonyms


def _targets(prefix):
    """Index the target ontology's own Gilda terms by normalized text."""
    generate = getattr(gilda.generate_terms, f"generate_{prefix}_terms", None)
    if generate is None:
        raise ValueError(f"gilda has no term generator for {prefix!r}")
    drop = "ignore_mappings" in inspect.signature(generate).parameters
    index = defaultdict(set)
    for term in generate(ignore_mappings=True) if drop else generate():
        curie = mapnet.to_curie(f"{term.db}:{term.id}")
        index[term.norm_text].add((curie, term.entry_name, term.status))
    return index


def _mapping(subject, obj, confidence):
    """Turn one matched pair into an SSSOM row."""
    return mapnet.SemanticMapping(
        subject=mapnet.to_reference(subject[0], name=subject[1]),
        predicate=PREDICATE,
        object=mapnet.to_reference(obj[0], name=obj[1]),
        justification=JUSTIFICATION,
        confidence=confidence,
    )


if __name__ == "__main__":
    raise SystemExit(GildaMapper.main())
