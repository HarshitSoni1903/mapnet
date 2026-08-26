# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mapnet",
#     "leonmap @ git+https://github.com/HarshitSoni1903/Weakly-Supervised-Representation-Learning-for-Cross-Ontology-Mapping.git",
# ]
#
# [tool.uv.sources]
# mapnet = { path = "..", editable = true }
# ///
"""Run the LeonMap embedding matcher over two ontologies."""

import csv
import importlib.metadata as md
import subprocess
import sys
from pathlib import Path

from mapnet import Mapper, SemanticMapping, to_prefix, to_reference

WORKDIR = Path("data/leonmap")
THRESHOLD = 0.9
TOP_K = 1

EXACT = to_reference("skos:exactMatch")
CLOSE = to_reference("skos:closeMatch")
LEXICAL = to_reference("semapv:LexicalMatching")
SEMANTIC = to_reference("semapv:SemanticSimilarityThresholdMatching")


class LeonMapMapper(Mapper):
    name = "leonmap"
    version = md.version("leonmap")

    def match(self, args):
        """Run leonmap-map and yield every prediction it wrote."""
        work = WORKDIR.resolve()
        work.mkdir(parents=True, exist_ok=True)
        out = work / f"{to_prefix(args.source)}_to_{to_prefix(args.target)}.tsv"
        subprocess.run(_command(args, work, out), check=True)
        yield from _rows(out)


def _command(args, work, out):
    """Build the leonmap-map invocation."""
    source, target = args.source.resolve(), args.target.resolve()
    command = [sys.executable, "-m", "leonmap.mapper"]
    command += ["--source", str(source), "--target", str(target)]
    command += ["--src-name", to_prefix(source), "--tgt-name", to_prefix(target)]
    command += ["--src-prefix", f"{to_prefix(source)}:"]
    command += ["--tgt-prefix", f"{to_prefix(target)}:"]
    command += ["--work-dir", str(work), "--out", str(out)]
    command += ["--threshold", str(THRESHOLD), "--top_k", str(TOP_K)]
    command += ["--build-missing"]
    if args.config:
        command += ["--config", str(args.config)]
    return command


def _rows(out):
    """Yield every scored prediction from the TSV leonmap copied to --out."""
    with out.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("score"):
                yield _mapping(row)


def _mapping(row):
    """Turn one LeonMap prediction into an SSSOM row."""
    same = row["src_label"].strip().lower() == row["tgt_label"].strip().lower()
    return SemanticMapping(
        subject=to_reference(row["src_id"], name=row["src_label"]),
        predicate=EXACT if same else CLOSE,
        object=to_reference(row["tgt_id"], name=row["tgt_label"]),
        justification=LEXICAL if same else SEMANTIC,
        confidence=float(row["score"]),
    )


if __name__ == "__main__":
    raise SystemExit(LeonMapMapper.main())
