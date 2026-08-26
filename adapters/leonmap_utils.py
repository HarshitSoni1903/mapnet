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

import importlib.metadata as md
import subprocess
import sys
from pathlib import Path

from mapnet import Mapper, SemanticMapping, table, to_prefix, to_reference

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
        out = work / f"{_prefix(args, 'src')}_to_{_prefix(args, 'tgt')}.tsv"
        subprocess.run(_command(args, work, out), check=True)
        yield from _rows(out)


def _prefix(args, side):
    """Take the prefix the run declared, else the one the file declares."""
    given = getattr(args, f"{side}_prefix", None)
    return given or to_prefix(args.source if side == "src" else args.target)


def _command(args, work, out):
    """Build the leonmap-map invocation."""
    source, target = args.source.resolve(), args.target.resolve()
    src, tgt = _prefix(args, "src"), _prefix(args, "tgt")
    command = [sys.executable, "-m", "leonmap.mapper"]
    command += ["--source", str(source), "--target", str(target)]
    command += ["--src-name", src, "--tgt-name", tgt]
    command += ["--src-prefix", f"{src}:"]
    command += ["--tgt-prefix", f"{tgt}:"]
    command += ["--work-dir", str(work), "--out", str(out)]
    command += ["--threshold", str(THRESHOLD), "--top_k", str(TOP_K)]
    command += ["--build-missing"]
    if args.config:
        command += ["--config", str(args.config)]
    return command


def _rows(out):
    """Yield every scored prediction from the TSV leonmap copied to --out."""
    for row in table(out):
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
