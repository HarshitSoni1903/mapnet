# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mapnet",
#     "leonmap @ git+https://github.com/HarshitSoni1903/Weakly-Supervised-Representation-Learning-for-Cross-Ontology-Mapping.git",
#     "huggingface-hub",
# ]
#
# [tool.uv.sources]
# mapnet = { path = "..", editable = true }
# ///
"""Run the LeonMap embedding matcher over two ontologies."""

import csv
import importlib.metadata as md
import sys
from pathlib import Path

import yaml
from curies import Reference
from sssom_pydantic import SemanticMapping

from mapnet import Mapper, to_reference

WORKDIR = Path("data/leonmap")
MODEL_REPO = "harshitsoni1903/sapbert-finetuned-semra"
STUDY = "mapnet"
THRESHOLD = 0.9

EXACT = Reference(prefix="skos", identifier="exactMatch")
CLOSE = Reference(prefix="skos", identifier="closeMatch")
LEXICAL = Reference(prefix="semapv", identifier="LexicalMatching")
SEMANTIC = Reference(prefix="semapv", identifier="SemanticSimilarityThresholdMatching")

# Identifier prefixes LeonMap filters candidates by, per ontology.
ID_PREFIXES = {
    "mondo": ["MONDO_"],
    "mesh": ["mesh_"],
    "doid": ["DOID_"],
    "icd10": ["icd10:"],
}
KINDS = {".owl": "owl_path", ".tsv": "csv_path"}


class LeonMapMapper(Mapper):
    name = "leonmap"
    version = md.version("leonmap")

    def match(self, args):
        """Build both collections, run the mapper, and read what it wrote."""
        work = WORKDIR.resolve()
        work.mkdir(parents=True, exist_ok=True)

        import leonmap.config as config

        config.PROJECT_ROOT = work

        source, target = _name(args.source), _name(args.target)
        _write_config(work / "config.yaml", args, work)
        _download_model(work)

        from leonmap.build_vdb import main as build
        from leonmap.mapper import main as mapper

        _call(build, ["--collections", source, target])
        _call(mapper, ["--study", STUDY])
        yield from _rows(work, source, target)


def _name(path):
    """Read the collection name from an ontology filename."""
    return path.stem.split("_")[0].lower()


def _collection(path):
    """Build one collection entry pointing at the file MapNet supplied."""
    kind = KINDS.get(path.suffix)
    if kind is None:
        raise ValueError(f"leonmap cannot read {path.suffix!r}, want {sorted(KINDS)}")
    name = _name(path)
    return {
        "source": "owl" if kind == "owl_path" else "csv",
        "model": "ft",
        kind: str(path.resolve()),
        "id_prefixes": ID_PREFIXES.get(name, [f"{name}_"]),
    }


def _write_config(path, args, work):
    """Write the run config and patch it over LeonMap's defaults."""
    source, target = _name(args.source), _name(args.target)
    path.write_text(
        yaml.safe_dump(
            {
                "build": {
                    "db_dir": str(work / "db"),
                    "data_dir": str(work / "data"),
                    "ft_model_path": str(work / "models" / "sap_FT"),
                    "log_dir": str(args.logs.resolve()),
                },
                "collections": {
                    source: _collection(args.source),
                    target: _collection(args.target),
                },
                "mappings": {
                    STUDY: {
                        "src_collection": source,
                        "tgt_collection": target,
                        "threshold": THRESHOLD,
                        "top_k": 1,
                        "reverse": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    from leonmap.config_loader import load_user_config

    load_user_config(str(path))


def _download_model(work):
    """Fetch the fine-tuned SapBERT weights when they are not on disk."""
    path = work / "models" / "sap_FT"
    if path.exists():
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_REPO, local_dir=str(path))


def _call(entry, argv):
    """Call a LeonMap entry point, which reads its arguments from sys.argv."""
    old = sys.argv
    sys.argv = [entry.__module__, *argv]
    try:
        entry()
    finally:
        sys.argv = old


def _rows(work, source, target):
    """Yield every prediction from the newest run directory."""
    runs = sorted((work / "mapper_results" / STUDY).glob("run_*"))
    if not runs:
        raise RuntimeError(f"leonmap wrote no run directory under {work}")
    with open(runs[-1] / f"{source}_to_{target}.tsv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
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
