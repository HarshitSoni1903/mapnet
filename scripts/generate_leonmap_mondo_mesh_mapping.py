"""
Generate MONDO <-> MeSH disease mappings using LeonMap.

Downloads data if missing, builds collections if missing, runs mapping.
Edit the variables below to customize. For full config overrides, point
CONFIG_YAML to a copy of leonmap/test_config.yaml with your changes.

This yaml also serves as a reference for the running of the code. Ideally all study configs are stored in this yaml file and 
if new study is to be added, the yaml should be modified and path added in the CONFIG_YAML variable.
Config: https://github.com/HarshitSoni1903/Weakly-Supervised-Representation-Learning-for-Cross-Ontology-Mapping/blob/main/leonmap/test_config.yaml

Install:
  requirements: PYTHON 3.10
  pip install git+https://github.com/HarshitSoni1903/Weakly-Supervised-Representation-Learning-for-Cross-Ontology-Mapping.git
  pip install indra gilda biomappings lxml
"""
from __future__ import annotations

import csv
import gzip
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from indra.databases import mesh_client
from lxml import etree


LEONMAP_ROOT = "."           # where data/, db/, models/ exist or will be created
STUDY = "mondo_mesh"         # key into YAML mappings section (leonmap/test_config.yaml for more details)
CONFIG_YAML = None           # path to YAML override, None = use package default


MONDO_OWL_URL = "http://purl.obolibrary.org/obo/mondo.owl"
MESH_OWL_GZ_URL = "https://w3id.org/biopragmatics/resources/mesh/mesh.owl.gz"
# BIOMAPPINGS_GOLD_URL = (
#     "https://raw.githubusercontent.com/biopragmatics/biomappings/"
#     "main/src/biomappings/resources/positive.sssom.tsv"
# )
BIOMAPPINGS_GOLD_URL = "https://raw.githubusercontent.com/HarshitSoni1903/mapnet/refs/heads/main/scripts/gilda_mondo_mesh_predictions.sssom.tsv"

MESH_DISEASE_TREE_PREFIXES = ("C", "F03")
HF_MODEL_REPO = "zeromorethanone/sapbert-finetuned-semra"


def download_mondo_owl(data_dir: Path) -> Path:
    out = data_dir / "mondo.owl"
    if out.exists():
        print(f"MONDO OWL exists, skipping: {out}")
        return out
    print(f"Downloading MONDO OWL -> {out}")
    urllib.request.urlretrieve(MONDO_OWL_URL, str(out))
    print(f"Done ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    return out


def download_and_filter_mesh_owl(data_dir: Path) -> Path:
    out = data_dir / "mesh_disease.owl"
    if out.exists():
        print(f"MeSH disease OWL exists, skipping: {out}")
        return out

    print("Getting MeSH disease IDs from indra.mesh_client ...")
    keep_iris = set()
    for mesh_id in mesh_client.mesh_id_to_name:
        if any(mesh_client.has_tree_prefix(mesh_id, p) for p in MESH_DISEASE_TREE_PREFIXES):
            keep_iris.add(f"http://id.nlm.nih.gov/mesh/{mesh_id}")
    print(f"{len(keep_iris)} disease descriptors (tree {', '.join(MESH_DISEASE_TREE_PREFIXES)})")

    gz_path = data_dir / "mesh.owl.gz"
    owl_full = data_dir / "mesh.owl"
    print("Downloading full MeSH OWL ...")
    urllib.request.urlretrieve(MESH_OWL_GZ_URL, str(gz_path))

    with gzip.open(str(gz_path), "rb") as f_in, open(str(owl_full), "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()

    print("Filtering to disease subset ...")
    RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    OWL_NS = "http://www.w3.org/2002/07/owl#"

    parser = etree.XMLParser(remove_comments=False, huge_tree=True)
    tree = etree.parse(str(owl_full), parser)
    root = tree.getroot()

    new_root = etree.Element(root.tag, nsmap=root.nsmap)
    children = list(root)
    kept = 0
    i = 0

    while i < len(children):
        node = children[i]
        if node.tag in (f"{{{OWL_NS}}}Ontology", f"{{{OWL_NS}}}AnnotationProperty"):
            new_root.append(node)
            i += 1
            continue
        if isinstance(node, etree._Comment):
            if i + 1 < len(children):
                nxt = children[i + 1]
                if nxt.get(f"{{{RDF_NS}}}about") in keep_iris:
                    new_root.append(node)
                    new_root.append(nxt)
                    kept += 1
            i += 2
            continue
        if node.get(f"{{{RDF_NS}}}about") in keep_iris:
            new_root.append(node)
            kept += 1
        i += 1

    etree.ElementTree(new_root).write(str(out), encoding="utf-8", xml_declaration=True, pretty_print=True)
    print(f"Kept {kept} blocks -> {out}")
    return out


def download_biomappings_gold(data_dir: Path, gold_filename: str) -> Path:
    out = data_dir / gold_filename
    if out.exists():
        print(f"Gold file exists, skipping: {out}")
        return out

    print("Downloading biomappings gold ...")
    tmp = data_dir / "_biomappings_raw.tsv"
    urllib.request.urlretrieve(BIOMAPPINGS_GOLD_URL, str(tmp))

    lines = [l for l in open(tmp, "r", encoding="utf-8") if not l.startswith("#")]
    rows = []
    if lines:
        for row in csv.DictReader(lines, delimiter="\t"):
            s, o = row.get("subject_id", "").lower(), row.get("object_id", "").lower()
            if (s.startswith("mondo:") and o.startswith("mesh:")) or \
               (s.startswith("mesh:") and o.startswith("mondo:")):
                rows.append(row)
    if rows:
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    tmp.unlink(missing_ok=True)
    print(f"{len(rows)} MONDO<->MeSH gold pairs -> {out}")
    return out


def _run(entry_main, cli_name: str, argv: list[str]) -> None:
    old = sys.argv
    sys.argv = [cli_name] + argv
    try:
        entry_main()
    finally:
        sys.argv = old

def export_sssom(mapper_tsv: Path, out_path: Path, include_gold: bool = False) -> Path:
    rows = []
    with open(mapper_tsv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            remarks = row.get("remarks", "")
            justification = ("semapv:SemanticSimilarity" if "cosine=" not in remarks
                             else "semapv:LexicalMatching")
            comment_parts = []
            if remarks:
                comment_parts.append(remarks)
            sssom_row = {
                "subject_id": row["src_id"],
                "subject_label": row["src_label"],
                "predicate_id": "skos:exactMatch",
                "object_id": row["tgt_id"],
                "object_label": row["tgt_label"],
                "confidence": row["score"],
                "mapping_justification": justification,
                "mapping_tool": "leonmap",
                "comment": "; ".join(comment_parts),
                "known_mapping": row["in_gold"],
            }
            rows.append(sssom_row)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("#curie_map:\n")
        f.write("#  mondo: http://purl.obolibrary.org/obo/MONDO_\n")
        f.write("#  mesh: https://meshb.nlm.nih.gov/record/ui?ui=\n")
        f.write("#  skos: http://www.w3.org/2004/02/skos/core#\n")
        f.write("#  semapv: https://w3id.org/semapv/vocab/\n")
        f.write(f"#mapping_set_id: leonmap_{STUDY}\n")
        f.write("#mapping_tool: leonmap\n")
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {len(rows)} SSSOM mappings -> {out_path}")
    return out_path


if __name__ == "__main__":
    root = Path(os.path.abspath(LEONMAP_ROOT))

    import leonmap.config as _cfg
    _cfg.PROJECT_ROOT = root

    if CONFIG_YAML:
        from leonmap.config_loader import load_user_config
        load_user_config(CONFIG_YAML)

    from huggingface_hub import snapshot_download
    from leonmap.config import BuildConfig, MAPPINGS, COLLECTIONS, resolve_path
    from leonmap.build_vdb import main as build_main
    from leonmap.mapper import main as mapper_main

    if STUDY not in MAPPINGS:
        raise SystemExit(f"Unknown study: {STUDY}. Available: {sorted(MAPPINGS.keys())}")

    study = MAPPINGS[STUDY]
    cfg = BuildConfig()
    data_dir = resolve_path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    model_dir = resolve_path(cfg.ft_model_path)
    if not model_dir.exists():
        print(f"Model not found locally, downloading from HF: {HF_MODEL_REPO}")
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=str(model_dir))
        print(f"Model downloaded -> {model_dir}")

    # Get data (skips if files exist)
    src_owl = COLLECTIONS[study["src_collection"]]["owl_path"]
    tgt_owl = COLLECTIONS[study["tgt_collection"]]["owl_path"]
    if "mondo.owl" in (src_owl, tgt_owl):
        download_mondo_owl(data_dir)
    if "mesh_disease.owl" in (src_owl, tgt_owl):
        download_and_filter_mesh_owl(data_dir)
    gold = study.get("gold_file")
    if gold:
        download_biomappings_gold(data_dir, gold)

    #Build collections and run mapping.
    cols = [study["src_collection"], study["tgt_collection"]]
    build_argv = ["--collections"] + cols
    _run(build_main, "leonmap-build", build_argv)

    map_argv = ["--study", STUDY]
    if CONFIG_YAML:
        map_argv += ["--config", CONFIG_YAML]
    _run(mapper_main, "leonmap-map", map_argv)

    #Export results.
    results_dir = resolve_path("mapper_results") / STUDY
    latest_run = max(results_dir.glob("run_*"), key=lambda p: p.name)
    mapper_tsv = latest_run / "mondo_to_mesh.tsv"
    if mapper_tsv.exists():
        sssom_out = root / f"leonmap_{STUDY}_predictions.sssom.tsv"
        export_sssom(mapper_tsv, sssom_out, include_gold=True)

    print(f"Done. Results in: {results_dir}/")