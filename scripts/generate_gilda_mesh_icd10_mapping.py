# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "mapnet @ git+https://github.com/gyorilab/mapnet.git@969d11b915",
#     "openacme @ git+https://github.com/gyorilab/openacme.git",
#     "gilda==1.5.0",
#     "indra==1.24.0",
#     "biomappings==0.4.2",
#     "pyobo==0.12.18",
#     "bioregistry==0.13.23",
#     "bioversions==0.8.289",
#     "bioontologies==0.7.4",
#     "networkx==3.6.1",
#     "polars==1.39.3",
#     "pandas==2.3.3",
#     "pystow==0.7.28",
# ]
#
# [tool.uv]
# override-dependencies = [
#     "torch ; sys_platform == 'nope'",
#     "torchvision ; sys_platform == 'nope'",
#     "torchaudio ; sys_platform == 'nope'",
#     "deeponto ; sys_platform == 'nope'",
#     "jpype1 ; sys_platform == 'nope'",
#     "transformers ; sys_platform == 'nope'",
#     "black ; sys_platform == 'nope'",
# ]
# ///
"""WHO ICD-10 -> full MeSH by Gilda lexical matching, classified against SemRA + Biomappings.
    uv run --script https://raw.githubusercontent.com/gyorilab/mapnet/refs/heads/main/scripts/generate_gilda_mesh_icd10_mapping.py
"""
import argparse
import importlib.metadata as md
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import bioontologies.robot
import pandas as pd
import polars as pl
import pystow
from biomappings.resources import POSITIVES_SSSOM_PATH, PREDICTIONS_SSSOM_PATH
from gilda.generate_terms import generate_mesh_terms
from gilda.process import normalize
from indra.databases import mesh_client
from openacme.icd10.icd10 import ICD10_XML_URL, get_icd10_graph

if not hasattr(bioontologies.robot, "ROBOT_COMMAND"):
    bioontologies.robot.ROBOT_COMMAND = ["robot"]

from mapnet.utils.filtering import (
    get_right_wrong_mappings,
    repair_names_with_semra,
)
from mapnet.utils.utils import make_undirected, sssom_to_biomappings

# Pinned data sources. Only SemRA's disease landscape carries icd10.
SEMRA_URL = "https://zenodo.org/records/15826693/files/processed.sssom.tsv.gz?download=1"
SEMRA_NAME = "semra_disease_landscape_mappings.tsv.gz"
MAPPING_TOOL = ("https://github.com/gyorilab/mapnet/blob/main/scripts/"
                "generate_gilda_mesh_icd10_mapping.py")

BASE = "gilda_icd10_mesh"

SSSOM_COLUMNS = ["subject_id", "subject_label", "predicate_id", "object_id", "object_label",
                 "mapping_justification", "subject_source_version", "object_source_version",
                 "mapping_tool", "mapping_tool_id", "mapping_tool_version", "mapping_date",
                 "confidence"]


def load_semra_landscape_df():
    """Fetch (once, cached) the SemRA disease landscape in biomappings column layout."""
    path = pystow.ensure_gunzip("semra", url=SEMRA_URL, name=SEMRA_NAME)
    return sssom_to_biomappings(pl.read_csv(path, separator="\t"))


def _canonical(curie):
    ns, _, local = curie.partition(":")
    return f"{ns.lower()}:{local}" if local else curie.lower()


def _biomappings_evidence():
    """icd10:<->mesh: pairs from Biomappings SSSOM exports."""
    frames = []
    for path in (PREDICTIONS_SSSOM_PATH, POSITIVES_SSSOM_PATH):
        d = pd.read_csv(path, comment="#", sep="\t").fillna("")
        d[["subject_id", "object_id"]] = d[["subject_id", "object_id"]].map(_canonical)
        fwd = d[d.subject_id.str.startswith("icd10:") & d.object_id.str.startswith("mesh:")]
        rev = d[d.subject_id.str.startswith("mesh:") & d.object_id.str.startswith("icd10:")].rename(
            columns={"subject_id": "object_id", "object_id": "subject_id",
                     "subject_label": "object_label", "object_label": "subject_label"})
        frames.append(pd.concat([fwd, rev]))
    ev = pd.concat(frames, ignore_index=True)
    return pl.DataFrame({"source identifier": ev.subject_id, "source name": ev.subject_label,
                         "source prefix": "icd10", "target identifier": ev.object_id,
                         "target name": ev.object_label, "target prefix": "mesh"})


def greedy_match(pref, incl, mesh):
    """Unambiguous 1:1 overlap in two passes: preferred labels claim src/tgt, then inclusions fill the rest."""
    pairs, used_src, used_tgt = {}, set(), set()
    for via_pref, table in ((True, pref), (False, incl)):
        for norm in sorted(table.keys() & mesh.keys()):  # sorted -> reproducible tie-breaking
            icds, meshes = table[norm], mesh[norm]
            if len(icds) != 1 or len(meshes) != 1:
                continue
            icd, msh = next(iter(icds)), next(iter(meshes))
            if icd in used_src or msh in used_tgt:
                continue
            used_src.add(icd)
            used_tgt.add(msh)
            pairs[(icd, msh)] = via_pref
    return pairs


def _provenance():
    """Column values shared by every row: source releases and tool identity."""
    icd10 = re.search(r"icd10(\d{4})en", ICD10_XML_URL)
    return {"subject_source_version": icd10.group(1) if icd10 else "",
            # indra's bundled mesh_id_label_mappings.tsv states no release.
            "object_source_version": "",
            "mapping_tool": MAPPING_TOOL,
            "mapping_tool_id": "gilda",
            "mapping_tool_version": md.version("gilda"),
            "mapping_date": date.today().isoformat()}


def _write_sssom(df, out_path, provenance):
    if df.is_empty():
        return
    rows = [{"subject_id": r["source identifier"], "subject_label": r["source name"],
             "predicate_id": "skos:exactMatch",
             "object_id": r.get("predicted identifier", r.get("target identifier")),
             "object_label": r.get("predicted name", r.get("target name", "")),
             "mapping_justification": "semapv:LexicalMatching",
             "confidence": r.get("confidence", ""), **provenance}
            for r in df.iter_rows(named=True)]
    out = pd.DataFrame(rows, columns=SSSOM_COLUMNS).fillna("")
    out.to_csv(out_path, sep="\t", index=False)
    print(f"[sssom] {len(rows)} -> {out_path}")


def classify(df, out_dir):
    """Split predictions into right/wrong/novel against SemRA + Biomappings evidence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    semra = load_semra_landscape_df()
    df = repair_names_with_semra(df, semra)
    evidence = make_undirected(pl.concat([semra, _biomappings_evidence()]).unique())
    print(f"[evidence] {evidence.height} undirected pairs")
    right, wrong, novel = get_right_wrong_mappings(df, evidence)
    provenance = _provenance()
    for tag, part in (("novel", novel), ("right", right), ("wrong", wrong)):
        _write_sssom(part, out_dir / f"{BASE}_{tag}.sssom.tsv", provenance)
    print(f"[classify] right={right.height} wrong={wrong.height} novel={novel.height}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path(f"{BASE}_classified"), help="directory for the right/wrong/novel SSSOM files")
    args = parser.parse_args()

    g = get_icd10_graph()
    pref_by_norm, incl_by_norm, icd_label = defaultdict(set), defaultdict(set), {}
    for code, data in g.nodes(data=True):
        rub = data.get("rubrics", {}) or {}
        label = (rub.get("preferred") or [""])[0]
        cid = f"icd10:{code}"
        icd_label[cid] = label
        if label:
            pref_by_norm[normalize(label)].add(cid)
        for term in rub.get("inclusion", []) or []:
            if term:
                incl_by_norm[normalize(term)].add(cid)

    # Index full MeSH by Gilda-normalized text.
    mesh_by_norm = defaultdict(set)
    for term in generate_mesh_terms(ignore_mappings=True):
        mesh_by_norm[term.norm_text].add(term.id)

    pairs = greedy_match(pref_by_norm, incl_by_norm, mesh_by_norm)
    rows = [{
        "source identifier": icd, "source name": icd_label.get(icd, ""), "source prefix": "icd10",
        "target identifier": f"mesh:{msh}",
        "target name": mesh_client.get_mesh_name(msh, offline=True) or "", "target prefix": "mesh",
        "confidence": 1.0 if via_pref else 0.95,
    } for (icd, msh), via_pref in sorted(pairs.items())]
    predictions = pl.DataFrame(rows)
    n_pref = sum(pairs.values())
    print(f"[gilda] predictions: {predictions.height} "
          f"(preferred={n_pref}, inclusion={predictions.height - n_pref})")

    classify(predictions, args.out_dir)


if __name__ == "__main__":
    main()
