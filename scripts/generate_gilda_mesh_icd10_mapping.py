import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import polars as pl
from gilda.process import normalize
from gilda.generate_terms import generate_mesh_terms
from indra.databases import mesh_client
from openacme.icd10.icd10 import get_icd10_graph

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

BASE = "gilda_icd10_mesh"


def _canonical(curie):
    ns, _, local = curie.partition(":")
    return f"{ns.lower()}:{local}" if local else curie.lower()


def _biomappings_evidence():
    """icd10:<->mesh: pairs from Biomappings SSSOM exports."""
    from biomappings.resources import PREDICTIONS_SSSOM_PATH, POSITIVES_SSSOM_PATH
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


def _write_sssom(df, out_path, set_id):
    rows = [{"subject_id": r["source identifier"], "subject_label": r["source name"],
             "predicate_id": "skos:exactMatch",
             "object_id": r.get("predicted identifier", r.get("target identifier")),
             "object_label": r.get("predicted name", r.get("target name", "")),
             "confidence": str(r.get("confidence", "")),
             "mapping_justification": "semapv:LexicalMatching", "mapping_tool": "gilda"}
            for r in df.iter_rows(named=True)]
    header = ("#curie_map:\n#  icd10: https://icd.who.int/browse10/2019/en#/\n"
              "#  mesh: https://meshb.nlm.nih.gov/record/ui?ui=\n"
              "#  skos: http://www.w3.org/2004/02/skos/core#\n"
              "#  semapv: https://w3id.org/semapv/vocab/\n"
              f"#mapping_set_id: {set_id}\n#mapping_tool: gilda\n")
    with open(out_path, "w") as f:
        f.write(header)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False, mode="a")
    print(f"[sssom] {len(rows)} -> {out_path}")


def classify(df, out_dir):
    """Split predictions into right/wrong/novel against SemRA + Biomappings evidence."""
    from mapnet.utils.filtering import (load_semera_landscape_df, repair_names_with_semra,
                                        get_right_wrong_mappings)
    from mapnet.utils.utils import make_undirected
    out_dir.mkdir(parents=True, exist_ok=True)
    semra = load_semera_landscape_df("disease", {"icd10": {}, "mesh": {}}, {"icd10": "icd10", "mesh": "mesh"})
    df = repair_names_with_semra(df, semra)
    evidence = make_undirected(pl.concat([semra, _biomappings_evidence()]).unique())
    print(f"[evidence] {evidence.height} undirected pairs")
    right, wrong, novel = get_right_wrong_mappings(df, evidence)
    for tag, part in (("novel", novel), ("right", right), ("wrong", wrong)):
        _write_sssom(part, out_dir / f"{BASE}_{tag}.sssom.tsv", f"{BASE}_{tag}")
    print(f"[classify] right={right.height} wrong={wrong.height} novel={novel.height}")


if __name__ == "__main__":
    g = get_icd10_graph()

    # Index ICD-10 nodes by normalized preferred label and by normalized inclusion terms.
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

    classify(predictions, Path(f"{BASE}_classified"))
