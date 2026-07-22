import argparse
import json
import os
import re
import sys
from itertools import chain
from pathlib import Path

import faiss
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from leonmap.utils import canonicalize_id

STUDY = "icd10_mesh_full"
HF_MODEL_REPO = "harshitsoni1903/sapbert-finetuned-semra"

# own-weight decays with distance to the frontier: alpha = max(MIN, A0 * DECAY**(dist-1))
A0, DECAY, ALPHA_MIN = 0.7, 0.7, 0.3

COLLECTION = {"icd10": {"source": "csv", "model": "ft",
                        "csv_path": "icd10_concepts.tsv", "id_prefixes": ["icd10:"]}}
MAPPING = {STUDY: {"src_collection": "icd10", "tgt_collection": "mesh_full",
                   "src_col": "subject_id", "tgt_col": "object_id",
                   "threshold": 0.9, "top_k": 1, "reverse": False}}


def _flat(x):
    # flatten nested lists of strings (UMLS nests defs/synonyms for higher-order codes), dropping blanks
    if isinstance(x, str):
        return [x] if x.strip() else []
    if isinstance(x, list):
        return list(chain.from_iterable(_flat(y) for y in x))
    return []


def _norm(s):
    # token-sorted, punctuation-free label form for exact-name comparison
    return " ".join(sorted(re.findall(r"[a-z0-9]+", (s or "").lower())))


def _cosine(remarks):
    m = re.search(r"cosine=([\d.]+)", str(remarks))
    return float(m.group(1)) if m else None


# ICD-10 preprocessing

def _umls_definitions():
    # {code: {definition, synonyms}} from UMLS; caller falls back to ClaML if this raises
    from openacme.icd10.map_definitions import map_icd10_to_definitions
    raw = map_icd10_to_definitions(umls_api_key=os.environ["UMLS_API_KEY"])
    return {code: {"definition": next(iter(_flat(v.get("definition"))), ""),
                   "synonyms": _flat(v.get("synonyms"))} for code, v in raw.items()}


def load_icd10():
    # ICD-10 graph and code->immediate-children map (is_a edges run child->parent)
    from openacme.icd10.icd10 import get_icd10_graph
    g = get_icd10_graph()
    children = {c: kids for c in g.nodes
                if (kids := [u for u, _, d in g.in_edges(c, data=True) if d.get("kind") == "is_a"])}
    return g, children


def write_concepts(g, data_dir, use_umls):
    # write icd10_concepts.tsv (id/label/definition/synonyms) -- only needed when (re)building the index
    defs = {}
    if use_umls:
        try:
            defs = _umls_definitions()
        except Exception as e:
            print(f"[WARN] UMLS unavailable ({e}); using ClaML labels only.")
    rows = []
    for code, data in g.nodes(data=True):
        rub = data.get("rubrics", {}) or {}
        label = (rub.get("preferred") or [""])[0].strip() or code
        umls = defs.get(code, {})
        synonyms = (rub.get("inclusion") or []) + (umls.get("synonyms") or [])
        synonyms = [s.strip() for s in synonyms if s.strip() and s.strip().lower() != label.lower()]
        rows.append({"id": f"icd10:{code}", "label": label,
                     "definition": umls.get("definition", ""), "synonyms": ";".join(dict.fromkeys(synonyms))})
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(data_dir / "icd10_concepts.tsv", sep="\t", index=False)
    print(f"[icd10] wrote {len(rows)} concepts -> icd10_concepts.tsv")


# Hierarchical vector blend

def _blend_pass(vecs, pos, order, neighbors_of):
    # one directional tree blend: own-weight decays with distance to the frontier, blended against neighbors
    dist = {}
    for code in order:
        nbrs = [pos[f"icd10:{c}"] for c in neighbors_of.get(code, []) if f"icd10:{c}" in pos]
        dist[code] = 1 + max((dist.get(c, 0) for c in neighbors_of.get(code, [])), default=-1)
        p = pos.get(f"icd10:{code}")
        if p is None or not nbrs:
            continue
        a = max(ALPHA_MIN, A0 * DECAY ** (dist[code] - 1))
        v = a * vecs[p] + (1 - a) * vecs[nbrs].mean(axis=0)
        vecs[p] = v / (np.linalg.norm(v) or 1.0)
    return vecs


def blend_collection(db_dir, children):
    # rewrite the icd10 index: blend every node toward descendants (bottom-up) and ancestors (top-down), then average
    cdir = db_dir / "icd10"
    index = faiss.read_index(str(cdir / "index.raw.faiss"))
    pos = json.loads((cdir / "id2pos.json").read_text())
    vecs = index.reconstruct_n(0, index.ntotal)

    tree = nx.DiGraph((p, c) for p, kids in children.items() for c in kids)
    leaves_first = list(reversed(list(nx.topological_sort(tree))))
    root_first = list(reversed(leaves_first))
    parent_of = {c: [p] for p, kids in children.items() for c in kids}

    # leaves are exempt from the top-down pass: a leaf's own label is its strongest signal (often an exact MeSH match), so pulling it toward its ancestors only buries it below the retrieval cutoff.
    down_order = [c for c in root_first if c in children]
    up = _blend_pass(vecs.copy(), pos, leaves_first, children)
    down = _blend_pass(vecs.copy(), pos, down_order, parent_of)
    combined = up + down
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    blended = combined / np.where(norms == 0, 1.0, norms)

    out = faiss.IndexFlatIP(index.d)
    out.add(blended)
    faiss.write_index(out, str(cdir / "index.faiss"))
    print(f"[blend] rewrote {index.ntotal} vectors")


# Lexical labeling

def refine_mapping(mapper_tsv, g, cfg):
    # label each baseline top-1 as exact/synonym/semantic via lexical lookup; else it defaults to semantic
    from leonmap.utils import load_collection
    mesh = load_collection(cfg, "mesh_full")

    rows = []
    for _, base in pd.read_csv(mapper_tsv, sep="\t").fillna("").iterrows():
        code = base["src_id"].split(":", 1)[1]
        rub = g.nodes[code].get("rubrics", {}) or {}
        label = base["src_label"] or (rub.get("preferred") or [code])[0]
        lexical = set(mesh.exact_match_ids(label, rub.get("inclusion", [])))
        cos = _cosine(base["remarks"])
        raw = round(min(cos if cos is not None else float(base["score"]), 1.0), 6)
        if base["tgt_id"] in lexical:
            kind = "exact" if _norm(label) == _norm(base["tgt_label"]) else "synonym"
            conf, remark = 1.0, f"{kind};cosine={raw:.6f}"
        else:
            conf, remark = raw, "semantic"
        rows.append({"src_id": base["src_id"], "src_label": label, "tgt_id": base["tgt_id"],
                     "tgt_label": base["tgt_label"], "rank": 1, "score": conf, "remarks": remark})

    out = mapper_tsv.with_suffix(".refined.tsv")
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"[refine] {len(rows)} -> {out.name}")
    return out


# Classification + SSSOM

def _predictions_df(tsv):
    # refined mapper TSV -> (predictions frame, {(src, tgt): remark})
    d = pd.read_csv(tsv, sep="\t").fillna("")
    d["src"], d["tgt"] = d.src_id.map(canonicalize_id), d.tgt_id.map(canonicalize_id)
    remark = {(s, t): r for s, t, r in zip(d.src, d.tgt, d.remarks)}
    frame = pl.DataFrame({
        "source identifier": d.src, "source name": d.src_label, "source prefix": "icd10",
        "target identifier": d.tgt, "target name": d.tgt_label, "target prefix": "mesh",
        "confidence": d.score.astype(float)})
    return frame, remark


def _to_common(df):
    # right/novel carry 'target *'; wrong carries 'predicted *' -- reduce both to one schema
    tid = "predicted identifier" if "predicted identifier" in df.columns else "target identifier"
    tnm = "predicted name" if "predicted name" in df.columns else "target name"
    conf = pl.col("confidence") if "confidence" in df.columns else pl.lit(1.0).alias("confidence")
    return df.select([pl.col("source identifier"), pl.col("source name"),
                      pl.col(tid).alias("target identifier"), pl.col(tnm).alias("target name"), conf])


def _write_sssom(df, out_path, set_id, remark):
    # emit SSSOM; remark is looked up by (src, tgt) since get_right_wrong_mappings drops it as a column
    records = []
    for r in df.iter_rows(named=True):
        obj = r.get("predicted identifier", r["target identifier"])
        kind, _, comment = remark.get((r["source identifier"], obj), "semantic").partition(";")
        records.append({"subject_id": r["source identifier"], "subject_label": r["source name"],
                        "predicate_id": "skos:exactMatch" if kind == "exact" else "skos:closeMatch",
                        "object_id": obj, "object_label": r.get("predicted name", r["target name"]),
                        "confidence": r["confidence"],
                        "mapping_justification": "semapv:LexicalMatching" if kind in ("exact", "synonym")
                        else "semapv:SemanticSimilarity",
                        "mapping_tool": "leonmap", "comment": comment})
    header = ("#curie_map:\n#  icd10: https://icd.who.int/browse10/2019/en#/\n"
              "#  mesh: https://meshb.nlm.nih.gov/record/ui?ui=\n"
              "#  skos: http://www.w3.org/2004/02/skos/core#\n"
              "#  semapv: https://w3id.org/semapv/vocab/\n"
              f"#mapping_set_id: {set_id}\n#mapping_tool: leonmap\n")
    out_path.write_text(header)
    pd.DataFrame(records).to_csv(out_path, sep="\t", index=False, mode="a")
    print(f"[sssom] {len(records)} -> {out_path.name}")


def _biomappings_evidence():
    # icd10:<->mesh: pairs from Biomappings SSSOM exports (read directly; mapnet's own loader expects an old schema)
    from biomappings.resources import PREDICTIONS_SSSOM_PATH, POSITIVES_SSSOM_PATH
    from mapnet.utils.utils import sssom_to_biomappings
    frames = []
    for path in (PREDICTIONS_SSSOM_PATH, POSITIVES_SSSOM_PATH):
        d = pd.read_csv(path, comment="#", sep="\t").fillna("")
        d[["subject_id", "object_id"]] = d[["subject_id", "object_id"]].map(canonicalize_id)
        fwd = d[d.subject_id.str.startswith("icd10:") & d.object_id.str.startswith("mesh:")]
        rev = d[d.subject_id.str.startswith("mesh:") & d.object_id.str.startswith("icd10:")].rename(
            columns={"subject_id": "object_id", "object_id": "subject_id",
                     "subject_label": "object_label", "object_label": "subject_label"})
        frames.append(pd.concat([fwd, rev]))
    ev = pd.concat(frames, ignore_index=True)
    return sssom_to_biomappings(pl.from_pandas(ev[["subject_id", "subject_label", "object_id", "object_label"]]))


def classify(predictions, remark, out_dir, semra_raw):
    # flatten n:1 collisions to the best 1:1 pick, split right/wrong/novel against evidence, write SSSOM
    from mapnet.utils.filtering import repair_names_with_semra, get_right_wrong_mappings
    from mapnet.utils.utils import make_undirected, sssom_to_biomappings
    out_dir.mkdir(parents=True, exist_ok=True)

    semra = sssom_to_biomappings(semra_raw, {"icd10": {}, "mesh": {}}, {"icd10": "icd10", "mesh": "mesh"})
    predictions = repair_names_with_semra(predictions, semra)

    # collapse n:1 collisions: exact label match beats synonym beats semantic, tie-broken by raw cosine
    kind_rank = {"exact": 2, "synonym": 1, "semantic": 0}

    def quality(r):
        kind, _, comment = remark.get((r["source identifier"], r["target identifier"]), "").partition(";")
        cos = _cosine(comment) if kind else r["confidence"]
        return kind_rank.get(kind, 0), cos if cos is not None else 0.0

    keep, losers, used_src, used_tgt = [], [], set(), set()
    for r in sorted(predictions.to_dicts(), key=quality, reverse=True):
        s, t = r["source identifier"], r["target identifier"]
        (losers if s in used_src or t in used_tgt else keep).append(r)
        used_src.add(s)
        used_tgt.add(t)
    schema = predictions.schema
    predictions, dup_losers = pl.DataFrame(keep, schema=schema), pl.DataFrame(losers, schema=schema)

    evidence = make_undirected(pl.concat([semra, _biomappings_evidence()]).unique())
    print(f"[evidence] {evidence.height} undirected pairs")

    right, wrong, novel = get_right_wrong_mappings(predictions, evidence)
    right, novel = _to_common(right), _to_common(novel)
    wrong = pl.concat([_to_common(wrong), _to_common(dup_losers)])
    for tag, part in (("novel", novel), ("right", right), ("wrong", wrong)):
        _write_sssom(part, out_dir / f"leonmap_{STUDY}_{tag}.sssom.tsv", f"leonmap_{STUDY}_{tag}", remark)
    print(f"[classify] right={right.height} wrong={wrong.height} novel={novel.height} "
          f"(dup_collisions={dup_losers.height})")


# Orchestration

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--rebuild", action="store_true", help="re-embed the icd10 collection")
    ap.add_argument("-t", "--threshold", type=float, default=0.99, help="mapper confidence threshold")
    ap.add_argument("--no-blend", action="store_true",
                    help="map against the original db (skip the tree re-arrangement)")
    ap.add_argument("--no-umls", action="store_true", help="rebuild without UMLS enrichment (ClaML labels only)")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip lexical-hit labeling (exact/synonym predicates fall back to semantic)")
    ap.add_argument("--out-dir", default=f"leonmap_{STUDY}_classified")
    args = ap.parse_args()

    root = Path(os.path.abspath("."))
    import leonmap.config as config
    config.PROJECT_ROOT = root
    from leonmap.config import BuildConfig, COLLECTIONS, MAPPINGS, resolve_path
    from leonmap.build_vdb import main as build_main
    from leonmap.mapper import main as mapper_main
    COLLECTIONS.update(COLLECTION)
    MAPPINGS.update(MAPPING)

    cfg = BuildConfig()
    if not resolve_path(cfg.ft_model_path).exists():
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=str(resolve_path(cfg.ft_model_path)))

    g, children = load_icd10()
    icd10_dir = resolve_path(cfg.db_dir) / "icd10"
    raw = icd10_dir / "index.raw.faiss"

    # UMLS enrichment, the concepts write, and re-embedding only matter when rebuilding
    if args.rebuild:
        write_concepts(g, resolve_path(cfg.data_dir), use_umls=not args.no_umls)
        sys.argv = ["leonmap-build", "--collections", "icd10", "--monitor", "0", "--rebuild"]
        build_main()
        raw.write_bytes((icd10_dir / "index.faiss").read_bytes())   # snapshot the pristine build
    elif not raw.exists():
        raise SystemExit("No index.raw.faiss — run once with --rebuild first.")

    from mapnet.utils.filtering import load_semera_landscape_df
    semra_raw = load_semera_landscape_df(
        "disease", {"icd10": {}, "mesh": {}}, {"icd10": "icd10", "mesh": "mesh"}, sssom=True)

    if args.no_blend:
        (icd10_dir / "index.faiss").write_bytes(raw.read_bytes())
        print("[blend] skipped; mapping against the original db")
    else:
        blend_collection(resolve_path(cfg.db_dir), children)

    sys.argv = ["leonmap-map", "--study", STUDY, "--threshold", str(args.threshold)]
    mapper_main()

    run = max((resolve_path("mapper_results") / STUDY).glob("run_*"), key=lambda p: p.name)
    tsv = run / "icd10_to_mesh_full.tsv"
    if not args.no_refine:
        tsv = refine_mapping(tsv, g, cfg)
    predictions, remark = _predictions_df(tsv)
    classify(predictions, remark, root / args.out_dir, semra_raw)
    print(f"Done -> {root / args.out_dir}/")
