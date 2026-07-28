"""Combine two SSSOM files on (subject_id, object_id): shared pairs merged, plus each side's unique rows.

Usage: python scripts/intersect_sssom.py A.sssom.tsv B.sssom.tsv [OUT_DIR]

Tool names come from each file's mapping_tool column; the shared dataset name and classification
category (e.g. "novel") are inferred from the two input filenames, so output names stay informative
even with several dataset/category comparisons sitting in the same directory. Writes into OUT_DIR
(default: current dir), e.g. for leonmap_icd10_mesh_full_novel.sssom.tsv vs gilda_icd10_mesh_novel.sssom.tsv:
  icd10_mesh_leonmap_gilda_common_novel.tsv        shared pairs, each tool's columns side by side
  icd10_mesh_leonmap_gilda_only_leonmap_novel.tsv  pairs only in the first file (kept as valid SSSOM)
  icd10_mesh_leonmap_gilda_only_gilda_novel.tsv    pairs only in the second file
"""
import argparse
from pathlib import Path

import pandas as pd

_ID = ["subject_id", "subject_label", "object_id", "object_label"]


def _canon(curie):
    ns, _, local = str(curie).partition(":")
    return f"{ns.lower()}:{local}" if local else str(curie).lower()


def _read(path):
    df = pd.read_csv(path, comment="#", sep="\t").fillna("")
    df["_key"] = list(zip(df.subject_id.map(_canon), df.object_id.map(_canon)))
    return df


def _header(path):
    """Leading SSSOM '#' metadata block, so the unique-row outputs stay valid SSSOM."""
    lines = []
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            lines.append(line)
    return "".join(lines)


def _tag(df, fallback):
    tool = df["mapping_tool"].iloc[0] if "mapping_tool" in df.columns and len(df) else ""
    return str(tool).strip() or fallback


def _write_sssom(df, header, out_path):
    out_path.write_text(header)
    df.drop(columns="_key").to_csv(out_path, sep="\t", index=False, mode="a")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Combine two SSSOM files into common + per-tool unique rows.")
    ap.add_argument("sssom_a")
    ap.add_argument("sssom_b")
    ap.add_argument("out_dir", nargs="?", default=".", help="output directory (default: current dir)")
    args = ap.parse_args()

    a, b = _read(args.sssom_a), _read(args.sssom_b)
    ta, tb = _tag(a, "a"), _tag(b, "b")

    # Output name: shared trailing token of the two filenames as the category (e.g. "novel"), longest
    # shared leading tokens (after stripping each tool's own name) as the dataset name.
    tok_a = Path(args.sssom_a).name.removesuffix(".sssom.tsv").removesuffix(".tsv").split("_")
    tok_b = Path(args.sssom_b).name.removesuffix(".sssom.tsv").removesuffix(".tsv").split("_")
    if tok_a[:len(ta.split("_"))] == ta.split("_"):
        tok_a = tok_a[len(ta.split("_")):]
    if tok_b[:len(tb.split("_"))] == tb.split("_"):
        tok_b = tok_b[len(tb.split("_")):]
    category = tok_a[-1] if tok_a and tok_b and tok_a[-1] == tok_b[-1] else ""
    if category:
        tok_a, tok_b = tok_a[:-1], tok_b[:-1]
    dataset = []
    for x, y in zip(tok_a, tok_b):
        if x != y:
            break
        dataset.append(x)
    dataset = "_".join(dataset or tok_a or tok_b or ["data"])

    if ta == tb:
        ta, tb = f"{ta}_1", f"{tb}_2"
    keys_a, keys_b = set(a._key), set(b._key)

    suffix = f"_{category}" if category else ""
    prefix = f"{dataset}_{ta}_{tb}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_sssom(a[~a._key.isin(keys_b)], _header(args.sssom_a), out / f"{prefix}_only_{ta}{suffix}.tsv")
    _write_sssom(b[~b._key.isin(keys_a)], _header(args.sssom_b), out / f"{prefix}_only_{tb}{suffix}.tsv")

    # Align columns so every field is present on both sides, then merge: identity columns once,
    # every other column tagged by its tool (works for any SSSOM columns, e.g. leonmap's `comment`).
    extra = sorted((set(a.columns) | set(b.columns)) - {"_key", "mapping_tool", *_ID})
    for col in extra:
        a[col] = a[col] if col in a.columns else ""
        b[col] = b[col] if col in b.columns else ""
    m = a.merge(b, on="_key", suffixes=(f"_{ta}", f"_{tb}"))
    common = pd.DataFrame({c: m[f"{c}_{ta}"] for c in _ID})
    for col in extra:
        common[f"{col}_{ta}"], common[f"{col}_{tb}"] = m[f"{col}_{ta}"], m[f"{col}_{tb}"]
    common.to_csv(out / f"{prefix}_common{suffix}.tsv", sep="\t", index=False)

    print(f"[combine] {ta}={len(a)} {tb}={len(b)} | common={len(m)} "
          f"only_{ta}={len(keys_a - keys_b)} only_{tb}={len(keys_b - keys_a)} -> {out}/")
