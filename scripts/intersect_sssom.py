# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "pandas==2.3.3",
# ]
# ///
"""Combine two mapping runs: every row of the first, plus only what the second adds.

    uv run --script intersect_sssom.py leonmap_..._novel.sssom.tsv gilda_..._novel.sssom.tsv
    uv run --script intersect_sssom.py generate_leonmap_....py generate_gilda_....py --work-dir leonmap_run

Arguments are SSSOM files, or generator scripts to run first. Mappings are joined on
(subject_id, object_id); the generators already emit canonical CURIEs.
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def _load(arg, category, passthrough, scratch):
    """Path to an SSSOM file, running the generator first when given a .py."""
    if not str(arg).endswith(".py"):
        return Path(arg)
    out_dir = scratch / Path(arg).stem
    cmd = ["uv", "run", "--script", str(arg), "--out-dir", str(out_dir)]
    help_text = subprocess.run(["uv", "run", "--script", str(arg), "--help"],
                               capture_output=True, text=True).stdout
    for flag, value in passthrough.items():
        if value is not None and flag in help_text:
            cmd += [flag, str(value)]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    hits = sorted(out_dir.rglob(f"*_{category}.sssom.tsv"))
    if not hits:
        raise SystemExit(f"{arg} produced no *_{category}.sssom.tsv")
    return hits[0]


def _tool(df, fallback):
    if "mapping_tool_id" in df.columns and len(df):
        return str(df["mapping_tool_id"].iloc[0]).strip() or fallback
    return fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("first", help="SSSOM file or generator script; contributes every row")
    ap.add_argument("second", help="SSSOM file or generator script; contributes only its unique rows")
    ap.add_argument("-c", "--category", default="novel", help="novel, right or wrong (default: novel)")
    ap.add_argument("-t", "--threshold", help="forwarded to generator scripts that accept it")
    ap.add_argument("--work-dir", help="forwarded to generator scripts that accept it")
    ap.add_argument("-o", "--out-dir", default=".", help="output directory (default: current dir)")
    args = ap.parse_args()

    passthrough = {"--threshold": args.threshold, "--work-dir": args.work_dir}
    scratch = Path(tempfile.mkdtemp(prefix="intersect_"))
    try:
        path_a = _load(args.first, args.category, passthrough, scratch)
        path_b = _load(args.second, args.category, passthrough, scratch)
        a = pd.read_csv(path_a, sep="\t").fillna("")
        b = pd.read_csv(path_b, sep="\t").fillna("")

        ta, tb = _tool(a, "tool1"), _tool(b, "tool2")
        dataset = (path_a.name.removesuffix(".sssom.tsv")
                   .removeprefix(f"{ta}_").removesuffix(f"_{args.category}"))
        stem = f"{dataset}_{args.category}"

        key = ["subject_id", "object_id"]
        keys_a = set(map(tuple, a[key].values))
        keys_b = set(map(tuple, b[key].values))
        shared = keys_a & keys_b
        in_ = lambda df, keys: df[[tuple(r) in keys for r in df[key].values]]

        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for df, name in ((a, f"{ta}_{tb}_{stem}.tsv"),                        # append 1st
                         (in_(b, keys_b - shared), f"{tb}_only_{stem}.tsv"),  # append 2nd
                         (in_(a, keys_a - shared), f"{ta}_only_{stem}.tsv"),
                         (in_(a, shared), f"{ta}_{tb}_common_{stem}.tsv")):
            df.to_csv(out / name, sep="\t", index=False)
            print(f"[out] {len(df):5d} -> {name}")

        print(f"[combine] {ta}={len(a)} {tb}={len(b)} | common={len(shared)} | "
              f"union={len(a) + len(keys_b - shared)}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
