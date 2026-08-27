# /// script
# requires-python = ">=3.11"
# dependencies = ["mapnet"]
#
# [tool.uv.sources]
# mapnet = { path = "..", editable = true }
# ///
"""Blend enriched ICD-10 concepts into OBO, then map them to MeSH with gilda."""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

CONCEPTS = Path("data/icd10/icd10_concepts.tsv")
ONTOLOGY = Path(tempfile.gettempdir()) / "mapnet_icd10" / "icd10.obo"
OUT = Path("outputs")


def blend(concepts: Path, out: Path) -> tuple[int, int]:
    """Write one OBO term per concept, with its inclusion terms as EXACT synonyms."""
    blocks = ["format-version: 1.4\ndata-version: 2019\nontology: icd10\n"]
    terms = synonyms = 0
    with concepts.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            label = (row["label"] or "").strip()
            if not label:
                continue
            terms += 1
            block = ["[Term]", f"id: {row['id']}", f"name: {label}"]
            for text in dict.fromkeys(t.strip() for t in row["synonyms"].split(";")):
                if text and text.lower() != label.lower():
                    block.append(f'synonym: "{text}" EXACT []')
                    synonyms += 1
            blocks.append("\n".join(block) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(blocks), encoding="utf-8")
    return terms, synonyms


if __name__ == "__main__":
    counts = blend(CONCEPTS, ONTOLOGY)
    print(f"[blend] {ONTOLOGY}: {counts[0]} terms, {counts[1]} synonyms")
    command = [sys.executable, "-m", "mapnet.cli", "map", "--tool", "gilda"]
    command += ["--src", str(ONTOLOGY), "--tgt", "mesh", "--out", str(OUT)]
    sys.exit(subprocess.run(command).returncode)
