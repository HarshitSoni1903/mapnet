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
"""Map MP to HP with LeonMap embeddings, split the result, and score it."""

from pathlib import Path

import mapnet
from adapters.leonmap_utils import LeonMapMapper

space = mapnet.MapNet(workdir=Path("."))
dataset = mapnet.Dataset(
    src="mp",
    tgt="hp",
    evidence=mapnet.EVIDENCE,
    mapnet=space,
)
result = LeonMapMapper(
    dataset=dataset,
    threshold=0.85,
    top_k=3,
    build_missing=True,
).run()
split = result.classify()

print(f"\n{result.directory}")
for name, rows in split.sets():
    print(f"  {name:10} {len(rows):6}")
