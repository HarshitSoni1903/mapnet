# MapNet

An aggregator for biomedical ontology mapping tools. MapNet runs matchers over a pair of
ontologies, classifies the results against known evidence, and exports them as SSSOM.

Each matcher runs in its own isolated environment. MapNet supplies the input a tool asks for and
collects its output.

## Install

```bash
pip install mapnet
```

Adapters resolve their own environments through [uv](https://docs.astral.sh/uv/), which must be
on the path to run a matcher.

## Usage

```bash
mapnet fetch mondo
mapnet fetch mondo --format owl
mapnet fetch mondo --version 2026-08-04
mapnet fetch https://example.org/my.obo
mapnet versions mondo
mapnet tools
mapnet map --tool gilda --src mondo --tgt mesh --out predictions.sssom.tsv
mapnet map --tool gilda --src https://example.org/my.obo --tgt mesh --out predictions.sssom.tsv
```

| Command | Options |
| --- | --- |
| `fetch <source>` | `--format {obo,owl}` `--version` `--redownload` `--data` |
| `versions <prefix>` | `--refresh` `--data` |
| `tools` | none |
| `evidence` | none |
| `map` | `--tool` `--src` `--tgt` `--out` `--classify` `--evidence` `--data` `--logs` |
| `aggregate <predictions...>` | `--out` |
| `classify <predictions>` | `--out` `--evidence` `--data` |

`map --classify` carries a run through the split in one command, writing the four sets beside
`--out`. Any flag `map` does not recognise is passed to the tool, which validates it, so
`--src-prefix mondo` or a tool's own `--threshold 0.8` need no change to MapNet.

`map` fetches both ontologies in the format the tool declares, runs the tool as a subprocess,
and writes SSSOM. Each run is logged to `logs/`.

A source, `--src` or `--tgt` is a [Bioregistry](https://bioregistry.io) prefix or a download
URL. Files land in `data/<prefix>/`, named after the prefix and the extension they were served
with.

`--version` names an OBO Foundry release. It applies to prefixes only, since a URL serves one
release and cannot be versioned. `fetch` prints the path, the size and the version it read from
the file.

## Python API

Everything the pipeline needs is on the facade. Nothing imports a submodule.

```python
from mapnet import classify, load_evidence, aggregate, read, write
from mapnet import get_ontology, get_evidence, get_version, list_versions
from mapnet import to_curie, to_prefix, to_reference, table
from mapnet import BUCKETS, EVIDENCE, Evidence, Mapper, Reference, SemanticMapping, Split
```

```python
rows = read(predictions)
split = classify(rows)                        # loads EVIDENCE, derives the prefixes
for bucket in BUCKETS:
    write(split.buckets[bucket], out / f"{bucket}.sssom.tsv")
```

`classify` returns a `Split`: the four `buckets`, the `evidence` behind them, the `prefixes` it
read off the rows, and `rescued`, the count of `right` rows resting on an uncurated prediction
alone. It takes evidence names, a single name, or an `Evidence` you already loaded, so one load
can judge many prediction files.

```python
path = get_evidence("semra")                  # data/evidence/semra/<record>/
count = aggregate([gilda_out, leonmap_out], combined)
```

`get_evidence` resolves a Zenodo concept record to its newest version at fetch time, falling
back to the newest already cached when Zenodo is unreachable. `aggregate` unions prediction
files, keeping the first row for each subject and object pair.

## Evidence

`classify` judges candidates against the sources named in `EVIDENCE`, and takes the same
names or file paths on `--evidence`.

| Name | Is | Effect on a candidate |
| --- | --- | --- |
| `biomappings` | curated mappings | an exact pair is right |
| `biomappings-negative` | curated rejections | an exact pair is wrong, before anything else |
| `biomappings-predicted` | uncurated predictions | an exact pair is right only where no curated set has ruled |
| `semra` | an assembled landscape | an exact pair is right |
| `obo-xref` | the xrefs of the ontologies being mapped | an exact pair is right |

An entity already mapped into the other prefix, by any curated source, makes a different
candidate for it wrong. What survives is novel, then reduced to one subject and one object.

## Configuration

`mapnet/manifest.py` is the central registry. Adding a source, an evidence set or a matcher is
an edit there.

| Name | Holds |
| --- | --- |
| `URLS` | download and API endpoints |
| `SOURCES` | every evidence set: what a match means, and where the file comes from |
| `EVIDENCE` | the subset classify actually consults |
| `TOOLS` | registered matchers and the format each one wants |

## Layout

```text
mapnet/          the core package
adapters/        one script per matcher
design/          architecture and design documents
data/            downloaded ontologies and caches
outputs/         mapping sets
logs/            per-run tool output
```

`data/`, `outputs/` and `logs/` are gitignored.

## Design

| Document | Covers |
| --- | --- |
| [design.md](design/design.md) | architecture, class and flow diagrams, classification, modules |
| [adapters.md](design/adapters.md) | the adapter contract, the manifest, and how to add a matcher |

## License

MIT
