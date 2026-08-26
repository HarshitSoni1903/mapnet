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
| `map` | `--tool` `--src` `--tgt` `--out` `--data` `--logs` |

`map` fetches both ontologies in the format the tool declares, runs the tool as a subprocess,
and writes SSSOM. Each run is logged to `logs/`.

A source, `--src` or `--tgt` is a [Bioregistry](https://bioregistry.io) prefix or a download
URL. Files land in `data/<prefix>/`, named after the prefix and the extension they were served
with.

`--version` names an OBO Foundry release. It applies to prefixes only, since a URL serves one
release and cannot be versioned. `fetch` prints the path, the size and the version it read from
the file.

## Python API

```python
from mapnet import get_ontology, get_evidence, get_version, list_versions
from mapnet import read, write, aggregate
from mapnet import to_curie, to_prefix, to_reference
from mapnet import Mapper, Reference, SemanticMapping
```

Evidence sets and aggregation have no command line yet.

```python
path = get_evidence("semra:disease")          # data/evidence/semra_disease/<record>/
count = aggregate([gilda_out, leonmap_out], combined)
```

`get_evidence` resolves the Zenodo concept record to its newest version at fetch time.
`aggregate` unions prediction files, keeping the first row for each subject and object pair.

## Configuration

`mapnet/manifest.py` is the central registry. Adding a source, an evidence set or a matcher is
an edit there.

| Name | Holds |
| --- | --- |
| `URLS` | download and API endpoints |
| `EVIDENCE` | evidence sets, keyed by Zenodo concept record |
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
