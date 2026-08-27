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
mapnet fetch biomappings
mapnet fetch
mapnet versions mondo
mapnet tools
mapnet map --tool gilda --src mondo --tgt mesh --out results/
mapnet map --tool gilda --src https://example.org/my.obo --tgt mesh --out results/
```

| Command | Options |
| --- | --- |
| `fetch [source]` | `--format {obo,owl}` `--version` `--redownload` `--data` |
| `versions <prefix>` | `--refresh` `--data` |
| `tools` | none |
| `evidence` | none |
| `map` | `--tool` `--src` `--tgt` `--out` `--reverse` `--classify` `--evidence` `--data` `--logs` |
| `aggregate <predictions...>` | `--out` |
| `classify <predictions>` | `--out` `--reverse` `--evidence` `--data` |

`--out` is a folder. `map` names every file from the tool and the prefixes the ontologies
declare, so one command leaves a complete run in one place:

```bash
mapnet map --tool gilda --src icd10.obo --tgt mesh --out results/ --reverse --classify
```

```text
results/gilda_icd10_mesh.sssom.tsv            the run
results/gilda_mesh_icd10.sssom.tsv            the reverse run
results/gilda_icd10_mesh_right.sssom.tsv      the four sets
results/gilda_icd10_mesh_wrong.sssom.tsv
results/gilda_icd10_mesh_novel.sssom.tsv
results/gilda_icd10_mesh_conflicts.sssom.tsv
```

The tool leads the name so two matchers on the same pair do not overwrite each other.
`conflicts` holds collisions nothing could separate, kept rather than resolved arbitrarily.

Reduction cascades. When a candidate wins its subject but then loses its object to a stronger
rival, the candidate it had beaten is reconsidered rather than lost with it, and this repeats
until nothing more can be settled.

`--reverse` runs the tool again with the sides swapped. A collision that confidence cannot
separate is then decided by whether the pair survives both directions, which is the last leg of
the reduction cascade. `--classify` splits the predictions once written.

Any flag `map` does not recognise is passed to the tool, which validates it, so `--src-prefix
mondo` or a tool's own `--threshold 0.8` need no change to MapNet.

`map` fetches both ontologies in the format the tool declares, runs the tool as a subprocess,
and writes SSSOM. Each run is logged to `logs/`.

A source, `--src` or `--tgt` is a [Bioregistry](https://bioregistry.io) prefix or a download
URL. Files land in `data/<prefix>/`, named after the prefix and the extension they were served
with.

`--version` names an OBO Foundry release. It applies to prefixes only, since a URL serves one
release and cannot be versioned.

## Fetching and refreshing

`fetch` is the only thing that downloads. It takes an ontology prefix, a URL, or an evidence
name, so everything a run needs lands under `data/` and nothing is pulled from elsewhere at
classification time.

```bash
mapnet fetch mondo                 # an ontology
mapnet fetch biomappings           # an evidence set
mapnet fetch                       # refresh every volatile source
```

Nothing ever refreshes on its own. `REFRESH` in the manifest names the sources that change
upstream, and a bare `mapnet fetch` refetches exactly those. Everything else stays pinned to
whatever is already on disk, so a run is reproducible until you ask for it not to be.

Every download is recorded in `data/downloads.json` with its URL, sha256, size and date, and
`fetch` reports what actually changed:

```text
biomappings            changed      2.1 MB  version latest  data/evidence/biomappings/latest/positive.sssom.tsv
biomappings-negative   unchanged    0.3 MB  version latest  data/evidence/biomappings-negative/latest/negative.sssom.tsv
semra                  changed    954.0 MB  version 21935586  data/evidence/semra/21935586/processed.sssom.tsv
```

A refresh is not small. Semra alone is about 950 MB, and the biomappings predictions about
21 MB. Since `data/` holds every downloaded file, version control it there if you need a run
pinned to an exact snapshot.

## Python API

Everything the pipeline needs is on the facade. Nothing imports a submodule.

```python
from mapnet import classify, load_evidence, aggregate, read, write
from mapnet import get_source, get_version, list_versions, downloads
from mapnet import to_curie, to_prefix, to_reference, table, check_prefixes
from mapnet import BUCKETS, EVIDENCE, REFRESH, Evidence, Mapper, Reference, Split
from mapnet import SemanticMapping
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
path = get_source("semra")                    # data/evidence/semra/<record>/
path = get_source("mondo", fmt="owl")         # data/mondo/mondo.owl
count = aggregate([gilda_out, leonmap_out], combined)
```

`get_source` is the one door: an evidence name, a prefix, or a URL. A Zenodo concept resolves
to the newest record already cached, and only asks Zenodo when you pass `redownload=True`, so
classification never depends on the network. `downloads()` reads the index of every file
fetched. `aggregate` unions prediction files, keeping the first row for each subject and
object pair.

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

A file path is asserted evidence by default. Tag it to say otherwise:

```bash
mapnet classify preds.sssom.tsv --out sets/ --evidence biomappings,rejected:mine.sssom.tsv
```

`pairs:`, `rejected:` and `predicted:` work on registered names too, so `rejected:biomappings`
reads the curated positives as rejections. `classify` reports how many pairs each source
actually contributed and warns on stderr about any that contributed none, which is the fastest
way to notice a source that is costing a download and earning nothing.

Because predictions are consulted last, they only ever move a candidate that no curated source
has ruled on. Moving one out of `novel` also removes it from the collision groups, so enabling
predictions changes which of the remaining candidates survive reduction, not just their labels.

## Configuration

`mapnet/manifest.py` is the central registry. Adding a source, an evidence set, a matcher or a
sink is an edit there. A local path is never one of them: those come from pystow.

| Name | Holds |
| --- | --- |
| `URLS` | download and API endpoints |
| `SOURCES` | every evidence set: what a match means, and where the file comes from |
| `EVIDENCE` | the subset classify actually consults |
| `REFRESH` | the sources a bare `mapnet fetch` refetches |
| `DEPOSITION` | the Zenodo concept the run's own mapping sets are published under |
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
