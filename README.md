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
mapnet map --tool gilda --src mondo --tgt mesh
mapnet map --tool gilda --src https://example.org/my.obo --tgt mesh
```

| Command | Options |
| --- | --- |
| `fetch [source]` | `--format {obo,owl}` `--version` `--redownload` `--workdir` |
| `versions <prefix>` | `--refresh` `--workdir` |
| `tools` | none |
| `evidence` | none |
| `map` | `--tool` `--src` `--tgt` `--reverse` `--classify` `--gold` `--evidence` `--workdir` |
| `aggregate <predictions...>` | `--out` |
| `classify <predictions>` | `--out` `--reverse` `--evidence` `--workdir` |

`--workdir` is the one location everything is created under, the current directory by default,
so the flag can be skipped entirely. `data/`, `logs/` and `outputs/` are made inside it. Each
run gets its own directory, and the files inside carry plain names:

```bash
mapnet map --tool gilda --src icd10 --tgt mesh --reverse --classify
```

```text
outputs/gilda/icd10_mesh/20260827_143012/run.sssom.tsv          the run
outputs/gilda/icd10_mesh/20260827_143012/run_reverse.sssom.tsv  the reverse run
outputs/gilda/icd10_mesh/20260827_143012/right.sssom.tsv        the four sets
outputs/gilda/icd10_mesh/20260827_143012/wrong.sssom.tsv
outputs/gilda/icd10_mesh/20260827_143012/novel.sssom.tsv
outputs/gilda/icd10_mesh/20260827_143012/conflicts.sssom.tsv
outputs/gilda/icd10_mesh/20260827_143012/run.eval.json          when --gold is given
```

`--workdir run1/` puts the whole run under `run1/`, downloads included, so a second workdir is
a fully separate sandbox. Two matchers on the same pair land in different directories, and two
runs of the same matcher in different ones. `conflicts` holds collisions nothing could separate, kept rather than
resolved arbitrarily.

`classify` on its own writes the four sets beside the predictions unless `--out` says otherwise,
so classifying a run's file lands the sets back in that run's directory.

Reduction cascades. When a candidate wins its subject but then loses its object to a stronger
rival, the candidate it had beaten is reconsidered rather than lost with it, and this repeats
until nothing more can be settled.

`--reverse` runs the tool again with the sides swapped. A collision that confidence cannot
separate is then decided by whether the pair survives both directions, which is the last leg of
the reduction cascade. `--classify` splits the predictions once written.

Any flag `map` does not recognise is passed to the tool, which validates it, so `--src-prefix
mondo` or a tool's own `--threshold 0.8` need no change to MapNet.

`map` fetches both ontologies in the format the tool declares, runs the tool as a subprocess,
and writes SSSOM. Each run is logged to `<workdir>/logs/`, named by the stamp that names the
run directory.

A source, `--src` or `--tgt` is a [Bioregistry](https://bioregistry.io) prefix or a download
URL. Files land in `<workdir>/data/<prefix>/`, named after the prefix and the extension they
were served with.

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
from mapnet import to_curie, to_prefix, to_reference, table
from mapnet import by_prefixes, check_prefixes
from mapnet import score, mrr, hits_at_k, Scores
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

## Evaluation

`--gold` names an SSSOM file of correct mappings and is passed straight to the tool. The tool
scores itself and writes `<run>.eval.json` beside its predictions. Without `--gold` no scoring
happens at all.

```bash
mapnet map --tool gilda --src mondo --tgt mesh --gold gold/mondo_mesh.sssom.tsv
```

There is no gold file in the repository. `by_prefixes` cuts one out of any curated set, keeping
only the rows whose two sides use the pair being mapped, either way round:

```python
rows = read(Path("data/evidence/biomappings/latest/positive.sssom.tsv"))
write(by_prefixes(rows, "mondo", "mesh"), Path("gold/mondo_mesh.sssom.tsv"))
```

Check the coverage before trusting the result. Biomappings holds 323 curated mondo to mesh
pairs and none at all for icd10, so it can score some pairs and not others.

```json
{"tool": "gilda", "version": "1.6.1",
 "metrics": {"hits": 1, "predicted": 2, "expected": 2,
             "precision": 0.5, "recall": 0.5, "f1": 0.5}}
```

Scoring belongs to the adapter because only the adapter can see its own ranking. A matcher
prunes its candidates before writing SSSOM, so a rank based metric cannot be recovered from the
output file. `Mapper.evaluate` scores the pairs written, and an adapter that keeps ranked
candidates overrides it to add more:

```python
def evaluate(self, rows, gold, args):
    """Score the pairs written, and the ranking only this adapter can see."""
    metrics = super().evaluate(rows, gold, args)
    metrics["mrr"] = mrr(self.ranked, gold)
    metrics["hits_at_1"] = hits_at_k(self.ranked, gold, 1)
    return metrics
```

Every adapter uses the same functions from `mapnet.eval`, so numbers from different tools are
comparable. The eval file says what that tool could measure, so nothing needs declaring twice.

| Function | Takes | Gives |
| --- | --- | --- |
| `score(predicted, gold)` | two sets of pairs | hits, counts, precision, recall, f1 |
| `mrr(ranked, gold)` | subject to ranked objects | mean reciprocal rank |
| `hits_at_k(ranked, gold, k)` | subject to ranked objects | share correct within the top k |

Pairs are compared without direction, since an exact match holds both ways round.

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
demos/           end to end scripts over real data
design/          architecture and design documents
```

Everything a run produces goes under the workdir, the current directory by default:

```text
<workdir>/data/       downloaded ontologies and evidence
<workdir>/outputs/    one directory per run
<workdir>/logs/       per-run tool output
```

All three are gitignored at the repository root, and `--workdir` moves them together, so a
second workdir is a fully separate sandbox. A tool that caches models puts them under the
workdir too, wherever it chooses.

## Demos

`demos/` holds end to end scripts. `gilda_icd10_mesh.py` blends the enriched ICD-10 concept
table into OBO, then maps it to MeSH:

```bash
uv run --script demos/gilda_icd10_mesh.py --classify
uv run --script demos/gilda_icd10_mesh.py --workdir /tmp/run1
```

## Design

| Document | Covers |
| --- | --- |
| [design.md](design/design.md) | architecture, class and flow diagrams, classification, modules |
| [adapters.md](design/adapters.md) | the adapter contract, the manifest, and how to add a matcher |

## License

MIT
