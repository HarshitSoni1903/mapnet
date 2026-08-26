# Adapters

An adapter is a script that runs one matcher. It lives outside the importable `mapnet` package
and runs as a subprocess.

## Arguments

MapNet appends these to the command from the manifest:

| Argument | Value |
| --- | --- |
| `--source` | ontology file to map from |
| `--target` | ontology file to map to |
| `--out` | path to write the SSSOM file |
| `--logs` | directory for the tool's own logs |
| `--config` | the manifest's `config` path, appended only when the manifest declares one |

- `--source` and `--target` are local files already downloaded in the format the manifest names.
- `--logs` defaults to `logs` when the adapter is run by hand.
- Any flag `mapnet map` does not recognise is appended verbatim, so `--src-prefix` and a tool's
  own options reach the adapter without MapNet declaring them. The adapter validates them.
- Thresholds and model settings are not arguments. They go in the file `--config` points at,
  which MapNet passes through without reading.

## Output

The deliverable is one SSSOM TSV file at `--out`.

- The adapter process writes every row to `<out>.tmp`.
- The adapter process then renames `<out>.tmp` onto `<out>` with `os.replace`.
- `mapnet.sssom.write` performs both steps, so an adapter that calls it inherits them.
- Nothing partial ever appears at `--out`.

The core checks two things after the subprocess ends:

- A non zero exit code is a failure.
- A missing file at `--out` is a failure.

Both raise with the last non empty line of the run log.

## Manifest

`mapnet/manifest.py` is the registry. Its `TOOLS` entry registers a matcher.

```python
TOOLS = {
    "gilda": {
        "command": ["uv", "run", "--script", "gilda_utils.py"],
        "wants_format": "obo",
    },
}
```

| Field | Value |
| --- | --- |
| `command` | argv list. A part naming a file in `adapters/` is made absolute. |
| `wants_format` | `obo` or `owl`. The format MapNet downloads and passes. |
| `config` | Optional path to the tool's config file, resolved against `adapters/`. |

- `command` and `wants_format` are required. A missing one raises on load.
- `config` is optional. No tool declares one.
- A containerised tool is `["docker", "run", ...]`.
- Dependencies go in the adapter's PEP 723 header, never in the manifest.

## Writing one

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["mapnet", "gilda", "obonet", "indra"]
#
# [tool.uv.sources]
# mapnet = { path = "..", editable = true }
# ///
"""Match two ontologies on Gilda-normalized labels."""

import importlib.metadata as md

from mapnet import Mapper


class GildaMapper(Mapper):
    name = "gilda"
    version = md.version("gilda")

    def match(self, args):
        """Yield a SemanticMapping for each candidate pair."""
        ...


if __name__ == "__main__":
    raise SystemExit(GildaMapper.main())
```

`Mapper.main` parses the arguments, reads the version of both ontology files, and calls
`sssom.write` with the class attributes `name`, `version` and `tool_id`.

- Subclass `Mapper` and implement `match`, which yields `SemanticMapping` objects.
- Set `name` and `version` as class attributes. `tool_id` is optional and takes a CURIE.
- Call `sssom.write` directly when the control flow does not fit `Mapper.main`.
- Name the file `<tool>_utils.py`. A module named after the tool shadows the package on import.
- Emit every candidate found. Reduction to one to one happens in `classify`.
- Keep confidence comparable within one adapter's output.

The `[tool.uv.sources]` block is a local override while `mapnet` is unpublished. It is removed
on release.

## Field ownership

| Column | Filled by |
| --- | --- |
| subject_id, object_id, and their labels | adapter |
| predicate_id | adapter |
| mapping_justification | adapter |
| confidence | adapter |
| mapping_tool, mapping_tool_version, mapping_tool_id | adapter |
| subject_source_version, object_source_version | core |
| mapping_date | core |
| curie map header | core |

The curie map covers the prefixes the rows use, resolved through bioregistry. A row that already
carries a core column keeps its own value.
