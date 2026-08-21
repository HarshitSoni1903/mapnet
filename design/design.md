# Design

## Architecture

The core is a light orchestrator that always installs. Adapters are heavy, tool specific units
that each live in their own environment and are never imported by the core.

```mermaid
flowchart LR
    cli[CLI] --> data

    subgraph core[MapNet core]
        direction LR
        data --> matchers --> classify --> store
    end

    subgraph adapters[Adapters, isolated]
        direction TB
        gilda
        leonmap
        logmap
        bertmap
        ...
    end

    matchers -->|arguments| adapters
    adapters -->|predictions| matchers
    classify <--> sets[(right, wrong, novel)]
    store --> zenodo[(Zenodo)]

    classDef default fill:transparent,stroke:#888888,color:#888888
    style core fill:transparent,stroke:#888888,color:#888888
    style adapters fill:transparent,stroke:#888888,color:#888888
```

`matchers` is the only module that talks to an adapter. It appends the standard arguments to the
tool's command, spawns it, and reads back the SSSOM file it wrote.

## Rules

1. MapNet runs a command. It never resolves a package. Dependencies belong to the adapter,
   declared in its PEP 723 header and resolved by uv.
2. Base dependencies stay light. Installing `mapnet` beside a tool does not conflict with it.
3. Tools are read only. The adapter translates. MapNet never patches a tool.
4. An adapter emits every candidate it finds. Reduction happens in `classify`.

## Classes

The classes MapNet defines are `Tool` and `Mapper`. Adapters subclass `Mapper`.

```mermaid
classDiagram
    direction TB

    class Tool {
        +name str
        +command list~str~
        +wants_format str
        +config Path
    }

    class Mapper {
        +name str
        +version str
        +tool_id str
        +match(args) Iterable~SemanticMapping~
        +main(argv) int
    }

    class GildaMapper {
        +match(args)
    }

    class LeonMapMapper {
        +match(args)
    }

    Tool ..> Mapper : process boundary
    Mapper <|-- GildaMapper
    Mapper <|-- LeonMapMapper

    classDef default fill:transparent,stroke:#888888,color:#888888
```

`Tool` is a frozen dataclass, one per manifest entry, and `matchers.run` spawns it. The dashed
edge is the process boundary: no core module imports an adapter. `Mapper` is a base class, not a
plugin registry.

The output model comes from `sssom_pydantic`.

```mermaid
classDiagram
    direction LR

    class SemanticMapping {
        +subject Reference
        +predicate Reference
        +object Reference
        +justification Reference
        +confidence float
        +mapping_tool MappingTool
    }

    class MappingTool {
        +name str
        +version str
        +reference Reference
    }

    class MappingSet {
        +id AnyUrl
        +title str
    }

    SemanticMapping --> MappingTool : mapping_tool
    MappingSet ..> SemanticMapping : one header per file of rows

    classDef default fill:transparent,stroke:#888888,color:#888888
```

Only the fields MapNet sets are shown. `MappingSet` is the file header and does not hold the
rows.

## Run flow

```mermaid
flowchart LR
    req[Request] --> files[Ontology files] --> spawn[Invoke adapter]
    spawn --> preds[Predictions] --> split[Right, wrong, novel] --> out[SSSOM files]
    out --> gate{Store requested}
    gate -->|yes| push[Publish]
    gate -->|no| done[Done]
    push --> done

    classDef default fill:transparent,stroke:#888888,color:#888888
```

One run takes one tool. Several tools per run is planned.

## Classification

Planned. Reduction runs before the split. It resolves collisions in a fixed order and stops at
the first step that separates the candidates.

```mermaid
flowchart LR
    A[candidates] --> B{collision}
    B -->|no| K[keep]
    B -->|yes| E{evidence backs one}
    E -->|no| C{confidence differs}
    C -->|no| V{survives reverse run}
    V -->|no| G[conflicts]
    E -->|yes| K
    C -->|yes| K
    V -->|yes| K

    classDef default fill:transparent,stroke:#888888,color:#888888
```

Each survivor is then split against evidence.

```mermaid
flowchart LR
    K[kept] --> P{pair known}
    P -->|yes| R[right]
    P -->|no| E{subject or target known}
    E -->|yes| W[wrong]
    E -->|no| N[novel]

    classDef default fill:transparent,stroke:#888888,color:#888888
```

| Pair known | Subject known | Target known | Bucket |
| --- | --- | --- | --- |
| yes | any | any | right |
| no | yes | any | wrong |
| no | no | yes | wrong |
| no | no | no | novel |

- A collision is a subject or object claimed by more than one candidate in the same run.
- Evidence is tried before confidence. Evidence is curated, confidence is tool relative.
- The reverse run calls the same adapter with source and target swapped. A pair kept in both
  directions survives.
- A collision that reaches the end unseparated goes to `conflicts`, not to a bucket.
- Reduction applies to `skos:exactMatch` only. `narrowMatch` and `broadMatch` are many to one.
- Confidence is comparable within one tool's output, not across tools. A cross tool tie that
  evidence cannot separate falls to tool precedence.
- Evidence is undirected. A mondo to mesh prediction is checked against mesh to mondo rows.
- A wrong row carries the mapping it conflicts with.
- The resolved evidence version is stamped on the output set.

Evidence sources are named per run, with no auto selection: `biomappings`, `obo-xref`, and a
landscape such as `semra:disease`. Each resolves the same three ways as an ontology: a URL is
used as given, a pinned version is fetched by record or ref, otherwise the latest is resolved
and recorded.

## Modules

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Package facade. Declares the public API and attaches a null log handler. |
| `utils.py` | Cross cutting helpers: CURIE normalisation, ontology prefixes, run log paths. |
| `data.py` | Resolve, download and cache ontology files in the format an adapter asks for. |
| `sssom.py` | Read and write SSSOM, infer the curie map, stamp tool identity, write atomically. |
| `mapper.py` | The `Mapper` base class and the argument parser every adapter shares. |
| `matchers.py` | Read the manifest, spawn a tool as a subprocess, capture its log. |
| `classify.py` | Reduce to one to one, then split into right, wrong and novel. Planned. |
| `store.py` | Publish mapping sets to Zenodo. Planned. |
| `cli.py` | Command line surface. One registration function per subcommand. |
