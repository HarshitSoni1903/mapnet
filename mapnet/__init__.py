"""MapNet: an aggregator for biomedical ontology mapping tools."""

import importlib.metadata as md
import logging

from curies import Reference
from sssom_pydantic import SemanticMapping

from mapnet.classify import BUCKETS, Evidence, Split, aggregate, classify, load_evidence
from mapnet.data import downloads, get_source, get_version, list_versions
from mapnet.eval import Scores, hits_at_k, mrr, score
from mapnet.manifest import EVIDENCE, REFRESH
from mapnet.mapper import Mapper
from mapnet.sssom import by_prefixes, read, write
from mapnet.utils import check_prefixes, table, to_curie, to_prefix, to_reference

__version__ = md.version("mapnet")

__all__ = [
    "BUCKETS",
    "EVIDENCE",
    "Evidence",
    "Mapper",
    "REFRESH",
    "Reference",
    "Scores",
    "SemanticMapping",
    "Split",
    "__version__",
    "aggregate",
    "by_prefixes",
    "check_prefixes",
    "classify",
    "downloads",
    "get_source",
    "get_version",
    "hits_at_k",
    "list_versions",
    "load_evidence",
    "mrr",
    "read",
    "score",
    "table",
    "to_curie",
    "to_prefix",
    "to_reference",
    "write",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
