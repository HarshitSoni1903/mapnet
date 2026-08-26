"""MapNet: an aggregator for biomedical ontology mapping tools."""

import importlib.metadata as md
import logging

from curies import Reference
from sssom_pydantic import SemanticMapping

from mapnet.classify import BUCKETS, Evidence, Split, aggregate, classify, load_evidence
from mapnet.data import get_evidence, get_ontology, get_version, list_versions
from mapnet.manifest import EVIDENCE
from mapnet.mapper import Mapper
from mapnet.sssom import read, write
from mapnet.utils import check_prefixes, table, to_curie, to_prefix, to_reference

__version__ = md.version("mapnet")

__all__ = [
    "BUCKETS",
    "EVIDENCE",
    "Evidence",
    "Mapper",
    "Reference",
    "SemanticMapping",
    "Split",
    "__version__",
    "aggregate",
    "check_prefixes",
    "classify",
    "get_evidence",
    "get_ontology",
    "get_version",
    "list_versions",
    "load_evidence",
    "read",
    "table",
    "to_curie",
    "to_prefix",
    "to_reference",
    "write",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
