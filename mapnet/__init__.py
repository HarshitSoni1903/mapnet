"""MapNet: an aggregator for biomedical ontology mapping tools."""

import logging

from curies import Reference
from sssom_pydantic import SemanticMapping

from mapnet.classify import aggregate
from mapnet.data import get_evidence, get_ontology, get_version, list_versions
from mapnet.mapper import Mapper
from mapnet.sssom import read, write
from mapnet.utils import check_prefixes, to_curie, to_prefix, to_reference

__version__ = "2.0.0.dev0"

__all__ = [
    "Mapper",
    "Reference",
    "SemanticMapping",
    "__version__",
    "aggregate",
    "check_prefixes",
    "get_evidence",
    "get_ontology",
    "get_version",
    "list_versions",
    "read",
    "to_curie",
    "to_prefix",
    "to_reference",
    "write",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
