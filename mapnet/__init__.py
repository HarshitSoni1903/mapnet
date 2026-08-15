"""MapNet: an aggregator for biomedical ontology mapping tools."""

import logging

from mapnet.data import get_ontology, get_version, list_versions
from mapnet.mapper import Mapper
from mapnet.sssom import write
from mapnet.utils import to_curie, to_reference

__version__ = "2.0.0.dev0"

__all__ = [
    "Mapper",
    "__version__",
    "get_ontology",
    "get_version",
    "list_versions",
    "to_curie",
    "to_reference",
    "write",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
