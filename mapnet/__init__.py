"""MapNet: an aggregator for biomedical ontology mapping tools."""

import logging

from mapnet.mapper import Mapper, write

__version__ = "2.0.0.dev0"

__all__ = ["Mapper", "__version__", "write"]

logging.getLogger(__name__).addHandler(logging.NullHandler())
