"""MapNet: an aggregator for biomedical ontology mapping tools.

The package facade. All public names are exported from here.
"""

import logging

__version__ = "2.0.0.dev0"

__all__ = ["__version__"]

logging.getLogger(__name__).addHandler(logging.NullHandler())
