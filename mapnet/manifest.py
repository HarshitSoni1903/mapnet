"""The central registry."""

from __future__ import annotations

URLS = {
    "github_tags": "https://api.github.com/repos/{repo}/tags?per_page=100",
    "obo_release": (
        "http://purl.obolibrary.org/obo/{prefix}/releases/{version}/{prefix}.{fmt}"
    ),
    "zenodo_latest": "https://zenodo.org/api/records/{concept}/versions/latest",
    "zenodo_file": "https://zenodo.org/records/{record}/files/{filename}",
}

MAPPING_SET_BASE = "https://w3id.org/mapnet/mappings"

BIOMAPPINGS = "https://raw.githubusercontent.com/biopragmatics/biomappings/main/"

# Zenodo concept record and filename. The record id is resolved at run time.
EVIDENCE_ZENODO: dict[str, tuple[int, str]] = {
    "semra:disease": (11091885, "processed.sssom.tsv.gz"),
}

# Evidence served straight from a URL, used as given.
EVIDENCE_URL: dict[str, str] = {
    "biomappings": BIOMAPPINGS + "src/biomappings/resources/positive.sssom.tsv",
    "biomappings:negative": BIOMAPPINGS
    + "src/biomappings/resources/negative.sssom.tsv",
    "biomappings:predictions": BIOMAPPINGS
    + "src/biomappings/resources/predictions.sssom.tsv",
}

TOOLS = {
    "gilda": {
        "command": ["uv", "run", "--script", "gilda_utils.py"],
        "wants_format": "obo",
    },
    "leonmap": {
        "command": ["uv", "run", "--script", "leonmap_utils.py"],
        "wants_format": "owl",
    },
}
