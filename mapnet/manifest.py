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

# Zenodo concept record and filename. The record id is resolved at run time.
EVIDENCE: dict[str, tuple[int, str]] = {
    "semra:disease": (11091885, "processed.sssom.tsv.gz"),
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
