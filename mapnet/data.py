"""Fetch ontology files for a tool to consume."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import bioregistry
import pystow

RELEASE_URL = (
    "http://purl.obolibrary.org/obo/{prefix}/releases/{version}/{prefix}.{fmt}"
)

DOWNLOADERS = {
    "obo": bioregistry.get_obo_download,
    "owl": bioregistry.get_owl_download,
}


def get_ontology(
    source: str,
    fmt: str = "obo",
    version: str | None = None,
    redownload: bool = False,
) -> Path:
    """Return a local path to an ontology, downloading it when absent."""
    prefix, url = _resolve(source, fmt, version)
    name = f"{prefix}_v_{version}.{fmt}" if version else f"{prefix}.{fmt}"
    try:
        return pystow.ensure(
            "mapnet", "data", prefix, url=url, name=name, force=redownload
        )
    except pystow.utils.DownloadError as error:
        raise ValueError(f"cannot download {name} from {url}") from error


def _resolve(source: str, fmt: str, version: str | None) -> tuple[str, str]:
    """Return the cache key and download URL for a prefix or a URL."""
    if _is_url(source):
        return Path(urlparse(source).path).name.split(".")[0], source
    if fmt not in DOWNLOADERS:
        raise ValueError(f"unknown format {fmt!r}, expected {sorted(DOWNLOADERS)}")
    if version:
        return source, _release_url(source, version, fmt)
    url = DOWNLOADERS[fmt](source)
    if url is None:
        raise ValueError(f"bioregistry has no {fmt} download for {source!r}")
    return source, url


def _release_url(prefix: str, version: str, fmt: str) -> str:
    """Build the OBO Foundry release URL for a pinned version."""
    if bioregistry.get_obofoundry_prefix(prefix) is None:
        raise ValueError(f"{prefix!r} is not an OBO Foundry ontology, pass a URL")
    return RELEASE_URL.format(prefix=prefix, version=version, fmt=fmt)


def _is_url(source: str) -> bool:
    """Whether the source is a URL rather than a prefix."""
    return source.startswith(("http://", "https://"))
