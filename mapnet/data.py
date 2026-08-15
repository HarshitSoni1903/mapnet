"""Fetch ontology files for a tool to consume."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import bioregistry
import pystow

DATA_ROOT = Path("data")

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
    root: Path | None = None,
) -> Path:
    """Return a local path to an ontology, downloading it when absent."""
    prefix, url = _resolve(source, fmt, version)
    name = f"{prefix}_v_{version}.{fmt}" if version else f"{prefix}.{fmt}"
    path = (root or DATA_ROOT) / prefix / name
    if path.exists() and not redownload:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _download(url, path)
    except pystow.utils.DownloadError as error:
        raise ValueError(f"cannot download {name} from {url}") from error
    return path


def get_version(path: Path) -> str | None:
    """Read an ontology's version from its filename or its header."""
    stem = path.name.split(".")[0]
    if "_v_" in stem:
        return stem.split("_v_", 1)[1]
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("data-version:"):
                return _version_part(line.split(":", 1)[1].strip())
            if line.startswith("["):
                break
    return None


def _download(url: str, path: Path) -> None:
    """Download a URL, decompressing it when the source is gzipped."""
    if not url.endswith(".gz"):
        pystow.utils.download(url, path, force=True)
        return
    archive = path.with_name(f"{path.name}.gz")
    pystow.utils.download(url, archive, force=True)
    pystow.utils.gunzip(archive, path, cleanup=True)


def _version_part(value: str) -> str:
    """Strip the release path OBO headers wrap a version in."""
    parts = value.split("/")
    if parts[-1].endswith((".obo", ".owl", ".json")):
        parts.pop()
    parts = [part for part in parts if part and part != "releases"]
    return parts[-1] if parts else value


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
