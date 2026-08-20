"""Tool registry and isolated run orchestration."""

from __future__ import annotations

import subprocess
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mapnet.utils import LOG_ROOT, run_log

MANIFEST = Path(__file__).parent.parent / "adapters" / "manifest.toml"


@dataclass(frozen=True)
class Tool:
    """One registered matcher and how to run it."""

    name: str
    command: list[str]
    wants_format: str
    config: Path | None


def load_tools(manifests: Sequence[Path] | None = None) -> dict[str, Tool]:
    """Read the manifests in order, later entries winning by tool name."""
    tools: dict[str, Tool] = {}
    for manifest in manifests or [MANIFEST]:
        if not manifest.is_file():
            continue
        entries = tomllib.loads(manifest.read_text(encoding="utf-8"))
        for name, entry in entries.items():
            tools[name] = _tool(name, entry, manifest.parent)
    return tools


def run(
    tool: Tool, source: Path, target: Path, out: Path, logs: Path = LOG_ROOT
) -> Path:
    """Run a tool over two ontologies and return the predictions it wrote."""
    log = run_log(tool.name, source, target, logs)
    command = [*tool.command, "--source", str(source), "--target", str(target)]
    command += ["--out", str(out), "--logs", str(logs)]
    if tool.config:
        command += ["--config", str(tool.config)]
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, text=True
        )
    if result.returncode != 0:
        raise RuntimeError(f"{tool.name} failed, see {log}: {_tail(log)}")
    if not out.is_file():
        raise RuntimeError(f"{tool.name} wrote no predictions at {out}, see {log}")
    return out


def _tail(log: Path) -> str:
    """Read the last non-empty line of a log."""
    lines = [line.strip() for line in log.read_text(encoding="utf-8").splitlines()]
    return next((line for line in reversed(lines) if line), "no output")


def _tool(name: str, entry: dict, directory: Path) -> Tool:
    """Build one registry entry, resolving its paths against its manifest."""
    if "command" not in entry or "wants_format" not in entry:
        raise ValueError(f"{name!r} needs both command and wants_format")
    config = entry.get("config")
    return Tool(
        name=name,
        command=[_resolve(part, directory) for part in entry["command"]],
        wants_format=entry["wants_format"],
        config=directory / config if config else None,
    )


def _resolve(part: str, directory: Path) -> str:
    """Make a command part absolute when it names a file beside the manifest."""
    candidate = directory / part
    return str(candidate) if candidate.is_file() else part
