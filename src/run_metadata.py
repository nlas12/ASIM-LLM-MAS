"""
Write a reproducibility snapshot (git state, config, package versions) into
each experiment run folder so results can be correlated with the code that
produced them.
"""

import json
import os
import platform
import subprocess
from datetime import datetime
from importlib import metadata as _im

import config

_TRACKED_PACKAGES = ("langchain", "langchain-openai", "langgraph", "pandas", "pydantic", "numpy")


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = _im.version(name)
        except _im.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def write_run_metadata(run_folder: str, **extra) -> None:
    """Write ``run_metadata.json`` into ``run_folder``.

    ``extra`` holds run-specific fields (mode, persona, coordination, n_runs, ...).
    Any non-JSON-serializable values are coerced via ``str``.
    """
    os.makedirs(run_folder, exist_ok=True)
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": commit or "unknown",
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "config": config.snapshot(),
        "run": extra,
    }
    path = os.path.join(run_folder, "run_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
