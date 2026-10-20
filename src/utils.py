"""Small reusable helpers for the FlowCast project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root based on this module's location."""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration, resolving relative paths from the project root."""
    path = Path(config_path) if config_path else project_root() / "config.yaml"
    if not path.is_absolute():
        path = project_root() / path
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def configured_path(config_value: str | Path) -> Path:
    """Resolve a configured project-relative path without hard-coding locations."""
    path = Path(config_value)
    return path if path.is_absolute() else project_root() / path
