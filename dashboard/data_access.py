"""Safe, cached access to FlowCast dashboard data and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.targets import TRAFFIC_VOLUME_TARGET_30M
from src.utils import configured_path, load_config

try:
    import streamlit as st
except ImportError:  # pragma: no cover - used before optional dashboard install
    st = None


def _cache_data(func):
    if st is None:
        return func
    return st.cache_data(show_spinner=False)(func)


def _default_processed_path() -> Path:
    settings = load_config()
    return configured_path(settings["data"]["processed_dir"]) / "flowcast_target_30m.csv"


@_cache_data
def load_processed_data(input_path: str | Path | None = None) -> pd.DataFrame:
    """Load historical prediction inputs without the training target column."""
    path = Path(input_path) if input_path is not None else _default_processed_path()
    if not path.is_file():
        raise FileNotFoundError(f"Processed dataset is missing: {path}")

    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"Processed dataset is missing required column: timestamp")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("Processed dataset contains invalid timestamps")
    if "road_id" not in frame.columns:
        raise ValueError("Processed dataset is missing required column: road_id")
    if frame["road_id"].isna().any():
        raise ValueError("Processed dataset contains missing road identifiers")

    result = frame.drop(columns=[TRAFFIC_VOLUME_TARGET_30M], errors="ignore")
    if TRAFFIC_VOLUME_TARGET_30M in result.columns:
        raise AssertionError("Prediction inputs must not contain the training target")
    return result


def _validate_report_schema(
    report: Any,
    *,
    path: Path,
    required_keys: Iterable[str],
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError(f"Report must contain a JSON object: {path}")
    missing = sorted(set(required_keys).difference(report))
    if missing:
        raise ValueError(
            f"Report is missing required keys ({', '.join(missing)}): {path}"
        )
    return report


@_cache_data
def load_json_report(
    report_path: str | Path,
    required_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Load and validate a JSON report with a clear schema error."""
    path = Path(report_path)
    if not path.is_file():
        raise FileNotFoundError(f"Report is missing: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON report: {path}") from exc
    return _validate_report_schema(
        report,
        path=path,
        required_keys=required_keys,
    )


def default_report_path(name: str) -> Path:
    """Resolve a report filename under the configured reports directory."""
    if not name or Path(name).name != name or Path(name).suffix.lower() != ".json":
        raise ValueError("Report name must be a JSON filename without directories")
    settings = load_config()
    return configured_path(settings["paths"]["reports_dir"]) / name
