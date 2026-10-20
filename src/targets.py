"""Supervised target construction for FlowCast."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import configured_path, load_config


TRAFFIC_VOLUME_TARGET_30M = "traffic_volume_target_30m"


def add_traffic_volume_target_30m(
    frame: pd.DataFrame,
    *,
    road_column: str = "road_id",
    timestamp_column: str = "timestamp",
    volume_column: str = "traffic_volume",
    target_column: str = TRAFFIC_VOLUME_TARGET_30M,
) -> pd.DataFrame:
    """Add the valid traffic volume exactly 30 minutes in the future.

    Matching is performed by the pair ``(road_column, timestamp_column)``.
    Missing, nonnumeric, and negative future volumes remain missing. The input
    frame is not modified, and no rows are added, removed, or reordered.
    """
    required = {road_column, timestamp_column, volume_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Required target columns are missing: {', '.join(missing)}")
    if target_column in frame.columns:
        raise ValueError(f"Target column already exists: {target_column}")

    timestamp = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if timestamp.isna().any():
        invalid_rows = frame.index[timestamp.isna()].tolist()[:5]
        raise ValueError(f"Invalid timestamps found in rows: {invalid_rows}")

    keys = pd.MultiIndex.from_arrays(
        [frame[road_column], timestamp],
        names=[road_column, timestamp_column],
    )
    if keys.duplicated().any():
        raise ValueError(f"Duplicate target keys found for {road_column} and {timestamp_column}")

    volume = pd.to_numeric(frame[volume_column], errors="coerce")
    valid_volume = volume.where(volume.ge(0) & volume.notna())
    future_values = pd.Series(valid_volume.to_numpy(), index=keys, dtype="float64")
    future_keys = pd.MultiIndex.from_arrays(
        [frame[road_column], timestamp + pd.Timedelta(minutes=30)],
        names=[road_column, timestamp_column],
    )

    result = frame.copy()
    result[target_column] = future_values.reindex(future_keys).to_numpy()
    return result


def run_traffic_volume_target_pipeline(
    config: dict[str, Any] | None = None,
    *,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Create the target-enriched dataset from the rolling-feature dataset."""
    settings = config or load_config()
    input_file = (
        Path(input_path)
        if input_path
        else configured_path(settings["data"]["interim_dir"]) / "flowcast_rolling_features.csv"
    )
    output_file = (
        Path(output_path)
        if output_path
        else configured_path(settings["data"]["processed_dir"]) / "flowcast_target_30m.csv"
    )
    if not input_file.is_file():
        raise FileNotFoundError(f"Rolling-feature input dataset is missing: {input_file}")

    frame = pd.read_csv(input_file, parse_dates=["timestamp"])
    targeted = add_traffic_volume_target_30m(frame)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    targeted.to_csv(output_file, index=False)
    return output_file


def main() -> None:
    output = run_traffic_volume_target_pipeline()
    print(f"FlowCast 30-minute target written to {output}")


if __name__ == "__main__":
    main()
