"""Basic, reproducible time-based feature engineering for FlowCast."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.utils import configured_path, load_config


TIME_FEATURE_COLUMNS = (
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "is_rush_hour",
)
LAG_FEATURE_COLUMNS = (
    "traffic_volume_lag_1",
    "traffic_volume_lag_2",
)
ROLLING_FEATURE_COLUMNS = (
    "traffic_volume_rolling_mean_3",
    "traffic_volume_rolling_mean_6",
)


def _parse_rush_interval(interval: Iterable[int], index: int) -> tuple[int, int]:
    values = tuple(interval)
    if len(values) != 2:
        raise ValueError(f"Rush-hour interval {index} must contain start and end hours")
    start, end = values
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        raise ValueError(f"Rush-hour interval {index} hours must be integers")
    if not 0 <= start < end <= 24:
        raise ValueError(f"Rush-hour interval {index} must satisfy 0 <= start < end <= 24")
    return start, end


def add_time_features(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    rush_intervals: Iterable[Iterable[int]] = ((7, 10), (16, 19)),
) -> pd.DataFrame:
    """Return a copy of ``frame`` with calendar features derived from a timestamp.

    ``day_of_week`` uses the pandas convention Monday=0 through Sunday=6.
    Rush intervals are half-open hour ranges, so ``(7, 10)`` includes 07:00
    through 09:59 and excludes 10:00.
    """
    if timestamp_column not in frame.columns:
        raise ValueError(f"Required timestamp column is missing: {timestamp_column}")

    timestamp = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if timestamp.isna().any():
        invalid_rows = frame.index[timestamp.isna()].tolist()[:5]
        raise ValueError(f"Invalid timestamps found in rows: {invalid_rows}")

    intervals = tuple(_parse_rush_interval(interval, index) for index, interval in enumerate(rush_intervals))
    result = frame.copy()
    result["hour_of_day"] = timestamp.dt.hour.astype("int8")
    result["day_of_week"] = timestamp.dt.dayofweek.astype("int8")
    result["is_weekend"] = result["day_of_week"].isin((5, 6))
    result["month"] = timestamp.dt.month.astype("int8")
    result["is_rush_hour"] = pd.Series(False, index=result.index, dtype="bool")
    for start, end in intervals:
        result["is_rush_hour"] |= timestamp.dt.hour.ge(start) & timestamp.dt.hour.lt(end)
    return result


def add_traffic_volume_lags(
    frame: pd.DataFrame,
    *,
    road_column: str = "road_id",
    timestamp_column: str = "timestamp",
    volume_column: str = "traffic_volume",
) -> pd.DataFrame:
    """Add exact 30- and 60-minute historical traffic-volume features.

    Lookups are performed by road and timestamp, not by row position. Invalid
    or missing historical volumes are treated as unavailable and remain null.
    The input row order and all existing columns are preserved.
    """
    required = {road_column, timestamp_column, volume_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Required lag columns are missing: {', '.join(missing)}")

    timestamp = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if timestamp.isna().any():
        invalid_rows = frame.index[timestamp.isna()].tolist()[:5]
        raise ValueError(f"Invalid timestamps found in rows: {invalid_rows}")
    keys = pd.MultiIndex.from_arrays(
        [frame[road_column], timestamp],
        names=[road_column, timestamp_column],
    )
    if keys.duplicated().any():
        raise ValueError(f"Duplicate historical keys found for {road_column} and {timestamp_column}")

    volume = pd.to_numeric(frame[volume_column], errors="coerce")
    valid_volume = volume.where(volume.ge(0) & volume.notna())
    history = pd.Series(valid_volume.to_numpy(), index=keys, dtype="float64")
    result = frame.copy()
    for periods, column in ((1, LAG_FEATURE_COLUMNS[0]), (2, LAG_FEATURE_COLUMNS[1])):
        lookup_keys = pd.MultiIndex.from_arrays(
            [frame[road_column], timestamp - pd.Timedelta(minutes=30 * periods)],
            names=[road_column, timestamp_column],
        )
        result[column] = history.reindex(lookup_keys).to_numpy()
    return result


def add_traffic_volume_rolling_features(
    frame: pd.DataFrame,
    *,
    road_column: str = "road_id",
    timestamp_column: str = "timestamp",
    volume_column: str = "traffic_volume",
) -> pd.DataFrame:
    """Add past-only rolling means over exact earlier half-hour periods.

    The 3-period feature uses timestamps ``t - 30`` through ``t - 90`` minutes,
    and the 6-period feature uses ``t - 30`` through ``t - 180`` minutes.
    Missing timestamps and invalid historical volumes are excluded; no
    positional rows or future values are used.
    """
    required = {road_column, timestamp_column, volume_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Required rolling columns are missing: {', '.join(missing)}")

    timestamp = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if timestamp.isna().any():
        invalid_rows = frame.index[timestamp.isna()].tolist()[:5]
        raise ValueError(f"Invalid timestamps found in rows: {invalid_rows}")
    keys = pd.MultiIndex.from_arrays(
        [frame[road_column], timestamp],
        names=[road_column, timestamp_column],
    )
    if keys.duplicated().any():
        raise ValueError(f"Duplicate historical keys found for {road_column} and {timestamp_column}")

    volume = pd.to_numeric(frame[volume_column], errors="coerce")
    valid_volume = volume.where(volume.ge(0) & volume.notna())
    history = pd.Series(valid_volume.to_numpy(), index=keys, dtype="float64")
    result = frame.copy()
    for periods, column in ((3, ROLLING_FEATURE_COLUMNS[0]), (6, ROLLING_FEATURE_COLUMNS[1])):
        values = []
        for offset in range(1, periods + 1):
            lookup_keys = pd.MultiIndex.from_arrays(
                [frame[road_column], timestamp - pd.Timedelta(minutes=30 * offset)],
                names=[road_column, timestamp_column],
            )
            values.append(history.reindex(lookup_keys).to_numpy())
        result[column] = pd.DataFrame(values, index=range(1, periods + 1)).mean(
            axis=0, skipna=True
        ).where(lambda series: series.notna()).to_numpy()
    return result


def run_time_feature_pipeline(
    config: dict[str, Any] | None = None,
    *,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Create the Phase 3B time-feature dataset without modifying the source CSV."""
    settings = config or load_config()
    input_file = (
        Path(input_path)
        if input_path
        else configured_path(settings["data"]["processed_dir"]) / "flowcast_clean.csv"
    )
    output_file = (
        Path(output_path)
        if output_path
        else configured_path(settings["data"]["interim_dir"]) / "flowcast_time_features.csv"
    )
    if not input_file.is_file():
        raise FileNotFoundError(f"Processed input dataset is missing: {input_file}")

    rush_intervals = settings.get("features", {}).get("rush_hour_intervals", ((7, 10), (16, 19)))
    frame = pd.read_csv(input_file, parse_dates=["timestamp"])
    featured = add_time_features(frame, rush_intervals=rush_intervals)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(output_file, index=False)
    return output_file


def run_lag_feature_pipeline(
    config: dict[str, Any] | None = None,
    *,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Create lag features from the existing time-feature dataset."""
    settings = config or load_config()
    input_file = (
        Path(input_path)
        if input_path
        else configured_path(settings["data"]["interim_dir"]) / "flowcast_time_features.csv"
    )
    output_file = (
        Path(output_path)
        if output_path
        else configured_path(settings["data"]["interim_dir"]) / "flowcast_lag_features.csv"
    )
    if not input_file.is_file():
        raise FileNotFoundError(f"Time-feature input dataset is missing: {input_file}")

    frame = pd.read_csv(input_file, parse_dates=["timestamp"])
    featured = add_traffic_volume_lags(frame)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(output_file, index=False)
    return output_file


def run_rolling_feature_pipeline(
    config: dict[str, Any] | None = None,
    *,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Create rolling features from the existing lag-feature dataset."""
    settings = config or load_config()
    input_file = (
        Path(input_path)
        if input_path
        else configured_path(settings["data"]["interim_dir"]) / "flowcast_lag_features.csv"
    )
    output_file = (
        Path(output_path)
        if output_path
        else configured_path(settings["data"]["interim_dir"]) / "flowcast_rolling_features.csv"
    )
    if not input_file.is_file():
        raise FileNotFoundError(f"Lag-feature input dataset is missing: {input_file}")

    frame = pd.read_csv(input_file, parse_dates=["timestamp"])
    featured = add_traffic_volume_rolling_features(frame)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(output_file, index=False)
    return output_file


def main() -> None:
    output = run_time_feature_pipeline()
    print(f"FlowCast time features written to {output}")


if __name__ == "__main__":
    main()
