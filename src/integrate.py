"""Integrate the independently cleaned FlowCast source datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import configured_path


def _ensure_datetime(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    values = pd.to_datetime(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{name} contains invalid values in {column!r}")
    return values


def _require_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame.duplicated(keys).any():
        examples = frame.loc[frame.duplicated(keys, keep=False), keys].head(4).to_dict("records")
        raise ValueError(f"{name} keys are not unique for {keys}: {examples}")


def integrate_sources(
    traffic: pd.DataFrame,
    weather: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Join weather and calendar metadata to cleaned traffic observations.

    Weather is a many-to-one join on traffic station plus floored hour.
    Calendar is a many-to-one join on the date derived from the traffic timestamp.
    """
    required_traffic = {"road_id", "timestamp", "weather_station_id"}
    required_weather = {"station_id", "timestamp"}
    required_calendar = {"date"}
    for required, frame, name in (
        (required_traffic, traffic, "traffic"),
        (required_weather, weather, "weather"),
        (required_calendar, calendar, "calendar"),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} dataset is missing required columns: {', '.join(missing)}")

    result = traffic.copy()
    result["timestamp"] = _ensure_datetime(result, "timestamp", "traffic")
    weather_frame = weather.copy()
    weather_frame["timestamp"] = _ensure_datetime(weather_frame, "timestamp", "weather")
    calendar_frame = calendar.copy()
    calendar_frame["calendar_date"] = (
        _ensure_datetime(calendar_frame, "parsed_date", "calendar")
        if "parsed_date" in calendar_frame
        else _ensure_datetime(calendar_frame, "date", "calendar")
    ).dt.normalize()

    _require_unique(result, ["road_id", "timestamp"], "traffic identity")
    result["weather_hour"] = result["timestamp"].dt.floor("h")
    weather_frame["weather_hour"] = weather_frame["timestamp"].dt.floor("h")
    if result[["weather_station_id", "weather_hour"]].isna().any().any():
        raise ValueError("traffic contains null weather station/hour join keys")
    if weather_frame[["station_id", "weather_hour"]].isna().any().any():
        raise ValueError("weather contains null station/hour join keys")
    _require_unique(weather_frame, ["station_id", "weather_hour"], "weather")

    weather_key = weather_frame[["station_id", "weather_hour"]].rename(
        columns={"station_id": "weather_station_id"}
    )
    weather_columns = [
        column
        for column in weather_frame.columns
        if column not in {"station_id", "timestamp", "weather_hour", "source_row_number"}
    ]
    weather_payload = weather_key.join(weather_frame[weather_columns].add_prefix("weather_"))
    result = result.merge(
        weather_payload,
        on=["weather_station_id", "weather_hour"],
        how="left",
        validate="many_to_one",
        indicator="_weather_merge",
    )
    weather_matched = result["_weather_merge"].eq("both")
    traffic_rows = len(result)
    weather_keys_used = set(
        zip(result.loc[weather_matched, "weather_station_id"], result.loc[weather_matched, "weather_hour"])
    )
    weather_keys = set(zip(weather_frame["station_id"], weather_frame["weather_hour"]))
    weather_not_used = len(weather_keys - weather_keys_used)
    result.drop(columns="_weather_merge", inplace=True)

    result["calendar_date"] = result["timestamp"].dt.normalize()
    if calendar_frame["calendar_date"].isna().any():
        raise ValueError("calendar contains invalid date join keys")
    _require_unique(calendar_frame, ["calendar_date"], "calendar")
    calendar_columns = [
        column
        for column in calendar_frame.columns
        if column not in {"date", "parsed_date", "calendar_date", "source_row_number"}
    ]
    calendar_payload = calendar_frame[["calendar_date"] + calendar_columns].copy()
    calendar_payload.rename(
        columns={column: f"calendar_{column}" for column in calendar_columns},
        inplace=True,
    )
    result = result.merge(
        calendar_payload,
        on="calendar_date",
        how="left",
        validate="many_to_one",
        indicator="_calendar_merge",
    )
    calendar_matched = result["_calendar_merge"].eq("both")
    result.drop(columns="_calendar_merge", inplace=True)
    _require_unique(result, ["road_id", "timestamp"], "integrated traffic identity")

    audit = {
        "traffic_rows_before_weather": len(traffic),
        "traffic_rows_after_weather": traffic_rows,
        "weather_matched_traffic_rows": int(weather_matched.sum()),
        "weather_unmatched_traffic_rows": int((~weather_matched).sum()),
        "weather_records_not_used": weather_not_used,
        "weather_records_total": len(weather_frame),
        "calendar_matched_traffic_rows": int(calendar_matched.sum()),
        "calendar_unmatched_traffic_rows": int((~calendar_matched).sum()),
        "calendar_dates_total": len(calendar_frame),
        "final_rows": len(result),
        "weather_join_row_delta": traffic_rows - len(traffic),
        "calendar_join_row_delta": len(result) - traffic_rows,
        "final_duplicate_identity_rows": int(result.duplicated(["road_id", "timestamp"]).sum()),
        "final_measurement_missing": {
            column: int(result[column].isna().sum())
            for column in ("traffic_volume", "avg_speed", "occupancy")
            if column in result.columns
        },
        "final_columns": list(result.columns),
        "final_dtypes": {column: str(dtype) for column, dtype in result.dtypes.items()},
    }
    return result.sort_values(["road_id", "timestamp"]).reset_index(drop=True), audit


def run_integration(config: dict[str, Any]) -> dict[str, Any]:
    """Read Phase 2A outputs, integrate them, and write Phase 2B artifacts."""
    interim = configured_path(config["data"]["interim_dir"])
    processed = configured_path(config["data"]["processed_dir"])
    traffic = pd.read_csv(interim / "traffic_cleaned.csv", parse_dates=["timestamp"])
    weather = pd.read_csv(interim / "weather_cleaned.csv", parse_dates=["timestamp"])
    calendar = pd.read_csv(interim / "calendar_validated.csv", parse_dates=["parsed_date"])
    final, audit = integrate_sources(traffic, weather, calendar)
    processed.mkdir(parents=True, exist_ok=True)
    final.to_csv(processed / "flowcast_clean.csv", index=False)
    (interim / "integration_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit
