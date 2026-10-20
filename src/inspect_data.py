"""Generate a reproducible, read-only report about the raw FlowCast CSV files."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

if __package__:
    from src.utils import configured_path, load_config
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import configured_path, load_config


DATASETS = ("traffic", "weather", "calendar")


def _to_markdown(frame: pd.DataFrame, index: bool = False) -> str:
    """Render a small table without requiring an optional reporting package."""
    values = frame.reset_index() if index else frame.copy()
    headers = [str(column) for column in values.columns]
    rows = [[str(value) for value in row] for row in values.astype(object).itertuples(index=False)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _dataset_paths(config: dict[str, Any]) -> dict[str, Path]:
    raw_dir = configured_path(config["data"]["raw_dir"])
    return {
        "traffic": raw_dir / config["data"]["traffic_file"],
        "weather": raw_dir / config["data"]["weather_file"],
        "calendar": raw_dir / config["data"]["calendar_file"],
    }


def _read_dataset(name: str, path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required raw dataset is missing: {path}")
    return pd.read_csv(path)


def _missing_table(frame: pd.DataFrame) -> str:
    missing = frame.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return "None"
    return "\n".join(f"- `{column}`: {count:,}" for column, count in missing.items())


def _numeric_summary(frame: pd.DataFrame) -> str:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return "No numeric columns."
    return _to_markdown(numeric.describe().transpose().round(3), index=True)


def _date_range(frame: pd.DataFrame, name: str) -> str:
    if "date" not in frame.columns:
        return "No date column."
    date_format = "%d/%m/%Y" if name == "weather" else "%Y-%m-%d"
    parsed = pd.to_datetime(frame["date"], format=date_format, errors="coerce")
    if "time" in frame.columns:
        parsed = pd.to_datetime(
            frame["date"].astype(str) + " " + frame["time"].astype(str),
            format=f"{date_format} %H:%M",
            errors="coerce",
        )
    valid = parsed.dropna()
    if valid.empty:
        return "No parseable date/time values."
    return f"{valid.min():%Y-%m-%d %H:%M} to {valid.max():%Y-%m-%d %H:%M}"


def _sample_table(frame: pd.DataFrame) -> str:
    return _to_markdown(frame.head(3))


def _dataset_section(name: str, frame: pd.DataFrame) -> str:
    columns = ", ".join(f"`{column}` ({dtype})" for column, dtype in frame.dtypes.items())
    return "\n".join(
        [
            f"## {name.title()} dataset",
            f"- **Shape:** {len(frame):,} rows × {len(frame.columns)} columns.",
            f"- **Columns and dtypes:** {columns}.",
            f"- **Duplicate rows:** {frame.duplicated().sum():,}.",
            f"- **Date/time coverage:** {_date_range(frame, name)}.",
            "- **Missing values:**",
            _missing_table(frame),
            "- **Sample records:**",
            _sample_table(frame),
            "- **Numerical summary:**",
            _numeric_summary(frame),
            "",
        ]
    )


def _quality_observations(traffic: pd.DataFrame, weather: pd.DataFrame) -> list[str]:
    expected_windows = (
        traffic["road_id"].nunique()
        * traffic["date"].nunique()
        * 48
    )
    distinct_keys = traffic[["road_id", "date", "time"]].drop_duplicates().shape[0]
    weather_labels = sorted(weather["weather_condition"].dropna().astype(str).unique())
    return [
        f"- Traffic has {traffic['traffic_volume'].lt(0).sum():,} negative `traffic_volume` values.",
        f"- Traffic has {traffic['avg_speed'].gt(200).sum():,} `avg_speed` values above 200 km/h.",
        f"- Traffic has {traffic['occupancy'].gt(100).sum():,} `occupancy` values above 100%.",
        f"- Traffic has {traffic['congestion_level'].isna().sum():,} blank `congestion_level` values.",
        f"- Traffic has {traffic.duplicated(['road_id', 'date', 'time']).sum():,} duplicate segment/date/time keys.",
        f"- The observed traffic keys cover {distinct_keys:,} of {expected_windows:,} possible road/date/half-hour combinations "
        "before accounting for duplicate records.",
        f"- Raw weather condition labels observed ({len(weather_labels):,}): "
        + ", ".join(f"`{label}`" for label in weather_labels)
        + ".",
    ]


def build_report(config: dict[str, Any]) -> str:
    frames = {
        name: _read_dataset(name, path)
        for name, path in _dataset_paths(config).items()
    }
    sections = [
        "# FlowCast Data Understanding",
        "",
        f"_Generated on {datetime.now():%Y-%m-%d %H:%M} by `src/inspect_data.py`._",
        "",
        "This report is descriptive only. Raw CSV files were read without cleaning, merging, "
        "label changes, imputation, outlier removal, feature derivation, or modelling.",
        "",
    ]
    for name in DATASETS:
        sections.append(_dataset_section(name, frames[name]))

    traffic = frames["traffic"]
    weather = frames["weather"]
    calendar = frames["calendar"]
    road_count = traffic["road_id"].nunique()
    station_count = weather["station_id"].nunique()
    traffic_days = traffic["date"].nunique()
    weather_days = weather["date"].nunique()
    sections.extend(
        [
            "## Observed relationships and Phase 2 questions",
            f"- Traffic contains {road_count:,} distinct road IDs and {traffic_days:,} distinct date strings.",
            f"- Weather contains {station_count:,} distinct station IDs and {weather_days:,} distinct date strings.",
            "- The traffic table is the documented modelling grain: one road segment per 30-minute window.",
            "- The weather table is hourly by station; the Data Dictionary states that both traffic windows in an hour share that hour's weather.",
            "- Calendar records are date-level and are intended to join to traffic by date.",
            "- The raw traffic date format is YYYY-MM-DD; weather uses DD/MM/YYYY. This is an observed schema difference, not corrected here.",
            "- `vehicle_type_dist` is stored as a JSON string and has not been expanded.",
            "- Phase 2 must investigate missing windows versus null readings, duplicate keys, invalid numeric values, blank congestion labels, inconsistent weather labels, and join coverage.",
            "- Phase 2 must define and document the timestamp alignment and validation rules before cleaning or merging.",
            "",
            "## Observed data-quality facts",
            *_quality_observations(traffic, weather),
            "",
            "## Expected issues documented by the specifications",
            "The Data Dictionary and PRD describe expected defects including traffic missing readings, duplicate detector retries, negative traffic volume, impossible speeds, occupancy above 100%, blank congestion labels, inconsistent weather labels, mixed date formats, and the nested vehicle-type field. These are listed as questions for Phase 2; this report does not claim that each defect was quantified or repaired.",
            "",
        ]
    )
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional path to config.yaml.")
    args = parser.parse_args()
    config = load_config(args.config)
    report_path = configured_path(config["paths"]["reports_dir"]) / "data_understanding.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(config), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
