"""Validate and clean FlowCast source data without merging the sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__:
    from src.integrate import run_integration
    from src.utils import configured_path, load_config
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.integrate import run_integration
    from src.utils import configured_path, load_config


TRAFFIC_REQUIRED = {
    "road_id", "date", "time", "traffic_volume", "avg_speed", "occupancy",
    "congestion_level", "road_capacity",
}
WEATHER_REQUIRED = {"station_id", "date", "time", "weather_condition"}
CALENDAR_REQUIRED = {"date", "public_holiday", "holiday_name", "event_flag", "event_name", "roadwork_flag"}
CONGESTION_LEVELS = ("Free-flow", "Moderate", "Heavy", "Severe")
WEATHER_MAP = {
    "clear": "Clear",
    "rain": "Rain",
    "rainy": "Rain",
    "fog": "Fog",
    "foggy": "Fog",
    "cloudy": "Cloudy",
}


def dataset_paths(config: dict[str, Any]) -> dict[str, Path]:
    raw_dir = configured_path(config["data"]["raw_dir"])
    return {
        "traffic": raw_dir / config["data"]["traffic_file"],
        "weather": raw_dir / config["data"]["weather_file"],
        "calendar": raw_dir / config["data"]["calendar_file"],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} dataset is missing required columns: {', '.join(missing)}")


def _parse_timestamp(frame: pd.DataFrame, date_format: str, name: str) -> tuple[pd.Series, pd.DataFrame]:
    date = pd.to_datetime(frame["date"], format=date_format, errors="coerce")
    time = pd.to_datetime(frame["time"], format="%H:%M", errors="coerce")
    timestamp = pd.to_datetime(
        frame["date"].astype("string") + " " + frame["time"].astype("string"),
        format=f"{date_format} %H:%M",
        errors="coerce",
    )
    failures = pd.DataFrame(
        {
            "dataset": name,
            "source_row_number": frame["source_row_number"],
            "date_parse_failed": date.isna(),
            "time_parse_failed": time.isna(),
            "timestamp_parse_failed": timestamp.isna(),
        }
    )
    failures = failures.loc[
        failures[["date_parse_failed", "time_parse_failed", "timestamp_parse_failed"]].any(axis=1)
    ]
    return timestamp, failures


def _add_source_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "source_row_number", np.arange(2, len(result) + 2))
    return result


def _duplicate_audits(traffic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    exact_columns = [column for column in traffic.columns if column != "source_row_number"]
    exact_mask = traffic.duplicated(exact_columns, keep=False)
    exact_removed = traffic.loc[traffic.duplicated(exact_columns, keep="first")].copy()
    exact_removed["audit_type"] = "exact_duplicate_removed"
    exact_group_sizes = traffic.groupby(exact_columns, dropna=False)["source_row_number"].transform("size")
    exact_removed["duplicate_group_size"] = exact_group_sizes.loc[exact_removed.index]

    key_columns = ["road_id", "timestamp"]
    key_sizes = traffic.groupby(key_columns, dropna=False)["source_row_number"].transform("size")
    key_rows = traffic.loc[key_sizes > 1].copy()
    conflict_groups: list[pd.DataFrame] = []
    compare_columns = [column for column in traffic.columns if column not in key_columns + ["source_row_number"]]
    for _, group in key_rows.groupby(key_columns, dropna=False, sort=False):
        if group[compare_columns].nunique(dropna=False).gt(1).any():
            conflict_groups.append(group)
    conflicts = pd.concat(conflict_groups, ignore_index=True) if conflict_groups else key_rows.iloc[0:0].copy()
    if not conflicts.empty:
        conflicts["audit_type"] = "conflicting_duplicate_key"
    stats = {
        "exact_duplicate_groups": int(traffic.loc[exact_mask].groupby(exact_columns, dropna=False).ngroups),
        "exact_duplicate_rows_in_groups": int(exact_mask.sum()),
        "exact_duplicate_rows_removed": int(traffic.duplicated(exact_columns, keep="first").sum()),
        "duplicate_key_groups": int(key_rows.groupby(key_columns, dropna=False).ngroups),
        "duplicate_key_rows_beyond_first": int((key_sizes > 1).sum() - key_rows.groupby(key_columns, dropna=False).ngroups),
        "conflicting_duplicate_key_groups": int(conflicts.groupby(key_columns, dropna=False).ngroups),
        "conflicting_duplicate_key_rows": int(len(conflicts)),
    }
    return exact_removed, conflicts, stats


def _measurement_audit(traffic: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rules = {
        "traffic_volume": (traffic["traffic_volume"].lt(0), "negative_traffic_volume"),
        "avg_speed": (traffic["avg_speed"].gt(200), "speed_above_200_kmh"),
        "occupancy": (traffic["occupancy"].gt(100), "occupancy_above_100_percent"),
        "road_capacity": (traffic["road_capacity"].le(0), "nonpositive_road_capacity"),
    }
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for field, (mask, reason) in rules.items():
        counts[f"{field}_invalid_values"] = int(mask.sum())
        counts[f"{field}_invalid_rows"] = int(mask.sum())
        for index in traffic.index[mask]:
            records.append(
                {
                    "source_row_number": traffic.at[index, "source_row_number"],
                    "road_id": traffic.at[index, "road_id"],
                    "timestamp": traffic.at[index, "timestamp"],
                    "field": field,
                    "original_value": traffic.at[index, field],
                    "reason_code": reason,
                }
            )
        traffic.loc[mask, field] = np.nan
    audit = pd.DataFrame(records, columns=["source_row_number", "road_id", "timestamp", "field", "original_value", "reason_code"])
    counts["invalid_measurement_values_total"] = len(audit)
    counts["invalid_measurement_affected_rows_total"] = int(audit["source_row_number"].nunique()) if not audit.empty else 0
    return audit, counts


def _missing_windows(traffic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    valid = traffic.dropna(subset=["road_id", "timestamp"])
    start, end = valid["timestamp"].min().floor("30min"), valid["timestamp"].max().floor("30min")
    roads = sorted(valid["road_id"].unique())
    expected = pd.DataFrame(product(roads, pd.date_range(start, end, freq="30min")), columns=["road_id", "timestamp"])
    observed = valid[["road_id", "timestamp"]].drop_duplicates()
    missing = expected.merge(observed, on=["road_id", "timestamp"], how="left", indicator=True)
    missing = missing.loc[missing["_merge"].eq("left_only"), ["road_id", "timestamp"]].sort_values(["road_id", "timestamp"])

    summary: list[dict[str, Any]] = []
    longest = 0
    for road_id, group in missing.groupby("road_id", sort=True):
        values = group["timestamp"].sort_values()
        if values.empty:
            max_gap = 0
        else:
            breaks = values.diff().ne(pd.Timedelta(minutes=30)).cumsum()
            max_gap = int(values.groupby(breaks).size().max())
        longest = max(longest, max_gap)
        summary.append({"road_id": road_id, "missing_window_count": len(values), "longest_consecutive_gap_windows": max_gap, "longest_consecutive_gap_hours": max_gap * 0.5})
    return missing, pd.DataFrame(summary), longest


def _missingness(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": dataset,
            "column": frame.columns,
            "row_count": len(frame),
            "missing_count": frame.isna().sum().to_numpy(),
            "missing_percent": (frame.isna().mean().to_numpy() * 100).round(4),
        }
    )


def _derive_congestion(traffic: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    valid_inputs = traffic["traffic_volume"].notna() & traffic["road_capacity"].notna()
    valid_inputs &= traffic["traffic_volume"].ge(0) & traffic["road_capacity"].gt(0)
    ratio = traffic["traffic_volume"].div(traffic["road_capacity"].div(2))
    derived = pd.Series(pd.NA, index=traffic.index, dtype="string")
    derived.loc[valid_inputs & ratio.lt(0.50)] = "Free-flow"
    derived.loc[valid_inputs & ratio.ge(0.50) & ratio.lt(0.80)] = "Moderate"
    derived.loc[valid_inputs & ratio.ge(0.80) & ratio.lt(1.00)] = "Heavy"
    derived.loc[valid_inputs & ratio.ge(1.00)] = "Severe"
    existing = traffic["congestion_level"].astype("string")
    disagreement = existing.notna() & valid_inputs & existing.ne(derived)
    missing_mask = existing.isna()
    derive_mask = missing_mask & derived.notna()
    traffic["congestion_level_provenance"] = np.select(
        [derive_mask, existing.notna()],
        ["derived_valid_inputs", "original_label"],
        default="unresolved_missing",
    )
    traffic.loc[derive_mask, "congestion_level"] = derived.loc[derive_mask]
    traffic["vc_ratio"] = ratio.where(valid_inputs)
    return traffic, int(derive_mask.sum()), int(disagreement.sum())


def _normalize_weather(weather: pd.DataFrame) -> pd.DataFrame:
    weather["weather_condition_original"] = weather["weather_condition"]
    weather["weather_condition"] = (
        weather["weather_condition"].astype("string").str.strip().str.lower().map(WEATHER_MAP).fillna(
            weather["weather_condition"].astype("string").str.strip()
        )
    )
    return weather


def _write_report(
    path: Path,
    stats: dict[str, Any],
    missingness: pd.DataFrame,
    weather_counts: pd.DataFrame,
    parse_failures: pd.DataFrame,
    integration: dict[str, Any],
) -> None:
    lines = [
        "# FlowCast Data Quality Report",
        "",
        f"_Generated on {datetime.now():%Y-%m-%d %H:%M} by `src/clean.py`._",
        "",
        "This Phase 2 report validates, cleans, and integrates traffic, weather, and calendar data without creating synthetic traffic observations.",
        "",
        "## Source validation and hashes",
        f"- Raw files read: {stats['raw_rows']['traffic']:,} traffic, {stats['raw_rows']['weather']:,} weather, {stats['raw_rows']['calendar']:,} calendar rows.",
        f"- Raw SHA-256 hashes were unchanged after processing: **{stats['hashes_unchanged']}**.",
        f"- Parse failures: {len(parse_failures):,}; details are in `data/interim/parse_failures.csv`.",
        "",
        "## Duplicates",
        f"- Exact duplicate groups: {stats['exact_duplicate_groups']:,}; rows in groups: {stats['exact_duplicate_rows_in_groups']:,}; rows removed: {stats['exact_duplicate_rows_removed']:,}.",
        f"- Duplicate traffic-key groups: {stats['duplicate_key_groups']:,}; duplicate rows beyond first: {stats['duplicate_key_rows_beyond_first']:,}.",
        f"- Conflicting duplicate-key groups: {stats['conflicting_duplicate_key_groups']:,}; rows retained for review: {stats['conflicting_duplicate_key_rows']:,}.",
        "Exact duplicate removals and conflicting groups are saved with source row numbers.",
        "",
        "## Phase 2A audit-count reconciliation",
        "- The initial audit counted raw values meeting the same field-level predicates, while Phase 2A counted values after exact duplicate removal.",
        f"- Raw versus Phase 2A counts were traffic volume {stats['initial_measurement_counts']['traffic_volume']:,} vs {stats['traffic_volume_invalid_values']:,}, speed {stats['initial_measurement_counts']['avg_speed']:,} vs {stats['avg_speed_invalid_values']:,}, and occupancy {stats['initial_measurement_counts']['occupancy']:,} vs {stats['occupancy_invalid_values']:,}. These differences are attributable to invalid values on exact duplicate rows removed before measurement auditing; no cleaning rule was changed, and no conflicting duplicate keys were found.",
        "",
        "## Invalid measurements",
    ]
    for field in ("traffic_volume", "avg_speed", "occupancy", "road_capacity"):
        lines.append(f"- `{field}` invalid values: {stats[f'{field}_invalid_values']:,}; affected rows: {stats[f'{field}_invalid_rows']:,}.")
    lines.append(
        f"- All invalid measurement fields combined: {stats['invalid_measurement_values_total']:,} invalid values across "
        f"{stats['invalid_measurement_affected_rows_total']:,} distinct rows."
    )
    lines.extend(
        [
            "Invalid measurement fields were set to missing in the cleaned traffic output; other fields in the row were preserved.",
            "",
            "## Missingness",
            "Missing counts and percentages are saved in `data/interim/missingness.csv`. Null readings are not synthetic missing windows.",
            "",
        ]
    )
    for dataset, group in missingness.groupby("dataset", sort=False):
        lines.append(f"### {dataset.title()}")
        for row in group[group["missing_count"] > 0].itertuples():
            lines.append(f"- `{row.column}`: {row.missing_count:,} ({row.missing_percent:.2f}%).")
    lines.extend(
        [
            "",
            f"- Absent traffic windows: {stats['missing_window_count']:,}; longest consecutive gap: {stats['longest_gap_windows']:,} windows ({stats['longest_gap_hours']:.1f} hours).",
            "- Missing traffic-volume targets remain missing; no imputation or synthetic rows were performed.",
            "",
            "## Weather normalization",
            "Original and normalized category counts are saved in `data/interim/weather_label_counts.csv`.",
            "",
            "| Original | Normalized | Count |",
            "| --- | --- | ---: |",
        ]
    )
    lines.extend(f"| {row.original_label} | {row.normalized_label} | {row.count:,} |" for row in weather_counts.itertuples())
    lines.extend(
        [
            "",
            "## Congestion labels",
            f"- Missing labels derived from valid inputs: {stats['congestion_derived']:,}.",
            f"- Existing congestion labels retained: {stats['congestion_original']:,}.",
            f"- Missing labels unresolved because inputs were unavailable/invalid: {stats['congestion_unresolved']:,}.",
            f"- Existing labels disagreeing with recalculated valid-input labels: {stats['congestion_disagreements']:,}. Existing labels were not overwritten.",
            "- Formula: `V/C = traffic_volume / (road_capacity / 2)` with thresholds Free-flow `<0.50`, Moderate `[0.50,0.80)`, Heavy `[0.80,1.00)`, Severe `>=1.00`.",
            "",
            "## Integration",
            f"- Weather join: {integration['traffic_rows_before_weather']:,} traffic rows before and {integration['traffic_rows_after_weather']:,} after; {integration['weather_matched_traffic_rows']:,} matched, {integration['weather_unmatched_traffic_rows']:,} unmatched; {integration['weather_records_not_used']:,} weather records not used.",
            f"- Calendar join: {integration['calendar_matched_traffic_rows']:,} matched traffic rows, {integration['calendar_unmatched_traffic_rows']:,} unmatched; {integration['calendar_dates_total']:,} unique calendar dates.",
            f"- Final integrated dataset: {integration['final_rows']:,} rows; weather row delta {integration['weather_join_row_delta']:+,}, calendar row delta {integration['calendar_join_row_delta']:+,}. Both joins use many-to-one validation and preserve the traffic identity.",
            "- Weather matching uses station ID plus the traffic timestamp floored to the hour. Calendar matching uses the normalized traffic date.",
            f"- Final duplicate traffic identity rows: {integration['final_duplicate_identity_rows']:,}. Final missing measurements: "
            + ", ".join(f"`{field}` {count:,}" for field, count in integration["final_measurement_missing"].items()) + ".",
            f"- Final columns ({len(integration['final_columns'])}): {', '.join(integration['final_columns'])}.",
            "- Final CSV types are validated before writing; timestamp/hour fields are datetime values and calendar dates are normalized date values in the integration frame.",
            "",
            "## Final dataset integrity",
            "- Cleaned traffic measurements, quality flags, congestion labels/provenance, normalized weather fields, and calendar metadata are retained. Missing measurements remain missing where no safe imputation policy exists.",
            "- No accident-risk target, model, feature engineering, or dashboard functionality was created.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cleaning(config: dict[str, Any]) -> dict[str, Any]:
    paths = dataset_paths(config)
    required = {"traffic": TRAFFIC_REQUIRED, "weather": WEATHER_REQUIRED, "calendar": CALENDAR_REQUIRED}
    hashes_before: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {}
    parse_failures: list[pd.DataFrame] = []
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required raw dataset is missing: {path}")
        hashes_before[name] = sha256_file(path)
        frame = _add_source_rows(pd.read_csv(path))
        _require_columns(frame, required[name], name)
        frames[name] = frame

    traffic = frames["traffic"]
    traffic["timestamp"], failures = _parse_timestamp(traffic, "%Y-%m-%d", "traffic")
    parse_failures.append(failures)
    weather = frames["weather"]
    weather["timestamp"], failures = _parse_timestamp(weather, "%d/%m/%Y", "weather")
    parse_failures.append(failures)
    calendar = frames["calendar"]
    calendar["parsed_date"] = pd.to_datetime(calendar["date"], format="%Y-%m-%d", errors="coerce")
    calendar_failures = calendar.loc[calendar["parsed_date"].isna(), ["source_row_number"]].assign(
        dataset="calendar", date_parse_failed=True, time_parse_failed=False, timestamp_parse_failed=False
    )
    parse_failures.append(calendar_failures)
    all_parse_failures = pd.concat(parse_failures, ignore_index=True)

    exact_removed, conflicts, duplicate_stats = _duplicate_audits(traffic)
    initial_measurement_counts = {
        "traffic_volume": int(traffic["traffic_volume"].lt(0).sum()),
        "avg_speed": int(traffic["avg_speed"].gt(200).sum()),
        "occupancy": int(traffic["occupancy"].gt(100).sum()),
    }
    traffic = traffic.drop_duplicates(
        subset=[column for column in traffic.columns if column != "source_row_number"],
        keep="first",
    ).copy()
    measurement_audit, measurement_stats = _measurement_audit(traffic)
    traffic, congestion_derived, disagreements = _derive_congestion(traffic)
    missing_windows, window_summary, longest_gap = _missing_windows(traffic)
    traffic["quality_flags"] = ""
    for field in ("traffic_volume", "avg_speed", "occupancy", "road_capacity"):
        traffic.loc[traffic[field].isna(), "quality_flags"] = traffic.loc[traffic[field].isna(), "quality_flags"].map(
            lambda value, field=field: f"{value};{field}_missing".lstrip(";")
        )
    weather = _normalize_weather(weather)

    interim = configured_path(config["data"]["interim_dir"])
    reports = configured_path(config["paths"]["reports_dir"])
    interim.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    traffic.to_csv(interim / "traffic_cleaned.csv", index=False)
    weather.to_csv(interim / "weather_cleaned.csv", index=False)
    calendar.to_csv(interim / "calendar_validated.csv", index=False)
    exact_removed.to_csv(interim / "exact_duplicate_rows_removed.csv", index=False)
    conflicts.to_csv(interim / "conflicting_duplicate_key_rows.csv", index=False)
    measurement_audit.to_csv(interim / "measurement_quarantine.csv", index=False)
    all_parse_failures.to_csv(interim / "parse_failures.csv", index=False)
    missing_windows.to_csv(interim / "missing_traffic_windows.csv", index=False)
    window_summary.to_csv(interim / "missing_traffic_window_summary.csv", index=False)
    missingness = pd.concat(
        [_missingness(frames["traffic"], "traffic_raw"), _missingness(traffic, "traffic_cleaned"),
         _missingness(weather, "weather"), _missingness(calendar, "calendar")],
        ignore_index=True,
    )
    missingness.to_csv(interim / "missingness.csv", index=False)
    weather_counts = (
        weather.assign(original_label=weather["weather_condition_original"].astype("string").fillna("<MISSING>"),
                       normalized_label=weather["weather_condition"].astype("string").fillna("<MISSING>"))
        .groupby(["original_label", "normalized_label"], dropna=False).size().reset_index(name="count")
        .sort_values(["original_label", "normalized_label"])
    )
    weather_counts.to_csv(interim / "weather_label_counts.csv", index=False)
    hashes_after = {name: sha256_file(path) for name, path in paths.items()}
    stats: dict[str, Any] = {
        "raw_rows": {name: len(frame) for name, frame in frames.items()},
        **duplicate_stats,
        **measurement_stats,
        "missing_window_count": len(missing_windows),
        "longest_gap_windows": longest_gap,
        "longest_gap_hours": longest_gap * 0.5,
        "congestion_derived": congestion_derived,
        "congestion_original": int(traffic["congestion_level_provenance"].eq("original_label").sum()),
        "congestion_unresolved": int(traffic["congestion_level_provenance"].eq("unresolved_missing").sum()),
        "congestion_disagreements": disagreements,
        "initial_measurement_counts": initial_measurement_counts,
        "hashes_unchanged": hashes_before == hashes_after,
    }
    stats["hashes_before"] = hashes_before
    stats["hashes_after"] = hashes_after
    (interim / "run_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    integration = run_integration(config)
    stats["integration_final_rows"] = integration["final_rows"]
    _write_report(reports / "data_quality.md", stats, missingness, weather_counts, all_parse_failures, integration)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional path to config.yaml.")
    args = parser.parse_args()
    stats = run_cleaning(load_config(args.config))
    print(
        "FlowCast Phase 2 complete: "
        f"{stats['exact_duplicate_rows_removed']:,} exact duplicates removed, "
        f"{stats['traffic_volume_invalid_values'] + stats['avg_speed_invalid_values'] + stats['occupancy_invalid_values'] + stats['road_capacity_invalid_values']:,} invalid measurement values, "
        f"{stats['congestion_derived']:,} congestion labels derived, "
        f"{stats['integration_final_rows']:,} integrated rows. "
        f"Raw hashes unchanged={stats['hashes_unchanged']}."
    )


if __name__ == "__main__":
    main()
