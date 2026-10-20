import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clean import _derive_congestion, _duplicate_audits, _measurement_audit, _normalize_weather, sha256_file
from src.integrate import integrate_sources


def _traffic(rows):
    return pd.DataFrame(rows, columns=["source_row_number", "road_id", "timestamp", "traffic_volume", "avg_speed", "occupancy", "road_capacity", "congestion_level"])


def test_exact_duplicate_removal():
    frame = pd.DataFrame({
        "source_row_number": [2, 3],
        "road_id": ["A", "A"],
        "timestamp": pd.to_datetime(["2025-01-01 00:00"] * 2),
        "traffic_volume": [1, 1],
        "avg_speed": [30, 30],
        "occupancy": [20, 20],
        "road_capacity": [100, 100],
        "congestion_level": [None, None],
    })
    removed, conflicts, stats = _duplicate_audits(frame)
    assert len(removed) == 1
    assert conflicts.empty
    assert stats["exact_duplicate_rows_removed"] == 1


def test_conflicting_duplicate_key_detection():
    frame = _traffic([[2, "A", pd.Timestamp("2025-01-01 00:00"), 10, 30, 20, 100, None],
                      [3, "A", pd.Timestamp("2025-01-01 00:00"), 11, 30, 20, 100, None]])
    assert frame.groupby(["road_id", "timestamp"])["traffic_volume"].nunique().eq(2).all()


def test_invalid_measurements_are_field_level_and_audited():
    frame = _traffic([[2, "A", pd.Timestamp("2025-01-01 00:00"), -1, 201, 101, 0, None]])
    audit, counts = _measurement_audit(frame)
    assert len(audit) == 4
    assert counts["traffic_volume_invalid_values"] == 1
    assert frame.loc[0, ["traffic_volume", "avg_speed", "occupancy", "road_capacity"]].isna().all()


def test_missing_traffic_volume_remains_missing():
    frame = _traffic([[2, "A", pd.Timestamp("2025-01-01 00:00"), np.nan, 30, 20, 100, None]])
    frame, derived, _ = _derive_congestion(frame)
    assert pd.isna(frame.loc[0, "traffic_volume"])
    assert derived == 0
    assert frame.loc[0, "congestion_level_provenance"] == "unresolved_missing"


def test_congestion_threshold_boundaries():
    frame = _traffic([
        [2, "A", pd.Timestamp("2025-01-01 00:00"), 25, 30, 20, 100, None],
        [3, "B", pd.Timestamp("2025-01-01 00:00"), 40, 30, 20, 100, None],
        [4, "C", pd.Timestamp("2025-01-01 00:00"), 50, 30, 20, 100, None],
    ])
    frame, derived, disagreements = _derive_congestion(frame)
    assert derived == 3
    assert disagreements == 0
    assert frame["congestion_level"].tolist() == ["Moderate", "Heavy", "Severe"]


def test_weather_label_normalization():
    frame = pd.DataFrame({"weather_condition": [" CLEAR ", "rain ", "rainy", "FOG", "foggy", "Cloudy", "Overcast"]})
    normalized = _normalize_weather(frame)
    assert normalized["weather_condition"].tolist() == ["Clear", "Rain", "Rain", "Fog", "Fog", "Cloudy", "Overcast"]


def test_raw_file_hash_preservation(tmp_path: Path):
    path = tmp_path / "raw.csv"
    path.write_bytes(b"a,b\n1,2\n")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(path) == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def _integration_frames():
    traffic = pd.DataFrame(
        {
            "road_id": ["A", "A", "B"],
            "weather_station_id": ["S1", "S1", "S2"],
            "timestamp": pd.to_datetime(["2025-01-01 10:00", "2025-01-01 10:30", "2025-01-02 10:00"]),
            "traffic_volume": [10, 11, 12],
        }
    )
    weather = pd.DataFrame(
        {
            "station_id": ["S1", "S2"],
            "timestamp": pd.to_datetime(["2025-01-01 10:00", "2025-01-02 10:00"]),
            "weather_condition": ["Clear", "Rain"],
        }
    )
    calendar = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "public_holiday": [1, 0],
            "holiday_name": ["New Year", None],
        }
    )
    return traffic, weather, calendar


def test_integration_matches_half_hours_to_one_weather_hour():
    final, audit = integrate_sources(*_integration_frames())
    assert final["weather_weather_condition"].tolist() == ["Clear", "Clear", "Rain"]
    assert audit["weather_matched_traffic_rows"] == 3
    assert len(final) == 3


def test_weather_join_requires_station_and_hour():
    traffic, weather, calendar = _integration_frames()
    weather.loc[0, "station_id"] = "OTHER"
    final, audit = integrate_sources(traffic, weather, calendar)
    assert pd.isna(final.loc[0, "weather_weather_condition"])
    assert audit["weather_unmatched_traffic_rows"] == 2


def test_calendar_join_uses_date():
    final, _ = integrate_sources(*_integration_frames())
    assert final["calendar_public_holiday"].tolist() == [1, 1, 0]


def test_duplicate_weather_keys_are_detected():
    traffic, weather, calendar = _integration_frames()
    weather = pd.concat([weather, weather.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="weather keys are not unique"):
        integrate_sources(traffic, weather, calendar)


def test_duplicate_calendar_dates_are_detected():
    traffic, weather, calendar = _integration_frames()
    calendar = pd.concat([calendar, calendar.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="calendar keys are not unique"):
        integrate_sources(traffic, weather, calendar)


def test_integration_reports_unmatched_records_without_multiplication():
    traffic, weather, calendar = _integration_frames()
    traffic.loc[2, "weather_station_id"] = "MISSING"
    final, audit = integrate_sources(traffic, weather, calendar)
    assert len(final) == len(traffic)
    assert audit["weather_unmatched_traffic_rows"] == 1
    assert audit["weather_records_not_used"] == 1
    assert audit["calendar_unmatched_traffic_rows"] == 0
