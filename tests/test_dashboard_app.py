import pandas as pd
import pytest

from dashboard.app import (
    CONGESTION_LEVELS,
    aggregate_congestion_heatmap,
    aggregate_historical_data,
    daily_volume_history,
    filter_historical_data,
    filter_historical_roads,
    historical_date_bounds,
    historical_summary,
    hourly_volume_history,
    road_comparison_summary,
    select_forecast_history,
)
from src.sequence_data import DL_NUMERIC_FEATURE_COLUMNS


def _frame():
    return pd.DataFrame(
        {
            "road_id": ["A", "A", "B", "B"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-01-01 01:00",
                    "2025-01-02 00:00",
                    "2025-01-02 01:00",
                ]
            ),
            "hour_of_day": [0, 1, 0, 1],
            "traffic_volume": [100.0, 200.0, 300.0, None],
            "avg_speed": [50.0, 40.0, None, 30.0],
        }
    )


def test_historical_summary_uses_available_values_only():
    summary = historical_summary(_frame())

    assert summary["traffic_observations"] == 3
    assert summary["unique_roads"] == 2
    assert summary["average_traffic_volume"] == pytest.approx(200.0)
    assert summary["average_speed"] == pytest.approx(40.0)


def test_daily_volume_history_is_readable_and_aggregated():
    chart = daily_volume_history(_frame())

    assert chart["date"].tolist() == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
    ]
    assert chart["average_traffic_volume"].tolist() == [150.0, 300.0]


def test_hourly_volume_history_aggregates_by_hour():
    chart = hourly_volume_history(_frame())

    assert chart["hour_of_day"].tolist() == [0, 1]
    assert chart["average_traffic_volume"].tolist() == [200.0, 200.0]


def test_transformations_return_empty_frames_when_required_columns_missing():
    frame = pd.DataFrame({"road_id": ["A"]})

    assert daily_volume_history(frame).empty
    assert hourly_volume_history(frame).empty


def test_historical_date_bounds_and_filter_are_inclusive():
    frame = _frame()

    assert historical_date_bounds(frame) == (
        pd.Timestamp("2025-01-01").date(),
        pd.Timestamp("2025-01-02").date(),
    )
    filtered = filter_historical_data(
        frame,
        pd.Timestamp("2025-01-01").date(),
        pd.Timestamp("2025-01-01").date(),
        "A",
    )

    assert len(filtered) == 2
    assert set(filtered["road_id"]) == {"A"}


def test_historical_filter_returns_empty_for_unmatched_road():
    filtered = filter_historical_data(
        _frame(),
        pd.Timestamp("2025-01-01").date(),
        pd.Timestamp("2025-01-02").date(),
        "UNKNOWN",
    )

    assert filtered.empty


def test_historical_aggregation_handles_missing_values_and_grains():
    frame = _frame()

    daily = aggregate_historical_data(frame, "Daily")
    hourly = aggregate_historical_data(frame, "Hourly")

    assert daily["average_traffic_volume"].tolist() == [150.0, 300.0]
    assert daily["average_speed"].tolist() == [45.0, 30.0]
    assert len(hourly) == 4
    assert hourly["average_traffic_volume"].iloc[:3].tolist() == [100.0, 200.0, 300.0]
    assert pd.isna(hourly["average_traffic_volume"].iloc[3])
    assert hourly["average_speed"].iloc[[0, 1, 3]].tolist() == [50.0, 40.0, 30.0]
    assert pd.isna(hourly["average_speed"].iloc[2])


def test_historical_aggregation_rejects_unknown_grain():
    with pytest.raises(ValueError, match="Aggregation must be"):
        aggregate_historical_data(_frame(), "Weekly")


def test_filter_historical_roads_selects_multiple_real_roads():
    filtered = filter_historical_roads(
        _frame(),
        pd.Timestamp("2025-01-01").date(),
        pd.Timestamp("2025-01-02").date(),
        ["A", "B"],
    )

    assert len(filtered) == 4
    assert set(filtered["road_id"]) == {"A", "B"}


def test_road_comparison_summary_aggregates_missing_values_per_road():
    comparison = road_comparison_summary(_frame())

    assert comparison["road_id"].tolist() == ["A", "B"]
    assert comparison["average_traffic_volume"].tolist() == [150.0, 300.0]
    assert comparison["average_speed"].tolist() == [45.0, 30.0]
    assert comparison["volume_observations"].tolist() == [2, 1]
    assert comparison["speed_observations"].tolist() == [2, 1]


def test_road_comparison_helpers_handle_empty_selection_and_missing_columns():
    empty = filter_historical_roads(
        _frame(),
        pd.Timestamp("2025-01-01").date(),
        pd.Timestamp("2025-01-02").date(),
        [],
    )

    assert empty.empty
    assert road_comparison_summary(pd.DataFrame({"road_id": ["A"]})).empty


def test_congestion_heatmap_preserves_category_order_and_ignores_missing_values():
    frame = pd.DataFrame(
        {
            "road_id": ["A", "A", "A", "A", "B"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 08:00",
                    "2025-01-01 08:30",
                    "2025-01-01 09:00",
                    "2025-01-01 09:30",
                    "2025-01-01 08:00",
                ]
            ),
            "congestion_level": [
                "Severe",
                "Free-flow",
                "Moderate",
                None,
                "Heavy",
            ],
        }
    )

    heatmap = aggregate_congestion_heatmap(frame)

    assert CONGESTION_LEVELS == ("Free-flow", "Moderate", "Heavy", "Severe")
    assert heatmap.loc[
        (heatmap["road_id"] == "A") & (heatmap["hour_of_day"] == 8),
        "congestion_score",
    ].iloc[0] == pytest.approx(1.5)
    assert heatmap.loc[
        (heatmap["road_id"] == "A") & (heatmap["hour_of_day"] == 9),
        "observations",
    ].iloc[0] == 1
    assert len(heatmap) == 3


def test_congestion_heatmap_returns_empty_without_valid_categories():
    frame = pd.DataFrame(
        {
            "road_id": ["A"],
            "timestamp": pd.to_datetime(["2025-01-01 08:00"]),
            "congestion_level": [None],
        }
    )

    assert aggregate_congestion_heatmap(frame).empty


def _forecast_frame(periods=14):
    frame = pd.DataFrame(
        {
            "road_id": ["NL-001"] * periods,
            "timestamp": pd.date_range("2025-01-01", periods=periods, freq="30min"),
        }
    )
    for position, column in enumerate(DL_NUMERIC_FEATURE_COLUMNS):
        frame[column] = [float(position + row) for row in range(periods)]
    return frame


def test_select_forecast_history_returns_latest_contiguous_window_without_target():
    frame = _forecast_frame()

    history = select_forecast_history(
        frame,
        "NL-001",
        sequence_length=12,
        supported_roads=("NL-001",),
    )

    assert len(history) == 12
    assert history["timestamp"].iloc[0] == pd.Timestamp("2025-01-01 01:00")
    assert history["timestamp"].iloc[-1] == pd.Timestamp("2025-01-01 06:30")
    assert "traffic_volume_target_30m" not in history


def test_select_forecast_history_rejects_unsupported_road():
    with pytest.raises(ValueError, match="Unsupported road"):
        select_forecast_history(
            _forecast_frame(),
            "UNKNOWN",
            sequence_length=12,
            supported_roads=("NL-001",),
        )


def test_select_forecast_history_rejects_insufficient_or_gapped_history():
    with pytest.raises(ValueError, match="does not have 12 consecutive"):
        select_forecast_history(
            _forecast_frame(periods=11),
            "NL-001",
            sequence_length=12,
            supported_roads=("NL-001",),
        )

    frame = _forecast_frame()
    frame.loc[6:, "timestamp"] += pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="does not have 12 consecutive"):
        select_forecast_history(
            frame,
            "NL-001",
            sequence_length=12,
            supported_roads=("NL-001",),
        )
