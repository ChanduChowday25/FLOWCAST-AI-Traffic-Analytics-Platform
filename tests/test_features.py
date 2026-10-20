import numpy as np
import pandas as pd
import pytest

from src.features import (
    LAG_FEATURE_COLUMNS,
    ROLLING_FEATURE_COLUMNS,
    TIME_FEATURE_COLUMNS,
    add_time_features,
    add_traffic_volume_lags,
    add_traffic_volume_rolling_features,
)


def test_time_features_extract_timestamp_components():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-06 07:30", "2025-01-07 12:00"])})
    result = add_time_features(frame)

    assert list(result.columns) == ["timestamp", *TIME_FEATURE_COLUMNS]
    assert result["hour_of_day"].tolist() == [7, 12]
    assert result["day_of_week"].tolist() == [0, 1]
    assert result["month"].tolist() == [1, 1]


def test_weekend_behavior():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-10 12:00", "2025-01-11 12:00", "2025-01-12 12:00"])})
    result = add_time_features(frame)

    assert result["is_weekend"].tolist() == [False, True, True]


def test_rush_hour_boundaries_are_half_open():
    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2025-01-06 06:59", "2025-01-06 07:00", "2025-01-06 09:59", "2025-01-06 10:00", "2025-01-06 16:00", "2025-01-06 19:00"])}
    )
    result = add_time_features(frame)

    assert result["is_rush_hour"].tolist() == [False, True, True, False, True, False]


def test_missing_measurements_are_preserved():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-06 08:00", "2025-01-06 08:30"]),
            "traffic_volume": [np.nan, 10.0],
            "avg_speed": [np.nan, 30.0],
            "occupancy": [np.nan, 20.0],
            "weather_temperature": [np.nan, 12.0],
        }
    )
    result = add_time_features(frame)

    pd.testing.assert_series_equal(result["traffic_volume"], frame["traffic_volume"])
    pd.testing.assert_series_equal(result["avg_speed"], frame["avg_speed"])
    pd.testing.assert_series_equal(result["occupancy"], frame["occupancy"])
    pd.testing.assert_series_equal(result["weather_temperature"], frame["weather_temperature"])


def test_invalid_timestamp_fails_clearly():
    with pytest.raises(ValueError, match="Invalid timestamps"):
        add_time_features(pd.DataFrame({"timestamp": ["not-a-timestamp"]}))


def test_traffic_volume_lags_match_exact_historical_timestamps():
    frame = pd.DataFrame(
        {
            "road_id": ["A", "A", "A"],
            "timestamp": pd.to_datetime(["2025-01-01 08:00", "2025-01-01 08:30", "2025-01-01 09:00"]),
            "traffic_volume": [300.0, 400.0, 500.0],
        }
    )
    result = add_traffic_volume_lags(frame)

    assert result[list(LAG_FEATURE_COLUMNS)].iloc[2].tolist() == [400.0, 300.0]


def test_traffic_volume_lags_do_not_cross_roads():
    frame = pd.DataFrame(
        {
            "road_id": ["A", "B", "B"],
            "timestamp": pd.to_datetime(["2025-01-01 08:30", "2025-01-01 08:00", "2025-01-01 08:30"]),
            "traffic_volume": [999.0, 100.0, 200.0],
        }
    )
    result = add_traffic_volume_lags(frame)

    assert pd.isna(result.loc[0, "traffic_volume_lag_1"])
    assert result.loc[2, "traffic_volume_lag_1"] == 100.0


def test_missing_timestamps_do_not_create_positional_lags():
    frame = pd.DataFrame(
        {
            "road_id": ["A", "A", "A"],
            "timestamp": pd.to_datetime(["2025-01-01 08:00", "2025-01-01 09:00", "2025-01-01 09:30"]),
            "traffic_volume": [300.0, 500.0, 600.0],
        }
    )
    result = add_traffic_volume_lags(frame)

    assert pd.isna(result.loc[1, "traffic_volume_lag_1"])
    assert result.loc[2, "traffic_volume_lag_1"] == 500.0
    assert pd.isna(result.loc[2, "traffic_volume_lag_2"])


def test_missing_or_invalid_historical_volume_stays_missing():
    frame = pd.DataFrame(
        {
            "road_id": ["A", "A", "A", "A"],
            "timestamp": pd.to_datetime(["2025-01-01 08:00", "2025-01-01 08:30", "2025-01-01 09:00", "2025-01-01 09:30"]),
            "traffic_volume": [np.nan, -1.0, 500.0, 600.0],
        }
    )
    result = add_traffic_volume_lags(frame)

    assert pd.isna(result.loc[2, "traffic_volume_lag_1"])
    assert pd.isna(result.loc[3, "traffic_volume_lag_2"])


def test_first_observations_have_missing_lags_and_future_values_are_unused():
    frame = pd.DataFrame(
        {
            "road_id": ["A", "A"],
            "timestamp": pd.to_datetime(["2025-01-01 09:00", "2025-01-01 09:30"]),
            "traffic_volume": [500.0, 600.0],
        }
    )
    result = add_traffic_volume_lags(frame)

    assert result[list(LAG_FEATURE_COLUMNS)].iloc[0].isna().all()
    assert pd.isna(result.loc[0, "traffic_volume_lag_1"])


def test_lags_preserve_existing_columns_order_and_missing_values():
    frame = pd.DataFrame(
        {
            "road_id": ["A", "A"],
            "timestamp": pd.to_datetime(["2025-01-01 08:00", "2025-01-01 08:30"]),
            "traffic_volume": [np.nan, 10.0],
            "other": [None, "kept"],
        }
    )
    result = add_traffic_volume_lags(frame)

    assert result.index.tolist() == frame.index.tolist()
    assert result.columns[: len(frame.columns)].tolist() == frame.columns.tolist()
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    assert result["traffic_volume"].isna().tolist() == frame["traffic_volume"].isna().tolist()


def test_rolling_means_use_exact_past_half_hour_periods():
    frame = pd.DataFrame(
        {
            "road_id": ["A"] * 7,
            "timestamp": pd.date_range("2025-01-01 08:00", periods=7, freq="30min"),
            "traffic_volume": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
        }
    )
    result = add_traffic_volume_rolling_features(frame)

    assert pd.isna(result.loc[0, ROLLING_FEATURE_COLUMNS[0]])
    assert result.loc[3, ROLLING_FEATURE_COLUMNS[0]] == 200.0
    assert result.loc[6, ROLLING_FEATURE_COLUMNS[0]] == 500.0
    assert result.loc[6, ROLLING_FEATURE_COLUMNS[1]] == 350.0


def test_rolling_features_do_not_cross_roads_or_bridge_missing_timestamps():
    frame = pd.DataFrame(
        {
            "road_id": ["A", "A", "B", "B"],
            "timestamp": pd.to_datetime(
                ["2025-01-01 08:00", "2025-01-01 09:00", "2025-01-01 08:00", "2025-01-01 08:30"]
            ),
            "traffic_volume": [100.0, 300.0, 900.0, 800.0],
        }
    )
    result = add_traffic_volume_rolling_features(frame)

    assert result.loc[1, ROLLING_FEATURE_COLUMNS[0]] == 100.0
    assert result.loc[3, ROLLING_FEATURE_COLUMNS[0]] == 900.0


def test_rolling_features_ignore_invalid_or_missing_history():
    frame = pd.DataFrame(
        {
            "road_id": ["A"] * 4,
            "timestamp": pd.date_range("2025-01-01 08:00", periods=4, freq="30min"),
            "traffic_volume": [100.0, np.nan, -1.0, 400.0],
        }
    )
    result = add_traffic_volume_rolling_features(frame)

    assert result.loc[3, ROLLING_FEATURE_COLUMNS[0]] == 100.0
    assert result.loc[3, ROLLING_FEATURE_COLUMNS[1]] == 100.0


def test_rolling_features_do_not_use_current_or_future_values_and_preserve_input():
    frame = pd.DataFrame(
        {
            "road_id": ["A", "A"],
            "timestamp": pd.to_datetime(["2025-01-01 08:00", "2025-01-01 08:30"]),
            "traffic_volume": [100.0, 200.0],
            "other": [None, "kept"],
        }
    )
    original = frame.copy(deep=True)
    result = add_traffic_volume_rolling_features(frame)

    assert result.loc[0, list(ROLLING_FEATURE_COLUMNS)].isna().all()
    assert result.loc[1, ROLLING_FEATURE_COLUMNS[0]] == 100.0
    assert result.loc[1, ROLLING_FEATURE_COLUMNS[1]] == 100.0
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(result[frame.columns], frame)


def test_rolling_features_reject_duplicate_identity_keys():
    frame = pd.DataFrame(
        {
            "road_id": ["A", "A"],
            "timestamp": pd.to_datetime(["2025-01-01 08:00", "2025-01-01 08:00"]),
            "traffic_volume": [100.0, 200.0],
        }
    )

    with pytest.raises(ValueError, match="Duplicate historical keys"):
        add_traffic_volume_rolling_features(frame)


def test_rolling_features_reject_invalid_timestamps():
    frame = pd.DataFrame(
        {
            "road_id": ["A"],
            "timestamp": ["not-a-timestamp"],
            "traffic_volume": [100.0],
        }
    )

    with pytest.raises(ValueError, match="Invalid timestamps"):
        add_traffic_volume_rolling_features(frame)


def test_rolling_features_exclude_nonnumeric_values_and_average_available_history():
    frame = pd.DataFrame(
        {
            "road_id": ["A"] * 4,
            "timestamp": pd.date_range("2025-01-01 08:00", periods=4, freq="30min"),
            "traffic_volume": [100.0, "bad", np.nan, 400.0],
        }
    )
    result = add_traffic_volume_rolling_features(frame)

    assert result.loc[3, ROLLING_FEATURE_COLUMNS[0]] == 100.0


def test_rolling_features_do_not_change_when_future_values_change():
    frame = pd.DataFrame(
        {
            "road_id": ["A"] * 5,
            "timestamp": pd.date_range("2025-01-01 08:00", periods=5, freq="30min"),
            "traffic_volume": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )
    baseline = add_traffic_volume_rolling_features(frame)

    changed = frame.copy()
    changed.loc[4, "traffic_volume"] = 9999.0
    updated = add_traffic_volume_rolling_features(changed)

    pd.testing.assert_frame_equal(
        baseline.loc[:3, list(ROLLING_FEATURE_COLUMNS)],
        updated.loc[:3, list(ROLLING_FEATURE_COLUMNS)],
    )
