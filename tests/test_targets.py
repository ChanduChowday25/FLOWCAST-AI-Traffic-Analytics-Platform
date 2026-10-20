import numpy as np
import pandas as pd
import pytest

from src.targets import TRAFFIC_VOLUME_TARGET_30M, add_traffic_volume_target_30m


def _frame(rows):
    return pd.DataFrame(rows, columns=["road_id", "timestamp", "traffic_volume", "other"])


def test_target_matches_exact_future_timestamp_and_preserves_input():
    frame = _frame(
        [
            ["A", "2025-01-01 08:00", 100.0, "first"],
            ["A", "2025-01-01 08:30", 125.0, "second"],
            ["A", "2025-01-01 09:00", 150.0, "third"],
        ]
    )
    original = frame.copy(deep=True)

    result = add_traffic_volume_target_30m(frame)

    assert result[TRAFFIC_VOLUME_TARGET_30M].iloc[:2].tolist() == [125.0, 150.0]
    assert pd.isna(result.loc[2, TRAFFIC_VOLUME_TARGET_30M])
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    assert result.index.tolist() == frame.index.tolist()


def test_target_does_not_cross_roads():
    frame = _frame(
        [
            ["A", "2025-01-01 08:00", 100.0, "A"],
            ["B", "2025-01-01 08:30", 900.0, "B"],
        ]
    )

    result = add_traffic_volume_target_30m(frame)

    assert pd.isna(result.loc[0, TRAFFIC_VOLUME_TARGET_30M])


def test_missing_future_timestamp_stays_missing_instead_of_using_next_row():
    frame = _frame(
        [
            ["A", "2025-01-01 08:00", 100.0, "current"],
            ["A", "2025-01-01 09:00", 300.0, "later"],
        ]
    )

    result = add_traffic_volume_target_30m(frame)

    assert pd.isna(result.loc[0, TRAFFIC_VOLUME_TARGET_30M])


@pytest.mark.parametrize("future_volume", [np.nan, "not-a-number", -1.0])
def test_invalid_future_volume_stays_missing(future_volume):
    frame = _frame(
        [
            ["A", "2025-01-01 08:00", 100.0, "current"],
            ["A", "2025-01-01 08:30", future_volume, "future"],
        ]
    )

    result = add_traffic_volume_target_30m(frame)

    assert pd.isna(result.loc[0, TRAFFIC_VOLUME_TARGET_30M])


def test_duplicate_keys_are_rejected():
    frame = _frame(
        [
            ["A", "2025-01-01 08:00", 100.0, "one"],
            ["A", "2025-01-01 08:00", 101.0, "two"],
        ]
    )

    with pytest.raises(ValueError, match="Duplicate target keys"):
        add_traffic_volume_target_30m(frame)


def test_invalid_timestamps_are_rejected():
    frame = _frame([["A", "not-a-timestamp", 100.0, "bad"]])

    with pytest.raises(ValueError, match="Invalid timestamps"):
        add_traffic_volume_target_30m(frame)


def test_target_does_not_use_a_same_position_value_from_another_road():
    frame = _frame(
        [
            ["A", "2025-01-01 08:00", 100.0, "A current"],
            ["B", "2025-01-01 08:30", 200.0, "B future"],
            ["A", "2025-01-01 09:00", 300.0, "A later"],
        ]
    )

    result = add_traffic_volume_target_30m(frame)

    assert pd.isna(result.loc[0, TRAFFIC_VOLUME_TARGET_30M])
