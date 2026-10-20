import pandas as pd
import pytest

from src.splitting import split_chronologically


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=10, freq="30min")
    return pd.DataFrame(
        {
            "road_id": ["A", "B"] * 10,
            "timestamp": [timestamp for timestamp in timestamps for _ in range(2)],
            "value": range(20),
        },
        index=range(100, 120),
    )


def test_split_uses_chronological_unique_timestamp_proportions():
    frame = _frame()

    train, validation, test = split_chronologically(frame)

    assert train["timestamp"].nunique() == 7
    assert validation["timestamp"].nunique() == 1
    assert test["timestamp"].nunique() == 2
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()


def test_all_roads_at_a_timestamp_share_the_same_split():
    frame = _frame()

    train, validation, test = split_chronologically(frame)

    for timestamp in frame["timestamp"].unique():
        containing_splits = [
            part
            for part in (train, validation, test)
            if timestamp in set(part["timestamp"])
        ]
        assert len(containing_splits) == 1
        assert containing_splits[0].loc[
            containing_splits[0]["timestamp"].eq(timestamp), "road_id"
        ].tolist() == frame.loc[frame["timestamp"].eq(timestamp), "road_id"].tolist()


def test_split_preserves_original_row_order_index_and_values():
    frame = _frame().iloc[
        [8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 17, 18, 19]
    ]

    train, validation, test = split_chronologically(frame)

    for part in (train, validation, test):
        expected = frame.loc[frame["timestamp"].isin(part["timestamp"].unique())]
        pd.testing.assert_frame_equal(part, expected)


def test_split_rejects_missing_timestamp_column():
    with pytest.raises(ValueError, match="Required timestamp column is missing"):
        split_chronologically(pd.DataFrame({"value": [1, 2, 3]}))


def test_split_rejects_invalid_timestamps():
    frame = pd.DataFrame({"timestamp": ["2025-01-01", "not-a-timestamp", "2025-01-02"]})

    with pytest.raises(ValueError, match="Invalid timestamps"):
        split_chronologically(frame)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_proportion": 0.8, "validation_proportion": 0.1, "test_proportion": 0.2},
        {"train_proportion": 0.0, "validation_proportion": 0.5, "test_proportion": 0.5},
    ],
)
def test_split_rejects_invalid_proportions(kwargs):
    with pytest.raises(ValueError, match="proportion"):
        split_chronologically(_frame(), **kwargs)


def test_split_rejects_too_few_unique_timestamps_for_nonempty_splits():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01", "2025-01-01"])})

    with pytest.raises(ValueError, match="At least three unique timestamps"):
        split_chronologically(frame)
