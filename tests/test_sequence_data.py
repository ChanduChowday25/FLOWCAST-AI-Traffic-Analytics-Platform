import numpy as np
import pandas as pd
import pytest

from src.sequence_data import (
    DL_FEATURE_COLUMNS,
    DL_NUMERIC_FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    SequencePreprocessor,
    build_sequences,
    prepare_sequence_dataset,
)
from src.splitting import split_chronologically
from src.targets import TRAFFIC_VOLUME_TARGET_30M


def _frame(periods=36, roads=("A", "B")):
    rows = []
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="30min")
    for road_number, road_id in enumerate(roads):
        for position, timestamp in enumerate(timestamps):
            row = {
                "road_id": road_id,
                "timestamp": timestamp,
                TRAFFIC_VOLUME_TARGET_30M: float(1000 + position * 10 + road_number),
            }
            for column_number, column in enumerate(DL_NUMERIC_FEATURE_COLUMNS):
                row[column] = float(position + column_number + road_number)
            rows.append(row)
    return pd.DataFrame(rows)


def test_sequences_are_ordered_and_targets_align_to_endpoint():
    result = build_sequences(_frame(periods=13, roads=("A",)))

    assert len(result.sequences) == 2
    assert result.sequences.shape == (
        2,
        SEQUENCE_LENGTH,
        len(DL_FEATURE_COLUMNS) - 1,
    )
    assert result.road_sequences[0].tolist() == [0] * SEQUENCE_LENGTH
    assert result.road_categories == ("A",)
    assert result.metadata.loc[0, "endpoint_timestamp"] == pd.Timestamp("2025-01-01 05:30")
    assert result.targets.tolist() == [1110.0, 1120.0]
    assert result.metadata.loc[0, "source_row_indices"] == tuple(range(12))


def test_target_and_future_derived_columns_are_not_inputs():
    assert TRAFFIC_VOLUME_TARGET_30M not in DL_FEATURE_COLUMNS
    assert "traffic_volume_lag_1" not in DL_FEATURE_COLUMNS
    assert "traffic_volume_rolling_mean_6" not in DL_FEATURE_COLUMNS
    assert "weather_weather_condition" not in DL_FEATURE_COLUMNS


def test_no_cross_road_sequences_and_exact_continuity():
    frame = _frame(periods=24, roads=("A", "B"))
    result = build_sequences(frame)

    assert len(result.sequences) == 26
    assert all(np.unique(roads).size == 1 for roads in result.road_sequences)
    assert result.road_categories == ("A", "B")
    frame = frame.drop(frame.index[(frame["road_id"] == "A") & (frame["timestamp"] == "2025-01-01 03:00")])
    result = build_sequences(frame)
    assert result.exclusions["non_contiguous_timestamp"] > 0
    assert len(result.sequences) < 26


@pytest.mark.parametrize("target", [np.nan, np.inf, -1.0])
def test_invalid_targets_are_excluded(target):
    frame = _frame(periods=12, roads=("A",))
    frame.loc[frame.index[-1], TRAFFIC_VOLUME_TARGET_30M] = target
    result = build_sequences(frame)

    assert len(result.sequences) == 0
    assert result.exclusions["missing_target"] == int(pd.isna(target))
    assert result.exclusions["non_finite_target"] == int(np.isinf(target))
    assert result.exclusions["negative_target"] == int(target == -1.0)


def test_training_only_scaler_and_stable_road_encoding():
    train = _frame(periods=12, roads=("A",))
    validation = _frame(periods=12, roads=("A", "B"))
    train.loc[:, "traffic_volume"] = 10.0
    validation.loc[:, "traffic_volume"] = 10000.0
    preprocessor = SequencePreprocessor().fit(train)

    transformed = preprocessor.transform(validation)
    assert preprocessor.road_categories_ == ("A",)
    assert transformed.shape[1] == len(preprocessor.feature_names_out_)
    assert np.all(transformed[12:, -1] == 0.0)
    assert np.allclose(preprocessor.numeric_imputer_.statistics_[0], 10.0)


def test_prepare_dataset_is_split_and_deterministic():
    frame = _frame(periods=40, roads=("A", "B"))
    first = prepare_sequence_dataset(frame)
    second = prepare_sequence_dataset(frame)

    for left, right in (
        (first.train, second.train),
        (first.validation, second.validation),
        (first.test, second.test),
    ):
        np.testing.assert_array_equal(left.features, right.features)
        np.testing.assert_array_equal(left.targets, right.targets)
        pd.testing.assert_frame_equal(left.metadata, right.metadata)
    assert set(first.train.metadata["endpoint_timestamp"]).isdisjoint(
        first.validation.metadata["endpoint_timestamp"]
    )
    assert set(first.validation.metadata["endpoint_timestamp"]).isdisjoint(
        first.test.metadata["endpoint_timestamp"]
    )
    for partition in (first.train, first.validation, first.test):
        assert len(partition.features) == len(partition.targets)
        assert len(partition.features) == len(partition.metadata)
        assert partition.metadata["endpoint_source_index"].is_unique


def test_prepare_dataset_preserves_endpoint_metadata_and_target_alignment():
    frame = _frame(periods=40, roads=("A", "B"))
    result = prepare_sequence_dataset(frame, progress=lambda _: None)
    train, validation, test = split_chronologically(frame)
    preprocessor = SequencePreprocessor().fit(train)
    expected = tuple(
        preprocessor.transform_sequences(build_sequences(partition))
        for partition in (train, validation, test)
    )

    for partition, expected_partition in zip(
        (result.train, result.validation, result.test), expected
    ):
        np.testing.assert_array_equal(partition.features, expected_partition.features)
        np.testing.assert_array_equal(partition.targets, expected_partition.targets)
        pd.testing.assert_frame_equal(partition.metadata, expected_partition.metadata)
        for row_number, metadata in partition.metadata.iterrows():
            source_index = metadata["endpoint_source_index"]
            assert metadata["endpoint_timestamp"] == frame.loc[source_index, "timestamp"]
            assert partition.targets[row_number] == frame.loc[
                source_index, TRAFFIC_VOLUME_TARGET_30M
            ]


def test_missing_columns_and_invalid_configuration_fail_clearly():
    with pytest.raises(ValueError, match="Required sequence columns"):
        build_sequences(_frame().drop(columns=["avg_speed"]))
    with pytest.raises(ValueError, match="positive integer"):
        build_sequences(_frame(), sequence_length=0)
    with pytest.raises(ValueError, match="target cannot"):
        build_sequences(_frame(), feature_columns=(*DL_FEATURE_COLUMNS, TRAFFIC_VOLUME_TARGET_30M))


def test_empty_and_small_partitions_return_compact_empty_arrays():
    frame = _frame(periods=2, roads=("A",))
    result = build_sequences(frame)

    assert result.sequences.shape == (0, SEQUENCE_LENGTH, len(DL_FEATURE_COLUMNS) - 1)
    assert result.road_sequences.shape == (0, SEQUENCE_LENGTH)
    assert result.targets.shape == (0,)
    assert result.metadata.empty
    assert result.exclusions["insufficient_history"] == 2
