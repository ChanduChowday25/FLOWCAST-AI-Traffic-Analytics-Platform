import numpy as np
import pandas as pd
import pytest

from src.evaluate_final_test import evaluate_gru_test
from src.sequence_data import DL_FEATURE_COLUMNS, build_sequences
from src.splitting import split_chronologically
from src.train_linear_regression import _metrics


def test_metrics_match_expected_regression_values():
    metrics = _metrics(
        pd.Series([10.0, 20.0]),
        np.asarray([11.0, 18.0]),
    )

    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["mape"] == pytest.approx(10.0)
    assert metrics["r2"] == pytest.approx(0.9)


def test_final_evaluation_uses_test_partition_only():
    rows = []
    timestamps = pd.date_range("2025-01-01", periods=30, freq="30min")
    for road_id in ("A", "B"):
        for position, timestamp in enumerate(timestamps):
            row = {
                "road_id": road_id,
                "timestamp": timestamp,
                "traffic_volume_target_30m": float(position),
            }
            for column in DL_FEATURE_COLUMNS:
                if column != "road_id":
                    row[column] = float(position)
            rows.append(row)
    frame = pd.DataFrame(rows)

    train, validation, test = split_chronologically(frame)
    raw_test = build_sequences(test, feature_columns=DL_FEATURE_COLUMNS)

    assert callable(evaluate_gru_test)
    assert set(train["timestamp"]).isdisjoint(test["timestamp"])
    assert set(validation["timestamp"]).isdisjoint(test["timestamp"])
    assert set(raw_test.metadata["endpoint_source_index"]).issubset(set(test.index))
    assert set(raw_test.metadata["endpoint_source_index"]).isdisjoint(set(validation.index))
