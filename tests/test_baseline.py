import numpy as np
import pandas as pd
import pytest

from src.baseline import (
    PERSISTENCE_PREDICTION_COLUMN,
    evaluate_persistence,
    predict_persistence,
)
from src.targets import TRAFFIC_VOLUME_TARGET_30M


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "road_id": ["A", "A", "B", "B"],
            "timestamp": pd.to_datetime(
                ["2025-01-01 08:00", "2025-01-01 08:30", "2025-01-01 08:00", "2025-01-01 08:30"]
            ),
            "traffic_volume_lag_1": [100.0, 125.0, 200.0, 250.0],
            "traffic_volume_lag_2": [90.0, 100.0, 180.0, 200.0],
            TRAFFIC_VOLUME_TARGET_30M: [125.0, 150.0, 250.0, 300.0],
            "future_observation": [9999.0] * 4,
        },
        index=[11, 12, 21, 22],
    )


def test_prediction_uses_latest_lag_one_not_lag_two_or_future_values():
    result = predict_persistence(_frame())

    assert result[PERSISTENCE_PREDICTION_COLUMN].tolist() == [100.0, 125.0, 200.0, 250.0]
    assert TRAFFIC_VOLUME_TARGET_30M not in result.columns
    assert "future_observation" not in result.columns


def test_predictions_preserve_target_row_alignment_and_timestamps():
    frame = _frame()

    result = predict_persistence(frame)

    assert result.index.tolist() == frame.index.tolist()
    assert result["road_id"].tolist() == frame["road_id"].tolist()
    assert result["timestamp"].tolist() == frame["timestamp"].tolist()


@pytest.mark.parametrize("invalid_value", [np.nan, -1.0, "not-a-number", np.inf])
def test_missing_or_invalid_baseline_values_are_unavailable(invalid_value):
    frame = _frame()
    frame["traffic_volume_lag_1"] = frame["traffic_volume_lag_1"].astype(object)
    frame.loc[12, "traffic_volume_lag_1"] = invalid_value

    result = predict_persistence(frame)

    assert pd.isna(result.loc[12, PERSISTENCE_PREDICTION_COLUMN])
    assert result.loc[12, PERSISTENCE_PREDICTION_COLUMN] != frame.loc[12, TRAFFIC_VOLUME_TARGET_30M]


def test_evaluation_filters_invalid_targets_and_reports_mape_zero_handling():
    frame = _frame()
    frame.loc[11, TRAFFIC_VOLUME_TARGET_30M] = 0.0
    frame.loc[12, TRAFFIC_VOLUME_TARGET_30M] = np.nan
    frame.loc[21, "traffic_volume_lag_1"] = np.nan

    evaluation = evaluate_persistence(frame)

    assert evaluation.valid_prediction_count == 2
    assert evaluation.invalid_target_count == 1
    assert evaluation.missing_baseline_count == 1
    assert evaluation.zero_target_count == 1
    assert evaluation.mape_eligible_count == 1
    assert evaluation.metrics["mape"] == pytest.approx(16.6666666667)
    assert evaluation.metrics["mae"] == 75.0
    assert evaluation.metrics["rmse"] == pytest.approx(np.sqrt(6250))


def test_metrics_are_calculated_on_valid_rows():
    frame = _frame()

    evaluation = evaluate_persistence(frame)

    assert evaluation.metrics["rmse"] == pytest.approx(np.sqrt(6250 / 4))
    assert evaluation.metrics["mae"] == pytest.approx(37.5)
    assert evaluation.metrics["mape"] == pytest.approx((20 + 16.6666666667 + 20 + 16.6666666667) / 4)
    assert evaluation.metrics["r2"] == pytest.approx(1 - 6250 / 20468.75)


def test_missing_inputs_and_invalid_timestamps_fail_clearly():
    with pytest.raises(ValueError, match="Required persistence columns"):
        predict_persistence(pd.DataFrame({"road_id": ["A"]}))
    with pytest.raises(ValueError, match="Required target column"):
        evaluate_persistence(_frame().drop(columns=TRAFFIC_VOLUME_TARGET_30M))
    invalid = _frame()
    invalid["timestamp"] = invalid["timestamp"].astype(object)
    invalid.loc[11, "timestamp"] = "invalid"
    with pytest.raises(ValueError, match="Invalid timestamps"):
        predict_persistence(invalid)
    with pytest.raises(TypeError, match="pandas DataFrame"):
        predict_persistence([])
