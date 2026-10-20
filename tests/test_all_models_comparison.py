import numpy as np
import pandas as pd
import pytest

from src.compare_all_models import (
    align_validation_predictions,
    calculate_comparison_metrics,
)


def _rows(targets, predictions):
    return pd.DataFrame(
        {
            "road_id": ["A"] * len(targets),
            "timestamp": pd.date_range(
                "2025-01-01", periods=len(targets), freq="30min"
            ),
            "actual_target_30m": targets,
            "prediction": predictions,
        }
    )


def test_alignment_requires_all_models_and_exact_target_matches():
    predictions = {
        "random_forest": _rows([10.0, 20.0, 30.0], [11.0, 21.0, 29.0]),
        "lstm": _rows([10.0, 999.0, 30.0], [12.0, 19.0, 31.0]),
        "gru": _rows([10.0, 20.0, 30.0], [9.0, 22.0, 28.0]),
    }

    aligned, exclusions = align_validation_predictions(predictions)

    assert len(aligned) == 2
    assert aligned["actual_target_30m"].tolist() == [10.0, 30.0]
    assert exclusions["shared_road_timestamp_rows"] == 3
    assert exclusions["target_mismatch_rows"] == 1
    assert exclusions["common_rows"] == 2


def test_comparison_metrics_match_shared_predictions():
    aligned = pd.DataFrame(
        {
            "actual_target_30m": [10.0, 20.0],
            "random_forest_prediction": [11.0, 21.0],
            "lstm_prediction": [12.0, 18.0],
            "gru_prediction": [10.0, 24.0],
        }
    )

    metrics = calculate_comparison_metrics(aligned)

    assert metrics["random_forest"]["mae"] == pytest.approx(1.0)
    assert metrics["lstm"]["mae"] == pytest.approx(2.0)
    assert metrics["gru"]["mae"] == pytest.approx(2.0)
    assert metrics["random_forest"]["rmse"] == pytest.approx(1.0)
    assert np.isfinite(list(metrics["gru"].values())).all()
