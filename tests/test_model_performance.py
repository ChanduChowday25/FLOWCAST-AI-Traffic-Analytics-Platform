import json

import pandas as pd

from dashboard.model_performance import (
    METRICS,
    load_model_performance_reports,
    load_validation_predictions,
    road_validation_metrics,
)


def _validation_report(metrics: dict[str, float]) -> dict:
    return {"validation": metrics}


def _write_reports(tmp_path, *, include_all=True):
    metrics = {"rmse": 1.0, "mae": 2.0, "mape": 3.0, "r2": 0.8}
    reports = {
        "persistence_baseline_evaluation.json": _validation_report(metrics),
        "linear_regression_validation.json": _validation_report(metrics),
        "decision_tree_regression_validation.json": _validation_report(metrics),
        "random_forest_regression_validation.json": _validation_report(metrics),
        "gradient_boosting_regression_validation.json": _validation_report(metrics),
        "lstm_volume_validation.json": _validation_report(metrics),
        "gru_volume_validation.json": _validation_report(metrics),
        "all_models_validation_comparison.json": {
            "validation_only": True,
            "metrics": {
                "random_forest": {"rmse": 10, "mae": 11, "mape": 12, "r2": 0.9},
                "lstm": {"rmse": 20, "mae": 21, "mape": 22, "r2": 0.91},
                "gru": {"rmse": 30, "mae": 31, "mape": 32, "r2": 0.92},
            },
        },
        "final_test_evaluation.json": {
            "selected_candidate": "gru",
            "test_partition_only_for_evaluation": True,
            "metrics": {"rmse": 40, "mae": 41, "mape": 42, "r2": 0.93},
        },
    }
    for name, report in reports.items():
        if include_all or name in {"all_models_validation_comparison.json", "final_test_evaluation.json"}:
            (tmp_path / name).write_text(json.dumps(report), encoding="utf-8")


def test_loads_all_validation_metrics_and_separates_final_test(tmp_path):
    _write_reports(tmp_path)

    result = load_model_performance_reports(tmp_path)

    assert result["validation"]["approach"].tolist() == [
        "Persistence baseline",
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "LSTM",
        "GRU",
    ]
    assert result["validation"].loc[3, "rmse"] == 10
    assert result["validation"].loc[5, "r2"] == 0.91
    assert result["final_test"] == {
        "rmse": 40.0,
        "mae": 41.0,
        "mape": 42.0,
        "r2": 0.93,
    }


def test_missing_report_keeps_metrics_unavailable(tmp_path):
    _write_reports(tmp_path)
    (tmp_path / "linear_regression_validation.json").unlink()

    result = load_model_performance_reports(tmp_path)
    row = result["validation"].loc[
        result["validation"]["approach"].eq("Linear Regression")
    ].iloc[0]

    assert pd.isna(row[list(METRICS)].astype(float)).all()
    assert any("linear_regression_validation.json" in warning for warning in result["warnings"])


def test_malformed_report_is_reported_without_replacement_values(tmp_path):
    _write_reports(tmp_path)
    (tmp_path / "gru_volume_validation.json").write_text("{", encoding="utf-8")

    result = load_model_performance_reports(tmp_path)
    row = result["validation"].loc[result["validation"]["approach"].eq("GRU")].iloc[0]

    assert row["rmse"] == 30
    assert row["mae"] == 31
    assert any("Could not read JSON report" in warning for warning in result["warnings"])


def test_incomplete_metric_is_unavailable_and_test_metrics_do_not_enter_validation(tmp_path):
    _write_reports(tmp_path)
    report_path = tmp_path / "gru_volume_validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["validation"]["mae"]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = load_model_performance_reports(tmp_path)
    gru = result["validation"].loc[result["validation"]["approach"].eq("GRU")].iloc[0]

    assert gru["rmse"] == 30
    assert gru["mae"] == 31
    assert result["final_test"]["rmse"] == 40
    assert result["final_test"]["rmse"] not in result["validation"]["rmse"].tolist()


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "road_id": ["A", "A", "B"],
            "actual_target_30m": [10.0, 0.0, 20.0],
            "persistence_prediction": [12.0, 3.0, 16.0],
            "linear_regression_prediction": [10.0, 4.0, 24.0],
            "decision_tree_prediction": [11.0, 5.0, 22.0],
            "random_forest_prediction": [9.0, 6.0, 18.0],
            "gradient_boosting_prediction": [10.0, 7.0, 21.0],
        }
    )


def test_road_metrics_use_shared_valid_rows_and_report_observations():
    predictions = _prediction_frame()
    predictions["random_forest_prediction"] = predictions[
        "random_forest_prediction"
    ].astype(object)
    predictions.loc[2, "random_forest_prediction"] = "invalid"

    metrics, observations = road_validation_metrics(predictions, "A")

    assert observations == 2
    assert metrics["observations"].tolist() == [2] * 5
    assert metrics.loc[0, "rmse"] == pd.Series([2.0, 3.0]).pow(2).mean() ** 0.5
    assert metrics.loc[0, "mape"] == 20.0


def test_saved_prediction_loader_handles_missing_and_malformed_columns(tmp_path):
    frame = _prediction_frame()
    frame.to_csv(tmp_path / "validation_prediction_comparison.csv", index=False)

    loaded, warning = load_validation_predictions(tmp_path)

    assert warning is None
    assert loaded is not None
    assert loaded.shape == frame.shape

    frame.drop(columns=["gru_prediction"], errors="ignore").to_csv(
        tmp_path / "validation_prediction_comparison.csv", index=False
    )
    loaded, warning = load_validation_predictions(tmp_path)
    assert loaded is not None
    assert warning is None

    (tmp_path / "validation_prediction_comparison.csv").write_text(
        "road_id,actual_target_30m\nA,10\n", encoding="utf-8"
    )
    loaded, warning = load_validation_predictions(tmp_path)
    assert loaded is None
    assert warning is not None
