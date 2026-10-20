import json

import joblib
import numpy as np
import pandas as pd

from src.selection import MODEL_FEATURE_COLUMNS
from src.targets import TRAFFIC_VOLUME_TARGET_30M
from src.train_random_forest import train_random_forest


def _frame() -> pd.DataFrame:
    rows = []
    timestamps = pd.date_range("2025-01-01", periods=10, freq="30min")
    for timestamp_index, timestamp in enumerate(timestamps):
        for road_index, road_id in enumerate(["A", "B"]):
            row = {
                "timestamp": timestamp,
                "road_id": road_id,
                TRAFFIC_VOLUME_TARGET_30M: float(100 + timestamp_index * 10 + road_index),
            }
            for feature_index, column in enumerate(
                column for column in MODEL_FEATURE_COLUMNS if column != "road_id"
            ):
                row[column] = float(timestamp_index + feature_index + road_index)
            rows.append(row)
    return pd.DataFrame(rows)


def test_training_saves_distinct_forest_artifacts_and_validation_report(tmp_path):
    source = tmp_path / "target.csv"
    frame = _frame()
    frame.loc[0, TRAFFIC_VOLUME_TARGET_30M] = np.nan
    frame.to_csv(source, index=False)

    artifacts = train_random_forest(
        input_path=source,
        models_dir=tmp_path / "models",
        report_path=tmp_path / "reports" / "validation.json",
    )

    assert all(path.is_file() for path in artifacts.values())
    assert artifacts["model"].name != "linear_regression_model.joblib"
    assert artifacts["model"].name != "decision_tree_regression_model.joblib"
    report = json.loads(artifacts["report"].read_text(encoding="utf-8"))
    assert report["model"] == "sklearn.ensemble.RandomForestRegressor"
    assert report["model_parameters"] == {
        "n_estimators": 50,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": 42,
    }
    assert report["training"]["excluded_target_row_count"] == 1
    assert report["validation"]["row_count"] == 2
    assert report["validation"]["valid_prediction_count"] == 2


def test_loaded_artifacts_keep_target_separate_and_prediction_shape(tmp_path):
    source = tmp_path / "target.csv"
    frame = _frame()
    frame.to_csv(source, index=False)
    artifacts = train_random_forest(input_path=source, models_dir=tmp_path / "models")

    preprocessor = joblib.load(artifacts["preprocessor"])
    model = joblib.load(artifacts["model"])
    validation = frame.iloc[-2:].copy()
    transformed = preprocessor.transform(validation)

    assert TRAFFIC_VOLUME_TARGET_30M not in transformed.columns
    assert transformed.shape[0] == len(validation)
    assert model.predict(transformed).shape == (len(validation),)


def test_invalid_targets_are_excluded_and_chronology_is_preserved(tmp_path):
    source = tmp_path / "target.csv"
    frame = _frame()
    frame[TRAFFIC_VOLUME_TARGET_30M] = frame[TRAFFIC_VOLUME_TARGET_30M].astype(object)
    frame.loc[0, TRAFFIC_VOLUME_TARGET_30M] = -1
    frame.loc[1, TRAFFIC_VOLUME_TARGET_30M] = "invalid"
    frame.to_csv(source, index=False)

    artifacts = train_random_forest(input_path=source, models_dir=tmp_path / "models")
    report = json.loads(artifacts["report"].read_text(encoding="utf-8"))

    assert report["training"]["excluded_target_row_count"] == 2
    assert report["training"]["timestamp_end"] < report["validation"]["timestamp_start"]
    assert TRAFFIC_VOLUME_TARGET_30M not in report["feature_columns"]


def test_same_random_state_produces_same_predictions(tmp_path):
    source = tmp_path / "target.csv"
    frame = _frame()
    frame.to_csv(source, index=False)

    first = train_random_forest(input_path=source, models_dir=tmp_path / "first")
    second = train_random_forest(input_path=source, models_dir=tmp_path / "second")

    first_preprocessor = joblib.load(first["preprocessor"])
    first_model = joblib.load(first["model"])
    second_preprocessor = joblib.load(second["preprocessor"])
    second_model = joblib.load(second["model"])
    validation = frame.iloc[-2:].copy()
    first_predictions = first_model.predict(first_preprocessor.transform(validation))
    second_predictions = second_model.predict(second_preprocessor.transform(validation))

    np.testing.assert_array_equal(first_predictions, second_predictions)
