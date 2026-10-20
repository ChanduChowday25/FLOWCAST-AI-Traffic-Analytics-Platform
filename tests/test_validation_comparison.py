import json

import joblib
import pandas as pd

from src.create_validation_comparison import create_validation_comparison
from src.selection import MODEL_FEATURE_COLUMNS
from src.targets import TRAFFIC_VOLUME_TARGET_30M
from src.train_decision_tree import train_decision_tree
from src.train_gradient_boosting import train_gradient_boosting
from src.train_linear_regression import train_linear_regression
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


def test_models_with_temporary_directories_do_not_write_project_reports(tmp_path):
    source = tmp_path / "target.csv"
    _frame().to_csv(source, index=False)
    project_report = tmp_path / "production-reports" / "existing.json"
    project_report.parent.mkdir()
    project_report.write_text(json.dumps({"untouched": True}), encoding="utf-8")

    trainers = (
        train_linear_regression,
        train_decision_tree,
        train_random_forest,
        train_gradient_boosting,
    )
    for trainer in trainers:
        artifacts = trainer(input_path=source, models_dir=tmp_path / "models")
        assert artifacts["report"] == tmp_path / "reports" / artifacts["report"].name
        assert artifacts["report"].parent != tmp_path / "production-reports"
    assert json.loads(project_report.read_text(encoding="utf-8")) == {"untouched": True}


def test_comparison_uses_same_validation_rows_and_source_order(tmp_path):
    source = tmp_path / "target.csv"
    frame = _frame()
    frame.to_csv(source, index=False)
    models_dir = tmp_path / "models"
    train_linear_regression(input_path=source, models_dir=models_dir)
    train_decision_tree(input_path=source, models_dir=models_dir)
    train_random_forest(input_path=source, models_dir=models_dir)
    train_gradient_boosting(input_path=source, models_dir=models_dir)

    output = create_validation_comparison(
        input_path=source,
        models_dir=models_dir,
        output_path=tmp_path / "reports" / "comparison.csv",
    )
    comparison = pd.read_csv(output)

    assert len(comparison) == 2
    assert comparison["original_row_index"].tolist() == [14, 15]
    assert comparison["road_id"].tolist() == ["A", "B"]
    assert comparison["timestamp"].tolist() == ["2025-01-01 03:30:00", "2025-01-01 03:30:00"]
    assert comparison.notna().all().all()
    assert list(comparison.columns) == [
        "original_row_index",
        "road_id",
        "timestamp",
        "actual_target_30m",
        "persistence_prediction",
        "linear_regression_prediction",
        "decision_tree_prediction",
        "random_forest_prediction",
        "gradient_boosting_prediction",
    ]
    for artifact in models_dir.glob("*.joblib"):
        assert joblib.load(artifact) is not None
