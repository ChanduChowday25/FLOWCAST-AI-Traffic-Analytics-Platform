"""Train and validate FlowCast's Decision Tree regression model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.preprocessing import FlowCastPreprocessor
from src.splitting import split_chronologically
from src.targets import TRAFFIC_VOLUME_TARGET_30M
from src.train_linear_regression import (
    _resolve_report_path,
    _validation_report,
    _with_invalid_targets_missing,
)
from src.utils import configured_path, load_config


MODEL_ARTIFACT_NAME = "decision_tree_regression_model.joblib"
PREPROCESSOR_ARTIFACT_NAME = "decision_tree_regression_preprocessor.joblib"
REPORT_NAME = "decision_tree_regression_validation.json"


def train_decision_tree(
    *,
    input_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Fit a Decision Tree on training data and evaluate validation data."""
    settings = config or load_config()
    input_file = (
        Path(input_path)
        if input_path is not None
        else configured_path(settings["data"]["processed_dir"]) / "flowcast_target_30m.csv"
    )
    artifact_dir = (
        Path(models_dir)
        if models_dir is not None
        else configured_path(settings["paths"]["models_dir"])
    )
    report_file = _resolve_report_path(
        report_path=report_path,
        models_dir=models_dir,
        settings=settings,
        report_name=REPORT_NAME,
    )
    if not input_file.is_file():
        raise FileNotFoundError(f"Processed target dataset is missing: {input_file}")

    frame = pd.read_csv(input_file, parse_dates=["timestamp"])
    train, validation, _ = split_chronologically(frame)
    clean_train = _with_invalid_targets_missing(train)

    preprocessor = FlowCastPreprocessor()
    training = preprocessor.fit_transform_supervised(clean_train)
    if training.retained_row_count == 0:
        raise ValueError("Cannot train Decision Tree: no valid training targets")

    model = DecisionTreeRegressor(random_state=42)
    model.fit(training.features, training.target.astype(float))
    validation_report = _validation_report(validation, preprocessor, model)

    report = {
        "input_dataset": str(input_file),
        "target_column": TRAFFIC_VOLUME_TARGET_30M,
        "feature_columns": list(preprocessor.feature_columns),
        "model": "sklearn.tree.DecisionTreeRegressor",
        "model_parameters": {"random_state": 42},
        "split": {
            "train_proportion": 0.70,
            "validation_proportion": 0.15,
            "test_proportion": 0.15,
            "train_row_count": len(train),
            "validation_row_count": len(validation),
        },
        "training": {
            "row_count": len(train),
            "valid_target_row_count": training.retained_row_count,
            "excluded_target_row_count": training.missing_target_count,
            "timestamp_start": str(train["timestamp"].min()),
            "timestamp_end": str(train["timestamp"].max()),
        },
        "validation": validation_report,
        "mape_policy": "Zero-valued targets are excluded because percentage error is undefined.",
        "preprocessor_artifact": str(artifact_dir / PREPROCESSOR_ARTIFACT_NAME),
        "model_artifact": str(artifact_dir / MODEL_ARTIFACT_NAME),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, artifact_dir / PREPROCESSOR_ARTIFACT_NAME)
    joblib.dump(model, artifact_dir / MODEL_ARTIFACT_NAME)
    report_file.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return {
        "preprocessor": artifact_dir / PREPROCESSOR_ARTIFACT_NAME,
        "model": artifact_dir / MODEL_ARTIFACT_NAME,
        "report": report_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--report", dest="report_path", type=Path)
    args = parser.parse_args()
    artifacts = train_decision_tree(
        input_path=args.input_path,
        models_dir=args.models_dir,
        report_path=args.report_path,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
