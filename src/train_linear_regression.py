"""Train and validate FlowCast's first Linear Regression model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.preprocessing import FlowCastPreprocessor
from src.splitting import split_chronologically
from src.targets import TRAFFIC_VOLUME_TARGET_30M
from src.utils import configured_path, load_config


MODEL_ARTIFACT_NAME = "linear_regression_model.joblib"
PREPROCESSOR_ARTIFACT_NAME = "linear_regression_preprocessor.joblib"
REPORT_NAME = "linear_regression_validation.json"


def _valid_target(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.notna() & np.isfinite(numeric) & numeric.ge(0)


def _with_invalid_targets_missing(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    target = pd.to_numeric(result[TRAFFIC_VOLUME_TARGET_30M], errors="coerce")
    result[TRAFFIC_VOLUME_TARGET_30M] = target.where(_valid_target(target))
    return result


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    actual = y_true.to_numpy(dtype=float)
    errors = y_pred - actual
    metrics = {"rmse": math.nan, "mae": math.nan, "mape": math.nan, "r2": math.nan}
    if len(actual):
        metrics["rmse"] = float(np.sqrt(np.mean(errors**2)))
        metrics["mae"] = float(np.mean(np.abs(errors)))
        nonzero = actual != 0
        if nonzero.any():
            metrics["mape"] = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100)
        if len(actual) >= 2 and not np.all(actual == actual[0]):
            metrics["r2"] = float(
                1 - np.sum(errors**2) / np.sum((actual - np.mean(actual)) ** 2)
            )
    return metrics


def _validation_report(
    frame: pd.DataFrame,
    preprocessor: FlowCastPreprocessor,
    model: LinearRegression,
) -> dict[str, Any]:
    valid_target = _valid_target(frame[TRAFFIC_VOLUME_TARGET_30M])
    supervised = preprocessor.transform_supervised(_with_invalid_targets_missing(frame))
    predictions = model.predict(supervised.features)
    metrics = _metrics(supervised.target.astype(float), predictions)
    return {
        "row_count": len(frame),
        "valid_prediction_count": supervised.retained_row_count,
        "excluded_target_row_count": int((~valid_target).sum()),
        "zero_target_count": int((valid_target & pd.to_numeric(frame[TRAFFIC_VOLUME_TARGET_30M], errors="coerce").eq(0)).sum()),
        "mape_eligible_count": int((supervised.target.astype(float).ne(0)).sum()),
        **metrics,
        "timestamp_start": str(frame["timestamp"].min()),
        "timestamp_end": str(frame["timestamp"].max()),
    }


def train_linear_regression(
    *,
    input_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Fit Linear Regression on training data and evaluate validation data."""
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
        raise ValueError("Cannot train Linear Regression: no valid training targets")
    model = LinearRegression()
    model.fit(training.features, training.target.astype(float))

    validation_report = _validation_report(validation, preprocessor, model)
    report = {
        "input_dataset": str(input_file),
        "target_column": TRAFFIC_VOLUME_TARGET_30M,
        "feature_columns": list(preprocessor.feature_columns),
        "model": "sklearn.linear_model.LinearRegression",
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


def _resolve_report_path(
    *,
    report_path: str | Path | None,
    models_dir: str | Path | None,
    settings: dict[str, Any],
    report_name: str,
) -> Path:
    if report_path is not None:
        return Path(report_path)
    if models_dir is not None:
        return Path(models_dir).parent / "reports" / report_name
    return configured_path(settings["paths"]["reports_dir"]) / report_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--report", dest="report_path", type=Path)
    args = parser.parse_args()
    artifacts = train_linear_regression(
        input_path=args.input_path,
        models_dir=args.models_dir,
        report_path=args.report_path,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
