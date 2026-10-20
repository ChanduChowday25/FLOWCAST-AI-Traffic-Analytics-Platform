"""Regenerate validation reports from existing FlowCast model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.splitting import split_chronologically
from src.targets import TRAFFIC_VOLUME_TARGET_30M
from src.train_decision_tree import (
    MODEL_ARTIFACT_NAME as DECISION_TREE_MODEL_NAME,
    PREPROCESSOR_ARTIFACT_NAME as DECISION_TREE_PREPROCESSOR_NAME,
    REPORT_NAME as DECISION_TREE_REPORT_NAME,
)
from src.train_gradient_boosting import (
    MODEL_ARTIFACT_NAME as GRADIENT_BOOSTING_MODEL_NAME,
    MODEL_PARAMETERS as GRADIENT_BOOSTING_PARAMETERS,
    PREPROCESSOR_ARTIFACT_NAME as GRADIENT_BOOSTING_PREPROCESSOR_NAME,
    REPORT_NAME as GRADIENT_BOOSTING_REPORT_NAME,
)
from src.train_linear_regression import (
    MODEL_ARTIFACT_NAME as LINEAR_MODEL_NAME,
    PREPROCESSOR_ARTIFACT_NAME as LINEAR_PREPROCESSOR_NAME,
    REPORT_NAME as LINEAR_REPORT_NAME,
    _validation_report,
)
from src.train_random_forest import (
    MODEL_ARTIFACT_NAME as RANDOM_FOREST_MODEL_NAME,
    MODEL_PARAMETERS as RANDOM_FOREST_PARAMETERS,
    PREPROCESSOR_ARTIFACT_NAME as RANDOM_FOREST_PREPROCESSOR_NAME,
    REPORT_NAME as RANDOM_FOREST_REPORT_NAME,
)
from src.utils import configured_path, load_config


MODEL_SPECS = (
    (
        "linear_regression",
        LINEAR_PREPROCESSOR_NAME,
        LINEAR_MODEL_NAME,
        LINEAR_REPORT_NAME,
        "sklearn.linear_model.LinearRegression",
        None,
    ),
    (
        "decision_tree",
        DECISION_TREE_PREPROCESSOR_NAME,
        DECISION_TREE_MODEL_NAME,
        DECISION_TREE_REPORT_NAME,
        "sklearn.tree.DecisionTreeRegressor",
        {"random_state": 42},
    ),
    (
        "random_forest",
        RANDOM_FOREST_PREPROCESSOR_NAME,
        RANDOM_FOREST_MODEL_NAME,
        RANDOM_FOREST_REPORT_NAME,
        "sklearn.ensemble.RandomForestRegressor",
        RANDOM_FOREST_PARAMETERS,
    ),
    (
        "gradient_boosting",
        GRADIENT_BOOSTING_PREPROCESSOR_NAME,
        GRADIENT_BOOSTING_MODEL_NAME,
        GRADIENT_BOOSTING_REPORT_NAME,
        "sklearn.ensemble.GradientBoostingRegressor",
        GRADIENT_BOOSTING_PARAMETERS,
    ),
)


def regenerate_validation_reports(
    *,
    input_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> list[Path]:
    """Recompute reports without fitting or changing saved model artifacts."""
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
    report_dir = (
        Path(reports_dir)
        if reports_dir is not None
        else configured_path(settings["paths"]["reports_dir"])
    )
    frame = pd.read_csv(input_file, parse_dates=["timestamp"])
    train, validation, _ = split_chronologically(frame)
    written: list[Path] = []

    for name, preprocessor_name, model_name, report_name, model_label, parameters in MODEL_SPECS:
        preprocessor_path = artifact_dir / preprocessor_name
        model_path = artifact_dir / model_name
        if not preprocessor_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"Missing {name} artifacts in {artifact_dir}")
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        report: dict[str, Any] = {
            "input_dataset": str(input_file),
            "target_column": TRAFFIC_VOLUME_TARGET_30M,
            "feature_columns": list(preprocessor.feature_columns),
            "model": model_label,
        }
        if name == "gradient_boosting":
            report.update(
                {
                    "xgboost_available": False,
                    "dependency_note": (
                        "XGBoost is not installed or declared; sklearn "
                        "GradientBoostingRegressor is used as the runnable fallback."
                    ),
                }
            )
        if parameters is not None:
            report["model_parameters"] = parameters
        report.update(
            {
                "split": {
                    "train_proportion": 0.70,
                    "validation_proportion": 0.15,
                    "test_proportion": 0.15,
                    "train_row_count": len(train),
                    "validation_row_count": len(validation),
                },
                "training": {
                    "row_count": len(train),
                    "valid_target_row_count": preprocessor.fit_row_count_,
                    "excluded_target_row_count": preprocessor.fit_missing_target_count_,
                    "timestamp_start": str(train["timestamp"].min()),
                    "timestamp_end": str(train["timestamp"].max()),
                },
                "validation": _validation_report(validation, preprocessor, model),
                "mape_policy": "Zero-valued targets are excluded because percentage error is undefined.",
                "preprocessor_artifact": str(preprocessor_path),
                "model_artifact": str(model_path),
            }
        )
        report_path = report_dir / report_name
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        written.append(report_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    args = parser.parse_args()
    for path in regenerate_validation_reports(
        input_path=args.input_path,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
    ):
        print(path)


if __name__ == "__main__":
    main()
