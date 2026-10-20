"""Create an aligned, validation-only prediction comparison for FlowCast."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.baseline import PERSISTENCE_PREDICTION_COLUMN, predict_persistence
from src.splitting import split_chronologically
from src.targets import TRAFFIC_VOLUME_TARGET_30M
from src.train_decision_tree import (
    MODEL_ARTIFACT_NAME as DECISION_TREE_MODEL_NAME,
    PREPROCESSOR_ARTIFACT_NAME as DECISION_TREE_PREPROCESSOR_NAME,
)
from src.train_gradient_boosting import (
    MODEL_ARTIFACT_NAME as GRADIENT_BOOSTING_MODEL_NAME,
    PREPROCESSOR_ARTIFACT_NAME as GRADIENT_BOOSTING_PREPROCESSOR_NAME,
)
from src.train_linear_regression import (
    MODEL_ARTIFACT_NAME as LINEAR_MODEL_NAME,
    PREPROCESSOR_ARTIFACT_NAME as LINEAR_PREPROCESSOR_NAME,
    _valid_target,
)
from src.train_random_forest import (
    MODEL_ARTIFACT_NAME as RANDOM_FOREST_MODEL_NAME,
    PREPROCESSOR_ARTIFACT_NAME as RANDOM_FOREST_PREPROCESSOR_NAME,
)
from src.utils import configured_path, load_config


COMPARISON_NAME = "validation_prediction_comparison.csv"
MODEL_FILES = {
    "linear_regression": (LINEAR_PREPROCESSOR_NAME, LINEAR_MODEL_NAME),
    "decision_tree": (DECISION_TREE_PREPROCESSOR_NAME, DECISION_TREE_MODEL_NAME),
    "random_forest": (RANDOM_FOREST_PREPROCESSOR_NAME, RANDOM_FOREST_MODEL_NAME),
    "gradient_boosting": (GRADIENT_BOOSTING_PREPROCESSOR_NAME, GRADIENT_BOOSTING_MODEL_NAME),
}


def create_validation_comparison(
    *,
    input_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Write predictions for the common valid-target validation rows only."""
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
    output_file = (
        Path(output_path)
        if output_path is not None
        else configured_path(settings["paths"]["reports_dir"]) / COMPARISON_NAME
    )
    if not input_file.is_file():
        raise FileNotFoundError(f"Processed target dataset is missing: {input_file}")

    frame = pd.read_csv(input_file, parse_dates=["timestamp"])
    _, validation, _ = split_chronologically(frame)
    valid_target = _valid_target(validation[TRAFFIC_VOLUME_TARGET_30M])
    common = validation.loc[valid_target].copy()
    persistence = predict_persistence(common)
    common = common.loc[persistence[PERSISTENCE_PREDICTION_COLUMN].notna()].copy()
    persistence = persistence.loc[common.index]

    result = pd.DataFrame(
        {
            "original_row_index": common.index,
            "road_id": common["road_id"].to_numpy(),
            "timestamp": common["timestamp"].to_numpy(),
            "actual_target_30m": common[TRAFFIC_VOLUME_TARGET_30M].to_numpy(dtype=float),
            "persistence_prediction": persistence[PERSISTENCE_PREDICTION_COLUMN].to_numpy(dtype=float),
        },
        index=common.index,
    )
    for model_name, (preprocessor_name, model_name_file) in MODEL_FILES.items():
        preprocessor_path = artifact_dir / preprocessor_name
        model_path = artifact_dir / model_name_file
        if not preprocessor_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(
                f"Missing saved {model_name} artifacts: {preprocessor_path}, {model_path}"
            )
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        predictions = model.predict(preprocessor.transform(common))
        if len(predictions) != len(result):
            raise ValueError(f"{model_name} prediction count does not match validation rows")
        result[f"{model_name}_prediction"] = predictions

    expected_columns = [
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
    result = result.loc[:, expected_columns]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--output", dest="output_path", type=Path)
    args = parser.parse_args()
    print(
        create_validation_comparison(
            input_path=args.input_path,
            models_dir=args.models_dir,
            output_path=args.output_path,
        )
    )


if __name__ == "__main__":
    main()
