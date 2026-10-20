"""Compare saved LSTM and Random Forest predictions on common validation rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.dl_models import LSTMRegressor
from src.sequence_data import (
    DL_FEATURE_COLUMNS,
    SequencePreprocessor,
    build_sequences,
)
from src.splitting import split_chronologically
from src.targets import TRAFFIC_VOLUME_TARGET_30M
from src.train_linear_regression import _metrics, _valid_target
from src.train_random_forest import (
    MODEL_ARTIFACT_NAME as RANDOM_FOREST_MODEL_NAME,
    PREPROCESSOR_ARTIFACT_NAME as RANDOM_FOREST_PREPROCESSOR_NAME,
)
from src.utils import configured_path, load_config


OUTPUT_NAME = "lstm_vs_random_forest_validation.json"
SUMMARY_NAME = "lstm_vs_random_forest_validation.md"


def align_validation_predictions(
    lstm_rows: pd.DataFrame,
    random_forest_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return rows sharing road, timestamp, and exactly equal targets."""
    required = {"road_id", "timestamp", "actual_target_30m", "prediction"}
    for name, frame in (("LSTM", lstm_rows), ("Random Forest", random_forest_rows)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} alignment rows are missing: {', '.join(missing)}")

    left = lstm_rows.rename(
        columns={
            "actual_target_30m": "lstm_actual_target_30m",
            "prediction": "lstm_prediction",
        }
    )
    right = random_forest_rows.rename(
        columns={
            "actual_target_30m": "random_forest_actual_target_30m",
            "prediction": "random_forest_prediction",
        }
    )
    merged = left.merge(
        right,
        on=["road_id", "timestamp"],
        how="inner",
        validate="one_to_one",
    )
    target_match = (
        merged["lstm_actual_target_30m"] == merged["random_forest_actual_target_30m"]
    )
    aligned = merged.loc[target_match].copy()
    aligned["actual_target_30m"] = aligned["lstm_actual_target_30m"]
    aligned = aligned.loc[
        :,
        ["road_id", "timestamp", "actual_target_30m", "lstm_prediction", "random_forest_prediction"],
    ]
    exclusions = {
        "lstm_validation_rows": len(lstm_rows),
        "random_forest_valid_validation_rows": len(random_forest_rows),
        "shared_road_timestamp_rows": len(merged),
        "target_mismatch_rows": int((~target_match).sum()),
        "common_rows": len(aligned),
        "lstm_only_rows": len(lstm_rows) - len(merged),
        "random_forest_only_rows": len(random_forest_rows) - len(merged),
    }
    return aligned, exclusions


def calculate_comparison_metrics(
    aligned: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Calculate the shared-set regression metrics for both models."""
    if aligned.empty:
        raise ValueError("Cannot calculate comparison metrics without common rows")
    actual = aligned["actual_target_30m"].astype(float)
    return {
        "lstm": _metrics(actual, aligned["lstm_prediction"].to_numpy(dtype=float)),
        "random_forest": _metrics(
            actual, aligned["random_forest_prediction"].to_numpy(dtype=float)
        ),
    }


def _lstm_validation_predictions(
    input_path: Path,
    artifact_dir: Path,
) -> pd.DataFrame:
    required_columns = tuple(
        dict.fromkeys((*DL_FEATURE_COLUMNS, "timestamp", TRAFFIC_VOLUME_TARGET_30M))
    )
    frame = pd.read_csv(
        input_path,
        parse_dates=["timestamp"],
        usecols=list(required_columns),
    )
    train, validation, _ = split_chronologically(frame)
    preprocessor = SequencePreprocessor(feature_columns=DL_FEATURE_COLUMNS)
    preprocessor.fit(train)
    raw_validation = build_sequences(
        validation,
        sequence_length=12,
        feature_columns=DL_FEATURE_COLUMNS,
    )
    validation_partition = preprocessor.transform_sequences(raw_validation)
    model_path = artifact_dir / "lstm_volume_model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing saved LSTM model artifact: {model_path}")
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    config = artifact["training_config"]
    model = LSTMRegressor(
        input_size=int(artifact["input_size"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    features = torch.from_numpy(validation_partition.features).float()
    predictions: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(features), batch_size=64, shuffle=False)
    with torch.no_grad():
        for (batch,) in loader:
            predictions.append(model(batch).numpy())
    return pd.DataFrame(
        {
            "road_id": validation_partition.metadata["road_id"].to_numpy(),
            "timestamp": pd.to_datetime(
                validation_partition.metadata["endpoint_timestamp"]
            ).to_numpy(),
            "actual_target_30m": validation_partition.targets.astype(float),
            "prediction": np.concatenate(predictions),
        }
    )


def _random_forest_validation_predictions(
    input_path: Path,
    artifact_dir: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(input_path, parse_dates=["timestamp"])
    _, validation, _ = split_chronologically(frame)
    valid = _valid_target(validation[TRAFFIC_VOLUME_TARGET_30M])
    validation = validation.loc[valid].copy()
    preprocessor_path = artifact_dir / RANDOM_FOREST_PREPROCESSOR_NAME
    model_path = artifact_dir / RANDOM_FOREST_MODEL_NAME
    if not preprocessor_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            f"Missing Random Forest artifacts: {preprocessor_path}, {model_path}"
        )
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    return pd.DataFrame(
        {
            "road_id": validation["road_id"].to_numpy(),
            "timestamp": validation["timestamp"].to_numpy(),
            "actual_target_30m": validation[TRAFFIC_VOLUME_TARGET_30M].to_numpy(
                dtype=float
            ),
            "prediction": model.predict(preprocessor.transform(validation)),
        }
    )


def create_lstm_random_forest_comparison(
    *,
    input_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Write a validation-only comparison without fitting either model."""
    settings = config or load_config()
    input_file = (
        Path(input_path)
        if input_path is not None
        else configured_path(settings["data"]["processed_dir"])
        / "flowcast_target_30m.csv"
    )
    artifact_dir = (
        Path(models_dir)
        if models_dir is not None
        else configured_path(settings["paths"]["models_dir"])
    )
    output_file = (
        Path(output_path)
        if output_path is not None
        else configured_path(settings["paths"]["reports_dir"]) / OUTPUT_NAME
    )
    summary_file = (
        Path(summary_path)
        if summary_path is not None
        else output_file.with_name(SUMMARY_NAME)
    )
    lstm_rows = _lstm_validation_predictions(input_file, artifact_dir)
    random_forest_rows = _random_forest_validation_predictions(input_file, artifact_dir)
    aligned, exclusions = align_validation_predictions(lstm_rows, random_forest_rows)
    metrics = calculate_comparison_metrics(aligned)
    if not all(
        math.isfinite(value)
        for model_metrics in metrics.values()
        for value in model_metrics.values()
    ):
        raise ValueError("Comparison metrics must be finite")

    result = {
        "comparison": "lstm_vs_random_forest",
        "validation_only": True,
        "alignment_keys": ["road_id", "timestamp", "actual_target_30m"],
        "exclusions": exclusions,
        "metrics": metrics,
        "source_reports": {
            "lstm": str(configured_path(settings["paths"]["reports_dir"]) / "lstm_volume_validation.json"),
            "random_forest": str(configured_path(settings["paths"]["reports_dir"]) / "random_forest_regression_validation.json"),
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary_file.write_text(
        "\n".join(
            [
                "# LSTM vs Random Forest validation comparison",
                "",
                f"Common validation examples: {exclusions['common_rows']:,}",
                "",
                "| Model | RMSE | MAE | MAPE | R² |",
                "|---|---:|---:|---:|---:|",
                *[
                    f"| {name.replace('_', ' ').title()} | {values['rmse']:.6f} | "
                    f"{values['mae']:.6f} | {values['mape']:.6f}% | {values['r2']:.6f} |"
                    for name, values in metrics.items()
                ],
                "",
                "Rows were aligned by exact road, timestamp, and target value. "
                "The initial validation timestamps unavailable to sequence windows "
                "were excluded from the common set.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--output", dest="output_path", type=Path)
    parser.add_argument("--summary", dest="summary_path", type=Path)
    args = parser.parse_args()
    print(
        create_lstm_random_forest_comparison(
            input_path=args.input_path,
            models_dir=args.models_dir,
            output_path=args.output_path,
            summary_path=args.summary_path,
        )
    )


if __name__ == "__main__":
    main()
