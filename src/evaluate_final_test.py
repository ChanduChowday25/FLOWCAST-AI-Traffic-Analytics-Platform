"""Evaluate the selected GRU artifact on the untouched chronological test split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dl_models import GRURegressor
from src.sequence_data import DL_FEATURE_COLUMNS, SequencePreprocessor, build_sequences
from src.splitting import split_chronologically
from src.targets import TRAFFIC_VOLUME_TARGET_30M
from src.train_linear_regression import _metrics
from src.utils import configured_path, load_config


OUTPUT_NAME = "final_test_evaluation.json"
SUMMARY_NAME = "final_test_evaluation.md"


def evaluate_gru_test(
    *,
    input_path: str | Path | None = None,
    model_path: str | Path | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Evaluate the saved GRU without fitting on validation or test data."""
    settings = config or load_config()
    input_file = (
        Path(input_path)
        if input_path is not None
        else configured_path(settings["data"]["processed_dir"])
        / "flowcast_target_30m.csv"
    )
    saved_model = (
        Path(model_path)
        if model_path is not None
        else configured_path(settings["paths"]["models_dir"])
        / "gru_volume_model.pt"
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
    required_columns = tuple(
        dict.fromkeys((*DL_FEATURE_COLUMNS, "timestamp", TRAFFIC_VOLUME_TARGET_30M))
    )
    frame = pd.read_csv(
        input_file,
        parse_dates=["timestamp"],
        usecols=list(required_columns),
    )
    train, _, test = split_chronologically(frame)
    preprocessor = SequencePreprocessor(feature_columns=DL_FEATURE_COLUMNS)
    preprocessor.fit(train)
    raw_test = build_sequences(
        test,
        sequence_length=12,
        feature_columns=DL_FEATURE_COLUMNS,
    )
    test_partition = preprocessor.transform_sequences(raw_test)

    if not saved_model.is_file():
        raise FileNotFoundError(f"Missing saved GRU model artifact: {saved_model}")
    artifact = torch.load(saved_model, map_location="cpu", weights_only=False)
    training_config = artifact["training_config"]
    model = GRURegressor(
        input_size=int(artifact["input_size"]),
        hidden_size=int(training_config["hidden_size"]),
        num_layers=int(training_config["num_layers"]),
        dropout=float(training_config["dropout"]),
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    features = torch.from_numpy(test_partition.features).float()
    predictions: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(features), batch_size=64, shuffle=False)
    with torch.no_grad():
        for (batch,) in loader:
            predictions.append(model(batch).numpy())
    predicted = (
        np.concatenate(predictions)
        if predictions
        else np.empty(0, dtype="float64")
    )
    actual = test_partition.targets.astype(float)
    metrics = _metrics(pd.Series(actual, dtype="float64"), predicted)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Final test metrics must be finite")

    test_row_count = len(test)
    valid_prediction_count = len(actual)
    exclusions = {
        "test_row_count": test_row_count,
        "valid_test_prediction_count": valid_prediction_count,
        "excluded_test_examples": test_row_count - valid_prediction_count,
        **raw_test.exclusions,
    }
    result = {
        "evaluation": "final_gru_test",
        "selected_candidate": "gru",
        "selection_basis": {
            "validation_comparison": "reports/all_models_validation_comparison.json",
            "reason": "GRU led on RMSE, MAE, and R2 on the common validation set; "
            "LSTM and Random Forest remain comparison baselines.",
        },
        "validation_only_for_selection": True,
        "test_partition_only_for_evaluation": True,
        "sequence_history_policy": (
            "Test sequences were built from the test partition only. The first "
            "11 test rows per road were excluded when they lacked 12 test rows "
            "of contiguous history."
        ),
        "preprocessing_policy": (
            "Sequence preprocessing was fitted on training rows only and applied "
            "to the test sequences without refitting."
        ),
        "model_artifact": str(saved_model),
        "model_configuration": training_config,
        "sequence_length": 12,
        "exclusions": exclusions,
        "metrics": metrics,
        "test_boundaries": {
            "timestamp_start": str(test["timestamp"].min()),
            "timestamp_end": str(test["timestamp"].max()),
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary_file.write_text(
        "\n".join(
            [
                "# Final GRU test evaluation",
                "",
                "GRU was evaluated as the primary candidate because it led the "
                "common validation comparison on RMSE, MAE, and R2. LSTM and "
                "Random Forest remain comparison baselines.",
                "",
                f"Valid test predictions: {valid_prediction_count:,}",
                f"Excluded test examples: {test_row_count - valid_prediction_count:,}",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| RMSE | {metrics['rmse']:.6f} |",
                f"| MAE | {metrics['mae']:.6f} |",
                f"| MAPE | {metrics['mape']:.6f}% |",
                f"| R² | {metrics['r2']:.6f} |",
                "",
                "Test sequences used only rows from the test partition. "
                "Preprocessing was fitted on training rows only; no test data "
                "was used for training or tuning.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path)
    parser.add_argument("--model", dest="model_path", type=Path)
    parser.add_argument("--output", dest="output_path", type=Path)
    parser.add_argument("--summary", dest="summary_path", type=Path)
    args = parser.parse_args()
    print(
        evaluate_gru_test(
            input_path=args.input_path,
            model_path=args.model_path,
            output_path=args.output_path,
            summary_path=args.summary_path,
        )
    )


if __name__ == "__main__":
    main()
