"""Run and report the FlowCast persistence baseline on chronological splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.baseline import evaluate_persistence
from src.splitting import split_chronologically
from src.utils import configured_path, load_config


def _partition_report(frame: pd.DataFrame) -> dict[str, Any]:
    evaluation = evaluate_persistence(frame)
    metrics = evaluation.metrics
    return {
        "row_count": evaluation.input_row_count,
        "valid_prediction_count": evaluation.valid_prediction_count,
        "excluded_row_count": evaluation.input_row_count - evaluation.valid_prediction_count,
        "missing_baseline_count": evaluation.missing_baseline_count,
        "invalid_target_count": evaluation.invalid_target_count,
        "zero_target_count": evaluation.zero_target_count,
        "mape_eligible_count": evaluation.mape_eligible_count,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "r2": metrics["r2"],
        "timestamp_start": str(frame["timestamp"].min()),
        "timestamp_end": str(frame["timestamp"].max()),
    }


def run_persistence_evaluation(
    *,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Evaluate persistence on validation and test data and write a JSON report."""
    settings = config or load_config()
    input_file = (
        Path(input_path)
        if input_path is not None
        else configured_path(settings["data"]["processed_dir"]) / "flowcast_target_30m.csv"
    )
    output_file = (
        Path(output_path)
        if output_path is not None
        else configured_path(settings["paths"]["reports_dir"]) / "persistence_baseline_evaluation.json"
    )
    if not input_file.is_file():
        raise FileNotFoundError(f"Processed target dataset is missing: {input_file}")

    frame = pd.read_csv(input_file, parse_dates=["timestamp"])
    train, validation, test = split_chronologically(frame)
    report = {
        "input_dataset": str(input_file),
        "target_column": "traffic_volume_target_30m",
        "persistence_input_column": "traffic_volume_lag_1",
        "split": {
            "train_proportion": 0.70,
            "validation_proportion": 0.15,
            "test_proportion": 0.15,
            "train_row_count": len(train),
            "validation_row_count": len(validation),
            "test_row_count": len(test),
        },
        "validation": _partition_report(validation),
        "test": _partition_report(test),
        "mape_policy": "Zero-valued targets are excluded because percentage error is undefined.",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path)
    parser.add_argument("--output", dest="output_path", type=Path)
    args = parser.parse_args()
    output = run_persistence_evaluation(
        input_path=args.input_path,
        output_path=args.output_path,
    )
    print(f"Persistence baseline evaluation written to {output}")


if __name__ == "__main__":
    main()
