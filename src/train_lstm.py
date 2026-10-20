"""Train FlowCast's CPU LSTM volume regressor on train and validation sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dl_models import LSTMRegressor
from src.dl_training import TrainingConfig, load_sequence_dataset, train_recurrent_model
from src.utils import configured_path, load_config


MODEL_NAME = "lstm"


def train_lstm(
    *,
    input_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    curve_path: str | Path | None = None,
    config: TrainingConfig | None = None,
    smoke_test: bool = False,
) -> dict[str, Path]:
    settings = load_config()
    training_config = (
        TrainingConfig(max_epochs=1, batch_size=64) if smoke_test else config
    )
    artifact_dir = Path(models_dir) if models_dir is not None else configured_path(settings["paths"]["models_dir"])
    report_file = Path(report_path) if report_path is not None else configured_path(settings["paths"]["reports_dir"]) / "lstm_volume_validation.json"
    curve_file = Path(curve_path) if curve_path is not None else configured_path(settings["paths"]["figures_dir"]) / "lstm_volume_training_curve.png"
    return train_recurrent_model(
        model_factory=LSTMRegressor,
        model_name=MODEL_NAME,
        dataset=load_sequence_dataset(input_path),
        artifact_dir=artifact_dir,
        report_path=report_file,
        curve_path=curve_file,
        config=training_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--report", dest="report_path", type=Path)
    parser.add_argument("--curve", dest="curve_path", type=Path)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run exactly one CPU training epoch with batch size 64.",
    )
    args = parser.parse_args()
    for name, path in train_lstm(
        input_path=args.input_path,
        models_dir=args.models_dir,
        report_path=args.report_path,
        curve_path=args.curve_path,
        smoke_test=args.smoke_test,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
