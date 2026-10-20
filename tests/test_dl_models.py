import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.dl_models import GRURegressor, LSTMRegressor
from src.dl_training import TrainingConfig
from src.sequence_data import DL_NUMERIC_FEATURE_COLUMNS, prepare_sequence_dataset
from src.targets import TRAFFIC_VOLUME_TARGET_30M
from src.train_gru import train_gru
from src.train_lstm import train_lstm


def _frame(periods=100, roads=("A", "B")):
    rows = []
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="30min")
    for road_number, road_id in enumerate(roads):
        for position, timestamp in enumerate(timestamps):
            row = {
                "road_id": road_id,
                "timestamp": timestamp,
                TRAFFIC_VOLUME_TARGET_30M: float(100 + position + road_number),
            }
            for column_number, column in enumerate(DL_NUMERIC_FEATURE_COLUMNS):
                row[column] = float(position + column_number + road_number)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("model_class", [LSTMRegressor, GRURegressor])
def test_recurrent_models_return_finite_scalar_predictions(model_class):
    torch.manual_seed(42)
    model = model_class(input_size=36)
    inputs = torch.randn(4, 12, 36)

    predictions = model(inputs)

    assert predictions.shape == (4,)
    assert torch.isfinite(predictions).all()


@pytest.mark.parametrize(("trainer", "model_name"), [(train_lstm, "lstm"), (train_gru, "gru")])
def test_training_saves_reloadable_artifacts_and_validation_report(
    trainer, model_name, tmp_path, capsys
):
    source = tmp_path / "synthetic.csv"
    frame = _frame()
    frame.to_csv(source, index=False)
    artifact_dir = tmp_path / "models"
    report_path = tmp_path / "reports" / f"{model_name}.json"
    curve_path = tmp_path / "figures" / f"{model_name}.png"
    config = TrainingConfig(max_epochs=3, early_stopping_patience=1, seed=7)

    artifacts = trainer(
        input_path=source,
        models_dir=artifact_dir,
        report_path=report_path,
        curve_path=curve_path,
        config=config,
    )

    assert all(path.is_file() for path in artifacts.values())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    history = json.loads(artifacts["history"].read_text(encoding="utf-8"))
    epoch_lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("Training: epoch ")
    ]
    assert report["model"] == model_name
    assert report["sequence_length"] == 12
    assert report["device"] == "cpu"
    assert report["sequence_counts"]["train"] > 0
    assert report["sequence_counts"]["validation"] > 0
    assert set(report["validation"]) >= {"rmse", "mae", "mape", "r2"}
    assert len(history) <= 3
    assert len(epoch_lines) == len(history)
    assert all(
        "train loss:" in line
        and "validation loss:" in line
        and "elapsed:" in line
        for line in epoch_lines
    )

    checkpoint = torch.load(artifacts["model"], map_location="cpu", weights_only=False)
    assert checkpoint["model_state_dict"]
    assert artifacts["model"].name == f"{model_name}_volume_model.pt"
    assert artifacts["report"].parent != artifact_dir


def test_training_is_reproducible_for_fixed_seed(tmp_path):
    source = tmp_path / "synthetic.csv"
    _frame().to_csv(source, index=False)
    config = TrainingConfig(max_epochs=2, early_stopping_patience=1, seed=19)

    first = train_lstm(
        input_path=source,
        models_dir=tmp_path / "first_models",
        report_path=tmp_path / "first_report.json",
        curve_path=tmp_path / "first_curve.png",
        config=config,
    )
    second = train_lstm(
        input_path=source,
        models_dir=tmp_path / "second_models",
        report_path=tmp_path / "second_report.json",
        curve_path=tmp_path / "second_curve.png",
        config=config,
    )
    first_checkpoint = torch.load(first["model"], map_location="cpu", weights_only=False)
    second_checkpoint = torch.load(second["model"], map_location="cpu", weights_only=False)

    for name, value in first_checkpoint["model_state_dict"].items():
        torch.testing.assert_close(value, second_checkpoint["model_state_dict"][name])


def test_training_rejects_empty_training_sequences(tmp_path):
    source = tmp_path / "too_small.csv"
    _frame(periods=5, roads=("A",)).to_csv(source, index=False)

    with pytest.raises(ValueError, match="no training sequences"):
        train_gru(
            input_path=source,
            models_dir=tmp_path / "models",
            report_path=tmp_path / "report.json",
            curve_path=tmp_path / "curve.png",
            config=TrainingConfig(max_epochs=1),
        )


def test_lstm_smoke_test_uses_one_epoch_and_batch_size_64(tmp_path, capsys):
    source = tmp_path / "synthetic.csv"
    _frame().to_csv(source, index=False)

    artifacts = train_lstm(
        input_path=source,
        models_dir=tmp_path / "models",
        report_path=tmp_path / "report.json",
        curve_path=tmp_path / "curve.png",
        smoke_test=True,
    )

    history = json.loads(artifacts["history"].read_text(encoding="utf-8"))
    checkpoint = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=False)
    epoch_lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("Training: epoch ")
    ]
    assert len(history) == 1
    assert checkpoint["training_config"]["max_epochs"] == 1
    assert checkpoint["training_config"]["batch_size"] == 64
    assert len(epoch_lines) == 1


def test_gru_smoke_test_uses_one_epoch_and_isolated_artifacts(tmp_path, capsys):
    source = tmp_path / "synthetic.csv"
    _frame().to_csv(source, index=False)
    models_dir = tmp_path / "models"
    report_path = tmp_path / "reports" / "gru_smoke_test_validation.json"
    curve_path = tmp_path / "reports" / "figures" / "gru_smoke_test.png"

    artifacts = train_gru(
        input_path=source,
        models_dir=models_dir,
        report_path=report_path,
        curve_path=curve_path,
        config=TrainingConfig(max_epochs=50, batch_size=8),
        smoke_test=True,
    )

    history = json.loads(artifacts["history"].read_text(encoding="utf-8"))
    checkpoint = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=False)
    epoch_lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("Training: epoch ")
    ]
    assert len(history) == 1
    assert checkpoint["training_config"]["max_epochs"] == 1
    assert checkpoint["training_config"]["batch_size"] == 64
    assert len(epoch_lines) == 1
    assert artifacts["model"].parent == models_dir / "gru_smoke_test"
    assert artifacts["report"] == report_path
    assert artifacts["curve"] == curve_path
    assert artifacts["model"].name == "gru_volume_model.pt"
    assert not (models_dir / "gru_volume_model.pt").exists()
