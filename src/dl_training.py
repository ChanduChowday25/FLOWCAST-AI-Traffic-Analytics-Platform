"""Shared CPU training and artifact utilities for FlowCast recurrent models."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.sequence_data import DL_FEATURE_COLUMNS, SequenceDataset
from src.targets import TRAFFIC_VOLUME_TARGET_30M
from src.train_linear_regression import _metrics
from src.utils import configured_path, load_config


DEFAULT_HIDDEN_SIZE = 32
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.2
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 50
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EARLY_STOPPING_PATIENCE = 7
DEFAULT_LR_PATIENCE = 3
DEFAULT_LR_FACTOR = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible CPU training configuration."""

    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_layers: int = DEFAULT_NUM_LAYERS
    dropout: float = DEFAULT_DROPOUT
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    lr_patience: int = DEFAULT_LR_PATIENCE
    lr_factor: float = DEFAULT_LR_FACTOR
    seed: int = 42


def set_reproducible_seed(seed: int) -> None:
    """Set practical CPU seeds without enabling unsupported GPU behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _loader(partition: Any, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    features = torch.from_numpy(partition.features).float()
    targets = torch.from_numpy(partition.targets).float()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _progress(message: str) -> None:
    print(message, flush=True)


def _evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for features, targets in loader:
            predictions = model(features)
            losses.append(float(criterion(predictions, targets).item()))
    return float(np.mean(losses)) if losses else math.nan


def _predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for features, expected in loader:
            predictions.append(model(features).cpu().numpy())
            targets.append(expected.cpu().numpy())
    if not predictions:
        return np.empty(0, dtype="float64"), np.empty(0, dtype="float64")
    return np.concatenate(targets), np.concatenate(predictions)


def _plot_history(history: list[dict[str, float]], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [entry["epoch"] for entry in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [entry["train_loss"] for entry in history], label="train")
    plt.plot(epochs, [entry["validation_loss"] for entry in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def _partition_boundaries(partition: Any) -> dict[str, str | None]:
    if partition.metadata.empty:
        return {"endpoint_timestamp_start": None, "endpoint_timestamp_end": None}
    timestamps = pd.to_datetime(partition.metadata["endpoint_timestamp"])
    return {
        "endpoint_timestamp_start": str(timestamps.min()),
        "endpoint_timestamp_end": str(timestamps.max()),
    }


def train_recurrent_model(
    *,
    model_factory: Callable[..., nn.Module],
    model_name: str,
    dataset: SequenceDataset,
    artifact_dir: str | Path,
    report_path: str | Path,
    curve_path: str | Path,
    config: TrainingConfig | None = None,
) -> dict[str, Path]:
    """Train one recurrent model using train/validation partitions only."""
    settings = config or TrainingConfig()
    if len(dataset.train.targets) == 0:
        raise ValueError("Cannot train recurrent model: no training sequences")
    if len(dataset.validation.targets) == 0:
        raise ValueError("Cannot train recurrent model: no validation sequences")

    set_reproducible_seed(settings.seed)
    artifact_directory = Path(artifact_dir)
    report_file = Path(report_path)
    curve_file = Path(curve_path)
    model_path = artifact_directory / f"{model_name}_volume_model.pt"
    preprocessor_path = artifact_directory / f"{model_name}_volume_preprocessor.joblib"
    history_path = artifact_directory / f"{model_name}_volume_history.json"
    checkpoint_path = artifact_directory / f"{model_name}_volume_best_checkpoint.pt"

    input_size = int(dataset.train.features.shape[-1])
    model = model_factory(
        input_size=input_size,
        hidden_size=settings.hidden_size,
        num_layers=settings.num_layers,
        dropout=settings.dropout,
    ).to("cpu")
    _progress("Training: model created")
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=settings.lr_factor,
        patience=settings.lr_patience,
    )
    train_loader = _loader(
        dataset.train, settings.batch_size, shuffle=True, seed=settings.seed
    )
    validation_loader = _loader(
        dataset.validation, settings.batch_size, shuffle=False, seed=settings.seed
    )
    _progress("Training: DataLoaders created")

    artifact_directory.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, settings.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        validation_loss = _evaluate_loss(model, validation_loader, criterion)
        scheduler.step(validation_loss)
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)
        _progress(
            f"Training: epoch {epoch}/{settings.max_epochs} - "
            f"train loss: {epoch_record['train_loss']:.6f}, "
            f"validation loss: {validation_loss:.6f}, "
            f"elapsed: {time.perf_counter() - epoch_started:.1f}s"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "input_size": input_size,
                    "training_config": asdict(settings),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings.early_stopping_patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    torch.save(
        {
            **checkpoint,
            "model_state_dict": model.state_dict(),
        },
        model_path,
    )
    joblib.dump(dataset.preprocessor, preprocessor_path)
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    _plot_history(history, curve_file, f"FlowCast {model_name.upper()} training")

    actual, predicted = _predictions(model, validation_loader)
    validation_metrics = _metrics(
        pd.Series(actual, dtype="float64"),
        predicted,
    )
    report = {
        "model": model_name,
        "target_column": "traffic_volume_target_30m",
        "feature_columns": list(dataset.preprocessor.feature_columns),
        "transformed_feature_names": list(dataset.preprocessor.feature_names_out_),
        "sequence_length": int(dataset.train.features.shape[1]),
        "device": "cpu",
        "configuration": asdict(settings),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "sequence_counts": {
            "train": len(dataset.train.targets),
            "validation": len(dataset.validation.targets),
        },
        "split_boundaries": {
            "train": _partition_boundaries(dataset.train),
            "validation": _partition_boundaries(dataset.validation),
        },
        "validation": {
            "row_count": len(actual),
            "valid_prediction_count": len(actual),
            "zero_target_count": int(np.count_nonzero(actual == 0)),
            "mape_eligible_count": int(np.count_nonzero(actual != 0)),
            **validation_metrics,
        },
        "mape_policy": "Zero-valued targets are excluded because percentage error is undefined.",
        "model_artifact": str(model_path),
        "checkpoint_artifact": str(checkpoint_path),
        "preprocessor_artifact": str(preprocessor_path),
        "history_artifact": str(history_path),
        "curve_artifact": str(curve_file),
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return {
        "model": model_path,
        "checkpoint": checkpoint_path,
        "preprocessor": preprocessor_path,
        "history": history_path,
        "curve": curve_file,
        "report": report_file,
    }


def load_sequence_dataset(input_path: str | Path | None = None) -> SequenceDataset:
    """Load and prepare the configured processed dataset."""
    settings = load_config()
    input_file = (
        Path(input_path)
        if input_path is not None
        else configured_path(settings["data"]["processed_dir"]) / "flowcast_target_30m.csv"
    )
    if not input_file.is_file():
        raise FileNotFoundError(f"Processed target dataset is missing: {input_file}")
    from src.sequence_data import prepare_sequence_dataset

    required_columns = tuple(
        dict.fromkeys(
            (*DL_FEATURE_COLUMNS, "timestamp", TRAFFIC_VOLUME_TARGET_30M)
        )
    )
    _progress(f"Sequence data: loading required columns from {input_file}")
    frame = pd.read_csv(
        input_file,
        parse_dates=["timestamp"],
        usecols=list(required_columns),
    )
    _progress(f"Sequence data: loaded {len(frame)} rows")
    return prepare_sequence_dataset(frame)
