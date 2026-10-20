"""Production inference for the saved FlowCast GRU volume model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.dl_models import GRURegressor
from src.features import add_time_features
from src.sequence_data import (
    DL_FEATURE_COLUMNS,
    SEQUENCE_FREQUENCY,
    SEQUENCE_LENGTH,
    SequencePreprocessor,
    build_inference_sequence,
)
from src.utils import configured_path, load_config


@dataclass(frozen=True)
class GRUPrediction:
    """One next-interval GRU traffic-volume forecast."""

    road_id: str
    forecast_timestamp: pd.Timestamp
    predicted_traffic_volume: float


class GRUInference:
    """Load and run the trained GRU without fitting or changing artifacts."""

    def __init__(
        self,
        model: nn.Module,
        preprocessor: SequencePreprocessor,
        *,
        sequence_length: int,
    ) -> None:
        self.model = model.eval()
        self.preprocessor = preprocessor
        self.sequence_length = sequence_length

    @classmethod
    def from_artifacts(
        cls,
        *,
        model_path: str | Path | None = None,
        preprocessor_path: str | Path | None = None,
    ) -> "GRUInference":
        settings = load_config()
        models_dir = configured_path(settings["paths"]["models_dir"])
        model_file = Path(model_path) if model_path is not None else models_dir / "gru_volume_model.pt"
        preprocessor_file = (
            Path(preprocessor_path)
            if preprocessor_path is not None
            else models_dir / "gru_volume_preprocessor.joblib"
        )
        if not model_file.is_file():
            raise FileNotFoundError(f"Missing GRU model artifact: {model_file}")
        if not preprocessor_file.is_file():
            raise FileNotFoundError(
                f"Missing GRU preprocessor artifact: {preprocessor_file}"
            )

        import joblib

        artifact = torch.load(model_file, map_location="cpu", weights_only=False)
        preprocessor = joblib.load(preprocessor_file)
        if not isinstance(preprocessor, SequencePreprocessor):
            raise TypeError("GRU preprocessor artifact has an unexpected type")
        config = artifact["training_config"]
        sequence_length = int(artifact.get("sequence_length", SEQUENCE_LENGTH))
        if sequence_length != SEQUENCE_LENGTH:
            raise ValueError(
                f"Unsupported GRU sequence length: {sequence_length}; expected {SEQUENCE_LENGTH}"
            )
        if tuple(preprocessor.feature_columns) != DL_FEATURE_COLUMNS:
            raise ValueError("GRU preprocessor feature columns do not match training")
        if int(artifact["input_size"]) != len(preprocessor.feature_names_out_):
            raise ValueError("GRU input size does not match the saved preprocessor")
        model = GRURegressor(
            input_size=int(artifact["input_size"]),
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
        )
        model.load_state_dict(artifact["model_state_dict"])
        return cls(model, preprocessor, sequence_length=sequence_length)

    def predict_next_30_minutes(self, history: pd.DataFrame) -> GRUPrediction:
        """Predict the interval immediately after the latest observed row."""
        if not isinstance(history, pd.DataFrame):
            raise TypeError("history must be a pandas DataFrame")
        if "traffic_volume_target_30m" in history.columns:
            raise ValueError(
                "Prediction history must not contain traffic_volume_target_30m"
            )
        required = set(DL_FEATURE_COLUMNS).difference(
            {"hour_of_day", "day_of_week", "is_weekend", "month", "is_rush_hour"}
        ) | {"timestamp"}
        missing = sorted(required.difference(history.columns))
        if missing:
            raise ValueError(f"Required prediction columns are missing: {', '.join(missing)}")
        if history["road_id"].isna().any():
            raise ValueError("Prediction history contains missing road identifiers")
        road_ids = history["road_id"].astype(str)
        unknown = sorted(set(road_ids).difference(self.preprocessor.road_categories_))
        if unknown:
            raise ValueError(f"Unsupported road identifier(s): {', '.join(unknown)}")

        prepared = add_time_features(history)
        sequence = build_inference_sequence(
            prepared,
            sequence_length=self.sequence_length,
            feature_columns=DL_FEATURE_COLUMNS,
        )
        transformed = self.preprocessor.transform_sequences(sequence)
        features = torch.from_numpy(transformed.features).float()
        with torch.no_grad():
            prediction = float(self.model(features).item())
        endpoint = pd.Timestamp(sequence.metadata.loc[0, "endpoint_timestamp"])
        return GRUPrediction(
            road_id=str(sequence.metadata.loc[0, "road_id"]),
            forecast_timestamp=endpoint + SEQUENCE_FREQUENCY,
            predicted_traffic_volume=prediction,
        )
