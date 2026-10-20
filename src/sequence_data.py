"""Leakage-safe sequence data preparation for FlowCast deep models."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.splitting import split_chronologically
from src.targets import TRAFFIC_VOLUME_TARGET_30M


DL_NUMERIC_FEATURE_COLUMNS = (
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "weather_temperature",
    "weather_rainfall",
    "weather_visibility",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "is_rush_hour",
)
DL_FEATURE_COLUMNS = ("road_id", *DL_NUMERIC_FEATURE_COLUMNS)
SEQUENCE_LENGTH = 12
SEQUENCE_FREQUENCY = pd.Timedelta(minutes=30)
ProgressReporter = Callable[[str], None]


def _report_progress(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class RawSequencePartition:
    """Compact untransformed sequence windows and their audit metadata."""

    sequences: np.ndarray
    road_sequences: np.ndarray
    road_categories: tuple[str, ...]
    targets: np.ndarray
    metadata: pd.DataFrame
    exclusions: dict[str, int]


@dataclass(frozen=True)
class SequencePartition:
    """Model-ready sequences and traceable endpoint metadata."""

    features: np.ndarray
    targets: np.ndarray
    metadata: pd.DataFrame
    exclusions: dict[str, int]


@dataclass(frozen=True)
class SequenceDataset:
    """Chronologically split, preprocessed sequence data."""

    train: SequencePartition
    validation: SequencePartition
    test: SequencePartition
    preprocessor: "SequencePreprocessor"


def _validate_configuration(
    sequence_length: int,
    feature_columns: tuple[str, ...],
) -> None:
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise ValueError("sequence_length must be a positive integer")
    if sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    if not feature_columns:
        raise ValueError("At least one sequence feature is required")
    if TRAFFIC_VOLUME_TARGET_30M in feature_columns:
        raise ValueError("The sequence target cannot be an input feature")
    if "road_id" not in feature_columns:
        raise ValueError("Sequence features must include road_id")


def _validate_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    road_column: str,
    timestamp_column: str,
    target_column: str,
) -> pd.DataFrame:
    required = set(feature_columns) | {road_column, timestamp_column, target_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Required sequence columns are missing: {', '.join(missing)}")
    result = frame.copy()
    result[timestamp_column] = pd.to_datetime(result[timestamp_column], errors="coerce")
    if result[timestamp_column].isna().any():
        raise ValueError("Invalid timestamps found in sequence data")
    if result[road_column].isna().any():
        raise ValueError("Sequence data contains missing road identifiers")
    if result.duplicated([road_column, timestamp_column]).any():
        raise ValueError("Duplicate road/timestamp keys found in sequence data")
    return result


def build_sequences(
    frame: pd.DataFrame,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    feature_columns: tuple[str, ...] = DL_FEATURE_COLUMNS,
    road_column: str = "road_id",
    timestamp_column: str = "timestamp",
    target_column: str = TRAFFIC_VOLUME_TARGET_30M,
) -> RawSequencePartition:
    """Build independent, contiguous windows ending at each prediction time.

    The target on the endpoint row is the observed volume at endpoint + 30
    minutes, as constructed by :func:`add_traffic_volume_target_30m`.
    """
    _validate_configuration(sequence_length, feature_columns)
    result = _validate_frame(
        frame,
        feature_columns=feature_columns,
        road_column=road_column,
        timestamp_column=timestamp_column,
        target_column=target_column,
    )
    numeric_values = result.loc[:, list(feature_columns)].drop(columns=[road_column]).apply(
        lambda column: pd.to_numeric(column, errors="coerce")
    ).to_numpy(dtype="float32")
    numeric_values[~np.isfinite(numeric_values)] = np.nan
    road_categories = tuple(sorted(result[road_column].astype(str).unique().tolist()))
    road_to_code = {road: code for code, road in enumerate(road_categories)}
    road_values = result[road_column].astype(str).map(road_to_code).to_numpy(dtype="int32")
    timestamps_ns = result[timestamp_column].astype("int64").to_numpy()
    source_indices = result.index.to_numpy()
    target_values = pd.to_numeric(result[target_column], errors="coerce").to_numpy(dtype="float64")
    sequence_values: list[np.ndarray] = []
    sequence_roads: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, Any]] = []
    exclusions = {
        "insufficient_history": 0,
        "non_contiguous_timestamp": 0,
        "missing_target": 0,
        "non_finite_target": 0,
        "negative_target": 0,
    }

    road_positions = result.groupby(road_column, sort=True, observed=True).indices
    for road_id in sorted(road_positions):
        positions = np.asarray(road_positions[road_id], dtype=np.int64)
        positions = positions[np.argsort(timestamps_ns[positions], kind="stable")]
        for end_position in range(len(positions)):
            start_position = end_position - sequence_length + 1
            if start_position < 0:
                exclusions["insufficient_history"] += 1
                continue
            window_positions = positions[start_position : end_position + 1]
            if not np.all(np.diff(timestamps_ns[window_positions]) == SEQUENCE_FREQUENCY.value):
                exclusions["non_contiguous_timestamp"] += 1
                continue

            numeric_target = target_values[window_positions[-1]]
            if pd.isna(numeric_target):
                exclusions["missing_target"] += 1
                continue
            if not np.isfinite(numeric_target):
                exclusions["non_finite_target"] += 1
                continue
            if numeric_target < 0:
                exclusions["negative_target"] += 1
                continue

            sequence_values.append(numeric_values[window_positions].copy())
            sequence_roads.append(road_values[window_positions].copy())
            targets.append(float(numeric_target))
            metadata.append(
                {
                    "road_id": road_id,
                    "endpoint_timestamp": pd.Timestamp(timestamps_ns[window_positions[-1]]),
                    "endpoint_source_index": source_indices[window_positions[-1]],
                    "source_row_indices": tuple(source_indices[window_positions].tolist()),
                }
            )

    metadata_frame = pd.DataFrame(
        metadata,
        columns=[
            "road_id",
            "endpoint_timestamp",
            "endpoint_source_index",
            "source_row_indices",
        ],
    )
    return RawSequencePartition(
        sequences=(
            np.stack(sequence_values).astype("float32", copy=False)
            if sequence_values
            else np.empty((0, sequence_length, len(feature_columns) - 1), dtype="float32")
        ),
        road_sequences=(
            np.stack(sequence_roads).astype("int32", copy=False)
            if sequence_roads
            else np.empty((0, sequence_length), dtype="int32")
        ),
        road_categories=road_categories,
        targets=np.asarray(targets, dtype="float64"),
        metadata=metadata_frame,
        exclusions=exclusions,
    )


def build_inference_sequence(
    frame: pd.DataFrame,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    feature_columns: tuple[str, ...] = DL_FEATURE_COLUMNS,
    road_column: str = "road_id",
    timestamp_column: str = "timestamp",
) -> RawSequencePartition:
    """Build one target-free, contiguous sequence for production inference."""
    _validate_configuration(sequence_length, feature_columns)
    if TRAFFIC_VOLUME_TARGET_30M in frame.columns:
        raise ValueError(
            f"Prediction input must not contain target column: {TRAFFIC_VOLUME_TARGET_30M}"
        )
    required = set(feature_columns) | {timestamp_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Required inference columns are missing: {', '.join(missing)}")
    result = frame.copy()
    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column], errors="coerce"
    )
    if result[timestamp_column].isna().any():
        raise ValueError("Inference input contains invalid timestamps")
    if result[road_column].isna().any():
        raise ValueError("Inference input contains missing road identifiers")
    if result.duplicated([road_column, timestamp_column]).any():
        raise ValueError("Inference input contains duplicate road/timestamp keys")

    result[road_column] = result[road_column].astype(str)
    if result[road_column].nunique() != 1:
        raise ValueError("Inference input must contain exactly one road identifier")
    result = result.sort_values(timestamp_column).reset_index(drop=True)
    if len(result) < sequence_length:
        raise ValueError(
            f"Inference requires at least {sequence_length} historical rows"
        )
    timestamps_ns = result[timestamp_column].astype("int64").to_numpy()
    window = np.arange(len(result) - sequence_length, len(result), dtype=np.int64)
    if not np.all(np.diff(timestamps_ns[window]) == SEQUENCE_FREQUENCY.value):
        raise ValueError("Inference history must have exact 30-minute continuity")

    numeric_values = (
        result.loc[:, list(feature_columns)]
        .drop(columns=[road_column])
        .apply(lambda column: pd.to_numeric(column, errors="coerce"))
        .to_numpy(dtype="float32")
    )
    numeric_values[~np.isfinite(numeric_values)] = np.nan
    road_id = result.loc[window[0], road_column]
    road_categories = (road_id,)
    metadata = pd.DataFrame(
        [
            {
                "road_id": road_id,
                "endpoint_timestamp": pd.Timestamp(
                    timestamps_ns[window[-1]]
                ),
                "endpoint_source_index": result.index[window[-1]],
                "source_row_indices": tuple(window.tolist()),
            }
        ],
        columns=[
            "road_id",
            "endpoint_timestamp",
            "endpoint_source_index",
            "source_row_indices",
        ],
    )
    return RawSequencePartition(
        sequences=numeric_values[window].reshape(
            1, sequence_length, len(feature_columns) - 1
        ),
        road_sequences=np.zeros((1, sequence_length), dtype="int32"),
        road_categories=road_categories,
        targets=np.empty(0, dtype="float64"),
        metadata=metadata,
        exclusions={},
    )


class SequencePreprocessor:
    """Fit numeric transformations and road categories on training rows only."""

    def __init__(
        self,
        *,
        feature_columns: tuple[str, ...] = DL_FEATURE_COLUMNS,
        numeric_columns: tuple[str, ...] = DL_NUMERIC_FEATURE_COLUMNS,
        road_column: str = "road_id",
    ) -> None:
        _validate_configuration(1, feature_columns)
        if tuple(column for column in feature_columns if column != road_column) != numeric_columns:
            raise ValueError("numeric_columns must match feature_columns excluding road_id")
        self.feature_columns = tuple(feature_columns)
        self.numeric_columns = tuple(numeric_columns)
        self.road_column = road_column

    def fit(self, frame: pd.DataFrame) -> "SequencePreprocessor":
        """Fit imputation, scaling, and one-hot mapping using this frame only."""
        missing = sorted(set(self.feature_columns).difference(frame.columns))
        if missing:
            raise ValueError(f"Required sequence columns are missing: {', '.join(missing)}")
        if frame.empty:
            raise ValueError("Cannot fit sequence preprocessing on an empty frame")
        numeric = self._numeric_frame(frame)
        self.numeric_imputer_ = SimpleImputer(strategy="median")
        imputed = self.numeric_imputer_.fit_transform(numeric.to_numpy(dtype="float64"))
        self.numeric_scaler_ = StandardScaler()
        self.numeric_scaler_.fit(imputed)
        categories = sorted(frame[self.road_column].dropna().astype(str).unique().tolist())
        if not categories:
            raise ValueError("Cannot fit sequence preprocessing without road identifiers")
        self.road_categories_ = tuple(categories)
        self.road_encoder_ = OneHotEncoder(
            categories=[np.asarray(categories, dtype=object)],
            handle_unknown="ignore",
            sparse_output=False,
        )
        self.road_encoder_.fit(frame[[self.road_column]].astype(str))
        self.feature_names_out_ = tuple(self.numeric_columns) + tuple(
            f"road_id_{category}" for category in categories
        )
        return self

    def _numeric_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame.loc[:, list(self.numeric_columns)].copy()
        for column in self.numeric_columns:
            values = pd.to_numeric(numeric[column], errors="coerce")
            numeric[column] = values.where(np.isfinite(values), np.nan)
        return numeric

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform rows with the already-fitted training transformations."""
        if not hasattr(self, "numeric_imputer_"):
            raise ValueError("SequencePreprocessor must be fitted before transform")
        missing = sorted(set(self.feature_columns).difference(frame.columns))
        if missing:
            raise ValueError(f"Required sequence columns are missing: {', '.join(missing)}")
        numeric_frame = self._numeric_frame(frame)
        numeric = self.numeric_scaler_.transform(
            self.numeric_imputer_.transform(numeric_frame.to_numpy(dtype="float64"))
        )
        roads = self.road_encoder_.transform(frame[[self.road_column]].astype(str))
        return np.concatenate([numeric, roads], axis=1).astype("float32")

    def transform_sequences(self, partition: RawSequencePartition) -> SequencePartition:
        """Transform every sequence without refitting any preprocessing step."""
        if len(partition.sequences):
            shape = partition.sequences.shape
            numeric = self.numeric_scaler_.transform(
                self.numeric_imputer_.transform(
                    partition.sequences.reshape(-1, shape[-1])
                )
            )
            category_to_position = {
                category: position
                for position, category in enumerate(self.road_categories_)
            }
            roads = np.zeros(
                (shape[0] * shape[1], len(self.road_categories_)),
                dtype="float32",
            )
            partition_categories = np.asarray(partition.road_categories, dtype=object)
            for position, road_code in enumerate(partition.road_sequences.reshape(-1)):
                if road_code >= len(partition_categories):
                    continue
                category_position = category_to_position.get(str(partition_categories[road_code]))
                if category_position is not None:
                    roads[position, category_position] = 1.0
            features = np.concatenate(
                [numeric.astype("float32"), roads], axis=1
            ).reshape(shape[0], shape[1], -1)
        else:
            features = np.empty(
                (0, partition.sequences.shape[1], len(self.feature_names_out_)),
                dtype="float32",
            )
        return SequencePartition(
            features=features,
            targets=partition.targets.copy(),
            metadata=partition.metadata.copy(),
            exclusions=dict(partition.exclusions),
        )


def prepare_sequence_dataset(
    frame: pd.DataFrame,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    feature_columns: tuple[str, ...] = DL_FEATURE_COLUMNS,
    progress: ProgressReporter = _report_progress,
) -> SequenceDataset:
    """Split chronologically, fit on train rows, and transform all partitions."""
    _validate_configuration(sequence_length, feature_columns)
    validated = _validate_frame(
        frame,
        feature_columns=feature_columns,
        road_column="road_id",
        timestamp_column="timestamp",
        target_column=TRAFFIC_VOLUME_TARGET_30M,
    )
    progress("Sequence data: creating chronological train/validation/test splits")
    train, validation, test = split_chronologically(validated)
    progress("Sequence data: chronological splits ready")
    preprocessor = SequencePreprocessor(feature_columns=feature_columns)
    progress("Sequence data: fitting training-only preprocessor")
    preprocessor.fit(train)
    progress("Sequence data: training-only preprocessor ready")
    transformed: list[SequencePartition] = []
    for name, partition in zip(
        ("train", "validation", "test"),
        (train, validation, test),
    ):
        progress(f"Sequence data: creating {name} sequences")
        raw_partition = build_sequences(
            partition,
            sequence_length=sequence_length,
            feature_columns=feature_columns,
        )
        progress(f"Sequence data: transforming {name} sequences")
        transformed.append(preprocessor.transform_sequences(raw_partition))
        del raw_partition
        progress(f"Sequence data: {name} sequences ready")
    return SequenceDataset(
        train=transformed[0],
        validation=transformed[1],
        test=transformed[2],
        preprocessor=preprocessor,
    )
