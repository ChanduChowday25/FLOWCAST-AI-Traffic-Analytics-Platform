"""A leakage-safe persistence baseline for 30-minute traffic-volume forecasts."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.targets import TRAFFIC_VOLUME_TARGET_30M


PERSISTENCE_INPUT_COLUMN = "traffic_volume_lag_1"
PERSISTENCE_PREDICTION_COLUMN = "traffic_volume_persistence_prediction"


@dataclass(frozen=True)
class BaselineEvaluation:
    """Persistence predictions and auditable evaluation counts."""

    predictions: pd.DataFrame
    metrics: dict[str, float]
    input_row_count: int
    valid_prediction_count: int
    missing_baseline_count: int
    invalid_target_count: int
    zero_target_count: int
    mape_eligible_count: int


def _require_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")


def _validate_columns(frame: pd.DataFrame, *, road_column: str, timestamp_column: str, baseline_column: str) -> None:
    required = {road_column, timestamp_column, baseline_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Required persistence columns are missing: {', '.join(missing)}")
    if TRAFFIC_VOLUME_TARGET_30M in frame.columns and TRAFFIC_VOLUME_TARGET_30M == baseline_column:
        raise ValueError("Persistence input cannot be the target column")


def _valid_nonnegative_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.notna() & np.isfinite(numeric) & numeric.ge(0)


def predict_persistence(
    frame: pd.DataFrame,
    *,
    road_column: str = "road_id",
    timestamp_column: str = "timestamp",
    baseline_column: str = PERSISTENCE_INPUT_COLUMN,
) -> pd.DataFrame:
    """Return one aligned persistence prediction for every input row.

    ``traffic_volume_lag_1`` is the observed traffic volume exactly 30 minutes
    before the prediction timestamp, as defined by ``add_traffic_volume_lags``.
    It is therefore the latest historical measurement represented in the
    existing feature set. Missing, nonnumeric, negative, or infinite values
    produce an unavailable prediction (``NaN``); no target or future column is
    consulted, and no older lag is substituted.
    """
    _require_frame(frame)
    _validate_columns(
        frame,
        road_column=road_column,
        timestamp_column=timestamp_column,
        baseline_column=baseline_column,
    )
    timestamp = pd.to_datetime(frame[timestamp_column], errors="coerce", format="mixed")
    if timestamp.isna().any():
        invalid_rows = frame.index[timestamp.isna()].tolist()[:5]
        raise ValueError(f"Invalid timestamps found in rows: {invalid_rows}")

    baseline = pd.to_numeric(frame[baseline_column], errors="coerce")
    baseline = baseline.where(_valid_nonnegative_numeric(frame[baseline_column]))
    return pd.DataFrame(
        {
            road_column: frame[road_column].copy(),
            timestamp_column: timestamp,
            PERSISTENCE_PREDICTION_COLUMN: baseline,
        },
        index=frame.index,
    )


def evaluate_persistence(
    frame: pd.DataFrame,
    *,
    road_column: str = "road_id",
    timestamp_column: str = "timestamp",
    baseline_column: str = PERSISTENCE_INPUT_COLUMN,
    target_column: str = TRAFFIC_VOLUME_TARGET_30M,
) -> BaselineEvaluation:
    """Evaluate persistence on rows with valid baseline and target values.

    MAPE excludes rows whose true target is zero because percentage error is
    undefined there. The count of excluded zero targets is returned explicitly.
    RMSE and MAE require at least one eligible row; R² additionally requires
    at least two eligible rows with non-constant targets.
    """
    _require_frame(frame)
    if target_column not in frame.columns:
        raise ValueError(f"Required target column is missing: {target_column}")
    if target_column == baseline_column:
        raise ValueError("Target column cannot be the persistence input")

    prediction_frame = predict_persistence(
        frame,
        road_column=road_column,
        timestamp_column=timestamp_column,
        baseline_column=baseline_column,
    )
    baseline_valid = prediction_frame[PERSISTENCE_PREDICTION_COLUMN].notna()
    target_numeric = pd.to_numeric(frame[target_column], errors="coerce")
    target_valid = target_numeric.notna() & np.isfinite(target_numeric) & target_numeric.ge(0)
    eligible = baseline_valid & target_valid
    y_true = target_numeric.loc[eligible].to_numpy(dtype=float)
    y_pred = prediction_frame.loc[eligible, PERSISTENCE_PREDICTION_COLUMN].to_numpy(dtype=float)

    metrics = {"rmse": math.nan, "mae": math.nan, "mape": math.nan, "r2": math.nan}
    if len(y_true):
        errors = y_pred - y_true
        metrics["rmse"] = float(np.sqrt(np.mean(errors**2)))
        metrics["mae"] = float(np.mean(np.abs(errors)))
        nonzero = y_true != 0
        if nonzero.any():
            metrics["mape"] = float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100)
        if len(y_true) >= 2 and not np.all(y_true == y_true[0]):
            metrics["r2"] = float(1 - np.sum(errors**2) / np.sum((y_true - np.mean(y_true)) ** 2))

    return BaselineEvaluation(
        predictions=prediction_frame,
        metrics=metrics,
        input_row_count=len(frame),
        valid_prediction_count=int(eligible.sum()),
        missing_baseline_count=int((~baseline_valid).sum()),
        invalid_target_count=int((~target_valid).sum()),
        zero_target_count=int((eligible & target_numeric.eq(0)).sum()),
        mape_eligible_count=int((eligible & target_numeric.ne(0)).sum()),
    )
