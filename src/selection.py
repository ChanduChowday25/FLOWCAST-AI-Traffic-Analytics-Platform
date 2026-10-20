"""Leakage-safe feature selection for FlowCast model preparation."""

from __future__ import annotations

import pandas as pd

from src.targets import TRAFFIC_VOLUME_TARGET_30M


MODEL_FEATURE_COLUMNS = (
    "road_id",
    "road_capacity",
    "traffic_volume_lag_1",
    "traffic_volume_lag_2",
    "traffic_volume_rolling_mean_3",
    "traffic_volume_rolling_mean_6",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "is_rush_hour",
)


def select_model_features(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
    target_column: str = TRAFFIC_VOLUME_TARGET_30M,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the explicit model feature matrix and target series.

    This function only selects columns. It does not impute missing values,
    alter dtypes, split rows, or train a model. The returned objects retain
    the input row order and index.
    """
    if target_column not in frame.columns:
        raise ValueError(f"Required target column is missing: {target_column}")
    if target_column in feature_columns:
        raise ValueError(f"Target column cannot be a model feature: {target_column}")

    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Required model features are missing: {', '.join(missing)}")

    features = frame.loc[:, list(feature_columns)].copy()
    target = frame.loc[:, target_column].copy()
    return features, target
