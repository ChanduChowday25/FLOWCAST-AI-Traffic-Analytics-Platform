"""Chronological dataset splitting for FlowCast."""

from __future__ import annotations

import math

import pandas as pd


def split_chronologically(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    train_proportion: float = 0.70,
    validation_proportion: float = 0.15,
    test_proportion: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into chronological train, validation, and test frames.

    Split boundaries are selected from unique timestamps, so all rows sharing
    a timestamp remain in the same split. Rows retain their original order and
    index within each returned frame. Proportions are applied to the number of
    unique timestamps using floor boundaries.
    """
    if timestamp_column not in frame.columns:
        raise ValueError(f"Required timestamp column is missing: {timestamp_column}")

    proportions = (
        ("train", train_proportion),
        ("validation", validation_proportion),
        ("test", test_proportion),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for _, value in proportions
    ):
        raise ValueError("Split proportions must be finite positive numbers")
    if not math.isclose(
        train_proportion + validation_proportion + test_proportion,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Split proportions must sum to 1.0")

    timestamp = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if timestamp.isna().any():
        invalid_rows = frame.index[timestamp.isna()].tolist()[:5]
        raise ValueError(f"Invalid timestamps found in rows: {invalid_rows}")

    unique_timestamps = pd.Index(timestamp.unique()).sort_values()
    timestamp_count = len(unique_timestamps)
    train_end = int(timestamp_count * train_proportion)
    validation_end = int(timestamp_count * (train_proportion + validation_proportion))
    if train_end < 1 or validation_end <= train_end or validation_end >= timestamp_count:
        raise ValueError(
            "At least three unique timestamps are required for non-empty chronological splits"
        )

    train_timestamps = unique_timestamps[:train_end]
    validation_timestamps = unique_timestamps[train_end:validation_end]
    test_timestamps = unique_timestamps[validation_end:]
    if (
        train_timestamps.intersection(validation_timestamps).size
        or train_timestamps.intersection(test_timestamps).size
        or validation_timestamps.intersection(test_timestamps).size
        or train_timestamps.max() >= validation_timestamps.min()
        or validation_timestamps.max() >= test_timestamps.min()
    ):
        raise ValueError("Chronological split boundaries overlap or are out of order")

    result = frame.copy()
    train_mask = timestamp.isin(train_timestamps)
    validation_mask = timestamp.isin(validation_timestamps)
    test_mask = timestamp.isin(test_timestamps)
    return (
        result.loc[train_mask].copy(),
        result.loc[validation_mask].copy(),
        result.loc[test_mask].copy(),
    )
