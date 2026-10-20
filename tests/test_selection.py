import pandas as pd
import pytest

from src.selection import MODEL_FEATURE_COLUMNS, select_model_features
from src.targets import TRAFFIC_VOLUME_TARGET_30M


def _frame() -> pd.DataFrame:
    columns = [*MODEL_FEATURE_COLUMNS, TRAFFIC_VOLUME_TARGET_30M, "unselected_numeric"]
    return pd.DataFrame(
        {
            column: range(3)
            for column in columns
        },
        index=[10, 20, 30],
    )


def test_select_model_features_uses_explicit_allowlist():
    features, target = select_model_features(_frame())

    assert features.columns.tolist() == list(MODEL_FEATURE_COLUMNS)
    assert target.name == TRAFFIC_VOLUME_TARGET_30M
    assert "unselected_numeric" not in features.columns


def test_select_model_features_rejects_missing_required_feature():
    frame = _frame().drop(columns=["road_capacity"])

    with pytest.raises(ValueError, match="Required model features are missing: road_capacity"):
        select_model_features(frame)


def test_select_model_features_rejects_missing_target():
    frame = _frame().drop(columns=[TRAFFIC_VOLUME_TARGET_30M])

    with pytest.raises(ValueError, match="Required target column is missing"):
        select_model_features(frame)


def test_select_model_features_rejects_target_in_allowlist():
    with pytest.raises(ValueError, match="Target column cannot be a model feature"):
        select_model_features(
            _frame(),
            feature_columns=(*MODEL_FEATURE_COLUMNS, TRAFFIC_VOLUME_TARGET_30M),
        )


def test_select_model_features_preserves_row_order_index_and_values():
    frame = _frame()

    features, target = select_model_features(frame)

    pd.testing.assert_frame_equal(features, frame.loc[:, list(MODEL_FEATURE_COLUMNS)])
    pd.testing.assert_series_equal(target, frame[TRAFFIC_VOLUME_TARGET_30M])
    assert features.index.tolist() == frame.index.tolist()
    assert target.index.tolist() == frame.index.tolist()
