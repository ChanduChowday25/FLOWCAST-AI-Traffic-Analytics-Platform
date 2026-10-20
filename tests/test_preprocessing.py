import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.preprocessing import FlowCastPreprocessor
from src.selection import MODEL_FEATURE_COLUMNS
from src.targets import TRAFFIC_VOLUME_TARGET_30M


def _frame() -> pd.DataFrame:
    rows = []
    for index, road_id in enumerate(["A", "B", "A", "B"]):
        row = {column: float(index + 1) for column in MODEL_FEATURE_COLUMNS if column != "road_id"}
        row["road_id"] = road_id
        row[TRAFFIC_VOLUME_TARGET_30M] = float(index + 10)
        rows.append(row)
    return pd.DataFrame(rows, index=[10, 20, 30, 40])


def test_fit_uses_only_rows_with_observed_targets():
    train = _frame()
    train.loc[40, "road_capacity"] = 9999
    train.loc[40, TRAFFIC_VOLUME_TARGET_30M] = np.nan

    preprocessor = FlowCastPreprocessor().fit(train)

    numeric_imputer = preprocessor.pipeline_.named_transformers_["numeric"].named_steps["imputer"]
    capacity_position = list(preprocessor.pipeline_.transformers_[0][2]).index("road_capacity")
    assert numeric_imputer.statistics_[capacity_position] != 9999
    assert preprocessor.fit_row_count_ == 3
    assert preprocessor.fit_missing_target_count_ == 1


def test_missing_lags_and_road_id_are_imputed_without_row_deletion():
    frame = _frame()
    frame.loc[20, ["traffic_volume_lag_1", "traffic_volume_lag_2", "road_id"]] = np.nan

    result = FlowCastPreprocessor().fit_transform_supervised(frame)

    assert result.retained_row_count == len(frame)
    assert result.features.shape[0] == len(frame)
    assert not result.features.isna().any().any()


def test_road_id_is_one_hot_encoded_and_unknown_categories_are_safe():
    preprocessor = FlowCastPreprocessor().fit(_frame())

    transformed = preprocessor.transform(_frame().assign(road_id=["A", "C", "A", "C"]))

    assert any(column.startswith("road_id_") for column in transformed.columns)
    assert "road_id" not in transformed.columns
    assert transformed.shape[1] > len(MODEL_FEATURE_COLUMNS) - 1


def test_transformed_columns_are_consistent_across_splits():
    preprocessor = FlowCastPreprocessor().fit(_frame())
    validation = _frame().assign(road_id=["B", "C", "B", "C"])
    test = _frame().assign(road_id=["A", "A", "C", "C"])

    assert preprocessor.transform(validation).columns.tolist() == list(preprocessor.feature_names_out_)
    assert preprocessor.transform(test).columns.tolist() == list(preprocessor.feature_names_out_)


def test_target_is_separate_and_missing_targets_are_excluded():
    frame = _frame()
    frame.loc[20, TRAFFIC_VOLUME_TARGET_30M] = np.nan

    result = FlowCastPreprocessor().fit_transform_supervised(frame)

    assert TRAFFIC_VOLUME_TARGET_30M not in result.features.columns
    assert result.target.index.tolist() == [10, 30, 40]
    assert result.missing_target_count == 1


def test_invalid_and_empty_inputs_fail_clearly():
    with pytest.raises(ValueError, match="empty"):
        FlowCastPreprocessor().fit(pd.DataFrame())
    with pytest.raises(ValueError, match="Required model features are missing"):
        FlowCastPreprocessor().fit(pd.DataFrame({TRAFFIC_VOLUME_TARGET_30M: [1]}))
    with pytest.raises(TypeError, match="pandas DataFrame"):
        FlowCastPreprocessor().fit_transform_supervised([])


def test_transform_requires_fitting():
    with pytest.raises(NotFittedError, match="fitted"):
        FlowCastPreprocessor().transform(_frame())
