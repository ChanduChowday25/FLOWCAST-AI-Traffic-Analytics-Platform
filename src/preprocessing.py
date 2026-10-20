"""Leakage-safe preprocessing for the FlowCast supervised feature set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.selection import MODEL_FEATURE_COLUMNS, select_model_features
from src.targets import TRAFFIC_VOLUME_TARGET_30M


CATEGORICAL_FEATURE_COLUMNS = ("road_id",)
NUMERICAL_FEATURE_COLUMNS = tuple(
    column for column in MODEL_FEATURE_COLUMNS if column not in CATEGORICAL_FEATURE_COLUMNS
)


@dataclass(frozen=True)
class PreprocessedSplit:
    """Transformed supervised rows and an audit of target-row filtering."""

    features: pd.DataFrame
    target: pd.Series
    input_row_count: int
    retained_row_count: int
    missing_target_count: int


class FlowCastPreprocessor(BaseEstimator, TransformerMixin):
    """Fit feature imputers and encoders on training rows only.

    The input to this component is a frame containing the explicit model
    allowlist and ``traffic_volume_target_30m``. Historical lag nulls are
    imputed as ordinary numeric feature missingness; they do not cause rows to
    be discarded. Rows with missing targets are excluded only by the supervised
    helpers and are never used to fit an imputer or encoder.
    """

    def __init__(
        self,
        *,
        feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
        target_column: str = TRAFFIC_VOLUME_TARGET_30M,
    ) -> None:
        if not feature_columns:
            raise ValueError("At least one model feature is required")
        if target_column in feature_columns:
            raise ValueError("Target column cannot be a model feature")
        if "road_id" not in feature_columns:
            raise ValueError("The model feature allowlist must include road_id")
        self.feature_columns = tuple(feature_columns)
        self.target_column = target_column

    def _build_pipeline(self) -> ColumnTransformer:
        categorical = [column for column in self.feature_columns if column in CATEGORICAL_FEATURE_COLUMNS]
        numerical = [column for column in self.feature_columns if column not in CATEGORICAL_FEATURE_COLUMNS]
        transformers: list[tuple[str, Pipeline, list[str]]] = []

        if numerical:
            transformers.append(
                (
                    "numeric",
                    Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]),
                    numerical,
                )
            )
        if categorical:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical,
                )
            )
        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

    def _validated_features(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        features, target = select_model_features(
            frame,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
        )
        # Coercion makes malformed numeric values safe missing values without
        # allowing them to enter the categorical encoder.
        for column in NUMERICAL_FEATURE_COLUMNS:
            if column in features:
                features[column] = pd.to_numeric(features[column], errors="coerce")
        if "road_id" in features:
            features["road_id"] = features["road_id"].astype("object")
        return features, target

    def fit(self, frame: pd.DataFrame, y: Any = None) -> "FlowCastPreprocessor":
        """Fit transformations using only rows whose target is present."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if frame.empty:
            raise ValueError("Cannot fit preprocessing on an empty frame")
        features, target = self._validated_features(frame)
        valid_target = target.notna()
        if not valid_target.any():
            raise ValueError("Cannot fit preprocessing: all target values are missing")

        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(features.loc[valid_target])
        self.fit_row_count_ = int(valid_target.sum())
        self.fit_missing_target_count_ = int((~valid_target).sum())
        self.feature_names_out_ = tuple(self.pipeline_.get_feature_names_out())
        return self

    def _transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "pipeline_"):
            raise NotFittedError("FlowCastPreprocessor must be fitted before transform")
        features, _ = self._validated_features(frame)
        if features.empty:
            return pd.DataFrame(index=features.index, columns=self.feature_names_out_, dtype="float64")
        transformed = self.pipeline_.transform(features)
        return pd.DataFrame(transformed, index=features.index, columns=self.feature_names_out_)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform all rows while preserving their index and row count."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        return self._transform_features(frame)

    def transform_supervised(self, frame: pd.DataFrame) -> PreprocessedSplit:
        """Transform only rows with observed targets and return an audit."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        features, target = self._validated_features(frame)
        valid_target = target.notna()
        transformed = self._transform_features(frame.loc[valid_target])
        retained_target = target.loc[valid_target].copy()
        return PreprocessedSplit(
            features=transformed,
            target=retained_target,
            input_row_count=len(frame),
            retained_row_count=int(valid_target.sum()),
            missing_target_count=int((~valid_target).sum()),
        )

    def fit_transform_supervised(self, frame: pd.DataFrame) -> PreprocessedSplit:
        """Fit on observed-target training rows and transform that split."""
        self.fit(frame)
        return self.transform_supervised(frame)
