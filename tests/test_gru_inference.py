import pandas as pd
import pytest

from src.gru_inference import GRUInference
from src.sequence_data import DL_NUMERIC_FEATURE_COLUMNS


def _history(periods=12, road_id="NL-001"):
    timestamps = pd.date_range("2025-01-01 00:00", periods=periods, freq="30min")
    frame = pd.DataFrame(
        {
            "road_id": road_id,
            "timestamp": timestamps,
        }
    )
    for position, column in enumerate(DL_NUMERIC_FEATURE_COLUMNS):
        frame[column] = [float(position + row) for row in range(periods)]
    return frame


@pytest.fixture
def predictor():
    return GRUInference.from_artifacts()


def test_valid_inference_returns_next_timestamp_and_finite_prediction(predictor):
    result = predictor.predict_next_30_minutes(_history())

    assert result.road_id == "NL-001"
    assert result.forecast_timestamp == pd.Timestamp("2025-01-01 06:00")
    assert pd.notna(result.predicted_traffic_volume)


def test_target_column_is_rejected_to_prevent_leakage(predictor):
    history = _history()
    history["traffic_volume_target_30m"] = 999999.0

    with pytest.raises(ValueError, match="must not contain"):
        predictor.predict_next_30_minutes(history)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(columns=["avg_speed"]), "Required prediction columns"),
        (
            lambda frame: frame.assign(timestamp="not-a-timestamp"),
            "(?i)invalid timestamps",
        ),
        (
            lambda frame: frame.assign(road_id=lambda value: value["road_id"].mask(value.index == 0)),
            "missing road identifiers",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate road/timestamp",
        ),
        (lambda frame: frame.assign(road_id="UNKNOWN"), "Unsupported road"),
    ],
)
def test_invalid_inputs_fail_clearly(predictor, mutator, message):
    with pytest.raises(ValueError, match=message):
        predictor.predict_next_30_minutes(mutator(_history()))


def test_insufficient_history_is_rejected(predictor):
    with pytest.raises(ValueError, match="at least 12"):
        predictor.predict_next_30_minutes(_history(periods=11))


def test_multiple_road_ids_are_rejected(predictor):
    history = _history()
    history.loc[0, "road_id"] = "NL-002"

    with pytest.raises(ValueError, match="exactly one road"):
        predictor.predict_next_30_minutes(history)


def test_non_contiguous_history_is_rejected(predictor):
    history = _history()
    history.loc[5:, "timestamp"] += pd.Timedelta(minutes=30)

    with pytest.raises(ValueError, match="30-minute continuity"):
        predictor.predict_next_30_minutes(history)


def test_missing_numeric_values_follow_saved_imputation(predictor):
    history = _history()
    history.loc[3, "avg_speed"] = None

    result = predictor.predict_next_30_minutes(history)

    assert pd.notna(result.predicted_traffic_volume)
