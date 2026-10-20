import json

import pandas as pd
import pytest

from dashboard.data_access import load_json_report, load_processed_data
from src.sequence_data import DL_FEATURE_COLUMNS
from src.targets import TRAFFIC_VOLUME_TARGET_30M


def _processed_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "road_id": ["NL-001", "NL-001"],
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="30min"),
            TRAFFIC_VOLUME_TARGET_30M: [101.0, 102.0],
        }
    )
    for column in DL_FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = 1.0
    return frame


def test_processed_data_excludes_training_target(tmp_path):
    source = tmp_path / "processed.csv"
    _processed_frame().to_csv(source, index=False)

    loaded = load_processed_data(source)

    assert TRAFFIC_VOLUME_TARGET_30M not in loaded.columns
    assert set(["road_id", "timestamp"]).issubset(loaded.columns)


def test_processed_data_missing_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Processed dataset is missing"):
        load_processed_data(tmp_path / "missing.csv")


def test_processed_data_invalid_timestamp_fails_clearly(tmp_path):
    source = tmp_path / "processed.csv"
    frame = _processed_frame()
    frame.loc[0, "timestamp"] = "invalid"
    frame.to_csv(source, index=False)

    with pytest.raises(ValueError, match="invalid timestamps"):
        load_processed_data(source)


def test_json_report_loads_and_validates_schema(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"metrics": {}, "model": "gru"}), encoding="utf-8")

    report = load_json_report(report_path, required_keys=("metrics", "model"))

    assert report["model"] == "gru"


def test_json_report_missing_file_and_keys_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Report is missing"):
        load_json_report(tmp_path / "missing.json")

    report_path = tmp_path / "invalid.json"
    report_path.write_text(json.dumps({"model": "gru"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_json_report(report_path, required_keys=("metrics",))
