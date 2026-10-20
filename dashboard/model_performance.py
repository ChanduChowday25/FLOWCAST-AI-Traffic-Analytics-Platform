"""Report loading and presentation data for the model performance page."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd

try:
    from .data_access import configured_path, default_report_path, load_json_report, load_config
except ImportError:  # pragma: no cover - used by direct Streamlit execution
    from data_access import configured_path, default_report_path, load_json_report, load_config

METRICS = ("rmse", "mae", "mape", "r2")
ROAD_METRICS = ("rmse", "mae", "mape")
PREDICTION_FILE = "validation_prediction_comparison.csv"
PREDICTION_COLUMNS = {
    "Persistence baseline": "persistence_prediction",
    "Linear Regression": "linear_regression_prediction",
    "Decision Tree": "decision_tree_prediction",
    "Random Forest": "random_forest_prediction",
    "Gradient Boosting": "gradient_boosting_prediction",
}
PREDICTION_REQUIRED_COLUMNS = (
    "road_id",
    "actual_target_30m",
    *PREDICTION_COLUMNS.values(),
)
APPROACHES = (
    ("Persistence baseline", "persistence_baseline_evaluation.json", "persistence_baseline"),
    ("Linear Regression", "linear_regression_validation.json", "linear_regression"),
    ("Decision Tree", "decision_tree_regression_validation.json", "decision_tree"),
    ("Random Forest", "random_forest_regression_validation.json", "random_forest"),
    ("Gradient Boosting", "gradient_boosting_regression_validation.json", "gradient_boosting"),
    ("LSTM", "lstm_volume_validation.json", "lstm"),
    ("GRU", "gru_volume_validation.json", "gru"),
)

ReportLoader = Callable[..., dict[str, Any]]


def load_validation_predictions(
    report_dir: str | Path | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """Load aligned, validation-only per-row predictions when available."""
    directory = Path(report_dir) if report_dir is not None else None
    path = (
        directory / PREDICTION_FILE
        if directory is not None
        else configured_path(load_config()["paths"]["reports_dir"]) / PREDICTION_FILE
    )
    if not path.is_file():
        return None, f"Saved validation predictions are missing: {path}"
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return None, f"Could not read saved validation predictions: {path} ({exc})"
    missing = sorted(set(PREDICTION_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        return None, (
            f"Saved validation predictions are missing required columns "
            f"({', '.join(missing)}): {path}"
        )
    return frame, None


def road_validation_metrics(
    predictions: pd.DataFrame,
    road_id: str | None = None,
) -> tuple[pd.DataFrame, int]:
    """Calculate comparable road metrics on rows valid for every saved model."""
    selected = predictions if road_id is None else predictions.loc[
        predictions["road_id"].astype(str).eq(road_id)
    ]
    numeric = selected[list(PREDICTION_COLUMNS.values()) + ["actual_target_30m"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = numeric.notna() & numeric.apply(lambda column: column.map(math.isfinite))
    common = numeric.loc[valid.all(axis=1)]
    rows: list[dict[str, Any]] = []
    for label, column in PREDICTION_COLUMNS.items():
        actual = common["actual_target_30m"]
        predicted = common[column]
        errors = predicted - actual
        mape_rows = actual.ne(0)
        rows.append(
            {
                "approach": label,
                "rmse": float((errors.pow(2).mean()) ** 0.5),
                "mae": float(errors.abs().mean()),
                "mape": float((errors.abs().loc[mape_rows] / actual.abs().loc[mape_rows]).mean() * 100)
                if mape_rows.any()
                else math.nan,
                "observations": int(len(common)),
            }
        )
    return pd.DataFrame(rows, columns=["approach", *ROAD_METRICS, "observations"]), len(common)


def _metric_values(section: Any, *, source: str, warnings: list[str]) -> dict[str, float]:
    if not isinstance(section, dict):
        warnings.append(f"Validation metrics are unavailable in {source}.")
        return {metric: math.nan for metric in METRICS}

    values: dict[str, float] = {}
    for metric in METRICS:
        value = section.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values[metric] = float(value)
        else:
            values[metric] = math.nan
            warnings.append(f"{metric.upper()} is unavailable in {source}.")
    return values


def load_model_performance_reports(
    report_dir: str | Path | None = None,
    *,
    loader: ReportLoader = load_json_report,
) -> dict[str, Any]:
    """Load dashboard evaluation reports without substituting missing metrics."""
    directory = Path(report_dir) if report_dir is not None else None
    warnings: list[str] = []
    loaded: dict[str, dict[str, Any]] = {}

    def read_report(filename: str, required_keys: tuple[str, ...]) -> dict[str, Any] | None:
        path = directory / filename if directory is not None else default_report_path(filename)
        try:
            return loader(path, required_keys=required_keys)
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(str(exc))
            return None

    comparison = read_report(
        "all_models_validation_comparison.json",
        ("metrics", "validation_only"),
    )
    if comparison is not None and comparison.get("validation_only") is not True:
        warnings.append("The all-models report is not marked validation-only.")
        comparison = None

    rows: list[dict[str, Any]] = []
    for label, filename, key in APPROACHES:
        report = read_report(filename, ("validation",))
        if report is not None:
            loaded[key] = report
        values = _metric_values(
            report.get("validation") if report is not None else None,
            source=filename,
            warnings=warnings,
        )
        rows.append({"approach": label, **values})

    if comparison is not None:
        comparison_metrics = comparison.get("metrics")
        if isinstance(comparison_metrics, dict):
            for row in rows:
                key = next((item[2] for item in APPROACHES if item[0] == row["approach"]), "")
                if key in {"random_forest", "lstm", "gru"}:
                    row.update(_metric_values(
                        comparison_metrics.get(key),
                        source="all_models_validation_comparison.json",
                        warnings=warnings,
                    ))

    final_report = read_report(
        "final_test_evaluation.json",
        ("metrics", "selected_candidate", "test_partition_only_for_evaluation"),
    )
    final_metrics = None
    if final_report is not None:
        if final_report.get("test_partition_only_for_evaluation") is not True:
            warnings.append("The final test report is not marked test-partition-only.")
        final_metrics = _metric_values(
            final_report.get("metrics"),
            source="final_test_evaluation.json",
            warnings=warnings,
        )

    return {
        "validation": pd.DataFrame(rows, columns=["approach", *METRICS]),
        "final_test": final_metrics,
        "warnings": tuple(dict.fromkeys(warnings)),
        "comparison_report": comparison,
        "final_report": final_report,
        "loaded_reports": loaded,
    }
