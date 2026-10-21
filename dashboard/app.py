"""FLOWCAST Streamlit dashboard entry point."""

from __future__ import annotations

import html
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit runs this file as a script, so ensure the project package root is
# importable before loading dashboard modules that depend on src.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __package__:
    from .data_access import load_processed_data
    from .model_performance import (
        METRICS,
        PREDICTION_COLUMNS,
        load_model_performance_reports,
        load_validation_predictions,
        road_validation_metrics,
    )
else:
    from data_access import load_processed_data
    from model_performance import (
        METRICS,
        PREDICTION_COLUMNS,
        load_model_performance_reports,
        load_validation_predictions,
        road_validation_metrics,
    )
from src.clean import CONGESTION_LEVELS
from src.gru_inference import GRUInference
from src.sequence_data import DL_NUMERIC_FEATURE_COLUMNS, SEQUENCE_FREQUENCY
from src.targets import TRAFFIC_VOLUME_TARGET_30M


TRAFFIC_VOLUME_COLUMN = "traffic_volume"
AVERAGE_SPEED_COLUMN = "avg_speed"
CONGESTION_COLUMN = "congestion_level"
ROAD_COLUMN = "road_id"
TIMESTAMP_COLUMN = "timestamp"
HOUR_COLUMN = "hour_of_day"
FORECAST_INPUT_COLUMNS = ("road_id", "timestamp", *DL_NUMERIC_FEATURE_COLUMNS)
NAVIGATION = (
    "Overview",
    "Road Forecast",
    "Historical Analytics",
    "Congestion Heatmap",
    "Road Comparison",
    "Model Performance",
)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(frame[column], errors="coerce")
    return values[values.notna() & values.map(math.isfinite)]


def historical_summary(frame: pd.DataFrame) -> dict[str, int | float | None]:
    """Return descriptive historical metrics from the processed dataset."""
    volumes = _numeric_series(frame, TRAFFIC_VOLUME_COLUMN)
    speeds = _numeric_series(frame, AVERAGE_SPEED_COLUMN)
    roads = frame[ROAD_COLUMN].nunique(dropna=True) if ROAD_COLUMN in frame else 0
    return {
        "traffic_observations": int(volumes.notna().sum()),
        "unique_roads": int(roads),
        "average_traffic_volume": float(volumes.mean()) if not volumes.empty else None,
        "average_speed": float(speeds.mean()) if not speeds.empty else None,
    }


def daily_volume_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed traffic volume by day for a readable historical chart."""
    required = {TIMESTAMP_COLUMN, TRAFFIC_VOLUME_COLUMN}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=["date", "average_traffic_volume"])
    chart = frame[[TIMESTAMP_COLUMN, TRAFFIC_VOLUME_COLUMN]].copy()
    chart[TRAFFIC_VOLUME_COLUMN] = pd.to_numeric(
        chart[TRAFFIC_VOLUME_COLUMN], errors="coerce"
    )
    chart = chart.dropna(subset=[TIMESTAMP_COLUMN, TRAFFIC_VOLUME_COLUMN])
    if chart.empty:
        return pd.DataFrame(columns=["date", "average_traffic_volume"])
    chart["date"] = chart[TIMESTAMP_COLUMN].dt.floor("D")
    return (
        chart.groupby("date", as_index=False)[TRAFFIC_VOLUME_COLUMN]
        .mean()
        .rename(columns={TRAFFIC_VOLUME_COLUMN: "average_traffic_volume"})
    )


def hourly_volume_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed traffic volume by hour of day."""
    if TRAFFIC_VOLUME_COLUMN not in frame.columns:
        return pd.DataFrame(columns=["hour_of_day", "average_traffic_volume"])
    chart = frame[[TRAFFIC_VOLUME_COLUMN]].copy()
    if HOUR_COLUMN in frame.columns:
        chart[HOUR_COLUMN] = pd.to_numeric(frame[HOUR_COLUMN], errors="coerce")
    elif TIMESTAMP_COLUMN in frame.columns:
        chart[HOUR_COLUMN] = frame[TIMESTAMP_COLUMN].dt.hour
    else:
        return pd.DataFrame(columns=["hour_of_day", "average_traffic_volume"])
    chart[TRAFFIC_VOLUME_COLUMN] = pd.to_numeric(
        chart[TRAFFIC_VOLUME_COLUMN], errors="coerce"
    )
    chart = chart.dropna(subset=[HOUR_COLUMN, TRAFFIC_VOLUME_COLUMN])
    if chart.empty:
        return pd.DataFrame(columns=["hour_of_day", "average_traffic_volume"])
    return (
        chart.groupby(HOUR_COLUMN, as_index=False)[TRAFFIC_VOLUME_COLUMN]
        .mean()
        .rename(columns={TRAFFIC_VOLUME_COLUMN: "average_traffic_volume"})
        .sort_values(HOUR_COLUMN)
    )


def historical_date_bounds(frame: pd.DataFrame) -> tuple[date, date] | None:
    """Return the available inclusive date range for historical observations."""
    if TIMESTAMP_COLUMN not in frame.columns:
        return None
    timestamps = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce").dropna()
    if timestamps.empty:
        return None
    return timestamps.min().date(), timestamps.max().date()


def filter_historical_data(
    frame: pd.DataFrame,
    start_date: date,
    end_date: date,
    road_id: str | None = None,
) -> pd.DataFrame:
    """Filter observations by inclusive dates and an optional real road identifier."""
    if start_date > end_date:
        raise ValueError("Historical start date must be on or before the end date")
    if TIMESTAMP_COLUMN not in frame.columns:
        return frame.iloc[0:0].copy()

    filtered = frame.copy()
    timestamps = pd.to_datetime(filtered[TIMESTAMP_COLUMN], errors="coerce")
    date_mask = timestamps.dt.date.ge(start_date) & timestamps.dt.date.le(end_date)
    filtered = filtered.loc[date_mask]
    if road_id is not None:
        if ROAD_COLUMN not in filtered.columns:
            return filtered.iloc[0:0].copy()
        filtered = filtered.loc[filtered[ROAD_COLUMN].astype(str).eq(road_id)]
    return filtered.reset_index(drop=True)


def filter_historical_roads(
    frame: pd.DataFrame,
    start_date: date,
    end_date: date,
    road_ids: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Filter historical observations for a selected set of real road IDs."""
    filtered = filter_historical_data(frame, start_date, end_date)
    if ROAD_COLUMN not in filtered.columns or not road_ids:
        return filtered.iloc[0:0].copy()
    selected = {str(road_id) for road_id in road_ids}
    return filtered.loc[filtered[ROAD_COLUMN].astype(str).isin(selected)].reset_index(
        drop=True
    )


def road_comparison_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate observed volume and speed means for each selected road."""
    required = {ROAD_COLUMN, TRAFFIC_VOLUME_COLUMN, AVERAGE_SPEED_COLUMN}
    if not required.issubset(frame.columns):
        return pd.DataFrame(
            columns=[
                ROAD_COLUMN,
                "average_traffic_volume",
                "average_speed",
                "volume_observations",
                "speed_observations",
            ]
        )
    comparison = frame[list(required)].copy()
    comparison[TRAFFIC_VOLUME_COLUMN] = pd.to_numeric(
        comparison[TRAFFIC_VOLUME_COLUMN], errors="coerce"
    )
    comparison[AVERAGE_SPEED_COLUMN] = pd.to_numeric(
        comparison[AVERAGE_SPEED_COLUMN], errors="coerce"
    )
    comparison = comparison.dropna(subset=[ROAD_COLUMN])
    if comparison.empty:
        return pd.DataFrame(
            columns=[
                ROAD_COLUMN,
                "average_traffic_volume",
                "average_speed",
                "volume_observations",
                "speed_observations",
            ]
        )
    return (
        comparison.groupby(ROAD_COLUMN, as_index=False)
        .agg(
            average_traffic_volume=(TRAFFIC_VOLUME_COLUMN, "mean"),
            average_speed=(AVERAGE_SPEED_COLUMN, "mean"),
            volume_observations=(TRAFFIC_VOLUME_COLUMN, "count"),
            speed_observations=(AVERAGE_SPEED_COLUMN, "count"),
        )
        .sort_values(ROAD_COLUMN)
    )


def aggregate_historical_data(
    frame: pd.DataFrame,
    aggregation: str,
) -> pd.DataFrame:
    """Average observed volume and speed at a daily or hourly time grain."""
    if aggregation not in {"Daily", "Hourly"}:
        raise ValueError("Aggregation must be either 'Daily' or 'Hourly'")
    required = {TIMESTAMP_COLUMN, TRAFFIC_VOLUME_COLUMN, AVERAGE_SPEED_COLUMN}
    if not required.issubset(frame.columns):
        return pd.DataFrame(
            columns=["interval", "average_traffic_volume", "average_speed"]
        )

    chart = frame[list(required)].copy()
    chart[TIMESTAMP_COLUMN] = pd.to_datetime(chart[TIMESTAMP_COLUMN], errors="coerce")
    chart[TRAFFIC_VOLUME_COLUMN] = pd.to_numeric(
        chart[TRAFFIC_VOLUME_COLUMN], errors="coerce"
    )
    chart[AVERAGE_SPEED_COLUMN] = pd.to_numeric(
        chart[AVERAGE_SPEED_COLUMN], errors="coerce"
    )
    chart = chart.dropna(subset=[TIMESTAMP_COLUMN])
    if chart.empty:
        return pd.DataFrame(
            columns=["interval", "average_traffic_volume", "average_speed"]
        )

    frequency = "D" if aggregation == "Daily" else "h"
    chart["interval"] = chart[TIMESTAMP_COLUMN].dt.floor(frequency)
    return (
        chart.groupby("interval", as_index=False)
        .agg(
            average_traffic_volume=(TRAFFIC_VOLUME_COLUMN, "mean"),
            average_speed=(AVERAGE_SPEED_COLUMN, "mean"),
        )
        .sort_values("interval")
    )


def aggregate_congestion_heatmap(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid observed congestion categories by road and hour."""
    required = {ROAD_COLUMN, TIMESTAMP_COLUMN, CONGESTION_COLUMN}
    if not required.issubset(frame.columns):
        return pd.DataFrame(
            columns=[ROAD_COLUMN, HOUR_COLUMN, "congestion_score", "observations"]
        )

    chart = frame[[ROAD_COLUMN, TIMESTAMP_COLUMN, CONGESTION_COLUMN]].copy()
    chart[TIMESTAMP_COLUMN] = pd.to_datetime(chart[TIMESTAMP_COLUMN], errors="coerce")
    chart[HOUR_COLUMN] = chart[TIMESTAMP_COLUMN].dt.hour
    chart[CONGESTION_COLUMN] = pd.Categorical(
        chart[CONGESTION_COLUMN],
        categories=CONGESTION_LEVELS,
        ordered=True,
    )
    chart = chart.dropna(subset=[ROAD_COLUMN, HOUR_COLUMN, CONGESTION_COLUMN])
    if chart.empty:
        return pd.DataFrame(
            columns=[ROAD_COLUMN, HOUR_COLUMN, "congestion_score", "observations"]
        )

    chart["congestion_score"] = chart[CONGESTION_COLUMN].cat.codes
    return (
        chart.groupby([ROAD_COLUMN, HOUR_COLUMN], as_index=False, observed=True)
        .agg(
            congestion_score=("congestion_score", "mean"),
            observations=("congestion_score", "size"),
        )
        .sort_values([ROAD_COLUMN, HOUR_COLUMN])
    )


def select_forecast_history(
    frame: pd.DataFrame,
    road_id: str,
    *,
    sequence_length: int,
    supported_roads: tuple[str, ...],
) -> pd.DataFrame:
    """Select the latest valid contiguous history without target information."""
    if road_id not in supported_roads:
        raise ValueError(f"Unsupported road identifier: {road_id}")
    missing = sorted(set(FORECAST_INPUT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Forecast input columns are missing: {', '.join(missing)}")

    road = frame.loc[
        frame[ROAD_COLUMN].astype(str).eq(road_id), list(FORECAST_INPUT_COLUMNS)
    ].copy()
    road[TIMESTAMP_COLUMN] = pd.to_datetime(road[TIMESTAMP_COLUMN], errors="coerce")
    for column in DL_NUMERIC_FEATURE_COLUMNS:
        road[column] = pd.to_numeric(road[column], errors="coerce")
    road = road.dropna(subset=["timestamp", *DL_NUMERIC_FEATURE_COLUMNS])
    road = road.loc[
        road[list(DL_NUMERIC_FEATURE_COLUMNS)]
        .apply(lambda column: column.map(math.isfinite))
        .all(axis=1)
    ].sort_values(TIMESTAMP_COLUMN)
    if road.empty:
        raise ValueError(f"No valid historical observations are available for {road_id}.")

    timestamps = road[TIMESTAMP_COLUMN].reset_index(drop=True)
    run_ids = timestamps.diff().ne(SEQUENCE_FREQUENCY).cumsum()
    candidates = [
        group
        for _, group in road.reset_index(drop=True).groupby(run_ids, sort=True)
        if len(group) >= sequence_length
    ]
    if not candidates:
        raise ValueError(
            f"Road {road_id} does not have {sequence_length} consecutive "
            "valid 30-minute observations."
        )
    return candidates[-1].tail(sequence_length).reset_index(drop=True)


def _format_metric(value: int | float | None, *, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{value:,.{decimals}f}"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: #050807; color: #eef6ef; }
        [data-testid="stHeader"] { background: #050807; }
        [data-testid="stSidebar"] { background: #080d0a; border-right: 1px solid #1a3321; }
        [data-testid="stSidebar"] > div:first-child { padding: 1.35rem .9rem; }
        [data-testid="stSidebar"] * { color: #c7d3c9; }
        [data-testid="stSidebar"] .stRadio label { font-size: .79rem; }
        [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { margin: 0; }
        [data-testid="stSidebar"] [data-baseweb="radio"] { padding: .42rem .55rem; border-radius: 7px; }
        [data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) { background: #112418; }
        [data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) label { color: #b6ff56 !important; }
        .block-container { max-width: 1480px; padding: 3.9rem 3.2rem 4.5rem; overflow: visible; }
        h1, h2, h3 { color: #f3f8f3 !important; letter-spacing: -.035em; }
        h1 { font-size: 2.55rem !important; line-height: 1.02 !important; margin: .22rem 0 .55rem !important; }
        .eyebrow { color: #92d63d; font-size: .66rem; font-weight: 800; letter-spacing: .2em; text-transform: uppercase; }
        .page-intro { margin-bottom: 1.4rem; padding-top: .2rem; overflow: visible; }
        .page-intro p { color: #8e9a91; font-size: .88rem; margin: 0; max-width: 670px; }
        .coverage-strip { align-items: center; background: #0a110c; border: 1px solid #1b3822; border-radius: 8px;
          color: #99a79b; display: flex; font-size: .73rem; justify-content: space-between; margin: 0 0 1.15rem;
          padding: .62rem .85rem; }
        .coverage-strip strong { color: #c7f779; }
        .coverage-dot { color: #9eea43; font-size: .9rem; margin-right: .35rem; }
        .metric-tile { background: #0a110c; border: 1px solid #1d3b24; border-radius: 10px; min-height: 112px;
          padding: 1rem 1.05rem; position: relative; }
        .metric-tile:before { background: #9eea43; border-radius: 2px; content: ""; height: 3px; left: 1.05rem;
          position: absolute; top: 0; width: 26px; }
        .metric-label { color: #8e9a91; font-size: .66rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; }
        .metric-value { color: #b8f566; font-size: 1.65rem; font-weight: 800; letter-spacing: -.045em; margin-top: .42rem; }
        .metric-note { color: #68766c; font-size: .68rem; margin-top: .24rem; }
        .section-heading { align-items: end; display: flex; justify-content: space-between; margin: 1.65rem 0 .6rem; }
        .panel-title { color: #edf6ee; font-size: .92rem; font-weight: 750; letter-spacing: -.01em; }
        .panel-subtitle { color: #76847a; font-size: .72rem; margin: .2rem 0 .62rem; }
        .chart-panel { background: #080f0a; border: 1px solid #1b3521; border-radius: 10px; padding: .9rem .95rem .35rem; }
        [data-testid="stPlotlyChart"] { background: #080f0a; border-radius: 8px; }
        [data-testid="stMetric"] { background: #0a110c; border: 1px solid #1d3b24; border-radius: 8px; padding: .65rem .75rem; }
        [data-testid="stMetricLabel"] { color: #849187 !important; font-size: .68rem !important; }
        [data-testid="stMetricValue"] { color: #d9f8d7 !important; font-size: 1.1rem !important; }
        .forecast-card { background: linear-gradient(135deg, #102318, #09110c); border: 1px solid #70ad38;
          border-radius: 10px; padding: 1.45rem 1.5rem; }
        .forecast-label { color: #93a49a; font-size: .67rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
        .forecast-value { color: #baff5d; font-size: 2.75rem; font-weight: 850; letter-spacing: -.06em; margin: .35rem 0 .2rem; }
        .context-panel { background: #080f0a; border: 1px solid #1b3521; border-radius: 10px; padding: 1rem 1.1rem; }
        .context-grid { display: grid; gap: .15rem 1.25rem; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: .85rem; }
        .context-label { color: #738177; font-size: .66rem; letter-spacing: .11em; text-transform: uppercase; }
        .context-value { color: #e6f1e7; font-size: .92rem; font-weight: 700; margin: .22rem 0 .75rem; }
        .status-line { color: #9eea43; font-size: .72rem; margin-top: .55rem; }
        @media (max-width: 800px) {
          .block-container { padding: 3.3rem 1.15rem 3.5rem; }
          .context-grid { gap: .05rem .8rem; }
          .context-value { font-size: .82rem; overflow-wrap: anywhere; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_tile(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="metric-tile"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def _chart_layout(figure: object, *, height: int) -> None:
    figure.update_layout(
        height=height,
        margin={"l": 8, "r": 12, "t": 12, "b": 8},
        paper_bgcolor="#080f0a",
        plot_bgcolor="#080f0a",
        font_color="#91a096",
        hovermode="x unified",
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 10}},
    )
    figure.update_xaxes(gridcolor="#16251a", zeroline=False, linecolor="#203a26")
    figure.update_yaxes(gridcolor="#16251a", zeroline=False, linecolor="#203a26")
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def _render_model_performance() -> None:
    st.markdown(
        '<div class="page-intro"><div class="eyebrow">FLOWCAST / MODEL EVALUATION</div>'
        '<h1>Model performance, clearly compared.</h1>'
        '<p>Historical-data evaluation of the trained forecasting approaches. These results are not live traffic performance.</p></div>',
        unsafe_allow_html=True,
    )
    reports = load_model_performance_reports()
    for warning in reports["warnings"]:
        st.warning(warning)

    st.markdown(
        '<div class="coverage-strip"><span><span class="coverage-dot">●</span><strong>VALIDATION COMPARISON</strong>'
        ' &nbsp;Historical validation partition</span><span>Lower error is better · higher R² is better</span></div>',
        unsafe_allow_html=True,
    )
    st.info(
        "RMSE, MAE, and MAPE measure prediction error, so lower values are better. "
        "R² measures fit, so higher values are better. Missing report values are shown as unavailable."
    )

    validation = reports["validation"]
    display = validation.copy()
    display.columns = ["Approach", "RMSE", "MAE", "MAPE", "R²"]
    for metric in ["RMSE", "MAE", "MAPE", "R²"]:
        display[metric] = display[metric].map(
            lambda value: "Unavailable" if pd.isna(value) else f"{value:,.3f}"
        )
    st.dataframe(display, hide_index=True, use_container_width=True)

    chart_data = validation.melt(
        id_vars="approach",
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    if chart_data.empty:
        st.warning("No valid validation metrics are available to chart.")
    else:
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown(
                '<div class="chart-panel"><div class="panel-title">Validation error metrics</div>'
                '<div class="panel-subtitle">Lower is better · only metrics present in the reports are charted.</div>',
                unsafe_allow_html=True,
            )
            errors = chart_data.loc[chart_data["metric"].isin(["rmse", "mae", "mape"])]
            if errors.empty:
                st.info("No RMSE, MAE, or MAPE values are available.")
            else:
                figure = px.bar(
                    errors,
                    x="approach",
                    y="value",
                    color="metric",
                    barmode="group",
                    labels={"approach": "Approach", "value": "Metric value", "metric": "Metric"},
                    color_discrete_map={"rmse": "#b8f566", "mae": "#42c4d1", "mape": "#d69bff"},
                )
                figure.update_xaxes(tickangle=-35)
                _chart_layout(figure, height=370)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown(
                '<div class="chart-panel"><div class="panel-title">Validation fit</div>'
                '<div class="panel-subtitle">Higher R² indicates a better fit.</div>',
                unsafe_allow_html=True,
            )
            fit = chart_data.loc[chart_data["metric"].eq("r2")]
            if fit.empty:
                st.info("No R² values are available.")
            else:
                figure = px.bar(
                    fit,
                    x="approach",
                    y="value",
                    labels={"approach": "Approach", "value": "R²"},
                    color_discrete_sequence=["#b8f566"],
                )
                figure.update_xaxes(tickangle=-35)
                figure.update_yaxes(range=[0, 1])
                _chart_layout(figure, height=370)
            st.markdown("</div>", unsafe_allow_html=True)

    predictions, prediction_warning = load_validation_predictions()
    if prediction_warning:
        st.warning(prediction_warning)
    elif predictions is not None:
        st.markdown(
            '<div class="section-heading"><div><div class="eyebrow">ROAD-LEVEL VALIDATION</div>'
            '<div class="panel-title">Compare models on shared road observations</div></div>'
            '<div class="panel-subtitle">Only saved, aligned validation predictions are used</div></div>',
            unsafe_allow_html=True,
        )
        roads = sorted(predictions["road_id"].dropna().astype(str).unique())
        selected_road = st.selectbox(
            "ROAD SEGMENT",
            ["All road segments", *roads],
            key="model_performance_road",
        )
        road_id = None if selected_road == "All road segments" else selected_road
        road_metrics, observation_count = road_validation_metrics(predictions, road_id)
        road_display = road_metrics.copy()
        road_display.columns = ["Approach", "RMSE", "MAE", "MAPE", "Observations"]
        for metric in ["RMSE", "MAE", "MAPE"]:
            road_display[metric] = road_display[metric].map(
                lambda value: "Unavailable" if pd.isna(value) else f"{value:,.3f}"
            )
        st.caption(
            f"{observation_count:,} shared valid validation observations evaluated. "
            "MAPE excludes zero-valued actual targets."
        )
        st.dataframe(road_display, hide_index=True, use_container_width=True)
        unavailable = [
            label for label in ("LSTM", "GRU") if label not in PREDICTION_COLUMNS
        ]
        st.info(
            "Road-specific results are available for Persistence baseline, Linear Regression, "
            "Decision Tree, Random Forest, and Gradient Boosting. "
            f"Per-row predictions are not saved for {', '.join(unavailable)}; their report-level "
            "validation metrics above are not presented as road-specific results."
        )

    st.markdown(
        '<div class="section-heading"><div><div class="eyebrow">HELD-OUT EVALUATION</div>'
        '<div class="panel-title">Final GRU test results</div></div>'
        '<div class="panel-subtitle">Separate test partition · not included in the validation comparison</div></div>',
        unsafe_allow_html=True,
    )
    final_metrics = reports["final_test"]
    if final_metrics is None:
        st.warning("Final GRU test evaluation is unavailable.")
    else:
        cards = st.columns(4, gap="medium")
        for card, metric in zip(cards, METRICS):
            with card:
                value = final_metrics[metric]
                _metric_tile(
                    metric.upper() if metric != "r2" else "R²",
                    "Unavailable" if pd.isna(value) else f"{value:,.3f}",
                    "final GRU test · historical data",
                )
        final_report = reports["final_report"]
        if final_report and final_report.get("test_boundaries"):
            boundaries = final_report["test_boundaries"]
            st.caption(
                f"GRU test period: {boundaries.get('timestamp_start', 'Unavailable')} "
                f"to {boundaries.get('timestamp_end', 'Unavailable')}. "
                "This is held-out historical evaluation, not live traffic performance."
            )


def _render_overview(frame: pd.DataFrame) -> None:
    st.markdown(
        '<div class="page-intro"><div class="eyebrow">FLOWCAST / HISTORICAL INTELLIGENCE</div>'
        '<h1>Traffic, clearly observed.</h1>'
        '<p>Read the daily rhythm of the network and its busiest hours from the processed historical record.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="coverage-strip"><span><span class="coverage-dot">●</span><strong>HISTORICAL DATASET</strong>'
        ' &nbsp;01 Jan 2025 — 31 May 2025</span><span>Observed only · not live traffic</span></div>',
        unsafe_allow_html=True,
    )
    summary = historical_summary(frame)
    cards = st.columns(4, gap="medium")
    metrics = (
        ("Historical observations", f"{summary['traffic_observations']:,}", "valid observed volume rows"),
        ("Road segments", f"{summary['unique_roads']:,}", "unique road identifiers"),
        ("Average volume", _format_metric(summary["average_traffic_volume"]), "vehicles / observation"),
        ("Average speed", f"{_format_metric(summary['average_speed'])} km/h", "observed mean"),
    )
    for card, (label, value, note) in zip(cards, metrics):
        with card:
            _metric_tile(label, value, note)

    daily = daily_volume_history(frame)
    hourly = hourly_volume_history(frame)
    st.markdown(
        '<div class="section-heading"><div><div class="eyebrow">NETWORK PULSE</div>'
        '<div class="panel-title">Historical traffic profile</div></div>'
        '<div class="panel-subtitle">Daily and hourly aggregation of observed vehicle volume</div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.55, 1], gap="medium")
    with left:
        st.markdown(
            '<div class="chart-panel"><div class="panel-title">Daily volume trajectory</div>'
            '<div class="panel-subtitle">Each point is the mean observed vehicle volume across all road readings for that day.</div>',
            unsafe_allow_html=True,
        )
        if daily.empty:
            st.info("Historical volume data is unavailable for this chart.")
        else:
            figure = px.area(
                daily,
                x="date",
                y="average_traffic_volume",
                labels={"date": "Date", "average_traffic_volume": "Observed volume (vehicles)"},
                hover_data={"average_traffic_volume": ":,.1f"},
            )
            figure.update_traces(line_color="#a7ed4d", fillcolor="rgba(167,237,77,.13)")
            _chart_layout(figure, height=310)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div class="chart-panel"><div class="panel-title">Volume by hour</div>'
            '<div class="panel-subtitle">Mean observed volume by hour of day · all 24 hours shown.</div>',
            unsafe_allow_html=True,
        )
        if hourly.empty:
            st.info("Hour-of-day data is unavailable.")
        else:
            hourly = (
                hourly.set_index("hour_of_day")
                .reindex(range(24))
                .rename_axis("hour_of_day")
                .reset_index()
            )
            figure = px.bar(
                hourly,
                x="hour_of_day",
                y="average_traffic_volume",
                labels={"hour_of_day": "Hour of day (0–23)", "average_traffic_volume": "Observed volume (vehicles)"},
                hover_data={"average_traffic_volume": ":,.1f"},
            )
            figure.update_traces(marker_color="#42c4d1", marker_line_width=0)
            figure.update_xaxes(tickmode="array", tickvals=list(range(24)), ticktext=[str(h) for h in range(24)], range=[-0.5, 23.5])
            _chart_layout(figure, height=310)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_historical_analytics(frame: pd.DataFrame) -> None:
    st.markdown(
        '<div class="page-intro"><div class="eyebrow">FLOWCAST / HISTORICAL ANALYTICS</div>'
        '<h1>Understand the observed network.</h1>'
        '<p>Explore recorded traffic volume and speed from the processed historical dataset.</p></div>',
        unsafe_allow_html=True,
    )
    bounds = historical_date_bounds(frame)
    if bounds is None:
        st.warning("Historical observations are unavailable.")
        return
    minimum_date, maximum_date = bounds
    st.markdown(
        '<div class="coverage-strip"><span><span class="coverage-dot">●</span><strong>HISTORICAL DATASET</strong>'
        f' &nbsp;{minimum_date:%d %b %Y} — {maximum_date:%d %b %Y}</span>'
        '<span>Historical observations only · not live traffic</span></div>',
        unsafe_allow_html=True,
    )

    roads = sorted(frame[ROAD_COLUMN].dropna().astype(str).unique())
    filter_col, road_col, aggregation_col = st.columns([1.25, 1, .85], gap="medium")
    with filter_col:
        selected_dates = st.date_input(
            "DATE RANGE",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
        )
    with road_col:
        selected_road = st.selectbox(
            "ROAD SEGMENT",
            ["All road segments", *roads],
        )
    with aggregation_col:
        aggregation = st.selectbox(
            "TIME AGGREGATION",
            ["Daily", "Hourly"],
            help="Charts show mean observed values within each selected time interval.",
        )

    if not isinstance(selected_dates, tuple) or len(selected_dates) != 2:
        st.info("Select both a start and end date to view historical analytics.")
        return
    start_date, end_date = selected_dates
    filtered = filter_historical_data(
        frame,
        start_date,
        end_date,
        None if selected_road == "All road segments" else selected_road,
    )
    summary = historical_summary(filtered)
    cards = st.columns(4, gap="medium")
    metrics = (
        ("Observed volume rows", f"{summary['traffic_observations']:,}", "valid volume observations"),
        ("Road segments", f"{summary['unique_roads']:,}", "in selected period"),
        ("Average volume", _format_metric(summary["average_traffic_volume"]), "vehicles / observation"),
        ("Average speed", f"{_format_metric(summary['average_speed'])} km/h", "observed mean"),
    )
    for card, (label, value, note) in zip(cards, metrics):
        with card:
            _metric_tile(label, value, note)

    if filtered.empty:
        st.info("No historical observations match the selected date range and road segment.")
        return
    aggregated = aggregate_historical_data(filtered, aggregation)
    if aggregated.empty:
        st.info("The selected observations do not contain chartable volume or speed values.")
        return

    interval_label = "day" if aggregation == "Daily" else "hour"
    st.markdown(
        f'<div class="section-heading"><div><div class="eyebrow">OBSERVED TRENDS</div>'
        f'<div class="panel-title">Historical traffic by {interval_label}</div></div>'
        f'<div class="panel-subtitle">Mean observed values per selected {interval_label} · no predictions shown</div></div>',
        unsafe_allow_html=True,
    )
    volume_col, speed_col = st.columns(2, gap="medium")
    with volume_col:
        st.markdown(
            '<div class="chart-panel"><div class="panel-title">Observed traffic volume</div>'
            f'<div class="panel-subtitle">Mean vehicles per {interval_label}</div>',
            unsafe_allow_html=True,
        )
        volume_figure = px.line(
            aggregated,
            x="interval",
            y="average_traffic_volume",
            markers=True,
            labels={
                "interval": "Time interval",
                "average_traffic_volume": "Mean observed volume (vehicles)",
            },
            hover_data={"average_traffic_volume": ":,.1f"},
        )
        volume_figure.update_traces(line_color="#a7ed4d")
        _chart_layout(volume_figure, height=310)
        st.markdown("</div>", unsafe_allow_html=True)
    with speed_col:
        st.markdown(
            '<div class="chart-panel"><div class="panel-title">Observed average speed</div>'
            f'<div class="panel-subtitle">Mean recorded speed per {interval_label}</div>',
            unsafe_allow_html=True,
        )
        speed_figure = px.line(
            aggregated,
            x="interval",
            y="average_speed",
            markers=True,
            labels={
                "interval": "Time interval",
                "average_speed": "Mean observed speed (km/h)",
            },
            hover_data={"average_speed": ":,.1f"},
        )
        speed_figure.update_traces(line_color="#42c4d1")
        _chart_layout(speed_figure, height=310)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_congestion_heatmap(frame: pd.DataFrame) -> None:
    st.markdown(
        '<div class="page-intro"><div class="eyebrow">FLOWCAST / CONGESTION HEATMAP</div>'
        '<h1>See observed congestion patterns.</h1>'
        '<p>Compare recorded congestion categories by road segment and hour of day across the historical dataset.</p></div>',
        unsafe_allow_html=True,
    )
    bounds = historical_date_bounds(frame)
    if bounds is None:
        st.warning("Historical congestion observations are unavailable.")
        return
    minimum_date, maximum_date = bounds
    st.markdown(
        '<div class="coverage-strip"><span><span class="coverage-dot">●</span><strong>HISTORICAL DATASET</strong>'
        f' &nbsp;{minimum_date:%d %b %Y} — {maximum_date:%d %b %Y}</span>'
        '<span>Observed congestion only · not live traffic or a prediction</span></div>',
        unsafe_allow_html=True,
    )

    roads = sorted(frame[ROAD_COLUMN].dropna().astype(str).unique())
    date_col, road_col = st.columns([1.25, 1], gap="medium")
    with date_col:
        selected_dates = st.date_input(
            "DATE RANGE",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
        )
    with road_col:
        selected_road = st.selectbox("ROAD SEGMENT", ["All road segments", *roads])

    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Select both a start and end date to view the congestion heatmap.")
        return
    start_date, end_date = selected_dates
    filtered = filter_historical_data(
        frame,
        start_date,
        end_date,
        None if selected_road == "All road segments" else selected_road,
    )
    if filtered.empty:
        st.info("No historical observations match the selected date range and road segment.")
        return

    heatmap = aggregate_congestion_heatmap(filtered)
    if heatmap.empty:
        st.info("No valid congestion categories are available for the selected observations.")
        return

    matrix = (
        heatmap.pivot(index=ROAD_COLUMN, columns=HOUR_COLUMN, values="congestion_score")
        .reindex(columns=range(24))
        .sort_index()
    )
    figure = px.imshow(
        matrix,
        x=list(range(24)),
        y=matrix.index,
        color_continuous_scale=[
            [0.0, "#102318"],
            [0.33, "#6d9f35"],
            [0.66, "#b8f566"],
            [1.0, "#f2c14e"],
        ],
        zmin=0,
        zmax=len(CONGESTION_LEVELS) - 1,
        aspect="auto",
        labels={
            "x": "Hour of day",
            "y": "Road segment",
            "color": "Observed congestion",
        },
        text_auto=False,
    )
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(24)),
        ticktext=[str(hour) for hour in range(24)],
    )
    figure.update_coloraxes(
        colorbar={
            "title": "Category",
            "tickmode": "array",
            "tickvals": list(range(len(CONGESTION_LEVELS))),
            "ticktext": list(CONGESTION_LEVELS),
        }
    )
    figure.update_traces(
        hovertemplate=(
            "Road segment: %{y}<br>Hour: %{x}:00"
            "<br>Observed category: %{customdata}<extra></extra>"
        ),
        customdata=matrix.apply(
            lambda column: column.map(
                lambda value: CONGESTION_LEVELS[int(round(value))]
                if pd.notna(value)
                else "No valid observation"
            )
        ).to_numpy(),
    )
    figure.update_layout(height=max(430, 26 * len(matrix.index) + 180))
    st.markdown(
        '<div class="section-heading"><div><div class="eyebrow">OBSERVED CONGESTION</div>'
        '<div class="panel-title">Historical congestion by road and hour</div></div>'
        '<div class="panel-subtitle">Cells show the mean ordered category across valid observations; blank cells have no valid category.</div></div>',
        unsafe_allow_html=True,
    )
    _chart_layout(figure, height=max(430, 26 * len(matrix.index) + 180))


def _render_road_comparison(frame: pd.DataFrame) -> None:
    st.markdown(
        '<div class="page-intro"><div class="eyebrow">FLOWCAST / ROAD COMPARISON</div>'
        '<h1>Compare the observed network.</h1>'
        '<p>Compare recorded traffic volume and speed across selected road segments in the historical dataset.</p></div>',
        unsafe_allow_html=True,
    )
    bounds = historical_date_bounds(frame)
    if bounds is None:
        st.warning("Historical road observations are unavailable.")
        return
    minimum_date, maximum_date = bounds
    st.markdown(
        '<div class="coverage-strip"><span><span class="coverage-dot">●</span><strong>HISTORICAL DATASET</strong>'
        f' &nbsp;{minimum_date:%d %b %Y} — {maximum_date:%d %b %Y}</span>'
        '<span>Observed values only · not live traffic or predictions</span></div>',
        unsafe_allow_html=True,
    )

    roads = sorted(frame[ROAD_COLUMN].dropna().astype(str).unique())
    date_col, road_col = st.columns([1.1, 1.9], gap="medium")
    with date_col:
        selected_dates = st.date_input(
            "DATE RANGE",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
        )
    with road_col:
        selected_roads = st.multiselect(
            "ROAD SEGMENTS",
            roads,
            default=roads[:3],
            help="Select one or more real road IDs from the processed dataset.",
        )

    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Select both a start and end date to compare roads.")
        return
    if not selected_roads:
        st.info("Select at least one road segment to compare.")
        return

    filtered = filter_historical_roads(
        frame, selected_dates[0], selected_dates[1], selected_roads
    )
    if filtered.empty:
        st.info("No historical observations match the selected dates and road segments.")
        return
    comparison = road_comparison_summary(filtered)
    if comparison.empty:
        st.info("The selected observations do not contain usable volume or speed values.")
        return

    st.markdown(
        '<div class="section-heading"><div><div class="eyebrow">OBSERVED SUMMARY</div>'
        '<div class="panel-title">Selected road performance</div></div>'
        '<div class="panel-subtitle">Means across valid observations in the selected date range</div></div>',
        unsafe_allow_html=True,
    )
    display = comparison.rename(
        columns={
            ROAD_COLUMN: "Road segment",
            "average_traffic_volume": "Average observed volume (vehicles)",
            "average_speed": "Average observed speed (km/h)",
            "volume_observations": "Volume observations",
            "speed_observations": "Speed observations",
        }
    ).copy()
    for column in (
        "Average observed volume (vehicles)",
        "Average observed speed (km/h)",
    ):
        display[column] = display[column].round(1)
    st.dataframe(display, hide_index=True, use_container_width=True)

    volume_col, speed_col = st.columns(2, gap="medium")
    with volume_col:
        st.markdown(
            '<div class="chart-panel"><div class="panel-title">Average observed traffic volume</div>'
            '<div class="panel-subtitle">Vehicles per observation · selected historical period</div>',
            unsafe_allow_html=True,
        )
        volume_figure = px.bar(
            comparison,
            x=ROAD_COLUMN,
            y="average_traffic_volume",
            labels={
                ROAD_COLUMN: "Road segment",
                "average_traffic_volume": "Average observed volume (vehicles)",
            },
            hover_data={"average_traffic_volume": ":,.1f"},
            color_discrete_sequence=["#a7ed4d"],
        )
        _chart_layout(volume_figure, height=330)
        st.markdown("</div>", unsafe_allow_html=True)
    with speed_col:
        st.markdown(
            '<div class="chart-panel"><div class="panel-title">Average observed speed</div>'
            '<div class="panel-subtitle">Kilometres per hour · selected historical period</div>',
            unsafe_allow_html=True,
        )
        speed_figure = px.bar(
            comparison,
            x=ROAD_COLUMN,
            y="average_speed",
            labels={
                ROAD_COLUMN: "Road segment",
                "average_speed": "Average observed speed (km/h)",
            },
            hover_data={"average_speed": ":,.1f"},
            color_discrete_sequence=["#42c4d1"],
        )
        _chart_layout(speed_figure, height=330)
        st.markdown("</div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _load_predictor() -> GRUInference:
    return GRUInference.from_artifacts()


def _render_forecast(frame: pd.DataFrame) -> None:
    st.markdown(
        '<div class="page-intro"><div class="eyebrow">FLOWCAST / GRU FORECAST</div>'
        '<h1>Project the next interval.</h1>'
        '<p>Generate a 30-minute traffic-volume estimate from the latest valid historical sequence for a supported road.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="coverage-strip"><span><span class="coverage-dot">●</span><strong>MODEL INPUT</strong>'
        ' &nbsp;Historical observations · January–May 2025</span><span>Not connected to live traffic sensors</span></div>',
        unsafe_allow_html=True,
    )
    try:
        predictor = _load_predictor()
    except (FileNotFoundError, TypeError, ValueError, KeyError) as exc:
        st.error(f"The GRU forecast is unavailable: {exc}")
        return
    supported_roads = tuple(predictor.preprocessor.road_categories_)
    available_roads = tuple(
        road
        for road in supported_roads
        if ROAD_COLUMN in frame.columns and frame[ROAD_COLUMN].astype(str).eq(road).any()
    )
    if not available_roads:
        st.warning("No supported road segments are available in the historical dataset.")
        return

    selector_col, info_col = st.columns([1, 2], gap="medium")
    with selector_col:
        road_id = st.selectbox("SUPPORTED ROAD SEGMENT", available_roads)
    with info_col:
        st.markdown(
            '<div class="status-line">● MODEL READY &nbsp;|&nbsp; Saved GRU artifacts &nbsp;|&nbsp; 30-minute horizon</div>',
            unsafe_allow_html=True,
        )
    try:
        history = select_forecast_history(
            frame, road_id, sequence_length=predictor.sequence_length, supported_roads=supported_roads
        )
        prediction = predictor.predict_next_30_minutes(history)
    except (TypeError, ValueError, KeyError) as exc:
        st.warning(f"A forecast could not be generated for {road_id}: {exc}")
        return

    latest = pd.Timestamp(history[TIMESTAMP_COLUMN].max())
    forecast_col, detail_col = st.columns([1.15, 1], gap="medium")
    with forecast_col:
        st.markdown(
            f'<div class="forecast-card"><div class="forecast-label">Predicted traffic volume · GRU output</div>'
            f'<div class="forecast-value">{prediction.predicted_traffic_volume:,.0f}</div>'
            f'<div class="panel-subtitle">vehicles expected in the interval beginning {prediction.forecast_timestamp:%d %b %Y · %H:%M}</div></div>',
            unsafe_allow_html=True,
        )
    with detail_col:
        st.markdown(
            f'<div class="context-panel"><div class="panel-title">Forecast context</div>'
            f'<div class="context-grid">'
            f'<div><div class="context-label">Road segment</div>'
            f'<div class="context-value">{html.escape(prediction.road_id)}</div></div>'
            f'<div><div class="context-label">Forecast interval</div>'
            f'<div class="context-value">{prediction.forecast_timestamp:%d %b %Y · %H:%M}</div></div>'
            f'<div><div class="context-label">Latest input observation</div>'
            f'<div class="context-value">{latest:%d %b %Y · %H:%M}</div></div>'
            f'<div><div class="context-label">Input sequence</div>'
            f'<div class="context-value">{len(history)} × 30 min</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    chart = history[[TIMESTAMP_COLUMN, TRAFFIC_VOLUME_COLUMN]].copy()
    chart[TRAFFIC_VOLUME_COLUMN] = pd.to_numeric(chart[TRAFFIC_VOLUME_COLUMN], errors="coerce")
    chart["series"] = "Observed history"
    chart = pd.concat(
        [
            chart,
            pd.DataFrame({TIMESTAMP_COLUMN: [prediction.forecast_timestamp], TRAFFIC_VOLUME_COLUMN: [prediction.predicted_traffic_volume], "series": ["GRU forecast"]}),
        ],
        ignore_index=True,
    )
    st.markdown(
        '<div class="section-heading"><div><div class="eyebrow">SEQUENCE CONTEXT</div>'
        '<div class="panel-title">Observed history → forecast</div></div>'
        '<div class="panel-subtitle">Observed input · neon green = model output</div></div>',
        unsafe_allow_html=True,
    )
    figure = px.line(
        chart,
        x=TIMESTAMP_COLUMN,
        y=TRAFFIC_VOLUME_COLUMN,
        color="series",
        markers=True,
        color_discrete_map={"Observed history": "#42c4d1", "GRU forecast": "#b8f566"},
        labels={TIMESTAMP_COLUMN: "Timestamp", TRAFFIC_VOLUME_COLUMN: "Traffic volume (vehicles)", "series": "Data type"},
    )
    figure.update_traces(selector={"name": "GRU forecast"}, marker={"size": 10}, line={"width": 3})
    _chart_layout(figure, height=335)


def main() -> None:
    st.set_page_config(
        page_title="FLOWCAST — AI Traffic Analytics",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    st.sidebar.markdown(
        '<div style="font-size:1.45rem;font-weight:850;color:#f0f7f0;">FLOW<span style="color:#b8f566;">CAST</span></div>'
        '<div style="color:#758478;font-size:.62rem;letter-spacing:.16em;">AI TRAFFIC ANALYTICS</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    page = st.sidebar.radio("WORKSPACE", NAVIGATION)
    st.sidebar.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div style="color:#92d63d;font-size:.63rem;font-weight:700;letter-spacing:.14em;">DATA COVERAGE</div>'
        '<div style="color:#e2ebe2;font-size:.79rem;margin-top:.35rem;">01 JAN — 31 MAY 2025</div>'
        '<div style="color:#718074;font-size:.68rem;margin-top:.2rem;">Historical · 30-minute grain</div>',
        unsafe_allow_html=True,
    )

    if page == "Model Performance":
        _render_model_performance()
        return

    try:
        frame = load_processed_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Historical data could not be loaded: {exc}")
        st.stop()
    if TRAFFIC_VOLUME_TARGET_30M in frame.columns:
        st.error("The training target was unexpectedly included in dashboard inputs.")
        st.stop()
    if page == "Overview":
        _render_overview(frame)
    elif page == "Road Forecast":
        _render_forecast(frame)
    elif page == "Historical Analytics":
        _render_historical_analytics(frame)
    elif page == "Congestion Heatmap":
        _render_congestion_heatmap(frame)
    elif page == "Road Comparison":
        _render_road_comparison(frame)


if __name__ == "__main__":
    main()
