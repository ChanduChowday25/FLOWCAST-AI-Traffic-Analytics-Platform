# FlowCast

FlowCast is a student portfolio project for short-horizon traffic analytics on the
Northline Corridor. The business problem is that operations teams react to
congestion and incidents after they form. The planned system will use traffic
sensor telemetry, weather observations, and calendar signals to forecast traffic
volume, speed, congestion, travel time, and accident risk for each road segment
30–120 minutes ahead.

## Input data

- `traffic_sensor_log.csv`: segment-level traffic readings at 30-minute grain.
- `weather_observations.csv`: station-level weather observations at hourly grain.
- `calendar_events.csv`: corridor-wide calendar and event flags by date.

The supplied reference documents in `project_reference/` are authoritative. Raw
inputs are preserved unchanged in `data/raw/`.

The Phase 4 supervised label `traffic_volume_target_30m` is the observed
traffic volume for the same road segment exactly 30 minutes after each row's
timestamp. It is written to `data/processed/flowcast_target_30m.csv` from the
rolling-feature dataset. This column is a training target, not an input
feature: model training must exclude it from `X` to prevent target leakage.

## Technology stack

The planned stack is Python 3.11, Pandas, NumPy, scikit-learn, XGBoost,
PyTorch or TensorFlow/Keras, Plotly, Matplotlib, Seaborn, Streamlit, PyYAML,
joblib, and tqdm. The current environment includes Pandas, NumPy, PyYAML, and
scikit-learn. The reusable preprocessing component in `src/preprocessing.py`
fits numeric imputers and categorical encoders on training rows only, using
the explicit allowlist in `src/selection.py`.

The persistence baseline in `src/baseline.py` uses `traffic_volume_lag_1`,
which is the exact valid traffic volume at `t - 30 minutes`. Missing or
invalid lag values remain unavailable rather than being replaced with future
observations or the target. MAPE excludes zero-valued targets because
percentage error is undefined for them.

To reproduce the persistence evaluation on the processed target dataset:

```powershell
.\.venv\Scripts\python.exe -m src.run_baseline_evaluation
```

The runner uses the chronological 70/15/15 split and writes separate
validation and test results to
`reports/persistence_baseline_evaluation.json`. It does not fit a model or
modify any dataset.

The first ML model is trained with:

```powershell
.\.venv\Scripts\python.exe -m src.train_linear_regression
```

This fits `sklearn.linear_model.LinearRegression` using only the approved
feature allowlist. Preprocessing is fit on the training partition only.
Validation metrics and the saved model/preprocessor artifacts are written to
`reports/linear_regression_validation.json` and `models/`.

The Decision Tree regression model can be trained with:

```powershell
.\.venv\Scripts\python.exe -m src.train_decision_tree
```

It uses the same allowlist, chronological split, target filtering, and
validation metric conventions as Linear Regression. Its artifacts are saved
under distinct Decision Tree filenames so the Linear Regression artifacts are
not overwritten.

The Random Forest regression model can be trained with:

```powershell
.\.venv\Scripts\python.exe -m src.train_random_forest
```

It uses 50 trees, `max_depth=20`, `min_samples_leaf=2`, all available CPU
cores, and `random_state=42`. These bounded settings keep the artifact
CPU-friendly while remaining reproducible. It follows the same approved
feature allowlist, chronological split, target filtering, and validation-only
reporting conventions. Its model, preprocessor, and validation report use
distinct Random Forest filenames.

XGBoost was checked before implementing the next boosting model but is not
installed or declared in the project environment. The runnable fallback is
Gradient Boosting:

```powershell
.\.venv\Scripts\python.exe -m src.train_gradient_boosting
```

It uses 100 estimators, learning rate `0.05`, maximum depth `3`, minimum leaf
size `5`, subsampling `0.8`, and `random_state=42`. It follows the same
approved feature allowlist, chronological split, target filtering, and
validation-only reporting conventions, with distinct artifact filenames.

To regenerate the four ML validation reports from existing saved artifacts
without retraining:

```powershell
.\.venv\Scripts\python.exe -m src.regenerate_validation_reports
```

To create the aligned manual comparison for the common valid validation rows:

```powershell
.\.venv\Scripts\python.exe -m src.create_validation_comparison
```

This writes `reports/validation_prediction_comparison.csv` with the original
row index, road, timestamp, actual target, persistence prediction, and each
ML model prediction. The comparison contains only validation rows for which
both the target and persistence input are valid; it never reads the test
partition. When a temporary `models_dir` is supplied to a trainer, its
default report path is isolated under the corresponding temporary directory.

## Planned workflow

1. Understand and document the raw data (current phase).
2. Validate, clean, align, and merge the sources without losing provenance.
3. Explore relationships and engineer modelling features.
4. Train and evaluate classical and sequence models using time-aware splits.
5. Build a Streamlit dashboard around real predictions and diagnostics.

No data cleaning, source merging, target derivation, prediction, or model
training is implemented in Phase 1.

## Setup and inspection

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\inspect_data.py
```

The inspection command writes `reports/data_understanding.md` and does not
modify the raw CSV files.

## Current phase

**Phase 1 — Project foundation and data understanding.** This phase establishes
the reproducible layout, configuration, raw-data snapshot, and evidence-based
inspection report. Phase 2 will implement the documented validation and
cleaning pipeline.
