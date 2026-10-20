# FlowCast Phase 3A — EDA Findings

Generated from `notebooks/03_eda.ipynb` using the Phase 2 integrated dataset. The notebook is descriptive only: it does not impute values, create an accident-risk target, engineer modelling features, or train a model.

## Dataset and coverage

- Source: `data/processed/flowcast_clean.csv`
- Shape: **176,701 rows × 36 source columns**. The notebook adds temporary `hour` and `day_of_week` analysis columns in memory only.
- Traffic grain: road segment × half-hour timestamp.
- Coverage: **25 road segments**, **7,248 unique timestamps**, from **2025-01-01 00:00** through **2025-05-31 23:30**.
- Rows per road range from **7,045** (`NL-015`) to **7,094** (`NL-013`), so row coverage is broadly comparable but not identical.
- The Phase 2 missing-window audit identified **4,499 absent traffic windows**; this EDA does not create them.

The final dataset retains the cleaned traffic identity and fields, normalized weather fields, and calendar metadata. The notebook confirms that `timestamp`, `weather_hour`, and `calendar_date` load as datetime columns; source `date` and `time` strings are retained for provenance.

## Missingness and validity

| Field | Missing rows | Missing % |
| --- | ---: | ---: |
| `traffic_volume` | 4,587 | 2.60% |
| `avg_speed` | 4,579 | 2.59% |
| `occupancy` | 4,573 | 2.59% |
| `congestion_level` | 681 | 0.39% |
| `weather_temperature` | 2,708 | 1.53% |
| `weather_visibility` | 1,806 | 1.02% |
| `weather_rainfall` | 0 | 0.00% |

The measurement distributions use existing valid cleaned observations: 172,114 traffic-volume values, 172,122 speeds within 0–200 km/h, and 172,128 occupancies within 0–100%. No missing values were imputed. Invalid measurements and missing measurements remain conceptually distinct and should remain distinct in Phase 3B.

## Historical traffic features

The Phase 3B feature pipeline adds two past-only traffic-volume summaries:

- `traffic_volume_rolling_mean_3` averages the exact earlier half-hour timestamps
  `t-30`, `t-60`, and `t-90` minutes.
- `traffic_volume_rolling_mean_6` averages the exact earlier half-hour timestamps
  `t-30` through `t-180` minutes.

Both features exclude the current timestamp and are calculated separately for each
`road_id`. They use exact `(road_id, timestamp)` keys rather than adjacent CSV
rows, so missing half-hour timestamps are not silently bridged and values from
another road segment cannot enter the calculation.

Missing, nonnumeric, and negative historical traffic volumes are treated as
unavailable and excluded from the mean. The implementation uses `skipna=True`,
so a rolling mean may be based on fewer than three or six valid observations
when some historical periods are unavailable. If every historical value in a
window is unavailable, the feature remains missing; it is not replaced with
zero or another imputed value. These summaries describe recent history and are
not predictions themselves.

## Traffic measurements

For valid observations:

- Traffic volume: mean **431.29**, median **362**, interquartile range **195–601**, maximum **2,090**.
- Average speed: mean **42.11 km/h**, median **44.9**, interquartile range **36.2–50.9**, maximum **64.8 km/h**.
- Occupancy: mean **31.16%**, median **27.0%**, interquartile range **14.2–43.7%**, maximum **100%**.
- The volume distribution is right-skewed, with substantially larger values during commute periods.

## Congestion class balance

| Class | Rows | Share |
| --- | ---: | ---: |
| Free-flow | 108,139 | 61.20% |
| Moderate | 41,905 | 23.72% |
| Heavy | 16,257 | 9.20% |
| Severe | 9,719 | 5.50% |
| Missing | 681 | 0.39% |

Free-flow is the majority class and Severe is relatively rare. Phase 3B classification evaluation should therefore report per-class metrics and macro-averaged metrics rather than accuracy alone.

## Temporal patterns

Hourly mean traffic volume ranges from **162.23 at 00:00** to **886.15 at 08:00**. A second evening peak appears at **18:00**, with mean volume **811.79**. Mean speed falls to **21.91 km/h at 08:00** and **25.68 km/h at 18:00**, while overnight speeds are approximately 52 km/h.

Mean volume by day of week is highest on Monday (**483.84**) and Friday (**482.49**) and lower on Saturday (**325.57**) and Sunday (**331.23**). The daily mean volume ranges from **186.40 on 2025-03-15** to **653.49 on 2025-02-28**. These are descriptive results over the available five-month period and may include calendar or coverage effects.

## Weather and measurement relationships

The largest normalized weather groups are Clear (**132,715 rows**), Cloudy (**29,926**), Rain (**6,959**), Overcast (**5,853**), and Fog (**1,248**). Group means show lower traffic volume in Rain (**367.83**) than Clear (**437.27**), while Fog has the lowest observed mean volume (**217.88**). These groups are not equally sized, and the results are associations rather than causal effects.

Pearson correlations calculated pairwise with at least 100 non-null observations include:

- Traffic volume vs average speed: **-0.896**
- Traffic volume vs occupancy: **0.938**
- Traffic volume vs weather temperature: **0.386**
- Traffic volume vs rainfall: **-0.038**
- Traffic volume vs visibility: **0.073**
- Rainfall vs visibility: **-0.759**

The strong traffic-volume/speed and traffic-volume/occupancy relationships are expected to be influenced by the traffic system and the derived congestion/V-C fields; they should not be interpreted as causal evidence. Weather-condition comparisons also need care because weather categories and rainfall are unevenly distributed and weather values have missingness.

## Charts generated

The notebook saves these reproducible charts under `reports/figures/`:

- `eda_road_coverage.png`
- `eda_measurement_distributions.png`
- `eda_congestion_distribution.png`
- `eda_temporal_patterns.png`
- `eda_daily_volume_trend.png`
- `eda_measurement_relationships.png`

## Recommendations for Phase 3B

1. Use hour-of-day, day-of-week, and calendar flags as candidate temporal predictors, while preserving chronological ordering for validation.
2. Add explicit missingness indicators for traffic and weather measurements; do not silently impute targets.
3. Treat congestion as imbalanced and evaluate rare Heavy/Severe classes explicitly.
4. Consider road-specific baselines because segment row counts and traffic levels differ.
5. Separate current-observation predictors from future-derived values to avoid target leakage.
6. Expand `vehicle_type_dist` only after defining a stable parsing and validation policy.
7. Treat weather-condition and weather-measurement associations as exploratory; test them with time-aware validation rather than assuming causal effects.

No feature engineering, prediction horizon, accident-risk target, dashboard work, or model training was started in Phase 3A.
