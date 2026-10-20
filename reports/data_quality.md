# FlowCast Data Quality Report

_Generated on 2026-10-13 20:48 by `src/clean.py`._

This Phase 2 report validates, cleans, and integrates traffic, weather, and calendar data without creating synthetic traffic observations.

## Source validation and hashes
- Raw files read: 178,468 traffic, 10,872 weather, 151 calendar rows.
- Raw SHA-256 hashes were unchanged after processing: **True**.
- Parse failures: 0; details are in `data/interim/parse_failures.csv`.

## Duplicates
- Exact duplicate groups: 1,767; rows in groups: 3,534; rows removed: 1,767.
- Duplicate traffic-key groups: 1,767; duplicate rows beyond first: 1,767.
- Conflicting duplicate-key groups: 0; rows retained for review: 0.
Exact duplicate removals and conflicting groups are saved with source row numbers.

## Phase 2A audit-count reconciliation
- The initial audit counted raw values meeting the same field-level predicates, while Phase 2A counted values after exact duplicate removal.
- Raw versus Phase 2A counts were traffic volume 241 vs 239, speed 237 vs 236, and occupancy 234 vs 229. These differences are attributable to invalid values on exact duplicate rows removed before measurement auditing; no cleaning rule was changed, and no conflicting duplicate keys were found.

## Invalid measurements
- `traffic_volume` invalid values: 239; affected rows: 239.
- `avg_speed` invalid values: 236; affected rows: 236.
- `occupancy` invalid values: 229; affected rows: 229.
- `road_capacity` invalid values: 0; affected rows: 0.
- All invalid measurement fields combined: 704 invalid values across 704 distinct rows.
Invalid measurement fields were set to missing in the cleaned traffic output; other fields in the row were preserved.

## Missingness
Missing counts and percentages are saved in `data/interim/missingness.csv`. Null readings are not synthetic missing windows.

### Traffic_Raw
- `traffic_volume`: 4,387 (2.46%).
- `avg_speed`: 4,382 (2.46%).
- `occupancy`: 4,383 (2.46%).
- `congestion_level`: 26,883 (15.06%).
### Traffic_Cleaned
- `traffic_volume`: 4,587 (2.60%).
- `avg_speed`: 4,579 (2.59%).
- `occupancy`: 4,573 (2.59%).
- `congestion_level`: 681 (0.39%).
- `vc_ratio`: 4,587 (2.60%).
### Weather
- `temperature`: 167 (1.54%).
- `visibility`: 111 (1.02%).
### Calendar
- `holiday_name`: 145 (96.03%).
- `event_name`: 145 (96.03%).

- Absent traffic windows: 4,499; longest consecutive gap: 2 windows (1.0 hours).
- Missing traffic-volume targets remain missing; no imputation or synthetic rows were performed.

## Weather normalization
Original and normalized category counts are saved in `data/interim/weather_label_counts.csv`.

| Original | Normalized | Count |
| --- | --- | ---: |
| CLEAR | Clear | 1,237 |
| Clear | Clear | 5,647 |
| Cloudy | Cloudy | 1,504 |
| FOG | Fog | 7 |
| Fog | Fog | 59 |
| Overcast | Overcast | 358 |
| RAIN | Rain | 34 |
| Rain | Rain | 307 |
| clear | Clear | 1,284 |
| cloudy | Cloudy | 340 |
| foggy | Fog | 10 |
| rain  | Rain | 48 |
| rainy | Rain | 37 |

## Congestion labels
- Missing labels derived from valid inputs: 25,943.
- Existing congestion labels retained: 150,077.
- Missing labels unresolved because inputs were unavailable/invalid: 681.
- Existing labels disagreeing with recalculated valid-input labels: 0. Existing labels were not overwritten.
- Formula: `V/C = traffic_volume / (road_capacity / 2)` with thresholds Free-flow `<0.50`, Moderate `[0.50,0.80)`, Heavy `[0.80,1.00)`, Severe `>=1.00`.

## Integration
- Weather join: 176,701 traffic rows before and 176,701 after; 176,701 matched, 0 unmatched; 0 weather records not used.
- Calendar join: 176,701 matched traffic rows, 0 unmatched; 151 unique calendar dates.
- Final integrated dataset: 176,701 rows; weather row delta +0, calendar row delta +0. Both joins use many-to-one validation and preserve the traffic identity.
- Weather matching uses station ID plus the traffic timestamp floored to the hour. Calendar matching uses the normalized traffic date.
- Final duplicate traffic identity rows: 0. Final missing measurements: `traffic_volume` 4,587, `avg_speed` 4,579, `occupancy` 4,573.
- Final columns (36): source_row_number, road_id, road_name, latitude, longitude, weather_station_id, date, time, traffic_volume, vehicle_count, vehicle_type_dist, avg_speed, occupancy, congestion_level, travel_time, accident_count, signal_timing, road_capacity, timestamp, congestion_level_provenance, vc_ratio, quality_flags, weather_hour, weather_date, weather_time, weather_weather_condition, weather_temperature, weather_rainfall, weather_visibility, weather_weather_condition_original, calendar_date, calendar_public_holiday, calendar_holiday_name, calendar_event_flag, calendar_event_name, calendar_roadwork_flag.
- Final CSV types are validated before writing; timestamp/hour fields are datetime values and calendar dates are normalized date values in the integration frame.

## Final dataset integrity
- Cleaned traffic measurements, quality flags, congestion labels/provenance, normalized weather fields, and calendar metadata are retained. Missing measurements remain missing where no safe imputation policy exists.
- No accident-risk target, model, feature engineering, or dashboard functionality was created.
