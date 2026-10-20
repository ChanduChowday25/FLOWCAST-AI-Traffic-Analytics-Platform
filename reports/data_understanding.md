# FlowCast Data Understanding

_Generated on 2026-10-12 20:49 by `src/inspect_data.py`._

This report is descriptive only. Raw CSV files were read without cleaning, merging, label changes, imputation, outlier removal, feature derivation, or modelling.

## Traffic dataset
- **Shape:** 178,468 rows × 17 columns.
- **Columns and dtypes:** `road_id` (object), `road_name` (object), `latitude` (float64), `longitude` (float64), `weather_station_id` (object), `date` (object), `time` (object), `traffic_volume` (float64), `vehicle_count` (int64), `vehicle_type_dist` (object), `avg_speed` (float64), `occupancy` (float64), `congestion_level` (object), `travel_time` (float64), `accident_count` (int64), `signal_timing` (int64), `road_capacity` (int64).
- **Duplicate rows:** 1,767.
- **Date/time coverage:** 2025-01-01 00:00 to 2025-05-31 23:30.
- **Missing values:**
- `traffic_volume`: 4,387
- `avg_speed`: 4,382
- `occupancy`: 4,383
- `congestion_level`: 26,883
- **Sample records:**
| road_id | road_name | latitude | longitude | weather_station_id | date | time | traffic_volume | vehicle_count | vehicle_type_dist | avg_speed | occupancy | congestion_level | travel_time | accident_count | signal_timing | road_capacity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NL-007 | Kingsway @ Park | 28.647903 | 77.149142 | WS-NORTH | 2025-02-26 | 07:30 | 706.0 | 706 | {"2W": 0.37, "Car": 0.43, "LCV": 0.11, "HCV": 0.09} | 31.2 | 59.6 | Moderate | 2.12 | 0 | 54 | 1928 |
| NL-014 | Industrial Link B | 28.594749 | 77.253501 | WS-CENTRAL | 2025-03-23 | 21:00 | 308.0 | 308 | {"2W": 0.43, "Car": 0.39, "LCV": 0.08, "HCV": 0.1} | 53.5 | 16.8 | Free-flow | 2.03 | 0 | 32 | 2327 |
| NL-022 | Old Town @ Clock | 28.674521 | 77.219835 | WS-NORTH | 2025-04-05 | 11:00 | 342.0 | 342 | {"2W": 0.4, "Car": 0.33, "LCV": 0.15, "HCV": 0.12} | 49.9 | 27.5 | Free-flow | 3.46 | 0 | 38 | 1771 |
- **Numerical summary:**
| index | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latitude | 178468.0 | 28.628 | 0.034 | 28.564 | 28.599 | 28.627 | 28.65 | 28.683 |
| longitude | 178468.0 | 77.199 | 0.051 | 77.107 | 77.162 | 77.196 | 77.239 | 77.312 |
| traffic_volume | 174081.0 | 430.619 | 283.745 | -59.0 | 195.0 | 361.0 | 600.0 | 2090.0 |
| vehicle_count | 178468.0 | 431.418 | 283.579 | 41.0 | 195.0 | 362.0 | 601.0 | 2090.0 |
| avg_speed | 174086.0 | 42.42 | 14.147 | 6.7 | 36.2 | 45.0 | 50.9 | 319.6 |
| occupancy | 174085.0 | 31.37 | 20.469 | 3.0 | 14.2 | 27.0 | 43.8 | 259.9 |
| travel_time | 178468.0 | 3.566 | 2.532 | 0.8 | 2.2 | 3.12 | 4.09 | 29.91 |
| accident_count | 178468.0 | 0.009 | 0.098 | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 |
| signal_timing | 178468.0 | 44.676 | 8.673 | 21.0 | 38.0 | 44.0 | 51.0 | 74.0 |
| road_capacity | 178468.0 | 1882.012 | 326.704 | 1259.0 | 1670.0 | 1888.0 | 2184.0 | 2327.0 |

## Weather dataset
- **Shape:** 10,872 rows × 7 columns.
- **Columns and dtypes:** `station_id` (object), `date` (object), `time` (object), `weather_condition` (object), `temperature` (float64), `rainfall` (float64), `visibility` (float64).
- **Duplicate rows:** 0.
- **Date/time coverage:** 2025-01-01 00:00 to 2025-05-31 23:00.
- **Missing values:**
- `temperature`: 167
- `visibility`: 111
- **Sample records:**
| station_id | date | time | weather_condition | temperature | rainfall | visibility |
| --- | --- | --- | --- | --- | --- | --- |
| WS-NORTH | 01/01/2025 | 00:00 | Clear | 12.7 | 0.0 | 10000.0 |
| WS-NORTH | 01/01/2025 | 01:00 | Clear | 8.8 | 0.0 | 10000.0 |
| WS-NORTH | 01/01/2025 | 02:00 | clear | 9.8 | 0.0 | 10000.0 |
- **Numerical summary:**
| index | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temperature | 10705.0 | 17.98 | 4.376 | 5.8 | 14.6 | 18.0 | 21.3 | 29.8 |
| rainfall | 10872.0 | 0.191 | 1.056 | 0.0 | 0.0 | 0.0 | 0.0 | 8.9 |
| visibility | 10761.0 | 9674.191 | 1499.341 | 301.0 | 10000.0 | 10000.0 | 10000.0 | 10000.0 |

## Calendar dataset
- **Shape:** 151 rows × 6 columns.
- **Columns and dtypes:** `date` (object), `public_holiday` (int64), `holiday_name` (object), `event_flag` (int64), `event_name` (object), `roadwork_flag` (int64).
- **Duplicate rows:** 0.
- **Date/time coverage:** 2025-01-01 00:00 to 2025-05-31 00:00.
- **Missing values:**
- `holiday_name`: 145
- `event_name`: 145
- **Sample records:**
| date | public_holiday | holiday_name | event_flag | event_name | roadwork_flag |
| --- | --- | --- | --- | --- | --- |
| 2025-01-01 | 1 | New Year's Day | 0 | nan | 0 |
| 2025-01-02 | 0 | nan | 0 | nan | 0 |
| 2025-01-03 | 0 | nan | 0 | nan | 0 |
- **Numerical summary:**
| index | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| public_holiday | 151.0 | 0.04 | 0.196 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| event_flag | 151.0 | 0.04 | 0.196 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| roadwork_flag | 151.0 | 0.073 | 0.261 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |

## Observed relationships and Phase 2 questions
- Traffic contains 25 distinct road IDs and 151 distinct date strings.
- Weather contains 3 distinct station IDs and 151 distinct date strings.
- The traffic table is the documented modelling grain: one road segment per 30-minute window.
- The weather table is hourly by station; the Data Dictionary states that both traffic windows in an hour share that hour's weather.
- Calendar records are date-level and are intended to join to traffic by date.
- The raw traffic date format is YYYY-MM-DD; weather uses DD/MM/YYYY. This is an observed schema difference, not corrected here.
- `vehicle_type_dist` is stored as a JSON string and has not been expanded.
- Phase 2 must investigate missing windows versus null readings, duplicate keys, invalid numeric values, blank congestion labels, inconsistent weather labels, and join coverage.
- Phase 2 must define and document the timestamp alignment and validation rules before cleaning or merging.

## Observed data-quality facts
- Traffic has 241 negative `traffic_volume` values.
- Traffic has 237 `avg_speed` values above 200 km/h.
- Traffic has 234 `occupancy` values above 100%.
- Traffic has 26,883 blank `congestion_level` values.
- Traffic has 1,767 duplicate segment/date/time keys.
- The observed traffic keys cover 176,701 of 181,200 possible road/date/half-hour combinations before accounting for duplicate records.
- Raw weather condition labels observed (13): `CLEAR`, `Clear`, `Cloudy`, `FOG`, `Fog`, `Overcast`, `RAIN`, `Rain`, `clear`, `cloudy`, `foggy`, `rain `, `rainy`.

## Expected issues documented by the specifications
The Data Dictionary and PRD describe expected defects including traffic missing readings, duplicate detector retries, negative traffic volume, impossible speeds, occupancy above 100%, blank congestion labels, inconsistent weather labels, mixed date formats, and the nested vehicle-type field. These are listed as questions for Phase 2; this report does not claim that each defect was quantified or repaired.
