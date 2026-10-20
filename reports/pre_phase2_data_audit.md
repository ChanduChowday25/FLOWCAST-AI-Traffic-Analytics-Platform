# FlowCast Pre-Phase 2 Data Audit

Read-only audit of raw CSVs. No records were removed, imputed, relabelled, merged, or overwritten. Rules: traffic segment x 30 minutes, weather station x hour, calendar date, congestion V/C = `traffic_volume / (road_capacity / 2)`.

## 1. Executive summary

- Traffic/weather/calendar rows: 178,468/10,872/151.
- Exact duplicate rows: 1,767; rows in exact groups: 3,534; exact groups: 1,767.
- Duplicate key rows beyond first: 1,767; rows in key groups: 3,534; key groups: 1,767.
- Conflicting key groups: 0; non-exact duplicate-key rows: 0.
- Missing traffic volume/speed/occupancy/congestion: 4,387/4,382/4,383/26,883.
- Accident count zero/positive: 176,799/1,669.
- No merge was performed or saved.
## 2. Exact duplicates versus duplicate keys

- Exact duplicates are 3,534 rows within duplicate-key groups; 0 duplicate-key rows differ in one or more fields.
- Conflicting groups: 0.
## 3. Missing-value analysis

### Traffic
| column             |   missing_count | missing_pct   |
|:-------------------|----------------:|:--------------|
| road_id            |               0 | 0.00%         |
| road_name          |               0 | 0.00%         |
| latitude           |               0 | 0.00%         |
| longitude          |               0 | 0.00%         |
| weather_station_id |               0 | 0.00%         |
| date               |               0 | 0.00%         |
| time               |               0 | 0.00%         |
| traffic_volume     |            4387 | 2.46%         |
| vehicle_count      |               0 | 0.00%         |
| vehicle_type_dist  |               0 | 0.00%         |
| avg_speed          |            4382 | 2.46%         |
| occupancy          |            4383 | 2.46%         |
| congestion_level   |           26883 | 15.06%        |
| travel_time        |               0 | 0.00%         |
| accident_count     |               0 | 0.00%         |
| signal_timing      |               0 | 0.00%         |
| road_capacity      |               0 | 0.00%         |

### Weather
| column            |   missing_count | missing_pct   |
|:------------------|----------------:|:--------------|
| station_id        |               0 | 0.00%         |
| date              |               0 | 0.00%         |
| time              |               0 | 0.00%         |
| weather_condition |               0 | 0.00%         |
| temperature       |             167 | 1.54%         |
| rainfall          |               0 | 0.00%         |
| visibility        |             111 | 1.02%         |

### Calendar
| column         |   missing_count | missing_pct   |
|:---------------|----------------:|:--------------|
| date           |               0 | 0.00%         |
| public_holiday |               0 | 0.00%         |
| holiday_name   |             145 | 96.03%        |
| event_flag     |               0 | 0.00%         |
| event_name     |             145 | 96.03%        |
| roadwork_flag  |               0 | 0.00%         |

- Traffic rows missing all three volume/speed/occupancy: 4,370; exactly one: 0; exactly two: 21.
- Missing volume with all other traffic fields non-missing: 4,387.
Missing traffic readings by road/month (nonzero only):
| road_id   | month   |   rows |   mv |   ms |   mo |   missing_volume |   missing_speed |   missing_occupancy |
|:----------|:--------|-------:|-----:|-----:|-----:|-----------------:|----------------:|--------------------:|
| NL-001    | 2025-01 |   1458 | 1414 | 1414 | 1415 |               44 |              44 |                  43 |
| NL-001    | 2025-02 |   1321 | 1283 | 1283 | 1283 |               38 |              38 |                  38 |
| NL-001    | 2025-03 |   1457 | 1416 | 1416 | 1416 |               41 |              41 |                  41 |
| NL-001    | 2025-04 |   1424 | 1397 | 1397 | 1397 |               27 |              27 |                  27 |
| NL-001    | 2025-05 |   1452 | 1416 | 1416 | 1416 |               36 |              36 |                  36 |
| NL-002    | 2025-01 |   1478 | 1451 | 1451 | 1451 |               27 |              27 |                  27 |
| NL-002    | 2025-02 |   1324 | 1282 | 1282 | 1282 |               42 |              42 |                  42 |
| NL-002    | 2025-03 |   1469 | 1435 | 1435 | 1435 |               34 |              34 |                  34 |
| NL-002    | 2025-04 |   1415 | 1378 | 1378 | 1378 |               37 |              37 |                  37 |
| NL-002    | 2025-05 |   1469 | 1436 | 1436 | 1436 |               33 |              33 |                  33 |
| NL-003    | 2025-01 |   1469 | 1436 | 1436 | 1436 |               33 |              33 |                  33 |
| NL-003    | 2025-02 |   1311 | 1275 | 1275 | 1275 |               36 |              36 |                  36 |
| NL-003    | 2025-03 |   1460 | 1423 | 1423 | 1423 |               37 |              37 |                  37 |
| NL-003    | 2025-04 |   1410 | 1371 | 1371 | 1371 |               39 |              39 |                  39 |
| NL-003    | 2025-05 |   1480 | 1445 | 1445 | 1445 |               35 |              35 |                  35 |
| NL-004    | 2025-01 |   1467 | 1442 | 1442 | 1442 |               25 |              25 |                  25 |
| NL-004    | 2025-02 |   1344 | 1309 | 1309 | 1309 |               35 |              35 |                  35 |
| NL-004    | 2025-03 |   1475 | 1433 | 1433 | 1433 |               42 |              42 |                  42 |
| NL-004    | 2025-04 |   1415 | 1381 | 1381 | 1381 |               34 |              34 |                  34 |
| NL-004    | 2025-05 |   1468 | 1431 | 1431 | 1432 |               37 |              37 |                  36 |
| NL-005    | 2025-01 |   1461 | 1426 | 1426 | 1426 |               35 |              35 |                  35 |
| NL-005    | 2025-02 |   1332 | 1294 | 1294 | 1294 |               38 |              38 |                  38 |
| NL-005    | 2025-03 |   1473 | 1429 | 1429 | 1429 |               44 |              44 |                  44 |
| NL-005    | 2025-04 |   1405 | 1373 | 1373 | 1373 |               32 |              32 |                  32 |
| NL-005    | 2025-05 |   1465 | 1429 | 1429 | 1429 |               36 |              36 |                  36 |
| NL-006    | 2025-01 |   1461 | 1433 | 1433 | 1433 |               28 |              28 |                  28 |
| NL-006    | 2025-02 |   1333 | 1296 | 1296 | 1296 |               37 |              37 |                  37 |
| NL-006    | 2025-03 |   1456 | 1423 | 1423 | 1423 |               33 |              33 |                  33 |
| NL-006    | 2025-04 |   1418 | 1387 | 1387 | 1387 |               31 |              31 |                  31 |
| NL-006    | 2025-05 |   1456 | 1414 | 1414 | 1414 |               42 |              42 |                  42 |
| NL-007    | 2025-01 |   1459 | 1418 | 1418 | 1418 |               41 |              41 |                  41 |
| NL-007    | 2025-02 |   1340 | 1302 | 1302 | 1302 |               38 |              38 |                  38 |
| NL-007    | 2025-03 |   1464 | 1432 | 1432 | 1432 |               32 |              32 |                  32 |
| NL-007    | 2025-04 |   1419 | 1385 | 1385 | 1385 |               34 |              34 |                  34 |
| NL-007    | 2025-05 |   1461 | 1421 | 1421 | 1422 |               40 |              40 |                  39 |
| NL-008    | 2025-01 |   1463 | 1432 | 1431 | 1431 |               31 |              32 |                  32 |
| NL-008    | 2025-02 |   1323 | 1290 | 1291 | 1290 |               33 |              32 |                  33 |
| NL-008    | 2025-03 |   1466 | 1430 | 1430 | 1430 |               36 |              36 |                  36 |
| NL-008    | 2025-04 |   1423 | 1394 | 1394 | 1394 |               29 |              29 |                  29 |
| NL-008    | 2025-05 |   1463 | 1430 | 1430 | 1431 |               33 |              33 |                  32 |
| NL-009    | 2025-01 |   1453 | 1410 | 1411 | 1410 |               43 |              42 |                  43 |
| NL-009    | 2025-02 |   1325 | 1294 | 1294 | 1294 |               31 |              31 |                  31 |
| NL-009    | 2025-03 |   1457 | 1413 | 1412 | 1412 |               44 |              45 |                  45 |
| NL-009    | 2025-04 |   1431 | 1394 | 1393 | 1393 |               37 |              38 |                  38 |
| NL-009    | 2025-05 |   1455 | 1425 | 1425 | 1425 |               30 |              30 |                  30 |
| NL-010    | 2025-01 |   1469 | 1439 | 1439 | 1439 |               30 |              30 |                  30 |
| NL-010    | 2025-02 |   1329 | 1310 | 1310 | 1310 |               19 |              19 |                  19 |
| NL-010    | 2025-03 |   1470 | 1434 | 1434 | 1434 |               36 |              36 |                  36 |
| NL-010    | 2025-04 |   1415 | 1378 | 1378 | 1378 |               37 |              37 |                  37 |
| NL-010    | 2025-05 |   1464 | 1420 | 1420 | 1421 |               44 |              44 |                  43 |
| NL-011    | 2025-01 |   1455 | 1414 | 1414 | 1414 |               41 |              41 |                  41 |
| NL-011    | 2025-02 |   1324 | 1296 | 1296 | 1296 |               28 |              28 |                  28 |
| NL-011    | 2025-03 |   1463 | 1436 | 1436 | 1436 |               27 |              27 |                  27 |
| NL-011    | 2025-04 |   1421 | 1383 | 1383 | 1383 |               38 |              38 |                  38 |
| NL-011    | 2025-05 |   1478 | 1445 | 1446 | 1445 |               33 |              32 |                  33 |
| NL-012    | 2025-01 |   1466 | 1430 | 1430 | 1430 |               36 |              36 |                  36 |
| NL-012    | 2025-02 |   1322 | 1282 | 1282 | 1282 |               40 |              40 |                  40 |
| NL-012    | 2025-03 |   1478 | 1439 | 1439 | 1439 |               39 |              39 |                  39 |
| NL-012    | 2025-04 |   1410 | 1381 | 1382 | 1381 |               29 |              28 |                  29 |
| NL-012    | 2025-05 |   1462 | 1445 | 1445 | 1445 |               17 |              17 |                  17 |
| NL-013    | 2025-01 |   1468 | 1433 | 1434 | 1433 |               35 |              34 |                  35 |
| NL-013    | 2025-02 |   1331 | 1293 | 1293 | 1293 |               38 |              38 |                  38 |
| NL-013    | 2025-03 |   1471 | 1436 | 1436 | 1436 |               35 |              35 |                  35 |
| NL-013    | 2025-04 |   1425 | 1385 | 1385 | 1385 |               40 |              40 |                  40 |
| NL-013    | 2025-05 |   1472 | 1431 | 1431 | 1431 |               41 |              41 |                  41 |
| NL-014    | 2025-01 |   1473 | 1431 | 1431 | 1431 |               42 |              42 |                  42 |
| NL-014    | 2025-02 |   1331 | 1297 | 1297 | 1297 |               34 |              34 |                  34 |
| NL-014    | 2025-03 |   1469 | 1430 | 1430 | 1430 |               39 |              39 |                  39 |
| NL-014    | 2025-04 |   1416 | 1387 | 1387 | 1387 |               29 |              29 |                  29 |
| NL-014    | 2025-05 |   1461 | 1417 | 1417 | 1417 |               44 |              44 |                  44 |
| NL-015    | 2025-01 |   1468 | 1440 | 1440 | 1440 |               28 |              28 |                  28 |
| NL-015    | 2025-02 |   1329 | 1293 | 1293 | 1293 |               36 |              36 |                  36 |
| NL-015    | 2025-03 |   1461 | 1432 | 1433 | 1432 |               29 |              28 |                  29 |
| NL-015    | 2025-04 |   1403 | 1364 | 1364 | 1364 |               39 |              39 |                  39 |
| NL-015    | 2025-05 |   1459 | 1427 | 1427 | 1427 |               32 |              32 |                  32 |
| NL-016    | 2025-01 |   1469 | 1432 | 1432 | 1432 |               37 |              37 |                  37 |
| NL-016    | 2025-02 |   1312 | 1276 | 1276 | 1276 |               36 |              36 |                  36 |
| NL-016    | 2025-03 |   1468 | 1445 | 1445 | 1446 |               23 |              23 |                  22 |
| NL-016    | 2025-04 |   1420 | 1376 | 1376 | 1376 |               44 |              44 |                  44 |
| NL-016    | 2025-05 |   1459 | 1406 | 1406 | 1406 |               53 |              53 |                  53 |
| NL-017    | 2025-01 |   1470 | 1434 | 1434 | 1434 |               36 |              36 |                  36 |
| NL-017    | 2025-02 |   1329 | 1301 | 1301 | 1301 |               28 |              28 |                  28 |
| NL-017    | 2025-03 |   1464 | 1428 | 1428 | 1428 |               36 |              36 |                  36 |
| NL-017    | 2025-04 |   1424 | 1385 | 1385 | 1385 |               39 |              39 |                  39 |
| NL-017    | 2025-05 |   1471 | 1440 | 1440 | 1440 |               31 |              31 |                  31 |
| NL-018    | 2025-01 |   1479 | 1448 | 1449 | 1448 |               31 |              30 |                  31 |
| NL-018    | 2025-02 |   1333 | 1296 | 1296 | 1296 |               37 |              37 |                  37 |
| NL-018    | 2025-03 |   1460 | 1423 | 1423 | 1424 |               37 |              37 |                  36 |
| NL-018    | 2025-04 |   1421 | 1382 | 1382 | 1382 |               39 |              39 |                  39 |
| NL-018    | 2025-05 |   1466 | 1439 | 1439 | 1439 |               27 |              27 |                  27 |
| NL-019    | 2025-01 |   1455 | 1428 | 1428 | 1428 |               27 |              27 |                  27 |
| NL-019    | 2025-02 |   1314 | 1276 | 1276 | 1276 |               38 |              38 |                  38 |
| NL-019    | 2025-03 |   1455 | 1413 | 1413 | 1413 |               42 |              42 |                  42 |
| NL-019    | 2025-04 |   1412 | 1375 | 1375 | 1375 |               37 |              37 |                  37 |
| NL-019    | 2025-05 |   1468 | 1432 | 1432 | 1432 |               36 |              36 |                  36 |
| NL-020    | 2025-01 |   1471 | 1435 | 1434 | 1434 |               36 |              37 |                  37 |
| NL-020    | 2025-02 |   1327 | 1294 | 1294 | 1294 |               33 |              33 |                  33 |
| NL-020    | 2025-03 |   1476 | 1439 | 1439 | 1439 |               37 |              37 |                  37 |
| NL-020    | 2025-04 |   1414 | 1365 | 1365 | 1365 |               49 |              49 |                  49 |
| NL-020    | 2025-05 |   1469 | 1423 | 1423 | 1423 |               46 |              46 |                  46 |
| NL-021    | 2025-01 |   1448 | 1412 | 1412 | 1412 |               36 |              36 |                  36 |
| NL-021    | 2025-02 |   1327 | 1292 | 1292 | 1292 |               35 |              35 |                  35 |
| NL-021    | 2025-03 |   1472 | 1441 | 1441 | 1441 |               31 |              31 |                  31 |
| NL-021    | 2025-04 |   1422 | 1383 | 1383 | 1383 |               39 |              39 |                  39 |
| NL-021    | 2025-05 |   1464 | 1426 | 1426 | 1426 |               38 |              38 |                  38 |
| NL-022    | 2025-01 |   1470 | 1432 | 1433 | 1432 |               38 |              37 |                  38 |
| NL-022    | 2025-02 |   1325 | 1307 | 1307 | 1307 |               18 |              18 |                  18 |
| NL-022    | 2025-03 |   1477 | 1441 | 1441 | 1441 |               36 |              36 |                  36 |
| NL-022    | 2025-04 |   1416 | 1378 | 1378 | 1378 |               38 |              38 |                  38 |
| NL-022    | 2025-05 |   1455 | 1414 | 1414 | 1414 |               41 |              41 |                  41 |
| NL-023    | 2025-01 |   1463 | 1428 | 1428 | 1428 |               35 |              35 |                  35 |
| NL-023    | 2025-02 |   1322 | 1289 | 1289 | 1289 |               33 |              33 |                  33 |
| NL-023    | 2025-03 |   1462 | 1430 | 1430 | 1431 |               32 |              32 |                  31 |
| NL-023    | 2025-04 |   1413 | 1384 | 1384 | 1384 |               29 |              29 |                  29 |
| NL-023    | 2025-05 |   1469 | 1432 | 1432 | 1432 |               37 |              37 |                  37 |
| NL-024    | 2025-01 |   1465 | 1435 | 1435 | 1435 |               30 |              30 |                  30 |
| NL-024    | 2025-02 |   1316 | 1279 | 1279 | 1279 |               37 |              37 |                  37 |
| NL-024    | 2025-03 |   1455 | 1423 | 1423 | 1423 |               32 |              32 |                  32 |
| NL-024    | 2025-04 |   1417 | 1376 | 1376 | 1376 |               41 |              41 |                  41 |
| NL-024    | 2025-05 |   1463 | 1434 | 1434 | 1434 |               29 |              29 |                  29 |
| NL-025    | 2025-01 |   1462 | 1437 | 1437 | 1437 |               25 |              25 |                  25 |
| NL-025    | 2025-02 |   1324 | 1286 | 1286 | 1286 |               38 |              38 |                  38 |
| NL-025    | 2025-03 |   1469 | 1441 | 1441 | 1441 |               28 |              28 |                  28 |
| NL-025    | 2025-04 |   1420 | 1399 | 1399 | 1399 |               21 |              21 |                  21 |
| NL-025    | 2025-05 |   1475 | 1435 | 1436 | 1435 |               40 |              39 |                  40 |
Missing windows by road using each road?s observed first-to-last date span:
| road_id   |   observed_rows |   distinct_windows |   expected_in_observed_span |   missing_windows |   longest_gap_windows |
|:----------|----------------:|-------------------:|----------------------------:|------------------:|----------------------:|
| NL-001    |            7112 |               7059 |                        7248 |               189 |                     2 |
| NL-002    |            7155 |               7081 |                        7248 |               167 |                     2 |
| NL-003    |            7130 |               7064 |                        7248 |               184 |                     2 |
| NL-004    |            7169 |               7091 |                        7248 |               157 |                     2 |
| NL-005    |            7136 |               7064 |                        7248 |               184 |                     2 |
| NL-006    |            7124 |               7053 |                        7248 |               195 |                     2 |
| NL-007    |            7143 |               7072 |                        7248 |               176 |                     2 |
| NL-008    |            7138 |               7066 |                        7248 |               182 |                     2 |
| NL-009    |            7121 |               7069 |                        7248 |               179 |                     2 |
| NL-010    |            7147 |               7081 |                        7248 |               167 |                     2 |
| NL-011    |            7141 |               7076 |                        7248 |               172 |                     2 |
| NL-012    |            7138 |               7066 |                        7248 |               182 |                     2 |
| NL-013    |            7167 |               7094 |                        7248 |               154 |                     2 |
| NL-014    |            7150 |               7075 |                        7248 |               173 |                     2 |
| NL-015    |            7120 |               7045 |                        7248 |               203 |                     2 |
| NL-016    |            7128 |               7053 |                        7248 |               195 |                     2 |
| NL-017    |            7158 |               7080 |                        7248 |               168 |                     2 |
| NL-018    |            7159 |               7066 |                        7248 |               182 |                     2 |
| NL-019    |            7104 |               7045 |                        7248 |               203 |                     2 |
| NL-020    |            7157 |               7093 |                        7248 |               155 |                     2 |
| NL-021    |            7133 |               7068 |                        7248 |               180 |                     2 |
| NL-022    |            7143 |               7052 |                        7248 |               196 |                     2 |
| NL-023    |            7129 |               7065 |                        7248 |               183 |                     2 |
| NL-024    |            7116 |               7058 |                        7248 |               190 |                     2 |
| NL-025    |            7150 |               7065 |                        7248 |               183 |                     2 |
- Observed-span missing windows total: 4,499; longest gap: 2 half-hour windows (1.0 hours).
- Theoretical full 151-day grid: 181,200; absent distinct keys: 4,499. This is an upper bound, not an assumption of full activity.
## 4. Invalid and suspicious values

- Negative traffic_volume (definite violation): 241 (0.14%).
| road_id   | date       | time   |   traffic_volume |   vehicle_count |   road_capacity |
|:----------|:-----------|:-------|-----------------:|----------------:|----------------:|
| NL-009    | 2025-02-05 | 21:30  |               -6 |             247 |            1518 |
| NL-014    | 2025-01-25 | 23:30  |              -33 |             119 |            2327 |
| NL-007    | 2025-04-08 | 20:30  |              -38 |             258 |            1928 |
| NL-012    | 2025-01-23 | 16:00  |               -6 |             528 |            1259 |
| NL-001    | 2025-04-13 | 00:00  |              -53 |             103 |            1888 |
- avg_speed > 200 km/h (definite violation): 237 (0.13%).
| road_id   | date       | time   |   avg_speed |   traffic_volume |
|:----------|:-----------|:-------|------------:|-----------------:|
| NL-002    | 2025-05-23 | 00:30  |       317   |              229 |
| NL-001    | 2025-02-23 | 01:30  |       243.4 |              144 |
| NL-015    | 2025-02-23 | 23:00  |       313.2 |              159 |
| NL-013    | 2025-02-10 | 12:00  |       307   |              429 |
| NL-015    | 2025-04-29 | 03:30  |       283.6 |              252 |
- occupancy > 100% (definite violation): 234 (0.13%).
| road_id   | date       | time   |   occupancy |   traffic_volume |
|:----------|:-----------|:-------|------------:|-----------------:|
| NL-011    | 2025-05-30 | 02:00  |       131.7 |              141 |
| NL-004    | 2025-05-27 | 21:30  |       125.2 |              nan |
| NL-023    | 2025-05-28 | 08:30  |       141.7 |             1111 |
| NL-013    | 2025-03-26 | 21:00  |       173.7 |              237 |
| NL-011    | 2025-03-15 | 20:30  |       141.3 |              142 |
- Zero/negative road_capacity: 0.
- Unparseable traffic date/time/timestamp: 0/0/0.
- Unparseable weather date/time/timestamp: 0/0/0.
- Unparseable calendar dates: 0.
- Traffic labels: ['Free-flow', 'Heavy', 'Moderate', 'Severe']; outside documented set: [].
- Weather labels (13): ['CLEAR', 'Clear', 'Cloudy', 'FOG', 'Fog', 'Overcast', 'RAIN', 'Rain', 'clear', 'cloudy', 'foggy', 'rain ', 'rainy'].
- Invalid vehicle_type_dist JSON: 0.
## 5. Weather and calendar join audit (no merge saved)

- Traffic station coverage: 3/3; unmatched traffic stations: [].
- Weather station/hour duplicate rows: 0; groups: 0.
- Weather station/hour keys without traffic: 0; distinct traffic station/hour keys without weather: 0; traffic rows without weather: 0. Weather timestamps were compared at the traffic timestamp floored to the hour, so both 30-minute traffic windows can match one hourly observation.
- Traffic dates absent from calendar: 0; dates: [].
- Duplicate calendar dates: 0 beyond first; groups: 0.
## 6. Congestion-label audit

- Missing labels safely derivable: 26,196; not safely derivable: 687.
- Existing non-missing labels with valid inputs: 147,644; formula disagreements: 0.
## 7. Accident-risk target feasibility

- Missing: 0; negative: 0; non-integer: 0.
- Zero: 176,799 (99.06%); positive: 1,669 (0.94%).
|   accident_count |   rows |
|-----------------:|-------:|
|                0 | 176799 |
|                1 |   1654 |
|                2 |     15 |
- Positive examples exist but are rare; use time-aware evaluation and report class balance. The source does not uniquely define whether next-window risk is t+1 or an aggregated 1-4 window horizon; decide before target construction and prevent leakage.
## 8. Recommended handling policy

- Keep raw CSVs immutable and write outputs only under configured interim/processed paths.
- Log exact duplicates; do not silently choose among conflicting keys.
- Profile missingness and contiguous gaps before any drop/flag/fill rule.
- Flag/reject definite violations and invalid JSON/date/time with reason codes; distinguish unusual values.
- Parse source-specific dates, normalize weather labels, validate station/hour uniqueness, then align hourly weather to both traffic half-hours.
- Validate calendar uniqueness/coverage before joining.
- Derive only approved blank congestion labels with valid inputs and provenance.
- Specify accident horizon before features; treat it as a rare-event target with time-aware metrics.
## 9. Decisions requiring care or clarification

- Whether all 25 segments were active every day or full-day absences are planned inactivity.
- How to resolve non-identical duplicate-key records.
- Whether accident risk is any positive count at t+1 or aggregated 30?120 minute risk.
- Whether Overcast/foggy/rainy variants map to documented classes or remain unknown.
- What downstream analyses permit when volume, speed, or occupancy is missing.
## 10. Proposed Phase 2 cleaning order

1. Validate required columns and raw hashes.
2. Parse source-specific dates/times and create non-destructive timestamps.
3. Log duplicates, conflicts, invalid JSON/ranges, and missingness.
4. Apply approved duplicate/conflict and validity policies with audit flags.
5. Normalize weather labels and validate station/hour keys.
6. Validate calendar keys.
7. Derive blank congestion labels with provenance.
8. Align weather and join calendar after key validation.
9. Persist outputs and rerun reconciliation checks.
## 11. Cleaned-output verification checklist

- [ ] Raw hashes unchanged.
- [ ] Required columns/dtypes validated.
- [ ] Duplicate key result reconciles with approved policy.
- [ ] Invalid records counted and reason-coded.
- [ ] Missingness by dataset/road/month reconciles.
- [ ] Timestamp continuity and window coverage reported.
- [ ] Weather station/hour keys unique and join coverage quantified.
- [ ] Calendar dates unique and coverage quantified.
- [ ] Congestion boundaries and derivations reproducible.
- [ ] Accident target has no future leakage.
- [ ] Output row counts and provenance documented.
