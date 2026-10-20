# LSTM vs Random Forest validation comparison

Common validation examples: 18,704

| Model | RMSE | MAE | MAPE | R² |
|---|---:|---:|---:|---:|
| Lstm | 56.234498 | 38.764358 | 9.701769% | 0.958998 |
| Random Forest | 62.582575 | 42.496093 | 10.286797% | 0.949219 |

Rows were aligned by exact road, timestamp, and target value. The initial validation timestamps unavailable to sequence windows were excluded from the common set.
