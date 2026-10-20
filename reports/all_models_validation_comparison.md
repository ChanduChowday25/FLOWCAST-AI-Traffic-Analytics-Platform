# Random Forest, LSTM, and GRU validation comparison

Common validation examples: 18,704

| Model | RMSE | MAE | MAPE | R² |
|---|---:|---:|---:|---:|
| Random Forest | 62.582575 | 42.496093 | 10.286797% | 0.949219 |
| Lstm | 56.234498 | 38.764358 | 9.701769% | 0.958998 |
| Gru | 55.921596 | 38.709622 | 9.820962% | 0.959453 |

Rows were aligned by exact road ID, target timestamp, and actual target value. Validation-only rows without enough sequence history were excluded from the common set; the test partition was not used.
