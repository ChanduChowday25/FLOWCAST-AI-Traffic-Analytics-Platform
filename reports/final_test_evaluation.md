# Final GRU test evaluation

GRU was evaluated as the primary candidate because it led the common validation comparison on RMSE, MAE, and R2. LSTM and Random Forest remain comparison baselines.

Valid test predictions: 18,640
Excluded test examples: 7,863

| Metric | Value |
|---|---:|
| RMSE | 61.429334 |
| MAE | 40.845811 |
| MAPE | 10.158373% |
| R² | 0.954946 |

Test sequences used only rows from the test partition. Preprocessing was fitted on training rows only; no test data was used for training or tuning.
