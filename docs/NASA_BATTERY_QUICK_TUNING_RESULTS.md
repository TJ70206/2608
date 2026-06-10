# NASA Battery Quick Tuning Results

## Protocol
- **Dataset**: NASA Battery B0005/B0006/B0007/B0018 processed cycle-level CSV.
- **Split**: train B0007+B0018, validation B0006, test B0005.
- **Main metric for this source-domain task**: dense per-cycle RMSE/MAE/Score.
- **Caution**: this is a single battery split; use leave-one-battery-out for formal robustness.

## Results
| Model | Role | RMSE | MAE | Score | Last RMSE | Last Score | Coverage | Width | Params | Latency ms/sample |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P-TCN baseline | baseline | 19.5367 | 13.3382 | 735.55 | 3.8047 | 0.4630 | 0.5912 | 11.8958 | 128129 | 0.3812 |
| P-SA-MCD base | rejected | 21.8385 | 13.8498 | 1015.96 | 0.8464 | 0.0883 | 0.6058 | 18.3649 | 58129 | 0.6027 |
| P-SA-MCD h64 | previous_best | 5.1775 | 2.9291 | 50.76 | 0.9571 | 0.0764 | 1.0000 | 53.2176 | 229913 | 0.7120 |
| P-SA-MCD w16 | rejected | 23.3876 | 15.3728 | 1504.75 | 1.0408 | 0.1097 | 0.5882 | 19.4122 | 58129 | 0.2262 |
| P-SA-MCD h64_w48 | new_best | 2.6299 | 1.9045 | 20.67 | 0.9121 | 0.0727 | 1.0000 | 24.2207 | 229913 | 0.2763 |
| P-SA-MCD h96 | rejected | 8.9212 | 5.0583 | 116.75 | 0.2594 | 0.0202 | 0.9343 | 40.8402 | 515361 | 0.7890 |
| P-SA-MCD h64_reg | rejected | 22.1154 | 14.1449 | 1064.01 | 1.3545 | 0.1451 | 0.5985 | 17.8840 | 229913 | 0.6684 |

## Decision
- **Adopt**: `configs/nasa_battery_processed_psa_mcd_tcn_h64_w48.yaml`.
- **Reason**: it improves over previous h64 on RMSE, MAE, Score, last-window RMSE/Score, conformal width, and latency while keeping coverage at 1.0.
