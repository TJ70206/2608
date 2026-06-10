# XJTU-SY Paper9 B1 Quick Experiment

## Protocol
- **Task**: B1, C1 -> C2.
- **Source labeled unit**: `Bearing1_1`.
- **Target alignment unit**: `Bearing2_1`.
- **Target test unit**: `Bearing2_3`.
- **Input**: processed low-frequency HI features from `data/processed/xjtu_sy_hi.npz`, not raw high-frequency vibration.
- **Model**: P-SA-MCD-TCN with stage-aware LMMD, 50 epochs.

## Result
| Method | MAE | RMSE | Paper9 Score | Note |
|---|---:|---:|---:|---|
| Paper9 HIWSAN | 0.1382 | 0.1725 | 0.3887 | Table 14, mean over 10 repeats |
| Ours P-SA-MCD-TCN + LMMD | 0.1824 | 0.2421 | 0.2978 | raw predictions |
| Ours P-SA-MCD-TCN + LMMD | 0.1814 | 0.2418 | 0.2987 | predictions clipped to [0,1], diagnostic only |

## Interpretation
- **RMSE**: our quick run is worse than Paper9 HIWSAN on B1.
- **MAE**: our quick run is worse than Paper9 HIWSAN on B1.
- This is a single quick run, while Paper9 reports mean  std over 10 repeats; treat it as a first sanity check rather than a final comparison.
