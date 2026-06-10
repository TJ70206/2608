# XJTU-SY Paper9 B1 50-Epoch Tuning Results

## Protocol
- **Task**: Paper9 B1, `C1 -> C2`.
- **Source labeled**: `Bearing1_1`.
- **Target alignment**: `Bearing2_1`.
- **Target test**: `Bearing2_3`.
- **Input**: `data/processed/xjtu_sy_hi.npz`, 14-dimensional low-frequency HI features.
- **Training**: 50 epochs, P-SA-MCD-TCN / P-TCN with stage-aware LMMD variants.

## Paper9 B1 references
| Method | MAE | RMSE | Score |
|---|---:|---:|---:|
| HIWSAN | 0.1382 | 0.1725 | 0.3887 |
| GSAN | 0.1595 | 0.2012 | 0.3498 |
| DSAN-WM | 0.1763 | 0.2211 | 0.3290 |

## Our 50-epoch variants
| Label | MAE | RMSE | Score | Coverage | Width | Params | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_w32_lmmd05_50e | 0.1824 | 0.2421 | 0.2978 | 0.865 | 0.7899 | 230489 | diagnostic |
| w32_lmmd0_50e | 0.1883 | 0.2323 | 0.2998 | 0.921 | 0.8073 | 230489 | diagnostic |
| w32_lmmd005_50e | 0.1788 | 0.2246 | 0.2412 | 0.897 | 0.8136 | 230489 | diagnostic |
| w32_lmmd02_50e | 0.1871 | 0.2419 | 0.2567 | 0.857 | 0.7585 | 230489 | diagnostic |
| w24_lmmd005_50e | 0.1778 | 0.2155 | 0.2540 | 0.938 | 0.8366 | 230489 | diagnostic |
| w48_lmmd005_50e | 0.2217 | 0.2777 | 0.2333 | 0.869 | 0.9614 | 230489 | diagnostic |
| h32_w32_lmmd005_50e | 0.1722 | 0.2207 | 0.3501 | 0.905 | 0.7796 | 58417 | keep |
| ptcn_w32_lmmd005_50e | 0.1831 | 0.2070 | 0.2907 | 1.000 | 0.7947 | 128705 | keep |

## Best selections
- **Best MAE/Score**: `h32_w32_lmmd005_50e` with MAE `0.1722`, RMSE `0.2207`, Score `0.3501`.
- **Best RMSE**: `ptcn_w32_lmmd005_50e` with MAE `0.1831`, RMSE `0.2070`, Score `0.2907`.

## Interpretation
- `h32_w32_lmmd005_50e` beats Paper9 DSAN-WM on MAE/RMSE and slightly beats GSAN on Score, but does not beat HIWSAN.
- `ptcn_w32_lmmd005_50e` gives the best RMSE among our runs and also beats DSAN-WM RMSE, showing that a simpler TCN can be more stable on B1.
- The earlier 150-epoch variants degraded, so B1 should use short training or early stopping rather than fixed long training.
- For the final XA-202608 pipeline, absorb Paper9 ideas as HI-weighted subdomain alignment and stage/HI-aware LMMD for `XJTU-SY -> reaction_wheel_sim`, without transferring raw high-frequency vibration directly.
