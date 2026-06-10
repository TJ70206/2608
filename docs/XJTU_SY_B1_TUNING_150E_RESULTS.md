# XJTU-SY Paper9 B1 150-Epoch Tuning Results

## Protocol
- **Task**: Paper9 B1, `C1 -> C2`.
- **Source labeled**: `Bearing1_1`.
- **Target alignment**: `Bearing2_1`.
- **Target test**: `Bearing2_3`.
- **Input**: `data/processed/xjtu_sy_hi.npz`, 14-dimensional low-frequency HI features.
- **Model**: P-SA-MCD-TCN + stage-aware LMMD.
- **Tuning request**: 5 parameter variants, each 150 epochs.

## Paper9 B1 references
| Method | MAE | RMSE | Score |
|---|---:|---:|---:|
| HIWSAN | 0.1382 | 0.1725 | 0.3887 |
| GSAN | 0.1595 | 0.2012 | 0.3498 |
| DSAN-WM | 0.1763 | 0.2211 | 0.3290 |

## Our runs
| Label | Epochs | MAE | RMSE | Score | Clipped MAE | Clipped RMSE | Clipped Score | Coverage | Width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_w32_50e | 50 | 0.1824 | 0.2421 | 0.2978 | 0.1814 | 0.2418 | 0.2987 | 0.865 | 0.7899 |
| w16_lmmd001_150e | 150 | 0.2144 | 0.2759 | 0.2288 | 0.2136 | 0.2755 | 0.2294 | 0.815 | 0.7961 |
| w16_lmmd01_150e | 150 | 0.2836 | 0.3432 | 0.1638 | 0.2476 | 0.3195 | 0.1690 | 0.723 | 0.7905 |
| w64_lmmd005_150e | 150 | 0.2086 | 0.2695 | 0.2234 | 0.2068 | 0.2686 | 0.2246 | 0.873 | 1.0053 |
| w32_lmmd001_lr5e4_150e | 150 | 0.2071 | 0.2618 | 0.2074 | 0.2050 | 0.2608 | 0.2092 | 0.849 | 0.7893 |
| h96_w32_lmmd005_150e | 150 | 0.2325 | 0.2920 | 0.2221 | 0.2120 | 0.2821 | 0.2272 | 0.802 | 0.7535 |

## Selection
- **Best among these runs**: `baseline_w32_50e` with MAE `0.1824`, RMSE `0.2421`, Score `0.2978`.
- **Important**: the best is still the previous 50-epoch baseline, not any of the new 150-epoch variants.
- **Interpretation**: extending training to 150 epochs degraded B1 generalization under the current simplified LMMD setup.
- **Report status**: treat this as a pipeline/tuning sanity check, not as a final strong comparison against HIWSAN.
