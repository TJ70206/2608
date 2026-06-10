# C-MAPSS Stage 1 Final Results
## Protocol
- **Main metric**: capped last-window RMSE / NASA Score, one prediction per test engine, test RUL capped at 125.
- **Diagnostic metric**: last-5 average RMSE / Score.
- **Leakage rule**: normalization is fitted on training units only; validation-only calibration is fitted on validation last-window predictions/labels only.

## Final / Robustness Table
| Subset | Role | RMSE | Score | Last5 RMSE | Last5 Score | Note |
|---|---|---:|---:|---:|---:|---|
| FD001 | Main | 12.3669 | 202.71 | 12.2946 | 215.37 | Strong single-condition baseline. |
| FD002 | RMSE-main | 16.1071 | 1049.27 | 15.7403 | 1025.99 | Validation-only piecewise calibration; best RMSE candidate. |
| FD002 | Score-main | 16.4611 | 994.53 | 16.0290 | 959.08 | Validation-only piecewise calibration with shrink=0.75; best Score candidate. |
| FD002 | Score-robustness-seed2026 | 16.6976 | 1050.67 | 16.1329 | 1036.65 | Robustness check for FD002 score-oriented candidate. |
| FD003 | Main | 13.2372 | 259.78 | 13.3268 | 277.32 | Tail15w3 + late prediction penalty + clipping. |
| FD004 | Main-isotonic | 16.5149 | 1069.38 | 15.9108 | 963.14 | h64 condition norm + validation-only isotonic calibration; seed42 main. |
| FD004 | Robustness-isotonic-seed2026 | 16.1339 | 1260.67 | 15.4904 | 1132.01 | Robustness check for FD004 isotonic candidate. |
| FD004 | Rejected-piecewise075-seed2026 | 16.2527 | 1316.53 | 15.6798 | 1108.42 | Final conservative piecewise check; not adopted because Score worsened. |

## Adopted Main Results
| Subset | Main config | Main posture |
|---|---|---|
| FD001 | `cmapss_fd001_psa_mcd_paper4_fullseq_tail5_capped_99e.yaml` | Adopt. |
| FD002 | `cmapss_fd002_psa_mcd_paper4_fullseq_tail5_condnorm_piecewise_capped_99e.yaml` / `...piecewise_score...yaml` | Report RMSE-main and Score-main as two fair validation-selected variants. |
| FD003 | `cmapss_fd003_psa_mcd_paper4_fullseq_tail15w3_latepred12_clip_capped_99e.yaml` | Adopt; no further tuning. |
| FD004 | `cmapss_fd004_psa_mcd_paper4_fullseq_h64_tail5_condnorm_isotonic_capped_99e.yaml` | Adopt isotonic seed42 as main; mention seed2026 Score fluctuation. |

## Final Optimization Decision
- **FD004 conservative piecewise check**: rejected. It obtained RMSE `16.2527` and Score `1316.53`, so it did not improve robustness over isotonic.
- **C-MAPSS stage**: close stage 1; do not continue broad C-MAPSS tuning.
- **Next stage**: implement/finalize `reaction_wheel_sim`, `satellite_battery_sim`, then run SA-FS-LMMD transfer experiments.
