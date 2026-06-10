# C-MAPSS FD001 Paper4 Full-Sequence Tail-Weighted Experiment Review

## Metric Protocol

C-MAPSS results must be reported with an explicit evaluation protocol.

| Protocol | Meaning | Use |
|---|---|---|
| All-window / dense RMSE | RMSE over all predicted time points/windows | Useful for full RUL curve quality and paper4-style dense-supervision analysis |
| Capped last-window RMSE | One prediction per test engine, with test RUL capped at 125 | Best paper-style comparison for many C-MAPSS papers using piecewise capped labels |
| Raw last-window RMSE | One prediction per test engine, using original test RUL labels | Strict official-style diagnostic result; harder when training labels are capped |
| Last-window NASA Score | One prediction per test engine, asymmetric penalty | Important C-MAPSS prognostics metric |

The current report should not mix these protocols. The main paper-style FD001 result should be capped last-window RMSE/Score, with all-window RMSE as an auxiliary curve-fitting metric and raw last-window RMSE as a strict robustness check.

## Compared Runs

| Run | Preprocessing / objective | Checkpoint metric | Test all-window RMSE | Test capped last-window RMSE | Test capped last-window Score |
|---|---|---|---:|---:|---:|
| `cmapss_fd001_psa_mcd_aligned_pseudo_match` | Windowed P-SA-MCD-TCN, minmax, cap=125 | weighted | 12.9744 | 12.7011 | 293.23 |
| `cmapss_fd001_psa_mcd_paper4_fullseq_capped_99e` | Full sequence, zscore, random end trim, dense supervision | weighted | 10.7599 | 17.0678 | 371.59 |
| `cmapss_fd001_psa_mcd_paper4_fullseq_tail5_capped_99e` | Full sequence + dense supervision + tail-weighted final 5 points | last-window RMSE | 9.3132 | 12.3669 | 202.71 |

## Strict Raw Re-evaluation

The `tail5` model was also re-evaluated with raw test RUL labels (`test_max_rul: null`).

| Run | Raw last-window RMSE | Raw last-window MAE | Raw last-window Score | Raw last-5 RMSE | Raw last-5 Score |
|---|---:|---:|---:|---:|---:|
| Old strict raw best | 13.8777 | N/A | 321.17 | 14.0531 | 346.29 |
| Paper4 tail5 raw re-eval | 14.1133 | 10.3701 | 260.87 | 13.9410 | 263.22 |

Interpretation: `tail5` does not beat the old strict raw RMSE best, but it improves raw NASA Score and last-5 RMSE/Score. This is consistent with stronger full-trajectory and late-stage behavior, while still being affected by raw labels above the 125 cap.

## Main Findings

- **Paper4 idea is useful**: full-sequence dense supervision substantially improves all-window RMSE from 12.9744 to 9.3132.
- **Dense supervision alone is not enough**: pure paper4 full-sequence training worsened capped last-window RMSE to 17.0678.
- **Tail weighting is the key adaptation**: emphasizing the final 5 valid sequence points converted the all-window gain into a capped last-window gain.
- **New capped best**: `paper4_fullseq_tail5_capped_99e` is now the best capped FD001 result: last-window RMSE 12.3669 and Score 202.71.
- **Raw RMSE best unchanged**: strict raw RMSE should still use the old raw best 13.8777 unless future experiments beat it.
- **Raw Score best candidate**: `tail5` has better raw Score than the old raw best, so it can be reported as a Score-oriented strict result.

## Plot Outputs

Each training or re-evaluation now generates two diagnostic figures:

| Figure | File pattern | Meaning |
|---|---|---|
| Last-window summary | `*_last_window_summary.png` | Last-window true/predicted curve, residuals, scatter, residual histogram |
| Single-unit interval trajectory | `*_unit_<id>_trajectory.png` | True RUL, predicted RUL, multi-level validation-calibrated intervals |

For the current `tail5` experiment:

- `outputs/cmapss_fd001_psa_mcd_paper4_fullseq_tail5_capped_99e/test_last_window_summary.png`
- `outputs/cmapss_fd001_psa_mcd_paper4_fullseq_tail5_capped_99e/test_unit_76_trajectory.png`
- `outputs/cmapss_fd001_psa_mcd_paper4_fullseq_tail5_capped_99e/test_raw_reeval_last_window_summary.png`
- `outputs/cmapss_fd001_psa_mcd_paper4_fullseq_tail5_capped_99e/test_raw_reeval_unit_76_trajectory.png`

The single-unit trajectory plot defaults to test unit 76 when available because it is a representative long trajectory. In the capped test evaluation, unit 76 has 205 observed points and a final capped RUL of 10, so the capped true RUL starts decreasing around index 90. This is close to the common MATLAB-style example where a full run-to-failure engine with length about 205 starts decreasing around index 80 under `RUL_max=125`.

If a selected test engine appears to decrease much later, it is usually not a plotting error. C-MAPSS test trajectories are partial trajectories, and each unit has a different final RUL from `RUL_FD001.txt`. For capped labels, the decrease onset is approximately:

```text
onset_index = observed_length - (125 - final_RUL)
```

For example, unit 45 has 152 observed points and final RUL 114, so the capped true RUL does not decrease until about index 141. Unit 83 has final capped RUL 125, so its capped true RUL stays flat throughout the observed test trajectory.

Representative examples from the current capped test predictions:

| Unit | Observed length | Final capped RUL | Capped RUL decrease onset | Interpretation |
|---:|---:|---:|---:|---|
| 21 | 148 | 57 | 80 | Similar onset to common MATLAB-style screenshots, but shorter observed trajectory |
| 76 | 205 | 10 | 90 | Best representative long trajectory; close to full run-to-failure style plots |
| 45 | 152 | 114 | 141 | Late decrease is expected because the test trajectory ends while the engine is still healthy |
| 83 | 73 | 125 | no decrease | Flat true capped RUL is expected because the observed segment never leaves the capped plateau |

Therefore, a curve dropping near index 140 is not necessarily wrong. It depends on which test unit is plotted. For visual reports, use unit 76 or unit 21 rather than automatically selecting the largest-error unit.

## Adoption Decision

| Item | Decision | Reason |
|---|---|---|
| Full-sequence dense supervision from paper4 | Adopt for capped paper-style branch | Produces paper-level all-window RMSE and improves curve fitting |
| Tail-weighted final points | Adopt | Fixes the mismatch between dense all-window training and official last-window evaluation |
| Z-score normalization in this branch | Adopt | Part of paper4-aligned preprocessing and works with tail weighting |
| Pure paper4 full-sequence without tail weighting | Do not adopt as final | Good all-window RMSE but poor last-window RMSE |
| Use all-window RMSE as the only headline metric | Do not adopt | Can be misleading for C-MAPSS official one-RUL-per-engine comparison |
| Report capped result only | Do not adopt | Also report raw last-window as a stricter supplement |

## Recommended Reporting

Recommended table entries:

| Result type | Config / output | Metric |
|---|---|---|
| Best paper-style capped result | `cmapss_fd001_psa_mcd_paper4_fullseq_tail5_capped_99e` | capped last-window RMSE 12.3669 / Score 202.71 |
| Paper4-style dense curve-fitting result | same | all-window RMSE 9.3132 |
| Best strict raw RMSE result | `cmapss_fd001_psa_mcd_paper_strict_80e_raw` | raw last-window RMSE 13.8777 |
| Score-oriented raw result | `cmapss_fd001_psa_mcd_paper4_fullseq_tail5_capped_99e` raw re-eval | raw last-window Score 260.87 |

## Follow-up Optimization Review

Several low-risk follow-up experiments were run after the `tail5` result.

| Experiment | Main change | Capped all-window RMSE | Capped last-window RMSE | Capped last-window Score | Decision |
|---|---|---:|---:|---:|---|
| `tail5_val5` | enable 5 validation trims | 9.3132 | 12.3669 | 202.71 | Same checkpoint/test result as seed42 best; useful stability check |
| `tail3_val5` | final 3 points, weight 5 | 11.0160 | 13.2724 | 239.55 | Do not adopt |
| `tail10w3_val5` | final 3 points, weight 10 | 11.5784 | 14.7102 | 273.91 | Do not adopt |

Conclusion: the current `sequence_tail_weight=5.0` and `sequence_tail_k=5` is the best tested tail-weight setting. Shorter or stronger tail weighting over-emphasizes local sequence endings and hurts both dense RMSE and last-window RMSE.

The data loader now supports `split_seed` and `val_trim_seed`, so train/validation unit split and validation trimming can be held fixed while varying model initialization. Fixed-split model-seed experiments were run with `split_seed=42` and `val_trim_seed=1042`.

| Run | Capped all-window RMSE | Capped last-window RMSE | Capped last-window Score | Capped last-5 RMSE | Decision |
|---|---:|---:|---:|---:|---|
| seed42 best | 9.3132 | 12.3669 | 202.71 | 12.2946 | Keep as FD001 best |
| fixedsplit seed40 | 9.5620 | 13.4729 | 226.88 | 12.8546 | Do not adopt |
| fixedsplit seed7 | 9.2699 | 13.3141 | 245.38 | 13.0883 | Better all-window only; do not adopt for last-window |
| fixedsplit seed123 | 9.4744 | 13.4313 | 236.51 | 12.9907 | Do not adopt |
| fixedsplit seed2026 | 9.9783 | 13.7798 | 244.88 | 13.8989 | Do not adopt |

Conclusion: FD001 single-model optimization has likely reached a practical plateau under the current budget. Seed42 remains the best capped last-window result. Some fixed-split seeds improve all-window RMSE slightly, but all worsen the official/paper-style last-window metrics. Further blind FD001 single-knob tuning is not recommended; the next step should be testing FD002/FD003/FD004 generalization or reporting seed42 as the main FD001 result with explicit seed/split documentation.

## FD002/FD004 Multi-Operating-Condition Review

The FD001 `P-SA-MCD-TCN-P4Tail` configuration was transferred to FD002, FD003, and FD004. For FD002 and FD004, `setting_1`, `setting_2`, and `setting_3` were retained as operating-condition inputs; FD003 kept the FD001 feature selection because it has a single operating condition.

| Run | Normalize | Capped all-window RMSE | Capped last-window RMSE | Capped last-window Score | Capped last-5 RMSE | Decision |
|---|---|---:|---:|---:|---:|---|
| FD002 P4Tail | global z-score | 11.7547 | 21.8800 | 1731.13 | 21.3870 | Baseline only |
| FD002 P4Tail condition-norm | condition z-score | 10.3285 | 19.5389 | 1350.59 | 18.9464 | Adopt for FD002 |
| FD003 P4Tail | global z-score | 9.2813 | 13.8634 | 416.79 | 14.3606 | Adopt for FD003 |
| FD004 P4Tail | global z-score | 13.5147 | 27.7328 | 3138.82 | 27.5925 | Baseline only |
| FD004 P4Tail condition-norm | condition z-score | 12.9443 | 21.1325 | 1610.26 | 20.8045 | Adopt for FD004 |

The condition-norm implementation is leakage-safe: operating-condition centers and normalization statistics are fitted only on training units; validation/test rows are assigned to the nearest training operating-condition center and normalized with the corresponding training statistics.

This is consistent with common FD002/FD004 handling in C-MAPSS codebases. The reviewed Transformer repository uses separate operating-condition scalers for FD002/FD004, and the reviewed CNN repository clusters `setting1/setting2/setting3` with KMeans and scales by `operating_condition`. The original Paper4 pipeline uses global z-score standardization, but its own discussion also notes that FD002/FD004 are harder because of multiple operating conditions. Therefore, condition-wise normalization is a justified extension of the Paper4 preprocessing chain for multi-operating-condition subsets, not an arbitrary tuning trick.

Implementation note: different public implementations handle operating settings slightly differently. The reviewed Transformer code keeps `setting1/setting2/setting3` in the feature list and applies per-operating-condition scalers to the selected features, which is closest to the current 17-channel `condition_zscore` setup. The reviewed CNN code uses KMeans on `setting1/setting2/setting3` to create `operating_condition`, then scales sensor channels by cluster and uses the 14 selected sensors as model input. The current project chooses the Transformer-style setup for FD002/FD004 because P-SA-MCD-TCN can exploit cross-dimension interactions and because the resolved configs explicitly preserve the three setting channels. All operating-condition centers and statistics are fitted only from training units, so the preprocessing is suitable for rigorous reporting.
