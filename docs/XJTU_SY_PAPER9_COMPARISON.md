# XJTU-SY Paper 9 Protocol and Results

## Local processed data
- **Raw folders**: `data/raw/xjtu_sy/35Hz12kN`, `37.5Hz11kN`, `40Hz10kN`.
- **Processed cache**: `data/processed/xjtu_sy_hi.npz`. This is not a new dataset; it is a low-frequency HI feature cache extracted from the raw CSV files.
- **Units**: 15 bearings, 5 per condition.
- **Feature dimension**: 14 = 2 vibration channels x 7 statistics (`RMS`, `STD`, `skewness`, `kurtosis`, `peak-to-peak`, `crest factor`, `trend slope`).

| Unit | Condition | Sequence length | Feature dim |
|---|---|---:|---:|
| Bearing1_1 | C1 / 35Hz12kN | 123 | 14 |
| Bearing1_2 | C1 / 35Hz12kN | 161 | 14 |
| Bearing1_3 | C1 / 35Hz12kN | 158 | 14 |
| Bearing1_4 | C1 / 35Hz12kN | 122 | 14 |
| Bearing1_5 | C1 / 35Hz12kN | 52 | 14 |
| Bearing2_1 | C2 / 37.5Hz11kN | 491 | 14 |
| Bearing2_2 | C2 / 37.5Hz11kN | 161 | 14 |
| Bearing2_3 | C2 / 37.5Hz11kN | 533 | 14 |
| Bearing2_4 | C2 / 37.5Hz11kN | 42 | 14 |
| Bearing2_5 | C2 / 37.5Hz11kN | 339 | 14 |
| Bearing3_1 | C3 / 40Hz10kN | 2538 | 14 |
| Bearing3_2 | C3 / 40Hz10kN | 2496 | 14 |
| Bearing3_3 | C3 / 40Hz10kN | 371 | 14 |
| Bearing3_4 | C3 / 40Hz10kN | 1515 | 14 |
| Bearing3_5 | C3 / 40Hz10kN | 114 | 14 |

## Paper 9 XJTU-SY protocol
- **Paper**: Remaining Useful Life Prediction Across Conditions Based on a Health Indicator-Weighted Subdomain Alignment Network.
- **Dataset**: XJTU-SY run-to-failure bearing dataset.
- **Conditions**: C1 = 12,000 N / 2100 rpm; C2 = 11,000 N / 2250 rpm; C3 = 10,000 N / 2400 rpm.
- **Model idea**: multi-scale encoder, health-indicator temporal weights, two subdomains, LMMD-based subdomain alignment.
- **Metrics**: normalized RUL `MAE`, `RMSE`, and paper-defined `Score`; lower MAE/RMSE is better and higher Score is better.

## Paper 9 Table 13: transfer tasks
| Task | Conditions | Labeled source | Unlabeled target | Test bearing |
|---|---|---|---|---|
| B1 | C1 -> C2 | B1_1 | B2_1 | B2_3 |
| B2 | C1 -> C3 | B1_1 | B3_3 | B3_1 |
| B3 | C2 -> C1 | B2_1 | B1_3 | B1_1 |
| B4 | C2 -> C3 | B2_1 | B3_3 | B3_1 |
| B5 | C3 -> C1 | B3_1 | B1_3 | B1_1 |
| B6 | C3 -> C2 | B3_1 | B2_3 | B2_1 |

## Paper 9 Table 14: XJTU-SY results
| Task | Model | MAE | RMSE | Score |
|---|---|---:|---:|---:|
| B1 | DSAN-WM | 0.1763 +/- 0.0109 | 0.2211 +/- 0.0236 | 0.3290 +/- 0.0294 |
| B1 | GSAN | 0.1595 +/- 0.0251 | 0.2012 +/- 0.0359 | 0.3498 +/- 0.0325 |
| B1 | HIWSAN | 0.1382 +/- 0.0161 | 0.1725 +/- 0.0255 | 0.3887 +/- 0.0364 |
| B2 | DSAN-WM | 0.1517 +/- 0.0178 | 0.2084 +/- 0.0459 | 0.3735 +/- 0.0257 |
| B2 | GSAN | 0.1289 +/- 0.0249 | 0.1623 +/- 0.0460 | 0.3831 +/- 0.0495 |
| B2 | HIWSAN | 0.0863 +/- 0.0117 | 0.1017 +/- 0.0332 | 0.4821 +/- 0.0310 |
| B3 | DSAN-WM | 0.2102 +/- 0.0383 | 0.2930 +/- 0.0642 | 0.3317 +/- 0.0306 |
| B3 | GSAN | 0.1704 +/- 0.0563 | 0.2297 +/- 0.0419 | 0.3442 +/- 0.0605 |
| B3 | HIWSAN | 0.1090 +/- 0.0143 | 0.1327 +/- 0.0225 | 0.4440 +/- 0.0331 |
| B4 | DSAN-WM | 0.1804 +/- 0.0493 | 0.2291 +/- 0.0645 | 0.3308 +/- 0.0801 |
| B4 | GSAN | 0.1608 +/- 0.0151 | 0.2096 +/- 0.0285 | 0.3517 +/- 0.0327 |
| B4 | HIWSAN | 0.1377 +/- 0.0213 | 0.1703 +/- 0.0249 | 0.3863 +/- 0.0451 |
| B5 | DSAN-WM | 0.1927 +/- 0.0521 | 0.2609 +/- 0.0564 | 0.3501 +/- 0.0487 |
| B5 | GSAN | 0.1623 +/- 0.0407 | 0.2107 +/- 0.0373 | 0.3749 +/- 0.0602 |
| B5 | HIWSAN | 0.1114 +/- 0.0155 | 0.1320 +/- 0.0264 | 0.4295 +/- 0.0362 |
| B6 | DSAN-WM | 0.1812 +/- 0.0190 | 0.2320 +/- 0.0804 | 0.3252 +/- 0.0303 |
| B6 | GSAN | 0.1597 +/- 0.0296 | 0.2002 +/- 0.0443 | 0.3550 +/- 0.0390 |
| B6 | HIWSAN | 0.1311 +/- 0.0230 | 0.1586 +/- 0.0313 | 0.3927 +/- 0.0478 |

## Average over B1-B6
| Model | Avg MAE | Avg RMSE | Avg Score |
|---|---:|---:|---:|
| DSAN-WM | 0.1821 | 0.2407 | 0.3401 |
| GSAN | 0.1569 | 0.2023 | 0.3598 |
| HIWSAN | 0.1190 | 0.1446 | 0.4205 |

## How we should compare
- For strict Paper 9 comparison, use the same B1-B6 cross-condition tasks and report normalized RUL MAE/RMSE/Score.
- For XA-202608 main line, XJTU-SY should be treated as the public source domain and transferred to `reaction_wheel_sim`; do not directly transfer raw high-frequency vibration.
- Our current `xjtu_sy_hi.npz` is suitable as a source-domain low-frequency HI representation for that transfer setting.
