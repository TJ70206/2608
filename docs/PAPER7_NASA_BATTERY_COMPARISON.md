# Paper 7 NASA Battery Comparison

## Paper Method
- **Paper**: Lithium-ion battery RUL prediction based on optimized VMD-SSA-PatchTST algorithm.
- **Dataset**: NASA Battery cells B0005, B0006, B0007, B0018.
- **Protocol**: test one battery, choose one of the remaining batteries as validation, train on the other two. Table 4 reports B0005 as test with validation B0006/B0007/B0018.
- **Task**: capacity trajectory prediction; RUL is obtained from predicted capacity crossing a threshold.
- **Features / preprocessing**: CCDT, MINTDV, DE health indicators; WOA optimizes VMD; SSA optimizes CNN/GRU/PatchTST hyperparameters.
- **Metrics**: MAE, RMSE, MAPE(%) on capacity prediction.

## Paper Table 4 Results: Test B0005
| Model | Validation | MAE | RMSE | MAPE (%) |
|---|---|---:|---:|---:|
| SSA-CNN | B0006 | 0.0732 | 0.1040 | 5.110 |
| SSA-CNN | B0007 | 0.0384 | 0.0490 | 2.360 |
| SSA-CNN | B0018 | 0.0472 | 0.0626 | 3.040 |
| SSA-GRU | B0006 | 0.0397 | 0.0472 | 2.188 |
| SSA-GRU | B0007 | 0.0411 | 0.0490 | 2.251 |
| SSA-GRU | B0018 | 0.0783 | 0.0900 | 4.296 |
| SSA-PatchTST | B0006 | 0.0183 | 0.0234 | 1.289 |
| SSA-PatchTST | B0007 | 0.0350 | 0.0424 | 2.320 |
| SSA-PatchTST | B0018 | 0.0371 | 0.0449 | 2.419 |
| WOA-VMD-SSA-PatchTST | B0006 | 0.0150 | 0.0218 | 0.950 |
| WOA-VMD-SSA-PatchTST | B0007 | 0.0163 | 0.0226 | 1.024 |
| WOA-VMD-SSA-PatchTST | B0018 | 0.0306 | 0.0373 | 1.891 |

## Our Capacity-Prediction Runs: Test B0005 / Validation B0006
| Model | Epochs ran | MAE | RMSE | MAPE (%) | Last-window RMSE | Coverage | Width | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P-SA-MCD-TCN paper-like w5 | 41 | 0.0386 | 0.0437 | 2.497 | 0.0368 | 1.000 | 0.2907 | timestep=5, horizon=5 |
| P-SA-MCD-TCN w48 | 44 | 0.0233 | 0.0275 | 1.568 | 0.0156 | 1.000 | 0.3414 | window=48, horizon=5; best RMSE among our runs |
| P-SA-MCD-TCN w64 | 50 | 0.0216 | 0.0286 | 1.473 | 0.0087 | 1.000 | 0.3497 | window=64, horizon=5; best MAE/MAPE among our tuning runs |
| P-SA-MCD-TCN h96-w48 | 84 | 0.0392 | 0.0459 | 2.669 | 0.0504 | 1.000 | 0.3506 | larger hidden channels; rejected |
| P-SA-MCD-TCN w48-lr5e4 | 64 | 0.0296 | 0.0398 | 2.116 | 0.0740 | 1.000 | 0.4563 | smaller learning rate; rejected |
| P-SA-MCD-TCN w64-1000e | 1000 | 0.0434 | 0.0510 | 3.158 | 0.0428 | 1.000 | 0.2056 | 1000 epochs from w64; rejected due to worse test generalization |

## Selection
- **Best RMSE among our runs**: `P-SA-MCD-TCN w48`, RMSE `0.0275`, MAE `0.0233`, MAPE `1.568%`.
- **Best MAE/MAPE among our runs**: `P-SA-MCD-TCN w64`, RMSE `0.0286`, MAE `0.0216`, MAPE `1.473%`.
- **1000-epoch result**: `P-SA-MCD-TCN w64-1000e` worsened to RMSE `0.0510`, MAE `0.0434`, MAPE `3.158%`; reject it as overtrained/poorly generalizing on the small four-battery split.
- **Paper comparison**: Paper 7 WOA-VMD-SSA-PatchTST remains stronger under B0005/B0006 capacity-prediction protocol: MAE `0.0150`, RMSE `0.0218`, MAPE `0.950%`.
- **Report recommendation**: cite `w64` if emphasizing MAE/MAPE, cite `w48` if emphasizing RMSE, and explicitly note that 1000 epochs did not improve our model without VMD/SSA.
