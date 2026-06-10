# Competition Delivery Cleanup Plan

Goal: convert the current experiment-heavy workspace into a competition-ready
evidence package while preserving reproducibility.

## Requirements Covered

- Two scenario tracks: reaction wheel and satellite battery.
- Public degradation source to satellite simulation target transfer flow.
- Strict unsupervised transfer comparison under the same data conditions.
- Supervised reference baselines kept separate from strict transfer claims.
- Ablations for PG-STDA, SAC, source-supervision balance, and R-SPA.
- Parameter sensitivity for LMMD and R-SPA weights.
- Prediction and metric visualizations for report and bonus-point evidence.
- Low-risk cache cleanup without deleting final reproducibility evidence.

## Implementation Steps

1. Add `scripts/prepare_competition_artifacts.py`.
2. Generate `competition_artifacts/` with ordered subfolders:
   - `00_requirement_mapping`
   - `01_datasets`
   - `02_experiments`
   - `03_results`
   - `04_figures`
   - `05_report_assets`
   - `99_cleanup`
3. Copy final configs, metrics, predictions, and selected evidence docs.
4. Write master CSV/Markdown tables for strict baselines, supervised references,
   ablations, and parameter sensitivity.
5. Generate matplotlib figures from existing prediction and metric files.
6. Remove low-risk generated caches and duplicate non-final checkpoints only
   after artifact generation succeeds.
7. Run unit tests and inspect generated artifact inventory.

## Safety Rules

- Do not delete raw public datasets or simulated datasets.
- Keep final first-transfer and second-transfer model checkpoints.
- Keep all metric JSON and prediction CSV files used by the artifact package.
- Delete only verified paths inside the project directory.
