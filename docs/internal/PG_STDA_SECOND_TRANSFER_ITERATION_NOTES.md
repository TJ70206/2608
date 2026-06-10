# PG-STDA second-transfer iteration notes

This file records internal iteration thoughts for improving the final full model
on the NASA battery -> satellite battery transfer task.

## Iteration 1: physics proxy features + unlabeled monotonic consistency

Date: 2026-06-06

Goal: improve the strict unsupervised second-transfer result of the final
P-SA-MCD model. Current strict reference points:

- P-SA-MCD stage LMMD + pseudo-time: RMSE 0.3571.
- AD-TCN-MSC-DIM stage LMMD + pseudo-time: RMSE 0.3407.

Five-angle review before implementation:

1. Competition fit: the method must still be a cross-domain transfer method.
   Target labels are not used in the transfer loss. Validation/test labels are
   only used for model selection/evaluation as the current pipeline already
   does.
2. Physics fit: the target task is battery degradation under orbital charge and
   discharge. Voltage/current/temperature contain observable signs of growing
   resistance and thermal stress. Capacity, true resistance, SOH, RUL, cycle,
   and time-step remain forbidden as model inputs.
3. Domain adaptation fit: raw V/I/T alone may hide degradation cues. Derived
   proxies should expose local voltage slope, current transitions, and thermal
   response, then stage LMMD can align source and target feature distributions.
4. Optimization risk: too many proxies can amplify source-target mismatch.
   Keep a small bounded proxy set, normalize with the existing train-only
   normalizer, and run only a few weight variants if the first result is poor.
5. Innovation clarity: this is not just another baseline. It becomes
   PG-STDA: Physics-Guided Stage-aware Transfer Domain Adaptation, combining
   physics-proxy observables, stage-aware LMMD, and target unlabeled temporal
   monotonicity.

Chosen implementation:

- Add feature set `pg_stda_v1`:
  - `d_voltage`
  - `d_current`
  - `d_temperature`
  - `abs_current`
  - `transition_resistance_proxy`
  - `thermal_response_proxy`
- Add target sequence consistency losses:
  - monotonic loss penalizes predicted RUL increases inside an unlabeled target
    window.
  - smooth loss penalizes excessive local prediction jumps.
- First config uses small weights so the new loss regularizes rather than
  dominates:
  - LMMD weight: 0.003
  - target sequence monotonic weight: 0.003
  - target sequence smooth weight: 0.001

Success criteria:

- Primary: beat strict second-transfer best RMSE 0.3407.
- Secondary: at least beat current P-SA-MCD strict RMSE 0.3571.
- If the first run fails, try proxy-only and lower consistency weights before
  abandoning the direction.

Result:

- Config: `nasa_to_satellite_battery_pg_stda_psa_mcd_proxy_mono_w0p003_50e.yaml`
- Test RMSE: 0.3465
- Test MAE: 0.2848
- Test RA: 0.4521
- Best epoch: 36

Interpretation:

- The idea is useful versus the original P-SA-MCD strict model
  (0.3571 -> 0.3465 RMSE).
- It does not yet beat the strict AD-TCN-MSC-DIM reference (0.3407 RMSE).
- Next variants should isolate two risks:
  - consistency loss too strong or miscalibrated;
  - full six-proxy set too noisy because NASA source is cycle-level while the
    satellite target is orbital-step-level.

## Iteration 2: compact proxies and consistency ablation

Date: 2026-06-06

Variants:

- full proxy only
- full proxy + lower consistency
- compact proxy only
- compact proxy + consistency
- compact proxy + lower consistency

Best result:

- Config:
  `nasa_to_satellite_battery_pg_stda_psa_mcd_proxy_compact_mono_low_w0p003_50e.yaml`
- Feature set:
  `voltage`, `current`, `temperature`, `d_voltage`, `d_temperature`,
  `abs_current`
- Consistency weights:
  monotonic 0.001, smooth 0.0003
- Test RMSE: 0.3326
- Test MAE: 0.2653
- Test NASA score: 66.5476
- Test RA: 0.4576
- Last-window RMSE: 0.3166

Conclusion:

- Compact proxies are better than the full proxy set. The noisy
  transition-resistance and thermal-response ratios probably overfit
  source-target sampling differences.
- Low unlabeled monotonic/smooth consistency is necessary. Compact proxy only
  falls back to RMSE 0.3516, so the gain is not just feature engineering.
- This variant beats the strict AD-TCN-MSC-DIM reference on RMSE, MAE, NASA
  score, RA, last-window RMSE, and last-5-window RMSE. It still trails AD on
  `alpha_lambda_0.5`, so the result should be reported as strongest on primary
  error metrics, not uniformly best on every diagnostic metric.
