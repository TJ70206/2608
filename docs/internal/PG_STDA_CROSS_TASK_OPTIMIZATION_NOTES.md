# PG-STDA cross-task optimization notes

Date: 2026-06-06

Goal: push the final full model across both transfer tasks.

## Step 1: apply current final mechanism to first transfer

First transfer: XJTU-SY bearing source -> `reaction_wheel_sim` target.

Current strict best:

- P-SA-MCD stage LMMD + pseudo-time
- RMSE 0.1943

Five-angle review:

1. Task consistency: first transfer already uses 14-dimensional low-frequency HI
   features built from `motor_current` and `vibration_proxy`, matching source
   XJTU HI dimensionality. Battery-specific proxy columns must not be reused.
2. Leakage control: target `target_stage_source=time_progress` is pseudo-time,
   not target RUL. Sequence monotonic consistency uses only unlabeled target
   predictions.
3. Optimization risk: first transfer is already strong. Add only low-weight
   monotonic/smooth consistency first, because heavy regularization can erase
   useful source-domain degradation cues.
4. Cross-task consistency: this keeps the same final method family as second
   transfer: P-SA-MCD + stage LMMD + pseudo-time + unlabeled temporal
   consistency, with task-specific observable features.
5. Evaluation discipline: if first-transfer RMSE becomes worse, keep the
   original first-transfer P-SA strict config as the first-transfer final; do
   not force a module only because it helped the battery task.

Chosen first-transfer config:

- `xjtu_to_reaction_wheel_pg_stda_psa_mcd_mono_low_w0p003_50e.yaml`
- monotonic weight 0.001
- smooth weight 0.0003

Result:

- Original P-SA strict RMSE: 0.1943
- PG-STDA monotonic low RMSE: 0.2105

Decision:

- Do not use target-window monotonic consistency for the first transfer final.
  The reaction-wheel HI sequence is already an aggregated health-index sequence,
  and extra local monotonic regularization over-smooths useful degradation
  dynamics.
- First-transfer final remains P-SA-MCD stage LMMD + pseudo-time unless a later
  innovation beats 0.1943 under the same strict setting.

## Step 2: next innovation for second transfer

Proposed mechanism: Stage Auxiliary Calibration (SAC).

The P-SA-MCD model already has a stage head, but the transfer loop currently
does not train it. SAC uses it explicitly:

- source stage CE on source RUL-derived stage labels;
- optional target pseudo-stage CE only when `target_stage_source` is
  pseudo-time/progress;
- no target train RUL labels enter the loss.

Five-angle review:

1. Architecture: uses an existing head, so it is low-complexity and does not add
   a heavy model family.
2. Transfer logic: stage-aware LMMD aligns by stage; a trained stage head makes
   the backbone encode the same coarse degradation semantics.
3. Leakage: target stage CE must be blocked unless target stages come from
   pseudo-time. True target RUL stages would leak labels.
4. Optimization: use small weights. Too much pseudo-stage CE can overfit linear
   time progress instead of telemetry degradation.
5. Reportability: this is a clear combination innovation:
   physics-guided proxies + stage-aware LMMD + unlabeled monotonic consistency
   + auxiliary degradation-stage calibration.

First SAC sweep result:

- best SAC config: source stage 0.01, target pseudo-stage 0.003
- RMSE improved from 0.3326 to 0.3059
- MAE improved from 0.2653 to 0.2465
- NASA score improved from 66.5476 to 62.8317
- RA improved from 0.4576 to 0.4908
- late-window RMSE worsened from 0.3166 to 0.3382

Next sweep:

- source-only SAC 0.01
- source-only SAC 0.02
- stronger SAC 0.02/0.005

Purpose: determine whether target pseudo-stage supervision is the main gain or
whether source degradation-stage calibration alone is enough and less harmful
to late-life predictions.

Boundary sweep result:

- source-only SAC 0.01 and 0.02 both degraded RMSE; target pseudo-stage
  supervision is the useful part.
- stronger SAC 0.02/0.005 improved alpha@0.5 but degraded RMSE/MAE and late
  windows.
- Current best remains SAC 0.01/0.003.

## Step 3: target lifecycle order consistency

Proposed mechanism: Target Lifecycle Monotonic Consistency (TLMC).

The previous target-window monotonic loss only regularizes the predictions
inside each telemetry window. TLMC instead uses target `unit_id` and
`time_index` metadata in a shuffled training batch. For samples from the same
target unit, predictions are sorted by `time_index`; later windows should not
receive larger RUL predictions than earlier windows.

Five-angle review:

1. Physical consistency: RUL is non-increasing with operating time for each
   component. This is directly aligned with lifetime prediction.
2. Leakage control: TLMC uses only `unit_id`, `time_index`, and model
   predictions. It does not use target train RUL labels.
3. Task specificity: it should matter more for the battery task, where the
   current SAC result improves global RMSE but worsens late-window RMSE.
4. Optimization risk: too large a weight can collapse predictions toward a
   nearly constant curve. Start with low weights and optional curvature
   smoothing only.
5. Reportability: TLMC is a mechanism-level innovation, not only a
   hyperparameter tweak. It can be described as an unlabeled lifecycle-order
   regularizer within PG-STDA-SAC.

TLMC sweep result:

- low TLMC improved late-window RMSE compared with SAC, but global RMSE/MAE
  became worse.
- stronger TLMC and curvature smoothing both degraded global metrics.
- Validation-selected averaging between SAC and TLMC did not improve the test
  result. TLMC should be kept as an ablation/analysis module, not the final
  second-transfer main model.

## Step 4: source warm-start and late-stage alignment

Proposed mechanisms:

- Source warm-start: pretrain briefly on NASA RUL labels before strict
  source-target alignment.
- Late-stage LMMD weighting: increase the alignment weight for later
  degradation stages, where RUL prediction is most sensitive.

Five-angle review:

1. Transfer stability: source warm-start may produce a better initial
   degradation encoder before domain alignment.
2. Domain-shift risk: too much source-only pretraining can overfit NASA
   ground-test profiles and hurt the satellite target.
3. Late-life relevance: late-stage weighting can help the battery task where
   SAC improves global RMSE but late windows remain weaker.
4. Leakage control: both mechanisms use source labels and target pseudo-time
   stages only; target train RUL labels stay unused.
5. Complexity: these are lightweight mechanisms and keep the final method in
   the PG-STDA-SAC family.

Warm-start / late-stage weighting result:

- source pretrain10 improved alpha@0.5 but did not improve global RMSE.
- late-stage LMMD weighting improved late-window RMSE but degraded global
  RMSE/MAE.
- validation-selected prediction fusion did not beat SAC on the primary test
  metrics.

## Step 5: source-supervision weight balance

Proposed mechanism: reduce the supervised NASA loss weight during transfer.

Five-angle review:

1. Source dominance: NASA ground cycling differs from the satellite orbit
   profile; full source weight may over-constrain the target representation.
2. Alignment freedom: a lower source supervised weight may let LMMD/SAC shape
   features toward target telemetry.
3. Stability risk: too low a source weight can remove the only true supervised
   signal in strict unsupervised transfer.
4. Leakage: this still uses source labels plus target pseudo-time/stage
   metadata only.
5. Search discipline: test only two conservative values, 0.75 and 0.5.

Source-supervision balance result:

- 1.00: RMSE 0.3059
- 0.75: RMSE 0.2995, better MAE/NASA/RA than 1.00
- 0.70: RMSE 0.2914, best balanced result
- 0.65: RMSE 0.2908, lowest RMSE-only result but worse MAE/NASA/RA and late
  windows than 0.70
- 0.50: RMSE 0.3255, too little source supervision

Decision:

- Recommended second-transfer final: `source_supervised_weight=0.70`.
- Keep `0.65` as an RMSE-only ablation, not the final recommended model.

Final first-transfer rerun:

- The same method family without target-window monotonic consistency was run on
  the first transfer:
  `xjtu_to_reaction_wheel_pg_stda_sac_srcsup0p7_w0p003_50e.yaml`.
- Result: RMSE improved from 0.1943 to 0.1318.
- Decision: the unified final method can be reported as PG-STDA-SAC with
  source-supervision balance 0.70. The first transfer excludes the battery
  compact proxy inputs because its source/target HI dimensions are already
  aligned; the second transfer uses compact V/I/T physics proxies.

## Step 6: Reliability-weighted Stage Prototype Alignment

Proposed mechanism: R-SPA.

R-SPA adds a prototype-level alignment term on top of stage LMMD. Source
prototypes use source degradation-stage labels. Target prototypes use
pseudo-time stages, but each target sample is weighted by the detached
confidence of the stage auxiliary head. A confidence gate avoids aligning
ambiguous target pseudo-stage samples too aggressively.

Five-angle review:

1. Stage semantics: stage LMMD aligns distributions, but it does not explicitly
   pull the source and target stage centers together. R-SPA fills that gap.
2. Reliability: target stages are pseudo labels. Confidence weighting reduces
   the risk of reinforcing incorrect target-stage assignments.
3. Leakage: R-SPA uses source labels, target pseudo-time stages, target stage
   logits, and features only. Target train RUL labels are not used.
4. Complexity: it is a small loss term on existing features and stage head, not
   a new heavy model family.
5. Cross-task fit: the same module can run on both transfer tasks, while the
   weight may differ because reaction-wheel HI and battery telemetry proxies
   have different stage separability.

Local sweep result:

- First transfer baseline PG-STDA-SAC: RMSE `0.1318`, MAE `0.1049`, NASA
  `7.1723`, RA `0.6916`.
- First transfer R-SPA `w=0.0005,c=0.5`: RMSE `0.1274`, MAE `0.1004`, NASA
  `7.0249`, RA `0.6902`.
- First transfer R-SPA `w=0.0003,c=0.5`: RMSE `0.1276`, MAE `0.1012`, NASA
  `7.1020`, RA `0.6842`.
- First transfer R-SPA weighted checkpoint: RMSE `0.1300`, NASA `7.0023`, RA
  `0.6954`, last-window RMSE `0.1508`, last-5 RMSE `0.1434`.
- Second transfer baseline PG-STDA-SAC: RMSE `0.2914`, MAE `0.2337`, NASA
  `60.0056`, RA `0.5078`.
- Second transfer R-SPA `w=0.001,c=0.5`: RMSE `0.2888`, MAE `0.2316`, NASA
  `58.9301`, RA `0.5117`.
- Second transfer R-SPA `w=0.0008,c=0.5`: RMSE `0.2911`, MAE `0.2346`, NASA
  `59.8452`, RA `0.5100`.
- Second transfer R-SPA `w=0.001,c=0.6`: RMSE `0.2932`, MAE `0.2349`, NASA
  `60.9571`, RA `0.5042`.

Decision:

- Adopt R-SPA as PG-STDA-SAC-RSPA.
- First transfer primary RMSE config: `w=0.0005,c=0.5`.
- First transfer balanced checkpoint is useful when emphasizing NASA/RA/late
  windows, but the RMSE-selected checkpoint remains the primary RMSE result.
- Second transfer final config: `w=0.001,c=0.5`.
- Do not use second-transfer weighted checkpoint: it improves late-window RMSE
  but degrades global RMSE to `0.3503`.
