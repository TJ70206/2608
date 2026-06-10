# R-SPA Stage Prototype Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight reliability-weighted stage prototype alignment loss to PG-STDA-SAC and test whether it improves strict unsupervised transfer, starting with NASA Battery -> satellite_battery_sim.

**Architecture:** Keep the existing P-SA-MCD-TCN backbone and PG-STDA-SAC losses. Add one optional transfer loss that computes per-stage source/target feature prototypes, weights target samples by pseudo-stage reliability from the stage head, and aligns matched stage prototypes without using target RUL labels.

**Tech Stack:** Python, PyTorch, YAML configs, existing `unittest` suite, existing `scripts/train_transfer.py`.

---

### Task 1: Add Failing Loss Tests

**Files:**
- Modify: `tests/test_train_transfer_options.py`
- Modify: `src/xa202608/losses.py`

- [ ] **Step 1: Write failing tests**

Add tests that import `reliability_weighted_stage_prototype_alignment_loss` and verify:

```python
def test_rspa_loss_is_lower_for_matched_stage_prototypes(self) -> None:
    source = torch.tensor([[0.0, 0.0], [0.2, 0.0], [3.0, 0.0], [3.2, 0.0]])
    target_good = torch.tensor([[0.1, 0.0], [0.3, 0.0], [3.1, 0.0], [3.3, 0.0]])
    target_bad = torch.tensor([[3.1, 0.0], [3.3, 0.0], [0.1, 0.0], [0.3, 0.0]])
    stages = torch.tensor([0, 0, 1, 1])
    logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])

    good = reliability_weighted_stage_prototype_alignment_loss(source, target_good, stages, stages, logits, 2)
    bad = reliability_weighted_stage_prototype_alignment_loss(source, target_bad, stages, stages, logits, 2)

    self.assertLess(float(good), float(bad))
```

```python
def test_rspa_loss_downweights_uncertain_target_stage_predictions(self) -> None:
    source = torch.tensor([[0.0, 0.0], [0.2, 0.0], [3.0, 0.0], [3.2, 0.0]])
    target_bad = torch.tensor([[3.1, 0.0], [3.3, 0.0], [0.1, 0.0], [0.3, 0.0]])
    stages = torch.tensor([0, 0, 1, 1])
    confident = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
    uncertain = torch.zeros_like(confident)

    confident_loss = reliability_weighted_stage_prototype_alignment_loss(source, target_bad, stages, stages, confident, 2)
    uncertain_loss = reliability_weighted_stage_prototype_alignment_loss(source, target_bad, stages, stages, uncertain, 2)

    self.assertLess(float(uncertain_loss), float(confident_loss))
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' -m unittest tests.test_train_transfer_options -v
```

Expected: import error for the missing R-SPA loss.

### Task 2: Implement R-SPA Loss

**Files:**
- Modify: `src/xa202608/losses.py`

- [ ] **Step 1: Add minimal implementation**

Implement:

```python
def reliability_weighted_stage_prototype_alignment_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_stages: torch.Tensor,
    target_stages: torch.Tensor,
    target_stage_logits: torch.Tensor | None,
    num_stages: int = 3,
    min_confidence: float = 0.0,
) -> torch.Tensor:
    ...
```

Use normalized stage-head confidence as target weights. Align source and target prototypes with squared Euclidean distance stage by stage.

- [ ] **Step 2: Run focused tests and verify GREEN**

Run:

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' -m unittest tests.test_train_transfer_options -v
```

Expected: all tests pass.

### Task 3: Connect R-SPA to Transfer Training

**Files:**
- Modify: `scripts/train_transfer.py`
- Modify: `tests/test_train_transfer_options.py`

- [ ] **Step 1: Add config parsing and history logging**

Add transfer config keys:

```yaml
prototype_alignment_weight: 0.001
prototype_min_confidence: 0.0
```

Require a stage head when the weight is positive.

- [ ] **Step 2: Add a focused training-option test**

Test that positive prototype alignment requires `return_aux=True` stage logits by exercising the helper/loss behavior with logits present.

- [ ] **Step 3: Run unit tests**

Run:

```powershell
& 'D:\anaconda\envs\jiebang\python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass.

### Task 4: Add Experiment Configs

**Files:**
- Create: `configs/nasa_to_satellite_battery_pg_stda_sac_rspa_w0p0005_srcsup0p7_50e.yaml`
- Create: `configs/nasa_to_satellite_battery_pg_stda_sac_rspa_w0p001_srcsup0p7_50e.yaml`
- Create: `configs/nasa_to_satellite_battery_pg_stda_sac_rspa_w0p003_srcsup0p7_50e.yaml`

- [ ] **Step 1: Copy final second-transfer config**

Start from `configs/nasa_to_satellite_battery_pg_stda_sac_srcsup0p7_w0p003_50e.yaml`.

- [ ] **Step 2: Add prototype weights**

Set `prototype_alignment_weight` to `0.0005`, `0.001`, and `0.003`.

### Task 5: Run Experiments and Decide

**Files:**
- Modify: `docs/SECOND_TRANSFER_PG_STDA_INNOVATION_RESULTS.md`
- Modify: `docs/internal/PG_STDA_CROSS_TASK_OPTIMIZATION_NOTES.md`
- Optionally modify: `docs/PG_STDA_SAC_FINAL_CROSS_TRANSFER_RESULTS.md`

- [ ] **Step 1: Run second-transfer R-SPA configs**

Run each config with the existing transfer training script.

- [ ] **Step 2: Compare to current final**

Baseline:

```text
RMSE 0.2914, MAE 0.2337, NASA 60.0056, RA 0.5078
```

- [ ] **Step 3: If R-SPA improves balanced metrics, run first transfer**

Create matching first-transfer config only for the best R-SPA weight and compare to:

```text
RMSE 0.1318, MAE 0.1049, NASA 7.1723, RA 0.6916
```

- [ ] **Step 4: If R-SPA fails, document why and pivot**

Next pivot candidate: physics-guided residual gating or inverse reliability weighting.
