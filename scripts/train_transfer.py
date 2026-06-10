from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from xa202608.config import load_config
from xa202608.data.satellite_battery import build_nasa_satellite_battery_transfer_loaders
from xa202608.data.synthetic import build_synthetic_transfer_debug_loaders
from xa202608.data.xjtu_reaction_wheel import build_xjtu_reaction_wheel_transfer_loaders
from xa202608.data.xjtu_sy import build_xjtu_transfer_loaders
from xa202608.experiment_io import benchmark_inference_latency, collect_environment_info, save_prediction_csv
from xa202608.losses import reliability_weighted_stage_prototype_alignment_loss, transfer_alignment_loss
from xa202608.models.factory import build_model
from xa202608.conformal import SplitConformalRegressor
from xa202608.metrics import rul_metrics_with_time
from xa202608.train_eval import predict
from xa202608.utils import count_parameters, ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XA-202608 transfer model with SA-FS-LMMD.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def _is_better_checkpoint(current: float, best: float, mode: str, min_delta: float) -> bool:
    mode_key = str(mode).lower()
    if mode_key == "min":
        return float(current) < float(best) - float(min_delta)
    if mode_key == "max":
        return float(current) > float(best) + float(min_delta)
    raise ValueError("checkpoint_mode must be 'min' or 'max'")


def _scheduled_alignment_weight(base_weight: float, epoch: int, warmup_epochs: int) -> float:
    if int(warmup_epochs) <= 0:
        return float(base_weight)
    progress = min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))
    return float(base_weight) * progress


def _resolve_transfer_steps(transfer_cfg: dict, source_loader, target_loader) -> int:
    explicit_steps = transfer_cfg.get("steps_per_epoch")
    if explicit_steps is not None:
        steps = int(explicit_steps)
        if steps <= 0:
            raise ValueError("transfer.steps_per_epoch must be positive")
        return steps
    epoch_steps = str(transfer_cfg.get("epoch_steps", "source")).lower()
    if epoch_steps in {"source", "source_train"}:
        return len(source_loader)
    if epoch_steps in {"target", "target_train"}:
        return len(target_loader)
    if epoch_steps in {"max", "maximum"}:
        return max(len(source_loader), len(target_loader))
    if epoch_steps in {"min", "minimum"}:
        return min(len(source_loader), len(target_loader))
    raise ValueError("transfer.epoch_steps must be source, target, max, min, or set transfer.steps_per_epoch")


def _supervised_loss(
    criterion: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
    stage: torch.Tensor,
    late_stage_weight: float = 1.0,
    late_stage_threshold: int = 2,
    late_prediction_weight: float = 1.0,
) -> torch.Tensor:
    loss = criterion(pred, target).reshape(-1)
    if late_stage_weight != 1.0:
        stage_weight = torch.where(
            stage.reshape(-1) >= int(late_stage_threshold),
            float(late_stage_weight),
            1.0,
        ).to(device=loss.device, dtype=loss.dtype)
        loss = loss * stage_weight
    if late_prediction_weight != 1.0:
        prediction_weight = torch.where(
            pred.detach().reshape(-1) > target.reshape(-1),
            float(late_prediction_weight),
            1.0,
        ).to(device=loss.device, dtype=loss.dtype)
        loss = loss * prediction_weight
    return loss.mean()


def _sequence_temporal_consistency_losses(
    sequence_pred: torch.Tensor,
    monotonic_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sequence_pred.ndim < 2:
        raise ValueError("sequence_pred must have shape [batch, time, ...]")
    if sequence_pred.size(1) < 2:
        zero = sequence_pred.sum() * 0.0
        return zero, zero
    diffs = sequence_pred[:, 1:, ...] - sequence_pred[:, :-1, ...]
    monotonic_loss = torch.relu(diffs - float(monotonic_margin)).mean()
    smooth_loss = diffs.pow(2).mean()
    return monotonic_loss, smooth_loss


def _batch_time_order_consistency_losses(
    pred: torch.Tensor,
    unit_id: torch.Tensor,
    time_index: torch.Tensor,
    monotonic_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_flat = pred.reshape(-1)
    unit_flat = unit_id.reshape(-1)
    time_flat = time_index.reshape(-1)
    monotonic_terms = []
    smooth_terms = []
    for uid in torch.unique(unit_flat):
        unit_mask = unit_flat == uid
        indices = torch.nonzero(unit_mask, as_tuple=False).reshape(-1)
        if indices.numel() < 2:
            continue
        order = torch.argsort(time_flat[indices])
        sorted_pred = pred_flat[indices[order]]
        diffs = sorted_pred[1:] - sorted_pred[:-1]
        monotonic_terms.append(torch.relu(diffs - float(monotonic_margin)))
        if sorted_pred.numel() >= 3:
            curvature = sorted_pred[2:] - 2.0 * sorted_pred[1:-1] + sorted_pred[:-2]
            smooth_terms.append(curvature.pow(2))
    zero = pred_flat.sum() * 0.0
    monotonic_loss = torch.cat(monotonic_terms).mean() if monotonic_terms else zero
    smooth_loss = torch.cat(smooth_terms).mean() if smooth_terms else zero
    return monotonic_loss, smooth_loss


def _stage_auxiliary_loss(stage_logits: torch.Tensor | None, stage: torch.Tensor) -> torch.Tensor:
    if stage_logits is None:
        raise ValueError("stage auxiliary loss requires model stage logits; enable a model stage head")
    return nn.functional.cross_entropy(stage_logits, stage.reshape(-1).long())


def _target_stage_aux_uses_unlabeled_stages(data_cfg: dict) -> bool:
    target_stage_source = str(data_cfg.get("target_stage_source", "rul")).lower()
    return target_stage_source in {"time", "time_progress", "progress", "pseudo_time"}


def _prototype_alignment_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_stages: torch.Tensor,
    target_stages: torch.Tensor,
    target_stage_logits: torch.Tensor | None,
    num_stages: int,
    min_confidence: float,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return target_features.sum() * 0.0
    if target_stage_logits is None:
        raise ValueError("prototype alignment requires model stage logits; enable a model stage head")
    return reliability_weighted_stage_prototype_alignment_loss(
        source_features,
        target_features,
        source_stages,
        target_stages,
        target_stage_logits,
        num_stages=int(num_stages),
        min_confidence=float(min_confidence),
    )


def _time_calibration_features(
    pred: dict[str, np.ndarray],
    time_min: float,
    time_scale: float,
    degree: int,
) -> np.ndarray:
    y_pred = pred["y_pred"].astype(np.float64).reshape(-1)
    time_index = pred["time_index"].astype(np.float64).reshape(-1)
    time_feature = np.clip((time_index - float(time_min)) / float(time_scale), 0.0, 1.0)
    columns = [np.ones_like(y_pred), y_pred, time_feature]
    if int(degree) >= 2:
        columns.extend([y_pred**2, time_feature**2, y_pred * time_feature])
    return np.vstack(columns).T


def _fit_time_aware_output_calibration(
    val_pred: dict[str, np.ndarray],
    calibration_cfg: dict,
) -> dict[str, Any]:
    degree = int(calibration_cfg.get("degree", 2))
    if degree not in {1, 2}:
        raise ValueError("time-aware output calibration degree must be 1 or 2")
    ridge = float(calibration_cfg.get("ridge", 0.01))
    if ridge < 0.0:
        raise ValueError("time-aware output calibration ridge must be non-negative")
    time_min = float(np.min(val_pred["time_index"].astype(np.float64)))
    time_max = float(np.max(val_pred["time_index"].astype(np.float64)))
    time_scale = max(time_max - time_min, 1e-8)
    x = _time_calibration_features(val_pred, time_min=time_min, time_scale=time_scale, degree=degree)
    y = val_pred["y_true"].astype(np.float64).reshape(-1)
    regularizer = ridge * np.eye(x.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
    return {
        "degree": degree,
        "ridge": ridge,
        "time_min": time_min,
        "time_scale": time_scale,
        "coef": coef,
    }


def _apply_time_aware_output_calibration(
    pred: dict[str, np.ndarray],
    calibration: dict[str, Any],
    clip_range: tuple[float, float] = (0.0, 1.0),
) -> None:
    x = _time_calibration_features(
        pred,
        time_min=float(calibration["time_min"]),
        time_scale=float(calibration["time_scale"]),
        degree=int(calibration["degree"]),
    )
    values = x @ calibration["coef"]
    pred["y_pred"] = np.clip(values, float(clip_range[0]), float(clip_range[1]))


def _train_source_pretrain_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    late_stage_weight: float = 1.0,
    late_stage_threshold: int = 2,
    late_prediction_weight: float = 1.0,
) -> float:
    model.train()
    losses = []
    for batch in tqdm(loader, desc="source-pretrain", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        stage = batch["stage"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = _supervised_loss(
            criterion,
            pred,
            y,
            stage,
            late_stage_weight=late_stage_weight,
            late_stage_threshold=late_stage_threshold,
            late_prediction_weight=late_prediction_weight,
        )
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_transfer(config: dict) -> dict[str, float]:
    seed = int(config["experiment"].get("seed", 42))
    set_seed(seed)
    output_dir = ensure_dir(PROJECT_ROOT / config["experiment"].get("output_dir", "outputs/transfer"))
    save_json(config, output_dir / "resolved_config.json")
    dataset = str(config["data"]["dataset"]).lower()
    if dataset == "synthetic_transfer_debug":
        source_train, target_train, target_val, target_test, input_channels = build_synthetic_transfer_debug_loaders(config)
    elif dataset == "xjtu_sy_transfer":
        source_train, target_train, target_val, target_test, input_channels = build_xjtu_transfer_loaders(config)
    elif dataset in {"xjtu_reaction_wheel_transfer", "xjtu_to_reaction_wheel"}:
        source_train, target_train, target_val, target_test, input_channels = build_xjtu_reaction_wheel_transfer_loaders(config)
    elif dataset in {"nasa_to_satellite_battery", "nasa_satellite_battery_transfer"}:
        source_train, target_train, target_val, target_test, input_channels = build_nasa_satellite_battery_transfer_loaders(config)
    else:
        raise ValueError(f"unknown transfer dataset: {config['data']['dataset']}")
    config["model"]["input_channels"] = input_channels
    model = build_model(config)
    device = get_device(str(config["train"].get("device", "auto")))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    criterion = nn.SmoothL1Loss()
    transfer_cfg = config.get("transfer", {})
    lmmd_weight = float(transfer_cfg.get("lmmd_weight", 0.1))
    alignment_mode = str(transfer_cfg.get("alignment", "stage"))
    num_stages = int(transfer_cfg.get("num_stages", 3))
    source_supervised_weight = float(transfer_cfg.get("source_supervised_weight", 1.0))
    target_supervised_weight = float(transfer_cfg.get("target_supervised_weight", 0.0))
    source_pretrain_epochs = int(transfer_cfg.get("source_pretrain_epochs", 0))
    lmmd_warmup_epochs = int(transfer_cfg.get("lmmd_warmup_epochs", 0))
    stage_weights_cfg = transfer_cfg.get("stage_weights")
    if stage_weights_cfg is not None and len(stage_weights_cfg) < num_stages:
        raise ValueError("transfer.stage_weights length must be at least transfer.num_stages")
    source_iter = cycle_loader(source_train)
    target_iter = cycle_loader(target_train)
    transfer_steps = _resolve_transfer_steps(transfer_cfg, source_train, target_train)
    history = []
    pretrain_history = []
    train_cfg = config["train"]
    criterion = nn.SmoothL1Loss(reduction="none")
    checkpoint_metric = str(train_cfg.get("checkpoint_metric", "target_val_rmse")).lower()
    checkpoint_weights = train_cfg.get("checkpoint_metric_weights", {})
    checkpoint_mode = str(train_cfg.get("checkpoint_mode", "min")).lower()
    checkpoint_min_epoch = int(train_cfg.get("checkpoint_min_epoch", 1))
    checkpoint_min_delta = float(train_cfg.get("checkpoint_min_delta", 0.0))
    if checkpoint_metric not in {"target_val_rmse", "val_rmse", "rmse", "weighted", "weighted_sum", "combined"}:
        raise ValueError("transfer checkpoint_metric must be target_val_rmse, val_rmse, rmse, or weighted")
    if checkpoint_mode not in {"min", "max"}:
        raise ValueError("checkpoint_mode must be 'min' or 'max'")
    best_val_metric = float("inf") if checkpoint_mode == "min" else -float("inf")
    best_epoch = 0
    best_state = None
    grad_clip = float(config["train"].get("grad_clip_norm", 0.0))
    late_stage_weight = float(transfer_cfg.get("late_stage_weight", train_cfg.get("late_stage_weight", 1.0)))
    late_stage_threshold = int(transfer_cfg.get("late_stage_threshold", train_cfg.get("late_stage_threshold", 2)))
    late_prediction_weight = float(
        transfer_cfg.get("late_prediction_weight", train_cfg.get("late_prediction_weight", 1.0))
    )
    target_sequence_monotonic_weight = float(
        transfer_cfg.get("target_sequence_monotonic_weight", transfer_cfg.get("target_monotonic_weight", 0.0))
    )
    target_sequence_smooth_weight = float(
        transfer_cfg.get("target_sequence_smooth_weight", transfer_cfg.get("target_smooth_weight", 0.0))
    )
    target_sequence_monotonic_margin = float(
        transfer_cfg.get("target_sequence_monotonic_margin", transfer_cfg.get("target_monotonic_margin", 0.0))
    )
    target_lifecycle_monotonic_weight = float(
        transfer_cfg.get("target_lifecycle_monotonic_weight", transfer_cfg.get("target_time_monotonic_weight", 0.0))
    )
    target_lifecycle_smooth_weight = float(
        transfer_cfg.get("target_lifecycle_smooth_weight", transfer_cfg.get("target_time_smooth_weight", 0.0))
    )
    target_lifecycle_monotonic_margin = float(
        transfer_cfg.get("target_lifecycle_monotonic_margin", transfer_cfg.get("target_time_monotonic_margin", 0.0))
    )
    source_stage_aux_weight = float(transfer_cfg.get("source_stage_aux_weight", transfer_cfg.get("stage_aux_weight", 0.0)))
    target_stage_aux_weight = float(transfer_cfg.get("target_stage_aux_weight", 0.0))
    prototype_alignment_weight = float(
        transfer_cfg.get("prototype_alignment_weight", transfer_cfg.get("rspa_weight", 0.0))
    )
    prototype_min_confidence = float(transfer_cfg.get("prototype_min_confidence", 0.0))
    if target_stage_aux_weight > 0.0 and not _target_stage_aux_uses_unlabeled_stages(config.get("data", {})):
        raise ValueError(
            "transfer.target_stage_aux_weight requires data.target_stage_source to be time_progress/pseudo_time "
            "so target train RUL labels are not used as stage supervision"
        )
    use_stage_aux = source_stage_aux_weight > 0.0 or target_stage_aux_weight > 0.0 or prototype_alignment_weight > 0.0
    for pretrain_epoch in range(1, source_pretrain_epochs + 1):
        pretrain_loss = _train_source_pretrain_epoch(
            model,
            source_train,
            optimizer,
            criterion,
            device,
            grad_clip,
            late_stage_weight=late_stage_weight,
            late_stage_threshold=late_stage_threshold,
            late_prediction_weight=late_prediction_weight,
        )
        val_pred = predict(model, target_val, device)
        val_rmse = float(np.sqrt(np.mean((val_pred["y_true"] - val_pred["y_pred"]) ** 2)))
        row = {
            "phase": "source_pretrain",
            "epoch": pretrain_epoch,
            "source_pretrain_loss": pretrain_loss,
            "target_val_rmse": val_rmse,
        }
        pretrain_history.append(row)
        print(row)
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        losses = []
        pred_losses = []
        target_pred_losses = []
        alignment_losses = []
        target_sequence_monotonic_losses = []
        target_sequence_smooth_losses = []
        target_lifecycle_monotonic_losses = []
        target_lifecycle_smooth_losses = []
        source_stage_aux_losses = []
        target_stage_aux_losses = []
        prototype_alignment_losses = []
        effective_lmmd_weight = _scheduled_alignment_weight(lmmd_weight, epoch, lmmd_warmup_epochs)
        for _ in tqdm(range(transfer_steps), desc="transfer-train", leave=False):
            source_batch = next(source_iter)
            target_batch = next(target_iter)
            sx = source_batch["x"].to(device)
            sy = source_batch["y"].to(device)
            ss = source_batch["stage"].to(device)
            tx = target_batch["x"].to(device)
            ty = target_batch["y"].to(device)
            ts = target_batch["stage"].to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_stage_aux:
                try:
                    spred, sfeat, source_stage_logits = model(sx, return_aux=True)
                    tpred, tfeat, target_stage_logits = model(tx, return_aux=True)
                except TypeError as exc:
                    raise ValueError("stage auxiliary losses require a model that supports return_aux=True") from exc
            else:
                spred, sfeat = model(sx, return_features=True)
                tpred, tfeat = model(tx, return_features=True)
                source_stage_logits = None
                target_stage_logits = None
            pred_loss = _supervised_loss(
                criterion,
                spred,
                sy,
                ss,
                late_stage_weight=late_stage_weight,
                late_stage_threshold=late_stage_threshold,
                late_prediction_weight=late_prediction_weight,
            )
            target_pred_loss = _supervised_loss(
                criterion,
                tpred,
                ty,
                ts,
                late_stage_weight=late_stage_weight,
                late_stage_threshold=late_stage_threshold,
                late_prediction_weight=late_prediction_weight,
            )
            stage_weights = (
                torch.as_tensor(stage_weights_cfg, device=device, dtype=sfeat.dtype)
                if stage_weights_cfg is not None
                else None
            )
            lmmd = transfer_alignment_loss(
                sfeat,
                tfeat,
                ss,
                ts,
                mode=alignment_mode,
                num_stages=num_stages,
                stage_weights=stage_weights,
            )
            if target_sequence_monotonic_weight > 0.0 or target_sequence_smooth_weight > 0.0:
                target_sequence_pred = model(tx, return_sequence=True)
                target_sequence_monotonic_loss, target_sequence_smooth_loss = _sequence_temporal_consistency_losses(
                    target_sequence_pred,
                    monotonic_margin=target_sequence_monotonic_margin,
                )
            else:
                target_sequence_monotonic_loss = tpred.sum() * 0.0
                target_sequence_smooth_loss = tpred.sum() * 0.0
            if source_stage_aux_weight > 0.0:
                source_stage_aux_loss = _stage_auxiliary_loss(source_stage_logits, ss)
            else:
                source_stage_aux_loss = tpred.sum() * 0.0
            if target_stage_aux_weight > 0.0:
                target_stage_aux_loss = _stage_auxiliary_loss(target_stage_logits, ts)
            else:
                target_stage_aux_loss = tpred.sum() * 0.0
            prototype_alignment_loss = _prototype_alignment_loss(
                sfeat,
                tfeat,
                ss,
                ts,
                target_stage_logits,
                num_stages=num_stages,
                min_confidence=prototype_min_confidence,
                enabled=prototype_alignment_weight > 0.0,
            )
            if target_lifecycle_monotonic_weight > 0.0 or target_lifecycle_smooth_weight > 0.0:
                target_lifecycle_monotonic_loss, target_lifecycle_smooth_loss = _batch_time_order_consistency_losses(
                    tpred,
                    target_batch["unit_id"].to(device),
                    target_batch["time_index"].to(device),
                    monotonic_margin=target_lifecycle_monotonic_margin,
                )
            else:
                target_lifecycle_monotonic_loss = tpred.sum() * 0.0
                target_lifecycle_smooth_loss = tpred.sum() * 0.0
            loss = (
                source_supervised_weight * pred_loss
                + target_supervised_weight * target_pred_loss
                + effective_lmmd_weight * lmmd
                + target_sequence_monotonic_weight * target_sequence_monotonic_loss
                + target_sequence_smooth_weight * target_sequence_smooth_loss
                + target_lifecycle_monotonic_weight * target_lifecycle_monotonic_loss
                + target_lifecycle_smooth_weight * target_lifecycle_smooth_loss
                + source_stage_aux_weight * source_stage_aux_loss
                + target_stage_aux_weight * target_stage_aux_loss
                + prototype_alignment_weight * prototype_alignment_loss
            )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pred_losses.append(float(pred_loss.detach().cpu()))
            target_pred_losses.append(float(target_pred_loss.detach().cpu()))
            alignment_losses.append(float(lmmd.detach().cpu()))
            target_sequence_monotonic_losses.append(float(target_sequence_monotonic_loss.detach().cpu()))
            target_sequence_smooth_losses.append(float(target_sequence_smooth_loss.detach().cpu()))
            target_lifecycle_monotonic_losses.append(float(target_lifecycle_monotonic_loss.detach().cpu()))
            target_lifecycle_smooth_losses.append(float(target_lifecycle_smooth_loss.detach().cpu()))
            source_stage_aux_losses.append(float(source_stage_aux_loss.detach().cpu()))
            target_stage_aux_losses.append(float(target_stage_aux_loss.detach().cpu()))
            prototype_alignment_losses.append(float(prototype_alignment_loss.detach().cpu()))
        val_pred = predict(model, target_val, device)
        val_metrics = rul_metrics_with_time(val_pred["y_true"], val_pred["y_pred"], val_pred["unit_id"], val_pred["time_index"])
        val_rmse = float(val_metrics["rmse"])
        if checkpoint_metric in {"weighted", "weighted_sum", "combined"}:
            if not isinstance(checkpoint_weights, dict) or not checkpoint_weights:
                raise ValueError("checkpoint_metric_weights must be a non-empty mapping for weighted transfer checkpointing")
            current_val_metric = float(
                sum(float(weight) * float(val_metrics[metric_name]) for metric_name, weight in checkpoint_weights.items())
            )
        else:
            metric_key = "rmse" if checkpoint_metric in {"target_val_rmse", "val_rmse"} else checkpoint_metric
            current_val_metric = float(val_metrics[metric_key])
        row = {
            "epoch": epoch,
            "transfer_loss": float(np.mean(losses)),
            "source_pred_loss": float(np.mean(pred_losses)),
            "target_pred_loss": float(np.mean(target_pred_losses)),
            "alignment_loss": float(np.mean(alignment_losses)),
            "target_sequence_monotonic_loss": float(np.mean(target_sequence_monotonic_losses)),
            "target_sequence_smooth_loss": float(np.mean(target_sequence_smooth_losses)),
            "target_lifecycle_monotonic_loss": float(np.mean(target_lifecycle_monotonic_losses)),
            "target_lifecycle_smooth_loss": float(np.mean(target_lifecycle_smooth_losses)),
            "source_stage_aux_loss": float(np.mean(source_stage_aux_losses)),
            "target_stage_aux_loss": float(np.mean(target_stage_aux_losses)),
            "prototype_alignment_loss": float(np.mean(prototype_alignment_losses)),
            "lmmd_weight_effective": float(effective_lmmd_weight),
            "target_sequence_monotonic_weight": float(target_sequence_monotonic_weight),
            "target_sequence_smooth_weight": float(target_sequence_smooth_weight),
            "target_lifecycle_monotonic_weight": float(target_lifecycle_monotonic_weight),
            "target_lifecycle_smooth_weight": float(target_lifecycle_smooth_weight),
            "source_stage_aux_weight": float(source_stage_aux_weight),
            "target_stage_aux_weight": float(target_stage_aux_weight),
            "prototype_alignment_weight": float(prototype_alignment_weight),
            "prototype_min_confidence": float(prototype_min_confidence),
            "source_supervised_weight": float(source_supervised_weight),
            "target_supervised_weight": float(target_supervised_weight),
            "transfer_steps": float(transfer_steps),
            "target_val_rmse": val_rmse,
            "val_checkpoint_metric": current_val_metric,
            **{f"target_val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(row)
        if epoch >= checkpoint_min_epoch and _is_better_checkpoint(
            current_val_metric,
            best_val_metric,
            checkpoint_mode,
            checkpoint_min_delta,
        ):
            best_val_metric = current_val_metric
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = predict(model, target_val, device)
    test_pred = predict(model, target_test, device)
    eval_cfg = config.get("evaluation", {})
    time_calibration: dict[str, Any] | None = None
    if bool(eval_cfg.get("calibrate_time_aware", False)):
        time_calibration = _fit_time_aware_output_calibration(val_pred, eval_cfg.get("time_aware", {}))
        clip_range = (
            float(eval_cfg.get("clip_min", 0.0)),
            float(eval_cfg.get("clip_max", 1.0)),
        )
        _apply_time_aware_output_calibration(val_pred, time_calibration, clip_range=clip_range)
        _apply_time_aware_output_calibration(test_pred, time_calibration, clip_range=clip_range)
    metrics = rul_metrics_with_time(test_pred["y_true"], test_pred["y_pred"], test_pred["unit_id"], test_pred["time_index"])
    metrics["best_epoch"] = float(best_epoch)
    metrics["best_target_val_rmse"] = float(best_val_metric)
    if time_calibration is not None:
        metrics["time_aware_calibration_enabled"] = 1.0
        metrics["time_aware_calibration_degree"] = float(time_calibration["degree"])
        metrics["time_aware_calibration_ridge"] = float(time_calibration["ridge"])
        metrics["time_aware_calibration_time_min"] = float(time_calibration["time_min"])
        metrics["time_aware_calibration_time_scale"] = float(time_calibration["time_scale"])
        metrics["time_aware_calibration_coef"] = time_calibration["coef"].tolist()
        metrics["calibration_source"] = "target_validation_only"
        metrics["time_aware_calibration_features"] = "y_pred,time_index"
    if config.get("conformal", {}).get("enabled", False):
        cp = SplitConformalRegressor(alpha=float(config["conformal"].get("alpha", 0.1)))
        cp.fit(val_pred["y_true"], val_pred["y_pred"])
        lower, upper = cp.predict_interval(test_pred["y_pred"])
        test_pred["lower"] = lower
        test_pred["upper"] = upper
        metrics["conformal_coverage"] = cp.coverage(test_pred["y_true"], lower, upper)
        metrics["conformal_avg_width"] = cp.average_width(lower, upper)
        metrics["conformal_q"] = float(cp.quantile_)
    metrics["num_parameters"] = float(count_parameters(model))
    metrics.update(benchmark_inference_latency(model, target_test, device))
    save_prediction_csv(val_pred, output_dir / "predictions_val.csv")
    save_prediction_csv(test_pred, output_dir / "predictions_test.csv")
    save_json(collect_environment_info(), output_dir / "env_info.json")
    torch.save(model.state_dict(), output_dir / "transfer_model.pt")
    save_json(
        {"source_pretrain_history": pretrain_history, "transfer_history": history, "final_metrics": metrics},
        output_dir / "transfer_metrics.json",
    )
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics = train_transfer(config)
    print("Final transfer metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
