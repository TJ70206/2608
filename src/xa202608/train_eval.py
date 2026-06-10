from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from xa202608.conformal import SplitConformalRegressor
from xa202608.experiment_io import benchmark_inference_latency, collect_environment_info, save_prediction_csv
from xa202608.metrics import rul_metrics_with_time
from xa202608.plotting import plot_evaluation_figures_csv
from xa202608.utils import count_parameters, save_json


def build_loss(name: str, smooth_l1_beta: float = 1.0) -> nn.Module:
    name = str(name).lower()
    if name in {"smooth_l1", "smoothl1", "huber"}:
        return nn.SmoothL1Loss(beta=float(smooth_l1_beta), reduction="none")
    if name in {"mse", "mse_loss"}:
        return nn.MSELoss(reduction="none")
    if name in {"l1", "mae"}:
        return nn.L1Loss(reduction="none")
    raise ValueError(f"unknown loss: {name}")


def build_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float | None = None,
    stage_loss_weight: float = 0.0,
    sequence_tail_weight: float = 1.0,
    sequence_tail_k: int = 1,
    late_stage_weight: float = 1.0,
    late_stage_threshold: int = 2,
    late_prediction_weight: float = 1.0,
) -> float:
    model.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        stage = batch["stage"].to(device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        is_sequence_batch = y.ndim == 3
        use_stage_aux = (not is_sequence_batch) and stage_loss_weight > 0.0 and getattr(model, "stage_head", None) is not None
        if is_sequence_batch:
            pred = model(x, return_sequence=True)
            stage_logits = None
        elif use_stage_aux:
            pred, _, stage_logits = model(x, return_aux=True)
        else:
            pred = model(x)
            stage_logits = None
        reg_loss = criterion(pred, y)
        if is_sequence_batch:
            reg_loss = reg_loss.squeeze(-1)
        else:
            reg_loss = reg_loss.reshape(-1)
        if is_sequence_batch and sequence_tail_weight != 1.0:
            tail_k = max(1, int(sequence_tail_k))
            if mask is not None:
                positions = torch.arange(reg_loss.size(1), device=device).unsqueeze(0)
                lengths = mask.long().sum(dim=1, keepdim=True)
                tail_mask = (positions >= (lengths - tail_k).clamp_min(0)) & (positions < lengths)
            else:
                positions = torch.arange(reg_loss.size(1), device=device).unsqueeze(0)
                tail_mask = positions >= max(0, reg_loss.size(1) - tail_k)
            tail_weight = torch.where(
                tail_mask,
                torch.as_tensor(float(sequence_tail_weight), device=device, dtype=reg_loss.dtype),
                torch.ones((), device=device, dtype=reg_loss.dtype),
            )
            reg_loss = reg_loss * tail_weight
        if late_stage_weight != 1.0:
            sample_weight = torch.where(stage >= late_stage_threshold, late_stage_weight, 1.0).to(reg_loss.dtype)
            reg_loss = reg_loss * sample_weight
        if late_prediction_weight != 1.0:
            pred_for_weight = pred.detach().squeeze(-1) if is_sequence_batch else pred.detach().reshape(-1)
            target_for_weight = y.squeeze(-1) if is_sequence_batch else y.reshape(-1)
            prediction_weight = torch.where(
                pred_for_weight > target_for_weight,
                late_prediction_weight,
                1.0,
            ).to(reg_loss.dtype)
            reg_loss = reg_loss * prediction_weight
        if mask is not None:
            loss = (reg_loss * mask.to(reg_loss.dtype)).sum() / mask.sum().clamp_min(1).to(reg_loss.dtype)
        else:
            loss = reg_loss.mean()
        if use_stage_aux and stage_logits is not None:
            loss = loss + stage_loss_weight * F.cross_entropy(stage_logits, stage)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    preds, targets, unit_ids, time_indices, stages = [], [], [], [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["x"].to(device)
        if "mask" in batch:
            pred = model(x, return_sequence=True).detach().cpu().numpy()
            mask = batch["mask"].numpy().astype(bool)
            preds.append(pred.squeeze(-1)[mask])
            targets.append(batch["y"].numpy().squeeze(-1)[mask])
            unit_ids.append(batch["unit_id"].numpy()[mask])
            time_indices.append(batch["time_index"].numpy()[mask])
            stages.append(batch["stage"].numpy()[mask])
        else:
            pred = model(x).detach().cpu().numpy()
            preds.append(pred)
            targets.append(batch["y"].numpy())
            unit_ids.append(batch["unit_id"].numpy())
            time_indices.append(batch["time_index"].numpy())
            stages.append(batch["stage"].numpy())
    return {
        "y_pred": np.concatenate(preds, axis=0).reshape(-1),
        "y_true": np.concatenate(targets, axis=0).reshape(-1),
        "unit_id": np.concatenate(unit_ids, axis=0).reshape(-1),
        "time_index": np.concatenate(time_indices, axis=0).reshape(-1),
        "stage": np.concatenate(stages, axis=0).reshape(-1),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    pred = predict(model, loader, device)
    return rul_metrics_with_time(pred["y_true"], pred["y_pred"], pred["unit_id"], pred["time_index"])


def _postprocess_predictions(pred: dict[str, np.ndarray], bias: float = 0.0, clip_range: tuple[float, float] | None = None) -> None:
    values = pred["y_pred"].astype(np.float64) + float(bias)
    if clip_range is not None:
        values = np.clip(values, float(clip_range[0]), float(clip_range[1]))
    pred["y_pred"] = values


def _last_window_objective(metrics: dict[str, float], objective: str, score_weight: float) -> float:
    objective = str(objective).lower()
    if objective == "rmse":
        return float(metrics["last_window_rmse"])
    if objective == "score":
        return float(metrics["last_window_nasa_score"])
    if objective in {"combined", "weighted", "weighted_sum"}:
        return float(metrics["last_window_rmse"]) + float(score_weight) * float(metrics["last_window_nasa_score"])
    raise ValueError(f"unknown bias calibration objective: {objective}")


def _fit_prediction_bias(
    val_pred: dict[str, np.ndarray],
    bias_cfg: dict,
    clip_range: tuple[float, float] | None,
) -> tuple[float, dict[str, float]]:
    min_bias = float(bias_cfg.get("min", -20.0))
    max_bias = float(bias_cfg.get("max", 20.0))
    step = float(bias_cfg.get("step", 0.5))
    if step <= 0.0:
        raise ValueError("bias calibration step must be positive")
    objective = str(bias_cfg.get("objective", "combined"))
    score_weight = float(bias_cfg.get("score_weight", 0.003))
    best_bias = 0.0
    best_objective = float("inf")
    best_metrics: dict[str, float] = {}
    base_pred = val_pred["y_pred"].astype(np.float64)
    for bias in np.arange(min_bias, max_bias + 0.5 * step, step):
        y_pred = base_pred + float(bias)
        if clip_range is not None:
            y_pred = np.clip(y_pred, float(clip_range[0]), float(clip_range[1]))
        metrics = rul_metrics_with_time(val_pred["y_true"], y_pred, val_pred["unit_id"], val_pred["time_index"])
        value = _last_window_objective(metrics, objective, score_weight)
        if value < best_objective:
            best_objective = value
            best_bias = float(bias)
            best_metrics = metrics
    return best_bias, best_metrics


def _last_window_prediction_frame(pred: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit_ids = pred["unit_id"]
    time_index = pred["time_index"]
    selected: list[int] = []
    for unit_id in np.unique(unit_ids):
        unit_indices = np.where(unit_ids == unit_id)[0]
        selected.append(int(unit_indices[np.argmax(time_index[unit_indices])]))
    indices = np.asarray(selected, dtype=np.int64)
    return pred["y_true"][indices].astype(np.float64), pred["y_pred"][indices].astype(np.float64), indices


def _fit_piecewise_prediction_offsets(val_pred: dict[str, np.ndarray], calib_cfg: dict) -> dict[str, np.ndarray]:
    edges = np.asarray(calib_cfg.get("edges", [0.0, 30.0, 60.0, 90.0, 125.0, 200.0]), dtype=np.float64)
    shrink = float(calib_cfg.get("shrink", 1.0))
    if edges.ndim != 1 or edges.size < 3:
        raise ValueError("piecewise calibration edges must contain at least three values")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("piecewise calibration edges must be strictly increasing")
    y_true, y_pred, _ = _last_window_prediction_frame(val_pred)
    residual = y_true - y_pred
    offsets: list[float] = []
    default_offset = float(np.median(residual)) if residual.size else 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (y_pred >= low) & (y_pred < high)
        offsets.append(float(np.median(residual[mask])) if np.any(mask) else default_offset)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {"centers": centers, "offsets": np.asarray(offsets, dtype=np.float64) * shrink, "edges": edges, "shrink": np.asarray([shrink], dtype=np.float64)}


def _apply_piecewise_prediction_offsets(pred: dict[str, np.ndarray], calibration: dict[str, np.ndarray]) -> None:
    values = pred["y_pred"].astype(np.float64)
    offsets = np.interp(
        values,
        calibration["centers"],
        calibration["offsets"],
        left=float(calibration["offsets"][0]),
        right=float(calibration["offsets"][-1]),
    )
    pred["y_pred"] = values + offsets


def _pava_increasing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    sorted_x = x[order].astype(np.float64)
    sorted_y = y[order].astype(np.float64)
    block_x: list[float] = []
    block_y: list[float] = []
    block_weight: list[float] = []
    for x_value, y_value in zip(sorted_x, sorted_y):
        block_x.append(float(x_value))
        block_y.append(float(y_value))
        block_weight.append(1.0)
        while len(block_y) >= 2 and block_y[-2] > block_y[-1]:
            merged_weight = block_weight[-2] + block_weight[-1]
            merged_x = (block_x[-2] * block_weight[-2] + block_x[-1] * block_weight[-1]) / merged_weight
            merged_y = (block_y[-2] * block_weight[-2] + block_y[-1] * block_weight[-1]) / merged_weight
            block_x[-2:] = [float(merged_x)]
            block_y[-2:] = [float(merged_y)]
            block_weight[-2:] = [float(merged_weight)]
    return np.asarray(block_x, dtype=np.float64), np.asarray(block_y, dtype=np.float64)


def _fit_isotonic_prediction_calibration(val_pred: dict[str, np.ndarray], calib_cfg: dict) -> dict[str, np.ndarray]:
    y_true, y_pred, _ = _last_window_prediction_frame(val_pred)
    target_min = calib_cfg.get("target_min")
    target_max = calib_cfg.get("target_max")
    if target_min is not None or target_max is not None:
        y_true = np.clip(
            y_true,
            -np.inf if target_min is None else float(target_min),
            np.inf if target_max is None else float(target_max),
        )
    knots_x, knots_y = _pava_increasing(y_pred, y_true)
    if knots_x.size == 1:
        knots_x = np.asarray([knots_x[0] - 1.0, knots_x[0] + 1.0], dtype=np.float64)
        knots_y = np.asarray([knots_y[0], knots_y[0]], dtype=np.float64)
    return {"knots_x": knots_x, "knots_y": knots_y}


def _apply_isotonic_prediction_calibration(pred: dict[str, np.ndarray], calibration: dict[str, np.ndarray]) -> None:
    values = pred["y_pred"].astype(np.float64)
    pred["y_pred"] = np.interp(
        values,
        calibration["knots_x"],
        calibration["knots_y"],
        left=float(calibration["knots_y"][0]),
        right=float(calibration["knots_y"][-1]),
    )


def fit_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: Path,
) -> dict[str, float]:
    train_cfg = config["train"]
    model.to(device)
    criterion = build_loss(str(train_cfg.get("loss", "smooth_l1")), float(train_cfg.get("smooth_l1_beta", 1.0)))
    optimizer = build_optimizer(model, train_cfg)
    checkpoint_metric = str(train_cfg.get("checkpoint_metric", "rmse"))
    checkpoint_weights = train_cfg.get("checkpoint_metric_weights", {})
    checkpoint_min_epoch = int(train_cfg.get("checkpoint_min_epoch", 1))
    checkpoint_mode = str(train_cfg.get("checkpoint_mode", "min")).lower()
    if checkpoint_mode not in {"min", "max"}:
        raise ValueError("checkpoint_mode must be 'min' or 'max'")
    best_val_metric = float("inf") if checkpoint_mode == "min" else -float("inf")
    best_state = None
    history = []
    patience = int(train_cfg.get("early_stopping_patience", 0))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip_norm=float(train_cfg.get("grad_clip_norm", 0.0)) or None,
            stage_loss_weight=float(train_cfg.get("stage_loss_weight", 0.0)),
            sequence_tail_weight=float(train_cfg.get("sequence_tail_weight", 1.0)),
            sequence_tail_k=int(train_cfg.get("sequence_tail_k", 1)),
            late_stage_weight=float(train_cfg.get("late_stage_weight", 1.0)),
            late_stage_threshold=int(train_cfg.get("late_stage_threshold", 2)),
            late_prediction_weight=float(train_cfg.get("late_prediction_weight", 1.0)),
        )
        val_metrics = evaluate(model, val_loader, device)
        if checkpoint_metric in {"weighted", "weighted_sum", "combined"}:
            if not isinstance(checkpoint_weights, dict) or not checkpoint_weights:
                raise ValueError("checkpoint_metric_weights must be a non-empty mapping for weighted checkpointing")
            current_val_metric = float(
                sum(float(weight) * float(val_metrics[metric_name]) for metric_name, weight in checkpoint_weights.items())
            )
        else:
            current_val_metric = float(val_metrics[checkpoint_metric])
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        row["val_checkpoint_metric"] = current_val_metric
        history.append(row)
        print(row)
        if epoch >= checkpoint_min_epoch:
            is_better = (
                current_val_metric < best_val_metric - min_delta
                if checkpoint_mode == "min"
                else current_val_metric > best_val_metric + min_delta
            )
            if is_better:
                best_val_metric = current_val_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = predict(model, val_loader, device)
    test_pred = predict(model, test_loader, device)
    eval_cfg = config.get("evaluation", {})
    clip_range = None
    if bool(eval_cfg.get("clip_predictions", False)):
        clip_range = (
            float(eval_cfg.get("clip_min", 0.0)),
            float(eval_cfg.get("clip_max", config.get("data", {}).get("test_max_rul", config.get("data", {}).get("max_rul", 125)))),
        )
    prediction_bias = 0.0
    bias_metrics: dict[str, float] = {}
    isotonic_calibration: dict[str, np.ndarray] | None = None
    if bool(eval_cfg.get("calibrate_isotonic", False)):
        isotonic_calibration = _fit_isotonic_prediction_calibration(val_pred, eval_cfg.get("isotonic", {}))
        _apply_isotonic_prediction_calibration(val_pred, isotonic_calibration)
        _apply_isotonic_prediction_calibration(test_pred, isotonic_calibration)
    piecewise_calibration: dict[str, np.ndarray] | None = None
    if bool(eval_cfg.get("calibrate_piecewise", False)):
        piecewise_calibration = _fit_piecewise_prediction_offsets(val_pred, eval_cfg.get("piecewise", {}))
        _apply_piecewise_prediction_offsets(val_pred, piecewise_calibration)
        _apply_piecewise_prediction_offsets(test_pred, piecewise_calibration)
    if bool(eval_cfg.get("calibrate_bias", False)):
        prediction_bias, bias_metrics = _fit_prediction_bias(val_pred, eval_cfg.get("bias", {}), clip_range)
    _postprocess_predictions(val_pred, bias=prediction_bias, clip_range=clip_range)
    _postprocess_predictions(test_pred, bias=prediction_bias, clip_range=clip_range)
    test_metrics = rul_metrics_with_time(test_pred["y_true"], test_pred["y_pred"], test_pred["unit_id"], test_pred["time_index"])
    test_metrics["prediction_bias"] = float(prediction_bias)
    if clip_range is not None:
        test_metrics["prediction_clip_min"] = float(clip_range[0])
        test_metrics["prediction_clip_max"] = float(clip_range[1])
    if piecewise_calibration is not None:
        test_metrics["piecewise_calibration_enabled"] = 1.0
        test_metrics["piecewise_calibration_edges"] = piecewise_calibration["edges"].tolist()
        test_metrics["piecewise_calibration_offsets"] = piecewise_calibration["offsets"].tolist()
        test_metrics["piecewise_calibration_shrink"] = float(piecewise_calibration["shrink"][0])
    if isotonic_calibration is not None:
        test_metrics["isotonic_calibration_enabled"] = 1.0
        test_metrics["isotonic_calibration_knots_x"] = isotonic_calibration["knots_x"].tolist()
        test_metrics["isotonic_calibration_knots_y"] = isotonic_calibration["knots_y"].tolist()
    for key, value in bias_metrics.items():
        test_metrics[f"bias_calib_val_{key}"] = value
    if config.get("conformal", {}).get("enabled", False):
        cp = SplitConformalRegressor(alpha=float(config["conformal"].get("alpha", 0.1)))
        cp.fit(val_pred["y_true"], val_pred["y_pred"])
        lower, upper = cp.predict_interval(test_pred["y_pred"])
        test_pred["lower"] = lower
        test_pred["upper"] = upper
        test_metrics["conformal_coverage"] = cp.coverage(test_pred["y_true"], lower, upper)
        test_metrics["conformal_avg_width"] = cp.average_width(lower, upper)
        test_metrics["conformal_q"] = float(cp.quantile_)
    test_metrics["num_parameters"] = float(count_parameters(model))
    test_metrics.update(benchmark_inference_latency(model, test_loader, device))
    output_dir.mkdir(parents=True, exist_ok=True)
    val_prediction_path = output_dir / "predictions_val.csv"
    test_prediction_path = output_dir / "predictions_test.csv"
    save_prediction_csv(val_pred, val_prediction_path)
    save_prediction_csv(test_pred, test_prediction_path)
    test_metrics["plot_paths"] = [
        str(path)
        for path in plot_evaluation_figures_csv(
            test_prediction_path,
            output_dir,
            prefix="test",
            calibration_csv=val_prediction_path,
        )
    ]
    save_json(collect_environment_info(), output_dir / "env_info.json")
    save_json({"history": history, "test_metrics": test_metrics}, output_dir / "metrics.json")
    torch.save(model.state_dict(), output_dir / "model.pt")
    return test_metrics
