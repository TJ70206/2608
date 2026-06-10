from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    diff = y_pred - y_true
    score = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(score))


def relative_accuracy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.mean(1.0 - np.minimum(np.abs(y_true - y_pred) / np.maximum(y_true, eps), 1.0)))


def alpha_lambda_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    life_progress: np.ndarray | None = None,
    alpha: float = 0.2,
    lambdas: tuple[float, ...] = (0.5, 0.8),
    tolerance: float = 0.05,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if life_progress is None:
        life_progress = np.linspace(0.0, 1.0, len(y_true))
    life_progress = np.asarray(life_progress, dtype=np.float64).reshape(-1)
    out: dict[str, float] = {}
    for lam in lambdas:
        mask = np.abs(life_progress - lam) <= tolerance
        if not mask.any():
            out[f"alpha_lambda_{lam:g}"] = float("nan")
            continue
        lower = y_true[mask] * (1.0 - alpha)
        upper = y_true[mask] * (1.0 + alpha)
        inside = (y_pred[mask] >= lower) & (y_pred[mask] <= upper)
        out[f"alpha_lambda_{lam:g}"] = float(np.mean(inside))
    return out


def prognostic_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_index: np.ndarray,
    alpha: float = 0.2,
) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    time_index = np.asarray(time_index, dtype=np.float64).reshape(-1)
    if len(y_true) == 0:
        return float("nan")
    within = np.abs(y_pred - y_true) <= alpha * np.maximum(y_true, 1e-8)
    for idx in range(len(within)):
        if within[idx] and within[idx:].all():
            eol_time = time_index[idx] + y_true[idx]
            return float(eol_time - time_index[idx])
    return 0.0


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "ra": relative_accuracy(y_true, y_pred),
    }


def last_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit_id: np.ndarray,
    time_index: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    unit_id = np.asarray(unit_id).reshape(-1)
    time_index = np.asarray(time_index, dtype=np.float64).reshape(-1)
    selected = []
    for uid in np.unique(unit_id):
        mask = unit_id == uid
        unit_indices = np.where(mask)[0]
        selected.append(unit_indices[np.argmax(time_index[unit_indices])])
    selected_idx = np.asarray(selected, dtype=np.int64)
    metrics = regression_metrics(y_true[selected_idx], y_pred[selected_idx])
    return {
        "last_window_rmse": metrics["rmse"],
        "last_window_mae": metrics["mae"],
        "last_window_nasa_score": metrics["nasa_score"],
        "last_window_ra": metrics["ra"],
        "last_window_num_units": float(len(selected_idx)),
    }


def last_k_average_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit_id: np.ndarray,
    time_index: np.ndarray,
    k: int = 5,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    unit_id = np.asarray(unit_id).reshape(-1)
    time_index = np.asarray(time_index, dtype=np.float64).reshape(-1)
    k = max(1, int(k))
    selected_true = []
    averaged_pred = []
    used_counts = []
    for uid in np.unique(unit_id):
        mask = unit_id == uid
        unit_indices = np.where(mask)[0]
        order = np.argsort(time_index[unit_indices])
        ordered_idx = unit_indices[order]
        last_idx = ordered_idx[-1]
        avg_idx = ordered_idx[-k:]
        selected_true.append(y_true[last_idx])
        averaged_pred.append(float(np.mean(y_pred[avg_idx])))
        used_counts.append(len(avg_idx))
    selected_true_arr = np.asarray(selected_true, dtype=np.float64)
    averaged_pred_arr = np.asarray(averaged_pred, dtype=np.float64)
    metrics = regression_metrics(selected_true_arr, averaged_pred_arr)
    return {
        f"last_{k}_avg_rmse": metrics["rmse"],
        f"last_{k}_avg_mae": metrics["mae"],
        f"last_{k}_avg_nasa_score": metrics["nasa_score"],
        f"last_{k}_avg_ra": metrics["ra"],
        f"last_{k}_avg_num_units": float(len(selected_true_arr)),
        f"last_{k}_avg_mean_windows": float(np.mean(used_counts)) if used_counts else float("nan"),
    }


def rul_metrics_with_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit_id: np.ndarray,
    time_index: np.ndarray,
    alpha: float = 0.2,
    lambdas: tuple[float, ...] = (0.5, 0.8),
) -> dict[str, float]:
    metrics = regression_metrics(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    unit_id = np.asarray(unit_id).reshape(-1)
    time_index = np.asarray(time_index, dtype=np.float64).reshape(-1)
    life_progress = np.zeros_like(y_true, dtype=np.float64)
    ph_values = []
    for uid in np.unique(unit_id):
        mask = unit_id == uid
        order = np.argsort(time_index[mask])
        idx = np.where(mask)[0][order]
        unit_true = y_true[idx]
        unit_pred = y_pred[idx]
        unit_time = time_index[idx]
        start_rul = max(float(unit_true[0]), 1.0)
        life_progress[idx] = np.clip(1.0 - unit_true / start_rul, 0.0, 1.0)
        ph_values.append(prognostic_horizon(unit_true, unit_pred, unit_time, alpha=alpha))
    metrics.update(alpha_lambda_accuracy(y_true, y_pred, life_progress, alpha=alpha, lambdas=lambdas))
    metrics.update(last_window_metrics(y_true, y_pred, unit_id, time_index))
    metrics.update(last_k_average_metrics(y_true, y_pred, unit_id, time_index, k=5))
    metrics["ph_mean"] = float(np.nanmean(ph_values)) if ph_values else float("nan")
    return metrics
