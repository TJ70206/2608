from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _validate_prediction_frame(df: pd.DataFrame) -> None:
    required = {"unit_id", "time_index", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing required columns: {sorted(missing)}")


def _last_window_frame(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("unit_id")["time_index"].idxmax()
    last_df = df.loc[idx].copy().sort_values("unit_id")
    last_df["error"] = last_df["y_pred"] - last_df["y_true"]
    last_df["abs_error"] = last_df["error"].abs()
    return last_df


def _rmse(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")


def _absolute_error_quantiles(df: pd.DataFrame, levels: list[float]) -> dict[float, float]:
    errors = np.abs(df["y_pred"].to_numpy(dtype=float) - df["y_true"].to_numpy(dtype=float))
    if errors.size == 0:
        return {level: float("nan") for level in levels}
    return {level: float(np.quantile(errors, level)) for level in levels}


def _select_representative_unit(df: pd.DataFrame, last_df: pd.DataFrame) -> int:
    available_units = {int(unit_id) for unit_id in df["unit_id"].unique()}
    if 76 in available_units:
        return 76
    candidates = []
    for unit_id, unit_df in df.groupby("unit_id"):
        unit_df = unit_df.sort_values("time_index")
        y_true = unit_df["y_true"].to_numpy(dtype=float)
        onset_positions = np.flatnonzero(y_true < 125.0)
        if len(onset_positions) == 0:
            continue
        onset = int(onset_positions[0])
        candidates.append(
            {
                "unit_id": int(unit_id),
                "length": int(len(unit_df)),
                "last_true": float(y_true[-1]),
                "onset_distance": abs(onset - 80),
            }
        )
    if candidates:
        candidates_df = pd.DataFrame(candidates)
        candidates_df = candidates_df.sort_values(["onset_distance", "last_true", "length"], ascending=[True, True, False])
        return int(candidates_df.iloc[0]["unit_id"])
    return int(last_df.sort_values("abs_error", ascending=False).iloc[0]["unit_id"])


def plot_evaluation_figures_csv(
    prediction_csv: str | Path,
    output_dir: str | Path,
    prefix: str = "test",
    unit_id: int | None = None,
    calibration_csv: str | Path | None = None,
) -> list[Path]:
    csv_path = Path(prediction_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    _validate_prediction_frame(df)
    last_df = _last_window_frame(df)
    units = last_df["unit_id"].to_numpy()
    errors = last_df["error"].to_numpy()
    rmse_value = _rmse(errors)
    mae_value = float(np.mean(np.abs(errors))) if len(errors) else float("nan")
    saved: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(units, last_df["y_true"], marker="o", linewidth=1.5, label="True RUL")
    ax.plot(units, last_df["y_pred"], marker="o", linewidth=1.5, label="Predicted RUL")
    ax.set_title("Last-window true vs predicted RUL")
    ax.set_xlabel("Unit ID")
    ax.set_ylabel("RUL")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.axhline(0.0, color="tab:red", linewidth=1.2)
    ax.vlines(units, 0.0, errors, color="tab:blue", alpha=0.75)
    ax.scatter(units, errors, s=20, color="tab:blue")
    ax.set_title("Last-window residuals")
    ax.set_xlabel("Unit ID")
    ax.set_ylabel("Prediction error")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(last_df["y_true"], last_df["y_pred"], s=28, alpha=0.8)
    min_value = float(min(last_df["y_true"].min(), last_df["y_pred"].min()))
    max_value = float(max(last_df["y_true"].max(), last_df["y_pred"].max()))
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="tab:red", label="Ideal")
    ax.set_title("Last-window scatter")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.hist(errors, bins=20, alpha=0.75, color="tab:purple", edgecolor="white")
    ax.axvline(0.0, color="tab:red", linewidth=1.2)
    ax.set_title(f"Residual distribution | RMSE={rmse_value:.3f}, MAE={mae_value:.3f}")
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    summary_path = out_dir / f"{prefix}_last_window_summary.png"
    fig.savefig(summary_path, dpi=180)
    plt.close(fig)
    saved.append(summary_path)

    selected_unit = int(unit_id) if unit_id is not None else _select_representative_unit(df, last_df)
    unit_df = df[df["unit_id"] == selected_unit].sort_values("time_index").copy()
    unit_df["error"] = unit_df["y_pred"] - unit_df["y_true"]
    fig, ax = plt.subplots(figsize=(11, 8))
    x_values = unit_df["time_index"].to_numpy()
    pred_values = unit_df["y_pred"].to_numpy(dtype=float)
    interval_levels = [0.95, 0.90, 0.75, 0.50, 0.25, 0.10, 0.05]
    calibration_path = Path(calibration_csv) if calibration_csv is not None else None
    quantiles: dict[float, float]
    if calibration_path is not None and calibration_path.exists():
        calibration_df = pd.read_csv(calibration_path)
        _validate_prediction_frame(calibration_df)
        quantiles = _absolute_error_quantiles(calibration_df, interval_levels)
    elif "lower" in unit_df.columns and "upper" in unit_df.columns:
        q = float(np.nanmean((unit_df["upper"].to_numpy(dtype=float) - unit_df["lower"].to_numpy(dtype=float)) / 2.0))
        quantiles = {level: q * level / 0.9 for level in interval_levels}
    else:
        quantiles = _absolute_error_quantiles(df, interval_levels)
    for level in interval_levels:
        q = quantiles[level]
        if not np.isfinite(q):
            continue
        alpha = 0.08 + 0.35 * (1.0 - level)
        ax.fill_between(x_values, pred_values - q, pred_values + q, color="forestgreen", alpha=alpha, label=f"{int(level * 100)}% interval")
    ax.plot(unit_df["time_index"], unit_df["y_true"], linestyle=":", marker=".", color="black", linewidth=1.6, markersize=3.5, label="True RUL")
    ax.plot(unit_df["time_index"], unit_df["y_pred"], color="red", linewidth=1.8, label="Predicted RUL")
    ax.set_title("Prediction results of intervals with different confidence levels")
    ax.set_xlabel("Time index")
    ax.set_ylabel("RUL")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    unit_path = out_dir / f"{prefix}_unit_{selected_unit}_trajectory.png"
    fig.savefig(unit_path, dpi=180)
    plt.close(fig)
    saved.append(unit_path)
    return saved


def plot_predictions_csv(prediction_csv: str | Path, output_dir: str | Path, max_units: int = 4) -> list[Path]:
    csv_path = Path(prediction_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    _validate_prediction_frame(df)
    saved: list[Path] = []
    for idx, unit_id in enumerate(sorted(df["unit_id"].unique())):
        if idx >= max_units:
            break
        unit_df = df[df["unit_id"] == unit_id].sort_values("time_index")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(unit_df["time_index"], unit_df["y_true"], label="True RUL", linewidth=2)
        ax.plot(unit_df["time_index"], unit_df["y_pred"], label="Predicted RUL", linewidth=2)
        if "lower" in unit_df.columns and "upper" in unit_df.columns:
            ax.fill_between(
                unit_df["time_index"].to_numpy(),
                unit_df["lower"].to_numpy(),
                unit_df["upper"].to_numpy(),
                alpha=0.2,
                label="Conformal interval",
            )
        ax.set_title(f"Unit {unit_id} RUL prediction")
        ax.set_xlabel("Time index")
        ax.set_ylabel("RUL")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        output_path = out_dir / f"prediction_unit_{unit_id}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        saved.append(output_path)
    return saved
