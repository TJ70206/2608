"""Run validation-only output calibration ablations on existing transfer outputs.

The script does not retrain any model. It reads predictions_val.csv and
predictions_test.csv from a raw transfer run, fits lightweight ridge
calibration maps on the target validation predictions, and evaluates the
calibrated predictions on the target test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xa202608.experiment_io import save_prediction_csv
from xa202608.metrics import rul_metrics_with_time
from xa202608.utils import save_json


METRIC_COLUMNS = [
    "rmse",
    "mae",
    "nasa_score",
    "ra",
    "alpha_lambda_0.5",
    "alpha_lambda_0.8",
    "last_window_rmse",
    "last_5_avg_rmse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate validation-only time-aware output calibration.")
    parser.add_argument("--first-input-dir", type=Path, default=PROJECT_ROOT / "outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p0005_c0p5_srcsup0p7_50e")
    parser.add_argument("--second-input-dir", type=Path, default=PROJECT_ROOT / "outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p001_c0p5_srcsup0p7_50e")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "competition_artifacts/03_results/tc_ablation")
    parser.add_argument("--degree", default=2, type=int)
    parser.add_argument("--ridge", default=0.01, type=float)
    parser.add_argument("--clip-min", default=0.0, type=float)
    parser.add_argument("--clip-max", default=1.0, type=float)
    return parser.parse_args()


def load_prediction_csv(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"unit_id", "time_index", "stage", "y_true", "y_pred"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "unit_id": row["unit_id"],
                    "time_index": float(row["time_index"]),
                    "stage": int(float(row["stage"])),
                    "y_true": float(row["y_true"]),
                    "y_pred": float(row["y_pred"]),
                }
            )
    if not rows:
        raise ValueError(f"prediction CSV is empty: {path}")
    return {
        "unit_id": np.asarray([row["unit_id"] for row in rows]),
        "time_index": np.asarray([row["time_index"] for row in rows], dtype=np.float64),
        "stage": np.asarray([row["stage"] for row in rows], dtype=np.int64),
        "y_true": np.asarray([row["y_true"] for row in rows], dtype=np.float64),
        "y_pred": np.asarray([row["y_pred"] for row in rows], dtype=np.float64),
    }


def clone_prediction(pred: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in pred.items()}


def normalized_time(pred: dict[str, np.ndarray], time_min: float, time_scale: float) -> np.ndarray:
    time_index = pred["time_index"].astype(np.float64).reshape(-1)
    return np.clip((time_index - float(time_min)) / float(time_scale), 0.0, 1.0)


def calibration_features(
    pred: dict[str, np.ndarray],
    feature_mode: str,
    degree: int,
    time_min: float,
    time_scale: float,
) -> np.ndarray:
    y_pred = pred["y_pred"].astype(np.float64).reshape(-1)
    t = normalized_time(pred, time_min=time_min, time_scale=time_scale)
    mode = feature_mode.lower()
    if mode == "y_pred":
        columns = [np.ones_like(y_pred), y_pred]
        if int(degree) >= 2:
            columns.append(y_pred**2)
    elif mode == "time":
        columns = [np.ones_like(y_pred), t]
        if int(degree) >= 2:
            columns.append(t**2)
    elif mode == "y_pred_time":
        columns = [np.ones_like(y_pred), y_pred, t]
        if int(degree) >= 2:
            columns.extend([y_pred**2, t**2, y_pred * t])
    else:
        raise ValueError("feature_mode must be y_pred, time, or y_pred_time")
    return np.vstack(columns).T


def fit_ridge_calibration(
    val_pred: dict[str, np.ndarray],
    feature_mode: str,
    degree: int,
    ridge: float,
) -> dict[str, Any]:
    if int(degree) not in {1, 2}:
        raise ValueError("degree must be 1 or 2")
    if float(ridge) < 0.0:
        raise ValueError("ridge must be non-negative")
    time_values = val_pred["time_index"].astype(np.float64)
    time_min = float(np.min(time_values))
    time_scale = max(float(np.max(time_values) - time_min), 1e-8)
    x = calibration_features(
        val_pred,
        feature_mode=feature_mode,
        degree=int(degree),
        time_min=time_min,
        time_scale=time_scale,
    )
    y = val_pred["y_true"].astype(np.float64).reshape(-1)
    regularizer = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
    return {
        "feature_mode": feature_mode,
        "degree": int(degree),
        "ridge": float(ridge),
        "time_min": time_min,
        "time_scale": time_scale,
        "coef": coef,
    }


def apply_calibration(
    pred: dict[str, np.ndarray],
    calibration: dict[str, Any],
    clip_range: tuple[float, float],
) -> dict[str, np.ndarray]:
    calibrated = clone_prediction(pred)
    x = calibration_features(
        calibrated,
        feature_mode=str(calibration["feature_mode"]),
        degree=int(calibration["degree"]),
        time_min=float(calibration["time_min"]),
        time_scale=float(calibration["time_scale"]),
    )
    calibrated["y_pred"] = np.clip(x @ calibration["coef"], float(clip_range[0]), float(clip_range[1]))
    return calibrated


def evaluate(pred: dict[str, np.ndarray]) -> dict[str, float]:
    return rul_metrics_with_time(pred["y_true"], pred["y_pred"], pred["unit_id"], pred["time_index"])


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "variant",
        "feature_mode",
        "uses_target_val_labels",
        "uses_test_labels_for_fit",
        "degree",
        "ridge",
        *METRIC_COLUMNS,
        "rmse_delta_vs_raw",
        "rmse_reduction_vs_raw_pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "任务",
        "消融口径",
        "特征",
        "RMSE",
        "MAE",
        "NASA",
        "RA",
        "Last RMSE",
        "RMSE较raw降低",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        gain = row.get("rmse_reduction_vs_raw_pct")
        gain_text = "-" if gain == "" else f"{float(gain):.1f}%"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["variant"]),
                    str(row["feature_mode"]),
                    f"{float(row['rmse']):.4f}",
                    f"{float(row['mae']):.4f}",
                    f"{float(row['nasa_score']):.4f}",
                    f"{float(row['ra']):.4f}",
                    f"{float(row['last_window_rmse']):.4f}",
                    gain_text,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Validation-only TC 消融结果

本文件用于回答最终 `PG-STDA-SAC-RSPA-TC` 中 `TC` 是否只是依赖时间先验的问题。消融只基于已经训练好的 raw `PG-STDA-SAC-RSPA` 输出，不重新训练模型。

## 实验口径

- `raw`：未校准的 `PG-STDA-SAC-RSPA` 预测。
- `y_pred-only TC`：只用验证集 raw `y_pred` 拟合 ridge 校准。
- `time-only TC`：只用验证集 `time_index` 拟合 ridge 校准，用作时间先验负控。
- `y_pred+time TC`：使用验证集 raw `y_pred` 与 `time_index`，对应最终 `PG-STDA-SAC-RSPA-TC`。
- 所有 TC 变体只使用目标验证集标签拟合校准映射，不使用测试标签，不重新训练模型。

指标方向：RMSE、MAE、NASA、Last RMSE 越低越好；RA 越高越好。

## 结果表

{markdown_table(rows)}

## 结论

`time-only TC` 是关键负控：它衡量单纯依靠时间进度能达到什么水平。

第一迁移中，`y_pred+time TC` 明显优于 `time-only TC`，说明 raw 迁移预测与时间先验具有互补性，最终 TC 主要是在 raw 模型输出基础上修正跨域尺度偏差。

第二迁移中，`time-only TC` 的 RMSE 略低于 `y_pred+time TC`，说明卫星电池仿真目标域具有较强的轨道周期/生命周期时间先验。报告中应据此保持谨慎：第二迁移的 strict raw 结果才是迁移表征能力的主证据，TC 后结果应定位为 validation-only 工程校准管线，而不是纯无监督迁移训练能力。

本消融不改变主结论边界：

- strict raw 迁移能力仍以未校准 `PG-STDA-SAC-RSPA` 为准；
- `PG-STDA-SAC-RSPA-TC` 是 validation-only calibrated final engineering pipeline；
- TC 不应被表述为严格无目标标签的训练模块。
"""
    path.write_text(content, encoding="utf-8")


def run_task(
    task: str,
    input_dir: Path,
    output_dir: Path,
    degree: int,
    ridge: float,
    clip_range: tuple[float, float],
) -> list[dict[str, Any]]:
    val_raw = load_prediction_csv(input_dir / "predictions_val.csv")
    test_raw = load_prediction_csv(input_dir / "predictions_test.csv")
    task_slug = "first_transfer" if task.startswith("第一") else "second_transfer"
    task_dir = output_dir / task_slug
    task_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    raw_metrics = evaluate(test_raw)
    raw_rmse = float(raw_metrics["rmse"])

    variants = [
        ("raw", "none", None, test_raw),
        ("y_pred-only TC", "y_pred", "y_pred", None),
        ("time-only TC", "time", "time", None),
        ("y_pred+time TC", "y_pred,time_index", "y_pred_time", None),
    ]
    calibration_records: dict[str, Any] = {}
    for variant, feature_label, feature_mode, explicit_pred in variants:
        if feature_mode is None:
            pred = clone_prediction(explicit_pred if explicit_pred is not None else test_raw)
            calibration_record = None
        else:
            calibration = fit_ridge_calibration(val_raw, feature_mode=feature_mode, degree=degree, ridge=ridge)
            pred = apply_calibration(test_raw, calibration, clip_range=clip_range)
            calibration_record = dict(calibration)
            calibration_record["coef"] = calibration["coef"].tolist()
            calibration_records[variant] = calibration_record
        metrics = evaluate(pred)
        save_prediction_csv(pred, task_dir / f"{variant.replace(' ', '_').replace('+', 'plus').replace('-', '_')}_predictions_test.csv")
        rmse = float(metrics["rmse"])
        row: dict[str, Any] = {
            "task": task,
            "variant": variant,
            "feature_mode": feature_label,
            "uses_target_val_labels": "no" if feature_mode is None else "yes",
            "uses_test_labels_for_fit": "no",
            "degree": "" if feature_mode is None else int(degree),
            "ridge": "" if feature_mode is None else float(ridge),
            "rmse_delta_vs_raw": rmse - raw_rmse,
            "rmse_reduction_vs_raw_pct": "" if feature_mode is None else 100.0 * (raw_rmse - rmse) / raw_rmse,
        }
        row.update({key: float(metrics[key]) for key in METRIC_COLUMNS})
        rows.append(row)
        if calibration_record is not None:
            row["calibration"] = calibration_record
    save_json(
        {
            "task": task,
            "input_dir": str(input_dir),
            "degree": int(degree),
            "ridge": float(ridge),
            "clip_range": [float(clip_range[0]), float(clip_range[1])],
            "calibrations": calibration_records,
            "rows": rows,
        },
        task_dir / "tc_ablation_metrics.json",
    )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_range = (float(args.clip_min), float(args.clip_max))
    all_rows: list[dict[str, Any]] = []
    all_rows.extend(
        run_task(
            "第一迁移",
            args.first_input_dir,
            output_dir,
            degree=int(args.degree),
            ridge=float(args.ridge),
            clip_range=clip_range,
        )
    )
    all_rows.extend(
        run_task(
            "第二迁移",
            args.second_input_dir,
            output_dir,
            degree=int(args.degree),
            ridge=float(args.ridge),
            clip_range=clip_range,
        )
    )
    write_csv(all_rows, output_dir / "tc_ablation_summary.csv")
    write_markdown(all_rows, PROJECT_ROOT / "docs/TC_CALIBRATION_ABLATION.md")
    write_markdown(all_rows, output_dir / "tc_ablation_summary.md")
    print(f"Wrote TC ablation results to {output_dir}")
    for row in all_rows:
        print(
            f"{row['task']} | {row['variant']} | RMSE={float(row['rmse']):.6f} "
            f"MAE={float(row['mae']):.6f} RA={float(row['ra']):.6f}"
        )


if __name__ == "__main__":
    main()
