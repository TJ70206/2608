"""Apply validation-only output calibration to an existing transfer run.

This script does not retrain the model. It fits a small time-aware calibration
map on predictions_val.csv and applies it to predictions_test.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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

from scripts.train_transfer import _apply_time_aware_output_calibration, _fit_time_aware_output_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate existing transfer predictions using target validation only.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--degree", default=2, type=int)
    parser.add_argument("--ridge", default=0.01, type=float)
    parser.add_argument("--clip-min", default=0.0, type=float)
    parser.add_argument("--clip-max", default=1.0, type=float)
    return parser.parse_args()


def load_prediction_csv(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
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


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    val_pred = load_prediction_csv(input_dir / "predictions_val.csv")
    test_pred = load_prediction_csv(input_dir / "predictions_test.csv")
    calibration = _fit_time_aware_output_calibration(
        val_pred,
        {"degree": int(args.degree), "ridge": float(args.ridge)},
    )
    clip_range = (float(args.clip_min), float(args.clip_max))
    _apply_time_aware_output_calibration(val_pred, calibration, clip_range=clip_range)
    _apply_time_aware_output_calibration(test_pred, calibration, clip_range=clip_range)

    metrics = rul_metrics_with_time(test_pred["y_true"], test_pred["y_pred"], test_pred["unit_id"], test_pred["time_index"])
    source_metrics_path = input_dir / "transfer_metrics.json"
    if source_metrics_path.exists():
        with source_metrics_path.open("r", encoding="utf-8") as f:
            source_metrics = json.load(f)
    else:
        source_metrics = {"source_pretrain_history": [], "transfer_history": [], "final_metrics": {}}
    final_metrics = dict(source_metrics.get("final_metrics", {}))
    final_metrics.update(metrics)
    final_metrics["time_aware_calibration_enabled"] = 1.0
    final_metrics["time_aware_calibration_degree"] = float(calibration["degree"])
    final_metrics["time_aware_calibration_ridge"] = float(calibration["ridge"])
    final_metrics["time_aware_calibration_time_min"] = float(calibration["time_min"])
    final_metrics["time_aware_calibration_time_scale"] = float(calibration["time_scale"])
    final_metrics["time_aware_calibration_coef"] = calibration["coef"].tolist()
    final_metrics["calibration_source"] = "target_validation_only"
    final_metrics["time_aware_calibration_features"] = "y_pred,time_index"
    final_metrics["uncalibrated_metrics_path"] = str(input_dir / "transfer_metrics.json")

    save_prediction_csv(val_pred, output_dir / "predictions_val.csv")
    save_prediction_csv(test_pred, output_dir / "predictions_test.csv")
    save_json(
        {
            "source_pretrain_history": source_metrics.get("source_pretrain_history", []),
            "transfer_history": source_metrics.get("transfer_history", []),
            "final_metrics": final_metrics,
        },
        output_dir / "transfer_metrics.json",
    )
    copied_resolved_config = False
    for filename in ("resolved_config.json", "env_info.json", "transfer_model.pt"):
        source = input_dir / filename
        if source.exists():
            if filename == "resolved_config.json":
                with source.open("r", encoding="utf-8") as f:
                    resolved_config = json.load(f)
                resolved_config.setdefault("experiment", {})
                resolved_config["experiment"]["output_dir"] = str(output_dir)
                resolved_config["experiment"]["name"] = output_dir.name
                resolved_config["evaluation"] = {
                    "calibrate_time_aware": True,
                    "time_aware": {
                        "degree": int(calibration["degree"]),
                        "ridge": float(calibration["ridge"]),
                    },
                    "clip_min": float(args.clip_min),
                    "clip_max": float(args.clip_max),
                    "calibration_source": "target_validation_only",
                    "calibration_features": ["y_pred", "time_index"],
                    "uncalibrated_output_dir": str(input_dir),
                }
                save_json(resolved_config, output_dir / filename)
                copied_resolved_config = True
            else:
                shutil.copy2(source, output_dir / filename)
    if not copied_resolved_config:
        save_json(
            {
                "experiment": {"name": output_dir.name, "output_dir": str(output_dir)},
                "evaluation": {
                    "calibrate_time_aware": True,
                    "time_aware": {
                        "degree": int(calibration["degree"]),
                        "ridge": float(calibration["ridge"]),
                    },
                    "clip_min": float(args.clip_min),
                    "clip_max": float(args.clip_max),
                    "calibration_source": "target_validation_only",
                    "calibration_features": ["y_pred", "time_index"],
                    "uncalibrated_output_dir": str(input_dir),
                },
            },
            output_dir / "resolved_config.json",
        )
    print(f"Wrote calibrated outputs to {output_dir}")
    print(f"RMSE={metrics['rmse']:.6f}; MAE={metrics['mae']:.6f}; RA={metrics['ra']:.6f}")


if __name__ == "__main__":
    main()
