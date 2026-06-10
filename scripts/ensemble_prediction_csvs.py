from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xa202608.experiment_io import save_prediction_csv
from xa202608.metrics import rul_metrics_with_time
from xa202608.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-weighted ensemble for two prediction CSVs.")
    parser.add_argument("--val-a", required=True)
    parser.add_argument("--test-a", required=True)
    parser.add_argument("--val-b", required=True)
    parser.add_argument("--test-b", required=True)
    parser.add_argument("--name-a", default="model_a")
    parser.add_argument("--name-b", default="model_b")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--objective", choices=["rmse", "last_window_rmse"], default="rmse")
    return parser.parse_args()


def _load_prediction(path: str | Path, prediction_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"unit_id", "time_index", "stage", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing prediction columns: {sorted(missing)}")
    return df[["unit_id", "time_index", "stage", "y_true", "y_pred"]].rename(columns={"y_pred": prediction_name})


def _merge_predictions(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> pd.DataFrame:
    keys = ["unit_id", "time_index", "stage"]
    merged = a.merge(b, on=keys, how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError("prediction CSVs have no overlapping unit/time/stage rows")
    if not np.allclose(merged["y_true_a"].to_numpy(), merged["y_true_b"].to_numpy(), atol=1e-6):
        raise ValueError("prediction CSV y_true columns do not match after merge")
    merged["y_true"] = merged["y_true_a"]
    return merged[keys + ["y_true", name_a, name_b]]


def _metrics(df: pd.DataFrame, y_pred: np.ndarray) -> dict[str, float]:
    return rul_metrics_with_time(
        df["y_true"].to_numpy(dtype=np.float64),
        y_pred.astype(np.float64),
        df["unit_id"].to_numpy(),
        df["time_index"].to_numpy(),
    )


def main() -> None:
    args = parse_args()
    if args.step <= 0 or args.step > 1:
        raise ValueError("--step must be in (0, 1]")
    out_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    val = _merge_predictions(
        _load_prediction(args.val_a, args.name_a),
        _load_prediction(args.val_b, args.name_b),
        args.name_a,
        args.name_b,
    )
    test = _merge_predictions(
        _load_prediction(args.test_a, args.name_a),
        _load_prediction(args.test_b, args.name_b),
        args.name_a,
        args.name_b,
    )
    best: tuple[float, float, dict[str, float]] | None = None
    weights = np.arange(0.0, 1.0 + 0.5 * args.step, args.step)
    for weight_a in weights:
        y_pred = weight_a * val[args.name_a].to_numpy() + (1.0 - weight_a) * val[args.name_b].to_numpy()
        metrics = _metrics(val, y_pred)
        objective_value = float(metrics[args.objective])
        if best is None or objective_value < best[0]:
            best = (objective_value, float(weight_a), metrics)
    if best is None:
        raise RuntimeError("failed to select an ensemble weight")
    _, weight_a, val_metrics = best
    test_pred = weight_a * test[args.name_a].to_numpy() + (1.0 - weight_a) * test[args.name_b].to_numpy()
    test_metrics = _metrics(test, test_pred)
    test_metrics[f"ensemble_weight_{args.name_a}"] = float(weight_a)
    test_metrics[f"ensemble_weight_{args.name_b}"] = float(1.0 - weight_a)
    test_metrics["ensemble_validation_objective"] = args.objective
    test_metrics["ensemble_validation_objective_value"] = float(val_metrics[args.objective])
    save_prediction_csv(
        {
            "unit_id": test["unit_id"].to_numpy(),
            "time_index": test["time_index"].to_numpy(),
            "stage": test["stage"].to_numpy(),
            "y_true": test["y_true"].to_numpy(),
            "y_pred": test_pred,
        },
        out_dir / "predictions_test.csv",
    )
    save_json({"validation_metrics": val_metrics, "test_metrics": test_metrics}, out_dir / "metrics.json")
    print("Selected ensemble weights:")
    print(f"  {args.name_a}: {weight_a}")
    print(f"  {args.name_b}: {1.0 - weight_a}")
    print("Final ensemble test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
