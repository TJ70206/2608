from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xa202608.config import load_config
from xa202608.data.factory import build_dataloaders
from xa202608.experiment_io import save_prediction_csv
from xa202608.metrics import rul_metrics_with_time
from xa202608.models.factory import build_model
from xa202608.plotting import plot_evaluation_figures_csv
from xa202608.train_eval import predict
from xa202608.utils import ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained XA-202608 baseline model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--model", type=str, default=None, help="Path to model state dict. Defaults to <output_dir>/model.pt.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save evaluation metrics JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["experiment"].get("seed", 42))
    set_seed(seed)
    output_dir = ensure_dir(PROJECT_ROOT / config["experiment"].get("output_dir", "outputs/default"))
    _, val_loader, test_loader, input_channels = build_dataloaders(config)
    config["model"]["input_channels"] = input_channels
    model = build_model(config)
    device = get_device(str(config["train"].get("device", "auto")))
    model_path = Path(args.model) if args.model is not None else output_dir / "model.pt"
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    loader = val_loader if args.split == "val" else test_loader
    pred = predict(model, loader, device)
    metrics = rul_metrics_with_time(pred["y_true"], pred["y_pred"], pred["unit_id"], pred["time_index"])
    metrics["model_path"] = str(model_path)
    metrics["split"] = args.split
    prediction_path = output_dir / f"predictions_{args.split}_reeval.csv"
    save_prediction_csv(pred, prediction_path)
    calibration_path = output_dir / "predictions_val.csv"
    metrics["plot_paths"] = [
        str(path)
        for path in plot_evaluation_figures_csv(
            prediction_path,
            output_dir,
            prefix=f"{args.split}_reeval",
            calibration_csv=calibration_path if calibration_path.exists() else None,
        )
    ]
    metrics_path = Path(args.output) if args.output is not None else output_dir / f"metrics_{args.split}_reeval.json"
    if not metrics_path.is_absolute():
        metrics_path = PROJECT_ROOT / metrics_path
    save_json(metrics, metrics_path)
    print(f"Evaluation metrics ({args.split}):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
