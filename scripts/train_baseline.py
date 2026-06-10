from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xa202608.config import load_config
from xa202608.data.factory import build_dataloaders
from xa202608.models.factory import build_model
from xa202608.train_eval import fit_baseline
from xa202608.utils import ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XA-202608 baseline model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["experiment"].get("seed", 42))
    set_seed(seed)
    output_dir = ensure_dir(PROJECT_ROOT / config["experiment"].get("output_dir", "outputs/default"))
    train_loader, val_loader, test_loader, input_channels = build_dataloaders(config)
    config["model"]["input_channels"] = input_channels
    save_json(config, output_dir / "resolved_config.json")
    model = build_model(config)
    device = get_device(str(config["train"].get("device", "auto")))
    metrics = fit_baseline(model, train_loader, val_loader, test_loader, config, device, output_dir)
    print("Final test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
