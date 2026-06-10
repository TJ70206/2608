from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xa202608.config import load_config
from xa202608.data.factory import build_dataloaders
from xa202608.models.factory import build_model
from xa202608.utils import set_seed


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "synthetic_debug.yaml")
    set_seed(int(config["experiment"].get("seed", 42)))
    train_loader, val_loader, test_loader, input_channels = build_dataloaders(config)
    config["model"]["input_channels"] = input_channels
    model = build_model(config)
    batch = next(iter(train_loader))
    pred = model(batch["x"])
    print("project check ok")
    print(f"train_batches={len(train_loader)}, val_batches={len(val_loader)}, test_batches={len(test_loader)}")
    print(f"input_shape={tuple(batch['x'].shape)}, pred_shape={tuple(pred.shape)}")


if __name__ == "__main__":
    main()
