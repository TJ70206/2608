from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xa202608.models.factory import build_model  # noqa: E402


class RecurrentModelTests(unittest.TestCase):
    def test_lstm_and_gru_support_transfer_features(self) -> None:
        for name in ["lstm", "gru"]:
            config = {
                "model": {
                    "name": name,
                    "input_channels": 3,
                    "hidden_channels": 16,
                    "num_layers": 1,
                    "dropout": 0.1,
                    "output_dim": 1,
                    "pooling": "last",
                }
            }
            model = build_model(config)
            x = torch.randn(4, 12, 3)
            pred, features = model(x, return_features=True)

            self.assertEqual(tuple(pred.shape), (4, 1))
            self.assertEqual(tuple(features.shape), (4, 16))


if __name__ == "__main__":
    unittest.main()
