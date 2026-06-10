from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xa202608.data.reaction_wheel import generate_reaction_wheel_dataset  # noqa: E402
from xa202608.data.xjtu_reaction_wheel import build_reaction_wheel_hi_loaders  # noqa: E402
from xa202608.losses import mmd_loss, transfer_alignment_loss, weighted_mmd_loss  # noqa: E402
from scripts.train_transfer import _is_better_checkpoint, _scheduled_alignment_weight  # noqa: E402


def _small_reaction_config(output_csv: Path) -> dict:
    return {
        "experiment": {"seed": 202608},
        "data": {
            "dataset": "reaction_wheel_hi",
            "reaction_wheel_csv_path": str(output_csv),
            "target_hi_feature_columns": ["motor_current", "vibration_proxy"],
            "target_hi_window_size": 8,
            "target_hi_stride": 4,
            "target_column": "normalized_rul",
            "window_size": 4,
            "stride": 2,
            "max_rul": None,
            "normalize": "zscore",
        },
        "train": {"batch_size": 4},
    }


class FirstTransferExperimentTests(unittest.TestCase):
    def test_transfer_alignment_loss_supports_none_global_and_stage_modes(self) -> None:
        source = torch.tensor(
            [[0.0, 0.0], [0.1, 0.2], [1.0, 1.1], [1.2, 0.9], [2.0, 2.1], [2.2, 2.0]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[0.05, 0.05], [1.1, 1.0], [1.3, 1.1], [2.1, 2.0], [2.3, 2.2]],
            dtype=torch.float32,
        )
        source_stage = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        target_stage = torch.tensor([0, 1, 1, 2, 2], dtype=torch.long)

        none_loss = transfer_alignment_loss(source, target, source_stage, target_stage, mode="none", num_stages=3)
        global_loss = transfer_alignment_loss(source, target, source_stage, target_stage, mode="global", num_stages=3)
        stage_loss = transfer_alignment_loss(source, target, source_stage, target_stage, mode="stage", num_stages=3)

        self.assertEqual(float(none_loss), 0.0)
        self.assertTrue(torch.allclose(global_loss, mmd_loss(source, target)))
        self.assertGreater(float(stage_loss), 0.0)

    def test_weighted_mmd_matches_unweighted_mmd_for_uniform_weights(self) -> None:
        source = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
        target = torch.tensor([[0.2, 0.1], [1.1, 0.9]], dtype=torch.float32)
        source_weights = torch.ones(source.shape[0], dtype=torch.float32)
        target_weights = torch.ones(target.shape[0], dtype=torch.float32)

        weighted = weighted_mmd_loss(source, target, source_weights, target_weights)
        unweighted = mmd_loss(source, target)

        self.assertTrue(torch.allclose(weighted, unweighted, atol=1e-6))

    def test_stage_alignment_accepts_late_stage_weights(self) -> None:
        source = torch.tensor(
            [[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.2, 4.0]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[0.0, 0.1], [0.2, 0.0], [7.0, 7.0], [7.2, 7.0]],
            dtype=torch.float32,
        )
        stages = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        plain = transfer_alignment_loss(source, target, stages, stages, mode="stage", num_stages=2)
        weighted = transfer_alignment_loss(
            source,
            target,
            stages,
            stages,
            mode="stage",
            num_stages=2,
            stage_weights=torch.tensor([0.1, 2.0], dtype=torch.float32),
        )

        self.assertGreater(float(weighted), float(plain))

    def test_reaction_wheel_hi_target_only_loader_uses_fourteen_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "reaction_wheel_sim.csv"
            df, _ = generate_reaction_wheel_dataset(
                {
                    "num_units": 8,
                    "min_length": 72,
                    "max_length": 92,
                    "split_ratios": {"train": 0.5, "val": 0.25, "calib": 0.125, "test": 0.125},
                    "missing_rate": 0.02,
                    "missing_burst_rate": 0.0,
                    "friction_threshold_multiplier": 6.0,
                },
                seed=202608,
            )
            df.to_csv(csv_path, index=False)

            train_loader, val_loader, test_loader, input_dim = build_reaction_wheel_hi_loaders(_small_reaction_config(csv_path))
            batch = next(iter(train_loader))

            self.assertEqual(input_dim, 14)
            self.assertEqual(batch["x"].shape[-1], 14)
            self.assertGreater(len(train_loader.dataset), 0)
            self.assertGreater(len(val_loader.dataset), 0)
            self.assertGreater(len(test_loader.dataset), 0)

    def test_transfer_checkpoint_comparison_supports_min_and_max_modes(self) -> None:
        self.assertTrue(_is_better_checkpoint(0.2, 0.3, "min", 0.0))
        self.assertFalse(_is_better_checkpoint(0.31, 0.3, "min", 0.0))
        self.assertTrue(_is_better_checkpoint(0.82, 0.8, "max", 0.0))
        self.assertFalse(_is_better_checkpoint(0.79, 0.8, "max", 0.0))

    def test_scheduled_alignment_weight_warms_up_linearly(self) -> None:
        self.assertEqual(_scheduled_alignment_weight(0.003, epoch=1, warmup_epochs=0), 0.003)
        self.assertAlmostEqual(_scheduled_alignment_weight(0.003, epoch=1, warmup_epochs=5), 0.0006)
        self.assertAlmostEqual(_scheduled_alignment_weight(0.003, epoch=3, warmup_epochs=5), 0.0018)
        self.assertAlmostEqual(_scheduled_alignment_weight(0.003, epoch=6, warmup_epochs=5), 0.003)


if __name__ == "__main__":
    unittest.main()
