from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xa202608.data.reaction_wheel import (  # noqa: E402
    LEAKAGE_COLUMNS,
    TELEMETRY_ONLY_COLUMNS,
    build_reaction_wheel_loaders,
    find_first_eol_index,
    generate_reaction_wheel_dataset,
)


def small_config(output_dir: str | Path | None = None) -> dict:
    data_cfg = {
        "dataset": "reaction_wheel",
        "num_units": 8,
        "min_length": 80,
        "max_length": 110,
        "time_step_seconds": 60,
        "split_ratios": {"train": 0.5, "val": 0.25, "calib": 0.125, "test": 0.125},
        "window_size": 16,
        "stride": 4,
        "max_rul": None,
        "feature_protocol": "telemetry_only",
        "missing_rate": 0.04,
        "missing_burst_rate": 0.01,
        "noise_scale": 0.01,
        "explicit_fault_fraction": 0.25,
    }
    if output_dir is not None:
        data_cfg["csv_path"] = str(Path(output_dir) / "reaction_wheel_sim.csv")
    return {
        "experiment": {"seed": 202608},
        "data": data_cfg,
        "train": {"batch_size": 4},
    }


class ReactionWheelSimTests(unittest.TestCase):
    def test_generate_dataset_structure_and_monotonic_labels(self) -> None:
        df, metadata = generate_reaction_wheel_dataset(small_config()["data"], seed=202608)

        required = {
            "unit_id",
            "time_step",
            "mission_time_s",
            "command_voltage",
            "wheel_speed",
            "wheel_speed_rpm",
            "motor_current",
            "temperature",
            "vibration_proxy",
            "friction_torque_proxy",
            "torque_noise_proxy",
            "kt",
            "kt_ratio",
            "lubricant_loss",
            "lubricant_hi",
            "friction_torque",
            "friction_torque_magnitude",
            "degradation_progress",
            "health_stage",
            "rul",
            "normalized_rul",
            "base_degradation",
            "explicit_fault_injection_type",
            "is_censored",
            "split",
        }
        self.assertTrue(required.issubset(df.columns))
        self.assertEqual(df["unit_id"].nunique(), 8)
        self.assertEqual(set(df["split"].unique()), {"train", "val", "calib", "test"})
        self.assertFalse(df[TELEMETRY_ONLY_COLUMNS].isna().any().any())
        self.assertTrue(set(TELEMETRY_ONLY_COLUMNS).isdisjoint(LEAKAGE_COLUMNS))
        self.assertIn("references", metadata)
        self.assertEqual(metadata["default_feature_protocol"], "telemetry_only")

        for _, group in df.groupby("unit_id", sort=False):
            group = group.sort_values("time_step")
            self.assertLessEqual(np.diff(group["kt_ratio"].to_numpy()).max(), 1e-6)
            self.assertGreaterEqual(np.diff(group["lubricant_loss"].to_numpy()).min(), -1e-6)
            self.assertLessEqual(np.diff(group["rul"].to_numpy()).max(), 1e-6)
            self.assertTrue(group["normalized_rul"].between(0.0, 1.0).all())
            self.assertTrue(group["health_stage"].isin([0, 1, 2]).all())

    def test_eol_uses_friction_magnitude_for_reverse_rotation(self) -> None:
        kt_ratio = np.asarray([1.0, 0.9, 0.8, 0.7], dtype=np.float32)
        lubricant_hi = np.asarray([1.0, 0.9, 0.8, 0.7], dtype=np.float32)
        friction = np.asarray([-0.001, -0.002, -0.009, -0.010], dtype=np.float32)

        index, reason = find_first_eol_index(
            kt_ratio=kt_ratio,
            lubricant_hi=lubricant_hi,
            friction_torque=friction,
            kt_threshold_ratio=0.3,
            lubricant_threshold=0.05,
            friction_threshold=0.008,
        )

        self.assertEqual(index, 2)
        self.assertEqual(reason, "friction_magnitude_threshold")

    def test_build_reaction_wheel_loaders_uses_telemetry_only_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = small_config(tmp)
            df, _ = generate_reaction_wheel_dataset(config["data"], seed=202608)
            csv_path = Path(config["data"]["csv_path"])
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)

            train_loader, val_loader, test_loader, input_dim = build_reaction_wheel_loaders(config)
            batch = next(iter(train_loader))

            self.assertEqual(input_dim, len(TELEMETRY_ONLY_COLUMNS))
            self.assertEqual(batch["x"].shape[-1], len(TELEMETRY_ONLY_COLUMNS))
            self.assertGreater(len(train_loader.dataset), 0)
            self.assertGreater(len(val_loader.dataset), 0)
            self.assertGreater(len(test_loader.dataset), 0)


if __name__ == "__main__":
    unittest.main()
