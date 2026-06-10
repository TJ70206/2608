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

from xa202608.data.reaction_wheel import generate_reaction_wheel_dataset  # noqa: E402
from xa202608.data.xjtu_reaction_wheel import (  # noqa: E402
    build_xjtu_reaction_wheel_transfer_loaders,
    reaction_wheel_dataframe_to_hi_unit_dicts,
)
from xa202608.data.windowing import WindowedTimeSeriesDataset  # noqa: E402


def _write_xjtu_npz(path: Path) -> None:
    rng = np.random.default_rng(202608)
    series = [
        rng.normal(0.0, 1.0, size=(44, 14)).astype(np.float32),
        rng.normal(0.1, 1.1, size=(48, 14)).astype(np.float32),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        series=np.asarray(series, dtype=object),
        unit_ids=np.asarray(["Bearing1_1", "Bearing1_2"], dtype=object),
        conditions=np.asarray(["condition_a", "condition_a"], dtype=object),
    )


def _small_reaction_config() -> dict:
    return {
        "num_units": 8,
        "min_length": 72,
        "max_length": 92,
        "time_step_seconds": 60,
        "split_ratios": {"train": 0.5, "val": 0.25, "calib": 0.125, "test": 0.125},
        "missing_rate": 0.02,
        "missing_burst_rate": 0.0,
        "noise_scale": 0.01,
        "explicit_fault_fraction": 0.25,
        "friction_threshold_multiplier": 6.0,
    }


class XjtuReactionTransferTests(unittest.TestCase):
    def test_reaction_wheel_hi_matches_xjtu_hi_dimension(self) -> None:
        df, _ = generate_reaction_wheel_dataset(_small_reaction_config(), seed=202608)

        series, rul = reaction_wheel_dataframe_to_hi_unit_dicts(
            df,
            feature_columns=["motor_current", "vibration_proxy"],
            hi_window_size=8,
            hi_stride=4,
            target_column="normalized_rul",
        )

        self.assertEqual(set(series), set(rul))
        self.assertEqual(next(iter(series.values())).shape[1], 14)
        for unit_id, values in series.items():
            self.assertEqual(values.shape[0], len(rul[unit_id]))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(np.all((rul[unit_id] >= 0.0) & (rul[unit_id] <= 1.0)))

    def test_reaction_wheel_hi_rejects_hidden_state_leakage_columns(self) -> None:
        df, _ = generate_reaction_wheel_dataset(_small_reaction_config(), seed=202608)

        with self.assertRaisesRegex(ValueError, "leakage"):
            reaction_wheel_dataframe_to_hi_unit_dicts(
                df,
                feature_columns=["kt_ratio", "vibration_proxy"],
                hi_window_size=8,
                hi_stride=4,
                target_column="normalized_rul",
            )

    def test_build_transfer_loaders_produces_aligned_source_and_target_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            xjtu_path = tmp_path / "xjtu_hi.npz"
            reaction_csv = tmp_path / "reaction_wheel_sim.csv"
            _write_xjtu_npz(xjtu_path)
            df, _ = generate_reaction_wheel_dataset(_small_reaction_config(), seed=202608)
            df.to_csv(reaction_csv, index=False)

            config = {
                "experiment": {"seed": 202608},
                "data": {
                    "dataset": "xjtu_reaction_wheel_transfer",
                    "source_processed_path": str(xjtu_path),
                    "reaction_wheel_csv_path": str(reaction_csv),
                    "source_units": ["Bearing1_1", "Bearing1_2"],
                    "target_train_split": "train",
                    "target_val_split": "val",
                    "target_test_split": "test",
                    "target_hi_feature_columns": ["motor_current", "vibration_proxy"],
                    "target_hi_window_size": 8,
                    "target_hi_stride": 4,
                    "target_column": "normalized_rul",
                    "window_size": 4,
                    "stride": 2,
                    "max_rul": None,
                },
                "train": {"batch_size": 4},
            }

            source_loader, target_loader, val_loader, test_loader, input_channels = (
                build_xjtu_reaction_wheel_transfer_loaders(config)
            )
            source_batch = next(iter(source_loader))
            target_batch = next(iter(target_loader))

            self.assertEqual(input_channels, 14)
            self.assertEqual(source_batch["x"].shape[-1], 14)
            self.assertEqual(target_batch["x"].shape[-1], 14)
            self.assertGreater(len(val_loader.dataset), 0)
            self.assertGreater(len(test_loader.dataset), 0)

    def test_windowed_dataset_can_use_external_stage_labels(self) -> None:
        series = {1: np.arange(24, dtype=np.float32).reshape(6, 4)}
        rul = {1: np.asarray([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=np.float32)}
        external_stage = {1: np.asarray([2, 2, 1, 1, 0, 0], dtype=np.int64)}

        dataset = WindowedTimeSeriesDataset(
            series,
            rul,
            window_size=2,
            stride=2,
            stage_by_unit=external_stage,
        )

        self.assertEqual(dataset.stage_labels, [2, 1, 0])


if __name__ == "__main__":
    unittest.main()
