from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xa202608.data.satellite_battery import (  # noqa: E402
    BATTERY_PROXY_V1_COLUMNS,
    LEAKAGE_COLUMNS,
    TELEMETRY_ONLY_COLUMNS,
    add_battery_physics_proxy_features,
    build_nasa_satellite_battery_transfer_loaders,
    build_satellite_battery_loaders,
    generate_satellite_battery_dataset,
    validate_satellite_battery_dataframe,
)


def _small_config(output_dir: str | Path | None = None) -> dict:
    data_cfg = {
        "dataset": "satellite_battery_sim",
        "num_units": 9,
        "min_eol_cycles": 24,
        "max_eol_cycles": 36,
        "steps_per_cycle": 8,
        "split_ratios": {"train": 0.55, "val": 0.22, "test": 0.23},
        "explicit_fault_fraction": 0.33,
        "voltage_noise_std": 0.002,
        "temperature_noise_std": 0.05,
        "current_noise_std": 0.002,
        "window_size": 12,
        "stride": 4,
        "target_column": "normalized_rul",
        "feature_protocol": "telemetry_only",
        "max_rul": None,
    }
    if output_dir is not None:
        data_cfg["csv_path"] = str(Path(output_dir) / "satellite_battery_sim.csv")
    return {"experiment": {"seed": 202608}, "data": data_cfg, "train": {"batch_size": 4}}


def _write_source_battery_csv(path: Path) -> None:
    rows = []
    for unit_id, life in [(5, 44), (6, 40)]:
        for cycle in range(1, life + 1):
            progress = (cycle - 1) / max(life - 1, 1)
            soh = 1.0 - 0.2 * progress
            rows.append(
                {
                    "unit_id": unit_id,
                    "cycle": cycle,
                    "voltage": 3.65 - 0.35 * progress,
                    "current": -1.8 + 0.02 * np.sin(cycle / 3.0),
                    "temperature": 30.0 + 1.5 * np.sin(cycle / 5.0),
                    "soh": soh,
                    "rul": max(life - cycle, 0),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


class SatelliteBatterySimTests(unittest.TestCase):
    def test_generate_dataset_structure_and_quality(self) -> None:
        df, metadata = generate_satellite_battery_dataset(_small_config()["data"], seed=202608)
        summary = validate_satellite_battery_dataframe(df)

        required = {
            "unit_id",
            "split",
            "cycle",
            "time_step",
            "elapsed_days",
            "orbit_phase",
            "orbit_phase_sin",
            "orbit_phase_cos",
            "sunlight",
            "current",
            "voltage",
            "temperature",
            "soc_true",
            "soc_est",
            "capacity_ah",
            "capacity_ratio",
            "internal_resistance_ohm",
            "resistance_ratio",
            "soh",
            "fused_damage",
            "rul_cycles",
            "normalized_rul",
            "health_stage",
            "fault_type",
            "fault_active",
        }
        self.assertTrue(required.issubset(df.columns))
        self.assertEqual(df["unit_id"].nunique(), 9)
        self.assertEqual(set(df["split"].unique()), {"train", "val", "test"})
        self.assertFalse(df[TELEMETRY_ONLY_COLUMNS].isna().any().any())
        self.assertTrue(set(TELEMETRY_ONLY_COLUMNS).isdisjoint(LEAKAGE_COLUMNS))
        self.assertEqual(metadata["default_feature_protocol"], "telemetry_only")
        self.assertEqual(summary["units"], 9)

        for _, cycle_group in df.groupby(["unit_id", "cycle"], sort=False):
            self.assertEqual(cycle_group["capacity_ah"].nunique(), 1)
            self.assertEqual(cycle_group["internal_resistance_ohm"].nunique(), 1)

        for _, group in df.groupby("unit_id", sort=False):
            cycle_level = group.sort_values(["cycle", "time_step"]).drop_duplicates("cycle")
            self.assertLessEqual(np.diff(cycle_level["capacity_ah"].to_numpy()).max(), 1e-6)
            self.assertGreaterEqual(np.diff(cycle_level["internal_resistance_ohm"].to_numpy()).min(), -1e-6)
            self.assertLessEqual(np.diff(cycle_level["rul_cycles"].to_numpy()).max(), 1e-6)
            self.assertTrue(group["soc_true"].between(0.0, 1.0).all())
            self.assertTrue(group["voltage"].between(2.5, 4.25).all())
            self.assertTrue(group["normalized_rul"].between(0.0, 1.0).all())

    def test_build_satellite_battery_loaders_use_vit_only_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _small_config(tmp)
            df, _ = generate_satellite_battery_dataset(config["data"], seed=202608)
            csv_path = Path(config["data"]["csv_path"])
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)

            train_loader, val_loader, test_loader, input_dim = build_satellite_battery_loaders(config)
            batch = next(iter(train_loader))

            self.assertEqual(input_dim, len(TELEMETRY_ONLY_COLUMNS))
            self.assertEqual(batch["x"].shape[-1], len(TELEMETRY_ONLY_COLUMNS))
            self.assertGreater(len(train_loader.dataset), 0)
            self.assertGreater(len(val_loader.dataset), 0)
            self.assertGreater(len(test_loader.dataset), 0)

    def test_build_nasa_to_satellite_transfer_loaders_aligns_vit_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_csv = tmp_path / "nasa_battery.csv"
            target_csv = tmp_path / "satellite_battery_sim.csv"
            _write_source_battery_csv(source_csv)
            target_df, _ = generate_satellite_battery_dataset(_small_config()["data"], seed=202608)
            target_df.to_csv(target_csv, index=False)

            config = {
                "experiment": {"seed": 202608},
                "data": {
                    "dataset": "nasa_to_satellite_battery",
                    "source_csv_path": str(source_csv),
                    "target_csv_path": str(target_csv),
                    "source_feature_columns": ["voltage", "current", "temperature"],
                    "target_feature_protocol": "telemetry_only",
                    "source_current_multiplier": -1.0,
                    "source_target_column": "normalized_rul",
                    "target_column": "normalized_rul",
                    "window_size": 12,
                    "source_stride": 2,
                    "target_stride": 4,
                    "max_rul": None,
                    "normalizer_scope": "source_target_train",
                },
                "train": {"batch_size": 4},
            }
            source_loader, target_loader, val_loader, test_loader, input_channels = (
                build_nasa_satellite_battery_transfer_loaders(config)
            )
            source_batch = next(iter(source_loader))
            target_batch = next(iter(target_loader))

            self.assertEqual(input_channels, len(TELEMETRY_ONLY_COLUMNS))
            self.assertEqual(source_batch["x"].shape[-1], len(TELEMETRY_ONLY_COLUMNS))
            self.assertEqual(target_batch["x"].shape[-1], len(TELEMETRY_ONLY_COLUMNS))
            self.assertGreater(float(source_batch["x"][..., 1].mean()), -5.0)
            self.assertGreater(len(val_loader.dataset), 0)
            self.assertGreater(len(test_loader.dataset), 0)

    def test_battery_physics_proxy_features_are_finite_and_non_leaking(self) -> None:
        df = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2],
                "cycle": [1, 2, 3, 1],
                "time_step": [0, 1, 2, 0],
                "voltage": [4.0, 3.9, 3.8, 4.1],
                "current": [0.5, 0.7, 0.7, -0.4],
                "temperature": [25.0, 25.5, 26.0, 24.0],
            }
        )

        out = add_battery_physics_proxy_features(df)

        self.assertTrue(set(BATTERY_PROXY_V1_COLUMNS).issubset(out.columns))
        self.assertTrue(set(BATTERY_PROXY_V1_COLUMNS).isdisjoint(LEAKAGE_COLUMNS))
        self.assertFalse(out[BATTERY_PROXY_V1_COLUMNS].isna().any().any())
        self.assertTrue(np.isfinite(out[BATTERY_PROXY_V1_COLUMNS].to_numpy(dtype=np.float64)).all())
        self.assertAlmostEqual(float(out.loc[0, "d_voltage"]), 0.0)
        self.assertLess(float(out.loc[1, "d_voltage"]), 0.0)
        self.assertGreaterEqual(float(out.loc[1, "abs_current"]), 0.0)

    def test_nasa_to_satellite_transfer_loaders_can_use_pg_stda_proxy_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_csv = tmp_path / "nasa_battery.csv"
            target_csv = tmp_path / "satellite_battery_sim.csv"
            _write_source_battery_csv(source_csv)
            target_df, _ = generate_satellite_battery_dataset(_small_config()["data"], seed=202608)
            target_df.to_csv(target_csv, index=False)
            proxy_columns = TELEMETRY_ONLY_COLUMNS + BATTERY_PROXY_V1_COLUMNS

            config = {
                "experiment": {"seed": 202608},
                "data": {
                    "dataset": "nasa_to_satellite_battery",
                    "source_csv_path": str(source_csv),
                    "target_csv_path": str(target_csv),
                    "battery_proxy_feature_set": "pg_stda_v1",
                    "source_feature_columns": proxy_columns,
                    "target_feature_columns": proxy_columns,
                    "source_current_multiplier": -1.0,
                    "source_target_column": "normalized_rul",
                    "target_column": "normalized_rul",
                    "target_stage_source": "time_progress",
                    "window_size": 12,
                    "source_stride": 2,
                    "target_stride": 4,
                    "max_rul": None,
                    "normalizer_scope": "source_target_train",
                },
                "train": {"batch_size": 4},
            }

            source_loader, target_loader, _, _, input_channels = build_nasa_satellite_battery_transfer_loaders(config)
            source_batch = next(iter(source_loader))
            target_batch = next(iter(target_loader))

            self.assertEqual(input_channels, len(proxy_columns))
            self.assertEqual(source_batch["x"].shape[-1], len(proxy_columns))
            self.assertEqual(target_batch["x"].shape[-1], len(proxy_columns))

    def test_nasa_to_satellite_transfer_loader_can_use_pseudo_time_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_csv = tmp_path / "nasa_battery.csv"
            target_csv = tmp_path / "satellite_battery_sim.csv"
            _write_source_battery_csv(source_csv)
            target_df, _ = generate_satellite_battery_dataset(_small_config()["data"], seed=202608)
            target_df.to_csv(target_csv, index=False)

            config = {
                "experiment": {"seed": 202608},
                "data": {
                    "dataset": "nasa_to_satellite_battery",
                    "source_csv_path": str(source_csv),
                    "target_csv_path": str(target_csv),
                    "source_feature_columns": ["voltage", "current", "temperature"],
                    "target_feature_protocol": "telemetry_only",
                    "source_current_multiplier": -1.0,
                    "source_target_column": "normalized_rul",
                    "target_column": "normalized_rul",
                    "target_stage_source": "time_progress",
                    "window_size": 12,
                    "source_stride": 2,
                    "target_stride": 4,
                    "max_rul": None,
                    "normalizer_scope": "source_target_train",
                },
                "train": {"batch_size": 4},
            }
            _, target_loader, _, _, _ = build_nasa_satellite_battery_transfer_loaders(config)
            stages = set(int(stage) for stage in target_loader.dataset.stage_labels)

            self.assertGreaterEqual(len(stages), 2)
            self.assertTrue(stages.issubset({0, 1, 2}))


if __name__ == "__main__":
    unittest.main()
