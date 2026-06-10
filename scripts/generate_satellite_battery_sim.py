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
from xa202608.data.satellite_battery import (
    generate_satellite_battery_dataset,
    validate_satellite_battery_dataframe,
    write_satellite_battery_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the satellite_battery_sim self-built dataset.")
    parser.add_argument("--config", type=str, default="configs/sim_satellite_battery.yaml", help="YAML config path.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory. Defaults to data.output_dir or data/simulated/satellite_battery.",
    )
    return parser.parse_args()


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    config = load_config(args.config)
    data_cfg = dict(config.get("data", config))
    seed = int(config.get("experiment", {}).get("seed", data_cfg.get("seed", 42)))
    output_dir = Path(args.output_dir or data_cfg.get("output_dir", "data/simulated/satellite_battery"))

    df, metadata = generate_satellite_battery_dataset(data_cfg, seed=seed)
    summary = validate_satellite_battery_dataframe(df)
    paths = write_satellite_battery_outputs(df, metadata, output_dir=output_dir)

    print("satellite_battery_sim generated")
    print(f"  rows: {summary['rows']}")
    print(f"  units: {summary['units']}")
    print(f"  split_unit_counts: {summary['split_unit_counts']}")
    print(f"  fault_counts_by_unit: {summary['fault_counts_by_unit']}")
    print(f"  eol_reason_counts_by_unit: {summary['eol_reason_counts_by_unit']}")
    print(f"  feature_protocol: {metadata.get('default_feature_protocol', 'telemetry_only')}")
    print(f"  default_input_columns: {metadata.get('telemetry_only_columns', [])}")
    for name, path in paths.items():
        resolved = path.resolve()
        try:
            display_path = resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = resolved
        print(f"  {name}: {display_path}")


if __name__ == "__main__":
    main()
