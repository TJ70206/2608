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
from xa202608.data.reaction_wheel import (
    generate_reaction_wheel_dataset,
    validate_reaction_wheel_dataframe,
    write_reaction_wheel_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the reaction_wheel_sim self-built dataset.")
    parser.add_argument("--config", type=str, default="configs/sim_reaction_wheel.yaml", help="YAML config path.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory. Defaults to data.output_dir or data/simulated/reaction_wheel.",
    )
    return parser.parse_args()


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    config = load_config(args.config)
    data_cfg = dict(config.get("data", config))
    seed = int(config.get("experiment", {}).get("seed", data_cfg.get("seed", 42)))
    output_dir = Path(args.output_dir or data_cfg.get("output_dir", "data/simulated/reaction_wheel"))

    df, metadata = generate_reaction_wheel_dataset(data_cfg, seed=seed)
    summary = validate_reaction_wheel_dataframe(df)
    paths = write_reaction_wheel_outputs(df, metadata, output_dir=output_dir)

    print("reaction_wheel_sim generated")
    print(f"  rows: {summary['rows']}")
    print(f"  units: {summary['units']}")
    print(f"  split_unit_counts: {summary['split_unit_counts']}")
    print(f"  fault_counts_by_unit: {summary['fault_counts_by_unit']}")
    print(f"  eol_reason_counts_by_unit: {summary['eol_reason_counts_by_unit']}")
    for name, path in paths.items():
        resolved = path.resolve()
        try:
            display_path = resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = resolved
        print(f"  {name}: {display_path}")


if __name__ == "__main__":
    main()
