from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xa202608.data.satellite_battery import load_satellite_battery_csv, validate_satellite_battery_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check satellite_battery_sim dataset quality.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/simulated/satellite_battery/satellite_battery_sim.csv",
        help="Path to satellite_battery_sim.csv.",
    )
    return parser.parse_args()


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    df = load_satellite_battery_csv(args.data)
    summary = validate_satellite_battery_dataframe(df)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
