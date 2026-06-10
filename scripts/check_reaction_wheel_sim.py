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

from xa202608.data.reaction_wheel import load_reaction_wheel_csv, validate_reaction_wheel_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check reaction_wheel_sim dataset quality.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/simulated/reaction_wheel/reaction_wheel_sim.csv",
        help="Path to reaction_wheel_sim.csv.",
    )
    return parser.parse_args()


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    df = load_reaction_wheel_csv(args.data)
    summary = validate_reaction_wheel_dataframe(df)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
