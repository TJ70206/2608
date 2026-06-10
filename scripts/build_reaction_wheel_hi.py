from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xa202608.data.xjtu_reaction_wheel import build_reaction_wheel_hi_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 14-dimensional HI sequences for reaction_wheel_sim transfer.")
    parser.add_argument("--csv", type=str, default="data/simulated/reaction_wheel/reaction_wheel_sim.csv")
    parser.add_argument("--output", type=str, default="data/processed/reaction_wheel_sim_hi.npz")
    parser.add_argument("--columns", nargs="+", default=["motor_current", "vibration_proxy"])
    parser.add_argument("--hi-window-size", type=int, default=16)
    parser.add_argument("--hi-stride", type=int, default=8)
    parser.add_argument("--target-column", type=str, default="normalized_rul")
    return parser.parse_args()


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    output = build_reaction_wheel_hi_npz(
        csv_path=args.csv,
        output_path=args.output,
        feature_columns=[str(column) for column in args.columns],
        hi_window_size=int(args.hi_window_size),
        hi_stride=int(args.hi_stride),
        target_column=str(args.target_column),
    )
    print(f"reaction_wheel_sim HI saved: {output}")
    print(f"  columns: {args.columns}")
    print(f"  hi_window_size: {args.hi_window_size}")
    print(f"  hi_stride: {args.hi_stride}")
    print(f"  target_column: {args.target_column}")


if __name__ == "__main__":
    main()
