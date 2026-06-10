from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xa202608.data.xjtu_sy import convert_xjtu_root_to_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert XJTU-SY high-frequency CSV files to low-frequency HI NPZ.")
    parser.add_argument("--root", type=str, default="data/raw/xjtu_sy", help="XJTU-SY root directory.")
    parser.add_argument("--output", type=str, default="data/processed/xjtu_sy_hi.npz", help="Output NPZ path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = convert_xjtu_root_to_npz(PROJECT_ROOT / args.root, PROJECT_ROOT / args.output)
    print(f"saved XJTU-SY low-frequency HI to {output}")


if __name__ == "__main__":
    main()
