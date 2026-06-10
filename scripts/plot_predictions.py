from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xa202608.plotting import plot_predictions_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot XA-202608 prediction CSV files.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions_test.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for PNG figures.")
    parser.add_argument("--max_units", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = plot_predictions_csv(args.predictions, args.output_dir, max_units=args.max_units)
    for path in saved:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
