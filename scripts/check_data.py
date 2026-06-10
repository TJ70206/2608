from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def exists(path: Path) -> str:
    return "OK" if path.exists() else "MISSING"


def check_cmapss() -> None:
    root = PROJECT_ROOT / "data" / "raw" / "cmapss"
    print("[C-MAPSS]", root)
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        for prefix in ["train", "test", "RUL"]:
            path = root / f"{prefix}_{subset}.txt"
            print(f"  {exists(path):7s} {path.relative_to(PROJECT_ROOT)}")


def check_xjtu() -> None:
    root = PROJECT_ROOT / "data" / "raw" / "xjtu_sy"
    processed = PROJECT_ROOT / "data" / "processed" / "xjtu_sy_hi.npz"
    print("[XJTU-SY]", root)
    print(f"  {exists(root):7s} {root.relative_to(PROJECT_ROOT)}")
    print(f"  {exists(processed):7s} {processed.relative_to(PROJECT_ROOT)}")


def check_battery() -> None:
    raw = PROJECT_ROOT / "data" / "raw" / "nasa_battery"
    processed = PROJECT_ROOT / "data" / "processed" / "nasa_battery.csv"
    print("[NASA Battery]")
    print(f"  {exists(raw):7s} {raw.relative_to(PROJECT_ROOT)}")
    print(f"  {exists(processed):7s} {processed.relative_to(PROJECT_ROOT)}")


def check_reaction_wheel() -> None:
    root = PROJECT_ROOT / "data" / "simulated" / "reaction_wheel"
    csv_path = root / "reaction_wheel_sim.csv"
    metadata = root / "metadata.json"
    npz_path = root / "reaction_wheel_sim.npz"
    print("[reaction_wheel_sim]")
    print(f"  {exists(root):7s} {root.relative_to(PROJECT_ROOT)}")
    print(f"  {exists(csv_path):7s} {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"  {exists(metadata):7s} {metadata.relative_to(PROJECT_ROOT)}")
    print(f"  {exists(npz_path):7s} {npz_path.relative_to(PROJECT_ROOT)}")


def check_satellite_battery() -> None:
    root = PROJECT_ROOT / "data" / "simulated" / "satellite_battery"
    csv_path = root / "satellite_battery_sim.csv"
    metadata = root / "metadata.json"
    npz_path = root / "arrays.npz"
    print("[satellite_battery_sim]")
    print(f"  {exists(root):7s} {root.relative_to(PROJECT_ROOT)}")
    print(f"  {exists(csv_path):7s} {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"  {exists(metadata):7s} {metadata.relative_to(PROJECT_ROOT)}")
    print(f"  {exists(npz_path):7s} {npz_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check XA-202608 expected data files.")
    parser.add_argument("--all", action="store_true", help="Check all expected datasets.")
    parser.parse_args()
    check_cmapss()
    check_xjtu()
    check_battery()
    check_reaction_wheel()
    check_satellite_battery()


if __name__ == "__main__":
    main()
