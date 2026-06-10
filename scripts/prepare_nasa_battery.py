from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NASA Battery .mat files to a processed cycle-level CSV.")
    parser.add_argument("--source", type=str, required=True, help="Directory containing B0005.mat, B0006.mat, B0007.mat and B0018.mat.")
    parser.add_argument("--raw-output", type=str, default="data/raw/nasa_battery", help="Project raw data output directory.")
    parser.add_argument("--output", type=str, default="data/processed/nasa_battery.csv", help="Processed CSV output path.")
    parser.add_argument("--soh-eol", type=float, default=0.8, help="SOH threshold used to derive RUL.")
    return parser.parse_args()


def _as_scalar(value: object) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    arr = np.real(arr.astype(np.complex128, copy=False)).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _mean_field(data: object, name: str) -> float:
    if not hasattr(data, name):
        return float("nan")
    return _as_scalar(getattr(data, name))


def _array_field(data: object, name: str) -> np.ndarray:
    if not hasattr(data, name):
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(getattr(data, name))
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    arr = np.real(arr.astype(np.complex128, copy=False)).reshape(-1)
    return arr[np.isfinite(arr)].astype(np.float64)


def _discharge_duration(time: np.ndarray) -> float:
    if time.size == 0:
        return float("nan")
    return float(np.max(time) - np.min(time))


def _discharge_energy(voltage: np.ndarray, current: np.ndarray, time: np.ndarray) -> float:
    n = min(len(voltage), len(current), len(time))
    if n < 2:
        return float("nan")
    voltage = voltage[:n]
    current = np.abs(current[:n])
    time = time[:n]
    order = np.argsort(time)
    return float(np.trapezoid(voltage[order] * current[order], time[order]) / 3600.0)


def _copy_raw_files(source_dir: Path, raw_output_dir: Path) -> list[Path]:
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in sorted(source_dir.glob("B*.mat")):
        target = raw_output_dir / path.name
        shutil.copy2(path, target)
        copied.append(target)
    readme = source_dir / "README.txt"
    if readme.exists():
        shutil.copy2(readme, raw_output_dir / readme.name)
    return copied


def _battery_id_from_path(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot infer battery id from {path.name}")
    return int(digits)


def _load_cycles(path: Path) -> list[object]:
    key = path.stem
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if key not in mat:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if not keys:
            raise ValueError(f"No battery key found in {path}")
        key = keys[0]
    cycles = mat[key].cycle
    if isinstance(cycles, np.ndarray):
        return list(cycles.reshape(-1))
    return [cycles]


def _convert_one_battery(path: Path, soh_eol: float) -> list[dict[str, float | int | str]]:
    unit_id = _battery_id_from_path(path)
    cycles = _load_cycles(path)
    impedance_by_global_cycle: dict[int, float] = {}
    rows: list[dict[str, float | int | str]] = []
    discharge_index = 0
    for global_idx, cycle in enumerate(cycles):
        cycle_type = str(cycle.type).lower()
        data = cycle.data
        if cycle_type == "impedance":
            re_value = _mean_field(data, "Re")
            rct_value = _mean_field(data, "Rct")
            internal_resistance = re_value + rct_value
            if np.isfinite(internal_resistance):
                impedance_by_global_cycle[global_idx] = internal_resistance
        elif cycle_type == "discharge":
            discharge_index += 1
            capacity = _mean_field(data, "Capacity")
            voltage_values = _array_field(data, "Voltage_measured")
            current_values = _array_field(data, "Current_measured")
            time_values = _array_field(data, "Time")
            rows.append(
                {
                    "unit_id": unit_id,
                    "battery_id": path.stem,
                    "cycle": discharge_index,
                    "global_cycle": global_idx,
                    "ambient_temperature": _as_scalar(cycle.ambient_temperature),
                    "voltage": _mean_field(data, "Voltage_measured"),
                    "current": _mean_field(data, "Current_measured"),
                    "temperature": _mean_field(data, "Temperature_measured"),
                    "min_voltage": float(np.min(voltage_values)) if voltage_values.size else float("nan"),
                    "discharge_duration": _discharge_duration(time_values),
                    "discharge_energy": _discharge_energy(voltage_values, current_values, time_values),
                    "capacity": capacity,
                    "internal_resistance": float("nan"),
                    "discharge_time": _mean_field(data, "Time"),
                }
            )
    if not rows:
        return rows
    impedance_items = sorted(impedance_by_global_cycle.items())
    for row in rows:
        global_cycle = int(row["global_cycle"])
        previous = [value for idx, value in impedance_items if idx <= global_cycle]
        next_values = [value for idx, value in impedance_items if idx > global_cycle]
        if previous:
            row["internal_resistance"] = float(previous[-1])
        elif next_values:
            row["internal_resistance"] = float(next_values[0])
    df = pd.DataFrame(rows)
    df["internal_resistance"] = df["internal_resistance"].interpolate(limit_direction="both")
    if df["internal_resistance"].isna().all():
        df["internal_resistance"] = 0.0
    elif df["internal_resistance"].isna().any():
        df["internal_resistance"] = df["internal_resistance"].fillna(float(df["internal_resistance"].median()))
    capacity0 = max(float(df["capacity"].iloc[0]), 1e-8)
    df["soh"] = df["capacity"] / capacity0
    eol_indices = np.where(df["soh"].to_numpy(dtype=np.float32) <= float(soh_eol))[0]
    eol_cycle = float(df["cycle"].iloc[int(eol_indices[0])]) if len(eol_indices) else float(df["cycle"].iloc[-1])
    df["rul"] = np.maximum(eol_cycle - df["cycle"].to_numpy(dtype=np.float32), 0.0)
    return df.to_dict(orient="records")


def convert_nasa_battery(source_dir: Path, raw_output_dir: Path, output_path: Path, soh_eol: float) -> pd.DataFrame:
    source_dir = source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"NASA Battery source directory not found: {source_dir}")
    copied = _copy_raw_files(source_dir, raw_output_dir)
    if not copied:
        raise FileNotFoundError(f"No B*.mat files found in {source_dir}")
    rows: list[dict[str, float | int | str]] = []
    for mat_path in copied:
        rows.extend(_convert_one_battery(mat_path, soh_eol=soh_eol))
    if not rows:
        raise ValueError("No discharge cycles were converted from NASA Battery files.")
    df = pd.DataFrame(rows).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    raw_output_dir = PROJECT_ROOT / args.raw_output
    output_path = PROJECT_ROOT / args.output
    df = convert_nasa_battery(source_dir, raw_output_dir, output_path, soh_eol=float(args.soh_eol))
    print(f"saved raw NASA Battery files to {raw_output_dir}")
    print(f"saved processed NASA Battery CSV to {output_path}")
    print(f"rows={len(df)} units={df['unit_id'].nunique()} columns={list(df.columns)}")
    print(df.groupby("unit_id").agg(cycles=("cycle", "max"), capacity_first=("capacity", "first"), capacity_last=("capacity", "last"), soh_last=("soh", "last"), rul_max=("rul", "max")).to_string())


if __name__ == "__main__":
    main()
