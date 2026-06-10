from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from xa202608.data.features import extract_low_freq_hi
from xa202608.data.windowing import WindowedTimeSeriesDataset, fit_normalizer


def _numeric_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits) if digits else 10**9, path.name


def read_xjtu_csv(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, header=None)
    numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    numeric = numeric.dropna(axis=0, how="any")
    if numeric.empty:
        raise ValueError(f"no numeric signal found in {csv_path}")
    return numeric.to_numpy(dtype=np.float32)


def bearing_folder_to_hi_sequence(bearing_dir: Path) -> np.ndarray:
    csv_files = sorted(bearing_dir.glob("*.csv"), key=_numeric_sort_key)
    if not csv_files:
        raise FileNotFoundError(f"no csv files found in bearing folder: {bearing_dir}")
    hi_rows = []
    for csv_path in csv_files:
        signal = read_xjtu_csv(csv_path)
        hi_rows.append(extract_low_freq_hi(signal))
    return np.stack(hi_rows, axis=0).astype(np.float32)


def convert_xjtu_root_to_npz(root_dir: str | Path, output_path: str | Path) -> Path:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"XJTU-SY root not found: {root}")
    series = []
    unit_ids = []
    condition_names = []
    for bearing_dir in sorted(root.glob("**/Bearing*")):
        if not bearing_dir.is_dir():
            continue
        try:
            hi = bearing_folder_to_hi_sequence(bearing_dir)
        except FileNotFoundError:
            continue
        series.append(hi)
        unit_ids.append(bearing_dir.name)
        condition_names.append(bearing_dir.parent.name)
    if not series:
        raise FileNotFoundError(
            f"No XJTU-SY bearing folders found under {root}. Expected folders like 35Hz12kN/Bearing1_1/*.csv"
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        series=np.asarray(series, dtype=object),
        unit_ids=np.asarray(unit_ids, dtype=object),
        conditions=np.asarray(condition_names, dtype=object),
    )
    return output


def _unit_name_to_int(unit_id: str) -> int:
    digits = "".join(ch if ch.isdigit() else " " for ch in unit_id).split()
    if len(digits) >= 2:
        return int(digits[0]) * 100 + int(digits[1])
    if digits:
        return int(digits[0])
    raise ValueError(f"Cannot convert XJTU-SY unit id to int: {unit_id}")


def _normalized_rul(length: int) -> np.ndarray:
    if length <= 1:
        return np.zeros((length,), dtype=np.float32)
    return (np.arange(length - 1, -1, -1, dtype=np.float32) / float(length - 1)).astype(np.float32)


def _load_processed_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"XJTU-SY processed NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    series = {str(uid): np.asarray(values, dtype=np.float32) for uid, values in zip(data["unit_ids"], data["series"])}
    conditions = {str(uid): str(cond) for uid, cond in zip(data["unit_ids"], data["conditions"])}
    return series, conditions


def _select_units(
    series_all: dict[str, np.ndarray],
    unit_names: list[str],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    series_by_unit: dict[int, np.ndarray] = {}
    rul_by_unit: dict[int, np.ndarray] = {}
    for unit_name in unit_names:
        if unit_name not in series_all:
            raise ValueError(f"XJTU-SY unit not found in processed data: {unit_name}")
        unit_key = _unit_name_to_int(unit_name)
        values = series_all[unit_name].astype(np.float32)
        series_by_unit[unit_key] = values
        rul_by_unit[unit_key] = _normalized_rul(len(values))
    return series_by_unit, rul_by_unit


def build_xjtu_transfer_loaders(config: dict):
    data_cfg = config["data"]
    series_all, _ = _load_processed_npz(data_cfg.get("processed_path", "data/processed/xjtu_sy_hi.npz"))
    source_units = [str(x) for x in data_cfg["source_units"]]
    target_train_units = [str(x) for x in data_cfg["target_train_units"]]
    target_val_units = [str(x) for x in data_cfg.get("target_val_units", target_train_units)]
    target_test_units = [str(x) for x in data_cfg["target_test_units"]]
    source_series, source_rul = _select_units(series_all, source_units)
    target_train_series, target_train_rul = _select_units(series_all, target_train_units)
    target_val_series, target_val_rul = _select_units(series_all, target_val_units)
    target_test_series, target_test_rul = _select_units(series_all, target_test_units)
    normalizer = fit_normalizer(list(source_series.values()))
    max_rul_value = data_cfg.get("max_rul")
    common_kwargs = {
        "window_size": int(data_cfg["window_size"]),
        "stride": int(data_cfg.get("stride", 1)),
        "max_rul": None if max_rul_value is None else float(max_rul_value),
        "normalizer": normalizer,
    }
    source_ds = WindowedTimeSeriesDataset(source_series, source_rul, **common_kwargs)
    target_train_ds = WindowedTimeSeriesDataset(target_train_series, target_train_rul, **common_kwargs)
    target_val_ds = WindowedTimeSeriesDataset(target_val_series, target_val_rul, **common_kwargs)
    target_test_ds = WindowedTimeSeriesDataset(target_test_series, target_test_rul, **common_kwargs)
    batch_size = int(config["train"]["batch_size"])
    return (
        DataLoader(source_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(target_train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(target_val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        DataLoader(target_test_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        int(next(iter(series_all.values())).shape[1]),
    )
