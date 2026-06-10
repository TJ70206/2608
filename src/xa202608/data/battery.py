from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from xa202608.data.windowing import WindowedTimeSeriesDataset, fit_normalizer, split_unit_ids


def _filter_units(source: dict[int, np.ndarray], ids: np.ndarray) -> dict[int, np.ndarray]:
    id_set = set(int(x) for x in ids.tolist())
    return {unit_id: value for unit_id, value in source.items() if unit_id in id_set}


def load_processed_battery_csv(path: str | Path, soh_eol: float = 0.8) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Battery CSV not found: {csv_path}. Expected a processed CSV with unit_id, cycle, telemetry columns, and either rul/soh/capacity."
        )
    df = pd.read_csv(csv_path)
    required = {"unit_id", "cycle"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Battery CSV missing required columns: {sorted(missing)}")
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    if "soh" not in df.columns and "capacity" in df.columns:
        df["soh"] = df.groupby("unit_id")["capacity"].transform(lambda x: x / max(float(x.iloc[0]), 1e-8))
    if "rul" not in df.columns:
        if "soh" not in df.columns:
            raise ValueError("Battery CSV must contain rul, or soh/capacity so RUL can be estimated from EOL.")
        rul_values = []
        for _, group in df.groupby("unit_id", sort=False):
            cycles = group["cycle"].to_numpy(dtype=np.float32)
            soh = group["soh"].to_numpy(dtype=np.float32)
            eol_candidates = np.where(soh <= soh_eol)[0]
            eol_cycle = cycles[eol_candidates[0]] if len(eol_candidates) else cycles[-1]
            rul_values.extend(np.maximum(eol_cycle - cycles, 0.0).tolist())
        df["rul"] = np.asarray(rul_values, dtype=np.float32)
    return df


def battery_dataframe_to_unit_dicts(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "rul",
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    series_by_unit: dict[int, np.ndarray] = {}
    rul_by_unit: dict[int, np.ndarray] = {}
    if target_column not in df.columns:
        raise ValueError(f"Battery target column not found: {target_column}")
    for unit_id, group in df.sort_values(["unit_id", "cycle"]).groupby("unit_id"):
        series_by_unit[int(unit_id)] = group[feature_columns].to_numpy(dtype=np.float32)
        rul_by_unit[int(unit_id)] = group[target_column].to_numpy(dtype=np.float32)
    return series_by_unit, rul_by_unit


def infer_battery_feature_columns(df: pd.DataFrame, explicit: list[str] | None = None) -> list[str]:
    if explicit:
        missing = [col for col in explicit if col not in df.columns]
        if missing:
            raise ValueError(f"Battery feature columns not found: {missing}")
        return explicit
    excluded = {"unit_id", "cycle", "time_step", "rul", "soh", "fault_type", "split"}
    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    features = [col for col in numeric_columns if col not in excluded]
    if not features:
        raise ValueError("No numeric battery feature columns inferred.")
    return features


def build_battery_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    data_cfg = config["data"]
    seed = int(config["experiment"].get("seed", 42))
    csv_path = Path(data_cfg.get("csv_path", "data/processed/nasa_battery.csv"))
    df = load_processed_battery_csv(csv_path, soh_eol=float(data_cfg.get("soh_eol", 0.8)))
    feature_columns = infer_battery_feature_columns(df, data_cfg.get("feature_columns"))
    target_column = str(data_cfg.get("target_column", "rul"))
    series_all, rul_all = battery_dataframe_to_unit_dicts(df, feature_columns, target_column=target_column)
    all_ids = np.asarray(list(series_all.keys()), dtype=np.int64)
    train_ids, val_ids, test_ids = split_unit_ids(
        all_ids,
        train_ratio=float(data_cfg.get("train_ratio", 0.7)),
        val_ratio=float(data_cfg.get("val_ratio", 0.15)),
        seed=seed,
    )
    train_series = _filter_units(series_all, train_ids)
    val_series = _filter_units(series_all, val_ids)
    test_series = _filter_units(series_all, test_ids)
    train_rul = _filter_units(rul_all, train_ids)
    val_rul = _filter_units(rul_all, val_ids)
    test_rul = _filter_units(rul_all, test_ids)
    normalizer = fit_normalizer(list(train_series.values()))
    max_rul_value = data_cfg.get("max_rul", 500)
    common_kwargs = {
        "window_size": int(data_cfg["window_size"]),
        "stride": int(data_cfg.get("stride", 1)),
        "max_rul": None if max_rul_value is None else float(max_rul_value),
        "normalizer": normalizer,
        "target_horizon": int(data_cfg.get("target_horizon", 0)),
    }
    train_ds = WindowedTimeSeriesDataset(train_series, train_rul, **common_kwargs)
    val_ds = WindowedTimeSeriesDataset(val_series, val_rul, **common_kwargs)
    test_ds = WindowedTimeSeriesDataset(test_series, test_rul, **common_kwargs)
    batch_size = int(config["train"]["batch_size"])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        len(feature_columns),
    )
