from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from xa202608.data.windowing import (
    FullSequenceTimeSeriesDataset,
    WindowedTimeSeriesDataset,
    collate_sequence_batch,
    fit_condition_normalizer,
    fit_normalizer,
    split_unit_ids,
)

INDEX_COLUMNS = ["unit_id", "cycle"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"s_{i}" for i in range(1, 22)]
CMAPSS_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def read_cmapss_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"C-MAPSS file not found: {path}. "
            "Please place train_FDxxx.txt, test_FDxxx.txt and RUL_FDxxx.txt under data/raw/cmapss/."
        )
    return pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)


def add_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = train_df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    out = train_df.merge(max_cycle, left_on="unit_id", right_index=True)
    out["rul"] = out["max_cycle"] - out["cycle"]
    return out.drop(columns=["max_cycle"])


def add_test_rul(test_df: pd.DataFrame, rul_path: Path) -> pd.DataFrame:
    if not rul_path.exists():
        raise FileNotFoundError(f"C-MAPSS RUL file not found: {rul_path}")
    final_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["final_rul"])
    final_rul["unit_id"] = np.arange(1, len(final_rul) + 1)
    max_cycle = test_df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    out = test_df.merge(max_cycle, left_on="unit_id", right_index=True)
    out = out.merge(final_rul, on="unit_id", how="left")
    out["rul"] = out["final_rul"] + (out["max_cycle"] - out["cycle"])
    return out.drop(columns=["max_cycle", "final_rul"])


def dataframe_to_unit_dicts(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    series_by_unit: dict[int, np.ndarray] = {}
    rul_by_unit: dict[int, np.ndarray] = {}
    for unit_id, group in df.sort_values(["unit_id", "cycle"]).groupby("unit_id"):
        series_by_unit[int(unit_id)] = group[feature_columns].to_numpy(dtype=np.float32)
        rul_by_unit[int(unit_id)] = group["rul"].to_numpy(dtype=np.float32)
    return series_by_unit, rul_by_unit


def _filter_units(source: dict[int, np.ndarray], ids: np.ndarray) -> dict[int, np.ndarray]:
    id_set = set(int(x) for x in ids.tolist())
    return {unit_id: value for unit_id, value in source.items() if unit_id in id_set}


def _trim_unit_ends(
    series_by_unit: dict[int, np.ndarray],
    rul_by_unit: dict[int, np.ndarray],
    trim_min: int,
    trim_max: int,
    seed: int,
    num_trims: int = 1,
    min_length: int = 1,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    trimmed_series: dict[int, np.ndarray] = {}
    trimmed_rul: dict[int, np.ndarray] = {}
    num_trims = max(1, int(num_trims))
    id_multiplier = 100000
    for trim_idx in range(num_trims):
        for unit_id in sorted(series_by_unit):
            values = series_by_unit[unit_id]
            rul = rul_by_unit[unit_id]
            max_trim = min(int(trim_max), max(0, len(values) - int(min_length)))
            min_trim = min(int(trim_min), max_trim)
            trim = int(rng.integers(min_trim, max_trim + 1)) if max_trim > 0 else 0
            keep_length = max(int(min_length), len(values) - trim)
            out_unit_id = int(unit_id) if num_trims == 1 else int(unit_id) + (trim_idx + 1) * id_multiplier
            trimmed_series[out_unit_id] = values[:keep_length]
            trimmed_rul[out_unit_id] = rul[:keep_length]
    return trimmed_series, trimmed_rul


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", "raw", "unbounded"}:
        return None
    return float(value)


def build_cmapss_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    data_cfg = config["data"]
    seed = int(config["experiment"].get("seed", 42))
    split_seed = int(data_cfg.get("split_seed", seed))
    root_dir = Path(data_cfg.get("root_dir", "data/raw/cmapss"))
    subset = str(data_cfg.get("subset", "FD001"))
    train_df = add_train_rul(read_cmapss_table(root_dir / f"train_{subset}.txt"))
    test_df = add_test_rul(read_cmapss_table(root_dir / f"test_{subset}.txt"), root_dir / f"RUL_{subset}.txt")
    drop_features = set(data_cfg.get("drop_features", []))
    feature_columns = [col for col in SETTING_COLUMNS + SENSOR_COLUMNS if col not in drop_features]
    train_series_all, train_rul_all = dataframe_to_unit_dicts(train_df, feature_columns)
    test_series, test_rul = dataframe_to_unit_dicts(test_df, feature_columns)
    all_train_ids = np.asarray(list(train_series_all.keys()), dtype=np.int64)
    train_ids, val_ids, _ = split_unit_ids(
        all_train_ids,
        train_ratio=1.0 - float(data_cfg.get("val_ratio", 0.15)),
        val_ratio=float(data_cfg.get("val_ratio", 0.15)),
        seed=split_seed,
    )
    train_series = _filter_units(train_series_all, train_ids)
    val_series = _filter_units(train_series_all, val_ids)
    train_rul = _filter_units(train_rul_all, train_ids)
    val_rul = _filter_units(train_rul_all, val_ids)
    if bool(data_cfg.get("train_on_full_train", False)):
        train_series = train_series_all
        train_rul = train_rul_all
    normalize_method = str(data_cfg.get("normalize", "zscore")).lower()
    if normalize_method in {"condition_zscore", "condition-zscore", "operating_condition_zscore"}:
        condition_indices = [feature_columns.index(col) for col in SETTING_COLUMNS if col in feature_columns]
        if not condition_indices:
            raise ValueError("condition_zscore requires setting_1/setting_2/setting_3 to be kept in feature_columns")
        normalizer = fit_condition_normalizer(
            list(train_series.values()),
            condition_indices=condition_indices,
            num_conditions=int(data_cfg.get("num_conditions", 6)),
            seed=int(data_cfg.get("condition_seed", split_seed)),
        )
    else:
        normalizer = fit_normalizer(list(train_series.values()), method=normalize_method)
    batch_size = int(config["train"]["batch_size"])
    training_mode = str(data_cfg.get("training_mode", "window")).lower()
    if training_mode in {"full_sequence", "full_sequence_random_trim", "sequence_random_trim"}:
        min_length = int(data_cfg.get("min_length", data_cfg.get("window_size", 30)))
        common_kwargs = {
            "max_rul": _optional_float(data_cfg.get("max_rul", 125)),
            "normalizer": normalizer,
            "min_length": min_length,
        }
        test_kwargs = {**common_kwargs, "max_rul": _optional_float(data_cfg.get("test_max_rul", data_cfg.get("max_rul", 125)))}
        val_random_end_trim = bool(data_cfg.get("val_random_end_trim", False))
        val_num_trims = int(data_cfg.get("val_num_trims", 1))
        if val_random_end_trim and val_num_trims > 1:
            val_series_for_ds, val_rul_for_ds = _trim_unit_ends(
                val_series,
                val_rul,
                trim_min=int(data_cfg.get("val_trim_min", data_cfg.get("trim_min", 10))),
                trim_max=int(data_cfg.get("val_trim_max", data_cfg.get("trim_max", 75))),
                seed=int(data_cfg.get("val_trim_seed", split_seed + 1000)),
                num_trims=val_num_trims,
                min_length=min_length,
            )
            val_random_end_trim_for_ds = False
        else:
            val_series_for_ds, val_rul_for_ds = val_series, val_rul
            val_random_end_trim_for_ds = val_random_end_trim
        train_ds = FullSequenceTimeSeriesDataset(
            train_series,
            train_rul,
            random_end_trim=bool(data_cfg.get("random_end_trim", training_mode != "full_sequence")),
            trim_min=int(data_cfg.get("trim_min", 10)),
            trim_max=int(data_cfg.get("trim_max", 75)),
            deterministic_trim=False,
            seed=seed,
            **common_kwargs,
        )
        val_ds = FullSequenceTimeSeriesDataset(
            val_series_for_ds,
            val_rul_for_ds,
            random_end_trim=val_random_end_trim_for_ds,
            trim_min=int(data_cfg.get("val_trim_min", data_cfg.get("trim_min", 10))),
            trim_max=int(data_cfg.get("val_trim_max", data_cfg.get("trim_max", 75))),
            deterministic_trim=True,
            seed=int(data_cfg.get("val_trim_seed", split_seed + 1000)),
            **common_kwargs,
        )
        test_ds = FullSequenceTimeSeriesDataset(test_series, test_rul, random_end_trim=False, **test_kwargs)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_sequence_batch,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_sequence_batch)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_sequence_batch)
    else:
        common_kwargs = {
            "window_size": int(data_cfg["window_size"]),
            "stride": int(data_cfg.get("stride", 1)),
            "max_rul": _optional_float(data_cfg.get("max_rul", 125)),
            "normalizer": normalizer,
        }
        test_kwargs = {**common_kwargs, "max_rul": _optional_float(data_cfg.get("test_max_rul", data_cfg.get("max_rul", 125)))}
        if bool(data_cfg.get("val_random_end_trim", False)):
            val_series_for_ds, val_rul_for_ds = _trim_unit_ends(
                val_series,
                val_rul,
                trim_min=int(data_cfg.get("val_trim_min", data_cfg.get("trim_min", 10))),
                trim_max=int(data_cfg.get("val_trim_max", data_cfg.get("trim_max", 75))),
                seed=int(data_cfg.get("val_trim_seed", split_seed + 1000)),
                num_trims=int(data_cfg.get("val_num_trims", 1)),
            )
        else:
            val_series_for_ds, val_rul_for_ds = val_series, val_rul
        train_ds = WindowedTimeSeriesDataset(train_series, train_rul, **common_kwargs)
        val_ds = WindowedTimeSeriesDataset(
            val_series_for_ds,
            val_rul_for_ds,
            pad_short=bool(data_cfg.get("val_pad_short", False)),
            **common_kwargs,
        )
        test_ds = WindowedTimeSeriesDataset(
            test_series,
            test_rul,
            pad_short=bool(data_cfg.get("test_pad_short", False)),
            **test_kwargs,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader, len(feature_columns)
