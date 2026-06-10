from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from xa202608.data.features import extract_low_freq_hi
from xa202608.data.reaction_wheel import LEAKAGE_COLUMNS, load_reaction_wheel_csv
from xa202608.data.windowing import WindowedTimeSeriesDataset, fit_normalizer
from xa202608.data.xjtu_sy import _load_processed_npz, _select_units


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", "raw", "unbounded"}:
        return None
    return float(value)


def _filter_units(source: dict[int, np.ndarray], ids: set[int]) -> dict[int, np.ndarray]:
    return {unit_id: value for unit_id, value in source.items() if int(unit_id) in ids}


def _time_progress_stage_by_unit(
    series_by_unit: dict[int, np.ndarray],
    bins: list[float] | tuple[float, ...] = (0.0, 0.3, 0.7, 1.0),
) -> dict[int, np.ndarray]:
    stage_bins = np.asarray(list(bins), dtype=np.float32)
    if len(stage_bins) < 2:
        raise ValueError("stage bins must contain at least two boundaries")
    stages: dict[int, np.ndarray] = {}
    for unit_id, values in series_by_unit.items():
        if len(values) <= 1:
            progress = np.zeros((len(values),), dtype=np.float32)
        else:
            progress = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
        stages[int(unit_id)] = np.digitize(progress, stage_bins[1:-1], right=False).astype(np.int64)
    return stages


def _validate_hi_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> None:
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"reaction_wheel target HI columns not found: {missing}")
    leakage = sorted(set(feature_columns) & LEAKAGE_COLUMNS)
    if leakage:
        raise ValueError(f"reaction_wheel target HI columns contain leakage columns: {leakage}")
    non_numeric = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"reaction_wheel target HI columns must be numeric: {non_numeric}")


def reaction_wheel_dataframe_to_hi_unit_dicts(
    df: pd.DataFrame,
    feature_columns: list[str],
    hi_window_size: int,
    hi_stride: int,
    target_column: str = "normalized_rul",
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    if int(hi_window_size) <= 0 or int(hi_stride) <= 0:
        raise ValueError("hi_window_size and hi_stride must be positive")
    if target_column not in df.columns:
        raise ValueError(f"reaction_wheel target column not found: {target_column}")
    _validate_hi_feature_columns(df, feature_columns)

    series_by_unit: dict[int, np.ndarray] = {}
    rul_by_unit: dict[int, np.ndarray] = {}
    for unit_id, group in df.sort_values(["unit_id", "time_step"]).groupby("unit_id", sort=False):
        values = group[feature_columns].to_numpy(dtype=np.float32)
        targets = group[target_column].to_numpy(dtype=np.float32)
        hi_rows: list[np.ndarray] = []
        hi_targets: list[float] = []
        for start in range(0, len(values) - int(hi_window_size) + 1, int(hi_stride)):
            end = start + int(hi_window_size)
            hi_rows.append(extract_low_freq_hi(values[start:end]))
            hi_targets.append(float(targets[end - 1]))
        if hi_rows:
            series_by_unit[int(unit_id)] = np.stack(hi_rows, axis=0).astype(np.float32)
            rul_by_unit[int(unit_id)] = np.asarray(hi_targets, dtype=np.float32)
    if not series_by_unit:
        raise ValueError("no reaction_wheel HI series generated; check hi_window_size and hi_stride")
    return series_by_unit, rul_by_unit


def build_reaction_wheel_hi_npz(
    csv_path: str | Path,
    output_path: str | Path,
    feature_columns: list[str],
    hi_window_size: int,
    hi_stride: int,
    target_column: str = "normalized_rul",
) -> Path:
    df = load_reaction_wheel_csv(csv_path)
    series_by_unit, rul_by_unit = reaction_wheel_dataframe_to_hi_unit_dicts(
        df,
        feature_columns=feature_columns,
        hi_window_size=int(hi_window_size),
        hi_stride=int(hi_stride),
        target_column=target_column,
    )
    unit_ids = np.asarray(sorted(series_by_unit), dtype=np.int64)
    split_by_unit = df.groupby("unit_id")["split"].first().to_dict()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        series=np.asarray([series_by_unit[int(unit_id)] for unit_id in unit_ids], dtype=object),
        rul=np.asarray([rul_by_unit[int(unit_id)] for unit_id in unit_ids], dtype=object),
        unit_ids=unit_ids,
        splits=np.asarray([str(split_by_unit[int(unit_id)]) for unit_id in unit_ids], dtype=object),
        feature_columns=np.asarray(feature_columns, dtype=object),
        hi_feature_names=np.asarray(_hi_feature_names(feature_columns), dtype=object),
        target_column=str(target_column),
        hi_window_size=int(hi_window_size),
        hi_stride=int(hi_stride),
    )
    return output


def _hi_feature_names(feature_columns: list[str]) -> list[str]:
    stats = ["rms", "std", "skewness", "kurtosis", "peak_to_peak", "crest_factor", "trend_slope"]
    return [f"{column}_{stat}" for stat in stats for column in feature_columns]


def _select_target_splits(
    series_all: dict[int, np.ndarray],
    rul_all: dict[int, np.ndarray],
    split_by_unit: dict[int, str],
    split_name: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    ids = {unit_id for unit_id, split in split_by_unit.items() if str(split) == str(split_name)}
    if not ids:
        raise ValueError(f"reaction_wheel target split is empty: {split_name}")
    return _filter_units(series_all, ids), _filter_units(rul_all, ids)


def build_reaction_wheel_hi_loaders(config: dict):
    data_cfg = config["data"]
    target_series_all, target_rul_all, split_by_unit = _load_or_build_target_hi(data_cfg)
    target_train_series, target_train_rul = _select_target_splits(
        target_series_all,
        target_rul_all,
        split_by_unit,
        str(data_cfg.get("target_train_split", "train")),
    )
    target_val_series, target_val_rul = _select_target_splits(
        target_series_all,
        target_rul_all,
        split_by_unit,
        str(data_cfg.get("target_val_split", "val")),
    )
    target_test_series, target_test_rul = _select_target_splits(
        target_series_all,
        target_rul_all,
        split_by_unit,
        str(data_cfg.get("target_test_split", "test")),
    )
    normalizer = fit_normalizer(list(target_train_series.values()), method=str(data_cfg.get("normalize", "zscore")))
    common_kwargs = {
        "window_size": int(data_cfg["window_size"]),
        "stride": int(data_cfg.get("stride", 1)),
        "max_rul": _optional_float(data_cfg.get("max_rul", None)),
        "normalizer": normalizer,
        "target_horizon": int(data_cfg.get("target_horizon", 0)),
    }
    train_ds = WindowedTimeSeriesDataset(target_train_series, target_train_rul, **common_kwargs)
    val_ds = WindowedTimeSeriesDataset(target_val_series, target_val_rul, **common_kwargs)
    test_ds = WindowedTimeSeriesDataset(target_test_series, target_test_rul, **common_kwargs)
    batch_size = int(config["train"]["batch_size"])
    input_dim = int(next(iter(target_series_all.values())).shape[1])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        input_dim,
    )


def _load_or_build_target_hi(data_cfg: dict[str, Any]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, str]]:
    csv_path = Path(data_cfg.get("reaction_wheel_csv_path", "data/simulated/reaction_wheel/reaction_wheel_sim.csv"))
    feature_columns = [str(x) for x in data_cfg.get("target_hi_feature_columns", ["motor_current", "vibration_proxy"])]
    hi_window_size = int(data_cfg.get("target_hi_window_size", 16))
    hi_stride = int(data_cfg.get("target_hi_stride", 8))
    target_column = str(data_cfg.get("target_column", "normalized_rul"))
    df = load_reaction_wheel_csv(csv_path)
    series, rul = reaction_wheel_dataframe_to_hi_unit_dicts(
        df,
        feature_columns=feature_columns,
        hi_window_size=hi_window_size,
        hi_stride=hi_stride,
        target_column=target_column,
    )
    split_by_unit = {int(unit_id): str(split) for unit_id, split in df.groupby("unit_id")["split"].first().to_dict().items()}
    return series, rul, split_by_unit


def build_xjtu_reaction_wheel_transfer_loaders(config: dict):
    data_cfg = config["data"]
    source_series_all, _ = _load_processed_npz(data_cfg.get("source_processed_path", "data/processed/xjtu_sy_hi.npz"))
    source_units = [str(x) for x in data_cfg["source_units"]]
    source_series, source_rul = _select_units(source_series_all, source_units)
    target_series_all, target_rul_all, split_by_unit = _load_or_build_target_hi(data_cfg)

    source_dim = int(next(iter(source_series.values())).shape[1])
    target_dim = int(next(iter(target_series_all.values())).shape[1])
    if source_dim != target_dim:
        raise ValueError(f"source and target feature dimensions must match, got {source_dim} and {target_dim}")

    target_train_series, target_train_rul = _select_target_splits(
        target_series_all,
        target_rul_all,
        split_by_unit,
        str(data_cfg.get("target_train_split", "train")),
    )
    target_val_series, target_val_rul = _select_target_splits(
        target_series_all,
        target_rul_all,
        split_by_unit,
        str(data_cfg.get("target_val_split", "val")),
    )
    target_test_series, target_test_rul = _select_target_splits(
        target_series_all,
        target_rul_all,
        split_by_unit,
        str(data_cfg.get("target_test_split", "test")),
    )

    normalize_scope = str(data_cfg.get("normalizer_scope", "source")).lower()
    if normalize_scope == "source":
        normalizer = fit_normalizer(list(source_series.values()), method=str(data_cfg.get("normalize", "zscore")))
    elif normalize_scope in {"source_target_train", "source+target_train"}:
        normalizer = fit_normalizer(
            list(source_series.values()) + list(target_train_series.values()),
            method=str(data_cfg.get("normalize", "zscore")),
        )
    else:
        raise ValueError(f"unknown normalizer_scope: {normalize_scope}")

    common_kwargs = {
        "window_size": int(data_cfg["window_size"]),
        "stride": int(data_cfg.get("stride", 1)),
        "max_rul": _optional_float(data_cfg.get("max_rul", None)),
        "normalizer": normalizer,
        "target_horizon": int(data_cfg.get("target_horizon", 0)),
    }
    target_stage_source = str(data_cfg.get("target_stage_source", "rul")).lower()
    target_stage_kwargs: dict[str, dict[int, np.ndarray]] = {}
    if target_stage_source in {"rul", "true_rul", "target_rul"}:
        target_stage_kwargs = {}
    elif target_stage_source in {"time", "time_progress", "progress", "pseudo_time"}:
        target_stage_kwargs = {"stage_by_unit": _time_progress_stage_by_unit(target_train_series)}
    else:
        raise ValueError(f"unknown target_stage_source: {target_stage_source}")
    source_ds = WindowedTimeSeriesDataset(source_series, source_rul, **common_kwargs)
    target_train_ds = WindowedTimeSeriesDataset(target_train_series, target_train_rul, **common_kwargs, **target_stage_kwargs)
    target_val_ds = WindowedTimeSeriesDataset(target_val_series, target_val_rul, **common_kwargs)
    target_test_ds = WindowedTimeSeriesDataset(target_test_series, target_test_rul, **common_kwargs)
    batch_size = int(config["train"]["batch_size"])
    return (
        DataLoader(source_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(target_train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(target_val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        DataLoader(target_test_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        source_dim,
    )
