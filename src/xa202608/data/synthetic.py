from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from xa202608.data.windowing import WindowedTimeSeriesDataset, fit_normalizer, split_unit_ids


def generate_synthetic_units(
    num_units: int,
    min_length: int,
    max_length: int,
    num_features: int,
    seed: int,
    domain_shift: float = 0.0,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    series_by_unit: dict[int, np.ndarray] = {}
    rul_by_unit: dict[int, np.ndarray] = {}
    for unit_id in range(num_units):
        length = int(rng.integers(min_length, max_length + 1))
        t = np.linspace(0.0, 1.0, length, dtype=np.float32)
        degradation = t ** rng.uniform(1.1, 2.2)
        features = []
        for feature_id in range(num_features):
            direction = rng.choice([-1.0, 1.0])
            slope = rng.uniform(0.3, 1.8) * direction
            periodic = 0.08 * np.sin(2 * np.pi * (feature_id + 1) * t + rng.uniform(0, np.pi))
            noise = rng.normal(0.0, 0.03, size=length)
            feature = slope * degradation + periodic + noise + rng.normal(0.0, 0.1)
            feature = feature + domain_shift * (feature_id + 1) / max(num_features, 1)
            features.append(feature.astype(np.float32))
        values = np.stack(features, axis=1).astype(np.float32)
        rul = np.arange(length - 1, -1, -1, dtype=np.float32)
        series_by_unit[unit_id] = values
        rul_by_unit[unit_id] = rul
    return series_by_unit, rul_by_unit


def _filter_units(source: dict[int, np.ndarray], ids: np.ndarray) -> dict[int, np.ndarray]:
    id_set = set(int(x) for x in ids.tolist())
    return {unit_id: value for unit_id, value in source.items() if unit_id in id_set}


def build_synthetic_debug_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    data_cfg = config["data"]
    seed = int(config["experiment"].get("seed", 42))
    series_by_unit, rul_by_unit = generate_synthetic_units(
        num_units=int(data_cfg["num_units"]),
        min_length=int(data_cfg["min_length"]),
        max_length=int(data_cfg["max_length"]),
        num_features=int(data_cfg["num_features"]),
        seed=seed,
    )
    all_ids = np.asarray(list(series_by_unit.keys()), dtype=np.int64)
    train_ids, val_ids, test_ids = split_unit_ids(
        all_ids,
        train_ratio=float(data_cfg["train_ratio"]),
        val_ratio=float(data_cfg["val_ratio"]),
        seed=seed,
    )
    train_series = _filter_units(series_by_unit, train_ids)
    val_series = _filter_units(series_by_unit, val_ids)
    test_series = _filter_units(series_by_unit, test_ids)
    train_rul = _filter_units(rul_by_unit, train_ids)
    val_rul = _filter_units(rul_by_unit, val_ids)
    test_rul = _filter_units(rul_by_unit, test_ids)
    normalizer = fit_normalizer(list(train_series.values()))
    common_kwargs = {
        "window_size": int(data_cfg["window_size"]),
        "stride": int(data_cfg.get("stride", 1)),
        "max_rul": float(data_cfg["max_rul"]),
        "normalizer": normalizer,
    }
    train_ds = WindowedTimeSeriesDataset(train_series, train_rul, **common_kwargs)
    val_ds = WindowedTimeSeriesDataset(val_series, val_rul, **common_kwargs)
    test_ds = WindowedTimeSeriesDataset(test_series, test_rul, **common_kwargs)
    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader, int(data_cfg["num_features"])


def build_synthetic_transfer_debug_loaders(config: dict):
    data_cfg = config["data"]
    seed = int(config["experiment"].get("seed", 42))
    source_series, source_rul = generate_synthetic_units(
        num_units=int(data_cfg["num_units_source"]),
        min_length=int(data_cfg["min_length"]),
        max_length=int(data_cfg["max_length"]),
        num_features=int(data_cfg["num_features"]),
        seed=seed,
        domain_shift=float(data_cfg.get("source_shift", 0.0)),
    )
    target_series_all, target_rul_all = generate_synthetic_units(
        num_units=int(data_cfg["num_units_target"]),
        min_length=int(data_cfg["min_length"]),
        max_length=int(data_cfg["max_length"]),
        num_features=int(data_cfg["num_features"]),
        seed=seed + 1000,
        domain_shift=float(data_cfg.get("target_shift", 0.0)),
    )
    target_ids = np.asarray(list(target_series_all.keys()), dtype=np.int64)
    target_train_ids, target_val_ids, target_test_ids = split_unit_ids(
        target_ids,
        train_ratio=float(data_cfg["train_ratio"]),
        val_ratio=float(data_cfg["val_ratio"]),
        seed=seed,
    )
    target_train_series = _filter_units(target_series_all, target_train_ids)
    target_val_series = _filter_units(target_series_all, target_val_ids)
    target_test_series = _filter_units(target_series_all, target_test_ids)
    target_train_rul = _filter_units(target_rul_all, target_train_ids)
    target_val_rul = _filter_units(target_rul_all, target_val_ids)
    target_test_rul = _filter_units(target_rul_all, target_test_ids)
    normalizer = fit_normalizer(list(source_series.values()))
    common_kwargs = {
        "window_size": int(data_cfg["window_size"]),
        "stride": int(data_cfg.get("stride", 1)),
        "max_rul": float(data_cfg["max_rul"]),
        "normalizer": normalizer,
    }
    source_ds = WindowedTimeSeriesDataset(source_series, source_rul, **common_kwargs)
    target_train_ds = WindowedTimeSeriesDataset(target_train_series, target_train_rul, **common_kwargs)
    target_val_ds = WindowedTimeSeriesDataset(target_val_series, target_val_rul, **common_kwargs)
    target_test_ds = WindowedTimeSeriesDataset(target_test_series, target_test_rul, **common_kwargs)
    batch_size = int(config["train"]["batch_size"])
    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    target_train_loader = DataLoader(target_train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    target_val_loader = DataLoader(target_val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    target_test_loader = DataLoader(target_test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return source_loader, target_train_loader, target_val_loader, target_test_loader, int(data_cfg["num_features"])
