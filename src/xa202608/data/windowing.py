from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    method: str = "zscore"
    condition_indices: np.ndarray | None = None
    condition_centers: np.ndarray | None = None


def get_stage_label(
    rul: np.ndarray | float | None = None,
    rul_start: np.ndarray | float | None = None,
    soh: np.ndarray | float | None = None,
    soh_eol: float = 0.8,
    bins: Iterable[float] = (0.0, 0.3, 0.7, 1.0),
) -> np.ndarray:
    stage_bins = np.asarray(list(bins), dtype=np.float32)
    if len(stage_bins) < 2:
        raise ValueError("stage bins must contain at least two boundaries")
    if soh is not None:
        progress = np.clip((1.0 - np.asarray(soh, dtype=np.float32)) / (1.0 - soh_eol), 0.0, 1.0)
    elif rul is not None and rul_start is not None:
        rul_arr = np.asarray(rul, dtype=np.float32)
        start_arr = np.maximum(np.asarray(rul_start, dtype=np.float32), 1e-6)
        progress = np.clip(1.0 - rul_arr / start_arr, 0.0, 1.0)
    else:
        raise ValueError("either soh or both rul and rul_start must be provided")
    labels = np.digitize(progress, stage_bins[1:-1], right=False)
    return labels.astype(np.int64)


def fit_normalizer(arrays: list[np.ndarray], method: str = "zscore") -> Normalizer:
    if not arrays:
        raise ValueError("arrays is empty")
    stacked = np.concatenate(arrays, axis=0)
    method = str(method).lower()
    if method in {"zscore", "standard", "standardize"}:
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return Normalizer(mean=mean.astype(np.float32), std=std.astype(np.float32), method="zscore")
    if method in {"minmax", "min-max"}:
        min_value = stacked.min(axis=0)
        max_value = stacked.max(axis=0)
        scale = max_value - min_value
        scale = np.where(scale < 1e-8, 1.0, scale)
        return Normalizer(mean=min_value.astype(np.float32), std=scale.astype(np.float32), method="minmax")
    raise ValueError(f"unsupported normalization method: {method}")


def _fit_kmeans(values: np.ndarray, num_clusters: int, seed: int, max_iter: int = 100) -> np.ndarray:
    if len(values) < num_clusters:
        raise ValueError("num_clusters cannot exceed number of rows")
    rng = np.random.default_rng(int(seed))
    centers = values[rng.choice(len(values), size=num_clusters, replace=False)].astype(np.float32)
    for _ in range(max_iter):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)
        next_centers = centers.copy()
        for idx in range(num_clusters):
            members = values[labels == idx]
            if len(members):
                next_centers[idx] = members.mean(axis=0)
        if np.allclose(next_centers, centers, atol=1e-6):
            break
        centers = next_centers.astype(np.float32)
    return centers.astype(np.float32)


def fit_condition_normalizer(
    arrays: list[np.ndarray],
    condition_indices: list[int] | np.ndarray,
    num_conditions: int = 6,
    seed: int = 42,
) -> Normalizer:
    if not arrays:
        raise ValueError("arrays is empty")
    stacked = np.concatenate(arrays, axis=0).astype(np.float32)
    indices = np.asarray(condition_indices, dtype=np.int64)
    condition_values = stacked[:, indices]
    centers = _fit_kmeans(condition_values, int(num_conditions), int(seed))
    distances = ((condition_values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)
    mean = np.zeros((int(num_conditions), stacked.shape[1]), dtype=np.float32)
    std = np.ones((int(num_conditions), stacked.shape[1]), dtype=np.float32)
    global_mean = stacked.mean(axis=0).astype(np.float32)
    global_std = stacked.std(axis=0).astype(np.float32)
    global_std = np.where(global_std < 1e-8, 1.0, global_std).astype(np.float32)
    for idx in range(int(num_conditions)):
        members = stacked[labels == idx]
        if len(members):
            mean[idx] = members.mean(axis=0)
            std[idx] = members.std(axis=0)
            std[idx] = np.where(std[idx] < 1e-8, 1.0, std[idx])
        else:
            mean[idx] = global_mean
            std[idx] = global_std
    return Normalizer(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        method="condition_zscore",
        condition_indices=indices,
        condition_centers=centers,
    )


def apply_normalizer(values: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    if normalizer.method == "condition_zscore":
        if normalizer.condition_indices is None or normalizer.condition_centers is None:
            raise ValueError("condition normalizer requires condition_indices and condition_centers")
        condition_values = values[:, normalizer.condition_indices]
        distances = ((condition_values[:, None, :] - normalizer.condition_centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)
        return ((values - normalizer.mean[labels]) / normalizer.std[labels]).astype(np.float32)
    return ((values - normalizer.mean) / normalizer.std).astype(np.float32)


def split_unit_ids(
    unit_ids: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_ids = np.asarray(sorted(set(unit_ids.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    n_total = len(unique_ids)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train + n_val]
    test_ids = unique_ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


class WindowedTimeSeriesDataset(Dataset):
    def __init__(
        self,
        series_by_unit: dict[int, np.ndarray],
        rul_by_unit: dict[int, np.ndarray],
        window_size: int,
        stride: int = 1,
        max_rul: float | None = None,
        normalizer: Normalizer | None = None,
        stage_bins: Iterable[float] = (0.0, 0.3, 0.7, 1.0),
        stage_by_unit: dict[int, np.ndarray] | None = None,
        pad_short: bool = False,
        target_horizon: int = 0,
    ) -> None:
        self.samples: list[np.ndarray] = []
        self.targets: list[float] = []
        self.unit_ids: list[int] = []
        self.end_indices: list[int] = []
        self.stage_labels: list[int] = []
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.target_horizon = int(target_horizon)
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError("window_size and stride must be positive")
        if self.target_horizon < 0:
            raise ValueError("target_horizon must be non-negative")
        for unit_id, values in series_by_unit.items():
            rul = rul_by_unit[unit_id].astype(np.float32)
            values = values.astype(np.float32)
            external_stage = None
            if stage_by_unit is not None:
                if unit_id not in stage_by_unit:
                    raise ValueError(f"stage labels missing for unit: {unit_id}")
                external_stage = np.asarray(stage_by_unit[unit_id], dtype=np.int64)
                if len(external_stage) != len(values):
                    raise ValueError(
                        f"stage label length mismatch for unit {unit_id}: {len(external_stage)} != {len(values)}"
                    )
            if normalizer is not None:
                values = apply_normalizer(values, normalizer)
            pad_offset = 0
            if len(values) < self.window_size:
                if not pad_short:
                    continue
                pad_offset = self.window_size - len(values)
                values = np.pad(values, ((pad_offset, 0), (0, 0)), mode="edge")
                rul = np.pad(rul, (pad_offset, 0), mode="edge")
                if external_stage is not None:
                    external_stage = np.pad(external_stage, (pad_offset, 0), mode="edge")
            rul_start = float(max(rul[0], 1.0))
            max_start = len(values) - self.window_size - self.target_horizon
            for start in range(0, max_start + 1, self.stride):
                end = start + self.window_size
                target_index = end - 1 + self.target_horizon
                target = float(rul[target_index])
                if max_rul is not None:
                    target = min(target, float(max_rul))
                if external_stage is not None:
                    stage = int(external_stage[target_index])
                else:
                    stage = int(get_stage_label(rul=target, rul_start=rul_start, bins=stage_bins))
                self.samples.append(values[start:end])
                self.targets.append(target)
                self.unit_ids.append(int(unit_id))
                self.end_indices.append(int(max(0, target_index - pad_offset)))
                self.stage_labels.append(stage)
        if not self.samples:
            raise ValueError("no windows were generated; check window_size and input lengths")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.tensor(self.samples[index], dtype=torch.float32),
            "y": torch.tensor([self.targets[index]], dtype=torch.float32),
            "unit_id": torch.tensor(self.unit_ids[index], dtype=torch.long),
            "time_index": torch.tensor(self.end_indices[index], dtype=torch.long),
            "stage": torch.tensor(self.stage_labels[index], dtype=torch.long),
        }


class FullSequenceTimeSeriesDataset(Dataset):
    def __init__(
        self,
        series_by_unit: dict[int, np.ndarray],
        rul_by_unit: dict[int, np.ndarray],
        max_rul: float | None = None,
        normalizer: Normalizer | None = None,
        stage_bins: Iterable[float] = (0.0, 0.3, 0.7, 1.0),
        random_end_trim: bool = False,
        trim_min: int = 10,
        trim_max: int = 75,
        min_length: int = 30,
        deterministic_trim: bool = False,
        seed: int = 42,
    ) -> None:
        self.series: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.unit_ids: list[int] = []
        self.stage_labels: list[np.ndarray] = []
        self.random_end_trim = bool(random_end_trim)
        self.trim_min = int(trim_min)
        self.trim_max = int(trim_max)
        self.min_length = int(min_length)
        self.deterministic_trim = bool(deterministic_trim)
        self.fixed_trims: list[int] = []
        rng = np.random.default_rng(seed)
        if self.min_length <= 0:
            raise ValueError("min_length must be positive")
        for unit_id, values in series_by_unit.items():
            rul = rul_by_unit[unit_id].astype(np.float32)
            values = values.astype(np.float32)
            if normalizer is not None:
                values = apply_normalizer(values, normalizer)
            if len(values) < self.min_length:
                continue
            targets = rul.copy()
            if max_rul is not None:
                targets = np.minimum(targets, float(max_rul))
            rul_start = float(max(rul[0], 1.0))
            stages = get_stage_label(rul=targets, rul_start=rul_start, bins=stage_bins)
            self.series.append(values)
            self.targets.append(targets.astype(np.float32))
            self.unit_ids.append(int(unit_id))
            self.stage_labels.append(stages.astype(np.int64))
            max_allowed = min(self.trim_max, len(values) - self.min_length)
            if self.random_end_trim and self.deterministic_trim:
                if max_allowed >= self.trim_min:
                    self.fixed_trims.append(int(rng.integers(self.trim_min, max_allowed + 1)))
                elif max_allowed > 0:
                    self.fixed_trims.append(int(rng.integers(0, max_allowed + 1)))
                else:
                    self.fixed_trims.append(0)
            else:
                self.fixed_trims.append(0)
        if not self.series:
            raise ValueError("no sequences were generated; check min_length and input lengths")

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        values = self.series[index]
        targets = self.targets[index]
        stages = self.stage_labels[index]
        end = len(values)
        if self.random_end_trim:
            if self.deterministic_trim:
                trim = self.fixed_trims[index]
            else:
                max_allowed = min(self.trim_max, end - self.min_length)
                if max_allowed >= self.trim_min:
                    trim = int(np.random.randint(self.trim_min, max_allowed + 1))
                elif max_allowed > 0:
                    trim = int(np.random.randint(0, max_allowed + 1))
                else:
                    trim = 0
            end -= trim
        unit_id = self.unit_ids[index]
        return {
            "x": torch.tensor(values[:end], dtype=torch.float32),
            "y": torch.tensor(targets[:end, None], dtype=torch.float32),
            "unit_id": torch.full((end,), unit_id, dtype=torch.long),
            "time_index": torch.arange(end, dtype=torch.long),
            "stage": torch.tensor(stages[:end], dtype=torch.long),
        }


def collate_sequence_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(int(item["x"].shape[0]) for item in batch)
    padded: dict[str, list[torch.Tensor]] = {"x": [], "y": [], "unit_id": [], "time_index": [], "stage": [], "mask": []}
    for item in batch:
        length = int(item["x"].shape[0])
        pad_len = max_len - length
        padded["x"].append(F.pad(item["x"], (0, 0, 0, pad_len)))
        padded["y"].append(F.pad(item["y"], (0, 0, 0, pad_len)))
        padded["unit_id"].append(F.pad(item["unit_id"], (0, pad_len), value=-1))
        padded["time_index"].append(F.pad(item["time_index"], (0, pad_len)))
        padded["stage"].append(F.pad(item["stage"], (0, pad_len)))
        padded["mask"].append(torch.cat([torch.ones(length, dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)]))
    return {key: torch.stack(value, dim=0) for key, value in padded.items()}
