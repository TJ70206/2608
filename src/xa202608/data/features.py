from __future__ import annotations

import numpy as np


def extract_low_freq_hi(signal_window: np.ndarray) -> np.ndarray:
    values = np.asarray(signal_window, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("signal_window must have shape [time] or [time, channels]")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    rms = np.sqrt(np.mean(values ** 2, axis=0))
    centered = values - mean
    safe_std = np.where(std < 1e-8, 1.0, std)
    skewness = np.mean((centered / safe_std) ** 3, axis=0)
    kurtosis = np.mean((centered / safe_std) ** 4, axis=0)
    peak_to_peak = values.max(axis=0) - values.min(axis=0)
    crest_factor = np.max(np.abs(values), axis=0) / np.where(rms < 1e-8, 1.0, rms)
    t = np.arange(values.shape[0], dtype=np.float32)
    t_centered = t - t.mean()
    denom = np.sum(t_centered ** 2)
    if denom < 1e-8:
        trend_slope = np.zeros(values.shape[1], dtype=np.float32)
    else:
        trend_slope = (t_centered[:, None] * centered).sum(axis=0) / denom
    features = np.concatenate(
        [rms, std, skewness, kurtosis, peak_to_peak, crest_factor, trend_slope],
        axis=0,
    )
    return features.astype(np.float32)


def windowed_low_freq_hi(
    signal: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    outputs = []
    for start in range(0, len(values) - window_size + 1, stride):
        outputs.append(extract_low_freq_hi(values[start:start + window_size]))
    if not outputs:
        raise ValueError("no HI windows generated; check window_size and stride")
    return np.stack(outputs, axis=0).astype(np.float32)
