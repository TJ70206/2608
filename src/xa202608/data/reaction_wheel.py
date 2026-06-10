from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from xa202608.data.windowing import WindowedTimeSeriesDataset, fit_normalizer, split_unit_ids

TELEMETRY_ONLY_COLUMNS = [
    "wheel_speed",
    "motor_current",
    "command_voltage",
    "temperature",
    "vibration_proxy",
]
TELEMETRY_PLUS_PROXY_COLUMNS = TELEMETRY_ONLY_COLUMNS + ["friction_torque_proxy"]

LEAKAGE_COLUMNS = {
    "kt",
    "kt_ratio",
    "lubricant_loss",
    "lubricant_hi",
    "friction_torque",
    "friction_torque_magnitude",
    "degradation_progress",
    "health_stage",
    "rul",
    "normalized_rul",
    "eol_index",
    "eol_reason",
    "is_censored",
    "split",
}

REQUIRED_COLUMNS = {
    "unit_id",
    "time_step",
    "mission_time_s",
    "command_voltage",
    "wheel_speed",
    "wheel_speed_rpm",
    "motor_current",
    "temperature",
    "vibration_proxy",
    "friction_torque_proxy",
    "torque_noise_proxy",
    "kt",
    "kt_ratio",
    "lubricant_loss",
    "lubricant_hi",
    "friction_torque",
    "friction_torque_magnitude",
    "degradation_progress",
    "health_stage",
    "rul",
    "normalized_rul",
    "base_degradation",
    "explicit_fault_injection_type",
    "is_censored",
    "split",
}

OBSERVED_COLUMNS = TELEMETRY_PLUS_PROXY_COLUMNS + [
    "wheel_speed_rpm",
    "torque_noise_proxy",
]


def _as_float(data_cfg: dict[str, Any], key: str, default: float) -> float:
    value = data_cfg.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _as_int(data_cfg: dict[str, Any], key: str, default: int) -> int:
    value = data_cfg.get(key, default)
    if value is None:
        return int(default)
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", "raw", "unbounded"}:
        return None
    return float(value)


def _filter_units(source: dict[int, np.ndarray], ids: np.ndarray) -> dict[int, np.ndarray]:
    id_set = set(int(x) for x in ids.tolist())
    return {unit_id: value for unit_id, value in source.items() if int(unit_id) in id_set}


def assign_unit_splits(
    unit_ids: list[int] | np.ndarray,
    split_ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[int, str]:
    ratios = split_ratios or {"train": 0.6, "val": 0.15, "calib": 0.1, "test": 0.15}
    names = ["train", "val", "calib", "test"]
    weights = np.asarray([max(float(ratios.get(name, 0.0)), 0.0) for name in names], dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("split ratios must contain at least one positive value")
    weights = weights / weights.sum()

    ids = np.asarray(sorted(int(x) for x in unit_ids), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(ids)
    n_total = len(ids)
    counts = np.floor(weights * n_total).astype(int)
    remainder = n_total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(weights * n_total - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    if n_total >= len(names):
        for idx in range(len(names)):
            if weights[idx] > 0 and counts[idx] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] <= 1:
                    break
                counts[donor] -= 1
                counts[idx] += 1
    assigned: dict[int, str] = {}
    start = 0
    for name, count in zip(names, counts, strict=True):
        stop = start + int(count)
        for unit_id in ids[start:stop]:
            assigned[int(unit_id)] = name
        start = stop
    return assigned


def find_first_eol_index(
    kt_ratio: np.ndarray,
    lubricant_hi: np.ndarray,
    friction_torque: np.ndarray,
    kt_threshold_ratio: float,
    lubricant_threshold: float,
    friction_threshold: float,
) -> tuple[int, str]:
    kt_ratio = np.asarray(kt_ratio, dtype=np.float64)
    lubricant_hi = np.asarray(lubricant_hi, dtype=np.float64)
    friction_magnitude = np.abs(np.asarray(friction_torque, dtype=np.float64))
    if not (len(kt_ratio) == len(lubricant_hi) == len(friction_magnitude)):
        raise ValueError("EOL arrays must have equal length")

    triggers: list[tuple[int, str]] = []
    kt_hits = np.flatnonzero(kt_ratio <= float(kt_threshold_ratio))
    if len(kt_hits):
        triggers.append((int(kt_hits[0]), "kt_threshold"))
    lubricant_hits = np.flatnonzero(lubricant_hi <= float(lubricant_threshold))
    if len(lubricant_hits):
        triggers.append((int(lubricant_hits[0]), "lubricant_threshold"))
    friction_hits = np.flatnonzero(friction_magnitude >= float(friction_threshold))
    if len(friction_hits):
        triggers.append((int(friction_hits[0]), "friction_magnitude_threshold"))
    if not triggers:
        return max(len(kt_ratio) - 1, 0), "end_of_record"
    return min(triggers, key=lambda item: item[0])


def _command_profile(length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, int(length), dtype=np.float64)
    centered = (
        1.4 * np.sin(2.0 * np.pi * (2.0 + rng.uniform(-0.25, 0.25)) * t + rng.uniform(0, 2 * np.pi))
        + 0.75 * np.sin(2.0 * np.pi * (7.0 + rng.uniform(-0.5, 0.5)) * t + rng.uniform(0, 2 * np.pi))
        + 0.35 * np.sign(np.sin(2.0 * np.pi * (3.0 + rng.uniform(-0.4, 0.4)) * t))
    )
    for _ in range(int(rng.integers(3, 6))):
        start = int(rng.integers(0, max(1, length - 12)))
        stop = min(length, start + int(rng.integers(6, 22)))
        centered[start:stop] += rng.uniform(-1.2, 1.2)
    for _ in range(int(rng.integers(2, 5))):
        start = int(rng.integers(0, max(1, length - 16)))
        stop = min(length, start + int(rng.integers(8, 28)))
        centered[start:stop] *= rng.uniform(0.04, 0.18)
        if stop - start > 4:
            centered[start:stop] += np.linspace(-0.18, 0.18, stop - start)
    centered += rng.normal(0.0, 0.04, size=length)
    centered = np.clip(centered, -2.35, 2.35)
    command_voltage = np.clip(centered + 2.5, 0.0, 5.0)
    return centered.astype(np.float64), command_voltage.astype(np.float64)


def _apply_missing_and_interpolate(
    df: pd.DataFrame,
    observed_columns: list[str],
    missing_rate: float,
    burst_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if missing_rate <= 0 and burst_rate <= 0:
        for column in observed_columns:
            df[f"missing_mask_{column}"] = False
        return df
    out = df.copy()
    for column in observed_columns:
        mask = np.zeros(len(out), dtype=bool)
        for _, group in out.groupby("unit_id", sort=False):
            indices = group.index.to_numpy()
            if missing_rate > 0:
                mask[indices] |= rng.random(len(indices)) < float(missing_rate)
            if burst_rate > 0:
                starts = rng.random(len(indices)) < float(burst_rate)
                for local_start in np.flatnonzero(starts):
                    burst_len = int(rng.integers(2, 8))
                    mask[indices[local_start:local_start + burst_len]] = True
        out[f"missing_mask_{column}"] = mask
        out.loc[mask, column] = np.nan
    for column in observed_columns:
        out[column] = (
            out.groupby("unit_id", sort=False)[column]
            .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
            .astype(np.float64)
        )
    return out


def _generate_unit(
    unit_id: int,
    split: str,
    data_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    length = int(rng.integers(_as_int(data_cfg, "min_length", 600), _as_int(data_cfg, "max_length", 1600) + 1))
    time_step_seconds = _as_float(data_cfg, "time_step_seconds", 60.0)
    target_low = max(24, int(length * _as_float(data_cfg, "target_eol_min_fraction", 0.78)))
    target_high = max(target_low + 1, int(length * _as_float(data_cfg, "target_eol_max_fraction", 0.94)))
    target_eol = int(rng.integers(target_low, min(length - 1, target_high) + 1))

    kt0 = float(np.clip(rng.normal(_as_float(data_cfg, "kt0_mean", 0.029), _as_float(data_cfg, "kt0_std", 0.0012)), 0.023, 0.034))
    kt_threshold_ratio = _as_float(data_cfg, "kt_threshold_ratio", 0.3)
    kt_decay_rate = -math.log(kt_threshold_ratio) / max(float(target_eol), 1.0)

    tau_c0 = float(_as_float(data_cfg, "coulomb_friction", 0.002) * rng.uniform(0.8, 1.2))
    static_ratio = float(rng.uniform(1.06, 1.18))
    viscous_coeff0 = float(_as_float(data_cfg, "viscous_coeff", 5.0e-6) * rng.uniform(0.7, 1.4))
    omega_stribeck = float(_as_float(data_cfg, "omega_stribeck", 0.42) * rng.uniform(0.7, 1.6))
    inertia = float(_as_float(data_cfg, "inertia", 0.0077) * rng.uniform(0.9, 1.12))
    drive_gain = _as_float(data_cfg, "drive_gain", 0.19)
    max_speed = _as_float(data_cfg, "speed_limit_rad_s", 680.0)
    dynamics_dt = _as_float(data_cfg, "dynamics_dt", 0.018)
    command_response = _as_float(data_cfg, "command_response", 0.10)

    fault_type = "none"
    explicit_fault_fraction = _as_float(data_cfg, "explicit_fault_fraction", 0.25)
    if rng.random() < explicit_fault_fraction:
        fault_type = str(rng.choice(["lubrication_accelerated_loss", "friction_step"]))
    fault_start = int(rng.integers(max(10, int(length * 0.45)), max(11, int(length * 0.75))))
    fault_step_factor = float(rng.uniform(1.7, 2.8))
    fault_loss_multiplier = float(rng.uniform(1.8, 3.5))

    lubricant_type = str(rng.choice(["nominal", "low_temp_sensitive", "high_loss"], p=[0.55, 0.25, 0.20]))
    lubricant_threshold = _as_float(data_cfg, "lubricant_threshold", 0.08)
    temperature_sensitivity = {
        "nominal": 3.0,
        "low_temp_sensitive": 4.4,
        "high_loss": 3.6,
    }[lubricant_type]
    beta_scale = {
        "nominal": 1.0,
        "low_temp_sensitive": 0.92,
        "high_loss": 1.25,
    }[lubricant_type]

    idx = np.arange(length, dtype=np.float64)
    command_centered, command_voltage = _command_profile(length, rng)
    desired_speed = command_centered / 2.5 * max_speed * rng.uniform(0.72, 0.90)
    progress_to_target = np.clip(idx / max(float(target_eol), 1.0), 0.0, 1.5)

    temperature_bias = rng.normal(0.0, 2.0)
    orbit_period = float(rng.uniform(75.0, 140.0))
    ambient_temp = (
        22.0
        + temperature_bias
        + 7.0 * np.sin(2.0 * np.pi * idx / orbit_period + rng.uniform(0, 2 * np.pi))
        + 1.5 * np.sin(2.0 * np.pi * idx / (orbit_period * 0.43) + rng.uniform(0, 2 * np.pi))
    )
    preliminary_temp = (
        ambient_temp
        + 0.0028 * np.abs(desired_speed)
        + 6.5 * np.power(progress_to_target, 1.7)
        + rng.normal(0.0, 0.35, size=length)
    )
    if fault_type == "lubrication_accelerated_loss":
        preliminary_temp[fault_start:] += np.linspace(0.8, 4.0, length - fault_start)
    preliminary_temp = np.clip(preliminary_temp, -10.0, 60.0)

    temp_kelvin = preliminary_temp + 273.15
    temp_ref_kelvin = _as_float(data_cfg, "temperature_reference_kelvin", 296.15)
    temp_factor = np.exp(temperature_sensitivity * (temp_kelvin - temp_ref_kelvin) / temp_ref_kelvin)
    fault_multiplier = np.ones(length, dtype=np.float64)
    if fault_type == "lubrication_accelerated_loss":
        fault_multiplier[fault_start:] *= fault_loss_multiplier
    lubricant_eol_target = max(float(target_eol) * rng.uniform(0.92, 1.18), 1.0)
    base_loss_rate = (1.0 - lubricant_threshold) / lubricant_eol_target / max(float(np.mean(temp_factor[:target_eol])), 1e-8)
    loss_rate = base_loss_rate * beta_scale * temp_factor * fault_multiplier * (0.65 + 0.55 * progress_to_target)
    loss_rate += rng.normal(0.0, base_loss_rate * 0.05, size=length)
    loss_rate = np.clip(loss_rate, base_loss_rate * 0.05, None)
    lubricant_loss = np.maximum.accumulate(np.cumsum(loss_rate))
    lubricant_hi = np.clip(1.0 - lubricant_loss, 0.0, 1.0)

    kt_ratio = np.exp(-kt_decay_rate * idx)
    kt = kt0 * kt_ratio
    wheel_speed = np.zeros(length, dtype=np.float64)
    motor_current = np.zeros(length, dtype=np.float64)
    friction_torque = np.zeros(length, dtype=np.float64)
    torque_noise_proxy = np.zeros(length, dtype=np.float64)
    temperature = preliminary_temp.copy()
    vibration_proxy = np.zeros(length, dtype=np.float64)

    current_response = float(np.clip(_as_float(data_cfg, "current_response", 0.22), 0.02, 0.95))
    disturbance_amp = _as_float(data_cfg, "periodic_disturbance_amplitude", 4.0e-4)
    noise_scale = _as_float(data_cfg, "noise_scale", 0.01)
    friction_threshold = tau_c0 * _as_float(data_cfg, "friction_threshold_multiplier", 6.0)

    previous_speed = 0.0
    previous_current = 0.0
    for i in range(length):
        damage = float(np.clip(1.0 - lubricant_hi[i], 0.0, 1.0))
        temp_modifier = float(np.clip(np.exp(-0.018 * (temperature[i] - 23.0)), 0.35, 1.9))
        friction_growth = 1.0 + 2.4 * math.pow(damage, 1.35)
        if fault_type == "friction_step" and i >= fault_start:
            friction_growth *= fault_step_factor
        tau_c = tau_c0 * friction_growth
        tau_s = tau_c * static_ratio
        viscous_coeff = viscous_coeff0 * temp_modifier * (1.0 + 1.7 * math.pow(damage, 1.2))

        sign_value = math.copysign(1.0, previous_speed if abs(previous_speed) > 1e-5 else command_centered[i])
        if abs(previous_speed) <= 1e-5 and abs(command_centered[i]) <= 1e-5:
            sign_value = 0.0
        stribeck = sign_value * (tau_c + (tau_s - tau_c) * math.exp(-((previous_speed / max(omega_stribeck, 1e-6)) ** 2)))
        viscous = viscous_coeff * previous_speed
        local_noise = rng.normal(0.0, tau_c0 * (0.015 + 0.08 * damage))
        friction_torque[i] = viscous + stribeck + local_noise
        torque_noise_proxy[i] = abs(local_noise) + abs(friction_torque[i]) * (0.025 + 0.12 * damage)

        desired_current = drive_gain * command_centered[i]
        previous_current = previous_current + current_response * (desired_current - previous_current)
        previous_current += rng.normal(0.0, max(0.002, abs(desired_current) * noise_scale * 0.18))
        motor_current[i] = previous_current

        motor_torque = kt[i] * previous_current
        disturbance = disturbance_amp * math.sin(36.0 * (idx[i] / max(orbit_period, 1.0)) + rng.uniform(-0.05, 0.05))
        acceleration = (motor_torque - friction_torque[i] - disturbance) / max(inertia, 1e-8)
        next_speed = 0.985 * previous_speed + dynamics_dt * acceleration + command_response * (desired_speed[i] - previous_speed)
        next_speed += rng.normal(0.0, 0.45 + 0.75 * damage)
        previous_speed = float(np.clip(next_speed, -0.98 * max_speed, 0.98 * max_speed))
        wheel_speed[i] = previous_speed

        low_speed_factor = math.exp(-((abs(previous_speed) / 45.0) ** 2))
        vibration_proxy[i] = (
            0.018
            + 0.000035 * abs(previous_speed)
            + 0.06 * math.pow(max(progress_to_target[i], 0.0), 1.8)
            + 0.18 * math.pow(damage, 1.7)
            + 0.035 * low_speed_factor * (1.0 + 3.0 * damage)
            + rng.normal(0.0, 0.004)
        )
        temperature[i] = np.clip(
            preliminary_temp[i] + 0.0018 * abs(previous_speed) + 120.0 * abs(friction_torque[i]) + rng.normal(0.0, 0.25),
            -12.0,
            68.0,
        )

    vibration_proxy = np.clip(vibration_proxy, 0.0, None)
    friction_proxy_noise = rng.normal(0.0, tau_c0 * (0.08 + 0.08 * np.clip(1.0 - lubricant_hi, 0.0, 1.0)), size=length)
    friction_torque_proxy = friction_torque + friction_proxy_noise
    wheel_speed_rpm = wheel_speed * 60.0 / (2.0 * np.pi)

    eol_idx, eol_reason = find_first_eol_index(
        kt_ratio=kt_ratio,
        lubricant_hi=lubricant_hi,
        friction_torque=friction_torque,
        kt_threshold_ratio=kt_threshold_ratio,
        lubricant_threshold=lubricant_threshold,
        friction_threshold=friction_threshold,
    )
    is_censored = eol_reason == "end_of_record"
    keep_len = min(length, eol_idx + 1 + _as_int(data_cfg, "post_eol_padding", 0))
    eol_for_rul = int(eol_idx)
    rul = np.maximum(eol_for_rul - idx, 0.0)
    normalized_rul = rul / max(float(rul[0]), 1.0)
    degradation_progress = np.clip(1.0 - normalized_rul, 0.0, 1.0)
    health_stage = np.digitize(degradation_progress, [0.3, 0.7]).astype(np.int64)

    rows = pd.DataFrame(
        {
            "unit_id": int(unit_id),
            "time_step": idx.astype(np.int64),
            "mission_time_s": idx * time_step_seconds,
            "command_voltage": command_voltage,
            "wheel_speed": wheel_speed,
            "wheel_speed_rpm": wheel_speed_rpm,
            "motor_current": motor_current,
            "temperature": temperature,
            "vibration_proxy": vibration_proxy,
            "friction_torque_proxy": friction_torque_proxy,
            "torque_noise_proxy": torque_noise_proxy,
            "kt": kt,
            "kt_ratio": kt_ratio,
            "lubricant_loss": lubricant_loss,
            "lubricant_hi": lubricant_hi,
            "friction_torque": friction_torque,
            "friction_torque_magnitude": np.abs(friction_torque),
            "degradation_progress": degradation_progress,
            "health_stage": health_stage,
            "rul": rul,
            "normalized_rul": normalized_rul,
            "base_degradation": "kt_exponential_decay",
            "explicit_fault_injection_type": fault_type,
            "lubricant_type": lubricant_type,
            "is_censored": bool(is_censored),
            "eol_index": int(eol_idx),
            "eol_reason": eol_reason,
            "split": split,
        }
    ).iloc[:keep_len].reset_index(drop=True)

    unit_metadata = {
        "unit_id": int(unit_id),
        "split": split,
        "length": int(len(rows)),
        "raw_length": int(length),
        "target_eol_index": int(target_eol),
        "eol_index": int(eol_idx),
        "eol_reason": eol_reason,
        "is_censored": bool(is_censored),
        "explicit_fault_injection_type": fault_type,
        "fault_start": int(fault_start),
        "lubricant_type": lubricant_type,
        "kt0": kt0,
        "kt_decay_rate_per_step": kt_decay_rate,
        "tau_c0": tau_c0,
        "viscous_coeff0": viscous_coeff0,
        "friction_threshold": friction_threshold,
        "lubricant_threshold": lubricant_threshold,
    }
    return rows, unit_metadata


def generate_reaction_wheel_dataset(
    data_cfg: dict[str, Any] | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = dict(data_cfg or {})
    if seed is None:
        seed = _as_int(cfg, "seed", 42)
    rng = np.random.default_rng(int(seed))
    num_units = _as_int(cfg, "num_units", 100)
    unit_ids = list(range(1, num_units + 1))
    split_map = assign_unit_splits(unit_ids, cfg.get("split_ratios"), seed=int(cfg.get("split_seed", seed)))

    unit_frames: list[pd.DataFrame] = []
    unit_metadata: list[dict[str, Any]] = []
    for unit_id in unit_ids:
        frame, meta = _generate_unit(unit_id, split_map[unit_id], cfg, rng)
        unit_frames.append(frame)
        unit_metadata.append(meta)
    df = pd.concat(unit_frames, ignore_index=True)
    df = _apply_missing_and_interpolate(
        df,
        observed_columns=OBSERVED_COLUMNS,
        missing_rate=_as_float(cfg, "missing_rate", 0.05),
        burst_rate=_as_float(cfg, "missing_burst_rate", 0.004),
        rng=rng,
    )
    df = df.sort_values(["unit_id", "time_step"]).reset_index(drop=True)

    split_unit_counts = df.groupby("split")["unit_id"].nunique().to_dict()
    split_row_counts = df.groupby("split").size().to_dict()
    metadata = {
        "dataset": "reaction_wheel_sim",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "config": cfg,
        "num_units": int(num_units),
        "num_rows": int(len(df)),
        "default_feature_protocol": "telemetry_only",
        "telemetry_only_columns": TELEMETRY_ONLY_COLUMNS,
        "telemetry_plus_proxy_columns": TELEMETRY_PLUS_PROXY_COLUMNS,
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
        "target_column": "rul",
        "split_unit_counts": {str(k): int(v) for k, v in split_unit_counts.items()},
        "split_row_counts": {str(k): int(v) for k, v in split_row_counts.items()},
        "fault_counts_by_unit": {
            str(k): int(v)
            for k, v in pd.Series([m["explicit_fault_injection_type"] for m in unit_metadata]).value_counts().to_dict().items()
        },
        "eol_reason_counts_by_unit": {
            str(k): int(v) for k, v in pd.Series([m["eol_reason"] for m in unit_metadata]).value_counts().to_dict().items()
        },
        "unit_metadata": unit_metadata,
        "references": [
            "Carrara-style reaction wheel Coulomb/viscous/Stribeck friction and torque-current relation.",
            "ITHACO Type-A/Bialke-style reaction wheel kt0=0.029 Nm/A and 30% nominal kt EOL threshold.",
            "Reaction wheel lubrication decay literature: temperature-driven cumulative lubricant loss and lubrication fault scenarios.",
            "Bearing health literature: cage instability, temperature, vibration, and torque anomaly proxies.",
            "Reaction wheel friction torque literature: speed/temperature/lubricant effects and low-speed/zero-crossing importance.",
        ],
        "engineering_assumptions": [
            "本数据集是机理约束遥测仿真，不是在轨真实遥测，也不是完整的 Simulink/Matlab 复现。",
            "Command voltage is stored on the literature-like 0-5 V scale; the dynamics use the centered command_voltage-2.5 V internally.",
            "Lubricant loss uses Kelvin temperature and a normalized monotone temperature sensitivity for numerical stability.",
            "Friction EOL is evaluated with abs(friction_torque), so reverse rotation failures are not missed.",
            "The viscous temperature term is implemented as a clipped modifier; it is not the literal OCR coefficient that can become negative.",
            "The insufficient-lubrication-injection case is reserved for a later dataset version.",
        ],
    }
    return df, metadata


def validate_reaction_wheel_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"reaction_wheel_sim missing columns: {sorted(missing)}")
    if df[OBSERVED_COLUMNS].isna().any().any():
        nan_cols = [col for col in OBSERVED_COLUMNS if df[col].isna().any()]
        raise ValueError(f"observed columns contain NaN after interpolation: {nan_cols}")
    if not np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("numeric columns contain non-finite values")

    split_overlaps: dict[str, list[int]] = {}
    split_units = {split: set(group["unit_id"].astype(int).tolist()) for split, group in df.groupby("split")}
    for left_name, left_ids in split_units.items():
        for right_name, right_ids in split_units.items():
            if left_name >= right_name:
                continue
            overlap = sorted(left_ids & right_ids)
            if overlap:
                split_overlaps[f"{left_name}/{right_name}"] = overlap
    if split_overlaps:
        raise ValueError(f"unit split leakage detected: {split_overlaps}")

    for unit_id, group in df.groupby("unit_id", sort=False):
        group = group.sort_values("time_step")
        if not group["time_step"].is_monotonic_increasing:
            raise ValueError(f"unit {unit_id} is not sorted by time_step")
        if np.diff(group["kt_ratio"].to_numpy(dtype=np.float64)).max(initial=0.0) > 1e-5:
            raise ValueError(f"unit {unit_id} kt_ratio is not monotone non-increasing")
        if np.diff(group["lubricant_loss"].to_numpy(dtype=np.float64)).min(initial=0.0) < -1e-5:
            raise ValueError(f"unit {unit_id} lubricant_loss is not monotone non-decreasing")
        if np.diff(group["rul"].to_numpy(dtype=np.float64)).max(initial=0.0) > 1e-5:
            raise ValueError(f"unit {unit_id} rul is not monotone non-increasing")
        if not group["normalized_rul"].between(0.0, 1.0).all():
            raise ValueError(f"unit {unit_id} normalized_rul is outside [0, 1]")

    summary = {
        "rows": int(len(df)),
        "units": int(df["unit_id"].nunique()),
        "split_unit_counts": {str(k): int(v) for k, v in df.groupby("split")["unit_id"].nunique().to_dict().items()},
        "split_row_counts": {str(k): int(v) for k, v in df.groupby("split").size().to_dict().items()},
        "fault_counts_by_unit": {
            str(k): int(v)
            for k, v in df.groupby("unit_id")["explicit_fault_injection_type"].first().value_counts().to_dict().items()
        },
        "stage_counts": {str(k): int(v) for k, v in df["health_stage"].value_counts().sort_index().to_dict().items()},
        "eol_reason_counts_by_unit": {
            str(k): int(v) for k, v in df.groupby("unit_id")["eol_reason"].first().value_counts().to_dict().items()
        },
        "missing_mask_counts": {
            col: int(df[col].sum())
            for col in df.columns
            if col.startswith("missing_mask_") and pd.api.types.is_bool_dtype(df[col])
        },
    }
    return summary


def write_reaction_wheel_outputs(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: str | Path,
    csv_name: str = "reaction_wheel_sim.csv",
    npz_name: str = "reaction_wheel_sim.npz",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / csv_name
    metadata_path = out_dir / "metadata.json"
    readme_path = out_dir / "README.md"
    npz_path = out_dir / npz_name

    df.to_csv(csv_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    readme_path.write_text(_dataset_readme(metadata), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        telemetry_only=df[TELEMETRY_ONLY_COLUMNS].to_numpy(dtype=np.float32),
        telemetry_plus_proxy=df[TELEMETRY_PLUS_PROXY_COLUMNS].to_numpy(dtype=np.float32),
        rul=df["rul"].to_numpy(dtype=np.float32),
        normalized_rul=df["normalized_rul"].to_numpy(dtype=np.float32),
        health_stage=df["health_stage"].to_numpy(dtype=np.int64),
        unit_id=df["unit_id"].to_numpy(dtype=np.int64),
        time_step=df["time_step"].to_numpy(dtype=np.int64),
        split=df["split"].astype(str).to_numpy(),
        telemetry_only_columns=np.asarray(TELEMETRY_ONLY_COLUMNS, dtype=object),
        telemetry_plus_proxy_columns=np.asarray(TELEMETRY_PLUS_PROXY_COLUMNS, dtype=object),
    )
    return {"csv": csv_path, "metadata": metadata_path, "readme": readme_path, "npz": npz_path}


def _dataset_readme(metadata: dict[str, Any]) -> str:
    return f"""# reaction_wheel_sim

本目录保存反作用轮寿命预测实验的机理约束仿真遥测数据。
该数据集用于 `XJTU-SY -> reaction_wheel_sim` 跨域迁移实验，不是在轨真实遥测。

生成行数：{metadata.get("num_rows")}
生成单元数：{metadata.get("num_units")}
随机种子：{metadata.get("seed")}

默认输入特征协议：

```text
{", ".join(TELEMETRY_ONLY_COLUMNS)}
```

可选的 `telemetry_plus_proxy` 协议会额外加入 `friction_torque_proxy`。
它是由可观测行为估计得到的含噪健康代理特征。`kt_ratio`、`lubricant_hi`、
`degradation_progress`、`rul`、`normalized_rul` 等隐藏状态、未来信息或标签字段
不得作为普通遥测输入使用。

主要标签：

```text
rul
normalized_rul
health_stage
```

`split` 按完整单元划分，train/val/calib/test 单元互不交叉，用于迁移实验和校准实验。
"""


def load_reaction_wheel_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"reaction_wheel_sim CSV not found: {csv_path}. "
            "Run scripts/generate_reaction_wheel_sim.py first."
        )
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"reaction_wheel_sim CSV missing required columns: {sorted(missing)}")
    return df.sort_values(["unit_id", "time_step"]).reset_index(drop=True)


def infer_reaction_wheel_feature_columns(
    df: pd.DataFrame,
    protocol: str = "telemetry_only",
    explicit: list[str] | None = None,
    allow_leakage: bool = False,
) -> list[str]:
    if explicit:
        missing = [column for column in explicit if column not in df.columns]
        if missing:
            raise ValueError(f"reaction_wheel feature columns not found: {missing}")
        leakage = sorted(set(explicit) & LEAKAGE_COLUMNS)
        if leakage and not allow_leakage:
            raise ValueError(
                "reaction_wheel feature columns contain hidden/label leakage columns: "
                f"{leakage}. Set allow_leakage_features only for explicit oracle ablations."
            )
        return list(explicit)
    protocol_key = str(protocol).lower()
    if protocol_key in {"telemetry_only", "telemetry-only"}:
        return TELEMETRY_ONLY_COLUMNS.copy()
    if protocol_key in {"telemetry_plus_proxy", "telemetry+proxy", "with_proxy", "telemetry_with_proxy"}:
        return TELEMETRY_PLUS_PROXY_COLUMNS.copy()
    raise ValueError(f"unknown reaction_wheel feature protocol: {protocol}")


def reaction_wheel_dataframe_to_unit_dicts(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "rul",
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    if target_column not in df.columns:
        raise ValueError(f"reaction_wheel target column not found: {target_column}")
    series_by_unit: dict[int, np.ndarray] = {}
    rul_by_unit: dict[int, np.ndarray] = {}
    for unit_id, group in df.sort_values(["unit_id", "time_step"]).groupby("unit_id"):
        series_by_unit[int(unit_id)] = group[feature_columns].to_numpy(dtype=np.float32)
        rul_by_unit[int(unit_id)] = group[target_column].to_numpy(dtype=np.float32)
    return series_by_unit, rul_by_unit


def _ids_for_split(df: pd.DataFrame, split: str) -> np.ndarray:
    ids = df.loc[df["split"] == split, "unit_id"].drop_duplicates().to_numpy(dtype=np.int64)
    return ids


def build_reaction_wheel_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    data_cfg = config["data"]
    seed = int(config["experiment"].get("seed", data_cfg.get("seed", 42)))
    csv_path = Path(data_cfg.get("csv_path", "data/simulated/reaction_wheel/reaction_wheel_sim.csv"))
    df = load_reaction_wheel_csv(csv_path)
    feature_columns = infer_reaction_wheel_feature_columns(
        df,
        protocol=str(data_cfg.get("feature_protocol", "telemetry_only")),
        explicit=data_cfg.get("feature_columns"),
        allow_leakage=bool(data_cfg.get("allow_leakage_features", False)),
    )
    target_column = str(data_cfg.get("target_column", "rul"))
    series_all, rul_all = reaction_wheel_dataframe_to_unit_dicts(df, feature_columns, target_column=target_column)

    if "split" in df.columns and {"train", "val", "test"}.issubset(set(df["split"].astype(str).unique())):
        train_ids = _ids_for_split(df, "train")
        val_ids = _ids_for_split(df, "val")
        test_ids = _ids_for_split(df, "test")
    else:
        all_ids = np.asarray(list(series_all.keys()), dtype=np.int64)
        train_ids, val_ids, test_ids = split_unit_ids(
            all_ids,
            train_ratio=float(data_cfg.get("train_ratio", 0.7)),
            val_ratio=float(data_cfg.get("val_ratio", 0.15)),
            seed=int(data_cfg.get("split_seed", seed)),
        )

    train_series = _filter_units(series_all, train_ids)
    val_series = _filter_units(series_all, val_ids)
    test_series = _filter_units(series_all, test_ids)
    train_rul = _filter_units(rul_all, train_ids)
    val_rul = _filter_units(rul_all, val_ids)
    test_rul = _filter_units(rul_all, test_ids)
    if not train_series or not val_series or not test_series:
        raise ValueError("reaction_wheel train/val/test split produced an empty split")

    normalizer = fit_normalizer(list(train_series.values()), method=str(data_cfg.get("normalize", "zscore")))
    max_rul_value = _optional_float(data_cfg.get("max_rul", None))
    common_kwargs = {
        "window_size": int(data_cfg["window_size"]),
        "stride": int(data_cfg.get("stride", 1)),
        "max_rul": max_rul_value,
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
