from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from xa202608.data.battery import load_processed_battery_csv
from xa202608.data.windowing import WindowedTimeSeriesDataset, fit_normalizer, split_unit_ids

TELEMETRY_ONLY_COLUMNS = ["voltage", "current", "temperature"]
BATTERY_PROXY_V1_COLUMNS = [
    "d_voltage",
    "d_current",
    "d_temperature",
    "abs_current",
    "transition_resistance_proxy",
    "thermal_response_proxy",
]
TARGET_RICH_COLUMNS = TELEMETRY_ONLY_COLUMNS + [
    "soc_est",
    "c_rate_nominal",
    "orbit_phase_sin",
    "orbit_phase_cos",
    "sunlight",
]

LEAKAGE_COLUMNS = {
    "soc_true",
    "capacity_ah",
    "capacity_ratio",
    "internal_resistance_ohm",
    "resistance_ratio",
    "soh_capacity",
    "soh_resistance",
    "soh",
    "capacity_damage",
    "resistance_damage",
    "fused_damage",
    "rul_cycles",
    "normalized_rul",
    "health_stage",
    "fault_type",
    "fault_active",
    "eol_cycle",
    "eol_reason",
    "is_censored",
    "split",
    "cycle",
    "time_step",
}

REQUIRED_COLUMNS = {
    "unit_id",
    "split",
    "cycle",
    "time_step",
    "elapsed_days",
    "orbit_phase",
    "orbit_phase_sin",
    "orbit_phase_cos",
    "sunlight",
    "current",
    "voltage",
    "temperature",
    "soc_true",
    "soc_est",
    "c_rate_nominal",
    "dod_cycle",
    "ocv",
    "polarization_voltage",
    "capacity_ah",
    "capacity_ratio",
    "internal_resistance_ohm",
    "resistance_ratio",
    "soh_capacity",
    "soh_resistance",
    "soh",
    "capacity_damage",
    "resistance_damage",
    "fused_damage",
    "rul_cycles",
    "normalized_rul",
    "health_stage",
    "fault_type",
    "fault_active",
    "eol_cycle",
    "eol_reason",
    "is_censored",
}


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


def _filter_units(source: dict[int, np.ndarray], ids: np.ndarray | set[int]) -> dict[int, np.ndarray]:
    if isinstance(ids, set):
        id_set = {int(x) for x in ids}
    else:
        id_set = {int(x) for x in ids.tolist()}
    return {unit_id: value for unit_id, value in source.items() if int(unit_id) in id_set}


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


def _uses_battery_proxy_features(data_cfg: dict[str, Any], *feature_groups: object) -> bool:
    proxy_key = str(data_cfg.get("battery_proxy_feature_set", "")).lower()
    if proxy_key in {"pg_stda_v1", "physics_proxy_v1", "battery_proxy_v1"}:
        return True
    protocol_keys = [
        str(data_cfg.get("feature_protocol", "")).lower(),
        str(data_cfg.get("target_feature_protocol", "")).lower(),
    ]
    if any(key in {"pg_stda_v1", "telemetry_proxy", "telemetry_with_proxy"} for key in protocol_keys):
        return True
    proxy_columns = set(BATTERY_PROXY_V1_COLUMNS)
    for group in feature_groups:
        if isinstance(group, (list, tuple, set)) and proxy_columns.intersection(str(column) for column in group):
            return True
    return False


def _clip_finite(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    out = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=float(upper), neginf=float(lower))
    return np.clip(out, float(lower), float(upper)).astype(np.float32)


def add_battery_physics_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add bounded V/I/T-derived proxies without using hidden health labels."""
    required = {"unit_id", "voltage", "current", "temperature"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"battery physics proxy features require columns: {missing}")
    out = df.copy()
    sort_columns = ["unit_id"]
    for column in ("time_step", "cycle", "global_cycle"):
        if column in out.columns:
            sort_columns.append(column)
    ordered = out.sort_values(sort_columns, kind="mergesort").copy()
    groups = ordered.groupby("unit_id", sort=False)

    d_voltage = groups["voltage"].diff().fillna(0.0).to_numpy(dtype=np.float64)
    d_current = groups["current"].diff().fillna(0.0).to_numpy(dtype=np.float64)
    d_temperature = groups["temperature"].diff().fillna(0.0).to_numpy(dtype=np.float64)
    abs_current = np.abs(ordered["current"].to_numpy(dtype=np.float64))

    transition_resistance = np.zeros_like(d_voltage, dtype=np.float64)
    transition_mask = np.abs(d_current) >= 0.02
    transition_resistance[transition_mask] = -d_voltage[transition_mask] / d_current[transition_mask]
    thermal_response = d_temperature / (abs_current + 0.05)

    ordered["d_voltage"] = _clip_finite(d_voltage, -1.0, 1.0)
    ordered["d_current"] = _clip_finite(d_current, -3.0, 3.0)
    ordered["d_temperature"] = _clip_finite(d_temperature, -10.0, 10.0)
    ordered["abs_current"] = _clip_finite(abs_current, 0.0, 10.0)
    ordered["transition_resistance_proxy"] = _clip_finite(transition_resistance, -5.0, 5.0)
    ordered["thermal_response_proxy"] = _clip_finite(thermal_response, -10.0, 10.0)

    out.loc[ordered.index, BATTERY_PROXY_V1_COLUMNS] = ordered[BATTERY_PROXY_V1_COLUMNS].to_numpy(dtype=np.float32)
    return out


def assign_unit_splits(
    unit_ids: list[int] | np.ndarray,
    split_ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[int, str]:
    ratios = split_ratios or {"train": 0.7, "val": 0.15, "test": 0.15}
    names = ["train", "val", "test"]
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


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def _ocv(soc_true: float, temperature_c: float, temp_coeff: float) -> float:
    soc = float(np.clip(soc_true, 0.0, 1.0))
    ocv_base = 3.0 + 1.20 * soc - 0.08 * np.sin(np.pi * soc)
    return float(np.clip(ocv_base + float(temp_coeff) * (float(temperature_c) - 25.0), 2.8, 4.20))


def _quantize(value: float, step: float) -> float:
    step = float(step)
    if step <= 0:
        return float(value)
    return float(np.round(float(value) / step) * step)


def _fault_type(explicit_fault_fraction: float, rng: np.random.Generator) -> str:
    if rng.random() >= float(explicit_fault_fraction):
        return "none"
    return str(rng.choice(["capacity_knee", "resistance_step", "thermal_stress"], p=[0.45, 0.45, 0.10]))


def _health_stage(fused_damage: float) -> int:
    return int(np.digitize(float(np.clip(fused_damage, 0.0, 1.0)), [0.3, 0.7], right=False))


def _generate_unit(
    unit_id: int,
    split: str,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    min_eol = _as_int(cfg, "min_eol_cycles", 300)
    max_eol = _as_int(cfg, "max_eol_cycles", 1200)
    steps_per_cycle = _as_int(cfg, "steps_per_cycle", 24)
    if min_eol <= 0 or max_eol < min_eol:
        raise ValueError("min_eol_cycles and max_eol_cycles must define a positive range")
    if steps_per_cycle < 4:
        raise ValueError("steps_per_cycle must be at least 4")

    orbit_period_min = _as_float(cfg, "orbit_period_min", 90.0)
    sunlight_minutes = _as_float(cfg, "sunlight_minutes", 55.0)
    sunlight_fraction = float(np.clip(sunlight_minutes / max(orbit_period_min, 1e-6), 0.2, 0.85))
    dt_seconds = orbit_period_min * 60.0 / steps_per_cycle
    dt_hours = dt_seconds / 3600.0
    dt_days = dt_seconds / 86400.0

    q_bol = float(np.clip(rng.normal(_as_float(cfg, "q_bol_mean", 2.0), _as_float(cfg, "q_bol_std", 0.04)), 1.85, 2.15))
    r_bol = float(
        np.clip(
            rng.normal(_as_float(cfg, "r_bol_mean", 0.045), _as_float(cfg, "r_bol_std", 0.004)),
            _as_float(cfg, "r_bol_min", 0.035),
            _as_float(cfg, "r_bol_max", 0.060),
        )
    )
    target_life = int(rng.integers(min_eol, max_eol + 1))
    explicit_fault_fraction = _as_float(cfg, "explicit_fault_fraction", 0.25)
    fault_type = _fault_type(explicit_fault_fraction, rng)
    fault_cycle = int(rng.integers(max(3, int(0.45 * target_life)), max(4, int(0.82 * target_life)) + 1))
    knee_cycle = int(rng.integers(max(3, int(0.55 * target_life)), max(4, int(0.82 * target_life)) + 1))
    knee_width = max(4.0, 0.05 * target_life)
    knee_strength = float(rng.uniform(1.2, 2.8)) if fault_type == "capacity_knee" else float(rng.uniform(0.0, 0.35))
    resistance_step_size = float(rng.uniform(0.10, 0.22)) if fault_type == "resistance_step" else 0.0
    resistance_step_applied = False

    nominal_dod = float(rng.uniform(_as_float(cfg, "dod_min", 0.20), _as_float(cfg, "dod_max", 0.40)))
    if rng.random() < _as_float(cfg, "stress_dod_fraction", 0.08):
        nominal_dod = float(rng.uniform(0.40, _as_float(cfg, "stress_dod_max", 0.50)))
    soc_max = float(rng.uniform(0.90, 0.96))
    soc_min = float(max(0.45, soc_max - nominal_dod))
    soc_true = float(rng.uniform(soc_min + 0.05, soc_max))
    soc_est = float(np.clip(soc_true + rng.normal(0.0, _as_float(cfg, "soc_est_initial_std", 0.025)), 0.0, 1.0))

    base_temp = float(rng.normal(_as_float(cfg, "base_temperature_mean", 24.0), _as_float(cfg, "base_temperature_std", 3.0)))
    thermal_stress = fault_type == "thermal_stress"
    if thermal_stress:
        base_temp += float(rng.uniform(5.0, 9.0))
    orbit_temp_amp = float(rng.uniform(_as_float(cfg, "orbit_temp_amp_min", 4.0), _as_float(cfg, "orbit_temp_amp_max", 9.0)))
    if thermal_stress:
        orbit_temp_amp += float(rng.uniform(1.5, 3.5))
    temp_phase = float(rng.uniform(-np.pi, np.pi))
    temp_coeff = float(rng.uniform(-0.0010, -0.0005))
    charge_c_rate = float(rng.uniform(0.22, 0.42))
    discharge_c_rate = float(rng.uniform(0.24, 0.48))
    pulse_c_rate = float(rng.uniform(0.20, 0.55))
    pulse_phase = int(rng.integers(max(1, int(steps_per_cycle * sunlight_fraction)), steps_per_cycle))
    pulse_width = max(1, int(round(steps_per_cycle * rng.uniform(0.04, 0.09))))
    rp = float(rng.uniform(_as_float(cfg, "polarization_resistance_min", 0.010), _as_float(cfg, "polarization_resistance_max", 0.028)))
    tau_seconds = float(rng.uniform(_as_float(cfg, "thevenin_tau_min_s", 420.0), _as_float(cfg, "thevenin_tau_max_s", 780.0)))
    adc_step_v = _as_float(cfg, "adc_step_v", 0.001)
    voltage_noise_std = _as_float(cfg, "voltage_noise_std", 0.006)
    current_noise_std = _as_float(cfg, "current_noise_std", 0.006)
    temperature_noise_std = _as_float(cfg, "temperature_noise_std", 0.12)
    if fault_type == "capacity_knee":
        dominant_failure_mode = "capacity"
    elif fault_type == "resistance_step":
        dominant_failure_mode = "resistance"
    else:
        dominant_failure_mode = str(rng.choice(["capacity", "resistance", "mixed"], p=[0.35, 0.35, 0.30]))
    if dominant_failure_mode == "capacity":
        cap_base_rate = float(rng.uniform(1.00, 1.18)) / float(target_life)
        res_base_rate = float(rng.uniform(0.58, 0.82)) / float(target_life)
    elif dominant_failure_mode == "resistance":
        cap_base_rate = float(rng.uniform(0.70, 0.95)) / float(target_life)
        res_base_rate = float(rng.uniform(0.98, 1.18)) / float(target_life)
    else:
        cap_base_rate = float(rng.uniform(0.86, 1.08)) / float(target_life)
        res_base_rate = float(rng.uniform(0.78, 1.02)) / float(target_life)
    thermal_sensitivity = float(rng.uniform(0.026, 0.044))
    crate_sensitivity = float(rng.uniform(0.35, 0.65))
    dod_sensitivity = float(rng.uniform(0.55, 0.90))

    capacity_damage = 0.0
    resistance_damage = 0.0
    polarization_voltage = 0.0
    rows: list[dict[str, Any]] = []
    cycle_summaries: list[dict[str, float]] = []
    max_cycles = int(max(max_eol * 2, target_life * 2, target_life + 60))
    eol_reason = "max_cycle_reached"
    is_censored = True
    cycle = 1

    while cycle <= max_cycles:
        capacity_ah = q_bol * (1.0 - 0.2 * float(np.clip(capacity_damage, 0.0, 1.25)))
        internal_resistance = r_bol * (1.0 + 0.33 * float(np.clip(resistance_damage, 0.0, 1.30)))
        capacity_ratio = capacity_ah / q_bol
        resistance_ratio = internal_resistance / r_bol
        capacity_eol_hit = capacity_ratio <= 0.8 + 1e-9
        resistance_eol_hit = resistance_ratio >= 1.33 - 1e-9
        eol_reached_at_start = capacity_eol_hit or resistance_eol_hit
        if capacity_eol_hit and resistance_eol_hit:
            current_eol_reason = "capacity_and_resistance_threshold"
        elif capacity_eol_hit:
            current_eol_reason = "capacity_threshold"
        elif resistance_eol_hit:
            current_eol_reason = "resistance_threshold"
        else:
            current_eol_reason = "not_eol"

        cycle_rows_start = len(rows)
        soc_values = []
        current_values = []
        temp_values = []
        abs_current_values = []
        max_soc_seen = soc_true
        min_soc_seen = soc_true
        for step in range(steps_per_cycle):
            orbit_phase = step / float(steps_per_cycle)
            sunlight = int(orbit_phase < sunlight_fraction)
            if sunlight:
                taper = 1.0
                if soc_true > 0.88:
                    taper = max(0.12, 1.0 - 0.75 * (soc_true - 0.88) / max(soc_max - 0.88, 0.02))
                current_true = -charge_c_rate * q_bol * taper
            else:
                in_pulse = pulse_phase <= step < pulse_phase + pulse_width
                pulse = pulse_c_rate * q_bol if in_pulse else 0.0
                current_true = discharge_c_rate * q_bol + pulse
                if soc_true < soc_min:
                    current_true *= 0.35
            seasonal = 1.5 * np.sin(2.0 * np.pi * cycle / 180.0 + 0.01 * unit_id)
            temperature_true = (
                base_temp
                + orbit_temp_amp * np.sin(2.0 * np.pi * orbit_phase + temp_phase)
                + 1.8 * abs(current_true) / max(q_bol, 1e-6)
                + seasonal
            )
            temperature = float(np.clip(temperature_true + rng.normal(0.0, temperature_noise_std), -10.0, 50.0))
            current = float(current_true + rng.normal(0.0, current_noise_std))
            ocv = _ocv(soc_true, temperature, temp_coeff=temp_coeff)
            decay = float(np.exp(-dt_seconds / max(tau_seconds, 1e-6)))
            polarization_voltage = decay * polarization_voltage + rp * (1.0 - decay) * current_true
            voltage_true = ocv - current_true * internal_resistance - polarization_voltage
            voltage = _quantize(float(np.clip(voltage_true + rng.normal(0.0, voltage_noise_std), 2.5, 4.25)), adc_step_v)
            voltage = float(np.clip(voltage, 2.5, 4.25))
            c_rate_nominal = abs(current_true) / max(q_bol, 1e-6)
            fault_active = int(fault_type != "none" and cycle >= fault_cycle)
            rows.append(
                {
                    "unit_id": int(unit_id),
                    "split": split,
                    "cycle": int(cycle),
                    "time_step": int((cycle - 1) * steps_per_cycle + step),
                    "cycle_step": int(step),
                    "elapsed_days": float(((cycle - 1) * steps_per_cycle + step) * dt_days),
                    "orbit_phase": float(orbit_phase),
                    "orbit_phase_sin": float(np.sin(2.0 * np.pi * orbit_phase)),
                    "orbit_phase_cos": float(np.cos(2.0 * np.pi * orbit_phase)),
                    "sunlight": int(sunlight),
                    "current": current,
                    "voltage": voltage,
                    "temperature": temperature,
                    "soc_true": float(soc_true),
                    "soc_est": float(soc_est),
                    "c_rate_nominal": float(c_rate_nominal),
                    "dod_cycle": 0.0,
                    "ocv": float(ocv),
                    "polarization_voltage": float(polarization_voltage),
                    "capacity_ah": float(capacity_ah),
                    "capacity_ratio": float(capacity_ratio),
                    "internal_resistance_ohm": float(internal_resistance),
                    "resistance_ratio": float(resistance_ratio),
                    "soh_capacity": float(capacity_ratio),
                    "soh_resistance": float(np.clip((1.33 - resistance_ratio) / 0.33, 0.0, 1.0)),
                    "capacity_damage": float(np.clip(capacity_damage, 0.0, 1.0)),
                    "resistance_damage": float(np.clip(resistance_damage, 0.0, 1.0)),
                    "fault_type": fault_type,
                    "fault_active": int(fault_active),
                    "eol_reason": current_eol_reason,
                    "is_censored": False,
                }
            )
            soc_values.append(soc_true)
            current_values.append(current_true)
            abs_current_values.append(abs(current_true))
            temp_values.append(temperature)
            max_soc_seen = max(max_soc_seen, soc_true)
            min_soc_seen = min(min_soc_seen, soc_true)
            soc_true = float(np.clip(soc_true - current_true * dt_hours / max(capacity_ah, 1e-6), soc_min - 0.03, soc_max + 0.01))
            soc_true = float(np.clip(soc_true, 0.0, 1.0))
            soc_est = float(
                np.clip(
                    soc_est - current * dt_hours / max(q_bol, 1e-6) + rng.normal(0.0, _as_float(cfg, "soc_est_drift_std", 0.00015)),
                    0.0,
                    1.0,
                )
            )

        dod_cycle = float(np.clip(max_soc_seen - min_soc_seen, 0.0, 1.0))
        for idx in range(cycle_rows_start, len(rows)):
            rows[idx]["dod_cycle"] = dod_cycle
        cycle_summaries.append(
            {
                "cycle": float(cycle),
                "avg_temperature": float(np.mean(temp_values)),
                "avg_c_rate": float(np.mean(abs_current_values) / max(q_bol, 1e-6)),
                "avg_soc": float(np.mean(soc_values)),
                "dod_cycle": float(dod_cycle),
            }
        )
        if eol_reached_at_start:
            eol_reason = current_eol_reason
            is_censored = False
            break

        avg_temp = cycle_summaries[-1]["avg_temperature"]
        avg_c_rate = cycle_summaries[-1]["avg_c_rate"]
        avg_soc = cycle_summaries[-1]["avg_soc"]
        temp_factor = float(np.clip(np.exp(thermal_sensitivity * (avg_temp - 25.0)), 0.45, 2.70))
        c_rate_factor = float(np.clip((avg_c_rate / 0.35) ** crate_sensitivity, 0.55, 2.25))
        dod_factor = float(np.clip((max(dod_cycle, 0.05) / 0.30) ** dod_sensitivity, 0.55, 2.20))
        soc_factor = float(np.clip(1.0 + 1.4 * (avg_soc - 0.72) ** 2, 0.85, 1.40))
        stress = float(np.clip(temp_factor * c_rate_factor * dod_factor * soc_factor, 0.55, 2.60))
        if fault_type == "thermal_stress" and cycle >= fault_cycle:
            stress *= 1.25
        knee_multiplier = 1.0 + knee_strength * _sigmoid((cycle - knee_cycle) / knee_width)
        cap_increment = cap_base_rate * stress * knee_multiplier
        res_increment = res_base_rate * (0.70 * stress + 0.30 * temp_factor) + 0.10 * cap_increment
        if fault_type == "resistance_step" and cycle >= fault_cycle and not resistance_step_applied:
            res_increment += resistance_step_size
            resistance_step_applied = True
        capacity_damage = float(min(1.30, capacity_damage + cap_increment))
        resistance_damage = float(min(1.35, resistance_damage + res_increment))
        cycle += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("satellite battery unit generation produced no rows")
    eol_candidates = df[(df["capacity_ratio"] <= 0.8 + 1e-9) | (df["resistance_ratio"] >= 1.33 - 1e-9)]
    if len(eol_candidates):
        eol_cycle = int(eol_candidates["cycle"].iloc[0])
        eol_reason = str(eol_candidates["eol_reason"].iloc[0])
        is_censored = False
    else:
        eol_cycle = int(df["cycle"].max())
        eol_reason = "max_cycle_reached"
        is_censored = True
    df["eol_cycle"] = int(eol_cycle)
    df["eol_reason"] = eol_reason
    df["is_censored"] = bool(is_censored)
    df["rul_cycles"] = np.maximum(float(eol_cycle) - df["cycle"].to_numpy(dtype=np.float32), 0.0)
    df["normalized_rul"] = df["rul_cycles"] / max(float(eol_cycle), 1.0)
    df["soh_capacity_eol_scaled"] = np.clip((df["capacity_ratio"] - 0.8) / 0.2, 0.0, 1.0)
    df["soh_resistance"] = np.clip((1.33 - df["resistance_ratio"]) / 0.33, 0.0, 1.0)
    df["soh"] = 0.5 * df["soh_capacity_eol_scaled"] + 0.5 * df["soh_resistance"]
    df["fused_damage"] = 0.5 * df["capacity_damage"].clip(0.0, 1.0) + 0.5 * df["resistance_damage"].clip(0.0, 1.0)
    df["health_stage"] = df["fused_damage"].map(_health_stage).astype(int)
    df.loc[df["cycle"] < int(fault_cycle), "fault_active"] = 0
    df.loc[df["fault_type"] == "none", "fault_active"] = 0
    unit_metadata = {
        "unit_id": int(unit_id),
        "split": split,
        "rows": int(len(df)),
        "eol_cycle": int(eol_cycle),
        "eol_reason": eol_reason,
        "is_censored": bool(is_censored),
        "fault_type": fault_type,
        "fault_cycle": int(fault_cycle),
        "target_life_cycles": int(target_life),
        "q_bol": float(q_bol),
        "r_bol": float(r_bol),
        "nominal_dod": float(nominal_dod),
        "soc_min": float(soc_min),
        "soc_max": float(soc_max),
        "base_temperature": float(base_temp),
        "orbit_temp_amp": float(orbit_temp_amp),
        "knee_cycle": int(knee_cycle),
        "knee_strength": float(knee_strength),
        "resistance_step_size": float(resistance_step_size),
        "dominant_failure_mode": dominant_failure_mode,
        "thevenin_tau_seconds": float(tau_seconds),
        "polarization_resistance": float(rp),
    }
    return df, unit_metadata


def generate_satellite_battery_dataset(
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

    frames: list[pd.DataFrame] = []
    unit_metadata: list[dict[str, Any]] = []
    for unit_id in unit_ids:
        frame, meta = _generate_unit(unit_id, split_map[unit_id], cfg, rng)
        frames.append(frame)
        unit_metadata.append(meta)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["unit_id", "time_step"]).reset_index(drop=True)
    split_unit_counts = df.groupby("split")["unit_id"].nunique().to_dict()
    split_row_counts = df.groupby("split").size().to_dict()
    metadata = {
        "dataset": "satellite_battery_sim",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "config": cfg,
        "num_units": int(num_units),
        "num_rows": int(len(df)),
        "default_feature_protocol": "telemetry_only",
        "telemetry_only_columns": TELEMETRY_ONLY_COLUMNS,
        "target_rich_columns": TARGET_RICH_COLUMNS,
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
        "target_column": "normalized_rul",
        "eol_definition": {
            "capacity_threshold": "capacity_ah <= 0.8 * Q_BOL",
            "resistance_threshold": "internal_resistance_ohm >= 1.33 * R_BOL",
        },
        "orbit_profile": {
            "orbit_period_min": _as_float(cfg, "orbit_period_min", 90.0),
            "sunlight_minutes": _as_float(cfg, "sunlight_minutes", 55.0),
            "eclipse_minutes": _as_float(cfg, "orbit_period_min", 90.0) - _as_float(cfg, "sunlight_minutes", 55.0),
            "steps_per_cycle": _as_int(cfg, "steps_per_cycle", 24),
        },
        "split_unit_counts": {str(k): int(v) for k, v in split_unit_counts.items()},
        "split_row_counts": {str(k): int(v) for k, v in split_row_counts.items()},
        "fault_counts_by_unit": {
            str(k): int(v) for k, v in pd.Series([m["fault_type"] for m in unit_metadata]).value_counts().to_dict().items()
        },
        "eol_reason_counts_by_unit": {
            str(k): int(v) for k, v in pd.Series([m["eol_reason"] for m in unit_metadata]).value_counts().to_dict().items()
        },
        "unit_metadata": unit_metadata,
        "references": [
            "NASA Battery Aging source domain: 18650 Li-ion cells with voltage/current/temperature/capacity/impedance telemetry.",
            "Guha and Patra-style capacity and internal-resistance fusion: Q_EOL=0.8*Q_BOL and R_intEOL=1.33*R_intBOL.",
            "Semi-empirical battery aging literature: temperature, C-rate, DOD, SOC, cycle aging, and calendar aging stress factors.",
            "Thevenin equivalent-circuit model with OCV(SOC,T), ohmic drop, and polarization voltage.",
            "LEO-like battery operation: 90 min orbit, sunlight charge, eclipse discharge, and 20%-40% DOD main regime.",
        ],
        "engineering_assumptions": [
            "本数据集是机理约束加速仿真，不是在轨真实遥测，也不是高保真电化学模型。",
            "The stored current convention is discharge-positive; NASA source current can be multiplied by -1 in transfer loaders.",
            "soc_true is an internal simulator state and is forbidden as a default model input.",
            "soc_est is a noisy coulomb-counted estimate and is reserved for target-rich ablations.",
            "The first dataset version generates all target units to explicit EOL and does not use right-censored RUL labels.",
            "The Thevenin time constant is a macro-equivalent setting compatible with 24/32 output samples per orbit.",
        ],
    }
    return df, metadata


def validate_satellite_battery_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"satellite_battery_sim missing columns: {sorted(missing)}")
    if df[TELEMETRY_ONLY_COLUMNS].isna().any().any():
        nan_cols = [column for column in TELEMETRY_ONLY_COLUMNS if df[column].isna().any()]
        raise ValueError(f"telemetry columns contain NaN: {nan_cols}")
    if not np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("numeric columns contain non-finite values")

    split_units = {split: set(group["unit_id"].astype(int).tolist()) for split, group in df.groupby("split")}
    split_overlaps: dict[str, list[int]] = {}
    split_names = sorted(split_units)
    for idx, left_name in enumerate(split_names):
        for right_name in split_names[idx + 1 :]:
            overlap = sorted(split_units[left_name] & split_units[right_name])
            if overlap:
                split_overlaps[f"{left_name}/{right_name}"] = overlap
    if split_overlaps:
        raise ValueError(f"unit split leakage detected: {split_overlaps}")

    if not df["soc_true"].between(0.0, 1.0).all():
        raise ValueError("soc_true outside [0, 1]")
    if not df["soc_est"].between(0.0, 1.0).all():
        raise ValueError("soc_est outside [0, 1]")
    if not df["voltage"].between(2.5, 4.25).all():
        raise ValueError("voltage outside [2.5, 4.25]")
    if not df["temperature"].between(-10.0, 50.0).all():
        raise ValueError("temperature outside [-10, 50] degC")
    if not df["normalized_rul"].between(0.0, 1.0).all():
        raise ValueError("normalized_rul outside [0, 1]")
    if (df["capacity_ah"] <= 0).any() or (df["internal_resistance_ohm"] <= 0).any():
        raise ValueError("capacity/internal resistance must be positive")

    for unit_id, group in df.groupby("unit_id", sort=False):
        group = group.sort_values(["cycle", "time_step"])
        if not group["time_step"].is_monotonic_increasing:
            raise ValueError(f"unit {unit_id} is not sorted by time_step")
        cycle_level = group.drop_duplicates("cycle")
        if np.diff(cycle_level["capacity_ah"].to_numpy(dtype=np.float64)).max(initial=0.0) > 1e-6:
            raise ValueError(f"unit {unit_id} capacity_ah is not monotone non-increasing")
        if np.diff(cycle_level["internal_resistance_ohm"].to_numpy(dtype=np.float64)).min(initial=0.0) < -1e-6:
            raise ValueError(f"unit {unit_id} internal_resistance_ohm is not monotone non-decreasing")
        if np.diff(cycle_level["rul_cycles"].to_numpy(dtype=np.float64)).max(initial=0.0) > 1e-6:
            raise ValueError(f"unit {unit_id} rul_cycles is not monotone non-increasing")
        eol_cycle = int(cycle_level["eol_cycle"].iloc[0])
        eol_rows = cycle_level.loc[cycle_level["cycle"] == eol_cycle]
        if len(eol_rows) and not bool(cycle_level["is_censored"].iloc[0]):
            row = eol_rows.iloc[0]
            if not (row["capacity_ratio"] <= 0.8 + 1e-6 or row["resistance_ratio"] >= 1.33 - 1e-6):
                raise ValueError(f"unit {unit_id} eol_cycle does not meet EOL threshold")

    return {
        "rows": int(len(df)),
        "units": int(df["unit_id"].nunique()),
        "split_unit_counts": {str(k): int(v) for k, v in df.groupby("split")["unit_id"].nunique().to_dict().items()},
        "split_row_counts": {str(k): int(v) for k, v in df.groupby("split").size().to_dict().items()},
        "fault_counts_by_unit": {
            str(k): int(v) for k, v in df.groupby("unit_id")["fault_type"].first().value_counts().to_dict().items()
        },
        "eol_reason_counts_by_unit": {
            str(k): int(v) for k, v in df.groupby("unit_id")["eol_reason"].first().value_counts().to_dict().items()
        },
        "stage_counts": {str(k): int(v) for k, v in df["health_stage"].value_counts().sort_index().to_dict().items()},
        "voltage_range": [float(df["voltage"].min()), float(df["voltage"].max())],
        "temperature_range": [float(df["temperature"].min()), float(df["temperature"].max())],
        "soc_true_range": [float(df["soc_true"].min()), float(df["soc_true"].max())],
        "dod_cycle_range": [float(df["dod_cycle"].min()), float(df["dod_cycle"].max())],
    }


def _dataset_readme(metadata: dict[str, Any]) -> str:
    columns = ", ".join(metadata.get("telemetry_only_columns", TELEMETRY_ONLY_COLUMNS))
    return f"""# satellite_battery_sim

本目录保存卫星电池寿命预测实验的机理约束加速仿真数据。
该数据集用于 `NASA Battery -> satellite_battery_sim` 跨域迁移实验，不是在轨真实遥测。

生成行数：{metadata.get("num_rows")}
生成单元数：{metadata.get("num_units")}
随机种子：{metadata.get("seed")}

默认输入特征协议：

```text
{columns}
```

`soc_true`、容量、内阻、SOH、fused damage、EOL、RUL、split、故障标签、
cycle 和 time-step 等字段属于隐藏状态、标签或溯源字段，不得作为默认遥测输入使用。

主要标签：

```text
normalized_rul
rul_cycles
soh
health_stage
```

目标域 EOL 定义为首次满足以下任一条件的循环：
容量下降到 BOL 容量的 80%，或内阻达到 BOL 内阻的 1.33 倍。
"""


def write_satellite_battery_outputs(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: str | Path,
    csv_name: str = "satellite_battery_sim.csv",
    npz_name: str = "arrays.npz",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / csv_name
    units_path = out_dir / "units.csv"
    metadata_path = out_dir / "metadata.json"
    readme_path = out_dir / "README.md"
    npz_path = out_dir / npz_name

    df.to_csv(csv_path, index=False)
    unit_columns = [
        "unit_id",
        "split",
        "eol_cycle",
        "eol_reason",
        "fault_type",
        "fault_active",
        "is_censored",
    ]
    unit_summary = df.groupby("unit_id", sort=True).agg(
        split=("split", "first"),
        eol_cycle=("eol_cycle", "first"),
        eol_reason=("eol_reason", "first"),
        fault_type=("fault_type", "first"),
        fault_active=("fault_active", "max"),
        is_censored=("is_censored", "first"),
        q_bol=("capacity_ah", "first"),
        capacity_last=("capacity_ah", "last"),
        resistance_bol=("internal_resistance_ohm", "first"),
        resistance_last=("internal_resistance_ohm", "last"),
    ).reset_index()
    unit_summary[unit_columns + [col for col in unit_summary.columns if col not in unit_columns]].to_csv(units_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    readme_path.write_text(_dataset_readme(metadata), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        telemetry_only=df[TELEMETRY_ONLY_COLUMNS].to_numpy(dtype=np.float32),
        target_rich=df[TARGET_RICH_COLUMNS].to_numpy(dtype=np.float32),
        normalized_rul=df["normalized_rul"].to_numpy(dtype=np.float32),
        rul_cycles=df["rul_cycles"].to_numpy(dtype=np.float32),
        soh=df["soh"].to_numpy(dtype=np.float32),
        health_stage=df["health_stage"].to_numpy(dtype=np.int64),
        unit_id=df["unit_id"].to_numpy(dtype=np.int64),
        cycle=df["cycle"].to_numpy(dtype=np.int64),
        time_step=df["time_step"].to_numpy(dtype=np.int64),
        split=df["split"].astype(str).to_numpy(),
        telemetry_only_columns=np.asarray(TELEMETRY_ONLY_COLUMNS, dtype=object),
        target_rich_columns=np.asarray(TARGET_RICH_COLUMNS, dtype=object),
    )
    return {"csv": csv_path, "units": units_path, "metadata": metadata_path, "readme": readme_path, "npz": npz_path}


def load_satellite_battery_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"satellite_battery_sim CSV not found: {csv_path}. Run scripts/generate_satellite_battery_sim.py first."
        )
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"satellite_battery_sim CSV missing required columns: {sorted(missing)}")
    return df.sort_values(["unit_id", "time_step"]).reset_index(drop=True)


def infer_satellite_battery_feature_columns(
    df: pd.DataFrame,
    protocol: str = "telemetry_only",
    explicit: list[str] | None = None,
    allow_leakage: bool = False,
) -> list[str]:
    if explicit:
        missing = [column for column in explicit if column not in df.columns]
        if missing:
            raise ValueError(f"satellite_battery feature columns not found: {missing}")
        leakage = sorted(set(explicit) & LEAKAGE_COLUMNS)
        if leakage and not allow_leakage:
            raise ValueError(
                "satellite_battery feature columns contain hidden/label leakage columns: "
                f"{leakage}. Set allow_leakage_features only for explicit oracle ablations."
            )
        return list(explicit)
    protocol_key = str(protocol).lower()
    if protocol_key in {"telemetry_only", "telemetry-only", "vit"}:
        return TELEMETRY_ONLY_COLUMNS.copy()
    if protocol_key in {"pg_stda_v1", "telemetry_proxy", "telemetry_with_proxy"}:
        return TELEMETRY_ONLY_COLUMNS + BATTERY_PROXY_V1_COLUMNS
    if protocol_key in {"target_rich", "target-rich", "with_soc_est"}:
        return TARGET_RICH_COLUMNS.copy()
    raise ValueError(f"unknown satellite_battery feature protocol: {protocol}")


def satellite_battery_dataframe_to_unit_dicts(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "normalized_rul",
    sort_columns: tuple[str, str] = ("unit_id", "time_step"),
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    if target_column not in df.columns:
        raise ValueError(f"satellite_battery target column not found: {target_column}")
    series_by_unit: dict[int, np.ndarray] = {}
    rul_by_unit: dict[int, np.ndarray] = {}
    for unit_id, group in df.sort_values(list(sort_columns)).groupby("unit_id", sort=False):
        series_by_unit[int(unit_id)] = group[feature_columns].to_numpy(dtype=np.float32)
        rul_by_unit[int(unit_id)] = group[target_column].to_numpy(dtype=np.float32)
    return series_by_unit, rul_by_unit


def _ids_for_split(df: pd.DataFrame, split: str) -> np.ndarray:
    return df.loc[df["split"] == split, "unit_id"].drop_duplicates().to_numpy(dtype=np.int64)


def build_satellite_battery_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    data_cfg = config["data"]
    seed = int(config["experiment"].get("seed", data_cfg.get("seed", 42)))
    csv_path = Path(data_cfg.get("csv_path", "data/simulated/satellite_battery/satellite_battery_sim.csv"))
    df = load_satellite_battery_csv(csv_path)
    if _uses_battery_proxy_features(data_cfg, data_cfg.get("feature_columns")):
        df = add_battery_physics_proxy_features(df)
    feature_columns = infer_satellite_battery_feature_columns(
        df,
        protocol=str(data_cfg.get("feature_protocol", "telemetry_only")),
        explicit=data_cfg.get("feature_columns"),
        allow_leakage=bool(data_cfg.get("allow_leakage_features", False)),
    )
    target_column = str(data_cfg.get("target_column", "normalized_rul"))
    series_all, rul_all = satellite_battery_dataframe_to_unit_dicts(df, feature_columns, target_column=target_column)
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
        raise ValueError("satellite_battery train/val/test split produced an empty split")
    normalizer = fit_normalizer(list(train_series.values()), method=str(data_cfg.get("normalize", "zscore")))
    common_kwargs = {
        "window_size": int(data_cfg["window_size"]),
        "stride": int(data_cfg.get("stride", 1)),
        "max_rul": _optional_float(data_cfg.get("max_rul", None)),
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


def _ensure_normalized_rul(df: pd.DataFrame, target_column: str, soh_eol: float = 0.8) -> pd.DataFrame:
    if target_column in df.columns:
        return df
    out = df.copy()
    if target_column == "normalized_rul":
        if "rul" not in out.columns:
            out = load_processed_battery_csv_from_frame(out, soh_eol=soh_eol)
        max_rul = out.groupby("unit_id")["rul"].transform(lambda x: max(float(x.max()), 1.0))
        out["normalized_rul"] = out["rul"].astype(float) / max_rul
        return out
    raise ValueError(f"cannot derive source target column: {target_column}")


def load_processed_battery_csv_from_frame(df: pd.DataFrame, soh_eol: float = 0.8) -> pd.DataFrame:
    out = df.copy()
    if "soh" not in out.columns and "capacity" in out.columns:
        out["soh"] = out.groupby("unit_id")["capacity"].transform(lambda x: x / max(float(x.iloc[0]), 1e-8))
    if "rul" not in out.columns:
        if "soh" not in out.columns:
            raise ValueError("source battery frame must contain rul, or soh/capacity so RUL can be estimated")
        rul_values = []
        for _, group in out.sort_values(["unit_id", "cycle"]).groupby("unit_id", sort=False):
            cycles = group["cycle"].to_numpy(dtype=np.float32)
            soh = group["soh"].to_numpy(dtype=np.float32)
            eol_candidates = np.where(soh <= float(soh_eol))[0]
            eol_cycle = cycles[eol_candidates[0]] if len(eol_candidates) else cycles[-1]
            rul_values.extend(np.maximum(eol_cycle - cycles, 0.0).tolist())
        out["rul"] = np.asarray(rul_values, dtype=np.float32)
    return out


def _source_battery_to_unit_dicts(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    if target_column not in df.columns:
        raise ValueError(f"source battery target column not found: {target_column}")
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"source battery feature columns not found: {missing}")
    series_by_unit: dict[int, np.ndarray] = {}
    rul_by_unit: dict[int, np.ndarray] = {}
    for unit_id, group in df.sort_values(["unit_id", "cycle"]).groupby("unit_id", sort=False):
        series_by_unit[int(unit_id)] = group[feature_columns].to_numpy(dtype=np.float32)
        rul_by_unit[int(unit_id)] = group[target_column].to_numpy(dtype=np.float32)
    return series_by_unit, rul_by_unit


def build_nasa_satellite_battery_transfer_loaders(config: dict):
    data_cfg = config["data"]
    source_csv = Path(data_cfg.get("source_csv_path", "data/processed/nasa_battery.csv"))
    target_csv = Path(data_cfg.get("target_csv_path", "data/simulated/satellite_battery/satellite_battery_sim.csv"))
    source_df = load_processed_battery_csv(source_csv, soh_eol=float(data_cfg.get("source_soh_eol", 0.8)))
    current_multiplier = float(data_cfg.get("source_current_multiplier", -1.0))
    if "current" in source_df.columns and current_multiplier != 1.0:
        source_df = source_df.copy()
        source_df["current"] = source_df["current"].astype(float) * current_multiplier
    source_target_column = str(data_cfg.get("source_target_column", "normalized_rul"))
    source_df = _ensure_normalized_rul(source_df, source_target_column, soh_eol=float(data_cfg.get("source_soh_eol", 0.8)))
    source_feature_columns = [str(column) for column in data_cfg.get("source_feature_columns", TELEMETRY_ONLY_COLUMNS)]
    if _uses_battery_proxy_features(data_cfg, source_feature_columns, data_cfg.get("target_feature_columns")):
        source_df = add_battery_physics_proxy_features(source_df)
    source_series_all, source_rul_all = _source_battery_to_unit_dicts(source_df, source_feature_columns, source_target_column)
    source_units = data_cfg.get("source_units")
    if source_units:
        source_ids = {int(unit_id) for unit_id in source_units}
        source_series = _filter_units(source_series_all, source_ids)
        source_rul = _filter_units(source_rul_all, source_ids)
    else:
        source_series = source_series_all
        source_rul = source_rul_all
    if not source_series:
        raise ValueError("source battery split produced no units")

    target_df = load_satellite_battery_csv(target_csv)
    if _uses_battery_proxy_features(data_cfg, source_feature_columns, data_cfg.get("target_feature_columns")):
        target_df = add_battery_physics_proxy_features(target_df)
    target_feature_columns = infer_satellite_battery_feature_columns(
        target_df,
        protocol=str(data_cfg.get("target_feature_protocol", "telemetry_only")),
        explicit=data_cfg.get("target_feature_columns"),
        allow_leakage=bool(data_cfg.get("allow_leakage_features", False)),
    )
    if len(source_feature_columns) != len(target_feature_columns):
        raise ValueError(
            f"source and target feature dimensions must match, got {len(source_feature_columns)} and {len(target_feature_columns)}"
        )
    target_column = str(data_cfg.get("target_column", "normalized_rul"))
    target_series_all, target_rul_all = satellite_battery_dataframe_to_unit_dicts(
        target_df,
        target_feature_columns,
        target_column=target_column,
    )
    target_train_ids = _ids_for_split(target_df, str(data_cfg.get("target_train_split", "train")))
    target_val_ids = _ids_for_split(target_df, str(data_cfg.get("target_val_split", "val")))
    target_test_ids = _ids_for_split(target_df, str(data_cfg.get("target_test_split", "test")))
    target_train_series = _filter_units(target_series_all, target_train_ids)
    target_val_series = _filter_units(target_series_all, target_val_ids)
    target_test_series = _filter_units(target_series_all, target_test_ids)
    target_train_rul = _filter_units(target_rul_all, target_train_ids)
    target_val_rul = _filter_units(target_rul_all, target_val_ids)
    target_test_rul = _filter_units(target_rul_all, target_test_ids)
    if not target_train_series or not target_val_series or not target_test_series:
        raise ValueError("satellite target split produced an empty split")

    normalize_scope = str(data_cfg.get("normalizer_scope", "source_target_train")).lower()
    if normalize_scope == "source":
        normalizer = fit_normalizer(list(source_series.values()), method=str(data_cfg.get("normalize", "zscore")))
    elif normalize_scope == "target_train":
        normalizer = fit_normalizer(list(target_train_series.values()), method=str(data_cfg.get("normalize", "zscore")))
    elif normalize_scope in {"source_target_train", "source+target_train"}:
        normalizer = fit_normalizer(
            list(source_series.values()) + list(target_train_series.values()),
            method=str(data_cfg.get("normalize", "zscore")),
        )
    else:
        raise ValueError(f"unknown normalizer_scope: {normalize_scope}")

    max_rul = _optional_float(data_cfg.get("max_rul", None))
    source_kwargs = {
        "window_size": int(data_cfg.get("source_window_size", data_cfg["window_size"])),
        "stride": int(data_cfg.get("source_stride", data_cfg.get("stride", 1))),
        "max_rul": max_rul,
        "normalizer": normalizer,
        "target_horizon": int(data_cfg.get("source_target_horizon", data_cfg.get("target_horizon", 0))),
    }
    target_kwargs = {
        "window_size": int(data_cfg.get("target_window_size", data_cfg["window_size"])),
        "stride": int(data_cfg.get("target_stride", data_cfg.get("stride", 1))),
        "max_rul": max_rul,
        "normalizer": normalizer,
        "target_horizon": int(data_cfg.get("target_target_horizon", data_cfg.get("target_horizon", 0))),
    }
    target_stage_source = str(data_cfg.get("target_stage_source", "rul")).lower()
    target_stage_kwargs: dict[str, dict[int, np.ndarray]] = {}
    if target_stage_source in {"rul", "true_rul", "target_rul"}:
        target_stage_kwargs = {}
    elif target_stage_source in {"time", "time_progress", "progress", "pseudo_time"}:
        target_stage_kwargs = {"stage_by_unit": _time_progress_stage_by_unit(target_train_series)}
    else:
        raise ValueError(f"unknown target_stage_source: {target_stage_source}")
    source_ds = WindowedTimeSeriesDataset(source_series, source_rul, **source_kwargs)
    target_train_ds = WindowedTimeSeriesDataset(target_train_series, target_train_rul, **target_kwargs, **target_stage_kwargs)
    target_val_ds = WindowedTimeSeriesDataset(target_val_series, target_val_rul, **target_kwargs)
    target_test_ds = WindowedTimeSeriesDataset(target_test_series, target_test_rul, **target_kwargs)
    batch_size = int(config["train"]["batch_size"])
    return (
        DataLoader(source_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(target_train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(target_val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        DataLoader(target_test_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        len(source_feature_columns),
    )
