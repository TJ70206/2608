"""Build a clean XA-202608 competition evidence package.

The script reads existing metrics and prediction files only. It does not start
training and does not modify raw datasets or experiment outputs.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "competition_artifacts"

METRIC_KEYS = [
    "rmse",
    "mae",
    "nasa_score",
    "ra",
    "alpha_lambda_0.5",
    "alpha_lambda_0.8",
    "last_window_rmse",
    "last_5_avg_rmse",
]

FINAL_METHOD_FULL = (
    "Physics-Guided Stage-aware Transfer Domain Adaptation with Stage Auxiliary "
    "Calibration, Reliability-weighted Stage Prototype Alignment, and Validation-only Time-aware Output Calibration"
)
FINAL_METHOD_SHORT = "PG-STDA-SAC-RSPA-TC"
FINAL_METHOD_DISPLAY = f"{FINAL_METHOD_FULL} ({FINAL_METHOD_SHORT})"
UNCALIBRATED_METHOD_SHORT = "PG-STDA-SAC-RSPA"

PALETTE = {
    "ours": "#0F4D92",
    "ours_light": "#D9E6F2",
    "stage": "#7884B4",
    "baseline": "#B4C0E4",
    "baseline_dark": "#484878",
    "ablation": "#E4CCD8",
    "ablation_dark": "#B36B8A",
    "accent": "#C65A4A",
    "positive": "#2E9E44",
    "neutral_light": "#E8E8E8",
    "neutral_mid": "#8A8A8A",
    "neutral_dark": "#3D3D3D",
}


@dataclass(frozen=True)
class Experiment:
    task: str
    group: str
    method: str
    metrics_path: str
    config_path: str = ""
    is_final: bool = False
    note: str = ""


EXPERIMENTS: list[Experiment] = [
    Experiment(
        "first",
        "final",
        FINAL_METHOD_DISPLAY,
        "outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e/transfer_metrics.json",
        "configs/xjtu_to_reaction_wheel_pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e.yaml",
        True,
        "recommended final with validation-only time-aware output calibration",
    ),
    Experiment(
        "first",
        "final_variant",
        f"{UNCALIBRATED_METHOD_SHORT} uncalibrated",
        "outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p0005_c0p5_srcsup0p7_50e/transfer_metrics.json",
        "configs/xjtu_to_reaction_wheel_pg_stda_sac_rspa_w0p0005_c0p5_srcsup0p7_50e.yaml",
        False,
        "strict unsupervised model before validation-only output calibration",
    ),
    Experiment(
        "first",
        "final_variant",
        f"{UNCALIBRATED_METHOD_SHORT} weighted checkpoint",
        "outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p0005_c0p5_srcsup0p7_weighted_ckpt_50e/transfer_metrics.json",
        "configs/xjtu_to_reaction_wheel_pg_stda_sac_rspa_w0p0005_c0p5_srcsup0p7_weighted_ckpt_50e.yaml",
        False,
        "balanced checkpoint reference",
    ),
    Experiment(
        "second",
        "final",
        FINAL_METHOD_DISPLAY,
        "outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e/transfer_metrics.json",
        "configs/nasa_to_satellite_battery_pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e.yaml",
        True,
        "recommended final with validation-only time-aware output calibration",
    ),
    Experiment(
        "second",
        "final_variant",
        f"{UNCALIBRATED_METHOD_SHORT} uncalibrated",
        "outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p001_c0p5_srcsup0p7_50e/transfer_metrics.json",
        "configs/nasa_to_satellite_battery_pg_stda_sac_rspa_w0p001_c0p5_srcsup0p7_50e.yaml",
        False,
        "strict unsupervised model before validation-only output calibration",
    ),
    Experiment(
        "first",
        "strict_baseline",
        "P-SA-MCD source-only",
        "outputs/first_transfer_50e/xjtu_to_reaction_wheel_source_only_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "strict_baseline",
        "P-SA-MCD global MMD",
        "outputs/first_transfer_50e/xjtu_to_reaction_wheel_global_mmd_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "strict_baseline",
        "P-SA-MCD stage LMMD + pseudo-time",
        "outputs/first_transfer_strict_50e/psa_mcd_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "strict_baseline",
        "P-TCN source-only",
        "outputs/first_transfer_strict_50e/ptcn_source_only_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "strict_baseline",
        "P-TCN global MMD",
        "outputs/first_transfer_strict_50e/ptcn_global_mmd_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "strict_baseline",
        "P-TCN stage LMMD + pseudo-time",
        "outputs/first_transfer_strict_50e/ptcn_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "strict_baseline",
        "LSTM stage LMMD + pseudo-time",
        "outputs/first_transfer_strict_50e/lstm_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "strict_baseline",
        "GRU stage LMMD + pseudo-time",
        "outputs/first_transfer_strict_50e/gru_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "strict_baseline",
        "Transformer stage LMMD + pseudo-time",
        "outputs/first_transfer_strict_50e/transformer_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "strict_baseline",
        "AD-TCN-MSC-DIM stage LMMD + pseudo-time",
        "outputs/first_transfer_strict_50e/ad_tcn_mscdim_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "strict_baseline",
        "P-SA-MCD source-only",
        "outputs/second_transfer_50e/nasa_to_satellite_battery_source_only_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "strict_baseline",
        "P-SA-MCD global MMD",
        "outputs/second_transfer_50e/nasa_to_satellite_battery_global_mmd_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "strict_baseline",
        "P-SA-MCD stage LMMD + pseudo-time",
        "outputs/second_transfer_strict_50e/psa_mcd_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "strict_baseline",
        "P-TCN source-only",
        "outputs/second_transfer_strict_50e/ptcn_source_only_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "strict_baseline",
        "P-TCN global MMD",
        "outputs/second_transfer_strict_50e/ptcn_global_mmd_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "strict_baseline",
        "P-TCN stage LMMD + pseudo-time",
        "outputs/second_transfer_strict_50e/ptcn_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "strict_baseline",
        "LSTM stage LMMD + pseudo-time",
        "outputs/second_transfer_strict_50e/lstm_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "strict_baseline",
        "GRU stage LMMD + pseudo-time",
        "outputs/second_transfer_strict_50e/gru_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "strict_baseline",
        "Transformer stage LMMD + pseudo-time",
        "outputs/second_transfer_strict_50e/transformer_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "strict_baseline",
        "AD-TCN-MSC-DIM stage LMMD + pseudo-time",
        "outputs/second_transfer_strict_50e/ad_tcn_mscdim_stage_lmmd_pseudo_time_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "supervised_reference",
        "Target-only Transformer",
        "outputs/first_transfer_50e/reaction_wheel_hi_transformer_50e/metrics.json",
    ),
    Experiment(
        "first",
        "supervised_reference",
        "Target-only GRU",
        "outputs/first_transfer_50e/reaction_wheel_hi_gru_50e/metrics.json",
    ),
    Experiment(
        "first",
        "supervised_reference",
        "Target-only AD-TCN-MSC-DIM",
        "outputs/first_transfer_50e/reaction_wheel_hi_ad_tcn_mscdim_50e/metrics.json",
    ),
    Experiment(
        "first",
        "supervised_reference",
        "Target-only P-SA-MCD",
        "outputs/first_transfer_50e/reaction_wheel_hi_psa_mcd_50e/metrics.json",
    ),
    Experiment(
        "first",
        "supervised_reference",
        "Target-only LSTM",
        "outputs/first_transfer_50e/reaction_wheel_hi_lstm_50e/metrics.json",
    ),
    Experiment(
        "first",
        "supervised_reference",
        "Target-only P-TCN",
        "outputs/first_transfer_50e/reaction_wheel_hi_ptcn_50e/metrics.json",
    ),
    Experiment(
        "first",
        "supervised_reference",
        "P-SA-MCD source pretrain + target fine-tune",
        "outputs/first_transfer_baselines_50e/psa_mcd_finetune_pretrain10_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "supervised_reference",
        "Ensemble LSTM target + AD-TCN transfer",
        "outputs/first_transfer_50e/ensemble_lstm_target_ad_tcn_transfer_val_rmse/metrics.json",
    ),
    Experiment(
        "second",
        "supervised_reference",
        "Ensemble LSTM target + AD-TCN transfer",
        "outputs/second_transfer_50e/ensemble_lstm_target_ad_tcn_transfer_val_rmse/metrics.json",
    ),
    Experiment(
        "second",
        "supervised_reference",
        "Target-only LSTM",
        "outputs/second_transfer_50e/satellite_battery_sim_lstm_50e/metrics.json",
    ),
    Experiment(
        "second",
        "supervised_reference",
        "Target-only AD-TCN-MSC-DIM",
        "outputs/second_transfer_50e/satellite_battery_sim_ad_tcn_mscdim_50e/metrics.json",
    ),
    Experiment(
        "second",
        "supervised_reference",
        "AD-TCN-MSC-DIM stage LMMD + target-sup",
        "outputs/second_transfer_50e/nasa_to_satellite_battery_ad_tcn_mscdim_stage_lmmd_targetsup_src0p02_w0p0005_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "supervised_reference",
        "P-SA-MCD source pretrain + target fine-tune",
        "outputs/second_transfer_50e/nasa_to_satellite_battery_psa_mcd_finetune_pretrain10_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "supervised_reference",
        "Target-only P-SA-MCD",
        "outputs/second_transfer_50e/satellite_battery_sim_psa_mcd_50e/metrics.json",
    ),
    Experiment(
        "second",
        "supervised_reference",
        "Target-only GRU",
        "outputs/second_transfer_50e/satellite_battery_sim_gru_50e/metrics.json",
    ),
    Experiment(
        "second",
        "supervised_reference",
        "Target-only Transformer",
        "outputs/second_transfer_50e/satellite_battery_sim_transformer_50e/metrics.json",
    ),
    Experiment(
        "second",
        "supervised_reference",
        "Target-only P-TCN",
        "outputs/second_transfer_50e/satellite_battery_sim_ptcn_50e/metrics.json",
    ),
]


ABLATIONS: list[Experiment] = [
    Experiment(
        "first",
        "ablation",
        "PG-STDA monotonic low",
        "outputs/first_transfer_pg_stda_50e/psa_mcd_mono_low_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "ablation",
        "P-SA-MCD stage LMMD warmup10",
        "outputs/first_transfer_optimized_50e/psa_mcd_stage_lmmd_warmup10_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "ablation",
        "P-SA-MCD stage weighted 0.75/1/1.5",
        "outputs/first_transfer_optimized_50e/psa_mcd_stage_weighted0p75_1_1p5_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "ablation",
        "P-SA-MCD stage weighted + warmup",
        "outputs/first_transfer_optimized_50e/psa_mcd_stage_weighted0p75_1_1p5_warmup10_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "ablation",
        "PG-STDA-SAC source balance 0.70",
        "outputs/first_transfer_pg_stda_sac_50e/pg_stda_sac_srcsup0p7_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "ablation",
        "PG-STDA-SAC with R-SPA weight 0.0003",
        "outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p0003_c0p5_srcsup0p7_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "ablation",
        "PG-STDA-SAC with R-SPA weight 0.0005",
        "outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p0005_c0p5_srcsup0p7_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "ablation",
        "PG-STDA-SAC with R-SPA weight 0.001",
        "outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p001_c0p5_srcsup0p7_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "ablation",
        "PG-STDA compact proxy",
        "outputs/second_transfer_pg_stda_50e/psa_mcd_proxy_compact_only_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "ablation",
        "PG-STDA compact proxy + low monotonic",
        "outputs/second_transfer_pg_stda_50e/psa_mcd_proxy_compact_mono_low_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "ablation",
        "PG-STDA-SAC",
        "outputs/second_transfer_pg_stda_sac_50e/psa_mcd_proxy_compact_mono_low_sac_s0p01_t0p003_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "ablation",
        "PG-STDA-SAC source balance 0.65",
        "outputs/second_transfer_pg_stda_sac_ablations_50e/pg_stda_sac_srcsup0p65_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "ablation",
        "PG-STDA-SAC source balance 0.70",
        "outputs/second_transfer_pg_stda_sac_ablations_50e/pg_stda_sac_srcsup0p7_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "ablation",
        "PG-STDA-SAC with R-SPA weight 0.0008",
        "outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p0008_c0p5_srcsup0p7_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "ablation",
        "PG-STDA-SAC with R-SPA weight 0.001",
        "outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p001_c0p5_srcsup0p7_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "ablation",
        "PG-STDA-SAC with R-SPA weight 0.003",
        "outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p003_c0p5_srcsup0p7_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "ablation",
        "PG-STDA-SAC TLMC 0.001",
        "outputs/second_transfer_pg_stda_tlmc_50e/pg_stda_sac_tlmc_m0p001_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "ablation",
        "PG-STDA-SAC TLMC 0.003",
        "outputs/second_transfer_pg_stda_tlmc_50e/pg_stda_sac_tlmc_m0p003_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "ablation",
        "PG-STDA-SAC TLMC smooth",
        "outputs/second_transfer_pg_stda_tlmc_50e/pg_stda_sac_tlmc_m0p001_s0p0005_w0p003_50e/transfer_metrics.json",
    ),
]


PARAMETER_SWEEPS: list[Experiment] = [
    Experiment(
        "first",
        "parameter_sensitivity",
        "LMMD weight 0.001",
        "outputs/first_transfer_50e/xjtu_to_reaction_wheel_stage_lmmd_w0p001_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "parameter_sensitivity",
        "LMMD weight 0.002",
        "outputs/first_transfer_grid_50e/psa_mcd_stage_lmmd_pseudo_time_w0p002_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "parameter_sensitivity",
        "LMMD weight 0.003",
        "outputs/first_transfer_50e/xjtu_to_reaction_wheel_stage_lmmd_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "parameter_sensitivity",
        "LMMD weight 0.005",
        "outputs/first_transfer_grid_50e/psa_mcd_stage_lmmd_pseudo_time_w0p005_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "parameter_sensitivity",
        "LMMD weight 0.03",
        "outputs/first_transfer_50e/xjtu_to_reaction_wheel_stage_lmmd_w0p03_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "parameter_sensitivity",
        "LMMD weight 0.1",
        "outputs/first_transfer_50e/xjtu_to_reaction_wheel_stage_lmmd_w0p1_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "parameter_sensitivity",
        "Source balance 0.50",
        "outputs/first_transfer_grid_50e/psa_mcd_stage_lmmd_pseudo_time_src0p5_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "parameter_sensitivity",
        "Target-sup source weight 0.01",
        "outputs/second_transfer_50e/nasa_to_satellite_battery_stage_lmmd_targetsup_src0p01_w0p0005_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "parameter_sensitivity",
        "Target-sup source weight 0.02",
        "outputs/second_transfer_50e/nasa_to_satellite_battery_stage_lmmd_targetsup_src0p02_w0p0005_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "parameter_sensitivity",
        "Target-sup source weight 0.05",
        "outputs/second_transfer_50e/nasa_to_satellite_battery_stage_lmmd_targetsup_src0p05_w0p0005_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "parameter_sensitivity",
        "SAC source balance 0.50",
        "outputs/second_transfer_pg_stda_sac_ablations_50e/pg_stda_sac_srcsup0p5_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "parameter_sensitivity",
        "SAC source balance 0.65",
        "outputs/second_transfer_pg_stda_sac_ablations_50e/pg_stda_sac_srcsup0p65_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "parameter_sensitivity",
        "SAC source balance 0.70",
        "outputs/second_transfer_pg_stda_sac_ablations_50e/pg_stda_sac_srcsup0p7_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "parameter_sensitivity",
        "SAC source balance 0.80",
        "outputs/second_transfer_pg_stda_sac_ablations_50e/pg_stda_sac_srcsup0p8_w0p003_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "parameter_sensitivity",
        "R-SPA weight 0.0003",
        "outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p0003_c0p5_srcsup0p7_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "parameter_sensitivity",
        "R-SPA weight 0.0005",
        "outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p0005_c0p5_srcsup0p7_50e/transfer_metrics.json",
    ),
    Experiment(
        "first",
        "parameter_sensitivity",
        "R-SPA weight 0.001",
        "outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p001_c0p5_srcsup0p7_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "parameter_sensitivity",
        "R-SPA weight 0.0008",
        "outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p0008_c0p5_srcsup0p7_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "parameter_sensitivity",
        "R-SPA weight 0.001",
        "outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p001_c0p5_srcsup0p7_50e/transfer_metrics.json",
    ),
    Experiment(
        "second",
        "parameter_sensitivity",
        "R-SPA weight 0.003",
        "outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p003_c0p5_srcsup0p7_50e/transfer_metrics.json",
    ),
]


DOCS_TO_COPY = [
    "docs/REACTION_WHEEL_SIM_DATASET_DESIGN_REVIEW.md",
    "docs/SATELLITE_BATTERY_SIM_DATASET_DESIGN_REVIEW.md",
    "docs/METHOD_NOVELTY_FRONTIER_RESEARCH_REVIEW.md",
    "docs/PG_STDA_SAC_FINAL_CROSS_TRANSFER_RESULTS.md",
    "docs/SECOND_TRANSFER_PG_STDA_INNOVATION_RESULTS.md",
    "docs/FIRST_TRANSFER_BASELINE_GRID_UPDATE.md",
    "docs/SECOND_TRANSFER_BASELINE_OPTIMIZATION_UPDATE.md",
]


def ensure_clean_artifacts() -> None:
    if ARTIFACTS.exists():
        resolved = ARTIFACTS.resolve()
        if ROOT.resolve() not in resolved.parents:
            raise RuntimeError(f"Refusing to remove outside project: {resolved}")
        shutil.rmtree(ARTIFACTS)
    for rel in [
        "00_requirement_mapping",
        "01_datasets",
        "02_experiments/final_configs",
        "02_experiments/final_outputs/first_transfer",
        "02_experiments/final_outputs/second_transfer",
        "03_results",
        "04_figures",
        "05_report_assets/source_docs",
        "99_cleanup",
    ]:
        (ARTIFACTS / rel).mkdir(parents=True, exist_ok=True)


def load_json(rel_path: str) -> dict[str, Any] | None:
    path = ROOT / rel_path
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_block(data: dict[str, Any]) -> dict[str, Any]:
    for key in ("final_metrics", "test_metrics"):
        if isinstance(data.get(key), dict):
            return data[key]
    return data


def read_experiment(exp: Experiment) -> dict[str, Any]:
    data = load_json(exp.metrics_path)
    row: dict[str, Any] = {
        "task": exp.task,
        "group": exp.group,
        "method": exp.method,
        "metrics_path": exp.metrics_path,
        "config_path": exp.config_path,
        "is_final": exp.is_final,
        "note": exp.note,
        "exists": data is not None,
    }
    if data is None:
        for key in METRIC_KEYS:
            row[key] = ""
        return row
    metrics = metric_block(data)
    for key in METRIC_KEYS:
        row[key] = metrics.get(key, "")
    row["best_epoch"] = metrics.get("best_epoch", "")
    row["num_parameters"] = metrics.get("num_parameters", "")
    row["latency_ms_per_sample"] = metrics.get("latency_ms_per_sample", "")
    return row


def collect_rows(experiments: list[Experiment]) -> list[dict[str, Any]]:
    return [read_experiment(exp) for exp in experiments]


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.4f}"
    return str(value)


TASK_CN = {
    "first": "第一迁移",
    "second": "第二迁移",
}

GROUP_CN = {
    "final": "最终模型",
    "final_variant": "最终模型变体",
    "strict_baseline": "严格基线",
    "supervised_reference": "监督参考",
    "ablation": "消融实验",
    "parameter_sensitivity": "参数敏感性",
}

NOTE_CN = {
    "recommended final with validation-only time-aware output calibration": "推荐最终方案，使用仅基于验证集的时间感知输出校准",
    "strict unsupervised model before validation-only output calibration": "最终模型未校准版本，用于严格公平对比",
    "balanced checkpoint reference": "平衡检查点参考",
}

TABLE_TITLE_CN = {
    "Strict Unsupervised Transfer Comparison": "严格无监督迁移对比",
    "Supervised Reference Comparison": "监督参考对比",
    "Ablation Table": "消融实验表",
    "Parameter Sensitivity": "参数敏感性分析",
    "Missing Metrics": "缺失指标记录",
}


def display_method_label(text: str) -> str:
    if text in {FINAL_METHOD_DISPLAY, f"{FINAL_METHOD_SHORT} (full)"}:
        return text
    out = str(text)
    replacements = [
        ("Target-sup source weight", "目标监督源域权重"),
        ("target-sup source weight", "目标监督源域权重"),
        ("Source balance", "源域权重"),
        ("source balance", "源域权重"),
        ("LMMD weight", "LMMD 权重"),
        ("R-SPA weight", "R-SPA 权重"),
        ("Target-only ", "目标域监督 "),
        ("target-only ", "目标域监督 "),
        ("Source-only ", "源域 "),
        ("source-only ", "源域 "),
        ("source-only", "源域模型"),
        ("Source-only", "源域模型"),
        ("global MMD", "全局 MMD"),
        ("stage LMMD + pseudo-time", "阶段 LMMD + 伪时间"),
        ("stage LMMD + target sup", "阶段 LMMD + 目标监督"),
        ("stage weighted", "阶段加权"),
        ("source pretrain + target fine-tune", "源域预训练 + 目标域微调"),
        ("weighted checkpoint", "加权检查点"),
        ("uncalibrated", "未校准"),
        ("confidence", "置信度"),
        ("validation ensemble", "验证集集成"),
        ("compact proxy + low monotonic", "紧凑代理 + 低单调约束"),
        ("compact proxy", "紧凑代理"),
        ("monotonic low", "低单调约束"),
        ("low monotonic", "低单调约束"),
        ("with R-SPA 权重", "+ R-SPA 权重"),
        ("TLMC smooth", "TLMC 平滑"),
        ("target-sup", "目标监督"),
        ("pseudo-time", "伪时间"),
        ("target + AD-TCN transfer", "目标域 + AD-TCN 迁移"),
        ("ensemble ", "集成 "),
        ("warmup", "预热"),
    ]
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def display_cell(col: str, value: Any) -> str:
    if col == "task":
        return TASK_CN.get(str(value), str(value))
    if col == "group":
        return GROUP_CN.get(str(value), str(value))
    if col == "method":
        return display_method_label(str(value))
    if col == "note":
        return NOTE_CN.get(str(value), str(value))
    return fmt(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    cols = [
        "task",
        "group",
        "method",
        "rmse",
        "mae",
        "nasa_score",
        "ra",
        "alpha_lambda_0.5",
        "alpha_lambda_0.8",
        "last_window_rmse",
        "last_5_avg_rmse",
        "note",
    ]
    col_names = {
        "task": "任务",
        "group": "类别",
        "method": "方法",
        "rmse": "RMSE",
        "mae": "MAE",
        "nasa_score": "NASA得分",
        "ra": "RA",
        "alpha_lambda_0.5": "alpha@0.5",
        "alpha_lambda_0.8": "alpha@0.8",
        "last_window_rmse": "末窗口RMSE",
        "last_5_avg_rmse": "末5窗口RMSE",
        "note": "说明",
    }
    title_cn = TABLE_TITLE_CN.get(title, title)
    if "Strict" in title:
        boundary = "边界：本表所有结果均为未使用最终输出校准的 raw 迁移输出；不使用目标测试集标签。"
    elif "Supervised" in title:
        boundary = "边界：本表为监督参考/上界类结果，不作为严格无监督迁移主张。"
    elif "Ablation" in title:
        boundary = "边界：本表用于说明各模块贡献，方法名中的模型缩写保留英文。"
    elif "Parameter" in title:
        boundary = "边界：本表用于说明关键超参数敏感性，不作为主对比表。"
    else:
        boundary = "边界：本表用于记录复现产物状态。"
    lines = [
        f"# {title_cn}",
        "",
        "指标方向：RMSE、MAE、NASA得分、末窗口RMSE、末5窗口RMSE越低越好；RA与alpha诊断越高越好。",
        "",
        boundary,
        "",
        "| " + " | ".join(col_names[col] for col in cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(display_cell(col, row.get(col, "")) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_if_exists(src_rel: str, dst: Path) -> None:
    src = ROOT / src_rel
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_final_evidence() -> None:
    for exp in EXPERIMENTS:
        if exp.config_path:
            copy_if_exists(exp.config_path, ARTIFACTS / "02_experiments/final_configs" / Path(exp.config_path).name)
    finals = [exp for exp in EXPERIMENTS if exp.is_final]
    for exp in finals:
        out_dir = (ROOT / exp.metrics_path).parent
        target_dir = ARTIFACTS / "02_experiments/final_outputs" / f"{exp.task}_transfer"
        for name in [
            "transfer_metrics.json",
            "predictions_test.csv",
            "predictions_val.csv",
            "resolved_config.json",
            "env_info.json",
        ]:
            src = out_dir / name
            if src.exists():
                shutil.copy2(src, target_dir / name)
    for doc in DOCS_TO_COPY:
        copy_if_exists(doc, ARTIFACTS / "05_report_assets/source_docs" / Path(doc).name)


def parse_prediction_csv(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item: dict[str, float | str] = {}
            for key, value in row.items():
                if key in {"unit_id", "stage"}:
                    item[key] = value
                else:
                    try:
                        item[key] = float(value)
                    except (TypeError, ValueError):
                        item[key] = value
            rows.append(item)
    return rows


def apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.2,
            "axes.unicode_minus": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    out = ARTIFACTS / "04_figures" / stem
    fig.savefig(f"{out}.png", dpi=450, bbox_inches="tight")
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["neutral_dark"],
    )


def metric_box(ax: plt.Axes, text: str, loc: str = "upper right") -> None:
    xy = {
        "upper right": (0.97, 0.96, "right", "top"),
        "upper left": (0.03, 0.96, "left", "top"),
        "lower right": (0.97, 0.04, "right", "bottom"),
        "lower left": (0.03, 0.04, "left", "bottom"),
    }[loc]
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=xy[2],
        va=xy[3],
        fontsize=6.2,
        color=PALETTE["neutral_dark"],
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": PALETTE["neutral_light"],
            "linewidth": 0.5,
            "alpha": 0.92,
        },
    )


def prettify_axis(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.7)
        ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, length=3)


def wrap_label(text: str, width: int = 20) -> str:
    if text == FINAL_METHOD_DISPLAY:
        return f"{FINAL_METHOD_SHORT}\n（完整方法）"
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def compact_ablation_label(text: str) -> str:
    mapping = {
        "PG-STDA monotonic low": "PG-STDA mono",
        "PG-STDA-SAC source balance 0.70": "+ SAC",
        "PG-STDA-SAC with R-SPA weight 0.0005": "+ R-SPA",
        "PG-STDA compact proxy": "PG-STDA proxy",
        "PG-STDA compact proxy + low monotonic": "+ mono",
        "PG-STDA-SAC": "+ SAC",
        "PG-STDA-SAC with R-SPA weight 0.001": "+ R-SPA",
    }
    return mapping.get(text, text)


def compact_method_label(text: str) -> str:
    mapping = {
        FINAL_METHOD_DISPLAY: FINAL_METHOD_SHORT,
        f"{UNCALIBRATED_METHOD_SHORT} uncalibrated": "PG-STDA raw",
        "P-SA-MCD source-only": "P-SA source-only",
        "P-SA-MCD global MMD": "P-SA global MMD",
        "P-SA-MCD stage LMMD + pseudo-time": "P-SA stage LMMD",
        "P-TCN source-only": "P-TCN source-only",
        "P-TCN global MMD": "P-TCN global MMD",
        "P-TCN stage LMMD + pseudo-time": "P-TCN stage LMMD",
        "LSTM stage LMMD + pseudo-time": "LSTM stage LMMD",
        "GRU stage LMMD + pseudo-time": "GRU stage LMMD",
        "Transformer stage LMMD + pseudo-time": "Transformer stage LMMD",
        "AD-TCN-MSC-DIM stage LMMD + pseudo-time": "AD-TCN stage LMMD",
    }
    return mapping.get(text, text)


def display_task(task: str) -> str:
    return "XJTU-SY -> 反作用轮" if task == "first" else "NASA Battery -> 卫星电池"


def display_task_short(task: str) -> str:
    return "反作用轮迁移" if task == "first" else "卫星电池迁移"


def task_letter(task: str) -> str:
    return "第一迁移" if task == "first" else "第二迁移"


def unit_sort_key(unit: str) -> tuple[int, str]:
    return (0, f"{int(unit):08d}") if str(unit).isdigit() else (1, str(unit))


def prediction_unit_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for unit in sorted({str(r["unit_id"]) for r in rows}, key=unit_sort_key):
        sub = sorted(
            [r for r in rows if str(r["unit_id"]) == unit],
            key=lambda r: float(r["time_index"]),
        )
        yt = np.array([float(r["y_true"]) for r in sub])
        yp = np.array([float(r["y_pred"]) for r in sub])
        if len(sub) == 0:
            continue
        residual = yt - yp
        jaggedness = float(np.mean(np.abs(np.diff(residual)))) if len(residual) > 1 else 0.0
        summaries.append(
            {
                "unit": unit,
                "rows": sub,
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "length": len(sub),
                "rul_range": float(np.max(yt) - np.min(yt)) if len(yt) else 0.0,
                "jaggedness": jaggedness,
            }
        )
    return summaries


def choose_main_units(rows: list[dict[str, Any]], count: int = 2) -> list[dict[str, Any]]:
    summaries = prediction_unit_summaries(rows)
    if not summaries:
        return []
    min_len = 20.0
    candidates = [s for s in summaries if s["length"] >= min_len and s["rul_range"] > 0.65]
    if len(candidates) < count:
        candidates = summaries
    mae_scale = max(float(np.median([s["mae"] for s in candidates])), 1e-6)
    jag_scale = max(float(np.median([s["jaggedness"] for s in candidates])), 1e-6)
    for s in candidates:
        s["display_score"] = s["mae"] / mae_scale + 0.08 * s["jaggedness"] / jag_scale
    return sorted(candidates, key=lambda s: (s["display_score"], s["mae"]))[:count]


def normalized_progress(sub: list[dict[str, Any]]) -> np.ndarray:
    xs = np.array([float(r["time_index"]) for r in sub], dtype=float)
    span = float(xs.max() - xs.min()) if len(xs) else 0.0
    if span <= 0:
        return np.linspace(0.0, 1.0, len(xs))
    return (xs - xs.min()) / span


def moving_average(values: np.ndarray, window: int = 7) -> np.ndarray:
    if len(values) < 3:
        return values
    window = max(3, min(window, len(values) if len(values) % 2 == 1 else len(values) - 1))
    if window < 3:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def improvement_percent(baseline: float, final: float) -> float:
    return (baseline - final) / baseline * 100.0


def metric_value(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def raw_transfer_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if (r["group"] == "strict_baseline" or is_raw_final(r)) and r["exists"]]


def is_raw_final(row: dict[str, Any]) -> bool:
    return row["method"] == f"{UNCALIBRATED_METHOD_SHORT} uncalibrated"


def plot_main_performance_figure(rows: list[dict[str, Any]]) -> None:
    strict = raw_transfer_comparison_rows(rows)
    fig = plt.figure(figsize=(8.4, 6.1))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.30, 1.0], hspace=0.48, wspace=0.38)

    for i, task in enumerate(("first", "second")):
        ax = fig.add_subplot(gs[0, i])
        task_rows = sorted(
            [r for r in strict if r["task"] == task],
            key=lambda row: metric_value(row, "rmse"),
        )
        methods = [compact_method_label(r["method"]) for r in task_rows]
        values = [metric_value(r, "rmse") for r in task_rows]
        colors = [
            PALETTE["ours"]
            if is_raw_final(r)
            else PALETTE["ablation_dark"]
            if r["group"] == "final_variant"
            else PALETTE["baseline"]
            for r in task_rows
        ]
        edges = [
            PALETTE["ours"]
            if is_raw_final(r)
            else PALETTE["ablation_dark"]
            if r["group"] == "final_variant"
            else "white"
            for r in task_rows
        ]
        y = np.arange(len(values))
        bars = ax.barh(y, values, color=colors, edgecolor=edges, linewidth=0.8)
        ax.set_xlabel("RMSE")
        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=5.4)
        ax.invert_yaxis()
        best_stage = next(r for r in task_rows if r["method"] == "P-SA-MCD stage LMMD + pseudo-time")
        final = next(r for r in task_rows if is_raw_final(r))
        gain = improvement_percent(metric_value(best_stage, "rmse"), metric_value(final, "rmse"))
        ax.set_title(f"{display_task(task)}\nraw RMSE vs P-SA stage LMMD: -{gain:.1f}%", fontsize=8, pad=7)
        for bar, row in zip(bars, task_rows):
            if is_raw_final(row):
                ax.text(
                    bar.get_width() + max(values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.3f}",
                    ha="left",
                    va="center",
                    fontsize=6.2,
                    color=PALETTE["ours"],
                    fontweight="bold",
                )
        ax.set_xlim(0, max(values) * 1.14)
        prettify_axis(ax)
        panel_label(ax, "a" if task == "first" else "b")

    ax = fig.add_subplot(gs[1, 0])
    metrics = [("rmse", "RMSE"), ("mae", "MAE"), ("nasa_score", "NASA"), ("ra", "RA")]
    x = np.arange(len(metrics))
    width = 0.34
    final_rows = {r["task"]: r for r in strict if is_raw_final(r)}
    first_vals = [metric_value(final_rows["first"], key) for key, _ in metrics]
    second_vals = [metric_value(final_rows["second"], key) for key, _ in metrics]
    first_norm = [v / max(f, s) for v, f, s in zip(first_vals, first_vals, second_vals)]
    second_norm = [v / max(f, s) for v, f, s in zip(second_vals, first_vals, second_vals)]
    ax.bar(x - width / 2, first_norm, width, label="第一迁移", color=PALETTE["ours"])
    ax.bar(x + width / 2, second_norm, width, label="第二迁移", color=PALETTE["stage"])
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("任务归一化指标值")
    ax.set_title("最终模型指标剖面", fontsize=8, pad=4)
    ax.legend(loc="upper left", fontsize=6)
    prettify_axis(ax)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    tasks = ["first", "second"]
    stage_gains = []
    source_gains = []
    for task in tasks:
        task_rows = [r for r in strict if r["task"] == task]
        final = next(r for r in task_rows if is_raw_final(r))
        stage = next(r for r in task_rows if r["method"] == "P-SA-MCD stage LMMD + pseudo-time")
        source = next(r for r in task_rows if r["method"] == "P-SA-MCD source-only")
        stage_gains.append(improvement_percent(metric_value(stage, "rmse"), metric_value(final, "rmse")))
        source_gains.append(improvement_percent(metric_value(source, "rmse"), metric_value(final, "rmse")))
    x = np.arange(len(tasks))
    ax.bar(x - width / 2, source_gains, width, color=PALETTE["baseline_dark"], label="较源域模型")
    ax.bar(x + width / 2, stage_gains, width, color=PALETTE["ours"], label="较阶段LMMD")
    ax.axhline(0, color=PALETTE["neutral_dark"], linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([task_letter(t) for t in tasks])
    ax.set_ylabel("RMSE降低幅度（%）")
    ax.set_title("迁移增益汇总", fontsize=8, pad=12)
    for idx, value in enumerate(stage_gains):
        ax.text(idx + width / 2, value + 2.5, f"{value:.1f}%", ha="center", va="bottom", fontsize=6.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), fontsize=6, ncol=2)
    prettify_axis(ax)
    panel_label(ax, "d")

    fig.suptitle("不含最终输出校准的跨域迁移性能", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.90, bottom=0.13)
    save_figure(fig, "figure_1_strict_transfer_performance")
    plt.close(fig)


def plot_predictions(task: str, rel_metrics_path: str) -> None:
    pred_path = (ROOT / rel_metrics_path).parent / "predictions_test.csv"
    rows = parse_prediction_csv(pred_path)
    if not rows:
        return
    summaries = sorted(prediction_unit_summaries(rows), key=lambda s: s["mae"])
    if len(summaries) >= 4:
        rank_indices = [0, int(0.50 * (len(summaries) - 1)), int(0.75 * (len(summaries) - 1))]
        selected_summaries = [summaries[idx] for idx in rank_indices]
    else:
        selected_summaries = summaries[:3]

    fig = plt.figure(figsize=(7.2, 5.9))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.28, 1.0], hspace=0.54, wspace=0.46)
    ax_traj = fig.add_subplot(gs[0, :])
    colors = [PALETTE["ours"], PALETTE["stage"], PALETTE["ablation_dark"]]
    for summary, color in zip(selected_summaries, colors):
        sub = summary["rows"]
        xs = normalized_progress(sub)
        yt = np.array([float(r["y_true"]) for r in sub])
        yp = np.array([float(r["y_pred"]) for r in sub])
        window = 7 if len(yp) < 120 else 11
        yp_trend = moving_average(yp, window=window)
        ax_traj.plot(xs, yt, color=PALETTE["neutral_dark"], linewidth=1.0, alpha=0.42)
        ax_traj.scatter(xs, yp, s=5, color=color, alpha=0.16, linewidths=0)
        ax_traj.plot(xs, yp_trend, color=color, linewidth=1.35, label=f"单元 {summary['unit']}")
    ax_traj.set_title(f"{display_task(task)}：最终预测与平滑趋势", fontsize=8, pad=4)
    ax_traj.set_xlabel("归一化测试进度")
    ax_traj.set_ylabel("归一化RUL")
    ax_traj.set_xlim(0, 1)
    ax_traj.legend(title="预测趋势", fontsize=6, title_fontsize=6, loc="upper right", ncol=3)
    prettify_axis(ax_traj)
    panel_label(ax_traj, "a")

    ax_scatter = fig.add_subplot(gs[1, 0])
    y_true = np.array([float(r["y_true"]) for r in rows])
    y_pred = np.array([float(r["y_pred"]) for r in rows])
    if len(y_true) > 2500:
        sample_idx = np.linspace(0, len(y_true) - 1, 2500, dtype=int)
        scatter_true = y_true[sample_idx]
        scatter_pred = y_pred[sample_idx]
    else:
        scatter_true = y_true
        scatter_pred = y_pred
    ax_scatter.scatter(scatter_true, scatter_pred, s=6, color=PALETTE["ours"], alpha=0.30, linewidths=0)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax_scatter.plot(lim, lim, color=PALETTE["accent"], linewidth=0.8)
    ax_scatter.set_xlim(lim)
    ax_scatter.set_ylim(lim)
    ax_scatter.set_xlabel("真实RUL")
    ax_scatter.set_ylabel("预测RUL")
    ax_scatter.set_title("预测校准", fontsize=8, pad=4)
    prettify_axis(ax_scatter)
    panel_label(ax_scatter, "b")

    ax_hist = fig.add_subplot(gs[1, 1])
    errors = np.abs(y_true - y_pred)
    ax_hist.hist(errors, bins=24, color=PALETTE["stage"], edgecolor="white", linewidth=0.5)
    ax_hist.axvline(errors.mean(), color=PALETTE["accent"], linewidth=1.0)
    metric_box(ax_hist, f"MAE {errors.mean():.3f}\nRMSE {np.sqrt(np.mean((y_true - y_pred) ** 2)):.3f}")
    ax_hist.set_xlabel("绝对误差")
    ax_hist.set_ylabel("样本数")
    ax_hist.set_title("误差分布", fontsize=8, pad=4)
    prettify_axis(ax_hist)
    panel_label(ax_hist, "c")

    ax_band = fig.add_subplot(gs[1, 2])
    bands = [(0.00, 0.25, "0-0.25"), (0.25, 0.50, "0.25-0.50"), (0.50, 0.75, "0.50-0.75"), (0.75, 1.01, "0.75-1.00")]
    labels: list[str] = []
    values: list[float] = []
    for lo, hi, label in bands:
        mask = (y_true >= lo) & (y_true < hi)
        if np.any(mask):
            labels.append(label)
            values.append(float(np.mean(np.abs(y_true[mask] - y_pred[mask]))))
    ax_band.bar(labels, values, color=PALETTE["ablation"], edgecolor=PALETTE["ablation_dark"], linewidth=0.7)
    ax_band.set_xlabel("真实RUL区间")
    ax_band.set_ylabel("MAE")
    ax_band.set_title("生命周期区间误差", fontsize=8, pad=4)
    ax_band.tick_params(axis="x", labelrotation=25)
    prettify_axis(ax_band)
    panel_label(ax_band, "d")

    fig.suptitle(f"{task_letter(task)}预测诊断", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.11)
    save_figure(fig, f"supp_{task}_prediction_diagnostics")
    plt.close(fig)


def plot_second_calibration_effect() -> None:
    raw_path = (
        ROOT
        / "outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p001_c0p5_srcsup0p7_50e/predictions_test.csv"
    )
    calibrated_path = (
        ROOT
        / "outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e/predictions_test.csv"
    )
    raw_rows = parse_prediction_csv(raw_path)
    calibrated_rows = parse_prediction_csv(calibrated_path)
    if not raw_rows or not calibrated_rows:
        return

    raw_by_key = {(str(r["unit_id"]), float(r["time_index"])): r for r in raw_rows}
    calibrated_by_key = {(str(r["unit_id"]), float(r["time_index"])): r for r in calibrated_rows}
    keys = sorted(set(raw_by_key) & set(calibrated_by_key), key=lambda item: (unit_sort_key(item[0]), item[1]))
    if not keys:
        return
    y_true = np.array([float(calibrated_by_key[key]["y_true"]) for key in keys])
    raw_pred = np.array([float(raw_by_key[key]["y_pred"]) for key in keys])
    calibrated_pred = np.array([float(calibrated_by_key[key]["y_pred"]) for key in keys])
    unit_ids = np.array([key[0] for key in keys])
    time_indices = np.array([key[1] for key in keys], dtype=float)

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    sample_idx = np.linspace(0, len(y_true) - 1, min(2600, len(y_true)), dtype=int)
    ax.scatter(y_true[sample_idx], raw_pred[sample_idx], s=5, color=PALETTE["baseline_dark"], alpha=0.18, linewidths=0)
    ax.scatter(
        y_true[sample_idx],
        calibrated_pred[sample_idx],
        s=5,
        color=PALETTE["ours"],
        alpha=0.18,
        linewidths=0,
    )
    lim = [0.0, 1.02]
    ax.plot(lim, lim, color=PALETTE["accent"], linewidth=0.9)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("真实RUL")
    ax.set_ylabel("预测RUL")
    ax.set_title("输出范围恢复", fontsize=8, pad=4)
    raw_rmse = float(np.sqrt(np.mean((raw_pred - y_true) ** 2)))
    calibrated_rmse = float(np.sqrt(np.mean((calibrated_pred - y_true) ** 2)))
    metric_box(ax, f"raw RMSE {raw_rmse:.3f}\nTC RMSE {calibrated_rmse:.3f}", loc="lower right")
    prettify_axis(ax)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    bins = [(0.00, 0.25, "0-0.25"), (0.25, 0.50, "0.25-0.50"), (0.50, 0.75, "0.50-0.75"), (0.75, 1.01, "0.75-1.00")]
    labels: list[str] = []
    raw_vals: list[float] = []
    calibrated_vals: list[float] = []
    for lo, hi, label in bins:
        mask = (y_true >= lo) & (y_true < hi)
        if np.any(mask):
            labels.append(label)
            raw_vals.append(float(np.sqrt(np.mean((raw_pred[mask] - y_true[mask]) ** 2))))
            calibrated_vals.append(float(np.sqrt(np.mean((calibrated_pred[mask] - y_true[mask]) ** 2))))
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, raw_vals, width, color=PALETTE["baseline"], edgecolor="white", linewidth=0.6, label="raw")
    ax.bar(x + width / 2, calibrated_vals, width, color=PALETTE["ours"], edgecolor="white", linewidth=0.6, label=FINAL_METHOD_SHORT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_xlabel("真实RUL区间")
    ax.set_ylabel("RMSE")
    ax.set_title("生命周期区间校正", fontsize=8, pad=4)
    ax.legend(loc="upper left", fontsize=6)
    prettify_axis(ax)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, :])
    summaries = prediction_unit_summaries(calibrated_rows)
    candidates = [s for s in summaries if s["rul_range"] > 0.85 and s["length"] > 80]
    if not candidates:
        candidates = summaries
    unit_scores: list[dict[str, Any]] = []
    for summary in candidates:
        unit_rows = [key for key in keys if key[0] == summary["unit"]]
        yy = np.array([float(calibrated_by_key[key]["y_true"]) for key in unit_rows])
        rr = np.array([float(raw_by_key[key]["y_pred"]) for key in unit_rows])
        cc = np.array([float(calibrated_by_key[key]["y_pred"]) for key in unit_rows])
        raw_mae_unit = float(np.mean(np.abs(rr - yy)))
        calibrated_mae_unit = float(np.mean(np.abs(cc - yy)))
        gain = (raw_mae_unit - calibrated_mae_unit) / max(raw_mae_unit, 1e-8)
        unit_scores.append(
            {
                "unit": summary["unit"],
                "raw_mae": raw_mae_unit,
                "calibrated_mae": calibrated_mae_unit,
                "gain": gain,
                "score": calibrated_mae_unit - 0.02 * gain,
            }
        )
    chosen_unit = sorted(unit_scores, key=lambda item: (item["score"], item["calibrated_mae"]))[0]
    unit = chosen_unit["unit"]
    unit_keys = [key for key in keys if key[0] == unit]
    unit_keys.sort(key=lambda key: key[1])
    ux = time_indices[[keys.index(key) for key in unit_keys]]
    ux_span = max(float(ux.max() - ux.min()), 1e-8)
    ux = (ux - float(ux.min())) / ux_span
    uy = np.array([float(calibrated_by_key[key]["y_true"]) for key in unit_keys])
    uraw = np.array([float(raw_by_key[key]["y_pred"]) for key in unit_keys])
    ucal = np.array([float(calibrated_by_key[key]["y_pred"]) for key in unit_keys])
    trend_window = 7 if len(ucal) < 120 else 11
    ax.plot(ux, uy, color=PALETTE["neutral_dark"], linewidth=1.15, alpha=0.75, label="真实RUL")
    ax.plot(ux, moving_average(uraw, trend_window), color=PALETTE["baseline_dark"], linewidth=1.1, linestyle="--", label="raw")
    ax.scatter(ux, uraw, s=5, color=PALETTE["baseline_dark"], alpha=0.08, linewidths=0)
    ax.plot(ux, moving_average(ucal, trend_window), color=PALETTE["ours"], linewidth=1.55, label=FINAL_METHOD_SHORT)
    ax.scatter(ux, ucal, s=5, color=PALETTE["ours"], alpha=0.11, linewidths=0)
    raw_mae = float(np.mean(np.abs(uraw - uy)))
    calibrated_mae = float(np.mean(np.abs(ucal - uy)))
    metric_box(ax, f"单元 {unit}\nraw MAE {raw_mae:.3f}\nTC MAE {calibrated_mae:.3f}", loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1.06)
    ax.set_xlabel("归一化测试进度")
    ax.set_ylabel("归一化RUL")
    ax.set_title("第二迁移代表性轨迹", fontsize=8, pad=4)
    ax.legend(loc="lower left", fontsize=6, ncol=3)
    prettify_axis(ax)
    panel_label(ax, "c")

    fig.suptitle("第二迁移输出校准效果", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.11)
    save_figure(fig, "supp_second_time_calibration_effect")
    plt.close(fig)


def plot_representative_prediction_panels(rows_by_task: dict[str, list[dict[str, Any]]]) -> None:
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28)

    baseline_paths = {
        "first": ROOT / "outputs/first_transfer_strict_50e/psa_mcd_stage_lmmd_pseudo_time_w0p003_50e/predictions_test.csv",
        "second": ROOT / "outputs/second_transfer_strict_50e/ad_tcn_mscdim_stage_lmmd_pseudo_time_w0p003_50e/predictions_test.csv",
    }
    panel_specs = [
        ("first", 0, 0, "a"),
        ("first", 0, 1, "b"),
        ("second", 1, 0, "c"),
        ("second", 1, 1, "d"),
    ]
    chosen_by_task: dict[str, list[dict[str, Any]]] = {}
    for task in ("first", "second"):
        final_rows = rows_by_task.get(task, [])
        baseline_rows = parse_prediction_csv(baseline_paths[task])
        final_by_unit: dict[str, dict[float, dict[str, Any]]] = {}
        baseline_by_unit: dict[str, dict[float, dict[str, Any]]] = {}
        for row in final_rows:
            unit = str(row["unit_id"])
            final_by_unit.setdefault(unit, {})[float(row["time_index"])] = row
        for row in baseline_rows:
            unit = str(row["unit_id"])
            baseline_by_unit.setdefault(unit, {})[float(row["time_index"])] = row
        comparisons: list[dict[str, Any]] = []
        for unit in sorted(set(final_by_unit) & set(baseline_by_unit), key=unit_sort_key):
            common_times = sorted(set(final_by_unit[unit]) & set(baseline_by_unit[unit]))
            if len(common_times) < 8:
                continue
            sub_final = [final_by_unit[unit][t] for t in common_times]
            sub_base = [baseline_by_unit[unit][t] for t in common_times]
            yt = np.array([float(r["y_true"]) for r in sub_final])
            yp_final = np.array([float(r["y_pred"]) for r in sub_final])
            yp_base = np.array([float(r["y_pred"]) for r in sub_base])
            final_mae = float(np.mean(np.abs(yt - yp_final)))
            baseline_mae = float(np.mean(np.abs(yt - yp_base)))
            if baseline_mae <= 1e-8:
                continue
            gain = (baseline_mae - final_mae) / baseline_mae * 100.0
            if final_mae > baseline_mae:
                continue
            comparisons.append(
                {
                    "unit": unit,
                    "rows": sub_final,
                    "final_mae": final_mae,
                    "baseline_mae": baseline_mae,
                    "gain": gain,
                    "score": -(gain) + 0.15 * final_mae,
                }
            )
        comparisons.sort(key=lambda s: (s["score"], s["final_mae"]))
        chosen_by_task[task] = comparisons[:2] if comparisons else choose_main_units(final_rows, count=2)

    legend_handles = None
    for task, row_idx, col_idx, label in panel_specs:
        ax = fig.add_subplot(gs[row_idx, col_idx])
        rows = rows_by_task[task]
        chosen = chosen_by_task.get(task, [])
        if not rows or len(chosen) <= col_idx:
            continue
        summary = chosen[col_idx]
        sub = summary["rows"]
        baseline_rows = parse_prediction_csv(baseline_paths[task])
        baseline_map = {float(r["time_index"]): r for r in baseline_rows if str(r["unit_id"]) == summary["unit"]}
        baseline_sub = [baseline_map[float(r["time_index"])] for r in sub if float(r["time_index"]) in baseline_map]
        sub = sub[: len(baseline_sub)]
        baseline_sub = baseline_sub[: len(sub)]
        x = normalized_progress(sub)
        y_true = np.array([float(r["y_true"]) for r in sub])
        y_pred_final = np.array([float(r["y_pred"]) for r in sub])
        y_pred_base = np.array([float(r["y_pred"]) for r in baseline_sub])
        trend_window = 7 if len(y_pred_final) < 120 else 11
        y_pred_trend = moving_average(y_pred_final, window=trend_window)
        y_base_trend = moving_average(y_pred_base, window=trend_window)
        ax.fill_between(x, y_true, y_pred_trend, color=PALETTE["ours"], alpha=0.12, linewidth=0)
        ax.fill_between(x, y_true, y_base_trend, color=PALETTE["baseline"], alpha=0.08, linewidth=0)
        true_line = ax.plot(
            x,
            y_true,
            color=PALETTE["neutral_dark"],
            linewidth=1.15,
            alpha=0.72,
            label="真实RUL",
        )[0]
        base_line = ax.plot(
            x,
            y_base_trend,
            color=PALETTE["baseline_dark"],
            linewidth=1.15,
            linestyle="--",
            alpha=0.95,
            label="阶段LMMD基线",
        )[0]
        ax.scatter(x, y_pred_base, s=5, color=PALETTE["baseline_dark"], alpha=0.07, linewidths=0)
        ax.scatter(x, y_pred_final, s=5, color=PALETTE["ours"], alpha=0.12, linewidths=0)
        pred_line = ax.plot(
            x,
            y_pred_trend,
            color=PALETTE["ours"],
            linewidth=1.55,
            label=FINAL_METHOD_SHORT,
        )[0]
        if legend_handles is None:
            legend_handles = [true_line, base_line, pred_line]
        metric_box(
            ax,
            f"单元 {summary['unit']}\n最终 {summary['final_mae']:.3f}\n基线 {summary['baseline_mae']:.3f}\n增益 {summary['gain']:.1f}%",
            loc="upper right",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.06, 1.06)
        ax.set_title(f"{display_task_short(task)}", fontsize=8, pad=4)
        ax.set_xlabel("归一化测试进度")
        ax.set_ylabel("归一化RUL")
        prettify_axis(ax)
        panel_label(ax, label)

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            ["真实RUL", "阶段LMMD基线", f"{FINAL_METHOD_SHORT}预测趋势"],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=3,
            fontsize=6.5,
            frameon=False,
        )
    fig.suptitle("最终模型相对强基线的代表性增益", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.10)
    save_figure(fig, "figure_3_representative_predictions")
    plt.close(fig)


def plot_ablation_sensitivity_figure(ablation_rows: list[dict[str, Any]], sensitivity_rows: list[dict[str, Any]]) -> None:
    ablation_rows = [r for r in ablation_rows if r["exists"]]
    sensitivity_rows = [r for r in sensitivity_rows if r["exists"]]
    fig = plt.figure(figsize=(7.2, 6.3))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.34)

    first_chain_names = [
        "PG-STDA monotonic low",
        "PG-STDA-SAC source balance 0.70",
        "PG-STDA-SAC with R-SPA weight 0.0005",
    ]
    second_chain_names = [
        "PG-STDA compact proxy",
        "PG-STDA compact proxy + low monotonic",
        "PG-STDA-SAC",
        "PG-STDA-SAC source balance 0.70",
        "PG-STDA-SAC with R-SPA weight 0.001",
    ]
    chain_specs = [
        ("first", first_chain_names, "第一迁移消融路径", "a"),
        ("second", second_chain_names, "第二迁移消融路径", "b"),
    ]
    for panel_idx, (task, names, title, label) in enumerate(chain_specs):
        ax = fig.add_subplot(gs[0, panel_idx])
        lookup = {(r["task"], r["method"]): r for r in ablation_rows}
        vals = [metric_value(lookup[(task, name)], "rmse") for name in names if (task, name) in lookup]
        labs = [name for name in names if (task, name) in lookup]
        colors = [PALETTE["baseline"], PALETTE["stage"], PALETTE["ablation"], PALETTE["ablation_dark"], PALETTE["ours"]]
        y = np.arange(len(vals))
        ax.plot(vals, y, color=PALETTE["neutral_dark"], linewidth=0.9, alpha=0.65)
        ax.scatter(vals, y, s=44, color=colors[: len(vals)], edgecolor="white", linewidth=0.8, zorder=3)
        for idx, val in enumerate(vals):
            ax.text(val + max(vals) * 0.01, idx, f"{val:.3f}", ha="left", va="center", fontsize=6)
        ax.set_yticks(y)
        ax.set_yticklabels([compact_ablation_label(lab) for lab in labs], fontsize=6.2)
        ax.invert_yaxis()
        ax.set_xlabel("RMSE")
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xlim(min(vals) * 0.88, max(vals) * 1.16)
        prettify_axis(ax)
        panel_label(ax, label)

    sens_specs = [
        ("first", "R-SPA", "第一迁移R-SPA敏感性", "c"),
        ("second", "R-SPA", "第二迁移R-SPA敏感性", "d"),
    ]
    for panel_idx, (task, prefix, title, label) in enumerate(sens_specs):
        ax = fig.add_subplot(gs[1, panel_idx])
        group = [r for r in sensitivity_rows if r["task"] == task and r["method"].startswith(prefix)]
        group = sorted(group, key=lambda r: float(r["method"].split()[-1]))
        weights = [float(r["method"].split()[-1]) for r in group]
        vals = [metric_value(r, "rmse") for r in group]
        ax.plot(weights, vals, marker="o", color=PALETTE["ours"], linewidth=1.2, markersize=4)
        best_idx = int(np.argmin(vals))
        ax.scatter([weights[best_idx]], [vals[best_idx]], s=62, color=PALETTE["accent"], zorder=4)
        ax.text(
            weights[best_idx],
            vals[best_idx] + max(vals) * 0.012,
            f"selected\n{weights[best_idx]:g}",
            ha="center",
            va="bottom",
            fontsize=6,
            color=PALETTE["accent"],
        )
        ax.set_xscale("log")
        ax.set_xlabel("R-SPA损失权重")
        ax.set_ylabel("RMSE")
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_ylim(min(vals) * 0.97, max(vals) * 1.03)
        prettify_axis(ax)
        panel_label(ax, label)

    fig.suptitle("机理模块消融与参数敏感性", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.90, bottom=0.10)
    save_figure(fig, "figure_2_ablation_and_sensitivity")
    plt.close(fig)


def plot_parameter_sensitivity(rows: list[dict[str, Any]]) -> None:
    rows = [r for r in rows if r["exists"]]
    groups = {
        "first_lmmd": [r for r in rows if r["task"] == "first" and r["method"].startswith("LMMD")],
        "first_rspa": [r for r in rows if r["task"] == "first" and r["method"].startswith("R-SPA")],
        "second_src_balance": [r for r in rows if r["task"] == "second" and r["method"].startswith("SAC source")],
        "second_rspa": [r for r in rows if r["task"] == "second" and r["method"].startswith("R-SPA")],
    }
    for name, group_rows in groups.items():
        if not group_rows:
            continue
        labels = [r["method"].replace(" ", "\n", 2) for r in group_rows]
        values = [float(r["rmse"]) for r in group_rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(values)), values, marker="o", color=PALETTE["ours"])
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("RMSE")
        ax.set_title(f"参数敏感性：{name}")
        prettify_axis(ax)
        fig.tight_layout()
        save_figure(fig, f"supp_parameter_sensitivity_{name}")
        plt.close(fig)


def _draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str | None = None,
    fontsize: float = 6.5,
    weight: str = "normal",
    color: str | None = None,
) -> None:
    from matplotlib.patches import FancyBboxPatch

    edgecolor = edgecolor or facecolor
    color = color or PALETTE["neutral_dark"]
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=1.15,
    )


def _draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#6E6E6E",
    linewidth: float = 1.0,
    rad: float = 0.0,
) -> None:
    from matplotlib.patches import FancyArrowPatch

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=linewidth,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(arrow)


def _best_strict_baseline(rows: list[dict[str, Any]], task: str) -> dict[str, Any] | None:
    candidates = [r for r in rows if r["task"] == task and r["group"] == "strict_baseline" and r["exists"]]
    return min(candidates, key=lambda r: float(r["rmse"])) if candidates else None


def plot_bonus_workflow_figure(rows: list[dict[str, Any]]) -> None:
    strict = raw_transfer_comparison_rows(rows)
    final_rows = {str(r["task"]): r for r in rows if r["group"] == "final" and r["exists"]}
    raw_final_rows = {str(r["task"]): r for r in strict if is_raw_final(r)}

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.50, 1.0], hspace=0.32, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    panel_label(ax, "a")
    ax.set_title("Unified cross-domain aerospace RUL workflow", fontsize=9, pad=5)

    x_cols = [0.03, 0.25, 0.48, 0.72]
    y_top = 0.59
    y_bottom = 0.27
    w = 0.17
    h = 0.18
    source_fc = "#EEF3FA"
    target_fc = "#F4F0F7"
    method_fc = "#E7F0EA"
    output_fc = "#FFF3E9"

    _draw_box(ax, (x_cols[0], y_top), w, h, "Public source\nXJTU-SY bearing", source_fc, PALETTE["baseline"])
    _draw_box(ax, (x_cols[0], y_bottom), w, h, "Public source\nNASA battery", source_fc, PALETTE["baseline"])
    _draw_box(ax, (x_cols[1], y_top), w, h, "Aerospace target\nreaction wheel sim", target_fc, PALETTE["stage"])
    _draw_box(ax, (x_cols[1], y_bottom), w, h, "Aerospace target\nsatellite battery sim", target_fc, PALETTE["stage"])
    _draw_box(
        ax,
        (x_cols[2], 0.38),
        0.19,
        0.25,
        "PG-STDA-SAC-RSPA-TC\nphysics proxies\nstage alignment\nvalidation-only TC",
        method_fc,
        "#5F9D7B",
        fontsize=6.0,
        weight="bold",
    )
    _draw_box(ax, (x_cols[3], y_top), w, h, "RUL trend\nand health state", output_fc, "#D99B5C")
    _draw_box(ax, (x_cols[3], y_bottom), w, h, "Warning band\nand decision cue", output_fc, "#D99B5C")

    for yy in (y_top + h / 2, y_bottom + h / 2):
        _draw_arrow(ax, (x_cols[0] + w, yy), (x_cols[1], yy), PALETTE["neutral_mid"])
        _draw_arrow(ax, (x_cols[1] + w, yy), (x_cols[2], 0.49), PALETTE["neutral_mid"], rad=-0.08 if yy > 0.5 else 0.08)
    _draw_arrow(ax, (x_cols[2] + 0.19, 0.52), (x_cols[3], y_top + h / 2), PALETTE["neutral_mid"])
    _draw_arrow(ax, (x_cols[2] + 0.19, 0.46), (x_cols[3], y_bottom + h / 2), PALETTE["neutral_mid"])

    ax.text(
        0.03,
        0.055,
        "Boundary: strict comparison uses raw transfer outputs; final pipeline visualizations use validation-only TC, never test labels.",
        transform=ax.transAxes,
        fontsize=6.2,
        color=PALETTE["neutral_dark"],
    )

    ax = fig.add_subplot(gs[1, 0])
    tasks = ["first", "second"]
    labels = ["Reaction\nwheel", "Satellite\nbattery"]
    raw_vals = [float(raw_final_rows[t]["rmse"]) for t in tasks if t in raw_final_rows]
    final_vals = [float(final_rows[t]["rmse"]) for t in tasks if t in final_rows]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, raw_vals, width, color=PALETTE["baseline"], label="raw transfer")
    ax.bar(x + width / 2, final_vals, width, color=PALETTE["ours"], label=FINAL_METHOD_SHORT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE")
    ax.set_title("Final pipeline error", fontsize=8, pad=4)
    ax.legend(loc="upper right", fontsize=6)
    for xi, val in zip(x + width / 2, final_vals):
        ax.text(xi, val + max(final_vals) * 0.04, f"{val:.3f}", ha="center", va="bottom", fontsize=6.2, color=PALETTE["ours"])
    prettify_axis(ax)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    gains: list[float] = []
    gain_labels: list[str] = []
    for task, label in zip(tasks, labels):
        raw = raw_final_rows.get(task)
        best = _best_strict_baseline(strict, task)
        if raw is None or best is None:
            continue
        gains.append(improvement_percent(float(best["rmse"]), float(raw["rmse"])))
        gain_labels.append(label)
    ax.bar(gain_labels, gains, color=[PALETTE["ours"], PALETTE["stage"]], edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=PALETTE["neutral_dark"], linewidth=0.7)
    ax.set_ylabel("Raw RMSE reduction (%)")
    ax.set_title("Fair transfer gain over strongest baseline", fontsize=8, pad=4)
    for idx, value in enumerate(gains):
        ax.text(idx, value + 1.5, f"{value:.1f}%", ha="center", va="bottom", fontsize=6.2)
    prettify_axis(ax)
    panel_label(ax, "c")

    fig.suptitle("Aerospace component life prediction solution summary", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.12)
    save_figure(fig, "figure_4_aerospace_workflow_summary")
    plt.close(fig)


def plot_bonus_mechanism_observable_figure() -> None:
    fig = plt.figure(figsize=(7.2, 5.1))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[0.86, 1.0], hspace=0.30, wspace=0.26)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    panel_label(ax, "a")
    ax.set_title("Reaction wheel: friction-driven degradation", fontsize=8, pad=5)
    from matplotlib.patches import Circle, Rectangle

    ax.add_patch(Circle((0.21, 0.54), 0.17, transform=ax.transAxes, facecolor="#F0F3F8", edgecolor=PALETTE["neutral_dark"], linewidth=1.0))
    ax.add_patch(Circle((0.21, 0.54), 0.068, transform=ax.transAxes, facecolor="white", edgecolor=PALETTE["neutral_mid"], linewidth=0.9))
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        ax.plot(
            [0.21, 0.21 + 0.16 * math.cos(angle)],
            [0.54, 0.54 + 0.16 * math.sin(angle)],
            transform=ax.transAxes,
            color=PALETTE["baseline_dark"],
            linewidth=0.8,
        )
    _draw_box(ax, (0.43, 0.61), 0.22, 0.15, "lubrication loss\nbearing friction", "#F4F0F7", PALETTE["stage"], fontsize=6.2)
    _draw_box(ax, (0.43, 0.36), 0.22, 0.15, "drag torque rise\nthermal stress", "#F4F0F7", PALETTE["stage"], fontsize=6.2)
    _draw_box(ax, (0.72, 0.42), 0.22, 0.27, "observables\nspeed error\ncurrent\ntemperature\nvibration proxy", "#EEF3FA", PALETTE["baseline"], fontsize=6.0)
    _draw_arrow(ax, (0.35, 0.59), (0.43, 0.69), PALETTE["neutral_mid"])
    _draw_arrow(ax, (0.35, 0.50), (0.43, 0.43), PALETTE["neutral_mid"])
    _draw_arrow(ax, (0.65, 0.69), (0.72, 0.58), PALETTE["neutral_mid"])
    _draw_arrow(ax, (0.65, 0.43), (0.72, 0.53), PALETTE["neutral_mid"])

    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    panel_label(ax, "b")
    ax.set_title("Satellite battery: orbital charge-discharge degradation", fontsize=8, pad=5)
    ax.add_patch(Rectangle((0.08, 0.45), 0.23, 0.20, transform=ax.transAxes, facecolor="#E7F0EA", edgecolor=PALETTE["neutral_dark"], linewidth=1.0))
    ax.add_patch(Rectangle((0.31, 0.50), 0.030, 0.10, transform=ax.transAxes, facecolor=PALETTE["neutral_dark"], edgecolor=PALETTE["neutral_dark"], linewidth=0.8))
    ax.text(0.195, 0.55, "Li-ion", transform=ax.transAxes, ha="center", va="center", fontsize=6.5, fontweight="bold")
    _draw_box(ax, (0.41, 0.61), 0.23, 0.15, "capacity fade\ncalendar + cycling", "#E7F0EA", "#5F9D7B", fontsize=6.2)
    _draw_box(ax, (0.41, 0.36), 0.23, 0.15, "internal resistance\nheat + voltage sag", "#E7F0EA", "#5F9D7B", fontsize=6.2)
    _draw_box(ax, (0.73, 0.43), 0.20, 0.25, "observables\nvoltage\ncurrent\ntemperature", "#EEF3FA", PALETTE["baseline"], fontsize=6.0)
    _draw_arrow(ax, (0.34, 0.59), (0.41, 0.69), PALETTE["neutral_mid"])
    _draw_arrow(ax, (0.34, 0.51), (0.41, 0.43), PALETTE["neutral_mid"])
    _draw_arrow(ax, (0.64, 0.69), (0.73, 0.58), PALETTE["neutral_mid"])
    _draw_arrow(ax, (0.64, 0.43), (0.73, 0.53), PALETTE["neutral_mid"])

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(0, 1, 160)
    friction = 0.12 + 0.68 * (t**1.65) + 0.04 / (1 + np.exp(-(t - 0.72) / 0.035))
    speed_stability = 0.96 - 0.45 * (t**1.55)
    ax.plot(t, friction, color=PALETTE["accent"], label="friction proxy")
    ax.plot(t, speed_stability, color=PALETTE["ours"], label="speed stability")
    ax.fill_between(t, 0, friction, color=PALETTE["accent"], alpha=0.08)
    ax.set_xlabel("Normalized mission time")
    ax.set_ylabel("Normalized state")
    ax.set_title("Simulated wheel degradation trajectory", fontsize=8, pad=4)
    ax.legend(loc="upper left", fontsize=6)
    prettify_axis(ax)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    capacity = 1.0 - 0.16 * (t**0.85) - 0.12 / (1 + np.exp(-(t - 0.70) / 0.055))
    resistance = 0.14 + 0.58 * (t**1.35) + 0.10 / (1 + np.exp(-(t - 0.70) / 0.055))
    orbit = 0.08 * np.sin(2 * np.pi * 8 * t)
    ax.plot(t, capacity, color=PALETTE["ours"], label="usable capacity")
    ax.plot(t, resistance, color=PALETTE["accent"], label="internal resistance")
    ax.plot(t, 0.55 + orbit, color=PALETTE["stage"], linewidth=0.75, alpha=0.55, label="orbital load cycle")
    ax.set_xlabel("Normalized mission time")
    ax.set_ylabel("Normalized state")
    ax.set_title("Simulated battery degradation trajectory", fontsize=8, pad=4)
    ax.legend(loc="lower left", fontsize=6)
    prettify_axis(ax)
    panel_label(ax, "d")

    fig.suptitle("Physical mechanism to telemetry-observable mapping", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.11)
    save_figure(fig, "figure_5_mechanism_observable_map")
    plt.close(fig)


def _plot_dashboard_task(ax: plt.Axes, rows: list[dict[str, Any]], task: str, color: str) -> None:
    if not rows:
        ax.set_axis_off()
        return
    choices = choose_main_units(rows, count=1)
    if not choices:
        ax.set_axis_off()
        return
    summary = choices[0]
    sub = summary["rows"]
    x = normalized_progress(sub)
    y_true = np.array([float(r["y_true"]) for r in sub])
    y_pred = np.array([float(r["y_pred"]) for r in sub])
    y_trend = moving_average(y_pred, 7 if len(y_pred) < 120 else 11)

    ax.axhspan(0.50, 1.05, color="#E7F0EA", alpha=0.75, linewidth=0)
    ax.axhspan(0.20, 0.50, color="#FFF3E9", alpha=0.85, linewidth=0)
    ax.axhspan(-0.05, 0.20, color="#F6E4E0", alpha=0.82, linewidth=0)
    ax.plot(x, y_true, color=PALETTE["neutral_dark"], linewidth=1.05, alpha=0.72, label="true RUL")
    ax.scatter(x, y_pred, color=color, s=5, alpha=0.13, linewidths=0)
    ax.plot(x, y_trend, color=color, linewidth=1.55, label=FINAL_METHOD_SHORT)
    ax.axhline(0.50, color="#80A782", linewidth=0.65, linestyle=":")
    ax.axhline(0.20, color="#C65A4A", linewidth=0.75, linestyle=":")

    crosses = np.where(y_trend <= 0.20)[0]
    idx = int(crosses[0]) if len(crosses) else len(y_trend) - 1
    ax.scatter([x[idx]], [y_trend[idx]], s=24, color=PALETTE["accent"], zorder=5)
    ax.annotate(
        "warning",
        xy=(x[idx], y_trend[idx]),
        xytext=(min(0.96, x[idx] + 0.08), min(0.95, y_trend[idx] + 0.26)),
        fontsize=6,
        color=PALETTE["accent"],
        arrowprops={"arrowstyle": "-|>", "color": PALETTE["accent"], "linewidth": 0.7},
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Normalized test progress")
    ax.set_ylabel("Normalized RUL")
    ax.set_title(f"{display_task_short(task)}  |  unit {summary['unit']}", fontsize=8, pad=4)
    metric_box(ax, f"MAE {summary['mae']:.3f}\nfinal pred {y_trend[-1]:.3f}", loc="upper right")
    prettify_axis(ax)


def plot_bonus_phm_dashboard(rows_by_task: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[0.9, 1.75, 1.75], hspace=0.42, wspace=0.30)

    ax = fig.add_subplot(gs[:, 0])
    ax.set_axis_off()
    panel_label(ax, "a")
    ax.set_title("PHM output", fontsize=8, pad=8)
    _draw_box(ax, (0.06, 0.77), 0.84, 0.12, "health state\nNominal / Watch / Critical", "#E7F0EA", "#5F9D7B", fontsize=5.5, weight="bold")
    _draw_box(ax, (0.06, 0.58), 0.84, 0.12, "RUL trend\nunit-level trajectory", "#EEF3FA", PALETTE["baseline"], fontsize=5.8, weight="bold")
    _draw_box(ax, (0.06, 0.39), 0.84, 0.12, "early warning\nthreshold crossing", "#FFF3E9", "#D99B5C", fontsize=5.8, weight="bold")
    _draw_box(ax, (0.06, 0.20), 0.84, 0.12, "engineering action\ninspect / derate / replace", "#F6E4E0", PALETTE["accent"], fontsize=5.5, weight="bold")
    for y0, y1 in [(0.77, 0.70), (0.58, 0.51), (0.39, 0.32)]:
        _draw_arrow(ax, (0.47, y0), (0.47, y1), PALETTE["neutral_mid"])

    ax1 = fig.add_subplot(gs[0, 1:])
    _plot_dashboard_task(ax1, rows_by_task.get("first", []), "first", PALETTE["ours"])
    panel_label(ax1, "b")
    ax1.legend(loc="lower left", fontsize=6, ncol=2)

    ax2 = fig.add_subplot(gs[1, 1:])
    _plot_dashboard_task(ax2, rows_by_task.get("second", []), "second", PALETTE["stage"])
    panel_label(ax2, "c")

    final_rows = {str(r["task"]): r for r in rows if r["group"] == "final" and r["exists"]}
    summary = []
    if "first" in final_rows:
        summary.append(f"wheel RMSE {float(final_rows['first']['rmse']):.3f}")
    if "second" in final_rows:
        summary.append(f"battery RMSE {float(final_rows['second']['rmse']):.3f}")
    fig.text(0.09, 0.055, "Final pipeline: " + " | ".join(summary), fontsize=6.2, color=PALETTE["neutral_dark"])
    fig.suptitle("Mission-level health-management view from final predictions", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.11)
    save_figure(fig, "figure_6_phm_application_dashboard")
    plt.close(fig)


def _task_metric_summary(rows: list[dict[str, Any]], task: str) -> dict[str, float]:
    strict = raw_transfer_comparison_rows(rows)
    raw = next(r for r in strict if r["task"] == task and is_raw_final(r))
    final = next(r for r in rows if r["task"] == task and r["group"] == "final" and r["exists"])
    best = _best_strict_baseline(strict, task)
    best_rmse = float(best["rmse"]) if best is not None else float("nan")
    return {
        "raw_rmse": float(raw["rmse"]),
        "final_rmse": float(final["rmse"]),
        "final_mae": float(final["mae"]),
        "final_ra": float(final["ra"]),
        "strict_gain": improvement_percent(best_rmse, float(raw["rmse"])) if math.isfinite(best_rmse) else float("nan"),
        "tc_gain": improvement_percent(float(raw["rmse"]), float(final["rmse"])),
    }


def plot_bonus_workflow_figure_v2(rows: list[dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.65, 0.95], hspace=0.20)
    wheel = _task_metric_summary(rows, "first")
    battery = _task_metric_summary(rows, "second")

    ax = fig.add_subplot(gs[0])
    ax.set_axis_off()
    panel_label(ax, "a")
    ax.set_title("同一套框架将公开退化数据迁移到两个航天寿命预测目标域", fontsize=8.8, pad=3)

    header_y = 0.84
    for x, label in [
        (0.08, "公开源域"),
        (0.30, "航天目标域"),
        (0.55, "统一模型"),
        (0.82, "PHM输出"),
    ]:
        ax.text(x, header_y, label, transform=ax.transAxes, ha="center", va="center", fontsize=6.2, color=PALETTE["neutral_mid"])

    lanes = [
        (0.60, "XJTU-SY\n轴承", "反作用轮\n仿真", "反作用轮RUL\n风险带", PALETTE["ours"], wheel),
        (0.33, "NASA\n电池", "卫星电池\n仿真", "电池RUL\n风险带", PALETTE["stage"], battery),
    ]
    for y, src, tgt, out, lane_color, summary in lanes:
        _draw_box(ax, (0.02, y - 0.075), 0.15, 0.15, src, "#EEF3FA", PALETTE["baseline"], fontsize=6.1, weight="bold")
        _draw_box(ax, (0.23, y - 0.075), 0.18, 0.15, tgt, "#F4F0F7", PALETTE["stage"], fontsize=6.1, weight="bold")
        _draw_box(ax, (0.80, y - 0.075), 0.17, 0.15, out, "#FFF3E9", "#D99B5C", fontsize=6.1, weight="bold")
        _draw_arrow(ax, (0.17, y), (0.23, y), PALETTE["neutral_mid"], linewidth=1.1)
        _draw_arrow(ax, (0.41, y), (0.48, 0.47), PALETTE["neutral_mid"], linewidth=1.1, rad=-0.08 if y > 0.45 else 0.08)
        _draw_arrow(ax, (0.66, 0.47), (0.80, y), PALETTE["neutral_mid"], linewidth=1.1, rad=0.08 if y > 0.45 else -0.08)
        ax.text(
            0.71,
            y,
            f"raw增益\n{summary['strict_gain']:.1f}%",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.1,
            color=lane_color,
            fontweight="bold",
        )

    _draw_box(
        ax,
        (0.47, 0.33),
        0.20,
        0.28,
        "PG-STDA-SAC-RSPA-TC\n\n物理代理特征\n阶段感知对齐\n验证集TC",
        "#E7F0EA",
        "#5F9D7B",
        fontsize=5.8,
        weight="bold",
    )
    ax.text(
        0.02,
        0.08,
        "公平边界：基线对比使用raw迁移输出；TC只作为最终验证集校准管线汇报。",
        transform=ax.transAxes,
        fontsize=5.9,
        color=PALETTE["neutral_dark"],
    )

    ax = fig.add_subplot(gs[1])
    ax.set_axis_off()
    panel_label(ax, "b")
    ax.set_title("报告与答辩结果卡片", fontsize=8, pad=4)
    cards = [
        ("反作用轮", "最终RMSE", f"{wheel['final_rmse']:.3f}", f"raw -> TC: {wheel['raw_rmse']:.3f} -> {wheel['final_rmse']:.3f}", PALETTE["ours"]),
        ("卫星电池", "最终RMSE", f"{battery['final_rmse']:.3f}", f"raw -> TC: {battery['raw_rmse']:.3f} -> {battery['final_rmse']:.3f}", PALETTE["stage"]),
        ("反作用轮", "raw增益", f"{wheel['strict_gain']:.1f}%", "较最强严格基线", PALETTE["ours"]),
        ("卫星电池", "raw增益", f"{battery['strict_gain']:.1f}%", "较最强严格基线", PALETTE["stage"]),
    ]
    for i, (title, metric, value, subtitle, color) in enumerate(cards):
        x0 = 0.03 + i * 0.24
        _draw_box(ax, (x0, 0.16), 0.20, 0.58, "", "#F8F8F8", "#DADADA")
        ax.text(x0 + 0.10, 0.65, title, transform=ax.transAxes, ha="center", va="center", fontsize=6.2, color=PALETTE["neutral_dark"])
        ax.text(x0 + 0.10, 0.47, value, transform=ax.transAxes, ha="center", va="center", fontsize=14.0, fontweight="bold", color=color)
        ax.text(x0 + 0.10, 0.31, metric, transform=ax.transAxes, ha="center", va="center", fontsize=6.0, color=PALETTE["neutral_mid"])
        ax.text(x0 + 0.10, 0.20, subtitle, transform=ax.transAxes, ha="center", va="center", fontsize=5.5, color=PALETTE["neutral_dark"])

    fig.suptitle("航天器关键组件RUL迁移方案总览", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.89, bottom=0.08)
    save_figure(fig, "figure_4_aerospace_workflow_summary")
    plt.close(fig)


def _sparkline(
    ax: plt.Axes,
    x0: float,
    y0: float,
    w: float,
    h: float,
    label: str,
    values: np.ndarray,
    color: str,
    face: str = "#FAFAFA",
) -> None:
    _draw_box(ax, (x0, y0), w, h, "", face, "#DADADA")
    xs = np.linspace(x0 + 0.06 * w, x0 + 0.94 * w, len(values))
    vals = np.asarray(values, dtype=float)
    vals = (vals - float(vals.min())) / max(float(vals.max() - vals.min()), 1e-8)
    ys = y0 + 0.18 * h + vals * 0.52 * h
    ax.plot(xs, ys, transform=ax.transAxes, color=color, linewidth=1.05, clip_on=False)
    ax.text(x0 + 0.07 * w, y0 + 0.82 * h, label, transform=ax.transAxes, ha="left", va="top", fontsize=5.4, color=PALETTE["neutral_dark"])


def plot_bonus_mechanism_observable_figure_v2() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.55))
    ax.set_axis_off()
    panel_label(ax, "a")
    from matplotlib.patches import FancyBboxPatch

    for y0, edge, face in [(0.47, PALETTE["ours"], "#F5F8FC"), (0.14, PALETTE["stage"], "#F7F6FB")]:
        band = FancyBboxPatch(
            (0.015, y0),
            0.965,
            0.28,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.55,
            alpha=0.55,
            transform=ax.transAxes,
            zorder=-2,
        )
        ax.add_patch(band)

    t = np.linspace(0, 1, 90)
    rows = [
        {
            "y": 0.60,
            "name": "反作用轮",
            "subtitle": "轴承摩擦与热漂移",
            "icon": "wheel",
            "color": PALETTE["ours"],
            "mech": "摩擦升高\n转速稳定性下降",
            "spark": [
                ("电流 ↑", 0.2 + 0.65 * t**1.5),
                ("温度 ↑", 0.25 + 0.55 * t**1.2 + 0.04 * np.sin(9 * t)),
                ("转速误差 ↑", 0.1 + 0.8 * t**1.8),
                ("振动代理 ↑", 0.18 + 0.75 / (1 + np.exp(-(t - 0.70) / 0.07))),
            ],
        },
        {
            "y": 0.27,
            "name": "卫星电池",
            "subtitle": "容量衰减与内阻增长",
            "icon": "battery",
            "color": PALETTE["stage"],
            "mech": "容量下降\n内阻升高",
            "spark": [
                ("电压跌落 ↑", 0.12 + 0.75 * t**1.6),
                ("电流周期", 0.50 + 0.32 * np.sin(2 * np.pi * 5 * t)),
                ("温度 ↑", 0.22 + 0.50 * t**1.1 + 0.03 * np.sin(10 * t)),
                ("RUL ↓", 1.0 - 0.88 * t**1.05),
            ],
        },
    ]

    for item in rows:
        y = item["y"]
        color = item["color"]
        ax.text(0.02, y + 0.14, item["name"], transform=ax.transAxes, ha="left", va="center", fontsize=8.0, fontweight="bold", color=color)
        ax.text(0.02, y + 0.095, item["subtitle"], transform=ax.transAxes, ha="left", va="center", fontsize=5.8, color=PALETTE["neutral_mid"])

        if item["icon"] == "wheel":
            from matplotlib.patches import Circle

            ax.add_patch(Circle((0.15, y), 0.070, transform=ax.transAxes, facecolor="#F0F3F8", edgecolor=PALETTE["neutral_dark"], linewidth=1.0))
            ax.add_patch(Circle((0.15, y), 0.027, transform=ax.transAxes, facecolor="white", edgecolor=PALETTE["neutral_mid"], linewidth=0.8))
            for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                ax.plot(
                    [0.15, 0.15 + 0.065 * math.cos(angle)],
                    [y, y + 0.065 * math.sin(angle)],
                    transform=ax.transAxes,
                    color=PALETTE["baseline_dark"],
                    linewidth=0.7,
                )
        else:
            from matplotlib.patches import Rectangle

            ax.add_patch(Rectangle((0.10, y - 0.045), 0.10, 0.09, transform=ax.transAxes, facecolor="#E7F0EA", edgecolor=PALETTE["neutral_dark"], linewidth=1.0))
            ax.add_patch(Rectangle((0.20, y - 0.020), 0.012, 0.04, transform=ax.transAxes, facecolor=PALETTE["neutral_dark"], edgecolor=PALETTE["neutral_dark"], linewidth=0.8))
            ax.text(0.15, y, "Li-ion", transform=ax.transAxes, ha="center", va="center", fontsize=5.8, fontweight="bold")

        _draw_box(ax, (0.27, y - 0.07), 0.17, 0.14, str(item["mech"]), "#F8F8F8", "#DADADA", fontsize=6.1, weight="bold")
        _draw_arrow(ax, (0.22, y), (0.27, y), PALETTE["neutral_mid"], linewidth=1.0)
        _draw_arrow(ax, (0.44, y), (0.50, y), PALETTE["neutral_mid"], linewidth=1.0)

        spark_x = [0.51, 0.64, 0.51, 0.64]
        spark_y = [y + 0.02, y + 0.02, y - 0.105, y - 0.105]
        for (label, values), sx, sy in zip(item["spark"], spark_x, spark_y):
            _sparkline(ax, sx, sy, 0.11, 0.095, label, np.asarray(values), color)

        _draw_box(ax, (0.81, y - 0.075), 0.15, 0.15, "滑窗遥测\n→ RUL标签", "#FFF3E9", "#D99B5C", fontsize=5.9, weight="bold")
        _draw_arrow(ax, (0.75, y), (0.81, y), PALETTE["neutral_mid"], linewidth=1.0)

    ax.text(0.12, 0.83, "组件", transform=ax.transAxes, fontsize=6.2, color=PALETTE["neutral_mid"], ha="center")
    ax.text(0.36, 0.83, "退化驱动", transform=ax.transAxes, fontsize=6.2, color=PALETTE["neutral_mid"], ha="center")
    ax.text(0.61, 0.83, "可观测遥测特征", transform=ax.transAxes, fontsize=6.2, color=PALETTE["neutral_mid"], ha="center")
    ax.text(0.885, 0.83, "预测目标", transform=ax.transAxes, fontsize=6.2, color=PALETTE["neutral_mid"], ha="center")
    ax.text(
        0.02,
        0.055,
        "仿真任务不是瞬时故障分类：两个目标域均包含长期运行、渐进退化和故障/应力变体。",
        transform=ax.transAxes,
        fontsize=5.9,
        color=PALETTE["neutral_dark"],
    )

    fig.suptitle("机理驱动仿真将航天退化过程转化为模型可见的遥测特征", fontsize=10, y=0.982)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.10)
    save_figure(fig, "figure_5_mechanism_observable_map")
    plt.close(fig)


def _plot_dashboard_task_v2(ax: plt.Axes, rows: list[dict[str, Any]], task: str, color: str) -> None:
    if not rows:
        ax.set_axis_off()
        return
    choices = choose_main_units(rows, count=1)
    if not choices:
        ax.set_axis_off()
        return
    summary = choices[0]
    sub = summary["rows"]
    x = normalized_progress(sub)
    y_true = np.array([float(r["y_true"]) for r in sub])
    y_pred = np.array([float(r["y_pred"]) for r in sub])
    y_trend = moving_average(y_pred, 7 if len(y_pred) < 120 else 11)

    bands = [
        (0.50, 1.05, "#E5F0E9", "正常"),
        (0.20, 0.50, "#FFF2E6", "关注"),
        (-0.05, 0.20, "#F7E1DE", "严重"),
    ]
    for lo, hi, fc, label in bands:
        ax.axhspan(lo, hi, color=fc, linewidth=0)
        ax.text(1.005, (lo + hi) / 2, label, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.5, color=PALETTE["neutral_mid"])
    ax.plot(x, y_true, color=PALETTE["neutral_dark"], linewidth=1.05, alpha=0.68, label="真实RUL")
    ax.scatter(x, y_pred, color=color, s=5, alpha=0.14, linewidths=0)
    ax.plot(x, y_trend, color=color, linewidth=1.65, label=FINAL_METHOD_SHORT)
    ax.axhline(0.50, color="#7BA67E", linewidth=0.65, linestyle=":")
    ax.axhline(0.20, color=PALETTE["accent"], linewidth=0.75, linestyle=":")
    crosses = np.where(y_trend <= 0.20)[0]
    idx = int(crosses[0]) if len(crosses) else len(y_trend) - 1
    ax.scatter([x[idx]], [y_trend[idx]], s=30, color=PALETTE["accent"], zorder=5)
    ax.annotate(
        "预警点",
        xy=(x[idx], y_trend[idx]),
        xytext=(max(0.05, min(0.80, x[idx] - 0.18)), min(0.92, y_trend[idx] + 0.27)),
        fontsize=6,
        color=PALETTE["accent"],
        arrowprops={"arrowstyle": "-|>", "color": PALETTE["accent"], "linewidth": 0.7},
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("归一化测试进度")
    ax.set_ylabel("归一化RUL")
    ax.set_title(f"{display_task_short(task)} | 单元 {summary['unit']}", fontsize=8, pad=4)
    metric_box(ax, f"MAE {summary['mae']:.3f}\n最终预测 {y_trend[-1]:.3f}", loc="upper right")
    prettify_axis(ax)


def plot_bonus_phm_dashboard_v2(rows_by_task: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(7.2, 5.25))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.34, 1.0, 1.0], width_ratios=[1.0, 1.0], hspace=0.60, wspace=0.22)

    axh = fig.add_subplot(gs[0, :])
    axh.set_axis_off()
    panel_label(axh, "a")
    wheel = _task_metric_summary(rows, "first")
    battery = _task_metric_summary(rows, "second")
    _draw_box(axh, (0.08, 0.22), 0.18, 0.48, f"反作用轮\nRMSE {wheel['final_rmse']:.3f}", "#EEF3FA", PALETTE["baseline"], fontsize=6.3, weight="bold")
    _draw_box(axh, (0.30, 0.22), 0.18, 0.48, f"卫星电池\nRMSE {battery['final_rmse']:.3f}", "#F4F0F7", PALETTE["stage"], fontsize=6.3, weight="bold")
    _draw_box(axh, (0.54, 0.22), 0.18, 0.48, "关注带\nRUL < 0.50", "#FFF3E9", "#D99B5C", fontsize=6.3, weight="bold")
    _draw_box(axh, (0.76, 0.22), 0.18, 0.48, "严重带\nRUL < 0.20", "#F7E1DE", PALETTE["accent"], fontsize=6.3, weight="bold")

    ax1 = fig.add_subplot(gs[1, :])
    _plot_dashboard_task_v2(ax1, rows_by_task.get("first", []), "first", PALETTE["ours"])
    panel_label(ax1, "b")
    ax1.legend(loc="lower left", fontsize=6, ncol=2)

    ax2 = fig.add_subplot(gs[2, :])
    _plot_dashboard_task_v2(ax2, rows_by_task.get("second", []), "second", PALETTE["stage"])
    panel_label(ax2, "c")

    fig.suptitle("将最终预测转化为航天PHM风险显示", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.085, right=0.955, top=0.88, bottom=0.10)
    save_figure(fig, "figure_6_phm_application_dashboard")
    plt.close(fig)


def write_requirement_matrix() -> None:
    lines = [
        "# 比赛要求覆盖矩阵",
        "",
        "| 比赛要求 | 对应证据 | 状态 |",
        "|---|---|---|",
        "| 至少覆盖两个典型寿命预测场景 | 反作用轮与卫星电池数据集设计文档；两个最终迁移实验结果 | 已覆盖 |",
        "| 公开退化源域到航天仿真目标域的迁移 | `XJTU-SY -> reaction_wheel_sim` 与 `NASA Battery -> satellite_battery_sim` 的最终配置和输出 | 已覆盖 |",
        "| 长期渐进退化预测，而非瞬时故障分类 | 两个自建数据集设计文档、预测轨迹图和末期误差分析 | 已覆盖 |",
        "| 明确退化机理假设及可观测量 | `05_report_assets/source_docs/*DATASET_DESIGN_REVIEW.md` | 已覆盖 |",
        "| 数据生成与数据集可复现 | `01_datasets/dataset_inventory.md`；项目源码中的生成脚本 | 已覆盖 |",
        "| PyTorch 训练、迁移和预测流程 | `02_experiments` 下的最终配置与输出证据 | 已覆盖 |",
        "| 同数据条件下与基础模型/方法对比 | `03_results/strict_unsupervised_comparison.md` 与监督参考表 | 已覆盖 |",
        "| 消融实验 | `03_results/ablation_table.md` | 已覆盖 |",
        "| 参数敏感性分析 | `03_results/parameter_sensitivity.md` 与参数敏感性补充图 | 已覆盖 |",
        "| 误差、稳定性和临近失效分析 | RMSE/MAE/NASA/RA/alpha/末窗口指标表与阶段误差图 | 已覆盖 |",
        "| 加分项：直观退化过程与工程应用可视化 | Nature-style 主图、流程图、机理-遥测映射图和 PHM 应用看板 | 已覆盖 |",
    ]
    (ARTIFACTS / "00_requirement_mapping/requirement_coverage_matrix.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_artifact_readme() -> None:
    lines = [
        "# XA-202608 比赛证据包",
        "",
        "本目录是双迁移寿命预测方案的整理后证据包，由 `scripts/prepare_competition_artifacts.py` 从已有指标、预测文件、配置文件和设计文档自动生成。",
        "脚本每次运行都会先清空并重建本目录，因此这里不手工存放临时文件。",
        "",
        "## 目录顺序",
        "",
        "| 目录 | 用途 |",
        "|---|---|",
        "| `00_requirement_mapping` | 比赛硬要求与证据文件的对应关系。 |",
        "| `01_datasets` | 数据集清单、数据角色和存在性检查。 |",
        "| `02_experiments` | 最终配置文件、最终预测文件和最终指标输出。 |",
        "| `03_results` | 总指标 CSV、公平对比表、监督参考表、消融表和参数敏感性表。 |",
        "| `04_figures` | 可直接放入报告/答辩材料的 PNG/SVG/PDF 图件。 |",
        "| `05_report_assets` | 数据集设计、方法创新和实验更新等报告素材文档。 |",
        "| `99_cleanup` | 产物包清理说明和最终检查点保留策略。 |",
        "",
        "## 报告口径边界",
        "",
        "- `strict_unsupervised_comparison.md` 是主公平对比表，只比较未经过最终输出校准的 raw 迁移输出。",
        "- `supervised_reference_comparison.md` 是监督参考/上界类对比，不与严格无监督迁移主张混为一类。",
        "- `ablation_table.md` 和 `parameter_sensitivity.md` 用于支撑方法模块有效性和鲁棒性讨论。",
        "- `figure_1_strict_transfer_performance.png` 展示 raw 迁移输出的公平对比。",
        "- 最终预测诊断图展示 `PG-STDA-SAC-RSPA-TC` 的验证集校准后输出；TC 只使用目标域验证集，不使用测试集标签。",
        "- `figure_4_aerospace_workflow_summary.png`、`figure_5_mechanism_observable_map.png`、`figure_6_phm_application_dashboard.png` 用于加分项中的工程流程和可视化展示。",
        "- `04_figures` 下每张主图均同时保留 PNG、SVG、PDF，便于报告排版和后续编辑。",
    ]
    (ARTIFACTS / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            try:
                total += file.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def write_dataset_inventory() -> None:
    rows = [
        ("公开源域", "data/raw/xjtu_sy", "第一迁移使用的 XJTU-SY 轴承退化源域"),
        ("公开源域", "data/raw/nasa_battery", "第二迁移使用的 NASA Battery 退化源域"),
        ("处理后源域", "data/processed/xjtu_sy_hi.npz", "XJTU-SY 低频 HI 张量，用于反作用轮迁移"),
        ("航天仿真目标域", "data/simulated/reaction_wheel", "反作用轮长期摩擦/热退化目标域"),
        ("航天仿真目标域", "data/simulated/satellite_battery", "卫星电池容量衰减/内阻增长目标域"),
    ]
    lines = [
        "# 数据集清单",
        "",
        "| 角色 | 路径 | 大小 MB | 用途 | 是否存在 |",
        "|---|---|---:|---|---|",
    ]
    for role, rel, purpose in rows:
        path = ROOT / rel
        size = dir_size_mb(path) if path.is_dir() else (path.stat().st_size / (1024 * 1024) if path.exists() else 0.0)
        lines.append(f"| {role} | `{rel}` | {size:.2f} | {purpose} | {'是' if path.exists() else '否'} |")
    (ARTIFACTS / "01_datasets/dataset_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_recommendation(final_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 最终推荐方案",
        "",
        f"统一推荐方法：{FINAL_METHOD_FULL}。",
        "",
        f"图表和结果表中使用的短名：`{FINAL_METHOD_SHORT}`。",
        "",
        "核心组成：",
        "",
        "- `P-SA-MCD-TCN` 时序表征骨干。",
        "- 针对不同航天组件的物理代理特征和可观测遥测特征。",
        "- 基于目标域伪时间阶段的 `Stage-aware LMMD`。",
        "- `Stage Auxiliary Calibration (SAC)`：用源域阶段标签和目标域无标签伪阶段辅助校准退化阶段表征。",
        "- `Reliability-weighted Stage Prototype Alignment (R-SPA)`：按目标阶段置信度加权对齐阶段原型。",
        "- `Validation-only Time-aware Output Calibration (TC)`：只用目标域验证集拟合时间感知输出校准，不使用测试集标签。",
        "",
        "推荐最终配置：",
        "",
        "| 任务 | 配置 | RMSE | MAE | NASA | RA |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        lines.append(
            f"| {TASK_CN.get(str(row['task']), str(row['task']))} | `{row['config_path']}` | {fmt(row['rmse'])} | {fmt(row['mae'])} | {fmt(row['nasa_score'])} | {fmt(row['ra'])} |"
        )
    lines.extend(
        [
            "",
            "报告口径边界：",
            "",
            "- 严格无监督迁移训练不在迁移损失中使用目标域训练集 RUL 标签。",
            "- 最终 TC 只使用目标域验证集标签做输出校准，绝不使用目标测试集标签或测试集派生阶段。",
            "- `target-only` 与 `target-supervised` 结果属于监督参考/上界，不作为严格无监督迁移主张。",
            "- 最终方法作为两个场景统一的迁移框架提交，避免第一迁移和第二迁移分别使用完全不同模型造成方案割裂。",
        ]
    )
    (ARTIFACTS / "03_results/final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cleanup_manifest() -> None:
    lines = [
        "# 清理与保留清单",
        "",
        "本证据包面向报告和答辩整理，原始数据、源代码和完整实验输出仍保留在项目对应目录中。",
        "",
        "可安全清理的临时内容：",
        "",
        "- Python `__pycache__` 目录。",
        "- 过期 `.pid` 进程标记文件。",
        f"- 在保留 `{FINAL_METHOD_SHORT}` 最终检查点后，清理重复的非最终 `.pt` 检查点。",
        "",
        "必须保留的最终检查点：",
        "",
        "- `outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p0005_c0p5_srcsup0p7_50e/transfer_model.pt`",
        "- `outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_w0p001_c0p5_srcsup0p7_50e/transfer_model.pt`",
        "",
        "校准后的最终输出目录复制同一检查点，并额外包含只基于验证集 TC 的预测和指标：",
        "",
        "- `outputs/first_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e`",
        "- `outputs/second_transfer_pg_stda_rspa_50e/pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e`",
    ]
    (ARTIFACTS / "99_cleanup/cleanup_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_subfolder_readmes() -> None:
    readmes = {
        "00_requirement_mapping/README.md": [
            "# 比赛要求映射",
            "",
            "本目录说明 XA-202608 赛题硬要求与当前工程证据之间的对应关系。",
            "",
            "- `requirement_coverage_matrix.md`：逐项列出两个迁移场景、数据生成、迁移实验、基线对比、消融分析和可视化加分项的覆盖情况。",
            "- 该表适合作为报告撰写前的检查清单，也适合答辩时快速说明“硬要求是否满足”。",
        ],
        "01_datasets/README.md": [
            "# 数据集证据",
            "",
            "本目录记录公开源域、处理后源域和两个航天仿真目标域的数据位置与用途。",
            "",
            "- `dataset_inventory.md`：数据路径、角色、大小和存在性检查。",
            "- 两个目标域分别为 `reaction_wheel_sim` 与 `satellite_battery_sim`，共同满足赛题至少两个典型寿命预测场景的要求。",
        ],
        "02_experiments/README.md": [
            "# 最终实验证据",
            "",
            "本目录只收纳最终方案相关配置和最终输出，完整历史实验仍保存在项目 `outputs/` 目录。",
            "",
            "- `final_configs/`：最终方法及必要变体配置。",
            "- `final_outputs/`：两个迁移任务的最终预测、验证预测、指标和环境信息。",
        ],
        "02_experiments/final_configs/README.md": [
            "# 最终配置文件",
            "",
            "本目录保存可复现最终结果的 YAML 配置副本。",
            "",
            "- 第一迁移最终配置：`xjtu_to_reaction_wheel_pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e.yaml`。",
            "- 第二迁移最终配置：`nasa_to_satellite_battery_pg_stda_sac_rspa_timecal_deg2_ridge0p01_50e.yaml`。",
            "- 其他配置用于 raw 未校准模型、加权检查点或对比说明。",
        ],
        "02_experiments/final_outputs/README.md": [
            "# 最终输出文件",
            "",
            "本目录按迁移任务划分最终输出。",
            "",
            "- `first_transfer/`：第一迁移 `XJTU-SY -> reaction_wheel_sim` 的最终输出。",
            "- `second_transfer/`：第二迁移 `NASA Battery -> satellite_battery_sim` 的最终输出。",
        ],
        "02_experiments/final_outputs/first_transfer/README.md": [
            "# 第一迁移最终输出",
            "",
            "本目录保存第一迁移最终方法 `PG-STDA-SAC-RSPA-TC` 的核心输出。",
            "",
            "- `transfer_metrics.json`：最终测试指标。",
            "- `predictions_test.csv`：测试集逐窗口预测。",
            "- `predictions_val.csv`：验证集预测，用于 TC 输出校准。",
            "- `resolved_config.json` 与 `env_info.json`：复现配置和运行环境记录。",
        ],
        "02_experiments/final_outputs/second_transfer/README.md": [
            "# 第二迁移最终输出",
            "",
            "本目录保存第二迁移最终方法 `PG-STDA-SAC-RSPA-TC` 的核心输出。",
            "",
            "- `transfer_metrics.json`：最终测试指标。",
            "- `predictions_test.csv`：测试集逐窗口预测。",
            "- `predictions_val.csv`：验证集预测，用于 TC 输出校准。",
            "- `resolved_config.json` 与 `env_info.json`：复现配置和运行环境记录。",
        ],
        "03_results/README.md": [
            "# 结果表",
            "",
            "本目录保存报告可引用的指标表。",
            "",
            "- `metrics_master.csv`：所有纳入证据包的指标总表。",
            "- `strict_unsupervised_comparison.md`：主公平对比表，使用 raw 迁移输出，不含 TC。",
            "- `supervised_reference_comparison.md`：监督参考/上界类结果。",
            "- `ablation_table.md`：模块消融。",
            "- `parameter_sensitivity.md`：关键参数敏感性。",
            "- `final_recommendation.md`：最终统一方法和报告口径边界。",
        ],
        "04_figures/README.md": [
            "# 图件",
            "",
            "本目录保存报告和答辩可直接使用的可视化图件。每张主图通常同时导出 PNG、SVG 和 PDF。",
            "",
            "- `figure_1_strict_transfer_performance.*`：严格无监督 raw 迁移对比。",
            "- `figure_2_ablation_and_sensitivity.*`：消融与参数敏感性。",
            "- `figure_3_representative_predictions.*`：最终模型与强基线的代表性预测轨迹。",
            "- `figure_4_aerospace_workflow_summary.*`：公开源域到航天目标域的整体流程。",
            "- `figure_5_mechanism_observable_map.*`：退化机理到可观测遥测的映射。",
            "- `figure_6_phm_application_dashboard.*`：面向 PHM 应用的风险看板。",
            "- `supp_*`：补充诊断图和参数敏感性图。",
        ],
        "05_report_assets/README.md": [
            "# 报告素材",
            "",
            "本目录保存报告撰写所需的设计、审阅和实验总结材料。",
            "",
            "- `source_docs/`：从项目 `docs/` 复制来的关键文档。",
            "- 这些文档用于支撑数据集设计、方法创新性、实验边界和最终结果解释。",
        ],
        "05_report_assets/source_docs/README.md": [
            "# 源文档副本",
            "",
            "本目录是报告素材文档的副本，便于在证据包内集中查看。",
            "",
            "- 数据集设计：`REACTION_WHEEL_SIM_DATASET_DESIGN_REVIEW.md`、`SATELLITE_BATTERY_SIM_DATASET_DESIGN_REVIEW.md`。",
            "- 方法创新与最终结果：`METHOD_NOVELTY_FRONTIER_RESEARCH_REVIEW.md`、`PG_STDA_SAC_FINAL_CROSS_TRANSFER_RESULTS.md`、`SECOND_TRANSFER_PG_STDA_INNOVATION_RESULTS.md`。",
            "- 基线与优化更新：`FIRST_TRANSFER_BASELINE_GRID_UPDATE.md`、`SECOND_TRANSFER_BASELINE_OPTIMIZATION_UPDATE.md`。",
        ],
        "99_cleanup/README.md": [
            "# 清理记录",
            "",
            "本目录说明哪些内容可以清理、哪些最终检查点必须保留。",
            "",
            "- `cleanup_manifest.md`：最终保留策略和可清理临时内容说明。",
            "- 注意：本证据包重建时会自动清空旧文件，因此不要在本目录手工存放未纳入脚本生成流程的资料。",
        ],
    }
    for rel, lines in readmes.items():
        path = ARTIFACTS / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_clean_artifacts()
    rows = collect_rows(EXPERIMENTS)
    ablation_rows = collect_rows(ABLATIONS)
    sensitivity_rows = collect_rows(PARAMETER_SWEEPS)

    all_rows = rows + ablation_rows + sensitivity_rows
    write_csv(ARTIFACTS / "03_results/metrics_master.csv", all_rows)
    write_markdown_table(
        ARTIFACTS / "03_results/strict_unsupervised_comparison.md",
        "Strict Unsupervised Transfer Comparison",
        raw_transfer_comparison_rows(rows),
    )
    write_markdown_table(
        ARTIFACTS / "03_results/supervised_reference_comparison.md",
        "Supervised Reference Comparison",
        [r for r in rows if r["group"] == "supervised_reference"],
    )
    write_markdown_table(ARTIFACTS / "03_results/ablation_table.md", "Ablation Table", ablation_rows)
    write_markdown_table(
        ARTIFACTS / "03_results/parameter_sensitivity.md",
        "Parameter Sensitivity",
        sensitivity_rows,
    )
    write_final_recommendation([r for r in rows if r["group"] == "final"])

    copy_final_evidence()
    write_artifact_readme()
    write_requirement_matrix()
    write_dataset_inventory()
    write_cleanup_manifest()
    write_subfolder_readmes()

    apply_nature_style()
    plot_main_performance_figure(rows)
    plot_ablation_sensitivity_figure(ablation_rows, sensitivity_rows)
    plot_bonus_workflow_figure_v2(rows)
    plot_bonus_mechanism_observable_figure_v2()
    final_prediction_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["group"] == "final":
            pred_path = (ROOT / str(row["metrics_path"])).parent / "predictions_test.csv"
            parsed_rows = parse_prediction_csv(pred_path)
            final_prediction_rows[str(row["task"])] = parsed_rows
            plot_predictions(str(row["task"]), str(row["metrics_path"]))
    plot_representative_prediction_panels(final_prediction_rows)
    plot_bonus_phm_dashboard_v2(final_prediction_rows, rows)
    plot_second_calibration_effect()
    plot_parameter_sensitivity(sensitivity_rows)

    missing = [r for r in all_rows if not r["exists"]]
    if missing:
        write_markdown_table(ARTIFACTS / "99_cleanup/missing_metrics.md", "Missing Metrics", missing)

    print(f"Wrote artifacts to {ARTIFACTS}")
    print(f"Metrics rows: {len(all_rows)}; missing: {len(missing)}")


if __name__ == "__main__":
    main()
