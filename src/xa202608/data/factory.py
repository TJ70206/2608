from __future__ import annotations

from torch.utils.data import DataLoader

from xa202608.data.battery import build_battery_loaders
from xa202608.data.cmapss import build_cmapss_loaders
from xa202608.data.reaction_wheel import build_reaction_wheel_loaders
from xa202608.data.satellite_battery import build_satellite_battery_loaders
from xa202608.data.synthetic import build_synthetic_debug_loaders
from xa202608.data.xjtu_reaction_wheel import build_reaction_wheel_hi_loaders


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    dataset = str(config["data"]["dataset"]).lower()
    if dataset == "synthetic_debug":
        return build_synthetic_debug_loaders(config)
    if dataset == "cmapss":
        return build_cmapss_loaders(config)
    if dataset == "xjtu_sy":
        raise ValueError(
            "dataset 'xjtu_sy' is a raw-source alias, not a target-only training dataset. "
            "Use train_transfer.py with dataset 'xjtu_sy_transfer' for source-domain transfer, "
            "or use dataset 'reaction_wheel_hi' for the first aerospace target domain."
        )
    if dataset == "nasa_battery":
        return build_battery_loaders(config)
    if dataset in {"reaction_wheel", "reaction_wheel_sim"}:
        return build_reaction_wheel_loaders(config)
    if dataset in {"reaction_wheel_hi", "reaction_wheel_sim_hi"}:
        return build_reaction_wheel_hi_loaders(config)
    if dataset in {"satellite_battery", "satellite_battery_sim"}:
        return build_satellite_battery_loaders(config)
    raise ValueError(f"unknown dataset: {config['data']['dataset']}")
