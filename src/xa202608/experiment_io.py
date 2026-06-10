from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


def collect_environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            info["git_commit"] = result.stdout.strip()
    except FileNotFoundError:
        info["git_commit"] = "git_not_available"
    return info


def save_prediction_csv(prediction: dict[str, np.ndarray], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "unit_id": prediction["unit_id"],
        "time_index": prediction["time_index"],
        "stage": prediction["stage"],
        "y_true": prediction["y_true"],
        "y_pred": prediction["y_pred"],
    }
    if "lower" in prediction and "upper" in prediction:
        data["lower"] = prediction["lower"]
        data["upper"] = prediction["upper"]
    pd.DataFrame(data).sort_values(["unit_id", "time_index"]).to_csv(output_path, index=False, encoding="utf-8")


@torch.no_grad()
def benchmark_inference_latency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int = 2,
    max_batches: int = 20,
) -> dict[str, float]:
    model.eval()
    batch_times = []
    num_samples = 0
    total_batches = len(loader)
    effective_warmup = min(warmup_batches, max(total_batches - 1, 0))
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= effective_warmup + max_batches:
            break
        x = batch["x"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if batch_idx >= effective_warmup:
            batch_times.append(elapsed)
            num_samples += int(x.shape[0])
    if not batch_times or num_samples == 0:
        return {"latency_ms_per_batch": float("nan"), "latency_ms_per_sample": float("nan")}
    total_time = float(np.sum(batch_times))
    return {
        "latency_ms_per_batch": float(np.mean(batch_times) * 1000.0),
        "latency_ms_per_sample": float(total_time / num_samples * 1000.0),
    }
