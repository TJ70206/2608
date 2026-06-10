from __future__ import annotations

import torch
from torch import nn


def gaussian_kernel_matrix(x: torch.Tensor, y: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    beta = 1.0 / (2.0 * sigmas.view(-1, 1, 1))
    dist = torch.cdist(x, y, p=2).pow(2).unsqueeze(0)
    return torch.exp(-beta * dist).sum(dim=0)


def _normalized_weights(weights: torch.Tensor | None, length: int, reference: torch.Tensor) -> torch.Tensor:
    if weights is None:
        return torch.full((length,), 1.0 / float(length), device=reference.device, dtype=reference.dtype)
    normalized = weights.to(device=reference.device, dtype=reference.dtype).reshape(-1)
    if normalized.numel() != int(length):
        raise ValueError(f"MMD weights length mismatch: {normalized.numel()} != {length}")
    normalized = normalized.clamp_min(0.0)
    weight_sum = normalized.sum()
    if float(weight_sum.detach().cpu()) <= 0.0:
        return torch.full((length,), 1.0 / float(length), device=reference.device, dtype=reference.dtype)
    return normalized / weight_sum


def mmd_loss(source: torch.Tensor, target: torch.Tensor, sigmas: torch.Tensor | None = None) -> torch.Tensor:
    if source.numel() == 0 or target.numel() == 0:
        return source.new_tensor(0.0)
    if sigmas is None:
        sigmas = source.new_tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    k_xx = gaussian_kernel_matrix(source, source, sigmas).mean()
    k_yy = gaussian_kernel_matrix(target, target, sigmas).mean()
    k_xy = gaussian_kernel_matrix(source, target, sigmas).mean()
    return k_xx + k_yy - 2.0 * k_xy


def weighted_mmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
    sigmas: torch.Tensor | None = None,
) -> torch.Tensor:
    if source.numel() == 0 or target.numel() == 0:
        return source.new_tensor(0.0)
    if sigmas is None:
        sigmas = source.new_tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    src_w = _normalized_weights(source_weights, source.shape[0], source)
    tgt_w = _normalized_weights(target_weights, target.shape[0], target)
    k_xx = gaussian_kernel_matrix(source, source, sigmas)
    k_yy = gaussian_kernel_matrix(target, target, sigmas)
    k_xy = gaussian_kernel_matrix(source, target, sigmas)
    xx = (k_xx * src_w[:, None] * src_w[None, :]).sum()
    yy = (k_yy * tgt_w[:, None] * tgt_w[None, :]).sum()
    xy = (k_xy * src_w[:, None] * tgt_w[None, :]).sum()
    return xx + yy - 2.0 * xy


def _stage_weight(stage_weights: torch.Tensor | None, stage: int, reference: torch.Tensor) -> torch.Tensor:
    if stage_weights is None:
        return reference.new_tensor(1.0)
    weights = stage_weights.to(device=reference.device, dtype=reference.dtype).reshape(-1)
    if int(stage) >= weights.numel():
        raise ValueError(f"stage_weights length must cover stage {stage}")
    return weights[int(stage)].clamp_min(0.0)


def stage_aware_lmmd_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_stages: torch.Tensor,
    target_stages: torch.Tensor,
    num_stages: int = 3,
    stage_weights: torch.Tensor | None = None,
    source_sample_weights: torch.Tensor | None = None,
    target_sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = []
    denominators = []
    for stage in range(num_stages):
        src_mask = source_stages == stage
        tgt_mask = target_stages == stage
        if src_mask.any() and tgt_mask.any():
            weight = _stage_weight(stage_weights, stage, source_features)
            if float(weight.detach().cpu()) <= 0.0:
                continue
            src_weights = source_sample_weights[src_mask] if source_sample_weights is not None else None
            tgt_weights = target_sample_weights[tgt_mask] if target_sample_weights is not None else None
            losses.append(
                weight
                * weighted_mmd_loss(
                    source_features[src_mask],
                    target_features[tgt_mask],
                    source_weights=src_weights,
                    target_weights=tgt_weights,
                )
            )
            denominators.append(weight)
    if not losses:
        return source_features.new_tensor(0.0)
    return torch.stack(losses).sum() / torch.stack(denominators).sum().clamp_min(1e-12)


def reliability_weighted_stage_prototype_alignment_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_stages: torch.Tensor,
    target_stages: torch.Tensor,
    target_stage_logits: torch.Tensor | None,
    num_stages: int = 3,
    min_confidence: float = 0.0,
) -> torch.Tensor:
    losses = []
    source_stages = source_stages.reshape(-1).long()
    target_stages = target_stages.reshape(-1).long()
    if target_stage_logits is None:
        target_confidence = target_features.new_ones((target_features.shape[0],))
    else:
        if target_stage_logits.shape[0] != target_features.shape[0]:
            raise ValueError("target_stage_logits batch dimension must match target_features")
        if target_stage_logits.shape[1] < int(num_stages):
            raise ValueError("target_stage_logits class dimension must cover num_stages")
        probs = torch.softmax(target_stage_logits[:, : int(num_stages)], dim=-1)
        stage_prob = probs.gather(1, target_stages.clamp(0, int(num_stages) - 1).unsqueeze(1)).squeeze(1)
        min_conf = float(min_confidence)
        if min_conf > 0.0:
            target_confidence = ((stage_prob - min_conf) / max(1.0 - min_conf, 1e-6)).clamp(0.0, 1.0)
        else:
            target_confidence = stage_prob.clamp(0.0, 1.0)
        target_confidence = target_confidence.detach()

    for stage in range(int(num_stages)):
        src_mask = source_stages == stage
        tgt_mask = target_stages == stage
        if not src_mask.any() or not tgt_mask.any():
            continue
        src_proto = source_features[src_mask].mean(dim=0)
        tgt_weights = target_confidence[tgt_mask].to(device=target_features.device, dtype=target_features.dtype)
        weight_sum = tgt_weights.sum()
        if float(weight_sum.detach().cpu()) <= 1e-12:
            continue
        tgt_stage_features = target_features[tgt_mask]
        tgt_proto = (tgt_stage_features * (tgt_weights / weight_sum).unsqueeze(1)).sum(dim=0)
        losses.append(tgt_weights.mean() * (src_proto - tgt_proto).pow(2).mean())

    if not losses:
        return source_features.new_tensor(0.0)
    return torch.stack(losses).mean()


def transfer_alignment_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_stages: torch.Tensor,
    target_stages: torch.Tensor,
    mode: str = "stage",
    num_stages: int = 3,
    stage_weights: torch.Tensor | None = None,
    source_sample_weights: torch.Tensor | None = None,
    target_sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    mode_key = str(mode).lower()
    if mode_key in {"none", "off", "source_only", "source-only"}:
        return source_features.new_tensor(0.0)
    if mode_key in {"global", "mmd"}:
        if source_sample_weights is not None or target_sample_weights is not None:
            return weighted_mmd_loss(source_features, target_features, source_sample_weights, target_sample_weights)
        return mmd_loss(source_features, target_features)
    if mode_key in {"stage", "stage_aware", "lmmd", "stage_aware_lmmd"}:
        return stage_aware_lmmd_loss(
            source_features,
            target_features,
            source_stages,
            target_stages,
            num_stages=int(num_stages),
            stage_weights=stage_weights,
            source_sample_weights=source_sample_weights,
            target_sample_weights=target_sample_weights,
        )
    raise ValueError(f"unknown transfer alignment mode: {mode}")


class RulRegressionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(prediction, target)
