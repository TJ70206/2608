from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


def make_regressor(input_dim: int, hidden_dims: list[int] | None, dropout: float, output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for hidden_dim in hidden_dims or [input_dim]:
        layers.extend(
            [
                nn.Linear(current_dim, int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        current_dim = int(hidden_dim)
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, use_batch_norm: bool = False) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        layers: list[nn.Module] = []
        for _ in range(2):
            layers.extend([nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation), Chomp1d(padding)])
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(channels))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + x)


class TcnEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [TemporalBlock(hidden_channels, kernel_size, dilation, dropout, use_batch_norm) for dilation in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = self.input_projection(x)
        for block in self.blocks:
            h = block(h)
        return h


class MultiScaleCrossDimensionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_scales: int,
        dropout: float,
        use_channel_attention: bool = False,
        use_temporal_descriptors: bool = False,
    ) -> None:
        super().__init__()
        self.use_channel_attention = bool(use_channel_attention)
        self.use_temporal_descriptors = bool(use_temporal_descriptors)
        descriptor_multiplier = 3 if self.use_temporal_descriptors else 1
        self.scale_gate = nn.Sequential(
            nn.Linear(channels * num_scales * descriptor_multiplier, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_scales),
            nn.Softmax(dim=-1),
        )
        reduction = max(channels // 4, 1)
        self.channel_gate = nn.Sequential(
            nn.Linear(channels, reduction),
            nn.ReLU(),
            nn.Linear(reduction, channels),
            nn.Sigmoid(),
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
        )

    def forward(self, scale_features: list[torch.Tensor]) -> torch.Tensor:
        last = [feat[:, :, -1] for feat in scale_features]
        mean = [feat.mean(dim=-1) for feat in scale_features]
        maximum = [feat.amax(dim=-1) for feat in scale_features]
        concat = torch.cat(last + mean + maximum, dim=-1) if self.use_temporal_descriptors else torch.cat(last, dim=-1)
        weights = self.scale_gate(concat)
        stacked = torch.stack(last, dim=1)
        mixed = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        if self.use_channel_attention:
            mixed = mixed * self.channel_gate(mixed)
        return self.channel_mixer(mixed) + mixed


class PTcnRegressor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
        output_dim: int = 1,
        use_batch_norm: bool = False,
        head_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = TcnEncoder(input_channels, hidden_channels, kernel_size, dilations, dropout, use_batch_norm)
        self.regressor = make_regressor(hidden_channels, head_hidden_dims, dropout, output_dim)

    def forward(self, x: torch.Tensor, return_features: bool = False, return_sequence: bool = False):
        encoded = self.encoder(x)
        if return_sequence:
            features = encoded.transpose(1, 2)
            pred = self.regressor(features)
            if return_features:
                return pred, features
            return pred
        features = encoded[:, :, -1]
        pred = self.regressor(features)
        if return_features:
            return pred, features
        return pred


class PSaMcdTcnRegressor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
        output_dim: int = 1,
        attention_heads: int = 4,
        use_temporal_attention: bool = False,
        use_channel_attention: bool = False,
        use_temporal_descriptors: bool = False,
        use_stage_head: bool = True,
        num_stages: int = 3,
        use_batch_norm: bool = False,
        head_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.use_temporal_attention = bool(use_temporal_attention)
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.scale_blocks = nn.ModuleList(
            [TemporalBlock(hidden_channels, kernel_size, dilation, dropout, use_batch_norm) for dilation in dilations]
        )
        if self.use_temporal_attention and hidden_channels % int(attention_heads) != 0:
            raise ValueError("hidden_channels must be divisible by attention_heads")
        self.temporal_attention = (
            nn.MultiheadAttention(hidden_channels, num_heads=int(attention_heads), dropout=dropout, batch_first=True)
            if self.use_temporal_attention
            else None
        )
        self.attention_norm = nn.LayerNorm(hidden_channels)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.mcd = MultiScaleCrossDimensionBlock(
            hidden_channels,
            len(dilations),
            dropout,
            use_channel_attention=use_channel_attention,
            use_temporal_descriptors=use_temporal_descriptors,
        )
        self.stage_head = nn.Linear(hidden_channels, int(num_stages)) if use_stage_head else None
        self.regressor = make_regressor(hidden_channels, head_hidden_dims, dropout, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_sequence: bool = False,
    ):
        h = self.input_projection(x.transpose(1, 2))
        scale_outputs = []
        current = h
        for block in self.scale_blocks:
            current = block(current)
            scale_outputs.append(current)
        sequence = current.transpose(1, 2)
        if self.temporal_attention is not None:
            attended, _ = self.temporal_attention(sequence, sequence, sequence, need_weights=False)
            attention_sequence = self.attention_norm(attended + sequence)
        else:
            attention_sequence = sequence
        if return_sequence:
            mcd_features = self.mcd(scale_outputs)
            mcd_sequence = mcd_features.unsqueeze(1).expand(-1, attention_sequence.size(1), -1)
            sequence_features = self.fusion(torch.cat([mcd_sequence, attention_sequence], dim=-1)) + mcd_sequence
            pred = self.regressor(sequence_features)
            if return_features:
                return pred, sequence_features
            return pred
        attention_features = attention_sequence[:, -1, :]
        mcd_features = self.mcd(scale_outputs)
        features = self.fusion(torch.cat([mcd_features, attention_features], dim=-1)) + mcd_features
        pred = self.regressor(features)
        stage_logits = self.stage_head(features) if self.stage_head is not None else None
        if return_aux:
            return pred, features, stage_logits
        if return_features:
            return pred, features
        return pred
