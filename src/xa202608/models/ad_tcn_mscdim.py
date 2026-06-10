from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import weight_norm

from xa202608.models.tcn import Chomp1d, make_regressor


class AdaptiveDilationBank(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
        use_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        if not dilations:
            raise ValueError("dilations must not be empty")
        self.dilations = [int(d) for d in dilations]
        init_dilation = (min(self.dilations) + max(self.dilations)) / 2.0
        logits = torch.tensor([-abs(float(d) - init_dilation) for d in self.dilations], dtype=torch.float32)
        self.logits = nn.Parameter(logits)
        branches = []
        for dilation in self.dilations:
            padding = (int(kernel_size) - 1) * int(dilation)
            conv = nn.Conv1d(channels, channels, int(kernel_size), padding=padding, dilation=int(dilation))
            if use_weight_norm:
                conv = weight_norm(conv)
            branches.append(nn.Sequential(conv, Chomp1d(padding)))
        self.branches = nn.ModuleList(branches)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.logits, dim=0)
        outputs = torch.stack([branch(x) for branch in self.branches], dim=0)
        mixed = (outputs * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return self.dropout(mixed)


class ADTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
        use_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = AdaptiveDilationBank(channels, kernel_size, dilations, dropout, use_weight_norm)
        self.conv2 = AdaptiveDilationBank(channels, kernel_size, dilations, dropout, use_weight_norm)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.conv1(x))
        h = self.conv2(h)
        return self.activation(h + x)


class ADTCNStack(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
        num_blocks: int = 2,
        use_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ADTCNBlock(channels, kernel_size, dilations, dropout, use_weight_norm) for _ in range(int(num_blocks))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class MSCDIModule(nn.Module):
    def __init__(
        self,
        window_size: int,
        hidden_channels: int,
        conv_kernel_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.hidden_channels = int(hidden_channels)
        branches = []
        for kernel_size in conv_kernel_sizes:
            kernel_size = int(kernel_size)
            padding = kernel_size // 2
            branches.append(
                nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=(kernel_size, kernel_size), padding=(padding, padding)),
                    nn.ReLU(),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fc_w = nn.Linear(self.window_size, self.window_size)
        self.fc_h = nn.Linear(self.hidden_channels, self.hidden_channels)
        fusion_dim = self.window_size + self.hidden_channels
        hidden_dim = max(fusion_dim // 2, 1)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.window_size or x.size(2) != self.hidden_channels:
            raise ValueError(
                f"MSCDIModule expected input shape [batch,{self.window_size},{self.hidden_channels}], "
                f"got {tuple(x.shape)}"
            )
        y = torch.stack([branch(x.unsqueeze(1)).squeeze(1) for branch in self.branches], dim=0).sum(dim=0)
        y_w = self.fc_w(y.mean(dim=2) + y.amax(dim=2))
        y_h = self.fc_h(y.mean(dim=1) + y.amax(dim=1))
        z = self.fusion(torch.cat([y_w, y_h], dim=-1))
        z_w, z_h = torch.split(z, [self.window_size, self.hidden_channels], dim=-1)
        sigma_w = torch.sigmoid(z_w).unsqueeze(-1)
        sigma_h = torch.sigmoid(z_h).unsqueeze(1)
        return x * sigma_w * sigma_h


class AdTcnMscdimRegressor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
        output_dim: int = 1,
        window_size: int = 40,
        num_ad_blocks: int = 2,
        mscdim_kernel_sizes: list[int] | None = None,
        use_weight_norm: bool = True,
        use_stage_head: bool = False,
        num_stages: int = 3,
        head_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.input_mapping = nn.Linear(int(input_channels), int(hidden_channels))
        self.time_ad_tcn = ADTCNStack(
            int(hidden_channels), int(kernel_size), dilations, float(dropout), int(num_ad_blocks), bool(use_weight_norm)
        )
        self.sensor_ad_tcn = ADTCNStack(
            self.window_size, int(kernel_size), dilations, float(dropout), int(num_ad_blocks), bool(use_weight_norm)
        )
        self.mscdim = MSCDIModule(
            self.window_size,
            int(hidden_channels),
            mscdim_kernel_sizes or [1, 3, 5],
            float(dropout),
        )
        self.norm = nn.LayerNorm(int(hidden_channels))
        self.stage_head = nn.Linear(int(hidden_channels), int(num_stages)) if use_stage_head else None
        self.regressor = make_regressor(int(hidden_channels), head_hidden_dims, float(dropout), int(output_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.window_size:
            raise ValueError(f"AdTcnMscdimRegressor expected window_size={self.window_size}, got {x.size(1)}")
        h = self.input_mapping(x)
        h = self.time_ad_tcn(h.transpose(1, 2)).transpose(1, 2)
        h = self.sensor_ad_tcn(h)
        h = self.mscdim(h)
        return self.norm(h)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_sequence: bool = False,
    ):
        sequence = self.encode(x)
        if return_sequence:
            pred = self.regressor(sequence)
            if return_features:
                return pred, sequence
            return pred
        features = sequence[:, -1, :]
        pred = self.regressor(features)
        stage_logits = self.stage_head(features) if self.stage_head is not None else None
        if return_aux:
            return pred, features, stage_logits
        if return_features:
            return pred, features
        return pred
