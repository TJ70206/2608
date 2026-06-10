from __future__ import annotations

import math

import torch
from torch import nn

from xa202608.models.tcn import make_regressor


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerRulRegressor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_dim: int = 1,
        max_len: int = 512,
        pooling: str = "last",
        head_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.pooling = str(pooling).lower()
        self.input_projection = nn.Linear(int(input_channels), int(d_model))
        self.position = SinusoidalPositionalEncoding(int(d_model), int(max_len), float(dropout))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(int(d_model))
        self.regressor = make_regressor(int(d_model), head_hidden_dims, float(dropout), int(output_dim))

    def forward(self, x: torch.Tensor, return_features: bool = False, return_sequence: bool = False):
        sequence = self.norm(self.encoder(self.position(self.input_projection(x))))
        if return_sequence:
            pred = self.regressor(sequence)
            if return_features:
                return pred, sequence
            return pred
        if self.pooling == "mean":
            features = sequence.mean(dim=1)
        else:
            features = sequence[:, -1, :]
        pred = self.regressor(features)
        if return_features:
            return pred, features
        return pred
