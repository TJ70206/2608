from __future__ import annotations

import torch
from torch import nn

from xa202608.models.tcn import make_regressor


class RecurrentRulRegressor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        output_dim: int = 1,
        cell_type: str = "lstm",
        bidirectional: bool = False,
        pooling: str = "last",
        use_stage_head: bool = False,
        num_stages: int = 3,
        head_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        cell_key = str(cell_type).lower()
        if cell_key not in {"lstm", "gru"}:
            raise ValueError("cell_type must be 'lstm' or 'gru'")
        self.pooling = str(pooling).lower()
        recurrent_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        recurrent_cls = nn.LSTM if cell_key == "lstm" else nn.GRU
        self.recurrent = recurrent_cls(
            input_size=int(input_channels),
            hidden_size=int(hidden_channels),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bool(bidirectional),
        )
        feature_dim = int(hidden_channels) * (2 if bool(bidirectional) else 1)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.stage_head = nn.Linear(feature_dim, int(num_stages)) if use_stage_head else None
        self.regressor = make_regressor(feature_dim, head_hidden_dims, float(dropout), int(output_dim))

    def _pool(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return sequence.mean(dim=1)
        if self.pooling == "last":
            return sequence[:, -1, :]
        raise ValueError("pooling must be 'last' or 'mean'")

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_sequence: bool = False,
    ):
        sequence, _ = self.recurrent(x)
        sequence = self.dropout(self.norm(sequence))
        if return_sequence:
            pred = self.regressor(sequence)
            if return_features:
                return pred, sequence
            return pred
        features = self._pool(sequence)
        pred = self.regressor(features)
        stage_logits = self.stage_head(features) if self.stage_head is not None else None
        if return_aux:
            return pred, features, stage_logits
        if return_features:
            return pred, features
        return pred
