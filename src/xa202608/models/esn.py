from __future__ import annotations

import torch
from torch import nn

from xa202608.models.tcn import make_regressor


class LeakyEsnRegressor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        reservoir_size: int = 256,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.3,
        input_scale: float = 0.5,
        reservoir_density: float = 0.1,
        dropout: float = 0.0,
        output_dim: int = 1,
        train_reservoir: bool = False,
        head_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.reservoir_size = int(reservoir_size)
        self.leak_rate = float(leak_rate)
        input_weight = torch.empty(self.reservoir_size, int(input_channels)).uniform_(-float(input_scale), float(input_scale))
        recurrent = torch.empty(self.reservoir_size, self.reservoir_size).uniform_(-1.0, 1.0)
        mask = torch.rand_like(recurrent).lt(float(reservoir_density))
        recurrent = recurrent * mask
        eigenvalues = torch.linalg.eigvals(recurrent).abs()
        radius = float(eigenvalues.max().real) if eigenvalues.numel() else 1.0
        if radius > 1e-8:
            recurrent = recurrent * (float(spectral_radius) / radius)
        self.input_weight = nn.Parameter(input_weight, requires_grad=bool(train_reservoir))
        self.recurrent_weight = nn.Parameter(recurrent, requires_grad=bool(train_reservoir))
        self.bias = nn.Parameter(torch.zeros(self.reservoir_size), requires_grad=bool(train_reservoir))
        self.dropout = nn.Dropout(float(dropout))
        self.regressor = make_regressor(self.reservoir_size, head_hidden_dims, float(dropout), int(output_dim))

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        state = x.new_zeros(batch_size, self.reservoir_size)
        states = []
        input_weight = self.input_weight.to(dtype=x.dtype, device=x.device)
        recurrent_weight = self.recurrent_weight.to(dtype=x.dtype, device=x.device)
        bias = self.bias.to(dtype=x.dtype, device=x.device)
        for step in range(seq_len):
            pre_activation = x[:, step, :] @ input_weight.t() + state @ recurrent_weight.t() + bias
            candidate = torch.tanh(pre_activation)
            state = (1.0 - self.leak_rate) * state + self.leak_rate * candidate
            states.append(state)
        return torch.stack(states, dim=1)

    def forward(self, x: torch.Tensor, return_features: bool = False, return_sequence: bool = False):
        sequence = self.dropout(self.encode_sequence(x))
        if return_sequence:
            pred = self.regressor(sequence)
            if return_features:
                return pred, sequence
            return pred
        features = sequence[:, -1, :]
        pred = self.regressor(features)
        if return_features:
            return pred, features
        return pred
