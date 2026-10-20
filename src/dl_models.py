"""PyTorch recurrent regressors for FlowCast sequence forecasting."""

from __future__ import annotations

import torch
from torch import nn


class RecurrentTrafficRegressor(nn.Module):
    """Unidirectional recurrent network with a scalar regression head."""

    recurrent_type = "recurrent"

    def __init__(
        self,
        input_size: int,
        *,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be positive")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = self._build_recurrent(
            input_size,
            hidden_size,
            num_layers,
            recurrent_dropout,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def _build_recurrent(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Module:
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"inputs must have shape (batch, sequence, {self.input_size})"
            )
        _, hidden = self.recurrent(inputs)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        final_hidden = hidden[-1]
        return self.output(self.head_dropout(final_hidden)).squeeze(-1)


class LSTMRegressor(RecurrentTrafficRegressor):
    """Unidirectional LSTM volume regressor."""

    recurrent_type = "lstm"

    def _build_recurrent(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Module:
        return nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
        )


class GRURegressor(RecurrentTrafficRegressor):
    """Unidirectional GRU volume regressor."""

    recurrent_type = "gru"

    def _build_recurrent(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Module:
        return nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
        )
