"""The spatial sublayer (spec 6.3): attention across one view's query tokens and its map
memory, with rotary positions, then a gated MLP."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernels
from .kernels import attention
from .layers import GatedMLP, apply_rotary, norm

Tables = tuple[torch.Tensor, torch.Tensor]  # (cos, sin), each (..., tokens, 1, pairs)


class SpatialBlock(nn.Module):
    """The spatial sublayer: rotary attention of the query tokens over themselves and the map
    memory, then a gated MLP, each pre-normed and residual."""

    def __init__(self, width: int, heads: int, map_width: int) -> None:
        super().__init__()
        self.heads = heads
        self.attention_norm = norm(width)
        self.query_key_value = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.map_norm = norm(map_width)
        self.map_key_value = nn.Linear(map_width, 2 * width, bias=False)
        self.feedforward_norm = norm(width)
        self.feedforward = GatedMLP(width, 4 * width)

    def forward(
        self,
        tokens: torch.Tensor,
        map_memory: torch.Tensor,
        token_tables: Tables,
        map_tables: Tables,
    ) -> torch.Tensor:
        """tokens (..., Q, D) attend over themselves and map_memory (..., M, Dm)."""
        query, key, value = (
            self.query_key_value(self.attention_norm(tokens))
            .unflatten(-1, (3, self.heads, -1))
            .unbind(-3)
        )
        map_key, map_value = (
            self.map_key_value(self.map_norm(map_memory))
            .unflatten(-1, (2, self.heads, -1))
            .unbind(-3)
        )
        if kernels.switches.attention and attention.applies(query, token_tables):
            attended = attention.attend(
                query, key, value, map_key, map_value, token_tables, map_tables
            )
        else:
            attended = self._attend(query, key, value, map_key, map_value, token_tables, map_tables)
        tokens = tokens + self.output(attended.flatten(-2))
        return tokens + self.feedforward(self.feedforward_norm(tokens))

    @staticmethod
    def _attend(query, key, value, map_key, map_value, token_tables, map_tables):
        """The plain twin of `kernels.attention.attend`: (..., Q, heads, width)."""
        query, key = apply_rotary(query, *token_tables), apply_rotary(key, *token_tables)
        map_key = apply_rotary(map_key, *map_tables)
        keys = torch.cat([key, map_key], -3).transpose(-3, -2)  # (..., heads, Q + M, width)
        values = torch.cat([value, map_value], -3).transpose(-3, -2)
        attended = F.scaled_dot_product_attention(query.transpose(-3, -2), keys, values)
        return attended.transpose(-3, -2)
