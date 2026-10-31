"""The encoder and the trunk (spec 6.1, 6.5).

The query tokens are the ships, then the global token. Fields enter only as map memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import norm
from .spatial import SpatialBlock, Tables
from .temporal import TemporalBlock


class KindProjection(nn.Module):
    """One token kind's first projection: gelu(norm(W x + b)), width 2D."""

    def __init__(self, inputs: int, width: int) -> None:
        super().__init__()
        self.first = nn.Linear(inputs, width)
        self.first_norm = norm(width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.first_norm(self.first(tokens)))


class Encoder(nn.Module):
    def __init__(self, ship_inputs: int, global_inputs: int, field_inputs: int, width: int,
                 map_width: int) -> None:
        super().__init__()
        self.ship = KindProjection(ship_inputs, 2 * width)
        self.field = KindProjection(field_inputs, 2 * width)
        self.global_token = KindProjection(global_inputs, 2 * width)
        self.shared = nn.Linear(2 * width, width)
        self.shared_norm = norm(width)
        self.map_memory = nn.Linear(width, map_width, bias=False)
        self.map_memory_norm = norm(map_width)

    def _shared(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.shared_norm(self.shared(tokens))

    def forward(self, ship: torch.Tensor, global_token: torch.Tensor,
                field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ship (..., N, Fs), global_token (..., 1, Fg), field (..., M, Ff) to the queries
        (..., N + 1, D) and the map memory (..., M, Dm)."""
        queries = self._shared(torch.cat([self.ship(ship), self.global_token(global_token)], -2))
        memory = self.map_memory_norm(self.map_memory(self._shared(self.field(field))))
        return queries, memory


class Block(nn.Module):
    def __init__(self, width: int, heads: int, map_width: int, spatial_per_block: int) -> None:
        super().__init__()
        self.spatial = nn.ModuleList(SpatialBlock(width, heads, map_width)
                                     for _ in range(spatial_per_block))
        self.temporal = TemporalBlock(width)

    def attend(self, tokens, map_memory, token_tables, map_tables):
        for layer in self.spatial:
            tokens = layer(tokens, map_memory, token_tables, map_tables)
        return tokens


class Trunk(nn.Module):
    """Blocks of spatial sublayers then one temporal sublayer. The trunk carry is
    state (..., blocks, Q, D) and convolution_buffer (..., blocks, Q, 3, D)."""

    def __init__(self, width: int, heads: int, map_width: int, blocks: int,
                 spatial_per_block: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(width, heads, map_width, spatial_per_block)
                                    for _ in range(blocks))

    def step(self, tokens: torch.Tensor, map_memory: torch.Tensor, token_tables: Tables,
             map_tables: Tables, state: torch.Tensor, buffer: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One decision: tokens (..., Q, D). Returns (tokens, state, buffer)."""
        states, buffers = [], []
        for index, block in enumerate(self.blocks):
            tokens = block.attend(tokens, map_memory, token_tables, map_tables)
            tokens, block_state, block_buffer = block.temporal.step(
                tokens, state[..., index, :, :], buffer[..., index, :, :, :])
            states.append(block_state)
            buffers.append(block_buffer)
        return tokens, torch.stack(states, -3), torch.stack(buffers, -4)

    def sequence(self, tokens: torch.Tensor, map_memory: torch.Tensor, token_tables: Tables,
                 map_tables: Tables, state: torch.Tensor, buffer: torch.Tensor,
                 starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A segment: tokens (L, ..., Q, D), the carry at its start (..., blocks, Q, ...),
        starts (L, ...) (the first is ignored). The spatial sublayers run over every step at
        once; the temporal ones scan over time. Returns (tokens, final state, final buffer)."""
        starts = starts[..., None]  # per query token
        states, buffers = [], []
        for index, block in enumerate(self.blocks):
            tokens = block.attend(tokens, map_memory, token_tables, map_tables)
            tokens, block_state, block_buffer = block.temporal.sequence(
                tokens, state[..., index, :, :], buffer[..., index, :, :, :], starts)
            states.append(block_state)
            buffers.append(block_buffer)
        return tokens, torch.stack(states, -3), torch.stack(buffers, -4)
