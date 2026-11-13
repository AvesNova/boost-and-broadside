"""The encoder and the trunk (spec 6.1, 6.5).

The query tokens are the ships, then the global token. Fields, and the mode's zones, enter
only as map memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernels
from .kernels import heads as head_kernel
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
        """gelu(norm(W x + b)): (..., inputs) to (..., width)."""
        return F.gelu(self.first_norm(self.first(tokens)))


class Encoder(nn.Module):
    """One first projection per token kind, then a shared one. Query kinds: ships and the
    global token; memory kinds: fields and the mode's zones.

    An absent zone (a zone slot of a mode without zones, mixed spec 4) is one
    learned map-memory token, `absent_zone`, selected where the zone is not present; the
    projection's output for its row is discarded. No mask, so the fused attention kernel
    stays usable."""

    def __init__(
        self,
        ship_inputs: int,
        global_inputs: int,
        field_inputs: int,
        width: int,
        map_width: int,
        zone_inputs: int,
    ) -> None:
        super().__init__()
        self.ship = KindProjection(ship_inputs, 2 * width)
        self.field = KindProjection(field_inputs, 2 * width)
        self.global_token = KindProjection(global_inputs, 2 * width)
        self.zone = KindProjection(zone_inputs, 2 * width)
        self.shared = nn.Linear(2 * width, width)
        self.shared_norm = norm(width)
        self.map_memory = nn.Linear(width, map_width, bias=False)
        self.map_memory_norm = norm(map_width)
        # Uninitialised here (no random draw): the model initialises it after every other
        # layer, so theirs do not depend on it.
        self.absent_zone = nn.Parameter(torch.empty(map_width))

    def _kinds(
        self,
        ship: torch.Tensor,
        global_token: torch.Tensor,
        field: torch.Tensor,
        zone: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """shared(kind(tokens)) of the query kinds (ships, then the global token) and of the
        memory kinds (fields, then zones), before the shared norm."""
        if kernels.switches.heads and head_kernel.applies(ship, self.ship.first.out_features):

            def project(kind: KindProjection, tokens: torch.Tensor) -> torch.Tensor:
                return head_kernel.head(tokens, kind.first, kind.first_norm, self.shared)

            queries = torch.cat(
                [project(self.ship, ship), project(self.global_token, global_token)], -2
            )
            memory = [project(self.field, field), project(self.zone, zone)]
            return queries, torch.cat(memory, -2)
        queries = self.shared(torch.cat([self.ship(ship), self.global_token(global_token)], -2))
        memory = torch.cat([self.field(field), self.zone(zone)], -2)
        return queries, self.shared(memory)

    def forward(
        self,
        ship: torch.Tensor,
        global_token: torch.Tensor,
        field: torch.Tensor,
        zone: torch.Tensor,
        zone_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ship (..., N, Fs), global_token (..., 1, Fg), field (..., M, Ff), zone (..., Z, Fz)
        and zone_present (..., Z) bool to the queries (..., N + 1, D) and the map memory
        (..., M + Z, Dm), fields first."""
        queries, memory = self._kinds(ship, global_token, field, zone)
        queries = self.shared_norm(queries)
        memory = self.map_memory_norm(self.map_memory(self.shared_norm(memory)))
        fields = field.shape[-2]
        zones = torch.where(
            zone_present[..., None], memory[..., fields:, :], self.absent_zone.to(memory.dtype)
        )
        memory = torch.cat([memory[..., :fields, :], zones], -2)
        return queries, memory


ORDERS = ("spatial_first", "temporal_first", "spatial_only")


class Block(nn.Module):
    """One trunk block: spatial_per_block spatial sublayers and one temporal sublayer (none
    when not `temporal`), in the trunk's order (`ORDERS`)."""

    def __init__(
        self, width: int, heads: int, map_width: int, spatial_per_block: int, temporal: bool
    ) -> None:
        super().__init__()
        self.spatial = nn.ModuleList(
            SpatialBlock(width, heads, map_width) for _ in range(spatial_per_block)
        )
        self.temporal = TemporalBlock(width) if temporal else None

    def attend(self, tokens, map_memory, token_tables, map_tables, key_mask=None):
        """The tokens (..., Q, D) after each of the block's spatial sublayers in turn, each
        attending over the map memory; key_mask: `SpatialBlock.forward`'s."""
        for layer in self.spatial:
            tokens = layer(tokens, map_memory, token_tables, map_tables, key_mask)
        return tokens

    def step(
        self,
        tokens,
        map_memory,
        token_tables,
        map_tables,
        state,
        buffer,
        temporal_first,
        key_mask=None,
    ):
        """One decision through the block, its temporal sublayer before (`temporal_first`) or
        after its spatial ones: (tokens, the block's state, its buffer); key_mask:
        `SpatialBlock.forward`'s."""
        if not temporal_first:
            tokens = self.attend(tokens, map_memory, token_tables, map_tables, key_mask)
        tokens, state, buffer = self.temporal.step(tokens, state, buffer)
        if temporal_first:
            tokens = self.attend(tokens, map_memory, token_tables, map_tables, key_mask)
        return tokens, state, buffer

    def sequence(
        self,
        tokens,
        map_memory,
        token_tables,
        map_tables,
        state,
        buffer,
        starts,
        temporal_first,
        key_mask=None,
    ):
        """A segment through the block (see `Trunk.sequence`): (tokens, the block's final
        state, its final buffer); key_mask: `SpatialBlock.forward`'s."""
        if not temporal_first:
            tokens = self.attend(tokens, map_memory, token_tables, map_tables, key_mask)
        tokens, state, buffer = self.temporal.sequence(tokens, state, buffer, starts)
        if temporal_first:
            tokens = self.attend(tokens, map_memory, token_tables, map_tables, key_mask)
        return tokens, state, buffer


class Trunk(nn.Module):
    """Blocks of spatial sublayers and one temporal sublayer each: per block the spatial
    sublayers then the temporal one ("spatial_first"), or the temporal one then the spatial
    ones ("temporal_first"); or the spatial sublayers alone ("spatial_only", the ablation
    without temporal layers: the carry holds no trunk state, only the model's beliefs).
    The trunk carry is state (..., temporal_blocks, Q, D) and convolution_buffer
    (..., temporal_blocks, Q, 3, D).
    """

    def __init__(
        self,
        width: int,
        heads: int,
        map_width: int,
        blocks: int,
        spatial_per_block: int,
        order: str = "spatial_first",
    ) -> None:
        super().__init__()
        if order not in ORDERS:
            raise ValueError(f"order {order!r} is not one of {ORDERS}")
        self.order = order
        temporal = order != "spatial_only"
        self.temporal_blocks = blocks if temporal else 0  # the carry's blocks
        self.blocks = nn.ModuleList(
            Block(width, heads, map_width, spatial_per_block, temporal) for _ in range(blocks)
        )

    def _attend(self, tokens, map_memory, token_tables, map_tables, key_mask) -> torch.Tensor:
        """Every block's spatial sublayers in turn ("spatial_only")."""
        for block in self.blocks:
            tokens = block.attend(tokens, map_memory, token_tables, map_tables, key_mask)
        return tokens

    def step(
        self,
        tokens: torch.Tensor,
        map_memory: torch.Tensor,
        token_tables: Tables,
        map_tables: Tables,
        state: torch.Tensor,
        buffer: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One decision: tokens (..., Q, D); key_mask (..., K) bool or None, the first K keys
        attention reads (`SpatialBlock.forward`). Returns (tokens, state, buffer)."""
        if not self.temporal_blocks:
            attended = self._attend(tokens, map_memory, token_tables, map_tables, key_mask)
            return attended, state, buffer
        states, buffers = [], []
        temporal_first = self.order == "temporal_first"
        for index, block in enumerate(self.blocks):
            tokens, block_state, block_buffer = block.step(
                tokens,
                map_memory,
                token_tables,
                map_tables,
                state[..., index, :, :],
                buffer[..., index, :, :, :],
                temporal_first,
                key_mask,
            )
            states.append(block_state)
            buffers.append(block_buffer)
        return tokens, torch.stack(states, -3), torch.stack(buffers, -4)

    def sequence(
        self,
        tokens: torch.Tensor,
        map_memory: torch.Tensor,
        token_tables: Tables,
        map_tables: Tables,
        state: torch.Tensor,
        buffer: torch.Tensor,
        starts: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A segment: tokens (L, ..., Q, D), the carry at its start (..., blocks, Q, ...),
        starts (L, ...) (the first is ignored); key_mask (L, ..., N) as `step`'s. The spatial
        sublayers run over every step at once; the temporal ones scan over time. Returns
        (tokens, final state, final buffer)."""
        if not self.temporal_blocks:
            attended = self._attend(tokens, map_memory, token_tables, map_tables, key_mask)
            return attended, state, buffer
        starts = starts[..., None]  # per query token
        states, buffers = [], []
        temporal_first = self.order == "temporal_first"
        for index, block in enumerate(self.blocks):
            tokens, block_state, block_buffer = block.sequence(
                tokens,
                map_memory,
                token_tables,
                map_tables,
                state[..., index, :, :],
                buffer[..., index, :, :, :],
                starts,
                temporal_first,
                key_mask,
            )
            states.append(block_state)
            buffers.append(block_buffer)
        return tokens, torch.stack(states, -3), torch.stack(buffers, -4)
