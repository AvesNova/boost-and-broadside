"""The spatial sublayer (spec 6.3): attention across one view's query tokens and its map
memory, with rotary positions, then a gated MLP."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernels
from .kernels import attention
from .layers import GatedMLP, apply_rotary, norm

Tables = tuple[torch.Tensor, torch.Tensor]  # (cos, sin), each (..., tokens, 1, pairs)

GATES = ("none", "elementwise", "headwise")  # `YemongConfig.gate`
GATE_BIAS = 2.0  # sigmoid(2) = 0.88: a new gate starts nearly open, its slope 0.10


class SpatialBlock(nn.Module):
    """The spatial sublayer: rotary attention of the query tokens over themselves and the map
    memory, then a gated MLP, each pre-normed and residual.

    gate (`GATES`): "none", or the attention output gate of Qiu et al., "Gated Attention for
    Large Language Models" (2025): each head's output times sigmoid(W x + b), x the normed
    tokens the query projection reads, before the output projection. "elementwise", one gate
    per output element (W: width to width); "headwise", one per head (W: width to heads).
    W starts at zero and b at `GATE_BIAS`, so every gate starts at sigmoid(2) = 0.88."""

    def __init__(self, width: int, heads: int, map_width: int, gate: str = "none") -> None:
        super().__init__()
        if gate not in GATES:
            raise ValueError(f"gate {gate!r} is not one of {GATES}")
        self.heads = heads
        self.attention_norm = norm(width)
        self.query_key_value = nn.Linear(width, 3 * width, bias=False)
        self.gate = None
        if gate != "none":
            self.gate = nn.Linear(width, width if gate == "elementwise" else heads)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, GATE_BIAS)
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
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """tokens (..., Q, D) attend over themselves and map_memory (..., M, Dm). key_mask
        (..., K) bool, K <= Q + M: which of the first K keys (the tokens, then the map memory)
        are keys and values (the ships, for `Yemong.mask_benched`; every key, for the
        `Ablations.absent_zones` mask); the rest always are. None: every token. Masked or
        not, the fused kernel runs where it applies, the plain path elsewhere."""
        normed = self.attention_norm(tokens)
        query, key, value = (
            self.query_key_value(normed).unflatten(-1, (3, self.heads, -1)).unbind(-3)
        )
        map_key, map_value = (
            self.map_key_value(self.map_norm(map_memory))
            .unflatten(-1, (2, self.heads, -1))
            .unbind(-3)
        )
        if kernels.switches.attention and attention.applies(query, token_tables):
            attended = attention.attend(
                query, key, value, map_key, map_value, token_tables, map_tables, key_mask
            )
        else:
            attended = self._attend(
                query, key, value, map_key, map_value, token_tables, map_tables, key_mask
            )
        if self.gate is not None:  # (..., Q, heads, width or 1): per element or per head
            gate = torch.sigmoid(self.gate(normed)).unflatten(-1, (self.heads, -1))
            attended = attended * gate.to(attended.dtype)
        tokens = tokens + self.output(attended.flatten(-2))
        return tokens + self.feedforward(self.feedforward_norm(tokens))

    @staticmethod
    def _attend(query, key, value, map_key, map_value, token_tables, map_tables, key_mask=None):
        """The plain twin of `kernels.attention.attend`: (..., Q, heads, width); with a
        key_mask (see `forward`) the masked keys take no weight."""
        query, key = apply_rotary(query, *token_tables), apply_rotary(key, *token_tables)
        map_key = apply_rotary(map_key, *map_tables)
        keys = torch.cat([key, map_key], -3).transpose(-3, -2)  # (..., heads, Q + M, width)
        values = torch.cat([value, map_value], -3).transpose(-3, -2)
        query = query.transpose(-3, -2)
        device = query.device.type
        if torch.is_autocast_enabled(device):
            # Autocast's own cast for attention, made explicit: under `torch.vmap` (the
            # stacked weight sets, `bnb.runtime.stacked`) autocast skips it, and the float32
            # query met the bf16 values. Bitwise the same where autocast does cast.
            dtype = torch.get_autocast_dtype(device)
            query, keys, values = query.to(dtype), keys.to(dtype), values.to(dtype)
        mask = None
        if key_mask is not None:
            shown = key_mask.new_ones(key_mask.shape[:-1] + (keys.shape[-2] - key_mask.shape[-1],))
            mask = torch.cat([key_mask, shown], -1)[..., None, None, :]  # (..., 1, 1, keys)
        attended = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
        return attended.transpose(-3, -2)
