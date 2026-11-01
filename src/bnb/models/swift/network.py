"""The network (spec 6 and 7): encoder, rotary, spatial and temporal sublayers, trunk, heads.

Module attribute names follow the fixture weight names (tests/reference/yemong/README.md),
so a fixture's flat weight dict loads with `load_state_dict`.

Layouts: tokens are (..., Q, D). The temporal sublayer and the trunk are time-major:
(T, B, Q, D), with `starts` (T, B) bool marking "an episode starts at t" (t > 0; the stored
carry continues the segment, so step 0 never counts as a start).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .scan import linear_scan

DECAY_EXPONENT = 8.0


class RMSNorm(nn.Module):
    """RMSNorm with a learned scale and the dtype's machine epsilon. Low-precision inputs
    (autocast) are normalised in float32, with float32's epsilon."""

    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        if x.dtype not in (torch.float32, torch.float64):
            x = x.float()
        return F.rms_norm(x, (x.shape[-1],), self.weight, eps=torch.finfo(x.dtype).eps)


def gelu(x):
    return F.gelu(x)


class GatedMLP(nn.Module):
    """down(gelu(gate x) * up x), no biases."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        gate_up = F.linear(x, torch.cat([self.gate.weight, self.up.weight]))
        gate, up = gate_up.chunk(2, -1)
        return self.down(gelu(gate) * up)


# --------------------------------------------------------------------------- rotary


def rotary_tables(frequencies_x, frequencies_y, frequencies_heading, position, heading_angle):
    """position (..., N, 2) m; heading_angle (..., N) rad (0 for headless tokens).
    Returns (cos, sin), each (..., N, 1, R) with R the total frequency count."""
    angles = torch.cat([position[..., 0:1] * frequencies_x, position[..., 1:2] * frequencies_y,
                        heading_angle.unsqueeze(-1) * frequencies_heading], -1)
    return torch.cos(angles).unsqueeze(-2), torch.sin(angles).unsqueeze(-2)


def apply_rotary(values, cos, sin):
    """values (..., N, H, W); cos, sin (..., N, 1, R). Rotates the first 2R dims as
    interleaved (even, odd) pairs; the rest pass through."""
    pairs = cos.shape[-1]
    rotated = values[..., : 2 * pairs].unflatten(-1, (pairs, 2))
    even, odd = rotated[..., 0], rotated[..., 1]
    out = torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)
    return torch.cat([out, values[..., 2 * pairs:]], -1)


# --------------------------------------------------------------------------- spatial


class SpatialBlock(nn.Module):
    """x += W_o attention(norm x) over [query keys; map keys]; x += mlp(norm x)."""

    def __init__(self, width: int, heads: int, map_width: int):
        super().__init__()
        self.heads = heads
        self.attention_norm = RMSNorm(width)
        self.query_key_value = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.map_norm = RMSNorm(map_width)
        self.map_key_value = nn.Linear(map_width, 2 * width, bias=False)
        self.feedforward_norm = RMSNorm(width)
        self.feedforward = GatedMLP(width, 4 * width)

    def map_keys_values(self, memory, map_tables):
        """memory (..., M, Dm) -> rotated keys, values, each (..., M, H, W)."""
        keys, values = self.map_key_value(self.map_norm(memory)).chunk(2, -1)
        keys = apply_rotary(keys.unflatten(-1, (self.heads, -1)), *map_tables)
        return keys, values.unflatten(-1, (self.heads, -1))

    def forward(self, x, memory, token_tables, map_tables, map_keys_values=None):
        """x (..., Q, D); memory (..., M, Dm); tables (..., tokens, 1, R)."""
        query, key, value = self.query_key_value(self.attention_norm(x)).chunk(3, -1)
        query = apply_rotary(query.unflatten(-1, (self.heads, -1)), *token_tables)
        key = apply_rotary(key.unflatten(-1, (self.heads, -1)), *token_tables)
        value = value.unflatten(-1, (self.heads, -1))
        if map_keys_values is None:
            map_keys_values = self.map_keys_values(memory, map_tables)
        map_key, map_value = map_keys_values
        key = torch.cat([key, map_key], -3).transpose(-2, -3)
        value = torch.cat([value, map_value], -3).transpose(-2, -3)
        query = query.transpose(-2, -3)
        batch = query.shape[:-3]
        attended = F.scaled_dot_product_attention(query.flatten(0, -4), key.flatten(0, -4),
                                                  value.flatten(0, -4)).unflatten(0, batch)
        x = x + self.output(attended.transpose(-2, -3).flatten(-2))
        return x + self.feedforward(self.feedforward_norm(x))


# --------------------------------------------------------------------------- temporal


def lag_keeps(starts, dtype):
    """starts (T, ...) bool -> keep (3, T, ...): keep[j - 1, t] = 1 when no episode starts at
    t - j + 1 ... t (lags j = 1, 2, 3)."""
    fresh = starts.to(dtype)
    keep_one = 1 - fresh
    keep_two = keep_one * (1 - F.pad(fresh[:-1], [0] * (2 * fresh.dim() - 2) + [1, 0]))
    keep_three = keep_two * (1 - F.pad(fresh[:-2], [0] * (2 * fresh.dim() - 2) + [2, 0]))
    return torch.stack([keep_one, keep_two, keep_three])


class RGLRU(nn.Module):
    """h_t = a_t h_{t-1} + sqrt(1 - a_t^2) i_t c_t, a_t = sigmoid(Lambda)^(8 r_t)."""

    def __init__(self, width: int):
        super().__init__()
        self.log_decay = nn.Parameter(torch.linspace(0, 4, width))
        self.recurrence_gate = nn.Linear(width, width)
        self.input_gate = nn.Linear(width, width)

    def gates(self, inputs, starts):
        """inputs (T, ..., D) -> (decay, contribution), each (T, ..., D)."""
        weight = torch.cat([self.recurrence_gate.weight, self.input_gate.weight])
        bias = torch.cat([self.recurrence_gate.bias, self.input_gate.bias])
        recurrence, gate = torch.sigmoid(F.linear(inputs, weight, bias)).chunk(2, -1)
        decay = torch.exp(DECAY_EXPONENT * recurrence * F.logsigmoid(self.log_decay))
        multiplier = torch.sqrt(torch.clamp_min((1 - decay) * (1 + decay), 1e-6))
        contribution = multiplier * gate * inputs
        if starts is not None:
            decay = decay * (1 - starts.to(decay.dtype)).unsqueeze(-1)
        return decay, contribution

    def forward(self, inputs, initial, starts=None):
        """inputs (T, ..., D); initial (..., D) -> (outputs (T, ..., D), final (..., D))."""
        decay, contribution = self.gates(inputs, starts)
        outputs = linear_scan(decay, contribution, initial)
        return outputs, outputs[-1]


class DepthwiseConvolution(nn.Module):
    """weight (D, 4): column k applies to the input 3 - k steps back."""

    def __init__(self, width: int):
        super().__init__()
        bound = 1 / math.sqrt(4)
        self.weight = nn.Parameter(torch.empty(width, 4).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(width).uniform_(-bound, bound))

    def forward(self, inputs, buffer, starts):
        """inputs (T, ..., D); buffer (..., 3, D) oldest first; starts (T, ...) or None.
        Returns (outputs (T, ..., D), new buffer (..., 3, D))."""
        steps = inputs.shape[0]
        history = torch.cat([buffer.movedim(-2, 0), inputs], 0)  # (T + 3, ..., D)
        out = inputs * self.weight[:, 3] + self.bias
        keeps = None if starts is None else lag_keeps(starts, inputs.dtype).unsqueeze(-1)
        for lag in (1, 2, 3):
            tap = history[3 - lag: 3 - lag + steps] * self.weight[:, 3 - lag]
            out = out + (tap if keeps is None else tap * keeps[lag - 1])
        return out, history[-3:].movedim(0, -2)


class TemporalBlock(nn.Module):
    """Per query token over time: convolution, RG-LRU, gated output, then a gated MLP."""

    def __init__(self, width: int):
        super().__init__()
        self.input_norm = RMSNorm(width)
        self.recurrent_input = nn.Linear(width, width, bias=False)
        self.convolution = DepthwiseConvolution(width)
        self.log_decay = nn.Parameter(torch.linspace(0, 4, width))
        self.recurrence_gate = nn.Linear(width, width)
        self.input_gate = nn.Linear(width, width)
        self.gate_branch = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.feedforward_norm = RMSNorm(width)
        self.feedforward = GatedMLP(width, 4 * width)

    gates = RGLRU.gates

    def forward(self, x, state, buffer, starts=None):
        """x (T, ..., D); state (..., D); buffer (..., 3, D); starts (T, ...) bool or None.
        Returns (x, final state, final buffer)."""
        normed = self.input_norm(x)
        recurrent, branch = F.linear(
            normed, torch.cat([self.recurrent_input.weight, self.gate_branch.weight])).chunk(2, -1)
        convolved, buffer = self.convolution(recurrent, buffer, starts)
        decay, contribution = self.gates(convolved, starts)
        hidden = linear_scan(decay, contribution, state)
        x = x + self.output(hidden * gelu(branch))
        return x + self.feedforward(self.feedforward_norm(x)), hidden[-1], buffer


# --------------------------------------------------------------------------- encoder, trunk


class KindProjection(nn.Module):
    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self.first = nn.Linear(inputs, hidden)
        self.first_norm = RMSNorm(hidden)

    def forward(self, x):
        return gelu(self.first_norm(self.first(x)))


class Encoder(nn.Module):
    """Per kind gelu(norm(W x + b)) to 2D, shared norm(W h + b) to D; map memory
    norm(W_m t_field) to D_m."""

    def __init__(self, ship_width: int, global_width: int, field_width: int, width: int,
                 map_width: int):
        super().__init__()
        self.kinds = nn.ModuleDict({
            "ship": KindProjection(ship_width, 2 * width),
            "global": KindProjection(global_width, 2 * width),
            "field": KindProjection(field_width, 2 * width),
        })
        self.shared = nn.Linear(2 * width, width)
        self.shared_norm = RMSNorm(width)
        self.map_memory = nn.Linear(width, map_width, bias=False)
        self.map_memory_norm = RMSNorm(map_width)

    def fixture_state(self, weights):
        """The fixture's flat names ('ship.first.weight') to this module's."""
        return {(f"kinds.{key}" if key.split(".")[0] in self.kinds else key): value
                for key, value in weights.items()}

    def shared_projection(self, hidden):
        return self.shared_norm(self.shared(hidden))

    def forward(self, ship, global_token, field):
        """ship (..., N, Fs), global_token (..., 1, Fg), field (..., M, Ff) ->
        (queries (..., N + 1, D), map memory (..., M, Dm))."""
        ships = self.shared_projection(self.kinds["ship"](ship))
        globals_ = self.shared_projection(self.kinds["global"](global_token))
        fields = self.shared_projection(self.kinds["field"](field))
        memory = self.map_memory_norm(self.map_memory(fields))
        return torch.cat([ships, globals_], -2), memory


class Block(nn.Module):
    def __init__(self, width, heads, map_width, spatial_per_block):
        super().__init__()
        self.spatial = nn.ModuleList(SpatialBlock(width, heads, map_width)
                                     for _ in range(spatial_per_block))
        self.temporal = TemporalBlock(width)


class Trunk(nn.Module):
    """Per block: the spatial sublayers over the queries (reading map memory), then one
    temporal sublayer."""

    def __init__(self, width: int, heads: int, map_width: int, blocks: int,
                 spatial_per_block: int):
        super().__init__()
        self.blocks = nn.ModuleList(Block(width, heads, map_width, spatial_per_block)
                                    for _ in range(blocks))

    def forward(self, tokens, memory, token_tables, map_tables, state, buffer, starts=None):
        """tokens (T, B, Q, D); memory (T, B, M, Dm); tables (T, B, tokens, 1, R); state
        (blocks, B, Q, D); buffer (blocks, B, Q, 3, D); starts (T, B) bool or None.
        Returns (tokens (T, B, Q, D), state, buffer)."""
        token_starts = None if starts is None else starts.unsqueeze(-1).expand(
            *starts.shape, tokens.shape[-2])
        states, buffers = [], []
        x = tokens
        for index, block in enumerate(self.blocks):
            for spatial in block.spatial:
                x = spatial(x, memory, token_tables, map_tables)
            x, block_state, block_buffer = block.temporal(x, state[index], buffer[index],
                                                          token_starts)
            states.append(block_state)
            buffers.append(block_buffer)
        return x, torch.stack(states), torch.stack(buffers)


# --------------------------------------------------------------------------- heads


class MLPHead(nn.Module):
    """last(gelu(norm(first x)))."""

    def __init__(self, inputs: int, hidden: int, outputs: int):
        super().__init__()
        self.first = nn.Linear(inputs, hidden)
        self.norm = RMSNorm(hidden)
        self.last = nn.Linear(hidden, outputs)

    def forward(self, x):
        return self.last(gelu(self.norm(self.first(x))))

    def initialise(self, last_gain: float = 0.01):
        nn.init.orthogonal_(self.first.weight, math.sqrt(2))
        nn.init.zeros_(self.first.bias)
        nn.init.orthogonal_(self.last.weight, last_gain)
        nn.init.zeros_(self.last.bias)


class ShotTrajectoryHead(nn.Module):
    """h = W_latent x + b + W_age phi(age); norm, gelu, Linear(D, D), norm, gelu, Linear(D, out).
    The latent is computed once per ship-step and broadcast over the queried ages."""

    def __init__(self, width: int, time_features: int, outputs: int):
        super().__init__()
        self.latent = nn.Linear(width, width)
        self.age = nn.Linear(time_features, width, bias=False)
        self.first_norm = RMSNorm(width)
        self.middle = nn.Linear(width, width)
        self.middle_norm = RMSNorm(width)
        self.last = nn.Linear(width, outputs)

    def initialise(self):
        nn.init.orthogonal_(self.latent.weight, math.sqrt(2))
        nn.init.zeros_(self.latent.bias)
        nn.init.orthogonal_(self.middle.weight, math.sqrt(2))
        nn.init.zeros_(self.middle.bias)
        nn.init.orthogonal_(self.last.weight, 0.01)
        nn.init.zeros_(self.last.bias)

    def query(self, latent, age_features):
        """latent (..., D); age_features (..., A, F) -> logits (..., A, outputs)."""
        hidden = latent.unsqueeze(-2) + self.age(age_features)
        hidden = gelu(self.middle_norm(self.middle(gelu(self.first_norm(hidden)))))
        return self.last(hidden)
