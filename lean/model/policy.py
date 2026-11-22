"""The policy: one token per ship plus a global token, blocks of one recurrent sublayer and
two masked rotary attention sublayers, and heads that read each ship's own token.

Sizes: T ticks, E environments, N ship slots, M = 1 + N trunk tokens (the global token
first, then the ship slots in the caller's order). The model has no parameter or buffer
whose shape depends on N, and nothing is computed across the entity axis except attention.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lean.model.value import value_bin_centres


def _default_position_periods() -> tuple[float, ...]:
    return tuple(float(period) for period in torch.logspace(math.log10(64.0), math.log10(20000.0), 6))


@dataclass
class ModelConfig:
    """Sizes and constants of the policy.

    position_periods: spatial periods in metres of the Fourier position code and of the
    rotary position rotation (one shared set). The coarsest must exceed twice the arena
    diameter so that no pair of ships aliases, and the ladder must not be a power of two.
    heading_harmonics: multiples of the heading angle used as rotary angles.
    rotate_values: also rotate the attention values by the key's position and heading and
    rotate the attended sum back by the query's (the audit's open arm).
    positionless_rotary: what the global and null tokens do in the rotated dimensions of
    queries and keys. "zero" removes those dimensions from every score that involves such
    a token; "identity" leaves them unrotated, which places the token at the arena origin
    with heading zero.
    value_low, value_high: per reward level, the outermost value bin centres in return
    units (level order: outcome, kill and death, damage, shaping).
    next_state_fields: the State fields whose one-tick change the next-state head
    predicts, each with the divisor that brings the change to order one.
    """

    width: int = 128
    blocks: int = 2
    spatial_per_block: int = 2
    heads: int = 2
    actions: int = 42
    levels: int = 4
    bins: int = 77
    recurrence_exponent: float = 8.0
    retention_low: float = 0.9
    retention_high: float = 0.999
    norm_epsilon: float = 1e-6
    position_periods: tuple[float, ...] = _default_position_periods()
    heading_harmonics: tuple[float, ...] = (1.0, 2.0, 4.0)
    rotate_values: bool = False
    positionless_rotary: str = "zero"
    value_low: tuple[float, ...] = (-2.0, -2.0, -2.0, -2.0)
    value_high: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0)
    hl_gauss_sigma_bins: float = 0.75
    next_state_fields: tuple[tuple[str, float], ...] = (
        ("x", 10.0),
        ("y", 10.0),
        ("heading_cos", 0.1),
        ("heading_sin", 0.1),
        ("speed", 5.0),
        ("energy", 10.0),
        ("shield", 10.0),
    )

    @property
    def position_axes(self) -> int:
        """Number of wave directions (60 degrees apart)."""
        return 3

    @property
    def position_phases(self) -> int:
        """Number of position wave vectors: axes times scales."""
        return self.position_axes * len(self.position_periods)

    @property
    def rotary_pairs(self) -> int:
        """Number of rotated dimension pairs in each attention head."""
        return self.position_phases + len(self.heading_harmonics)

    @property
    def next_state_size(self) -> int:
        """NEXT_STATE in the contract: the width of the next-state head's output."""
        return len(self.next_state_fields)


def position_wave_vectors(periods: tuple[float, ...]) -> Tensor:
    """Wave vectors in radians per metre, (3 * len(periods), 2) float32.

    Three unit directions at 0, 60 and 120 degrees (a grid-cell layout, even resolution in
    every direction), each at every period; ordered axis-major.
    """
    angles = torch.tensor([0.0, math.pi / 3.0, 2.0 * math.pi / 3.0])
    directions = torch.stack([angles.cos(), angles.sin()], dim=-1)
    rates = 2.0 * math.pi / torch.tensor(periods, dtype=torch.float32)
    return (directions[:, None, :] * rates[None, :, None]).reshape(-1, 2)


def draw_position_offset(config: ModelConfig, envs: int, generator: torch.Generator | None, device) -> Tensor:
    """A random translation per environment for the token position code, (E, 2) float32 metres.

    Uniform over a square one coarsest period wide, so the phase of every scale is
    uniform and absolute positions cannot be memorised. Draw once per match.
    """
    span = max(config.position_periods)
    uniform = torch.rand(envs, 2, generator=generator, device=generator.device if generator else "cpu")
    return ((uniform - 0.5) * span).to(device)


def next_state_target(config: ModelConfig, state, next_state) -> Tensor:
    """The regression target of the next-state head, (..., N, NEXT_STATE) float32.

    `state` and `next_state` are consecutive lean.contract.State objects (or anything with
    the same field names). Each column is the change of one field over the tick divided
    by its scale in config.next_state_fields. Rows that are dead or absent in either
    state, and ticks where the environment was re-dealt, must be masked by the caller.
    """
    columns = [
        (getattr(next_state, name) - getattr(state, name)) / scale
        for name, scale in config.next_state_fields
    ]
    return torch.stack(columns, dim=-1).float()


def rotate(tensor: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the first 2 * pairs dimensions of each head, pairing dimension i with i + pairs.

    tensor: (batch, heads, tokens, head_width). cos, sin: (batch, 1, tokens, pairs).
    Dimensions past 2 * pairs pass through unchanged.
    """
    pairs = cos.shape[-1]
    first, second, rest = tensor[..., :pairs], tensor[..., pairs : 2 * pairs], tensor[..., 2 * pairs :]
    return torch.cat([first * cos - second * sin, first * sin + second * cos, rest], dim=-1)


class Encoder(nn.Module):
    """Linear, RMSNorm, GELU, Linear to the trunk width; applied to each row on its own."""

    def __init__(self, features: int, config: ModelConfig):
        super().__init__()
        self.input = nn.Linear(features, config.width)
        self.norm = nn.RMSNorm(config.width, eps=config.norm_epsilon)
        self.output = nn.Linear(config.width, config.width)

    def forward(self, features: Tensor) -> Tensor:
        return self.output(F.gelu(self.norm(self.input(features))))


class Temporal(nn.Module):
    """Temporal sublayer: x + RG-LRU(RMSNorm(x)), an independent recurrence per token.

    The RG-LRU of Griffin (De et al. 2024): with r_t and i_t sigmoid gates of the input,
    a_t = a^(c r_t), a = sigmoid(decay), and
    h_t = a_t h_{t-1} + sqrt(1 - a_t^2) (i_t x_t).
    The decay is initialised so that a^c is uniform in [retention_low, retention_high].
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.exponent = config.recurrence_exponent
        self.norm = nn.RMSNorm(config.width, eps=config.norm_epsilon)
        self.recurrence_gate = nn.Linear(config.width, config.width)
        self.input_gate = nn.Linear(config.width, config.width)
        retention = torch.empty(config.width).uniform_(config.retention_low, config.retention_high)
        base = retention ** (1.0 / self.exponent)
        self.decay = nn.Parameter(torch.log(base / (1.0 - base)))

    def retention_at_open_gate(self) -> Tensor:
        """a^c per channel, (width,): the per-tick retention when the recurrence gate is 1."""
        return torch.sigmoid(self.decay) ** self.exponent

    def forward(self, tokens: Tensor, carry: Tensor, fresh: Tensor, keep: Tensor) -> tuple[Tensor, Tensor]:
        """Run the recurrence over the leading time axis.

        tokens: (T, E, M, width). carry: (E, M, width) float32, the state before tick 0.
        fresh: (T, E, 1, 1) float32, 0 where the carry is zeroed before that tick (an
        episode start) and 1 elsewhere. keep: (T, E, M, 1) float32, 0 for rows that are
        not attendable, whose state is held at zero. Returns the tokens with the
        recurrent state added, (T, E, M, width), and the state after the last tick.
        """
        normed = self.norm(tokens)
        # The recurrence is float32 even under bf16 autocast: 1 - a_t^2 with a_t near 1
        # has no precision left in bf16.
        recurrence = torch.sigmoid(self.recurrence_gate(normed)).float()
        gated_input = torch.sigmoid(self.input_gate(normed)).float() * normed.float()
        retention = torch.exp(self.exponent * recurrence * F.logsigmoid(self.decay.float()))
        update = torch.sqrt(1.0 - retention * retention) * gated_input
        states = []
        for tick in range(tokens.shape[0]):
            carry = (retention[tick] * (carry * fresh[tick]) + update[tick]) * keep[tick]
            states.append(carry)
        return tokens + torch.stack(states).to(tokens.dtype), carry


class Spatial(nn.Module):
    """Spatial sublayer: pre-norm masked multi-head self-attention with rotary position and
    heading on queries and keys, then a pre-norm gated MLP. The learned null token is
    appended as a key (and value) only; it is never a query and has no residual stream.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.rotate_values = config.rotate_values
        self.attention_norm = nn.RMSNorm(config.width, eps=config.norm_epsilon)
        self.query_key_value = nn.Linear(config.width, 3 * config.width, bias=False)
        self.attention_output = nn.Linear(config.width, config.width)
        self.mlp_norm = nn.RMSNorm(config.width, eps=config.norm_epsilon)
        self.mlp_input = nn.Linear(config.width, 2 * config.width)
        self.mlp_output = nn.Linear(config.width, config.width)

    def forward(self, tokens: Tensor, null: Tensor, key_mask: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """tokens: (B, M, width). null: (width,), the null token. key_mask: (B, 1, 1, M + 1)
        bool, True where a key may be attended (the null key is last). cos, sin:
        (B, 1, M + 1, pairs) rotary terms of every key; the first M are also the queries'.
        Returns (B, M, width)."""
        batch, count, width = tokens.shape
        keys_in = torch.cat([tokens, null.to(tokens.dtype).expand(batch, 1, width)], dim=1)
        projected = self.query_key_value(self.attention_norm(keys_in))
        projected = projected.view(batch, count + 1, 3, self.heads, -1)
        query, key, value = projected.permute(2, 0, 3, 1, 4)
        query_cos, query_sin = cos[:, :, :count], sin[:, :, :count]
        query = rotate(query[:, :, :count], query_cos, query_sin)
        key = rotate(key, cos, sin)
        if self.rotate_values:
            value = rotate(value, cos, sin)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=key_mask)
        if self.rotate_values:
            attended = rotate(attended, query_cos, -query_sin)
        tokens = tokens + self.attention_output(attended.transpose(1, 2).reshape(batch, count, width))
        hidden, gate = self.mlp_input(self.mlp_norm(tokens)).chunk(2, dim=-1)
        return tokens + self.mlp_output(F.gelu(hidden) * gate)


class Block(nn.Module):
    """One temporal sublayer, then `spatial_per_block` spatial sublayers."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.temporal = Temporal(config)
        self.spatial = nn.ModuleList(Spatial(config) for _ in range(config.spatial_per_block))


class Policy(nn.Module):
    """The policy and critic for one team's view (lean/CONTRACT.md).

    ship_features and global_features default to lean.contract's SHIP_FEATURES and
    GLOBAL_FEATURES. The carry is (blocks, E, 1 + N, width) float32: token 0 is the
    global token, token 1 + n is ship slot n.
    """

    def __init__(self, config: ModelConfig, ship_features: int | None = None, global_features: int | None = None):
        super().__init__()
        if ship_features is None or global_features is None:
            from lean import contract

            ship_features = contract.SHIP_FEATURES if ship_features is None else ship_features
            global_features = contract.GLOBAL_FEATURES if global_features is None else global_features
        if config.width % config.heads != 0:
            raise ValueError("width must be a multiple of heads")
        if 2 * config.rotary_pairs > config.width // config.heads:
            raise ValueError("the rotary pairs do not fit in one attention head")
        if config.positionless_rotary not in ("zero", "identity"):
            raise ValueError("positionless_rotary must be 'zero' or 'identity'")
        self.config = config
        self.ship_encoder = Encoder(ship_features + 2 * config.position_phases, config)
        self.global_encoder = Encoder(global_features, config)
        self.null_token = nn.Parameter(torch.randn(config.width))
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.blocks))
        self.final_norm = nn.RMSNorm(config.width, eps=config.norm_epsilon)
        self.policy_head = nn.Linear(config.width, config.actions)
        self.value_head = nn.Linear(config.width, config.levels * config.bins)
        self.next_state_head = nn.Linear(config.width, config.next_state_size)
        # A near-uniform first policy and an exactly uniform first value distribution.
        with torch.no_grad():
            self.policy_head.weight.mul_(0.01)
            self.policy_head.bias.zero_()
            self.value_head.weight.zero_()
            self.value_head.bias.zero_()
        self.register_buffer("wave_vectors", position_wave_vectors(config.position_periods), persistent=False)
        self.register_buffer("harmonics", torch.tensor(config.heading_harmonics), persistent=False)
        self.register_buffer(
            "bin_centres",
            value_bin_centres(torch.tensor(config.value_low), torch.tensor(config.value_high), config.bins),
            persistent=False,
        )

    def initial_carry(self, envs: int, slots: int, device) -> Tensor:
        """The zero recurrent state, (blocks, envs, 1 + slots, width) float32."""
        return torch.zeros(self.config.blocks, envs, 1 + slots, self.config.width, device=device)

    def step(self, obs, carry: Tensor, position_offset: Tensor | None = None, reset: Tensor | None = None):
        """One tick.

        obs: lean.contract.Obs with fields shaped (E, N, ...). carry: from initial_carry
        or a previous call. position_offset: optional (E, 2) metres added to positions in
        the token code only (see draw_position_offset). reset: optional (E,) bool, zero
        the carry of those environments before this tick.
        Returns policy logits (E, N, 42), value logits (E, N, LEVELS, BINS) and the new
        carry. Outputs of rows with key_mask False are meaningless and their carry is zero.
        """
        logits, value_logits, _, carry = self._forward(
            obs.ship[None],
            obs.pos[None],
            obs.heading[None],
            obs.key_mask[None],
            obs.globals[None],
            carry,
            None if reset is None else reset[None],
            position_offset,
            with_next_state=False,
        )
        return logits[0], value_logits[0], carry

    def sequence(self, obs, carry: Tensor, reset: Tensor, position_offset: Tensor | None = None):
        """T ticks for the learner; equal to T calls of step.

        obs: Obs whose fields carry a leading T, (T, E, N, ...). carry: the state before
        tick 0. reset: (T, E) bool, zero the carry before that tick. position_offset:
        optional (E, 2) or (T, E, 2) metres.
        Returns policy logits (T, E, N, 42), value logits (T, E, N, LEVELS, BINS),
        next_state (T, E, N, NEXT_STATE) (the predicted scaled change of
        config.next_state_fields to the next tick) and the carry after the last tick.
        """
        return self._forward(
            obs.ship, obs.pos, obs.heading, obs.key_mask, obs.globals, carry, reset, position_offset,
            with_next_state=True,
        )

    def _forward(self, ship, pos, heading, key_mask, globals_, carry, reset, position_offset, with_next_state):
        ticks, envs, slots = key_mask.shape
        pairs = self.config.rotary_pairs

        coded = pos if position_offset is None else pos + position_offset.unsqueeze(-2)
        # Phases are float32 even under bf16 autocast (which would cast these matmuls):
        # thousands of metres over a 64 m period is hundreds of radians, and bf16 keeps
        # 8 bits. Measured: see NOTES.md.
        with torch.autocast(pos.device.type, enabled=False):
            code_phase = coded.float() @ self.wave_vectors.T
            rotary_phase = torch.cat(
                [pos.float() @ self.wave_vectors.T, heading.float().unsqueeze(-1) * self.harmonics], dim=-1
            )
        position_code = torch.cat([code_phase.cos(), code_phase.sin()], dim=-1)
        ship_tokens = self.ship_encoder(torch.cat([ship, position_code], dim=-1))
        global_token = self.global_encoder(globals_).unsqueeze(-2)
        tokens = torch.cat([global_token, ship_tokens], dim=-2)

        # Rotary uses the true positions: only differences enter a score, so the offset
        # would cancel and only cost float32 precision.
        positionless_cos = 0.0 if self.config.positionless_rotary == "zero" else 1.0
        one_cos = rotary_phase.new_full((ticks, envs, 1, pairs), positionless_cos)
        one_sin = rotary_phase.new_zeros((ticks, envs, 1, pairs))
        cos = torch.cat([one_cos, rotary_phase.cos(), one_cos], dim=-2).flatten(0, 1).unsqueeze(1)
        sin = torch.cat([one_sin, rotary_phase.sin(), one_sin], dim=-2).flatten(0, 1).unsqueeze(1)

        always = torch.ones_like(key_mask[..., :1])
        live = torch.cat([always, key_mask], dim=-1)
        attention_mask = torch.cat([live, always], dim=-1).flatten(0, 1)[:, None, None, :]
        keep = live.unsqueeze(-1).float()
        if reset is None:
            fresh = keep.new_ones((ticks, envs, 1, 1))
        else:
            fresh = (~reset)[:, :, None, None].float()

        new_carry = []
        for index, block in enumerate(self.blocks):
            tokens, state = block.temporal(tokens, carry[index], fresh, keep)
            new_carry.append(state)
            tokens = tokens.flatten(0, 1)
            for spatial in block.spatial:
                tokens = spatial(tokens, self.null_token, attention_mask, cos, sin)
            tokens = tokens.view(ticks, envs, 1 + slots, -1)

        ship_out = self.final_norm(tokens[:, :, 1:])
        logits = self.policy_head(ship_out)
        value_logits = self.value_head(ship_out).view(ticks, envs, slots, self.config.levels, self.config.bins)
        next_state = self.next_state_head(ship_out) if with_next_state else None
        return logits, value_logits, next_state, torch.stack(new_carry)
