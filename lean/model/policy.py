"""The policy: v2's Yemong trunk without beliefs (src/bnb/models/yemong/): an encoder with one
first projection per token kind and a shared second; blocks of one temporal sublayer
(Griffin's recurrent block) and two spatial sublayers (gated rotary attention of the ship
and game tokens over themselves and the fields, then a gated MLP); per-ship MLP heads.

Sizes: T ticks, E environments, N ship slots, M = 1 + N trunk tokens (the game token first,
then the ship slots in the caller's order), F field slots (keys and values only). The model
has no parameter or buffer whose shape depends on N or F, and nothing is computed across the
entity axis except attention.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lean.model.position import Rotary, make_position_code
from lean.model.temporal import Temporal


def _default_position_periods() -> tuple[float, ...]:
    return tuple(float(period) for period in torch.logspace(math.log10(64.0), math.log10(20000.0), 6))


@dataclass
class ModelConfig:
    """Sizes and constants of the policy (v2's defaults where v2 has one).

    position_code: the token inputs' position code, "fourier", "grid_cells" or "v2_codes"
    (lean.model.position). position_periods: the Fourier periods and the grid spacings
    (metres; the coarsest sets the position offset's range). grid_phases: grid-cell phases
    per lattice axis (phases^2 cells per spacing). colour_period, colour_levels: v2's
    categorical code (a new v2 run's: 236,196 m, 9 levels, finest colour 12 m).
    rotary_*: the rotary bands, v2's layout (two axes per band, each band's axes turned
    rotary_turn degrees from the coarser one's, periods rotary_finest * rotary_ratio^k),
    keeping only the bands up to a little above the arena's diameter (8.4 km).
    heading_harmonics: multiples of the heading angle used as rotary angles.
    rotate_values: also rotate the attention values by the key's angles and the attended sum
    back by the query's.
    positionless_rotary: the game token's rotary angles. "centre" (v2): the arena centre
    (the origin) with heading 0, the identity rotation. "zero": cos = sin = 0, which removes
    the rotated dimensions from every score and value involving the game token.
    next_state_fields: the State fields whose one-tick change the next-state head predicts,
    each with the divisor that brings the change to order one.
    """

    width: int = 128
    blocks: int = 2
    spatial_per_block: int = 2
    heads: int = 2
    map_width: int = 64
    mlp_ratio: int = 4
    head_hidden: int = 256
    actions: int = 42
    levels: int = 4
    bins: int = 64
    recurrence_exponent: float = 8.0
    retention_low: float = 0.9
    retention_high: float = 0.999
    norm_epsilon: float | None = None  # torch's default: the dtype's machine epsilon (v2)
    position_code: str = "fourier"
    position_periods: tuple[float, ...] = _default_position_periods()
    grid_phases: int = 2
    colour_period: float = 36.0 * 3.0**8
    colour_levels: int = 9
    rotary_finest: float = 64.0
    rotary_ratio: float = 3.405
    rotary_bands: int = 5
    rotary_turn: float = 35.0
    heading_harmonics: tuple[float, ...] = (1.0, 2.0)
    rotate_values: bool = True
    positionless_rotary: str = "centre"
    next_state_fields: tuple[tuple[str, float], ...] = (
        ("x", 10.0),
        ("y", 10.0),
        ("heading_cos", 0.1),
        ("heading_sin", 0.1),
        ("speed", 5.0),
        ("energy", 400.0),
        ("shield", 10.0),
    )

    @property
    def rotary_periods(self) -> tuple[float, ...]:
        """The rotary bands' periods in metres, finest first."""
        return tuple(self.rotary_finest * self.rotary_ratio**band for band in range(self.rotary_bands))

    @property
    def rotary_pairs(self) -> int:
        """Rotated dimension pairs in each attention head: two axes per band, then heading."""
        return 2 * self.rotary_bands + len(self.heading_harmonics)

    @property
    def offset_span(self) -> float:
        """The position offset's range in metres: the position code's coarsest period."""
        return self.colour_period if self.position_code == "v2_codes" else max(self.position_periods)

    @property
    def next_state_size(self) -> int:
        """NEXT_STATE in the contract: the width of the next-state head's output."""
        return len(self.next_state_fields)


def draw_position_offset(config: ModelConfig, envs: int, generator: torch.Generator | None, device) -> Tensor:
    """A random translation per environment for the token position code, (E, 2) float32 metres.

    Uniform over a square one coarsest period wide (config.offset_span), so the phase of
    every scale is uniform and absolute positions cannot be memorised. Draw once per match.
    """
    uniform = torch.rand(envs, 2, generator=generator, device=generator.device if generator else "cpu")
    return ((uniform - 0.5) * config.offset_span).to(device)


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

    tensor: (..., head_width); cos, sin: (..., pairs), broadcast against the tensor's
    leading axes. Dimensions past 2 * pairs pass through unchanged.
    """
    pairs = cos.shape[-1]
    first, second, rest = tensor[..., :pairs], tensor[..., pairs : 2 * pairs], tensor[..., 2 * pairs :]
    return torch.cat([first * cos - second * sin, first * sin + second * cos, rest], dim=-1)


class KindProjection(nn.Module):
    """One token kind's first projection: gelu(norm(W x + b)), to 2 * width (v2)."""

    def __init__(self, inputs: int, config: ModelConfig):
        super().__init__()
        self.first = nn.Linear(inputs, 2 * config.width)
        self.norm = nn.RMSNorm(2 * config.width, eps=config.norm_epsilon)

    def forward(self, tokens: Tensor) -> Tensor:
        return F.gelu(self.norm(self.first(tokens)))


class Encoder(nn.Module):
    """v2's Encoder without zones: a first projection per kind (ship, game, field), one
    shared Linear(2 width, width) and norm; the fields then go to the map memory width
    through a Linear without bias and a norm."""

    def __init__(self, ship_inputs: int, game_inputs: int, field_inputs: int, config: ModelConfig):
        super().__init__()
        self.ship = KindProjection(ship_inputs, config)
        self.game = KindProjection(game_inputs, config)
        self.field = KindProjection(field_inputs, config)
        self.shared = nn.Linear(2 * config.width, config.width)
        self.shared_norm = nn.RMSNorm(config.width, eps=config.norm_epsilon)
        self.map_memory = nn.Linear(config.width, config.map_width, bias=False)
        self.map_memory_norm = nn.RMSNorm(config.map_width, eps=config.norm_epsilon)

    def forward(self, ship: Tensor, game: Tensor, field: Tensor) -> tuple[Tensor, Tensor]:
        """ship (..., N, ship_inputs), game (..., 1, game_inputs), field (..., F, field_inputs)
        to the trunk tokens (..., 1 + N, width), game first, and the map memory (..., F,
        map_width)."""
        tokens = self.shared_norm(self.shared(torch.cat([self.game(game), self.ship(ship)], -2)))
        memory = self.map_memory_norm(self.map_memory(self.shared_norm(self.shared(self.field(field)))))
        return tokens, memory


class Spatial(nn.Module):
    """Spatial sublayer (v2's SpatialBlock): pre-norm rotary attention of the trunk tokens
    over themselves and the map memory in one softmax, an element-wise output gate, then a
    pre-norm gated MLP of 4 x width. No biases.

    The gate (Qiu et al. 2025, position G1): each head's attention output times
    sigmoid(W x), x the normed query token, before the output projection. W is extra output
    columns of the query projection with no bias, so it starts as the query projection does
    (centred on 0.5), as the released code does (docs/rewrite/rotary-and-gate-check.md).
    The map memory has its own norm and key/value projection; it is never a query.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        width, hidden = config.width, config.mlp_ratio * config.width
        self.heads = config.heads
        self.rotate_values = config.rotate_values
        self.attention_norm = nn.RMSNorm(width, eps=config.norm_epsilon)
        self.query_key_value_gate = nn.Linear(width, 4 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.map_norm = nn.RMSNorm(config.map_width, eps=config.norm_epsilon)
        self.map_key_value = nn.Linear(config.map_width, 2 * width, bias=False)
        self.mlp_norm = nn.RMSNorm(width, eps=config.norm_epsilon)
        self.mlp_input = nn.Linear(width, 2 * hidden, bias=False)  # gate rows, then up rows
        self.mlp_output = nn.Linear(hidden, width, bias=False)

    def forward(self, tokens: Tensor, memory: Tensor, key_mask: Tensor, rotary: tuple[Tensor, ...]) -> Tensor:
        """tokens (B, M, width); memory (B, F, map_width); key_mask (B, 1, 1, M + F) float,
        0 where a key may be attended and -inf elsewhere; rotary: cos and sin of the tokens (B, M, 1, 1,
        pairs) and of the memory (B, F, 1, 1, pairs). Returns (B, M, width)."""
        token_cos, token_sin, memory_cos, memory_sin = rotary
        batch, count, width = tokens.shape
        head_width = width // self.heads
        # (B, M, [query, key, value, gate], heads, head_width): one rotation for every part
        # that takes the token's angles, one for the memory's keys (and values).
        projected = self.query_key_value_gate(self.attention_norm(tokens)).view(batch, count, 4, self.heads, head_width)
        memory = self.map_key_value(self.map_norm(memory)).view(batch, memory.shape[1], 2, self.heads, head_width)
        rotated = 3 if self.rotate_values else 2
        projected = torch.cat([rotate(projected[:, :, :rotated], token_cos, token_sin), projected[:, :, rotated:]], 2)
        kept = rotated - 1  # the memory's key, and its value with rotate_values
        memory = torch.cat([rotate(memory[:, :, :kept], memory_cos, memory_sin), memory[:, :, kept:]], 2)
        query, key, value, gate = projected.unbind(2)
        key = torch.cat([key, memory[:, :, 0]], 1).transpose(1, 2)
        value = torch.cat([value, memory[:, :, 1]], 1).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query.transpose(1, 2), key, value, attn_mask=key_mask).transpose(1, 2)
        if self.rotate_values:  # back from the keys' frames to the query's
            attended = rotate(attended, token_cos[:, :, 0], -token_sin[:, :, 0])
        attended = attended * torch.sigmoid(gate)
        tokens = tokens + self.output(attended.reshape(batch, count, width))
        gate, up = self.mlp_input(self.mlp_norm(tokens)).chunk(2, -1)
        return tokens + self.mlp_output(F.gelu(gate) * up)


class HeadGroup(nn.Module):
    """v2's HeadGroup: MLP heads reading the same token, each last(gelu(norm(first x))), the
    first projections one matrix. Initialisation (v2 spec 7): each head's block of `first`
    orthogonal with gain sqrt 2, each `last` orthogonal with gain 0.01, biases zero."""

    def __init__(self, inputs: int, hidden: int, outputs: tuple[int, ...], config: ModelConfig):
        super().__init__()
        self.hidden = hidden
        self.first = nn.Linear(inputs, hidden * len(outputs))
        self.norms = nn.ModuleList(nn.RMSNorm(hidden, eps=config.norm_epsilon) for _ in outputs)
        self.lasts = nn.ModuleList(nn.Linear(hidden, size) for size in outputs)
        with torch.no_grad():
            for block in self.first.weight.split(hidden):
                nn.init.orthogonal_(block, math.sqrt(2.0))
            self.first.bias.zero_()
            for last in self.lasts:
                nn.init.orthogonal_(last.weight, 0.01)
                last.bias.zero_()

    def forward(self, inputs: Tensor) -> list[Tensor]:
        parts = self.first(inputs).split(self.hidden, -1)
        return [last(F.gelu(norm(part))) for part, norm, last in zip(parts, self.norms, self.lasts, strict=True)]


class Block(nn.Module):
    """One temporal sublayer, then `spatial_per_block` spatial sublayers (v2's temporal_first)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.temporal = Temporal(config, residual_layers=config.blocks * (1 + config.spatial_per_block))
        self.spatial = nn.ModuleList(Spatial(config) for _ in range(config.spatial_per_block))


class Policy(nn.Module):
    """The policy and critic for one team's view (lean/CONTRACT.md).

    ship_features, global_features and field_features default to lean.contract's. The carry
    is (blocks, E, 1 + N, 4, width) float32: token 0 is the game token, token 1 + n ship
    slot n; per token row 0 is the RG-LRU state and rows 1..3 the convolution's last three
    inputs, oldest first. Field slots have no carry.
    """

    def __init__(
        self,
        config: ModelConfig,
        ship_features: int | None = None,
        global_features: int | None = None,
        field_features: int | None = None,
    ):
        super().__init__()
        from lean import contract

        ship_features = contract.SHIP_FEATURES if ship_features is None else ship_features
        global_features = contract.GLOBAL_FEATURES if global_features is None else global_features
        field_features = contract.FIELD_FEATURES if field_features is None else field_features
        if config.width % config.heads != 0:
            raise ValueError("width must be a multiple of heads")
        if 2 * config.rotary_pairs > config.width // config.heads:
            raise ValueError("the rotary pairs do not fit in one attention head")
        if config.positionless_rotary not in ("centre", "zero"):
            raise ValueError("positionless_rotary must be 'centre' or 'zero'")
        self.config = config
        self.position_code = make_position_code(config)
        self.rotary = Rotary(config)
        code = self.position_code.features
        self.encoder = Encoder(ship_features + code, global_features + code, field_features + code, config)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.blocks))
        self.heads = HeadGroup(config.width, config.head_hidden, (config.actions, config.levels * config.bins), config)
        self.next_state_head = HeadGroup(config.width, config.head_hidden, (config.next_state_size,), config)

    def initial_carry(self, envs: int, slots: int, device) -> Tensor:
        """The zero recurrent state, (blocks, envs, 1 + slots, 4, width) float32."""
        return torch.zeros(self.config.blocks, envs, 1 + slots, 4, self.config.width, device=device)

    def step(self, obs, carry: Tensor, position_offset: Tensor | None = None, reset: Tensor | None = None):
        """One tick.

        obs: lean.contract.Obs with fields shaped (E, ...). carry: from initial_carry or a
        previous call. position_offset: optional (E, 2) metres added to positions in the
        token code only (see draw_position_offset). reset: optional (E,) bool, zero the
        carry of those environments before this tick.
        Returns policy logits (E, N, 42), value logits (E, N, LEVELS, BINS) and the new
        carry. Outputs of rows with key_mask False are meaningless and their carry is zero.
        """
        logits, value_logits, _, carry = self._forward(
            {name: getattr(obs, name)[None] for name in OBS_FIELDS},
            carry,
            None if reset is None else reset[None],
            None if position_offset is None else position_offset[None],
            with_next_state=False,
        )
        return logits[0], value_logits[0], carry

    def sequence(self, obs, carry: Tensor, reset: Tensor, position_offset: Tensor | None = None):
        """T ticks for the learner; equal to T calls of step.

        obs: Obs whose fields carry a leading T, (T, E, ...). carry: the state before tick
        0. reset: (T, E) bool, zero the carry before that tick. position_offset: optional
        (E, 2) or (T, E, 2) metres.
        Returns policy logits (T, E, N, 42), value logits (T, E, N, LEVELS, BINS),
        next_state (T, E, N, NEXT_STATE) (the predicted scaled change of
        config.next_state_fields to the next tick) and the carry after the last tick.
        """
        if position_offset is not None and position_offset.dim() == 2:
            position_offset = position_offset[None]
        return self._forward(
            {name: getattr(obs, name) for name in OBS_FIELDS}, carry, reset, position_offset, with_next_state=True
        )

    def _forward(self, obs: dict, carry, reset, position_offset, with_next_state):
        ticks, envs, slots = obs["key_mask"].shape
        pos, field_pos = obs["pos"], obs["field_pos"]
        offset = pos.new_zeros(ticks, envs, 2) if position_offset is None else position_offset.expand(ticks, envs, 2)
        offset = offset.unsqueeze(-2)

        # Token inputs: the features and the position code of the offset positions; the game
        # token reads the arena centre's code (v2's inputs.py).
        code = self.position_code
        ship_in = torch.cat([obs["ship"], code(pos + offset).to(obs["ship"].dtype)], -1)
        game_in = torch.cat([obs["globals"].unsqueeze(-2), code(offset).to(obs["globals"].dtype)], -1)
        field_in = torch.cat([obs["field"], code(field_pos + offset).to(obs["field"].dtype)], -1)
        tokens, memory = self.encoder(ship_in, game_in, field_in)
        memory = memory.flatten(0, 1)

        # Rotary uses the true positions: only differences enter a ship-ship score, and the
        # game token sits at the arena centre (the origin) or nowhere ("zero"). Fields take
        # their centre and heading 0.
        ship_angles = self.rotary.angles(pos, obs["heading"])
        pairs = ship_angles.shape[-1]
        centre = 1.0 if self.config.positionless_rotary == "centre" else 0.0
        game_cos = ship_angles.new_full((ticks, envs, 1, pairs), centre)
        game_sin = ship_angles.new_zeros((ticks, envs, 1, pairs))
        field_angles = self.rotary.position_angles(field_pos)
        rotary = tuple(
            part.flatten(0, 1)[:, :, None, None, :]
            for part in (
                torch.cat([game_cos, ship_angles.cos()], -2),
                torch.cat([game_sin, ship_angles.sin()], -2),
                field_angles.cos(),
                field_angles.sin(),
            )
        )

        live = torch.cat([torch.ones_like(obs["key_mask"][..., :1]), obs["key_mask"]], -1)
        attendable = torch.cat([live, obs["field_mask"]], -1).flatten(0, 1)[:, None, None, :]
        # Additive, built once: a boolean mask costs a conversion in every sublayer.
        attention_mask = torch.zeros(attendable.shape, device=attendable.device).masked_fill(~attendable, -math.inf)
        keep = live.unsqueeze(-1).float()
        if reset is None:
            fresh = keep.new_ones((ticks, envs, 1, 1))
        else:
            fresh = (~reset)[:, :, None, None].float()

        new_carry = []
        for index, block in enumerate(self.blocks):
            tokens, block_carry = block.temporal(tokens, carry[index], fresh, keep)
            new_carry.append(block_carry)
            tokens = tokens.flatten(0, 1)
            for spatial in block.spatial:
                tokens = spatial(tokens, memory, attention_mask, rotary)
            tokens = tokens.view(ticks, envs, 1 + slots, -1)

        ships = tokens[:, :, 1:]
        logits, value_logits = self.heads(ships)
        value_logits = value_logits.view(ticks, envs, slots, self.config.levels, self.config.bins)
        next_state = self.next_state_head(ships)[0] if with_next_state else None
        return logits, value_logits, next_state, torch.stack(new_carry)


OBS_FIELDS = ("ship", "pos", "heading", "key_mask", "globals", "field", "field_pos", "field_mask")
