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
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lean import contract
from lean.model.geometry import (
    EDGE_COLUMN,
    FIELD_COLUMN,
    PAIR_COLUMNS,
    GeometryColumn,
    PairHead,
    PairQuery,
    column_centres,
)
from lean.model.position import Rotary, make_position_code
from lean.model.temporal import Temporal
from lean.model.value import COLUMNS, buffered_centres, hl_gauss_targets


def _default_position_periods() -> tuple[float, ...]:
    return tuple(
        float(period) for period in torch.logspace(math.log10(64.0), math.log10(20000.0), 6)
    )


@dataclass(frozen=True)
class NextStateColumn:
    """One column of the next-state head: a categorical over `bins` bins whose centres are
    uniform in the column's bin space (lean.model.value.symlog with `knee`, target units; 0
    linear) from `low` to `high`, the measured range, plus the buffer columns beyond each end
    (next_state_centres; lean/model/NOTES.md has the table and the measurements behind the
    ranges and knees)."""

    name: str
    knee: float
    low: float
    high: float


# The next-state columns (next_state_target). Measured from the per-tick changes of living
# ships, scripted against scripted under the training game (42.7M rows, 2026-11-25;
# lean/model/NOTES.md; docs/rewrite/sections/scoreboard/target_ranges.py): each range is
# symmetric, +-the larger measured extreme; each knee from the flat-core data rule
# (docs/rewrite/sections/scoreboard/symlog-preview.md). The shield's low end, -90, was set when
# a hit dealt about 21 (four in one tick); a hit now deals 8, so bullets take at most 32 per
# tick at 4v4 and only a collision (up to the shield held) reaches -90, rare, clamping into the
# buffer columns. Kept until the ranges are re-measured (NOTES "Open"). High end 0, knee 1. The
# buffer columns lie beyond both ends.
NEXT_STATE_COLUMNS = (
    NextStateColumn("along", 2.4e-3, -1.89, 1.89),  # metres
    NextStateColumn("across", 0.0244, -1.82, 1.82),  # metres
    NextStateColumn("heading", 7.7e-5, -0.100, 0.100),  # radians
    NextStateColumn("speed", 0.156, -5.39, 5.39),  # metres per second
    NextStateColumn("energy", 21.8, -976.0, 976.0),  # store units
    NextStateColumn("shield", 1.0, -90.0, 0.0),  # shield points
)


# The rotary reach in arena radii: twice the arena diameter (author 2026-11-25: held at a constant
# ratio of the arena; revisit if the map gets too large).
ROTARY_REACH_ARENA_RADII = 4.0


@dataclass
class ModelConfig:
    """Sizes and constants of the policy (v2's defaults where v2 has one).

    position_code: the token inputs' position code, "fourier", "grid_cells" or "v2_codes"
    (lean.model.position). position_periods: the Fourier periods and the grid spacings
    (metres; the coarsest sets the position offset's range). grid_phases: grid-cell phases
    per lattice axis (phases^2 cells per spacing). colour_period, colour_levels: v2's
    categorical code (a new v2 run's: 236,196 m, 9 levels, finest colour 12 m).
    rotary_*: the rotary bands, two axes per band (v2's layout), periods rotary_finest *
    rotary_ratio^k up to the first period above rotary_reach (metres: ROTARY_REACH_ARENA_RADII
    arena radii, twice the largest separation one band alone must tell apart, the arena
    diameter; `for_arena` sets it from the game's arena radius), each band's axes
    turned rotary_turn degrees from the finer band's (an irrational fraction of 90 degrees,
    the golden section, so no band's axes repeat another's).
    heading_harmonics: multiples of the heading angle used as rotary angles.
    rotate_values: also rotate the attention values by the key's angles and the attended sum
    back by the query's.
    next_state_columns: the next-state head's columns (NextStateColumn, next_state_target's
    names), each a categorical over `bins` bins; next_state_sigma: the HL-Gauss targets'
    standard deviation in bin widths (the geometry columns' too).
    geometry_heads: the relative-geometry heads (lean.model.geometry: the pairwise head over
    ship pairs and ship-field pairs, the map-edge column on the auxiliary heads); False (an
    ablation) builds none. pair_hidden: the pair layer's width; pair_combine: "sum" (u_i +
    v_j) or "product" (u_i x v_j); pair_columns, field_column, edge_column: the targets.
    hit_classes: the hit head's outcomes for the bullet a ship emits on a tick, in
    lean.contract's StepOut.shot_outcome order (0 missed, 1 hit an enemy, 2 hit a team-mate);
    hit_class_frequencies: their measured shares, the hit head's initial biases (log).
    """

    width: int = 128
    blocks: int = 2
    spatial_per_block: int = 2
    heads: int = 2
    map_width: int = 64
    mlp_ratio: int = 4
    head_hidden: int = 256
    bins: int = COLUMNS  # 57 across the measured range, 3 buffer columns each end
    hit_classes: int = 3  # the hit head: miss, enemy, team-mate
    # Shares of the outcomes of every bullet that ended (ghosts included), 1.6M bullets of
    # the scripted yardstick against itself, 4v4 under the training game, 48 environments x
    # 4,200 ticks re-dealt (2026-11-28, hit damage 8, the egg; lean/model/NOTES.md).
    hit_class_frequencies: tuple[float, ...] = (0.9865, 0.0110, 0.0025)
    recurrence_exponent: float = 8.0
    retention_low: float = 0.9
    retention_high: float = 0.999
    norm_epsilon: float | None = None  # torch's default: the dtype's machine epsilon (v2)
    position_code: str = "fourier"
    position_periods: tuple[float, ...] = _default_position_periods()
    grid_phases: int = (
        3  # 2 cannot tell a position from its point reflection (literature-model-choices.md)
    )
    colour_period: float = 36.0 * 3.0**8
    colour_levels: int = 9
    rotary_finest: float = 64.0
    rotary_ratio: float = 2.0
    # The default game's (lean.game.GameConfig.arena_radius 4,200 m) reach; a run builds its
    # config with `for_arena` (lean/model/tests checks this default matches the game's).
    rotary_reach: float = ROTARY_REACH_ARENA_RADII * 4200.0
    rotary_turn: float = 90.0 * (math.sqrt(5.0) - 1.0) / 2.0
    heading_harmonics: tuple[float, ...] = (1.0, 2.0)
    rotate_values: bool = True
    next_state_columns: tuple[NextStateColumn, ...] = NEXT_STATE_COLUMNS
    next_state_sigma: float = 0.75
    geometry_heads: bool = True
    pair_hidden: int = 128
    pair_combine: str = "sum"
    pair_columns: tuple[GeometryColumn, ...] = PAIR_COLUMNS
    field_column: GeometryColumn = FIELD_COLUMN
    edge_column: GeometryColumn = EDGE_COLUMN

    @classmethod
    def for_arena(cls, arena_radius: float, **fields) -> ModelConfig:
        """The config for a game with this arena radius (metres): rotary_reach
        ROTARY_REACH_ARENA_RADII x arena_radius; the other fields their defaults or `fields`."""
        return cls(rotary_reach=ROTARY_REACH_ARENA_RADII * arena_radius, **fields)

    @property
    def rotary_periods(self) -> tuple[float, ...]:
        """The rotary bands' periods in metres, finest first: up to the first above rotary_reach."""
        periods = [self.rotary_finest]
        while periods[-1] <= self.rotary_reach:
            periods.append(periods[-1] * self.rotary_ratio)
        return tuple(periods)

    @property
    def rotary_pairs(self) -> int:
        """Rotated dimension pairs in each attention head: two axes per band, then heading."""
        return 2 * len(self.rotary_periods) + len(self.heading_harmonics)

    @property
    def offset_span(self) -> float:
        """The position offset's range in metres: the position code's coarsest period."""
        return (
            self.colour_period if self.position_code == "v2_codes" else max(self.position_periods)
        )

    @property
    def next_state_size(self) -> int:
        """NEXT_STATE in the contract: the next-state head's columns."""
        return len(self.next_state_columns)


def draw_position_offset(
    config: ModelConfig, envs: int, generator: torch.Generator | None, device
) -> Tensor:
    """A random translation per environment for the token position code, (E, 2) float32 metres.

    Uniform over a square one coarsest period wide (config.offset_span), so the phase of
    every scale is uniform and absolute positions cannot be memorised. Draw once per match.
    Drawn on `device` from `generator` (torch's default generator of the device when None, as
    inside a compiled graph).
    """
    uniform = torch.rand(envs, 2, **contract.random_keywords(generator, device))
    return (uniform - 0.5) * config.offset_span


def next_state_target(
    config: ModelConfig, state, next_state, step_out, tick_length: float
) -> Tensor:
    """The next-state head's targets, (E, N, NEXT_STATE) float32 in the columns' units.

    `state` and `next_state` are consecutive lean.contract.State objects, `step_out` the
    StepOut of the step between them, `tick_length` the game's tick in seconds. Columns, by
    name (config.next_state_columns picks and orders them):
    along, across: the move against dead reckoning (the move minus the world velocity times
    the tick), in the ship's frame at the tick's start (along the nose, then to its left).
    heading: the heading's change against constant turn (minus the nose rate times the tick).
    speed, energy: the plain change of the proper speed and of the store.
    shield: the change less the scheduled recharge (StepOut.shield_restored, which the game
    gives only on an unhurt quiet tick): minus the damage the shield took, 0 on a tick
    without damage. A recharging tick is set to 0 outright: subtracting the recharge back
    out of (shield + recharge) - shield leaves float32 residues of up to a few 1e-6.
    Rows that are dead or absent in either state, and ticks where the episode ended, must
    be masked by the caller.
    """
    h = tick_length
    cos, sin = state.heading_cos, state.heading_sin
    inverse_index = torch.exp(-state.log_index)  # the world velocity is the stored one over n
    drift_x = next_state.x - state.x - state.velocity_x * inverse_index * h
    drift_y = next_state.y - state.y - state.velocity_y * inverse_index * h
    turned = torch.atan2(
        cos * next_state.heading_sin - sin * next_state.heading_cos,
        cos * next_state.heading_cos + sin * next_state.heading_sin,
    )
    columns = {
        "along": cos * drift_x + sin * drift_y,
        "across": cos * drift_y - sin * drift_x,
        "heading": turned - state.nose_rate * h,
        "speed": next_state.speed - state.speed,
        "energy": next_state.energy - state.energy,
        "shield": torch.where(
            step_out.shield_restored > 0.0, 0.0, next_state.shield - state.shield
        ),
    }
    return torch.stack([columns[column.name] for column in config.next_state_columns], -1).float()


def next_state_knees(config: ModelConfig, device=None) -> Tensor:
    """The next-state columns' knees, (NEXT_STATE,) float32 in target units (0 linear)."""
    return torch.tensor(
        [column.knee for column in config.next_state_columns], dtype=torch.float32, device=device
    )


def next_state_centres(config: ModelConfig, device=None) -> Tensor:
    """The next-state bins' centres, (NEXT_STATE, BINS) float32, in each column's bin space:
    the column's measured range plus the buffer columns at each end
    (lean.model.value.buffered_centres)."""
    low = torch.tensor([column.low for column in config.next_state_columns], dtype=torch.float32)
    high = torch.tensor([column.high for column in config.next_state_columns], dtype=torch.float32)
    return buffered_centres(low, high, next_state_knees(config), config.bins).to(device)


def next_state_distribution(
    config: ModelConfig, target: Tensor, centres: Tensor | None = None, knees: Tensor | None = None
) -> Tensor:
    """HL-Gauss target probabilities of next_state_target's columns: target (..., NEXT_STATE)
    in target units to (..., NEXT_STATE, BINS), each column in its own bin space. `centres`
    and `knees` the support already on the device (Policy's buffers), else built here."""
    return hl_gauss_targets(
        target,
        next_state_centres(config, target.device) if centres is None else centres,
        next_state_knees(config, target.device) if knees is None else knees,
        config.next_state_sigma,
    )


def next_state_cross_entropy(
    config: ModelConfig,
    logits: Tensor,
    target: Tensor,
    centres: Tensor | None = None,
    knees: Tensor | None = None,
) -> Tensor:
    """The next-state head's cross-entropy in nats per column, (..., NEXT_STATE): logits
    (..., NEXT_STATE, BINS) against the HL-Gauss distribution of target (..., NEXT_STATE)."""
    distribution = next_state_distribution(config, target, centres, knees)
    return -(distribution * torch.log_softmax(logits.float(), -1)).sum(-1)


def rotate(tensor: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the first 2 * pairs dimensions of each head, pairing dimension i with i + pairs.

    tensor: (..., head_width); cos, sin: (..., pairs), broadcast against the tensor's
    leading axes. Dimensions past 2 * pairs pass through unchanged.
    """
    pairs = cos.shape[-1]
    first, second, rest = (
        tensor[..., :pairs],
        tensor[..., pairs : 2 * pairs],
        tensor[..., 2 * pairs :],
    )
    return torch.cat([first * cos - second * sin, first * sin + second * cos, rest], dim=-1)


class KindProjection(nn.Module):
    """One token kind's first projection: gelu(norm(W x + b)), to 2 * width (v2).

    With `categories` > 0 the token also reads one category index (the ship's pending
    action): a learned row per category added to W x + b. This is a one-hot input's
    function (the row is that input's column of W) without building the one-hot: one gather
    in place of a 42-wide one-hot and 42 more input columns. The index `categories` means
    none (enemies, dead and absent slots): a zero row never trained (padding_idx), as an
    all-zero one-hot. The rows start as the one-hot columns of the Linear would (uniform
    within 1/sqrt(fan_in), the fan-in counting the categories).
    """

    def __init__(self, inputs: int, config: ModelConfig, categories: int = 0):
        super().__init__()
        self.first = nn.Linear(inputs, 2 * config.width)
        self.norm = nn.RMSNorm(2 * config.width, eps=config.norm_epsilon)
        self.category = None
        if categories:
            self.category = nn.Embedding(categories + 1, 2 * config.width, padding_idx=categories)
            bound = 1.0 / math.sqrt(inputs + categories)
            with torch.no_grad():
                self.category.weight.uniform_(-bound, bound)
                self.category.weight[categories].zero_()

    def forward(self, tokens: Tensor, category: Tensor | None = None) -> Tensor:
        projected = self.first(tokens)
        if self.category is not None:
            projected = projected + self.category(category).to(projected.dtype)
        return F.gelu(self.norm(projected))


class Encoder(nn.Module):
    """v2's Encoder without zones: a first projection per kind (ship, game, field), one
    shared Linear(2 width, width) and norm; the fields then go to the map memory width
    through a Linear without bias and a norm."""

    def __init__(self, ship_inputs: int, game_inputs: int, field_inputs: int, config: ModelConfig):
        super().__init__()
        self.ship = KindProjection(ship_inputs, config, categories=contract.JOINT)
        self.game = KindProjection(game_inputs, config)
        self.field = KindProjection(field_inputs, config)
        self.shared = nn.Linear(2 * config.width, config.width)
        self.shared_norm = nn.RMSNorm(config.width, eps=config.norm_epsilon)
        self.map_memory = nn.Linear(config.width, config.map_width, bias=False)
        self.map_memory_norm = nn.RMSNorm(config.map_width, eps=config.norm_epsilon)

    def forward(
        self, ship: Tensor, pending: Tensor, game: Tensor, field: Tensor
    ) -> tuple[Tensor, Tensor]:
        """ship (..., N, ship_inputs), pending (..., N) long (0 .. actions, `actions` for none),
        game (..., 1, game_inputs), field (..., F, field_inputs) to the trunk tokens (..., 1 +
        N, width), game first, and the map memory (..., F, map_width)."""
        tokens = self.shared_norm(
            self.shared(torch.cat([self.game(game), self.ship(ship, pending)], -2))
        )
        memory = self.map_memory_norm(
            self.map_memory(self.shared_norm(self.shared(self.field(field))))
        )
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

    Initialisation: recurrentgemma's rule, as the temporal sublayer's
    (docs/rewrite/literature-model-choices.md section 3): the input projections (query,
    key, value and gate; the map's key and value; the MLP's gate and up rows) N(0,
    1/fan_in); the output projection and the MLP's down projection N(0, s/fan_in), s = 2 /
    residual_layers.
    """

    def __init__(self, config: ModelConfig, residual_layers: int):
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

        final_scale = 2.0 / residual_layers
        with torch.no_grad():
            for layer in (self.query_key_value_gate, self.map_key_value, self.mlp_input):
                layer.weight.normal_(0.0, math.sqrt(1.0 / layer.in_features))
            self.output.weight.normal_(0.0, math.sqrt(final_scale / width))
            self.mlp_output.weight.normal_(0.0, math.sqrt(final_scale / hidden))

    def forward(
        self, tokens: Tensor, memory: Tensor, key_mask: Tensor, rotary: tuple[Tensor, ...]
    ) -> Tensor:
        """tokens (B, M, width); memory (B, F, map_width); key_mask (B, 1, 1, M + F) float,
        0 where a key may be attended and -inf elsewhere; rotary: cos and sin of the tokens (B, M,
        1, 1,
        pairs) and of the memory (B, F, 1, 1, pairs). Returns (B, M, width)."""
        token_cos, token_sin, memory_cos, memory_sin = rotary
        batch, count, width = tokens.shape
        head_width = width // self.heads
        # (B, M, [query, key, value, gate], heads, head_width): one rotation for every part
        # that takes the token's angles, one for the memory's keys (and values).
        projected = self.query_key_value_gate(self.attention_norm(tokens)).view(
            batch, count, 4, self.heads, head_width
        )
        memory = self.map_key_value(self.map_norm(memory)).view(
            batch, memory.shape[1], 2, self.heads, head_width
        )
        rotated = 3 if self.rotate_values else 2
        projected = torch.cat(
            [rotate(projected[:, :, :rotated], token_cos, token_sin), projected[:, :, rotated:]], 2
        )
        kept = rotated - 1  # the memory's key, and its value with rotate_values
        memory = torch.cat(
            [rotate(memory[:, :, :kept], memory_cos, memory_sin), memory[:, :, kept:]], 2
        )
        query, key, value, gate = projected.unbind(2)
        key = torch.cat([key, memory[:, :, 0]], 1).transpose(1, 2)
        value = torch.cat([value, memory[:, :, 1]], 1).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2), key, value, attn_mask=key_mask
        ).transpose(1, 2)
        if self.rotate_values:  # back from the keys' frames to the query's
            attended = rotate(attended, token_cos[:, :, 0], -token_sin[:, :, 0])
        attended = attended * torch.sigmoid(gate)
        tokens = tokens + self.output(attended.reshape(batch, count, width))
        gate, up = self.mlp_input(self.mlp_norm(tokens)).chunk(2, -1)
        return tokens + self.mlp_output(F.gelu(gate) * up)


class HeadGroup(nn.Module):
    """v2's HeadGroup: MLP heads reading the same token, each last(gelu(norm(first x))), the
    first projections one matrix. Initialisation: each head's block of `first` orthogonal with
    gain sqrt 2 (v2), each `last` orthogonal with gain 0.01 (near-uniform categoricals:
    docs/rewrite/literature-model-choices.md section 3), biases zero (the caller may set a
    last layer's bias)."""

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
        return [
            last(F.gelu(norm(part)))
            for part, norm, last in zip(parts, self.norms, self.lasts, strict=True)
        ]


class Block(nn.Module):
    """One temporal sublayer, then `spatial_per_block` spatial sublayers (v2's temporal_first).
    Every sublayer is one residual block of recurrentgemma's depth scale."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        residual_layers = config.blocks * (1 + config.spatial_per_block)
        self.temporal = Temporal(config, residual_layers)
        self.spatial = nn.ModuleList(
            Spatial(config, residual_layers) for _ in range(config.spatial_per_block)
        )


class Policy(nn.Module):
    """The policy and critic for one team's view (lean/CONTRACT.md).

    Inputs are lean.contract's (SHIP_FEATURES, GLOBAL_FEATURES, FIELD_FEATURES). The carry
    is (blocks, E, 1 + N, 4, width) float32: token 0 is the game token, token 1 + n ship
    slot n; per token row 0 is the RG-LRU state and rows 1..3 the convolution's last three
    inputs, oldest first. Field slots have no carry.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.width % config.heads != 0:
            raise ValueError("width must be a multiple of heads")
        if 2 * config.rotary_pairs > config.width // config.heads:
            raise ValueError("the rotary pairs do not fit in one attention head")
        self.config = config
        self.position_code = make_position_code(config)
        self.rotary = Rotary(config)
        code = self.position_code.features
        self.encoder = Encoder(
            contract.SHIP_FEATURES + FOG_FEATURES + code,
            contract.GLOBAL_FEATURES + code,
            contract.FIELD_FEATURES + code,
            config,
        )
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.blocks))
        self.heads = HeadGroup(
            config.width,
            config.head_hidden,
            (contract.JOINT, contract.LEVELS * config.bins),
            config,
        )
        # The learner's auxiliary heads, one group: next state (a categorical per column), hit
        # and, with the geometry heads, the map edge.
        auxiliary = (config.next_state_size * config.bins, config.hit_classes)
        if config.geometry_heads:
            auxiliary += (config.edge_column.bins,)
        self.auxiliary_heads = HeadGroup(config.width, config.head_hidden, auxiliary, config)
        self.pair_head = PairHead(config) if config.geometry_heads else None
        # The geometry columns' supports, built once (as the next-state support).
        if config.geometry_heads:
            for column in (*config.pair_columns, config.field_column, config.edge_column):
                self.register_buffer(
                    f"geometry_centres_{column.name}", column_centres(column), persistent=False
                )
        # The next-state support on the device, built once (a table made inside a compiled
        # graph is a host-to-device copy at every call); not in the state dict.
        self.register_buffer("next_state_centres", next_state_centres(config), persistent=False)
        self.register_buffer("next_state_knees", next_state_knees(config), persistent=False)
        # The hit head starts at the measured class shares (Lin et al.'s prior bias).
        frequencies = torch.tensor(config.hit_class_frequencies, dtype=torch.float32)
        with torch.no_grad():
            self.auxiliary_heads.lasts[1].bias.copy_(torch.log(frequencies / frequencies.sum()))

    def initial_carry(self, envs: int, slots: int, device) -> Tensor:
        """The zero recurrent state, (blocks, envs, 1 + slots, 4, width) float32."""
        return torch.zeros(self.config.blocks, envs, 1 + slots, 4, self.config.width, device=device)

    def step(
        self, obs, carry: Tensor, position_offset: Tensor | None = None, reset: Tensor | None = None
    ):
        """One tick.

        obs: lean.contract.Obs with fields shaped (E, ...). carry: from initial_carry or a
        previous call. position_offset: optional (E, 2) metres added to positions in the
        token code only (see draw_position_offset). reset: optional (E,) bool, zero the
        carry of those environments before this tick.
        Returns policy logits (E, N, 42), value logits (E, N, LEVELS, BINS) and the new
        carry. Outputs of rows with key_mask False are meaningless and their carry is zero.
        """
        logits, value_logits, _, _, carry, *_ = self._forward(
            {name: getattr(obs, name)[None] for name in OBS_FIELDS},
            carry,
            None if reset is None else reset[None],
            None if position_offset is None else position_offset[None],
            with_auxiliary=False,
        )
        return logits[0], value_logits[0], carry

    def geometry_centres(self, column: GeometryColumn) -> Tensor:
        """The geometry column's bin-space centres on the device (lean.model.geometry)."""
        return getattr(self, f"geometry_centres_{column.name}")

    def sequence(
        self,
        obs,
        carry: Tensor,
        reset: Tensor,
        position_offset: Tensor | None = None,
        pairs: PairQuery | None = None,
    ) -> Sequence:
        """T ticks for the learner; equal to T calls of step.

        obs: Obs whose fields carry a leading T, (T, E, ...). carry: the state before tick
        0. reset: (T, E) bool, zero the carry before that tick. position_offset: optional
        (E, 2) or (T, E, 2) metres.
        pairs: the pairs the pairwise head evaluates (lean.model.geometry.PairQuery), or None.
        Returns a Sequence: policy logits (T, E, N, 42), value logits (T, E, N, LEVELS,
        BINS), next-state logits (T, E, N, NEXT_STATE, BINS) (a categorical per column of
        next_state_target, over the bins of next_state_centres), hit logits (T, E, N,
        HIT_CLASSES) (the outcome of the bullet each ship emits on that tick), the carry
        after the last tick; with the geometry heads the map-edge logits (T, E, N,
        edge bins) and, given `pairs`, the pairwise head's logits (PairHead.forward).
        """
        if position_offset is not None and position_offset.dim() == 2:
            position_offset = position_offset[None]
        return self._forward(
            {name: getattr(obs, name) for name in OBS_FIELDS},
            carry,
            reset,
            position_offset,
            with_auxiliary=True,
            query=pairs,
        )

    def _forward(self, obs: dict, carry, reset, position_offset, with_auxiliary, query=None):
        ticks, envs, slots = obs["key_mask"].shape
        pos, field_pos = obs["pos"], obs["field_pos"]
        offset = (
            pos.new_zeros(ticks, envs, 2)
            if position_offset is None
            else position_offset.expand(ticks, envs, 2)
        )
        offset = offset.unsqueeze(-2)

        # Token inputs: the features and the position code of the offset positions; the game
        # token reads the arena centre's code (the origin, offset like any token; v2's inputs.py).
        code = self.position_code
        # Fog (C10): whether the row is seen this tick and how long since it was (symlog
        # seconds); with fog off every row is visible and current.
        dtype = obs["ship"].dtype
        seen = [obs["visible"].to(dtype)[..., None], obs["since_seen"].to(dtype)[..., None]]
        ship_in = torch.cat([obs["ship"], *seen, code(pos + offset).to(dtype)], -1)
        game_in = torch.cat(
            [obs["globals"].unsqueeze(-2), code(offset).to(obs["globals"].dtype)], -1
        )
        field_in = torch.cat([obs["field"], code(field_pos + offset).to(obs["field"].dtype)], -1)
        tokens, field_memory = self.encoder(ship_in, obs["pending"], game_in, field_in)
        memory = field_memory.flatten(0, 1)

        # Rotary uses the true positions, in which the arena centre (the boundary's) is the
        # origin: only differences enter a ship-ship score. The game token sits at the centre
        # with heading 0 (the identity rotation); a field at its centre, heading towards the
        # arena centre (so the field's terms keep the arena's rotational symmetry).
        ship_angles = self.rotary.angles(pos, obs["heading"])
        pairs = ship_angles.shape[-1]
        game_cos = ship_angles.new_ones((ticks, envs, 1, pairs))
        game_sin = ship_angles.new_zeros((ticks, envs, 1, pairs))
        field_angles = self.rotary.angles(field_pos, field_heading(field_pos))
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
        every_field = live.new_ones(live.shape[:-1] + (field_pos.shape[-2],))  # always present
        attendable = torch.cat([live, every_field], -1).flatten(0, 1)[:, None, None, :]
        # Additive, built once: a boolean mask costs a conversion in every sublayer.
        attention_mask = torch.zeros(attendable.shape, device=attendable.device).masked_fill(
            ~attendable, -math.inf
        )
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
        value_logits = value_logits.view(ticks, envs, slots, contract.LEVELS, self.config.bins)
        next_state = hit_logits = edge_logits = pair_logits = field_logits = None
        if with_auxiliary:
            next_state, hit_logits, *edge = self.auxiliary_heads(ships)
            next_state = next_state.view(
                ticks, envs, slots, self.config.next_state_size, self.config.bins
            )
            edge_logits = edge[0] if edge else None
            if query is not None and self.pair_head is not None:
                pair_logits, field_logits = self.pair_head(ships, field_memory, query)
        return Sequence(
            logits,
            value_logits,
            next_state,
            hit_logits,
            torch.stack(new_carry),
            edge_logits,
            pair_logits,
            field_logits,
        )


class Sequence(NamedTuple):
    """Policy.sequence's outputs (step keeps the policy and value logits and the carry)."""

    logits: Tensor
    value_logits: Tensor
    next_state: Tensor | None
    hit: Tensor | None
    carry: Tensor
    edge: Tensor | None = None
    pairs: list[Tensor] | None = None
    fields: Tensor | None = None


def field_heading(field_pos: Tensor) -> Tensor:
    """A field's rotary heading, (..., F) radians: from its centre towards the arena centre
    (the origin; atan2 gives -pi for a field at the origin, where every heading is as good)."""
    return torch.atan2(-field_pos[..., 1], -field_pos[..., 0])


OBS_FIELDS = (
    "ship",
    "pos",
    "heading",
    "key_mask",
    "globals",
    "field",
    "field_pos",
    "visible",
    "since_seen",
    "pending",
)
FOG_FEATURES = 2  # visible, since_seen
