"""Yemong: one decision (`step`) and a stored segment (`unroll`) over one team's views.

The model owns its inputs: the belief, the pending actions and the view's position offset
live in its carry, and it composes its tokens from the raw observation. `step` returns a
replay record (what it read from the carry), so `unroll` recomputes a segment in parallel
without replaying the belief. `unroll` equals a loop of `step` over the same segment.

`step` skips the rows nothing reads: the enemy-action and next-state heads run on the enemy
rows only (own ships are always visible, so truth replaces their forecast on the next tick).
`unroll` computes every row, for the learner.
"""

import dataclasses
import math

import torch
import torch.nn as nn

from bnb.contracts import Obs
from bnb.contracts.action import NUM_JOINT
from bnb.contracts.obs import SHIP
from bnb.contracts.tree import contract
from bnb.game.rules import Rules
from bnb.game.ship_types import ShipTypes

from . import belief as beliefs
from . import critic, density, shots
from .belief import Belief
from .heads import HeadGroup, ShotTrajectoryHead, critic_values, next_state_logits
from .inputs import (
    FIELD_INPUTS,
    GLOBAL_INPUTS,
    POSITION_CODE,
    STORED_DTYPE,
    Replay,
    Tokens,
    compose,
    ship_inputs,
    truth,
)
from .layers import rotary_tables
from .network import Encoder, Trunk
from .state_code import MEANS, POSITION_LEVELS, POSITION_PERIOD, ShipStateCode, velocity_reach

SHOT_HARMONICS = 4  # the trajectory head's age features


@dataclasses.dataclass(frozen=True)
class YemongConfig:
    """Production defaults (v1's known-good run). Nothing in the weights depends on the
    numbers of ships or fields or on the arena.

    levels: the reward levels in their configured order; outcome_level names the one read
    off the global token (the others are per-ship categorical values).
    rotary_period: the coarsest period (m) of the position frequencies (2 pi / p) 2^k,
    k < rotary_octaves; heading_harmonics: the heading's.
    bfloat16_matmuls: run the matrix products of the encoder, the trunk and the heads under
    bf16 autocast. Off by default. On, `step` and `unroll` agree only to bf16 rounding (the
    fix-round notes measure it), where float32 gives float32 rounding.
    """

    width: int = 128
    blocks: int = 2
    spatial_per_block: int = 2
    heads: int = 2
    map_width: int = 64
    value_bins: int = 51
    value_limit: float = 2.0
    levels: tuple[str, ...] = ("reward", "outcome")
    outcome_level: str = "outcome"
    hex_rings: int = 10
    presence_radius: float = 150.0  # m
    rotary_period: float = 8192.0  # m, the coarsest
    rotary_octaves: int = 8
    heading_harmonics: tuple[int, ...] = (1, 2, 4, 8)
    bfloat16_matmuls: bool = False

    def __post_init__(self) -> None:
        if self.outcome_level not in self.levels:
            raise ValueError(f"outcome level {self.outcome_level!r} is not in {self.levels}")
        if len(self.levels) < 2:
            raise ValueError("the critic needs at least one per-ship level besides the outcome")
        pairs = 2 * self.rotary_octaves + len(self.heading_harmonics)
        if 2 * pairs > self.width // self.heads:
            raise ValueError(f"{pairs} rotary pairs do not fit a head of "
                             f"{self.width // self.heads}")

    @property
    def local_levels(self) -> tuple[int, ...]:
        return tuple(i for i, name in enumerate(self.levels) if name != self.outcome_level)


@contract
class Carry:
    """Per view (R leading). state (R, blocks, Q, D) and convolution_buffer
    (R, blocks, Q, 3, D): the trunk's; belief: per ship row, positions in the shifted frame;
    previous_action (R, S) int32, the own ships' last joint actions; offset (R, 2) m, the
    view's position offset, drawn uniformly in [0, P) at every reset; nonfinite (R,) int64,
    a running count of non-finite forecast entries (replaced by zeros)."""

    state: torch.Tensor
    convolution_buffer: torch.Tensor
    belief: Belief
    previous_action: torch.Tensor
    offset: torch.Tensor
    nonfinite: torch.Tensor


@contract
class Decision:
    """One step's outputs, per view: policy_logits (R, S, 42); enemy_action_logits and
    next_state_logits (R, N - S, ...) for the enemy rows; values (R, S, K) in the configured
    level order; outcome_probabilities (R, 4); the next carry (previous_action not yet
    replaced by the sample); the replay record."""

    policy_logits: torch.Tensor
    enemy_action_logits: torch.Tensor
    next_state_logits: torch.Tensor
    values: torch.Tensor
    outcome_probabilities: torch.Tensor
    carry: Carry
    replay: Replay


@contract
class Unrolled:
    """A segment's outputs, (L, R, ...) leading: policy_logits (S rows), enemy_action_logits
    and next_state_logits (every row), value_logits (S, K - 1, bins), values (S, K),
    outcome_logits (4), density_log_rates (2C: ally cells, then enemy cells),
    shot_outcome_logits (N, 19) and shot_latents (N, D) for `Yemong.shot_trajectory`."""

    policy_logits: torch.Tensor
    enemy_action_logits: torch.Tensor
    next_state_logits: torch.Tensor
    value_logits: torch.Tensor
    values: torch.Tensor
    outcome_logits: torch.Tensor
    density_log_rates: torch.Tensor
    shot_outcome_logits: torch.Tensor
    shot_latents: torch.Tensor


BELIEF_FIELDS = tuple(field.name for field in dataclasses.fields(Belief))  # (traced as constants)


def _rows(belief: Belief, rows: slice) -> Belief:
    return Belief(**{name: getattr(belief, name)[:, rows] for name in BELIEF_FIELDS})


def _nearest(position: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """The representative of position (mod P) nearest to reference: decoded positions come
    back in [0, P), and the belief's frame is continuous."""
    return position + POSITION_PERIOD * torch.round((reference - position) / POSITION_PERIOD)


class Yemong(nn.Module):
    """rules and types give the game's constants (tick, stall angle, shield delay, the ship
    card); rules.num_ships only sets the density head's starting rate."""

    def __init__(self, config: YemongConfig, rules: Rules, types: ShipTypes) -> None:
        super().__init__()
        self.config = config
        width, hidden = config.width, 2 * config.width
        self.ship_types = types.max_shield.shape[-1]
        self.tick = rules.tick_length
        self.state_code = ShipStateCode(rules, types)
        self.shot_code = shots.shot_code_parameters(
            float(types.muzzle_speed.max()), velocity_reach(),
            rules.bullet_moves * rules.tick_length, float(types.hit_radius.min()))
        self.bullet_moves = rules.bullet_moves
        self.encoder = Encoder(ship_inputs(self.state_code.width, self.ship_types), GLOBAL_INPUTS,
                               FIELD_INPUTS, width, config.map_width)
        self.trunk = Trunk(width, config.heads, config.map_width, config.blocks,
                           config.spatial_per_block)
        levels = len(config.levels) - 1
        # Heads grouped by the token they read: own ships, every ship, the global token.
        self.own_heads = HeadGroup(width, hidden, (NUM_JOINT, levels * config.value_bins))
        self.ship_heads = HeadGroup(width, hidden, (NUM_JOINT, self.state_code.width))
        self.outcome = HeadGroup(width, hidden, (critic.OUTCOMES,))
        cells = density.cell_count(config.hex_rings)
        self.density = HeadGroup(width, hidden, (2 * cells,))
        self.shot_outcome = HeadGroup(width, hidden, (self.shot_code.outcome_width,))
        self.shot_trajectory_head = ShotTrajectoryHead(width, 1 + 2 * SHOT_HARMONICS,
                                                       self.shot_code.trajectory_width)
        self._initialise(rules.num_ships, cells)
        # Frequencies (2 pi / period) 2^k, held as their periods (exact powers of two metres).
        octaves = 2.0 ** torch.arange(config.rotary_octaves, dtype=torch.float32)
        self.register_buffer("position_periods", config.rotary_period / octaves,
                             persistent=False)
        self.register_buffer("heading_frequencies",
                             torch.tensor(config.heading_harmonics, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("bins", critic.value_bins(config.value_bins, config.value_limit),
                             persistent=False)

    def _initialise(self, num_ships: int, cells: int) -> None:
        """Code columns of each first projection ~ N(0, 1 / groups); the next-state head
        starts as "nothing changes"; the density head at an even spread of the ships."""
        with torch.no_grad():
            self.encoder.ship.first.weight[:, :self.state_code.width].normal_(
                0.0, self.state_code.groups**-0.5)
            for kind in (self.encoder.field, self.encoder.global_token):
                kind.first.weight[:, :POSITION_CODE].normal_(0.0, POSITION_LEVELS**-0.5)
            nn.init.zeros_(self.ship_heads.lasts[1].weight)
            self.density.lasts[0].bias.fill_(math.log(max(num_ships, 2) / 2 / cells))

    # ------------------------------------------------------------------ carry

    def initial_carry(self, views: int, ships_per_team: int, num_ships: int,
                      device=None) -> Carry:
        """No history: the trunk carry zero, no beliefs, every previous action null, a
        fresh position offset per view."""
        config, dtype = self.config, self.bins.dtype
        queries = num_ships + 1
        state = torch.zeros((views, config.blocks, queries, config.width), dtype=dtype,
                            device=device)
        belief = beliefs.blank((views, num_ships), len(MEANS), self.state_code.spread_count,
                               NUM_JOINT, device=device, dtype=dtype)
        return Carry(state=state,
                     convolution_buffer=state.new_zeros(state.shape[:-1] + (3, config.width)),
                     belief=belief,
                     previous_action=torch.zeros((views, ships_per_team), dtype=torch.int32,
                                                 device=device),
                     offset=POSITION_PERIOD * torch.rand((views, 2), dtype=dtype, device=device),
                     nonfinite=torch.zeros(views, dtype=torch.int64, device=device))

    @staticmethod
    def _cleared(carry: Carry, reset: torch.Tensor) -> Carry:
        """The carry with every view whose environment reset made fresh, with a new offset
        drawn (from the default generator: a captured graph draws afresh on each replay)."""
        keep = ~reset

        def masked(tensor: torch.Tensor, fill=0):
            shape = keep.shape + (1,) * (tensor.dim() - keep.dim())
            return torch.where(keep.view(shape), tensor, fill)

        prior = carry.belief
        belief = Belief(valid=prior.valid & keep[..., None], age=masked(prior.age),
                        means=masked(prior.means), spreads=masked(prior.spreads),
                        action=masked(prior.action, 1.0 / NUM_JOINT))
        offset = torch.where(keep[..., None], carry.offset,
                             POSITION_PERIOD * torch.rand_like(carry.offset))
        return Carry(state=masked(carry.state), convolution_buffer=masked(carry.convolution_buffer),
                     belief=belief, previous_action=masked(carry.previous_action), offset=offset,
                     nonfinite=carry.nonfinite)

    # ------------------------------------------------------------------ shared

    def _matmuls(self, device: torch.device):
        return torch.autocast(device.type, torch.bfloat16, enabled=self.config.bfloat16_matmuls)

    def tokens(self, obs: Obs, replay: Replay) -> Tokens:
        """The encoder inputs of views and their replay records."""
        return compose(obs, replay, self.state_code, self.ship_types,
                       self.config.presence_radius, self.tick)

    def _encode(self, tokens: Tokens):
        """Queries (..., Q, D), map memory (..., M, Dm) and the rotary tables."""
        queries, memory = self.encoder(tokens.ship, tokens.global_token, tokens.field)
        frequencies = (self.position_periods, self.position_periods, self.heading_frequencies)
        token_tables = rotary_tables(tokens.query_position,
                                     tokens.query_heading, *frequencies)
        map_tables = rotary_tables(tokens.field_position,
                                   torch.zeros_like(tokens.field_position[..., 0]), *frequencies)
        return queries, memory, token_tables, map_tables

    def _values(self, value_logits: torch.Tensor, outcome_logits: torch.Tensor):
        value_logits = value_logits.unflatten(-1, (-1, self.config.value_bins))
        values = critic_values(value_logits, outcome_logits, self.bins, self.config.local_levels,
                               self.config.levels.index(self.config.outcome_level))
        return value_logits, values

    # ------------------------------------------------------------------ step

    def step(self, carry: Carry, obs: Obs, reset: torch.Tensor) -> Decision:
        """One decision for views (R, ...); reset (R,) bool: the environment reset before
        this observation."""
        carry = self._cleared(carry, reset)
        ships, dtype = obs.ships, self.bins.dtype
        own = carry.previous_action.shape[-1]
        visible = ships[..., SHIP["visible"]] > 0.5
        spawned = ships[..., SHIP["spawned"]] > 0.5
        seen = beliefs.observe(carry.belief, truth(ships, carry.offset), visible, spawned)
        replay = Replay(valid=seen.valid, age=seen.age, means=seen.means,
                        spreads=seen.spreads.to(STORED_DTYPE),
                        enemy_action=seen.action[:, own:].to(STORED_DTYPE),
                        previous_action=carry.previous_action, offset=carry.offset)
        tokens = self.tokens(obs, replay)
        ships_count = tokens.code.shape[-2]
        with self._matmuls(ships.device):
            queries, memory, token_tables, map_tables = self._encode(tokens)
            hidden, state, buffer = self.trunk.step(queries, memory, token_tables, map_tables,
                                                    carry.state, carry.convolution_buffer)
            policy_logits, value_logits = self.own_heads(hidden[:, :own])
            enemy_action_logits, change = self.ship_heads(hidden[:, own:ships_count])
            outcome_logits, = self.outcome(hidden[:, ships_count])
        policy_logits, enemy_action_logits, outcome_logits = \
            policy_logits.to(dtype), enemy_action_logits.to(dtype), outcome_logits.to(dtype)
        _, values = self._values(value_logits.to(dtype), outcome_logits)
        forecast_logits = next_state_logits(tokens.code[:, own:], change.to(dtype))
        enemy = _rows(seen, slice(own, None))
        means, spreads = self.state_code.decode_logits(forecast_logits, tokens.ship_type[:, own:])
        position = _nearest(means[..., 0:2], enemy.means[..., 0:2])
        means = torch.cat([position, means[..., 2:]], -1)
        advanced, nonfinite = beliefs.advance(enemy, torch.cat([means, spreads], -1),
                                              enemy_action_logits)
        mine = _rows(seen, slice(None, own))
        belief = Belief(**{name: torch.cat([getattr(mine, name), getattr(advanced, name)], 1)
                           for name in BELIEF_FIELDS})
        carry = Carry(state=state.to(dtype), convolution_buffer=buffer.to(dtype), belief=belief,
                      previous_action=carry.previous_action, offset=carry.offset,
                      nonfinite=carry.nonfinite + nonfinite)
        return Decision(policy_logits=policy_logits, enemy_action_logits=enemy_action_logits,
                        next_state_logits=forecast_logits, values=values,
                        outcome_probabilities=torch.softmax(outcome_logits, -1), carry=carry,
                        replay=replay)

    # ------------------------------------------------------------------ unroll

    def unroll(self, carry: Carry, obs: Obs, replay: Replay, reset: torch.Tensor) -> Unrolled:
        """A stored segment (L, R, ...) from the carry at its start and its replay records;
        reset (L, R) as `step` received it. The encoder and the spatial sublayers run over
        every step at once, the temporal sublayers scan over time."""
        keep = ~reset[0]  # step clears the trunk carry at a reset; so does unroll
        state = torch.where(keep[:, None, None, None], carry.state, 0.0)
        buffer = torch.where(keep[:, None, None, None, None], carry.convolution_buffer, 0.0)
        tokens = self.tokens(obs, replay)
        own, ships = replay.previous_action.shape[-1], tokens.code.shape[-2]
        dtype = self.bins.dtype
        with self._matmuls(obs.ships.device):
            queries, memory, token_tables, map_tables = self._encode(tokens)
            hidden, _, _ = self.trunk.sequence(queries, memory, token_tables, map_tables,
                                               state, buffer, reset)
            policy_logits, value_logits = self.own_heads(hidden[..., :own, :])
            ship_hidden = hidden[..., :ships, :]
            enemy_action_logits, change = self.ship_heads(ship_hidden)
            outcome_logits, = self.outcome(hidden[..., ships, :])
            density_log_rates, = self.density(hidden[..., ships, :])
            shot_outcome_logits, = self.shot_outcome(ship_hidden)
            shot_latents = self.shot_trajectory_head.latent(ship_hidden)
        outcome_logits = outcome_logits.to(dtype)
        value_logits, values = self._values(value_logits.to(dtype), outcome_logits)
        return Unrolled(policy_logits=policy_logits.to(dtype),
                        enemy_action_logits=enemy_action_logits.to(dtype),
                        next_state_logits=next_state_logits(tokens.code, change.to(dtype)),
                        value_logits=value_logits, values=values, outcome_logits=outcome_logits,
                        density_log_rates=density_log_rates.to(dtype),
                        shot_outcome_logits=shot_outcome_logits.to(dtype),
                        shot_latents=shot_latents.to(dtype))

    def shot_trajectory(self, latents: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        """Trajectory code logits (..., ages, 154) of latents (..., D) at bullet ages (in
        moves) broadcasting against (..., ages)."""
        features = shots.time_features(ages, float(self.bullet_moves), SHOT_HARMONICS)
        return self.shot_trajectory_head.query(latents, features)
