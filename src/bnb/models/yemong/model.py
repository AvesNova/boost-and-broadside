"""Yemong: one decision (`step`) and a stored segment (`unroll`) over one team's views.

The model owns its inputs: the belief and the pending actions live in its carry, and it
composes its tokens from the raw observation. `step` returns a replay record (what it read
from the belief and the carry), so `unroll` recomputes a segment in parallel without
replaying the belief. `unroll` equals a loop of `step` over the same segment.
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
from .heads import MLPHead, NextStateHead, ShotTrajectoryHead, critic_values
from .inputs import FIELD_INPUTS, GLOBAL_INPUTS, Tokens, compose, ship_inputs, truth
from .layers import rotary_tables
from .network import Encoder, Trunk
from .state_code import MEANS, SPREADS, ShipStateCode, velocity_reach

SHOT_OUTCOMES = 19
SHOT_HARMONICS = 4


@dataclasses.dataclass(frozen=True)
class YemongConfig:
    """Production defaults (v1's known-good run). Nothing in the weights depends on the
    numbers of ships or fields or on the arena.

    levels: the reward levels in their configured order; outcome_level names the one read
    off the global token (the others are per-ship categorical values).
    rotary_period: the base period (m) of the position frequencies (2 pi / p) 2^k,
    k < rotary_octaves; heading_harmonics: the heading's.
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
    rotary_period: float = 64.0  # m
    rotary_octaves: int = 8
    heading_harmonics: tuple[int, ...] = (1, 2, 4, 8)

    def __post_init__(self) -> None:
        if self.outcome_level not in self.levels:
            raise ValueError(f"outcome level {self.outcome_level!r} is not in {self.levels}")
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
    (R, blocks, Q, 3, D): the trunk's; belief: per ship row; previous_action (R, S) int32,
    the own ships' last joint actions; nonfinite (R,) int64, a running count of non-finite
    forecast entries (replaced by zeros)."""

    state: torch.Tensor
    convolution_buffer: torch.Tensor
    belief: Belief
    previous_action: torch.Tensor
    nonfinite: torch.Tensor


@contract
class Replay:
    """What one decision read from the belief and the carry, after assimilating sight:
    valid (R, N), means (R, N, 12), spreads (R, N, 15), action (R, N, 42), previous_action
    (R, S)."""

    valid: torch.Tensor
    means: torch.Tensor
    spreads: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor


@contract
class Decision:
    """One step's outputs, per view: policy_logits (R, S, 42); enemy_action_logits and
    next_state_logits (R, N, ...) for every row; values (R, S, K) in the configured level
    order; outcome (R, 4) probabilities; the next carry (previous_action not yet replaced
    by the sample); the replay record."""

    policy_logits: torch.Tensor
    enemy_action_logits: torch.Tensor
    next_state_logits: torch.Tensor
    values: torch.Tensor
    outcome: torch.Tensor
    carry: Carry
    replay: Replay


@contract
class Unrolled:
    """A segment's outputs, (L, R, ...) leading: everything `Decision` has (outcome as
    logits), plus value_logits (L, R, S, K - 1, bins), density_log_rates (L, R, 2C) (ally
    cells, then enemy cells), shot_outcome_logits (L, R, N, 19) and shot_latents (L, R, N, D)
    for `Yemong.shot_trajectory`."""

    policy_logits: torch.Tensor
    enemy_action_logits: torch.Tensor
    next_state_logits: torch.Tensor
    value_logits: torch.Tensor
    values: torch.Tensor
    outcome_logits: torch.Tensor
    density_log_rates: torch.Tensor
    shot_outcome_logits: torch.Tensor
    shot_latents: torch.Tensor


class Yemong(nn.Module):
    """rules and types give the game's constants (tick, stall angle, shield delay, the ship
    card); rules.num_ships only sets the density head's starting rate."""

    def __init__(self, config: YemongConfig, rules: Rules, types: ShipTypes) -> None:
        super().__init__()
        self.config = config
        width, hidden = config.width, 2 * config.width
        self.ship_types = types.max_shield.shape[-1]
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
        self.policy = MLPHead(width, hidden, NUM_JOINT)
        self.enemy_action = MLPHead(width, hidden, NUM_JOINT)
        self.next_state = NextStateHead(width, hidden, self.state_code.width)
        self.value = MLPHead(width, hidden, levels * config.value_bins)
        self.outcome = MLPHead(width, hidden, critic.OUTCOMES)
        cells = density.cell_count(config.hex_rings)
        self.density = MLPHead(width, hidden, 2 * cells)
        self.shot_outcome = MLPHead(width, hidden, SHOT_OUTCOMES)
        self.shot_trajectory_head = ShotTrajectoryHead(width, 1 + 2 * SHOT_HARMONICS,
                                                       self.shot_code.trajectory_width)
        self._initialise(rules.num_ships, cells)
        octaves = 2.0 ** torch.arange(config.rotary_octaves, dtype=torch.float32)
        position_frequencies = 2.0 * math.pi / config.rotary_period * octaves
        self.register_buffer("position_frequencies", position_frequencies, persistent=False)
        self.register_buffer("heading_frequencies",
                             torch.tensor(config.heading_harmonics, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("bins", critic.value_bins(config.value_bins, config.value_limit),
                             persistent=False)

    def _initialise(self, num_ships: int, cells: int) -> None:
        """Code columns of each first projection ~ N(0, 1 / groups); the density head starts
        at an even spread of the expected ships."""
        groups = self.state_code.groups
        with torch.no_grad():
            self.encoder.ship.first.weight[:, :self.state_code.width].normal_(0.0, groups**-0.5)
            field_code = FIELD_INPUTS - 3
            self.encoder.field.first.weight[:, :field_code].normal_(0.0, (field_code // 9) ** -0.5)
            self.density.last.bias.fill_(math.log(max(num_ships, 2) / 2 / cells))

    # ------------------------------------------------------------------ carry

    def initial_carry(self, views: int, ships_per_team: int, num_ships: int,
                      device=None) -> Carry:
        """No history: the trunk carry zero, no beliefs, every previous action null."""
        config, dtype = self.config, self.bins.dtype
        queries = num_ships + 1
        state = torch.zeros((views, config.blocks, queries, config.width), dtype=dtype,
                            device=device)
        return Carry(state=state,
                     convolution_buffer=state.new_zeros(state.shape[:-1] + (3, config.width)),
                     belief=beliefs.blank((views, num_ships), len(MEANS), len(SPREADS), NUM_JOINT,
                                          device=device, dtype=dtype),
                     previous_action=torch.zeros((views, ships_per_team), dtype=torch.int32,
                                                 device=device),
                     nonfinite=torch.zeros(views, dtype=torch.int64, device=device))

    @staticmethod
    def _cleared(carry: Carry, reset: torch.Tensor) -> Carry:
        """The carry with every view whose environment reset made fresh."""
        keep = ~reset

        def masked(tensor: torch.Tensor, fill=0):
            shape = keep.shape + (1,) * (tensor.dim() - keep.dim())
            return torch.where(keep.view(shape), tensor, fill)

        prior = carry.belief
        belief = Belief(valid=masked(prior.valid, False), age=masked(prior.age),
                        means=masked(prior.means), spreads=masked(prior.spreads),
                        action=masked(prior.action, 1.0 / NUM_JOINT))
        return Carry(state=masked(carry.state), convolution_buffer=masked(carry.convolution_buffer),
                     belief=belief, previous_action=masked(carry.previous_action),
                     nonfinite=carry.nonfinite)

    # ------------------------------------------------------------------ shared

    def tokens(self, obs: Obs, replay: Replay) -> Tokens:
        """The encoder inputs of views and their replay records."""
        return compose(obs, replay.valid, replay.means, replay.spreads, replay.action,
                       replay.previous_action, self.state_code, self.ship_types,
                       self.config.presence_radius)

    def _encode(self, tokens: Tokens):
        """Queries (..., Q, D), map memory (..., M, Dm) and the rotary tables."""
        queries, memory = self.encoder(tokens.ship, tokens.global_token, tokens.field)
        position = torch.cat([tokens.ship_position,
                              torch.zeros_like(tokens.ship_position[..., :1, :])], -2)
        heading = torch.cat([tokens.ship_heading,
                             torch.zeros_like(tokens.ship_heading[..., :1])], -1)
        frequencies = (self.position_frequencies, self.position_frequencies,
                       self.heading_frequencies)
        token_tables = rotary_tables(position, heading, *frequencies)
        map_tables = rotary_tables(tokens.field_position,
                                   torch.zeros_like(tokens.field_position[..., 0]), *frequencies)
        return queries, memory, token_tables, map_tables

    def _decision_heads(self, hidden: torch.Tensor, code: torch.Tensor, own: int):
        """The heads `step` needs, from the trunk's output (..., Q, D)."""
        ships = code.shape[-2]
        own_hidden, ship_hidden = hidden[..., :own, :], hidden[..., :ships, :]
        global_hidden = hidden[..., ships, :]
        value_logits = self.value(own_hidden).unflatten(-1, (-1, self.config.value_bins))
        outcome_logits = self.outcome(global_hidden)
        values = critic_values(value_logits, outcome_logits, self.bins, self.config.local_levels,
                               self.config.levels.index(self.config.outcome_level))
        return (self.policy(own_hidden), self.enemy_action(ship_hidden),
                self.next_state(ship_hidden, code), value_logits, values, outcome_logits)

    # ------------------------------------------------------------------ step

    def step(self, carry: Carry, obs: Obs, reset: torch.Tensor) -> Decision:
        """One decision for views (R, ...); reset (R,) bool: the environment reset before
        this observation."""
        carry = self._cleared(carry, reset)
        ships = obs.ships
        visible = ships[..., SHIP["visible"]] > 0.5
        spawned = ships[..., SHIP["spawned"]] > 0.5
        seen = beliefs.observe(carry.belief, truth(ships), visible, spawned)
        replay = Replay(valid=seen.valid, means=seen.means, spreads=seen.spreads,
                        action=seen.action, previous_action=carry.previous_action)
        tokens = self.tokens(obs, replay)
        queries, memory, token_tables, map_tables = self._encode(tokens)
        hidden, state, buffer = self.trunk.step(queries, memory, token_tables, map_tables,
                                                carry.state, carry.convolution_buffer)
        policy_logits, enemy_action_logits, next_state_logits, _, values, outcome_logits = \
            self._decision_heads(hidden, tokens.code, carry.previous_action.shape[-1])
        means, spreads = self.state_code.decode_logits(next_state_logits, tokens.ship_type)
        advanced, nonfinite = beliefs.advance(seen, torch.cat([means, spreads], -1),
                                              enemy_action_logits)
        carry = Carry(state=state, convolution_buffer=buffer, belief=advanced,
                      previous_action=carry.previous_action,
                      nonfinite=carry.nonfinite + nonfinite)
        return Decision(policy_logits=policy_logits, enemy_action_logits=enemy_action_logits,
                        next_state_logits=next_state_logits, values=values,
                        outcome=torch.softmax(outcome_logits, -1), carry=carry, replay=replay)

    # ------------------------------------------------------------------ unroll

    def unroll(self, carry: Carry, obs: Obs, replay: Replay, reset: torch.Tensor) -> Unrolled:
        """A stored segment (L, R, ...) from the carry at its start and its replay records;
        reset (L, R) as `step` received it. The encoder and the spatial sublayers run over
        every step at once, the temporal sublayers scan over time."""
        carry = self._cleared(carry, reset[0])
        tokens = self.tokens(obs, replay)
        queries, memory, token_tables, map_tables = self._encode(tokens)
        hidden, _, _ = self.trunk.sequence(queries, memory, token_tables, map_tables,
                                           carry.state, carry.convolution_buffer, reset)
        policy_logits, enemy_action_logits, next_state_logits, value_logits, values, \
            outcome_logits = self._decision_heads(hidden, tokens.code,
                                                  replay.previous_action.shape[-1])
        ships = tokens.code.shape[-2]
        ship_hidden = hidden[..., :ships, :]
        return Unrolled(policy_logits=policy_logits, enemy_action_logits=enemy_action_logits,
                        next_state_logits=next_state_logits, value_logits=value_logits,
                        values=values, outcome_logits=outcome_logits,
                        density_log_rates=self.density(hidden[..., ships, :]),
                        shot_outcome_logits=self.shot_outcome(ship_hidden),
                        shot_latents=self.shot_trajectory_head.latent(ship_hidden))

    def shot_trajectory(self, latents: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        """Trajectory code logits (..., ages, 154) of latents (..., D) at bullet ages (in
        moves) broadcasting against (..., ages)."""
        features = shots.time_features(ages, float(self.bullet_moves), SHOT_HARMONICS)
        return self.shot_trajectory_head.query(latents, features)
