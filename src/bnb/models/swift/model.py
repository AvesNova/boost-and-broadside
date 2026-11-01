"""Yemong: the policy and value model (spec 4-7), with `step` (one decision) and `unroll` (a
stored segment). The model owns its inputs, its carry and its belief.

Shapes: R views, N ships per view (own S = N / 2 first), M fields, Q = N + 1 queries
(ships, then the global token), D width, L steps. Positions are metres, times seconds.
"""

import dataclasses
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from bnb.contracts.action import NUM_JOINT
from bnb.contracts.obs import FIELD, GLOBAL, SHIP
from bnb.contracts.trajectory import REWARD_LEVELS

from . import belief, targets
from .codes import code_baseline, position_encode
from .network import Encoder, MLPHead, ShotTrajectoryHead, Trunk, rotary_tables
from .state_code import (
    MEAN_CHANNELS,
    POSITION_LEVELS,
    POSITION_PERIOD,
    SPREAD_CHANNELS,
    ShipCard,
    ShipStateCodec,
)

GRADIENT_SCALE = 1.875 * 2 * math.log(2.0) / 40.0  # g*, per metre
PRESENCE_RADIUS = 150.0  # m
RESPAWN_HORIZON = 16.0  # s
TIME_HARMONICS = 4
ROTARY_POSITION_FREQUENCIES = 8
ROTARY_HEADING_HARMONICS = (1.0, 2.0, 4.0, 8.0)
OUTCOMES = 4
SHOT_OUTCOME_WIDTH = 19
SHOT_TRAJECTORY_WIDTH = 154
FIELD_FEATURES = 81 + 3
GLOBAL_FEATURES = 3


@dataclasses.dataclass(frozen=True)
class YemongConfig:
    """Production defaults (spec 2)."""

    width: int = 128
    blocks: int = 2
    spatial_per_block: int = 2
    heads: int = 2
    map_width: int = 64
    value_bins: int = 51
    value_limit: float = 2.0
    reward_levels: tuple[str, ...] = REWARD_LEVELS
    outcome_level: str = "outcome"
    hex_rings: int = 10
    ships: int = 10  # N at construction: only the density head's initial bias reads it
    card: ShipCard = ShipCard()

    @property
    def level_count(self) -> int:
        return len(self.reward_levels)

    @property
    def outcome_index(self) -> int:
        return self.reward_levels.index(self.outcome_level)

    @property
    def local_levels(self) -> list[int]:
        return [i for i in range(self.level_count) if i != self.outcome_index]

    @property
    def hex_cell_count(self) -> int:
        return 1 + 3 * self.hex_rings * (self.hex_rings + 1)


class Carry(NamedTuple):
    """Per view (leading R). state (R, blocks, Q, D); convolution_buffer (R, blocks, Q, 3, D);
    belief: valid (R, N) bool, age (R, N) int32 ticks, means (R, N, 12), spreads (R, N, 15),
    action (R, N, 42) probabilities; previous_action (R, S) int64 joint index; nonfinite (R,)
    int64, the running count of non-finite forecast entries."""

    state: torch.Tensor
    convolution_buffer: torch.Tensor
    valid: torch.Tensor
    age: torch.Tensor
    means: torch.Tensor
    spreads: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    nonfinite: torch.Tensor


class Replay(NamedTuple):
    """What a decision read from the belief and the carry (after observing), leading (R,):
    valid (R, N), age (R, N), means (R, N, 12), spreads (R, N, 15), action (R, N, 42),
    previous_action (R, S)."""

    valid: torch.Tensor
    age: torch.Tensor
    means: torch.Tensor
    spreads: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor


class Inputs(NamedTuple):
    ship: torch.Tensor  # (..., N, 553)
    global_token: torch.Tensor  # (..., 1, 3)
    field: torch.Tensor  # (..., M, 84)
    code: torch.Tensor  # (..., N, 490), the ships' state code
    token_tables: tuple  # (cos, sin), each (..., Q, 1, 20)
    map_tables: tuple  # (..., M, 1, 20)


def upcast(tensor):
    """At least float32 (autocast leaves matmul outputs in bfloat16)."""
    return tensor if tensor.dtype in (torch.float32, torch.float64) else tensor.float()


class Yemong(nn.Module):
    def __init__(self, config: YemongConfig = YemongConfig()):
        super().__init__()
        self.config = config
        card = config.card
        self.codec = ShipStateCodec(card)
        width = config.width
        self.ship_features = (self.codec.width + 2 + card.ship_types + 3 + 1
                              + (1 + 2 * TIME_HARMONICS) + NUM_JOINT + 2 + 2)
        self.encoder = Encoder(self.ship_features, GLOBAL_FEATURES, FIELD_FEATURES, width,
                               config.map_width)
        self.trunk = Trunk(width, config.heads, config.map_width, config.blocks,
                           config.spatial_per_block)
        hidden = 2 * width
        self.action_head = MLPHead(width, hidden, NUM_JOINT)
        self.enemy_action_head = MLPHead(width, hidden, NUM_JOINT)
        self.next_state_head = MLPHead(width, hidden, self.codec.width)
        self.value_head = MLPHead(width, hidden, (config.level_count - 1) * config.value_bins)
        self.outcome_head = MLPHead(width, hidden, OUTCOMES)
        self.density_head = MLPHead(width, hidden, 2 * config.hex_cell_count)
        self.shot_outcome_head = MLPHead(width, hidden, SHOT_OUTCOME_WIDTH)
        self.shot_trajectory_head = ShotTrajectoryHead(width, 1 + 2 * TIME_HARMONICS,
                                                       SHOT_TRAJECTORY_WIDTH)

        position_frequencies = (2 * math.pi / 8192.0) * 2.0 ** torch.arange(
            ROTARY_POSITION_FREQUENCIES, dtype=torch.float64)
        self.register_buffer("rotary_position", position_frequencies.float(), persistent=False)
        self.register_buffer("rotary_heading", torch.tensor(ROTARY_HEADING_HARMONICS),
                             persistent=False)
        self.register_buffer("bins", targets.value_bins(config.value_bins, config.value_limit)
                             .float(), persistent=False)
        self.initialise()

    # ------------------------------------------------------------------ initialisation

    def initialise(self):
        for head in (self.action_head, self.enemy_action_head, self.next_state_head,
                     self.value_head, self.outcome_head, self.density_head,
                     self.shot_outcome_head):
            head.initialise()
        nn.init.zeros_(self.next_state_head.last.weight)
        cells = self.config.hex_cell_count
        nn.init.constant_(self.density_head.last.bias,
                          math.log(max(self.config.ships, 2) / 2 / cells))
        self.shot_trajectory_head.initialise()
        with torch.no_grad():
            kinds = self.encoder.kinds
            kinds["ship"].first.weight[:, : self.codec.width].normal_(
                0, 1 / math.sqrt(self.codec.groups))
            kinds["field"].first.weight[:, :81].normal_(0, 1 / math.sqrt(POSITION_LEVELS))

    # ------------------------------------------------------------------ carry

    def initial_carry(self, views: int, ships: int, device=None, dtype=torch.float32) -> Carry:
        config = self.config
        queries = ships + 1
        width = config.width
        return Carry(
            state=torch.zeros(views, config.blocks, queries, width, device=device, dtype=dtype),
            convolution_buffer=torch.zeros(views, config.blocks, queries, 3, width, device=device,
                                           dtype=dtype),
            valid=torch.zeros(views, ships, dtype=torch.bool, device=device),
            age=torch.zeros(views, ships, dtype=torch.int32, device=device),
            means=torch.zeros(views, ships, MEAN_CHANNELS, device=device, dtype=dtype),
            spreads=torch.zeros(views, ships, SPREAD_CHANNELS, device=device, dtype=dtype),
            action=torch.full((views, ships, NUM_JOINT), 1.0 / NUM_JOINT, device=device,
                              dtype=dtype),
            previous_action=torch.zeros(views, ships // 2, dtype=torch.int64, device=device),
            nonfinite=torch.zeros(views, dtype=torch.int64, device=device),
        )

    # ------------------------------------------------------------------ inputs

    @staticmethod
    def truth(ships):
        """obs ships (..., N, columns) -> the belief channels (..., N, 12)."""
        column = {name: ships[..., index] for name, index in SHIP.items()}
        return torch.stack([
            column["x"], column["y"], column["velocity_x"], column["velocity_y"],
            torch.atan2(column["heading_sin"], column["heading_cos"]), column["nose_rate"],
            column["angle_of_attack"], column["shield_recharge_wait"], column["shield"],
            column["energy"], column["gun_cooldown"], column["log_refractive_index"],
        ], -1)

    def ship_type(self, ships):
        return torch.clamp(ships[..., SHIP["ship_type"]].long(), 0, self.config.card.ship_types - 1)

    def compose(self, ships, fields, globals_, replay: Replay) -> Inputs:
        """Builds the tokens of spec 4 from the observation and the belief (after observing).

        ships (..., N, 24), fields (..., M, 5), globals_ (..., 11); replay leaves (..., N, ...)."""
        card = self.config.card
        dtype = replay.means.dtype
        own = replay.previous_action.shape[-1]
        ship_type = self.ship_type(ships)
        visible = ships[..., SHIP["visible"]] > 0.5
        spawned = ships[..., SHIP["spawned"]] > 0.5
        team = ships[..., SHIP["team_relative"]].long().clamp(0, 1)
        code = self.codec.encode(replay.means, replay.spreads, ship_type)

        pending_own = F.one_hot(replay.previous_action, NUM_JOINT).to(dtype)
        null = F.one_hot(torch.zeros_like(spawned[..., own:], dtype=torch.long), NUM_JOINT)
        pending_enemy = torch.where(spawned[..., own:].unsqueeze(-1), null.to(dtype),
                                    replay.action[..., own:, :])
        pending = torch.cat([pending_own, pending_enemy], -2)
        gradient = ships[..., SHIP["log_index_gradient_x"]:SHIP["log_index_gradient_y"] + 1]
        gradient = torch.where(visible.unsqueeze(-1), gradient, 0.0) / GRADIENT_SCALE
        presence = targets.local_presence(replay.means[..., 0:2], team, replay.valid,
                                          PRESENCE_RADIUS)
        time_since_seen = replay.age.to(dtype) * card.tick_seconds
        flags = ships[..., [SHIP["alive"], SHIP["visible"], SHIP["fired"]]]
        ship = torch.cat([
            code,
            F.one_hot(team, 2).to(dtype),
            F.one_hot(ship_type, card.ship_types).to(dtype),
            flags.to(dtype),
            targets.symlog(time_since_seen).unsqueeze(-1),
            targets.respawn_features(ships[..., SHIP["respawn_wait"]].to(dtype), RESPAWN_HORIZON,
                                     TIME_HARMONICS),
            pending,
            gradient.to(dtype),
            presence,
        ], -1)

        field_position = fields[..., FIELD["x"]:FIELD["y"] + 1].to(dtype)
        zero = torch.zeros_like(field_position[..., 0])
        field = torch.cat([
            position_encode((POSITION_PERIOD, POSITION_PERIOD), POSITION_LEVELS, field_position,
                            zero).flatten(-2),
            (fields[..., [FIELD["radius"], FIELD["band_width"], FIELD["level"]]].to(dtype)
             / torch.tensor([1000.0, 40.0, 2.0], dtype=dtype, device=fields.device)),
        ], -1)
        global_token = (globals_[..., [GLOBAL["episode_progress"], GLOBAL["arena_radius"],
                                       GLOBAL["fogged"]]].to(dtype)
                        / torch.tensor([1.0, 2600.0, 1.0], dtype=dtype,
                                       device=globals_.device)).unsqueeze(-2)

        frequencies_position = self.rotary_position.to(dtype)
        frequencies_heading = self.rotary_heading.to(dtype)
        token_position = torch.cat([replay.means[..., 0:2],
                                    torch.zeros_like(replay.means[..., :1, 0:2])], -2)
        token_heading = torch.cat([replay.means[..., 4],
                                   torch.zeros_like(replay.means[..., :1, 4])], -1)
        token_tables = rotary_tables(frequencies_position, frequencies_position,
                                     frequencies_heading, token_position, token_heading)
        map_tables = rotary_tables(frequencies_position, frequencies_position,
                                   frequencies_heading, field_position, zero)
        return Inputs(ship, global_token, field, code, token_tables, map_tables)

    # ------------------------------------------------------------------ the network

    def network(self, inputs: Inputs, state, buffer, starts):
        """inputs leading (T, B); state (B, blocks, Q, D); buffer (B, blocks, Q, 3, D);
        starts (T, B) bool or None. Returns (x (T, B, Q, D), state, buffer)."""
        queries, memory = self.encoder(inputs.ship, inputs.global_token, inputs.field)
        x, state, buffer = self.trunk(queries, memory, inputs.token_tables, inputs.map_tables,
                                      state.transpose(0, 1), buffer.transpose(0, 1), starts)
        return x, state.transpose(0, 1), buffer.transpose(0, 1)

    def decision_heads(self, x, code, own: int):
        """The heads `step` gives. x (..., Q, D); code (..., N, 490)."""
        config = self.config
        ships = x[..., :-1, :]
        global_token = x[..., -1, :]
        action_logits = upcast(self.action_head(ships[..., :own, :]))
        enemy_action_logits = upcast(self.enemy_action_head(ships))
        next_state_logits = code_baseline(code) + upcast(self.next_state_head(ships))
        value_logits = upcast(self.value_head(ships[..., :own, :])).unflatten(
            -1, (config.level_count - 1, config.value_bins))
        outcome_logits = upcast(self.outcome_head(global_token))
        values = targets.critic_values(value_logits, outcome_logits, self.bins.to(
            value_logits.dtype), config.local_levels, config.outcome_index)
        return {
            "action_logits": action_logits,
            "enemy_action_logits": enemy_action_logits,
            "next_state_logits": next_state_logits,
            "value_logits": value_logits,
            "outcome_logits": outcome_logits,
            "values": values,
            "value": values.sum(-1),
            "outcome": torch.softmax(outcome_logits, -1),
        }

    # ------------------------------------------------------------------ step

    def step(self, carry: Carry, obs, reset, sample: bool = True):
        """One decision for R views.

        obs: `Obs` with ships (R, N, 24), fields (R, M, 5), globals (R, 11); reset (R,) bool,
        the environment reset before this observation.
        Returns (carry, outputs): outputs holds the decision heads (action_logits (R, S, 42),
        enemy_action_logits (R, N, 42), next_state_logits (R, N, 490), values (R, S, K), value
        (R, S), outcome (R, 4)), the sampled `action` (R, S) joint index with its `logp` and
        the policy `entropy` (R, S), and `replay` (a `Replay`)."""
        ships, fields, globals_ = obs.ships, obs.fields, obs.globals
        dtype = carry.means.dtype
        own = carry.previous_action.shape[-1]
        keep = ~reset
        view = keep.view(-1, 1)
        state = carry.state * keep.view(-1, 1, 1, 1).to(dtype)
        buffer = carry.convolution_buffer * keep.view(-1, 1, 1, 1, 1).to(dtype)
        valid, age, means, spreads, action = belief.observe(
            carry.valid & view, torch.where(view, carry.age, 0),
            torch.where(view.unsqueeze(-1), carry.means, 0.0),
            torch.where(view.unsqueeze(-1), carry.spreads, 0.0),
            torch.where(view.unsqueeze(-1), carry.action, 1.0 / NUM_JOINT),
            self.truth(ships).to(dtype), ships[..., SHIP["visible"]] > 0.5,
            ships[..., SHIP["spawned"]] > 0.5)
        previous_action = torch.where(view, carry.previous_action, 0)
        replay = Replay(valid, age, means, spreads, action, previous_action)

        inputs = self.compose(ships, fields, globals_, replay)
        inputs = Inputs(*(tuple(t.unsqueeze(0) for t in item) if isinstance(item, tuple)
                          else item.unsqueeze(0) for item in inputs))
        x, state, buffer = self.network(inputs, state, buffer, None)
        heads = self.decision_heads(x[0], inputs.code[0], own)

        prediction = torch.cat(self.codec.decode_logits(heads["next_state_logits"],
                                                        self.ship_type(ships)), -1)
        means, spreads, action, nonfinite = belief.advance(
            valid, means, spreads, action, prediction, heads["enemy_action_logits"])

        log_probabilities = torch.log_softmax(heads["action_logits"], -1)
        if sample:
            uniform = torch.rand_like(log_probabilities).clamp_(min=1e-20)
            chosen = torch.argmax(log_probabilities - torch.log(-torch.log(uniform)), -1)
        else:
            chosen = torch.argmax(log_probabilities, -1)
        heads["action"] = chosen
        heads["logp"] = log_probabilities.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        heads["entropy"] = -(log_probabilities.exp() * log_probabilities).sum(-1)
        heads["replay"] = replay
        carry = Carry(state, buffer, valid, age, means, spreads, action, chosen,
                      carry.nonfinite + nonfinite)
        return carry, heads

    # ------------------------------------------------------------------ unroll

    def unroll(self, carry: Carry, obs, replay: Replay, reset, shot_ages=None):
        """A stored segment at once.

        carry: the carry at the segment's start (only the trunk's state and buffer are read);
        obs leaves (L, R, ...); replay leaves (L, R, ...); reset (L, R) bool; shot_ages
        (L, R, N, A) moves or None.
        Returns the decision heads of `step` with leading (L, R) (no sampling), plus
        density_log_rates (L, R, 2 * 331), shot_outcome_logits (L, R, N, 19), shot_latent
        (L, R, N, D) and, with shot_ages, shot_trajectory_logits (L, R, N, A, 154)."""
        dtype = carry.means.dtype
        replay = Replay(*(leaf.detach() for leaf in replay))  # stored data: no gradient
        first = (~reset[0]).to(dtype)
        state = carry.state * first.view(-1, 1, 1, 1)
        buffer = carry.convolution_buffer * first.view(-1, 1, 1, 1, 1)
        starts = torch.cat([torch.zeros_like(reset[:1]), reset[1:]], 0)
        inputs = self.compose(obs.ships, obs.fields, obs.globals, replay)
        x, state, buffer = self.network(inputs, state, buffer, starts)
        heads = self.decision_heads(x, inputs.code, replay.previous_action.shape[-1])
        ships = x[..., :-1, :]
        heads["density_log_rates"] = upcast(self.density_head(x[..., -1, :]))
        heads["shot_outcome_logits"] = upcast(self.shot_outcome_head(ships))
        latent = self.shot_trajectory_head.latent(ships)
        heads["shot_latent"] = latent
        if shot_ages is not None:
            heads["shot_trajectory_logits"] = self.shot_trajectory(latent, shot_ages)
        heads["final_state"] = state
        heads["final_convolution_buffer"] = buffer
        return heads

    def shot_trajectory(self, latent, ages):
        """latent (..., D); ages (..., A) in moves -> logits (..., A, 154)."""
        dtype = upcast(latent).dtype
        features = targets.shot_time_features(ages.to(dtype),
                                              float(self.config.card.bullet_life_moves),
                                              TIME_HARMONICS)
        return upcast(self.shot_trajectory_head.query(latent, features))
