"""The training loop: rollout segments into the store, the learner's update, the league's
hooks, progress metrics.

Lineup: self-play and the league's seat (`bnb.runtime.opponents`), half each, or the league's
seat alone for a run that trains one seat or an exploiter (`training_lineup`). A mixed batch
(mixed spec 5) crosses the lineup with the games of a training mix (`TrainConfig.mode_mix`).
The learner is Yemong, the toy relational model (learner spec 9) or the nearest-neighbour
MLP (`bnb.models.nearest`). The objective is PPO, or behaviour cloning of Imugi
(`bnb.learn.bc`, with its own lineup and stop rule in `bnb.runtime.cloning`).
"""

import copy
import dataclasses
import math
import time
from collections.abc import Callable
from typing import Protocol

import torch

from bnb.contracts import ActiveFields, ArenaScale, GameConfig, Pool, TeamSizes
from bnb.contracts.pools import pool_game
from bnb.controllers.imugi import Imugi
from bnb.controllers.relational import RelationalController
from bnb.controllers.scripted import Chaser, Random
from bnb.controllers.yemong import YemongController
from bnb.game import api
from bnb.game.api import PRESETS, ShipTypes
from bnb.game.handling.characteristics import characteristics
from bnb.learn.bc import (
    BehaviourCloning,
    StopConfig,
    StopRule,
    TeacherConfig,
    TeacherOpponent,
    teacher_agent,
)
from bnb.learn.ppo import Learner, LearnerConfig
from bnb.learn.rewards import RewardConfig
from bnb.models.nearest import Nearest, NearestBuild, NearestConfig
from bnb.models.relational import Relational, RelationalBuild, RelationalConfig
from bnb.models.yemong.codes import PADDING_BINS
from bnb.models.yemong.inputs import MODE_INPUTS
from bnb.models.yemong.model import (
    COLOUR_PERIOD,
    ROTARY_BANDS,
    ROTARY_COARSEST,
    ROTARY_RATIO,
    ROTARY_TURN,
    Yemong,
    YemongBuild,
    YemongConfig,
    yemong_build,
)

from . import analytics, lanes, metrics
from .evaluate import first_results, shares
from .lanes import LANES, LIVE, LaneConfig, LaneSeat
from .lineup import Lineup
from .metrics import Collector
from .opponents import OPPONENTS, LeagueConfig, OpponentSlots
from .rollout import Rollout
from .snapshots import RL_SNAPSHOTS, SnapshotSchedule

LEARNER = "learner"
IMUGI_SEAT = "scripted"  # the seat kind Imugi plays; run configs record it as "scripted"
# The default training mix (mixed spec 1): each game preset's share of the batch. The battle
# map (deathmatch_wide) is out for now (author, 2026-11-06: decisions, BC round three).
TRAINING_MIX = (("frontline", 0.5), ("deathmatch", 0.5))
# the mixes `bnb train --preset` names: the standard ship set's, and the real-scale set's
# (the same two modes on the real-scale maps, `PRESETS["real_scale_*"]`); and
# real_scale_all_modes, the final experiments' mix: the three real-scale modes, frontline
# 6/16, deathmatch 5/16, elimination 5/16 (decided 2026-11-14; 8/5/3 from 2026-11-12): of
# 2048 environments 768 / 640 / 640, each mode's two pools whole: 384 / 320 / 320; shares
# of sixteenths so that 2048 splits exactly and its views divide every minibatch count
TRAINING_MIXES = {
    "mixed": TRAINING_MIX,
    "real_scale_mixed": (("real_scale_frontline", 0.5), ("real_scale_deathmatch", 0.5)),
    "real_scale_all_modes": (
        ("real_scale_frontline", 0.375),
        ("real_scale_deathmatch", 0.3125),
        ("real_scale_elimination", 0.3125),
    ),
}
# Self-play is a fixed share of each mode's batch, the league's seat the rest (league spec 1).
LINEUP = (Pool("self_play", 0.5, (LEARNER, LEARNER)), Pool("league", 0.5, (LEARNER, OPPONENTS)))
# The lineup of a run that trains one seat or an exploiter: every game against the league's
# seat (self-play, if any, is its "latest" slots: `LeagueConfig.latest_share`).
LEAGUE_ONLY = (Pool("league", 1.0, (LEARNER, OPPONENTS)),)
SELF_PLAY_SEATS = ("both", "one")
MODELS = ("yemong", "relational", "nearest")
OBJECTIVES = ("ppo", "bc")
# `TrainConfig.update_policies`: which runs write every update's policy files, by stage (a
# PPO run is "rl", a BC run "bc"; "all" both, "off" neither).
UPDATE_POLICIES = ("off", "rl", "bc", "all")
PRECISIONS = ("float32", "tf32", "bfloat16")


def mixed(
    lineup: tuple[Pool, ...] = LINEUP,
    mix: tuple[tuple[str, float], ...] = TRAINING_MIX,
    base: GameConfig | None = None,
) -> tuple[GameConfig, tuple[Pool, ...]]:
    """A mixed batch: the shared config (`base`, by default the first game's) and every game
    of the mix (a preset name and its share) crossed with every pool of the lineup, game by
    game, so each game's environments are one contiguous slice. Pool "frontline/chaser"
    takes the frontline share of the chaser share."""
    base_name = "the run's game" if base is not None else mix[0][0]
    base = base if base is not None else PRESETS[mix[0][0]]
    pools = []
    for name, share in mix:
        if name not in PRESETS:
            raise ValueError(f"{name!r} is not a game preset; one of {sorted(PRESETS)}")
        game = PRESETS[name]
        if (game.ships_per_team, game.num_fields) != (base.ships_per_team, base.num_fields):
            raise ValueError(f"{name} has other sizes than {base_name}: one batch, one size")
        if game.ship_set != base.ship_set:
            raise ValueError(f"{name} plays another ship set than {base_name}: one batch, one card")
        pools.extend(
            Pool(
                f"{name}/{pool.name}",
                share * pool.share,
                pool.seats,
                pool.fog_share,
                mode=game.mode,
                arena_radius=game.arena_radius,
                spawn_groups=game.spawn_groups,
                episode_seconds=game.episode_seconds,
                respawn_seconds=game.respawn_seconds,
            )
            for pool in lineup
        )
    return base, tuple(pools)


def training_lineup(config: TrainConfig) -> tuple[Pool, ...]:
    """The PPO lineup the settings name: `LINEUP` (self-play training both seats, and the
    league's seat), or `LEAGUE_ONLY` for a run that trains one seat
    (`TrainConfig.self_play_seats="one"`) or an exploiter (`LeagueConfig.fixed_opponent`)."""
    if config.self_play_seats == "one" or config.league.fixed_opponent is not None:
        return LEAGUE_ONLY
    return LINEUP


def arranged(config: TrainConfig, preset: str | None = None) -> TrainConfig:
    """`config` with the pools its lineup settings (`training_lineup`) and its mode mix name:
    `TrainConfig.mode_mix`, else the training mix `preset` names (`TRAINING_MIXES`), else
    the lineup alone, each game's pools with that game's own map, clocks, respawn and spawn
    groups (`mixed`). A config that sets neither (`mode_mix` None, the default lineup) or
    clones is returned as it is, its pools untouched."""
    lineup = training_lineup(config)
    if config.objective != "ppo" or (config.mode_mix is None and lineup == LINEUP):
        return config
    mix = config.mix() or TRAINING_MIXES.get(preset)
    pools = lineup if mix is None else mixed(lineup, mix, config.game)[1]
    return dataclasses.replace(config, pools=pools)


def check_pools(config: TrainConfig) -> None:
    """Refuse pools that contradict the settings (`arranged` builds them): a self-play pool
    in a run that trains one seat or an exploiter, or games and shares other than the mode
    mix's."""
    if training_lineup(config) == LEAGUE_ONLY:
        both = [pool.name for pool in config.pools if pool.seats == (LEARNER, LEARNER)]
        if both:
            raise ValueError(
                f"pools {both} train both seats, but the run trains one seat "
                f"(self_play_seats={config.self_play_seats!r}, fixed_opponent="
                f"{config.league.fixed_opponent!r}): build the pools with `arranged`"
            )
    mix = config.mix()
    if mix is not None:
        shares: dict[str, float] = {}
        for pool in config.pools:
            name = pool.name.split("/")[0]
            shares[name] = shares.get(name, 0.0) + pool.share
        want = dict(mix)
        if shares.keys() != want.keys() or any(
            not math.isclose(shares[name], want[name]) for name in want
        ):
            raise ValueError(
                f"the pools' games and shares {shares} are not mode_mix's {want}: build the "
                f"pools with `arranged`"
            )


def training_pools(config: TrainConfig) -> tuple[Pool, ...]:
    """The config's training pools with its map-size sampling (`TrainConfig.arena_scale`),
    its field-count sampling (`TrainConfig.active_fields`) and its team-size sampling
    (`TrainConfig.team_sizes`)."""
    return tuple(
        dataclasses.replace(
            pool,
            arena_scale=config.arena_scale,
            active_fields=config.active_fields,
            team_sizes=config.team_sizes,
        )
        for pool in config.pools
    )


def samples_team_sizes(pools: tuple[Pool, ...]) -> bool:
    """Some pool draws its team size per episode (`Pool.team_sizes`). A Yemong playing it
    masks the benched ships (`Yemong.mask_benched`)."""
    return any(pool.team_sizes is not None for pool in pools)


def samples_field_counts(game: GameConfig, pools: tuple[Pool, ...] = ()) -> bool:
    """Some game of the batch draws its active field count per episode: the game's
    (`GameConfig.active_fields`) or a pool's. A Yemong playing it masks the absent fields
    (`Yemong.mask_absent_fields`)."""
    games = [pool_game(game, pool) for pool in pools] or [game]
    return any(one.active_fields is not None for one in games)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """model: "yemong", "relational" (the toy learner) or "nearest" (the nearest-neighbour
    MLP), built with the settings of the same name (`yemong`, `relational`, `nearest`);
    their `bfloat16_matmuls` stays False, as `precision`
    sets it, and a new run's Yemong decodes its belief's spreads with
    `spread_fit="coarsest"`, runs each block's temporal sublayer first
    (`order="temporal_first"`, arm C's order, the champion's since 2026-11-07), reads
    every mode's flag (`mode_inputs`, elimination's since 2026-11-12) and Frontline's
    timers (`frontline_timer_inputs`, since 2026-11-13), and has the position
    codes of 2026-11-12: the rotary bands (ROTARY_*: eight at ratio 3.405 from 48 m, each
    turned 35 degrees, heading harmonics 1, 2, 4), the colour code to 236 196 m
    (`position_period`), the arena radius's code (`arena_code`) and the geometry head
    (`geometry_head`), and every model's value support has `codes.PADDING_BINS` bins past
    each end (`value_padding`, since 2026-11-15), and a new nearest-neighbour MLP reads
    every mode's inputs (`NearestConfig.full_mode_inputs`, since 2026-11-15); the config's
    own defaults, "v1", "spatial_first", the two older modes, no timers, the older codes,
    no value padding and the MLP's first inputs, are what older build records hold (every build
    reads the boundary's radius and time to zero). ship_card:
    the ship card's values by field, one per ship type, every field given
    (`bnb.game.api.default_card`). environments: the training environments; length: ticks
    per segment; shards: consecutive segments per update (v1's batch, twice over:
    2,048 x 4 x 128); lookahead: ticks played after the last shard on every environment,
    never trained on, for the targets that need the future (the returns' tail, the shot
    labels); the next update continues from its end. graph_ticks: the most ticks in one
    captured rollout chunk; the device holds two chunks of the learner's records, streamed
    to the store as the next one plays, so it bounds the rollout's memory whatever the
    segment's length (memory and speed only). lanes: the rating lanes
    (`bnb.runtime.lanes`), environments beyond the training ones. precision: "bfloat16" (the
    model's matmuls under autocast; the default), "float32" or "tf32" (a process-wide
    setting, which the entry point applies with `bnb.cli.set_precision`). reward: the reward
    levels' weights. host_store: stream the segments through pinned host memory (on a GPU)
    rather than keep them on the device.
    seed: keys every random draw (`bnb.streams`): the model's initial weights, the
    rollout's maps, seats and decisions, the league's swaps, the learner's streams.
    objective: "ppo", or "bc" (clone Imugi; `learner` should then be
    `bnb.learn.bc.config()`, `teacher` sets its labels and how the scripted opponent
    plays, `bnb.learn.bc.TeacherOpponent`, and `stop` when it stops, its budget included).
    update_policies: which runs write the live, exponential-average and uniform-average
    policy files after every update (and their starting weights before the first) to
    `policies/` (about 11 MB each, never pruned; `bnb.runs.session`), by the run's stage:
    "off" (the default), "rl" (a PPO run, a chain's RL stage among them), "bc" (a BC run)
    or "all". One setting for both stages of a chain: "rl" keeps a chain's BC stage on its
    snapshots (BC's files would add about 1 GB). On for the run that feeds the averaging
    sweep (`bnb average`, R0 and R5); a resume may change it.
    policy_snapshots: for a run that does not write every update's policy files
    (`update_policies`), the environment steps at which it keeps them
    under `policies/step_<steps>/`, the live weights and both averages
    (`bnb.runtime.snapshots.SnapshotSchedule`; `RL_SNAPSHOTS` by default, decided
    2026-11-14: 1, 2, 3, 4, 6, 8, 10, 12, 16 and 20M steps, then every 5M), and always at
    the end of each process's segment (`bnb.runs.session.train`).
    arena_scale: None (the default) plays each pool's own arena every episode; map-size
    sampling for the training pools (`bnb.contracts.ArenaScale`: ln scale ~ Normal(0, 0.35)
    truncated to [0.5, 2], each environment's arena drawn at every reset) is opt-in, for
    the map-size arm only (the author's rule, 2026-11-15: every map is fixed-size until an
    agent has trained successfully on variable-size maps; `--set
    'arena_scale={"sigma": 0.35, "low": 0.5, "high": 2.0}'`). The rating lanes, evaluation
    and play keep their games' own sizes.
    active_fields: field-count sampling for the training pools (`bnb.contracts.ActiveFields`:
    each episode plays a drawn number of the game's field slots, the rest absent, and the
    model masks the absent ones, `Yemong.mask_absent_fields`), or None for the games' own
    (every field in every preset). The rating lanes play their games' own fields.
    team_sizes: team-size sampling for the training pools (arm A2; `bnb.contracts.TeamSizes`:
    each episode plays an N-against-N game, N drawn from low to high, the game's ships per
    team by default, within each mode's smallest team, Frontline 3; each team's slots past
    N benched, and the Yemong masks them, `Yemong.mask_benched`), or None (the default)
    for every slot in every game. Both BC's data and RL's games follow it. A2's setting is
    `--set 'team_sizes={"low": 1}'`: uniform over 1v1 to 5v5 in deathmatch and
    elimination, 3v3 to 5v5 in Frontline. The rating lanes play their games' own sizes.
    lineup: each slot's ship type by name in the game's ship set (`api.type_names`), the
    same for both teams, in every game of the batch (the rating lanes' too); None plays the
    game's own lineup (`Rules.team_types`). A tournament of such policies plays the same
    lineup (`bnb ratings --lineup`).
    lineup_draw: each episode of the batch (the rating lanes' too) draws its lineup, every
    slot one of the ship set's drawn types with equal chance (`api.DRAWN_LINEUPS`: Yongma
    or Jujak in the real-scale set), both teams flying it slot for slot, as the evaluation
    suite's games do (decided 2026-11-14); keyed by the episode (`bnb.runtime.rollout`).
    False plays the game's own lineup; a fixed `lineup` replaces the draw.
    compile_split: the learner's micro-batch loss compiles as two units, the unroll and the
    loss over its outputs (`Learner`'s split; compile time only, the same arithmetic within
    float rounding, `docs/rewrite/sections/compile-split`).
    mode_mix: the training mix as game presets and their weights, such as
    `{"real_scale_frontline": 8, "real_scale_deathmatch": 4, "real_scale_elimination": 4}`
    (arm A5), each game's share its weight over their sum; `arranged` builds the pools
    from it, every game with its own map, clocks, respawn and spawn groups. None (the
    default): the pools the preset gives (`TRAINING_MIXES`).
    self_play_seats: "both" (the default): a self-play game trains both seats; "one" (arm
    A6): no self-play pool, every game is against the league's seat, whose "latest" slots
    (`LeagueConfig.latest_share`, then required) are the self-play: a frozen copy of the
    live weights loaded after every update, so each game trains one seat
    (`training_lineup`)."""

    game: GameConfig
    environments: int = 2048
    updates: int = 100
    length: int = 128
    shards: int = 4
    lookahead: int = 128
    graph_ticks: int = 8
    lanes: LaneConfig = LaneConfig()
    model: str = "yemong"
    # a new build's settings that differ from the record defaults (see the docstring)
    yemong: YemongConfig = YemongConfig(
        spread_fit="coarsest",
        order="temporal_first",
        mode_inputs=MODE_INPUTS,
        frontline_timer_inputs=True,
        rotary_period=ROTARY_COARSEST,
        rotary_octaves=ROTARY_BANDS,
        rotary_ratio=ROTARY_RATIO,
        rotary_turn=ROTARY_TURN,
        heading_harmonics=(1, 2, 4),
        position_period=COLOUR_PERIOD,
        arena_code=True,
        geometry_head=True,
        # the champion's (2026-11-09-*-deadreckon-split-hiddenphysics): the delta next state
        # on the extrapolation baseline, split into the physics and error heads, the physics
        # head's hidden rows decayed (0.5 s then; halved for the x2 game)
        next_state="delta",
        delta_baseline="extrapolation",
        delta_split=True,
        physics_hidden_decay=0.25,  # s (0.5 before the x2 game: motion changes twice as fast)
        value_padding=PADDING_BINS,  # the HL-Gauss value targets' padding (2026-11-15)
    )
    relational: RelationalConfig = RelationalConfig(value_padding=PADDING_BINS)
    nearest: NearestConfig = NearestConfig(value_padding=PADDING_BINS, full_mode_inputs=True)
    ship_card: dict[str, tuple[float, ...]] | None = None  # None: the game's ship set's card
    league: LeagueConfig = LeagueConfig()
    pools: tuple[Pool, ...] = LINEUP
    learner: LearnerConfig = LearnerConfig()
    reward: RewardConfig = RewardConfig()
    precision: str = "bfloat16"
    host_store: bool = True
    seed: int = 0
    compiled: bool = True
    compile_split: bool = False
    objective: str = "ppo"
    teacher: TeacherConfig = TeacherConfig()
    stop: StopConfig = StopConfig()
    update_policies: str = "off"
    policy_snapshots: SnapshotSchedule = RL_SNAPSHOTS
    arena_scale: ArenaScale | None = None
    active_fields: ActiveFields | None = None
    team_sizes: TeamSizes | None = None
    lineup: tuple[str, ...] | None = None
    lineup_draw: bool = True
    # end a PPO run after this many consecutive updates with a combat-collapse alert
    # (`bnb.runs.triggers.CombatCollapse`), recording why; 0: never (the default)
    combat_collapse_stop: int = 0
    mode_mix: dict[str, int] | None = None
    self_play_seats: str = "both"

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective {self.objective!r} is not one of {OBJECTIVES}")
        if self.self_play_seats not in SELF_PLAY_SEATS:
            raise ValueError(
                f"self_play_seats {self.self_play_seats!r} is not one of {SELF_PLAY_SEATS}"
            )
        if self.objective == "bc" and (
            self.self_play_seats != "both"
            or self.mode_mix is not None
            or self.league.fixed_opponent is not None
        ):
            raise ValueError("self_play_seats, mode_mix and league.fixed_opponent are PPO's")
        if self.self_play_seats == "one" and not self.league.latest_share:
            raise ValueError(
                "self_play_seats 'one' needs league.latest_share > 0, its self-play share"
            )
        if self.mode_mix is not None:
            if not self.mode_mix or any(weight < 1 for weight in self.mode_mix.values()):
                raise ValueError(f"mode_mix {self.mode_mix} needs positive whole weights")
            unknown = sorted(set(self.mode_mix) - set(PRESETS))
            if unknown:
                raise ValueError(f"mode_mix names {unknown}, not game presets")
        if self.update_policies not in UPDATE_POLICIES:
            raise ValueError(
                f"update_policies {self.update_policies!r} is not one of {UPDATE_POLICIES}"
            )
        if self.objective == "bc" and self.learner.entropy_floor is not None:
            raise ValueError(
                "learner.entropy_floor is RL's: BC has no entropy term (a chain sets it with "
                "--rl-set)"
            )
        if self.combat_collapse_stop < 0:
            raise ValueError(f"combat_collapse_stop {self.combat_collapse_stop} is negative")
        if self.model not in MODELS:
            raise ValueError(f"model {self.model!r} is not one of {MODELS}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision {self.precision!r} is not one of {PRECISIONS}")
        if self.active_fields is not None:  # the game checks the range against its fields
            dataclasses.replace(self.game, active_fields=self.active_fields)
        if self.team_sizes is not None:
            if self.model != "yemong":
                raise ValueError(
                    f"team_sizes needs the Yemong model, which masks benched ships; "
                    f"{self.model!r} does not"
                )
            for pool in self.pools:  # each mode's sizes within the game's
                mode = pool_game(self.game, pool).mode
                self.team_sizes.probabilities(
                    self.game.ships_per_team, api.MINIMUM_SHIPS_PER_TEAM.get(mode, 1)
                )
        if self.shards < 1 or self.lookahead < 0:
            raise ValueError(f"{self.shards} shards and a lookahead of {self.lookahead} ticks")
        for name in MODELS:
            if getattr(self, name).bfloat16_matmuls:
                raise ValueError(
                    f"{name}.bfloat16_matmuls is set by precision ({self.precision!r}), "
                    f"not by the model's settings"
                )
        if self.lineup is not None:
            lineup_indices(self.lineup, self.game)  # fails for a wrong size or an unknown name
        elif self.lineup_draw:
            api.drawn_lineup_types(self.game.ship_set)  # fails for a set without drawn types
        card = api.default_card(self.game.ship_set)
        if self.ship_card is None:
            object.__setattr__(self, "ship_card", card)
        if self.ship_card.keys() != card.keys():
            missing = sorted(card.keys() - self.ship_card.keys())
            unknown = sorted(self.ship_card.keys() - card.keys())
            raise ValueError(f"ship_card lacks {missing} and has unknown fields {unknown}")
        for name, values in self.ship_card.items():
            if len(values) != len(card[name]):
                raise ValueError(
                    f"ship_card.{name} has {len(values)} values; the card has "
                    f"{len(card[name])} ship types"
                )

    def mix(self) -> tuple[tuple[str, float], ...] | None:
        """`mode_mix` as (preset, share) pairs, the shares summing to 1; None without one."""
        if self.mode_mix is None:
            return None
        total = sum(self.mode_mix.values())
        return tuple((name, weight / total) for name, weight in self.mode_mix.items())

    @property
    def writes_update_policies(self) -> bool:
        """Whether this run writes every update's policy files (`update_policies` for its
        stage: "rl" for PPO, "bc" for BC)."""
        stage = "bc" if self.objective == "bc" else "rl"
        return self.update_policies in ("all", stage)

    def uniform_start(self) -> int:
        """The environment steps the uniform average starts after: `league.uniform_from`,
        or by default the learner's warm-up (`LeagueConfig.uniform_start`)."""
        return self.league.uniform_start(self.learner.warmup_steps)

    def ship_types(self, device: torch.device | None = None) -> ShipTypes:
        """The run's ship card as tensors, on `device`."""
        return api.ship_types(device, **self.ship_card)


Build = YemongBuild | RelationalBuild | NearestBuild


def lineup_indices(lineup: tuple[str, ...], game: GameConfig) -> tuple[int, ...]:
    """A lineup's ship type names as type indices of the game's ship set, one per slot of a
    team. Fails for a lineup of another size than the team or a name not in the set."""
    names = api.type_names(game.ship_set)
    if len(lineup) != game.ships_per_team:
        raise ValueError(
            f"lineup {list(lineup)} has {len(lineup)} ships for a team of {game.ships_per_team}"
        )
    unknown = [name for name in lineup if name not in names]
    if unknown:
        raise ValueError(f"lineup names {unknown}, not ship types of {game.ship_set!r}: {names}")
    return tuple(names.index(name) for name in lineup)


def model_build(config: TrainConfig) -> Build:
    """The build record of the run's model: its settings (`TrainConfig.yemong`,
    `TrainConfig.relational` or `TrainConfig.nearest`) with the run's precision, and the
    numbers the run's game and ship card derive (none for the nearest-neighbour MLP)."""
    game, bfloat16 = config.game, config.precision == "bfloat16"
    if config.model == "nearest":
        return NearestBuild(dataclasses.replace(config.nearest, bfloat16_matmuls=bfloat16))
    if config.model == "relational":
        architecture = dataclasses.replace(config.relational, bfloat16_matmuls=bfloat16)
        return RelationalBuild(architecture, game.ships_per_team, game.num_ships)
    architecture = dataclasses.replace(config.yemong, bfloat16_matmuls=bfloat16)
    types = config.ship_types()
    return yemong_build(architecture, api.rules_of(game, types=types), types)


def build_model(build: Build, device: torch.device, seed: int = 0) -> torch.nn.Module:
    """A fresh model of the build record, initialised on the CPU from `seed` by parameter
    path (`bnb.models.initialise`), then moved to `device`."""
    if isinstance(build, NearestBuild):
        return Nearest(build, seed).to(device)
    kind = Relational if isinstance(build, RelationalBuild) else Yemong
    return kind(build, seed).to(device)


def model_controller(model: torch.nn.Module, game: GameConfig, distribution: bool = False):
    """The controller for a model: RelationalController for a stateless joint-policy model
    (Relational, Nearest), otherwise YemongController. distribution: its outputs carry the
    whole joint action distribution (the learner's, for `LearnerConfig.records_distribution`;
    Yemong only)."""
    if isinstance(model, Relational | Nearest):
        if distribution:
            raise ValueError("only a Yemong learner records its joint action distribution")
        return RelationalController(model)
    return YemongController(model, game.ships_per_team, game.num_ships, distribution)


def compiler(enabled: bool) -> Callable:
    """The runtime's stage cut for the learner: the micro-batch loss, compiled whole.

    Without Inductor's comprehensive padding: it pads the row stride of a wide intermediate
    (a code's logits' gradient, (L, R, N, ages, 162)) to 128 bytes, so the head kernel's
    and the weight gradient's (rows, outputs) views of it each need a copy; without it a
    real-scale micro-batch's step is 5% faster (`bench/learn_micro.py`)."""
    if not enabled:
        return lambda function: function
    return lambda function: torch.compile(
        function, fullgraph=True, dynamic=False, options={"comprehensive_padding": False}
    )


def mode_key(pool: Pool, game: GameConfig) -> str:
    """The gauge a pool's games count for (league spec 4): its game of the mix ("frontline"
    of "frontline/league"), or the run's mode for a single-game lineup."""
    return pool.name.split("/")[0] if "/" in pool.name else pool.mode or game.mode


class LeagueHooks(Protocol):
    """What the league (`bnb.league`) does inside an update: after each shard, after the
    whole collection (before learning), and after learning (returning its metrics)."""

    def after_shard(self, trainer: Trainer, shard: int) -> None:
        """Read the shard's results; end and begin drains."""

    def collected(self, trainer: Trainer) -> None:
        """The rollout is done and the weights are still the ones that played it."""

    def updated(self, trainer: Trainer) -> dict:
        """The update's learning is done; returns the league's series on the host."""

    def state(self) -> tuple[dict, dict[str, torch.Tensor]]:
        """What a training checkpoint keeps: (a JSON record, named tensors)."""


class Trainer:
    """The run's model (built from `model_build(config)`), the league's seat, opponents,
    rollout and learner, playing the config's ship card (`types`). `league`: the league's
    hooks, attached by `bnb.league` (None: the seat's slots keep what they hold).
    diagnostics: measure the learner's diagnostic series (diagnostics spec 1); none of them
    changes the update."""

    def __init__(self, config: TrainConfig, device: torch.device, diagnostics: bool = True) -> None:
        game = config.game
        check_pools(config)
        self.config = config
        self.model = build_model(model_build(config), device, config.seed)
        if isinstance(self.model, Yemong) and samples_field_counts(game, training_pools(config)):
            self.model.mask_absent_fields = True  # its copies (the seats', lanes') keep it
        if isinstance(self.model, Yemong) and samples_team_sizes(training_pools(config)):
            self.model.mask_benched = True  # its copies (the seats', lanes') keep it
        self.league: LeagueHooks | None = None
        # the trigger rules (`bnb.runs.triggers`), attached by the run's recorder: their
        # state is part of the training checkpoint
        self.triggers = None
        cloning = config.objective == "bc"
        seated = {name for pool in config.pools for name in pool.seats}
        self.types = types = config.ship_types(device)
        # The card's characteristics, when Imugi plays, teaches or is the ladder,
        # or the chaser plays (its distances are in the card's lengths).
        self.characteristics = (
            characteristics(types, api.rules_of(game))
            if cloning or {IMUGI_SEAT, OPPONENTS, "chaser"} & seated or config.lanes.environments
            else None
        )
        # the training pools and the rating lanes' (`bnb.runtime.lanes`), in one batch
        self.pools, self.lanes, self.batch_environments = lanes.lane_pools(
            training_pools(config),
            lanes.mode_games(config.pools, game, mode_key),
            config.environments,
            config.lanes,
        )
        lineup = Lineup(self.pools, self.batch_environments, device)
        opponent_pools = [self.pools[index] for index in lineup.pool_of.get(OPPONENTS, [])]
        # the league's gauges: one per mode of the opponents' pools, in order
        self.modes = tuple(dict.fromkeys(mode_key(pool, game) for pool in opponent_pools))
        self.mode_rows = tuple(
            sum(mode_key(pool, game) == mode for pool in opponent_pools) for mode in self.modes
        )
        builders = {
            LEARNER: lambda: model_controller(
                self.model, game, not cloning and config.learner.records_distribution
            ),
            OPPONENTS: lambda: OpponentSlots(
                model_controller(copy.deepcopy(self.model).requires_grad_(False), game),
                torch.tensor(
                    [self.modes.index(mode_key(pool, game)) for pool in opponent_pools],
                    device=device,
                ),
                len(self.modes),
                self.characteristics,
                game.ships_per_team,
                config.league,
            ),
            "chaser": lambda: Chaser(self.characteristics, game.ships_per_team),
            "random": lambda: Random(game.ships_per_team),
            # BC scores the learner against the teacher its labels describe
            IMUGI_SEAT: lambda: (
                TeacherOpponent(self.characteristics, game.ships_per_team, config.teacher)
                if cloning
                else Imugi(self.characteristics, game.ships_per_team)
            ),
        }
        if self.lanes:
            first_lane = len(config.pools)  # the lane pools follow the training pools
            builders[LANES] = lambda: LaneSeat(
                model_controller(copy.deepcopy(self.model).requires_grad_(False), game),
                self.lanes,
                lineup.pool_of[LANES] - first_lane,
                lineup.role[LANES],
                self.characteristics,
                game.ships_per_team,
            )
            seated.add(LANES)
        self.controllers = {
            name: build() for name, build in builders.items() if name in seated or name == LEARNER
        }
        self.opponents: OpponentSlots | None = self.controllers.get(OPPONENTS)
        self.lane_seat: LaneSeat | None = self.controllers.get(LANES)
        # the last update's lane games, {Lane: (wins, ties, losses)} from the first player's side
        self.lane_record: dict = {}
        self.rollout = Rollout(
            game,
            types,
            self.pools,
            self.controllers,
            LEARNER,
            self.batch_environments,
            config.length,
            device,
            compiled=config.compiled,
            reward_config=config.reward,
            shards=config.shards,
            host_store=config.host_store,
            extra_reads=Imugi.reads if cloning else (),
            lookahead=config.lookahead,
            graph_ticks=config.graph_ticks,
            ship_type=None
            if config.lineup is None
            else torch.tensor(lineup_indices(config.lineup, game) * api.TEAMS).repeat(
                self.batch_environments, 1
            ),
            lineup_types=api.drawn_lineup_types(game.ship_set)
            if config.lineup is None and config.lineup_draw
            else None,
            seed=config.seed,
        )
        lineup = self.rollout.lineup
        self.pool_names = tuple(pool.name for pool in lineup.pools)
        self.learner_pool = lineup.pool_of[LEARNER]
        rules, stage = self.rollout.rules, compiler(config.compiled)
        # each pool's game mode ("frontline", "team_deathmatch", "elimination"), and the
        # learner's: the advantage and combat series per mode
        pool_modes = tuple(pool_game(game, pool).mode for pool in lineup.pools)
        view_modes = tuple(dict.fromkeys(pool_modes[int(p)] for p in self.learner_pool.tolist()))
        view_mode = torch.tensor(  # on the host: the learner reads it once
            [view_modes.index(pool_modes[int(p)]) for p in self.learner_pool.tolist()]
        ).repeat(config.shards)
        if cloning:  # the labels are Imugi's distribution, transformed
            teacher = teacher_agent(self.characteristics, game.ships_per_team, config.teacher)
            self.ppo = BehaviourCloning(
                self.model,
                config.learner,
                rules,
                types,
                config.reward.outcome,
                device,
                teacher,
                config.teacher,
                self.pool_names[: len(config.pools)],  # the learner plays no lane
                self.learner_pool.repeat(config.shards),
                stage=stage,
                seed=config.seed,
                diagnostics=diagnostics,
                split=config.compile_split,
            )
        else:
            self.ppo = Learner(
                self.model,
                config.learner,
                rules,
                types,
                config.reward.outcome,
                device,
                stage=stage,
                seed=config.seed,
                diagnostics=diagnostics,
                split=config.compile_split,
                reward_weights=config.reward.weights,
                view_modes=view_modes,
                view_mode=view_mode,
            )
        self.updates_done = 0
        # BC's stop rule (`bnb.runtime.cloning.update` feeds it): its window is part of the
        # training checkpoint
        self.stop_rule = StopRule(config.stop) if cloning else None
        environment_pool = torch.repeat_interleave(
            torch.arange(len(lineup.pools), device=device),
            torch.tensor(lineup.sizes, device=device),
        )
        # the analytics read one shard at a time: one shard's views (R,)
        self.analytics = analytics.Analytics(
            self.pool_names,
            self.learner_pool,
            environment_pool,
            rules.tick_length,
            game.mode,
            view_fogged=lineup.fogged[lineup.environments[LEARNER]],
            environment_fogged=lineup.fogged,
            pool_modes=pool_modes,
        )

    @property
    def steps_per_update(self) -> int:
        """Environment steps one update collects."""
        return self.config.environments * self.config.length * self.config.shards

    def collect(self):
        """Every shard of one update, then the lookahead, and the bootstrap after it (no
        learning): (the store, the bootstrap). The league's hooks run after each shard and
        after the lookahead (as shard `shards`); the lanes' games are taken into
        `lane_record`."""
        if self.lane_seat is not None:
            self.lane_seat.load(LIVE, self.model.state_dict())
        for shard in range(self.config.shards):
            self.rollout.run(shard)
            if self.league is not None:
                self.league.after_shard(self, shard)
        if self.config.lookahead:
            self.rollout.run_lookahead()
            if self.league is not None:
                self.league.after_shard(self, self.config.shards)
        bootstrap = self.rollout.bootstrap()
        if self.lane_seat is not None:
            self.lane_record = lanes.results(self.rollout.take_tally(LANES), self.lanes)
        return self.rollout.store, bootstrap

    @torch.no_grad()
    def reset_opponents(self) -> None:
        """Every opponent slot holds the model's current weights (a warm start, before a
        league draws the slots)."""
        if self.opponents is not None:
            self.opponents.fill(self.model.state_dict())

    def update(self) -> dict:
        """One update: collect every shard, learn from them, with the league's hooks around
        the learning. Returns the update's series on the host (`bnb.runtime.metrics`), read
        back in one transfer, and its two phases' wall-clock times
        (`train/time/rollout_seconds`, `train/time/learner_seconds`)."""
        started = time.perf_counter()
        store, bootstrap = self.collect()
        collector = Collector()
        collector.add(self.analytics.episodes(self.rollout.take_episodes()))
        game_sums, outcome_tables = self.rollout.take_games()
        collector.set(self.analytics.games(game_sums, outcome_tables))
        for shard in range(store.shards):
            collector.add(self.analytics.shard(store.shard(analytics.NAMES, shard)))
        if self.league is not None:
            self.league.collected(self)
        collected = time.perf_counter()
        cadence = self.ppo.diagnostics and metrics.due(self.updates_done + 1)
        collector.set(
            self.ppo.update(store, bootstrap, self.steps_per_update, cadence=cadence, heads=cadence)
        )
        self.updates_done += 1
        collector.set(
            {
                "counters/updates": self.updates_done,
                "counters/environment_steps": self.ppo.environment_steps,
                "counters/nonfinite_beliefs": self.nonfinite_beliefs(),
            }
        )
        row = collector.read()  # the update's one transfer
        learned = time.perf_counter()
        if cadence:
            row["train/time/cadence_seconds"] = self.ppo.cadence_stopwatch.seconds()
            if outcome_tables is not None:  # the cadence's own read
                calibration = metrics.read_back(outcome_tables["calibration"].flatten())
                rows = [calibration[i : i + 3] for i in range(0, len(calibration), 3)]
                row["outcome/reliability"] = analytics.reliability_table(rows).as_json()
        row.update(lanes.series(self.lane_record))
        if self.league is not None:
            row.update(self.league.updated(self))
        row["train/time/rollout_seconds"] = collected - started
        row["train/time/learner_seconds"] = learned - collected
        return row

    def nonfinite_beliefs(self) -> torch.Tensor | int:
        """Non-finite belief entries the learner's carry has counted since its rollouts began,
        on the device (Yemong's `Carry.nonfinite`; 0 for a model without one)."""
        counts = getattr(self.rollout.carry["controllers"][LEARNER], "nonfinite", None)
        return 0 if counts is None else counts.sum()


def win_rate(trainer: Trainer, opponent: str = "chaser", games: int = 256) -> dict:
    """The learner against a scripted opponent over `games` fresh games (one per
    environment, every clock from zero, the seat drawn per environment): the shares won,
    drawn and lost."""
    config = trainer.config
    device = next(trainer.model.parameters()).device
    controllers = {LEARNER: trainer.controllers[LEARNER], opponent: trainer.controllers[opponent]}
    pool = Pool("evaluation", 1.0, (LEARNER, opponent), config.pools[0].fog_share)
    return shares(
        first_results(
            config.game,
            trainer.types,
            (pool,),
            controllers,
            LEARNER,
            games,
            device,
            config.length,
            config.compiled,
            seed=config.seed,
        )
    )


def train(
    config: TrainConfig, device: torch.device, log: Callable[[dict], None] = print
) -> Trainer:
    """The loop. Process-wide settings (the seed, `bnb.cli.set_precision`) are the entry
    point's."""
    trainer = Trainer(config, device)
    for _ in range(config.updates):
        log(trainer.update())
    return trainer
