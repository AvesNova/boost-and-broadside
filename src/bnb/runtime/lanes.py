"""The rating lanes: extra environments of the rollout that play rated matchups, in the same
tensors as the training games, never stored for the learner nor counted in its losses.

A matchup is a pair of players. A player is a network (`LIVE`, the weights the learner is
training; or a weight set loaded by name: the averages of the live weights, which the
league and behaviour cloning keep, or `league_<k>`, the entry the league's network slot k
holds) or a rung of the
stationary ladder (`random`, `rung_T<t>`, `scripted`: the scripted agent at temperature t,
uniform random actions for `random`). Every mode of the training mix plays every matchup: the
lanes are split evenly over the modes, and within a mode by the matchups' weights. Each
matchup is a pool of its own (`<mode>/lane/<first>_vs_<second>`) whose two seats are both
the one controller `LANES` (`LaneSeat`), so the seat is redrawn at every reset as in any
pool.

The rollout's shapes are fixed at capture: which row plays which player is static, and only
the weights of a network player change, copied in place between segments (`LaneSeat.load`).
The rollout tallies every ended lane game by matchup, from the first player's side.
"""

import dataclasses
import math

import torch
from torch.utils import _pytree

from bnb.contracts import Action, GameConfig, Obs, Outputs, Pool
from bnb.contracts.tree import contract
from bnb.controllers.scripted import Scripted
from bnb.game.handling.characteristics import Characteristics

from . import stacked
from .opponents import (
    OUTCOMES,
    SCRIPTED_PLAYER,
    Rung,
    ladder_name,
    ladder_rung,
    whole_game_begins,
)

LANES = "lanes"  # the lanes' controller name
LIVE = "live"  # the player the learner trains
# the averages of the live weights the league (and behaviour cloning) keeps
# (`bnb.league.pool.Averages`); RL's lanes rate both
UNIFORM_AVERAGE = "uniform_average"
EXPONENTIAL_AVERAGE = "exponential_average"
LEAGUE_PREFIX = "league_"  # a league slot's player: the entry the league's slot k holds


@dataclasses.dataclass(frozen=True)
class Matchup:
    """Two players and the matchup's share of each mode's lanes (relative to the others')."""

    first: str
    second: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.first == self.second:
            raise ValueError(f"a matchup needs two players, got {self.first} twice")
        if not self.weight > 0.0:
            raise ValueError(f"matchup {self.name}: weight {self.weight} must be positive")

    @property
    def name(self) -> str:
        """The matchup's name, `<first>_vs_<second>`."""
        return f"{self.first}_vs_{self.second}"


def ladder_chain(rungs: tuple[Rung, ...], weight: float) -> tuple[Matchup, ...]:
    """Matchups that calibrate a ladder from the scripted agent down: each rung against the
    next one up (`rungs` those below the scripted agent, weakest first), the top one against
    the scripted agent."""
    names = [ladder_name(t) for t in rungs] + [SCRIPTED_PLAYER]
    return tuple(
        Matchup(low, high, weight) for low, high in zip(names[:-1], names[1:], strict=True)
    )


def league_slot(index: int) -> str:
    """The lane player of the league's network slot `index`: `league_<index>`."""
    return f"{LEAGUE_PREFIX}{index}"


LEAGUE_SLOTS = 4  # `bnb.runtime.opponents.LeagueConfig.slots`' default
# The default matchups of a reinforcement-learning run (the author, 2026-11-12): every
# network player (the live policy, both averages, each league slot's entry) against the
# full-strength scripted agent, at equal weight, so each has the same lane rows and the
# seven stack as one network run with no padding (`LaneSeat.network_runs`). A game tells
# the rating fit p (1 - p) about its players' gap, 0.25 at even odds and 0.004 at the 950
# points v1's policy reached above the scripted agent. (The ladder's rung pairs and the
# network-against-network matchups of 2026-11-05 are gone; the training pool's ladder is
# unchanged.)
RL_MATCHUPS = tuple(
    Matchup(player, SCRIPTED_PLAYER, 1.0)
    for player in (
        LIVE,
        EXPONENTIAL_AVERAGE,
        UNIFORM_AVERAGE,
        *(league_slot(index) for index in range(LEAGUE_SLOTS)),
    )
)
# Behaviour cloning's: the live clone and both averages of its weights against the raw
# scripted agent (decisions: BC round three), a third of each mode's lanes each (the
# author, 2026-11-12: no reason for the live clone's half weight; equal rows also make the
# three one stacked network run with no padding, `LaneSeat.network_runs`).
BC_MATCHUPS = (
    Matchup(LIVE, SCRIPTED_PLAYER, 1.0),
    Matchup(UNIFORM_AVERAGE, SCRIPTED_PLAYER, 1.0),
    Matchup(EXPONENTIAL_AVERAGE, SCRIPTED_PLAYER, 1.0),
)


@dataclasses.dataclass(frozen=True)
class LaneConfig:
    """environments: the lanes, beyond the training environments (0: none); matchups: what
    they play; window: the updates of games that rate a moving player (a network: the live
    policy, an average), while a stationary player's (a rung, a snapshot) count from the
    start."""

    environments: int = 1024
    matchups: tuple[Matchup, ...] = RL_MATCHUPS
    window: int = 10

    def __post_init__(self) -> None:
        if self.environments < 0:
            raise ValueError(f"lanes: {self.environments} environments")
        if self.environments and not self.matchups:
            raise ValueError("lanes need at least one matchup")
        if self.window < 1:
            raise ValueError(f"lanes: a window of {self.window} updates")
        names = [matchup.name for matchup in self.matchups]
        if len(set(names)) != len(names):
            raise ValueError(f"lanes: a matchup is listed twice in {names}")


@dataclasses.dataclass(frozen=True)
class Lane:
    """One lane pool: its mode and its matchup."""

    mode: str
    matchup: Matchup

    @property
    def name(self) -> str:
        """The pool's name, `<mode>/lane/<matchup>`."""
        return f"{self.mode}/lane/{self.matchup.name}"


def split(total: int, weights: list[float]) -> list[int]:
    """`total` in whole parts proportional to `weights`, each at least 1, by the largest
    remainder."""
    if total < len(weights):
        raise ValueError(f"{total} lanes cannot give each of {len(weights)} pools one")
    rest = total - len(weights)
    shares = [rest * weight / sum(weights) for weight in weights]
    parts = [1 + math.floor(share) for share in shares]
    order = sorted(
        range(len(weights)), key=lambda i: shares[i] - math.floor(shares[i]), reverse=True
    )
    for index in order[: total - sum(parts)]:
        parts[index] += 1
    return parts


def lane_pools(
    pools: tuple[Pool, ...], modes: dict[str, Pool], environments: int, config: LaneConfig
) -> tuple[tuple[Pool, ...], tuple[Lane, ...], int]:
    """The training pools and the lanes' as one lineup: (every pool, shares of the whole
    batch; the lanes in their pools' order; the batch's environments). pools: the training
    pools over `environments`; modes: each mode's game, as a pool of it (its fog share and
    clocks are the lanes'). The lanes are split evenly over the modes, then by the matchups'
    weights. A lane plays its game's own arena every episode (no map-size sampling: a
    rating compares games of one size)."""
    total = environments + config.environments
    if not config.environments:
        return pools, (), environments
    scaled = tuple(
        dataclasses.replace(pool, share=pool.share * environments / total) for pool in pools
    )
    lanes, lane_sizes = [], []
    per_mode = split(config.environments, [1.0] * len(modes))
    for (mode, game), count in zip(modes.items(), per_mode, strict=True):
        sizes = split(count, [matchup.weight for matchup in config.matchups])
        for matchup, size in zip(config.matchups, sizes, strict=True):
            lanes.append(Lane(mode, matchup))
            lane_sizes.append((game, size))
    lane = tuple(
        dataclasses.replace(
            game,
            name=item.name,
            share=size / total,
            seats=(LANES, LANES),
            fog_share=round(game.fog_share * size) / size,
            arena_scale=None,
        )
        for item, (game, size) in zip(lanes, lane_sizes, strict=True)
    )
    return scaled + lane, tuple(lanes), total


def mode_games(pools: tuple[Pool, ...], game: GameConfig, mode_of) -> dict[str, Pool]:
    """Each mode of the training pools (`mode_of(pool, game)`), in order, with its first
    pool."""
    out: dict[str, Pool] = {}
    for pool in pools:
        out.setdefault(mode_of(pool, game), pool)
    return out


@contract
class LaneCarry:
    """Per lane row (R leading): model, the network players' carry (their controller's);
    started (R,) int64, the generation of the row's player's weights when its game began
    (`LaneSeat.load` with `fresh`), or -1 for the row's first game, which the rollout's
    staggered clocks (or a resume) cut short and so is never tallied; stepped (R,) bool,
    whether the row has played a tick. ladder: the ladder players' carry, one per ladder
    row (`LaneSeat.ladder_rows`' order)."""

    model: object
    started: torch.Tensor
    stepped: torch.Tensor
    ladder: object


class _Decision(torch.nn.Module):
    """A network controller's decision as a module call, for `functional_call`."""

    def __init__(self, controller) -> None:
        super().__init__()
        self.model = controller.model
        self.controller = controller

    def forward(self, carry, obs: Obs, reset: torch.Tensor, decision_key: torch.Tensor):
        action, carry, _ = self.controller.step(carry, obs, reset, decision_key)
        return action.packed(), carry


# A run of lane players (`LaneSeat.network_runs`) takes the next player while its rows are
# within this fraction of the run's first player's: the padding it costs against the network
# copy a run of its own would add to the compiled stage (about 8,000 traced ops, 2.5-3 min
# of a cold compile, `compile-split` notes). Up to about twice (2.05x, room for the splits'
# rounding): BC's lanes (three players of equal rows) are one run, and RL's seven
# (`RL_MATCHUPS`).
RUN_SLACK = 1.05


class LaneSeat:
    """Every lane row's player, in one controller. controller: a network controller of the
    live policy's architecture over a model of its own (the forward swaps each player's
    weights in); lanes: the lane pools in order; pool (R,) and role (R,): each lane row's
    lane (an index into `lanes`) and seat in it, from the lineup; characteristics and
    ships_per_team: the scripted agent's, for the ladder players. Every network player
    starts with the controller's model's weights.

    A game is tallied only if both its players still have the weights it began with: a
    player loaded `fresh` (another snapshot into a slot) starts a new generation, and the
    games in progress then are left out, not credited to the new weights."""

    def __init__(
        self,
        controller,
        lanes: tuple[Lane, ...],
        pool: torch.Tensor,
        role: torch.Tensor,
        characteristics: Characteristics,
        ships_per_team: int,
    ) -> None:
        device = pool.device
        players = tuple(
            dict.fromkeys(
                player for lane in lanes for player in (lane.matchup.first, lane.matchup.second)
            )
        )
        self.networks = tuple(p for p in players if ladder_rung(p) is None)
        self.decision = _Decision(controller)
        self.decision.requires_grad_(False)
        self.reads = tuple(dict.fromkeys(controller.reads + Scripted.reads))
        self.lanes, self.role = lanes, role
        first = torch.tensor([players.index(lane.matchup.first) for lane in lanes], device=device)
        second = torch.tensor([players.index(lane.matchup.second) for lane in lanes], device=device)
        player = torch.where(role == 0, first[pool], second[pool]).cpu()
        self.column = pool  # what each row's games are tallied under: its lane
        self.rows = role.shape[0]
        rows_of = {
            name: (player == players.index(name)).nonzero().flatten().to(device)
            for name in self.networks
        }
        # The network players ordered by their row count, and cut into runs of players with
        # nearly as many rows as each other (within RUN_SLACK of the run's first): each run
        # is one stacked forward over its players' rows (`bnb.runtime.stacked`), each padded
        # to the run's first with its own first row, and its weights are a slice of the
        # stack, not a copy. One run for every player padded the league's slots to the live
        # policy's rows (about half again as many network rows); a run per exact count put
        # a network copy per count into the compiled stage (counts differ by a few rows).
        self.networks = tuple(sorted(self.networks, key=lambda name: -rows_of[name].numel()))
        self.network_rows = [rows_of[name] for name in self.networks]
        # (players, (players, most) rows padded, the real entries' flat positions, their rows)
        self.network_runs: list[tuple[slice, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        start = 0
        while start < len(self.networks):
            most = self.network_rows[start].numel()
            end = start + 1
            while end < len(self.networks) and (
                self.network_rows[end].numel() * (1.0 + RUN_SLACK) >= most
            ):
                end += 1
            if most:
                run = self.network_rows[start:end]
                index = torch.stack(
                    [
                        torch.cat(
                            [r, (r[:1] if r.numel() else r.new_zeros(1)).expand(most - r.numel())]
                        )
                        for r in run
                    ]
                )
                # made here, not in the graph: a boolean mask's size is data-dependent there
                kept = torch.cat(
                    [
                        position * most + torch.arange(r.numel(), device=device)
                        for position, r in enumerate(run)
                    ]
                )
                self.network_runs.append((slice(start, end), index, kept, index.reshape(-1)[kept]))
            start = end
        ladder = torch.tensor([ladder_rung(name) is not None for name in players])
        ladder_rows = ladder[player].nonzero().flatten()
        self.ladder_rows = ladder_rows.to(device)
        rungs = [ladder_rung(players[int(index)]) for index in player[ladder_rows]]
        self.ladder_agent = Scripted(
            characteristics,
            ships_per_team,
            temperature=torch.tensor(
                [rung.temperature for rung in rungs], dtype=torch.float32, device=device
            ),
            share=torch.tensor([rung.share for rung in rungs], device=device),
        )
        state = self.decision.state_dict()
        self.parameters = {
            name: value.detach()[None].repeat((len(self.networks),) + (1,) * value.dim()).clone()
            for name, value in state.items()
        }
        self.columns, self.mode_count = len(lanes), 1
        # each row's network player (its index in `networks`), or the spare last entry of
        # `generation` for a ladder row, which never changes
        network = torch.full((self.rows,), len(self.networks), dtype=torch.int64)
        for index, rows in enumerate(self.network_rows):
            network[rows.cpu()] = index
        self.player_generation = network.to(device)
        self.generation = torch.zeros(len(self.networks) + 1, dtype=torch.int64, device=device)
        # the row of the same game's other seat
        partner = torch.empty(self.rows, dtype=torch.int64)
        pool_cpu, role_cpu = pool.cpu(), role.cpu()
        for lane in range(len(lanes)):
            first_rows = ((pool_cpu == lane) & (role_cpu == 0)).nonzero().flatten()
            second_rows = ((pool_cpu == lane) & (role_cpu == 1)).nonzero().flatten()
            partner[first_rows], partner[second_rows] = second_rows, first_rows
        self.partner = partner.to(device)

    # ---- the controller

    def init_carry(self, views: int, device: torch.device) -> LaneCarry:
        """Every row's network carry (a ladder row's is never read) and the ladder rows'."""
        return LaneCarry(
            model=self.decision.controller.init_carry(views, device),
            started=torch.full((views,), -1, dtype=torch.int64, device=device),
            stepped=torch.zeros(views, dtype=torch.bool, device=device),
            ladder=self.ladder_agent.init_carry(self.ladder_rows.numel(), device),
        )

    def step(self, carry: LaneCarry, obs: Obs, reset: torch.Tensor, decision_key: torch.Tensor):
        """One decision for every lane row; see `Controller.step`."""
        S = obs.ships.shape[-2] // 2
        action = torch.zeros((self.rows, S, 3), dtype=torch.int32, device=reset.device)
        model_carry = carry.model
        for players, index, kept, rows in self.network_runs:  # a stacked forward each
            inputs = _pytree.tree_map_only(
                torch.Tensor,
                lambda x, index=index: x[index.reshape(-1)].unflatten(0, index.shape),
                (carry.model, obs, reset, decision_key),
            )
            weights = {name: value[players] for name, value in self.parameters.items()}
            packed, chosen = stacked.call(self.decision, weights, inputs)
            action = action.index_copy(0, rows, packed.flatten(0, 1).index_select(0, kept))
            if chosen is not None:
                model_carry = _pytree.tree_map(
                    lambda whole, part, rows=rows, kept=kept: whole.index_copy(
                        0, rows, part.flatten(0, 1).index_select(0, kept)
                    ),
                    model_carry,
                    chosen,
                )
        ladder_carry = carry.ladder
        if self.ladder_rows.numel():
            rows = self.ladder_rows
            ladder_obs = _pytree.tree_map_only(torch.Tensor, lambda x: x[rows], obs)
            ladder_action, ladder_carry, _ = self.ladder_agent.step(
                carry.ladder, ladder_obs, reset[rows], decision_key[rows]
            )
            action = action.index_copy(0, rows, ladder_action.packed())
        generation = self.generation[self.player_generation]
        # the row's first game, cut short, keeps -1 (`whole_game_begins`)
        started = torch.where(whole_game_begins(reset, carry.stepped), generation, carry.started)
        carry = LaneCarry(
            model=model_carry,
            started=started,
            stepped=torch.ones_like(carry.stepped),
            ladder=ladder_carry,
        )
        return Action.unpack(action), carry, Outputs()

    def tally(self, carry: LaneCarry, done: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        """Each ended lane game (`done` (R,)) once, by (lane, outcome) with the outcome from
        the first player's side (`result` (R,) int32 is the row's own: +1 won, 0 tied, -1
        lost): flat (lanes * 3,) int64, as `bnb.runtime.opponents.OUTCOMES`. A game either of
        whose players' weights changed since it began is left out."""
        outcome = (1 - result).to(torch.int64)  # the row's win 0, tie 1, loss 2
        flat = self.column * len(OUTCOMES) + outcome
        same = carry.started == self.generation[self.player_generation]
        counted = done & (self.role == 0) & same & same[self.partner]
        counts = torch.zeros(self.columns * len(OUTCOMES), dtype=torch.int64, device=done.device)
        return counts.index_add(0, flat, counted.to(torch.int64))

    # ---- the host's side, between segments

    @torch.no_grad()
    def load(self, player: str, state: dict[str, torch.Tensor], fresh: bool = False) -> None:
        """Copy a network's state (its model's state dict) into `player`'s weights, in place
        (a player the lanes do not play is ignored). fresh: another player now (a snapshot
        into a slot), so the games in progress are not tallied; else the same player's
        weights moved on (the live policy, an average)."""
        if player not in self.networks:
            return
        index = self.networks.index(player)
        if fresh:
            self.generation[index] += 1
        for name, value in state.items():
            self.parameters[f"model.{name}"][index].copy_(value)


def results(tally: torch.Tensor, lanes: tuple[Lane, ...]) -> dict[Lane, tuple[int, int, int]]:
    """A taken tally (lanes * 3,) as {lane: (wins, ties, losses)} from the first player's
    side."""
    rows = tally.reshape(len(lanes), len(OUTCOMES)).tolist()
    return {lane: tuple(row) for lane, row in zip(lanes, rows, strict=True)}


def series(record: dict[Lane, tuple[int, int, int]]) -> dict:
    """An update's lane games as series: `lanes/games/<mode>/<matchup>` and, with games,
    `lanes/score/<mode>/<matchup>` (the first player's wins plus half the ties, per game)."""
    out = {}
    for lane, (wins, ties, losses) in record.items():
        games = wins + ties + losses
        key = f"{lane.mode}/{lane.matchup.name}"
        out[f"lanes/games/{key}"] = games
        if games:
            out[f"lanes/score/{key}"] = (wins + 0.5 * ties) / games
    return out
