"""The rating lanes: extra environments of the rollout that play rated matchups, in the same
tensors as the training games, never stored for the learner nor counted in its losses.

A matchup is a pair of players. A player is a network (`LIVE`, the weights the learner is
training; or a weight set the league loads by name: its averages, or a snapshot slot,
`snapshot_<i>`, which holds a frozen past live policy) or a rung of the
stationary ladder (`random`, `rung_<p>`, `scripted`: the scripted agent at probability p,
uniform random actions otherwise). Every mode of the training mix plays every matchup: the
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

from .opponents import OUTCOMES

LANES = "lanes"  # the lanes' controller name
LIVE = "live"  # the player the learner trains
SCRIPTED_PLAYER = "scripted"
EXPONENTIAL_AVERAGE = "exponential_average"  # the league's average the lanes rate by default
RUNG_PREFIX = "rung_"
SNAPSHOT_PREFIX = "snapshot_"  # a snapshot slot's player: a frozen past live policy


def ladder_probability(player: str) -> float | None:
    """A ladder player's scripted probability ("random" 0, "scripted" 1, "rung_<p>" p), or
    None for a network player."""
    if player == "random":
        return 0.0
    if player == SCRIPTED_PLAYER:
        return 1.0
    if player.startswith(RUNG_PREFIX):
        return float(player[len(RUNG_PREFIX) :])
    return None


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


def ladder_chain(rungs: tuple[float, ...], weight: float) -> tuple[Matchup, ...]:
    """Matchups that calibrate a ladder from the scripted agent down: each rung against the
    next one up (`rungs` increasing probabilities below 1), the top one against the
    scripted agent."""
    names = ["random" if p == 0.0 else f"{RUNG_PREFIX}{p:g}" for p in rungs] + [SCRIPTED_PLAYER]
    return tuple(
        Matchup(low, high, weight) for low, high in zip(names[:-1], names[1:], strict=True)
    )


def snapshot_slot(index: int) -> str:
    """The lane player of snapshot slot `index`: `snapshot_<index>`."""
    return f"{SNAPSHOT_PREFIX}{index}"


def snapshot_matchups(slots: int, weight: float, anchor_weight: float) -> tuple[Matchup, ...]:
    """The snapshot slots' matchups: each slot against the live policy and the averaged
    agent (`weight`), against the next slot round the ring (`weight`; a slot's neighbours
    in age are among them whatever the slots' order), and against the scripted agent
    (`anchor_weight`)."""
    names = [snapshot_slot(index) for index in range(slots)]
    out = []
    for name in names:
        out += [Matchup(name, LIVE, weight), Matchup(name, EXPONENTIAL_AVERAGE, weight)]
    if slots > 1:
        out += [
            Matchup(first, second, weight)
            for first, second in zip(names, names[1:] + names[:1], strict=True)
            if slots > 2 or first < second
        ]
    out += [Matchup(name, SCRIPTED_PLAYER, anchor_weight) for name in names]
    return tuple(out)


SNAPSHOT_SLOTS = 4
# The default matchups of a reinforcement-learning run (author, 2026-11-04). A game tells the
# fit p (1 - p) about the gap between its players, 0.25 at even odds and 0.004 at the 950
# points v1's policy reached above the scripted agent, so the moving players are rated
# against opponents near their own strength: per mode, the live policy and the averaged
# agent against each other, against frozen snapshots of the live policy in a few slots,
# and against the scripted agent; the snapshots against each other and the scripted agent,
# which chain them to its 1000; and the ladder's rungs chained up to the scripted agent,
# which calibrate the ladder. The snapshots and rungs are stationary, so their games gather
# over the run, and the rungs' lanes run no network: a small share.
RL_MATCHUPS = (
    Matchup(LIVE, SCRIPTED_PLAYER, 0.5),
    Matchup(EXPONENTIAL_AVERAGE, SCRIPTED_PLAYER, 0.5),
    Matchup(LIVE, EXPONENTIAL_AVERAGE, 0.5),
    *snapshot_matchups(SNAPSHOT_SLOTS, 1.0, 0.5),
    *ladder_chain((0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95), 0.05),
)
# Behaviour cloning's: the live policy against the raw scripted agent (its score stop).
BC_MATCHUPS = (Matchup(LIVE, SCRIPTED_PLAYER, 1.0),)


@dataclasses.dataclass(frozen=True)
class LaneConfig:
    """environments: the lanes, beyond the training environments (0: none); matchups: what
    they play; window: the updates of games that rate a moving player (a network: the live
    policy, an average), while a stationary player's (a rung, a snapshot) count from the
    start; snapshot_every: updates between frozen snapshots of the live policy, each into
    the next snapshot slot round robin (the slots are the `snapshot_<i>` players the
    matchups name)."""

    environments: int = 1024
    matchups: tuple[Matchup, ...] = RL_MATCHUPS
    window: int = 10
    snapshot_every: int = 8

    def __post_init__(self) -> None:
        if self.environments < 0:
            raise ValueError(f"lanes: {self.environments} environments")
        if self.environments and not self.matchups:
            raise ValueError("lanes need at least one matchup")
        if self.window < 1 or self.snapshot_every < 1:
            raise ValueError(
                f"lanes: a window of {self.window}, snapshots every {self.snapshot_every}"
            )
        names = [matchup.name for matchup in self.matchups]
        if len(set(names)) != len(names):
            raise ValueError(f"lanes: a matchup is listed twice in {names}")

    @property
    def slots(self) -> int:
        """The snapshot slots the matchups name."""
        names = {p for m in self.matchups for p in (m.first, m.second)}
        return sum(1 for name in names if name.startswith(SNAPSHOT_PREFIX))


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
    weights."""
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
    """Per lane row (R leading): model, the network players' carry (their controller's)."""

    model: object


class _Decision(torch.nn.Module):
    """A network controller's decision as a module call, for `functional_call`."""

    def __init__(self, controller) -> None:
        super().__init__()
        self.model = controller.model
        self.controller = controller

    def forward(self, carry, obs: Obs, reset: torch.Tensor):
        action, carry, _ = self.controller.step(carry, obs, reset)
        return action.packed(), carry


class LaneSeat:
    """Every lane row's player, in one controller. controller: a network controller of the
    live policy's architecture over a model of its own (the forward swaps each player's
    weights in); lanes: the lane pools in order; pool (R,) and role (R,): each lane row's
    lane (an index into `lanes`) and seat in it, from the lineup; characteristics: the
    scripted agent's, for the ladder players. Every network player starts with the
    controller's model's weights."""

    def __init__(
        self,
        controller,
        lanes: tuple[Lane, ...],
        pool: torch.Tensor,
        role: torch.Tensor,
        characteristics: Characteristics,
    ) -> None:
        device = pool.device
        players = tuple(
            dict.fromkeys(
                player for lane in lanes for player in (lane.matchup.first, lane.matchup.second)
            )
        )
        self.networks = tuple(p for p in players if ladder_probability(p) is None)
        self.decision = _Decision(controller)
        self.decision.requires_grad_(False)
        self.reads = tuple(dict.fromkeys(controller.reads + Scripted.reads))
        self.lanes, self.role = lanes, role
        first = torch.tensor([players.index(lane.matchup.first) for lane in lanes], device=device)
        second = torch.tensor([players.index(lane.matchup.second) for lane in lanes], device=device)
        player = torch.where(role == 0, first[pool], second[pool]).cpu()
        self.column = pool  # what each row's games are tallied under: its lane
        self.rows = role.shape[0]
        self.network_rows = [
            (player == players.index(name)).nonzero().flatten().to(device) for name in self.networks
        ]
        ladder = torch.tensor([ladder_probability(name) is not None for name in players])
        ladder_rows = ladder[player].nonzero().flatten()
        self.ladder_rows = ladder_rows.to(device)
        self.ladder_agent = Scripted(
            characteristics,
            scripted_probability=torch.tensor(
                [ladder_probability(players[int(index)]) for index in player[ladder_rows]],
                dtype=torch.float32,
                device=device,
            ),
        )
        state = self.decision.state_dict()
        self.parameters = {
            name: value.detach()[None].repeat((len(self.networks),) + (1,) * value.dim()).clone()
            for name, value in state.items()
        }
        self.columns, self.mode_count = len(lanes), 1

    # ---- the controller

    def init_carry(self, views: int, device: torch.device) -> LaneCarry:
        """Every row's network carry (a ladder row's is never read)."""
        return LaneCarry(model=self.decision.controller.init_carry(views, device))

    def step(self, carry: LaneCarry, obs: Obs, reset: torch.Tensor):
        """One decision for every lane row; see `Controller.step`."""
        S = obs.ships.shape[-2] // 2
        action = torch.zeros((self.rows, S, 3), dtype=torch.int32, device=reset.device)
        model_carry = carry.model
        for index, rows in enumerate(self.network_rows):
            if not rows.numel():
                continue
            parameters = {name: value[index] for name, value in self.parameters.items()}
            inputs = _pytree.tree_map_only(
                torch.Tensor, lambda x: x[rows], (carry.model, obs, reset)
            )
            packed, chosen = torch.func.functional_call(self.decision, parameters, inputs)
            action = action.index_copy(0, rows, packed)
            if chosen is not None:
                model_carry = _pytree.tree_map(
                    lambda whole, part: whole.index_copy(0, rows, part), model_carry, chosen
                )
        if self.ladder_rows.numel():
            rows = self.ladder_rows
            ladder_obs = _pytree.tree_map_only(torch.Tensor, lambda x: x[rows], obs)
            ladder_action, _, _ = self.ladder_agent.step(None, ladder_obs, reset[rows])
            action = action.index_copy(0, rows, ladder_action.packed())
        return Action.unpack(action), LaneCarry(model=model_carry), Outputs()

    def tally(self, carry: LaneCarry, done: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        """Each ended lane game (`done` (R,)) once, by (lane, outcome) with the outcome from
        the first player's side (`result` (R,) int32 is the row's own: +1 won, 0 tied, -1
        lost): flat (lanes * 3,) int64, as `bnb.runtime.opponents.OUTCOMES`."""
        outcome = (1 - result).to(torch.int64)  # the row's win 0, tie 1, loss 2
        flat = self.column * len(OUTCOMES) + outcome
        counts = torch.zeros(self.columns * len(OUTCOMES), dtype=torch.int64, device=done.device)
        return counts.index_add(0, flat, (done & (self.role == 0)).to(torch.int64))

    # ---- the host's side, between segments

    @torch.no_grad()
    def load(self, player: str, state: dict[str, torch.Tensor]) -> None:
        """Copy a network's state (its model's state dict) into `player`'s weights, in place
        (a player the lanes do not play is ignored)."""
        if player not in self.networks:
            return
        index = self.networks.index(player)
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
