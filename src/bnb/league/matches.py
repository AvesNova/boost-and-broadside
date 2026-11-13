"""The offline match engine (league spec 6): players named on the command line, and batches
of games between pairs of them on fixed slots.

A player is a policy file (or a training checkpoint, whose policy part plays), the newest
checkpoint of a run, a ladder entry ("scripted", "random" or "rung_T<t>"), or a teacher:
the scripted agent at its own temperature T with a uniform floor on its actions,
`scripted@T=<temperature>,floor=<floor>` (either key optional: T defaults to 1, floor to 0;
BC's teacher is `scripted`, T = 1 and floor 0). A ladder entry and a teacher are scripted
players; the others are networks. Networks of one build form a group (`Arena.groups`): the
groups' builds may differ (a field may mix architectures); the networks of one group have
their weights stacked.

A Yemong network may play ablated, `<policy>:<ablation>[+<ablation>...]` (for example
`PATH:rotary`, `PATH:forecast`, `PATH:rotary+forecast`; `bnb.models.yemong.model.ABLATIONS`):
the same weights, with the ablations applied on that player's rows only
(`bnb.models.yemong.model.Ablations`). It is the same build, so it shares its policy's group
and weights. An ablation is inference-time: it measures how much a policy trained with the
feature relies on it, not what one trained without it would reach.

A network may play at a sampling temperature, `<policy>@T=<temperature>` (after any
ablations, `PATH:rotary@T=0.7`; `temperature_of`): the same weights, its joint action drawn
from logits / T on that player's rows only (`bnb.models.yemong.policy.tempered`; each row's
temperature held in place, 1 for every other row, which samples exactly as without one).
Like an ablated copy it shares its policy's group and weights. Each row's first game also
sums its network's entropy (`STATISTICS`), so a field of temperatures measures the
entropy each one plays at.

The engine (`Arena`) is one rollout of one game preset, built once: `blocks` matchup blocks
of `block_environments` environments each. Each block is a pool of its own whose two seats
are both the one controller `MatchSeat`; block b's first seat is slot 2b and its second
slot 2b + 1, so each slot is a fixed, contiguous run of rows. A slot holds a network or a
scripted player. Each network group has a template of its own and a fixed number of
**lanes** (`Arena.lanes`; `default_lanes` splits the slots over the groups as their
players): a lane holds one slot's weights (stacked by lane) and the slot it plays this
batch, both written in place; a group's lanes run as one forward over their stacked
weights, each on its slot's rows (`bnb.runtime.stacked`, as the rating lanes'
`bnb.runtime.lanes.LaneSeat` does), writing back each slot's actions and carry. A batch may hold
at most a group's lanes of its slots (the tournament's allocation keeps to them); an idle
lane runs on a slot of its own and writes nothing. With lanes that sum to the slots, as
by default, a batch costs one network pass over every row however many builds the field
mixes. A scripted slot's rows take the scripted pass, one over every row, at each row's
temperature, share and floor.
The shapes never change, so the rollout's graph is captured once; a batch only copies the
chosen players into the lanes and the slots, in place.

A batch: every environment starts a fresh game (`Rollout.restart`), environment i of a
block seating the block's first player on team i mod 2, so a block's games split evenly
between the seats. Each environment's first game is played to its end and tallied once,
from the block's first player's side (`MatchSeat.tally`).

An engine built with benching plays unequal teams (the crossover, `bnb.league.crossover`):
each block's batch may give its two players their active ship counts (at most the game's
ships per team; the rest of a team's slots benched, `bnb.game.api`), set per environment
with the seats, so one graph plays every handicap at once.
"""

import dataclasses
import math
import pathlib

import torch
from torch.utils import _pytree

from bnb.contracts import ASPECTS, Action, GameConfig, Obs, Outputs, Pool
from bnb.contracts.obs import SHIP
from bnb.contracts.tree import contract
from bnb.controllers.scripted import Scripted, uniform_actions
from bnb.controllers.scripted_defaults import MODE_DEFAULTS
from bnb.controllers.scripted_memory import ScriptedCarry
from bnb.controllers.scripted_parameters import ScriptedParameters, table, unpack
from bnb.game import api
from bnb.game.handling.characteristics import Characteristics, characteristics
from bnb.learn.bc import TeacherConfig
from bnb.models.yemong.model import ABLATIONS, Ablations, Yemong
from bnb.runs import checkpoint
from bnb.runs.records import Run
from bnb.runtime import stacked
from bnb.runtime.opponents import OUTCOMES, Rung, ladder_rung
from bnb.runtime.rollout import Rollout
from bnb.runtime.train import model_controller

SEAT = "matches"  # the match seat's controller name


@dataclasses.dataclass(frozen=True)
class Player:
    """One player: its name, and either a policy directory, a ladder entry's rung
    (`bnb.runtime.opponents.Rung`; +inf random), or a teacher (the scripted agent at its
    temperature with a floor); ablations: a network's inference-time ablations (names of
    `ABLATIONS`, sorted); temperature: a network's sampling temperature (1: its policy as it
    is); ship_type: every ship of its team flies this type (an index into
    the game's card; None: the game's lineup), so a scripted player with a ship type is a
    ship type as the scripted agent flies it (`ship_type_players`); a tuple gives each ship
    of the team its own type (a mixed team, one entry per ship); parameters: a scripted
    player's own parameter set (its doctrine; only in an arena built with a `doctrine`), else
    the arena's."""

    name: str
    policy: pathlib.Path | None = None
    rung: Rung | None = None
    teacher: TeacherConfig | None = None
    ablations: tuple[str, ...] = ()
    temperature: float = 1.0
    ship_type: int | tuple[int, ...] | None = None
    parameters: ScriptedParameters | None = None


def ship_type_players(ship_set: str, rung: Rung = Rung(1.0)) -> list[Player]:
    """One player per type of a ship set, named by the type's name: the scripted agent at
    `rung` (the scripted agent itself by default) flying that type with every ship."""
    return [
        Player(name, rung=rung, ship_type=index)
        for index, name in enumerate(api.type_names(ship_set))
    ]


def teacher_of(name: str) -> TeacherConfig | None:
    """The teacher a `scripted@T=<t>,floor=<f>` name means (keys in any order, each
    optional, none repeated; T defaults to 1 and floor to 0), or None for another name.
    Raises ValueError for a malformed or out-of-range one."""
    if not name.startswith("scripted@"):
        return None
    values = {"T": 1.0, "floor": 0.0}
    seen = set()
    for item in name[len("scripted@") :].split(","):
        key, equals, text = item.partition("=")
        if not equals or key not in values or key in seen:
            raise ValueError(f"{name!r}: expected scripted@T=<temperature>,floor=<floor>")
        seen.add(key)
        try:
            values[key] = float(text)
        except ValueError:
            raise ValueError(f"{name!r}: {text!r} is not a number") from None
    return TeacherConfig(temperature=values["T"], floor=values["floor"])


def ablations_of(name: str) -> tuple[str, tuple[str, ...]]:
    """A name's policy part and its ablations: `<policy>:<ablation>[+<ablation>...]` gives
    (policy, the ablations sorted); a name without a known ablation suffix, (name, ())."""
    policy, colon, suffix = name.rpartition(":")
    parts = suffix.split("+")
    if not colon or not policy or not all(part in ABLATIONS for part in parts):
        return name, ()
    if len(set(parts)) < len(parts):
        raise ValueError(f"{name!r}: an ablation is named twice")
    return policy, tuple(sorted(parts))


def temperature_of(name: str) -> tuple[str, float | None]:
    """A name's player part and its sampling temperature: `<player>@T=<temperature>` gives
    (player, temperature); a name without the suffix, or a teacher's (`scripted@...`,
    `teacher_of`), (name, None). Raises ValueError for a temperature that is not a positive
    finite number."""
    base, at, suffix = name.rpartition("@")
    if not at or not base or not suffix.startswith("T=") or name.startswith("scripted@"):
        return name, None
    try:
        temperature = float(suffix[len("T=") :])
    except ValueError:
        raise ValueError(
            f"{name!r}: {suffix[len('T=') :]!r} is not a number (a temperature comes after "
            "any ablations: <policy>[:<ablation>...]@T=<temperature>)"
        ) from None
    if not (math.isfinite(temperature) and temperature > 0.0):
        raise ValueError(f"{name!r}: a temperature must be positive and finite")
    return base, temperature


def player(name: str, runs_directory: pathlib.Path | None = None) -> Player:
    """The player a command-line name means: a ladder entry, a policy directory (one with a
    checkpoint.json; a path, or one inside `runs_directory` such as
    `RUN/policies/step_<steps>/uniform_average`), or a run's newest checkpoint; either of
    the last two with ablations (`ablations_of`) and at a temperature (`temperature_of`)."""
    base, temperature = temperature_of(name)
    if temperature is not None:
        network = player(base, runs_directory)
        if network.policy is None:
            raise ValueError(
                f"{name!r}: only a network plays at a temperature (the scripted agent's is "
                "scripted@T=<temperature>)"
            )
        if network.temperature != 1.0:
            raise ValueError(f"{name!r}: a temperature is named twice")
        return dataclasses.replace(network, name=name, temperature=temperature)
    base, ablations = ablations_of(name)
    if ablations:
        network = player(base, runs_directory)
        if network.policy is None:
            raise ValueError(f"{name!r}: only a network plays ablated")
        return dataclasses.replace(network, name=name, ablations=ablations)
    rung = ladder_rung(name)
    if rung is not None:
        return Player(name, rung=rung)
    teacher = teacher_of(name)
    if teacher is not None:
        return Player(name, teacher=teacher)
    path = pathlib.Path(name)
    if (path / checkpoint.RECORD).exists():
        return Player(name, policy=path)
    inside = None if runs_directory is None else pathlib.Path(runs_directory) / name
    if inside is not None and (inside / checkpoint.RECORD).exists():
        return Player(name, policy=inside)
    if inside is not None and inside.is_dir():
        return Player(name, policy=Run.open(pathlib.Path(runs_directory), name).checkpoint_at())
    raise ValueError(
        f"{name!r} is not a policy file, a run or a ladder entry (scripted, random, "
        "rung_T<t>, rung_T<t>_p<share>)"
    )


@dataclasses.dataclass(frozen=True)
class ScriptedPlay:
    """How a scripted player plays: the scripted agent at `temperature` (+inf random), each
    ship playing it with probability `share` per game and uniformly otherwise (a mixture
    rung), then a uniform joint action with probability `floor` per decision (a teacher's)."""

    temperature: float
    share: float = 1.0
    floor: float = 0.0
    parameters: ScriptedParameters | None = None


def scripted_play(player: Player) -> ScriptedPlay | None:
    """A scripted player's play (a rung's or a teacher's), or None for a network."""
    if player.rung is not None:
        return ScriptedPlay(
            player.rung.temperature, player.rung.share, parameters=player.parameters
        )
    if player.teacher is not None:
        return ScriptedPlay(
            player.teacher.temperature, floor=player.teacher.floor, parameters=player.parameters
        )
    return None


@contract
class MatchCarry:
    """Per row (R leading): model, the networks' carry by group (the group's controller's; a
    memoryless network has none); scripted, the scripted agent's; finished (R,) bool, the
    row's first game has ended; started (1,) bool, a decision has been made (the first
    observation's reset is the batch's start, not a game's end); statistics (R,
    len(STATISTICS)) float, the row's team's first game so far (`MatchSeat.count`)."""

    model: object
    scripted: ScriptedCarry
    finished: torch.Tensor
    started: torch.Tensor
    statistics: torch.Tensor


# Each row's team's first game, summed (`MatchSeat.count_events`): kills (enemy ships killed by its
# hits), deaths (its ships', any cause), shots fired, hits (shots of its that damaged an
# enemy), its living ship-ticks and their proper speeds (m/s, summed: over the ship-ticks
# it is the mean speed), as each tick's observation shows them, and a network's entropy (nats,
# its sampled joint action distribution's, at its temperature) summed over the same
# ship-ticks (over them it is the mean entropy; 0 for a scripted player). Then, by the hit's
# aspect class (`bnb.contracts.events.ASPECTS`), the hit damage its ships dealt to enemy
# ships and its kills (the killing hit's class). New columns go last: a record keeps the
# columns of its time, a prefix of these.
STATISTICS = (
    "kills",
    "deaths",
    "shots",
    "hits",
    "ship_ticks",
    "speed",
    "entropy",
    *(f"damage_{aspect}" for aspect in ASPECTS),
    *(f"kills_{aspect}" for aspect in ASPECTS),
)
ASPECT_STATISTICS = STATISTICS.index(f"damage_{ASPECTS[0]}")


class _Decision(torch.nn.Module):
    """A network controller's decision as a module call, for `functional_call`."""

    def __init__(self, controller) -> None:
        super().__init__()
        self.model = controller.model
        self.controller = controller

    def forward(
        self,
        carry,
        obs: Obs,
        reset: torch.Tensor,
        temperature: torch.Tensor,
        ablations: Ablations | None = None,
    ):
        if ablations is None:
            action, carry, outputs = self.controller.step(
                carry, obs, reset, temperature=temperature
            )
        else:
            action, carry, outputs = self.controller.step(
                carry, obs, reset, ablations, temperature=temperature
            )
        return action.packed(), carry, outputs.entropy


class MatchSeat:
    """Every row's player, in one controller (see the module docstring). controllers: one
    network controller per group, over a model of its own (the forward swaps each lane's
    weights in), none when every player is scripted; lanes: each group's lane count, the
    most of its slots a batch may hold (one per group by default); blocks and
    block_environments: the layout (rows are block by block, each block's first seat's rows
    then its second's, as the lineup lists them); characteristics and ships_per_team: the
    scripted agent's; doctrine: the scripted agent's parameter set for every row whose slot's
    player brings none (None: the mode's set, and no player may bring one); ablations: a
    network slot may play ablated (each row's ablations held
    in place and passed to every Yemong group's network as `Ablations`; else the networks
    are called without them). Each row's network sampling temperature is held in place
    (`network_temperature`, 1 but on a tempered player's rows). Every slot starts as the
    scripted agent, every lane idle."""

    def __init__(
        self,
        controllers: list,
        blocks: int,
        block_environments: int,
        characteristics: Characteristics,
        ships_per_team: int,
        device: torch.device,
        ablations: bool = False,
        lanes: tuple[int, ...] | None = None,
        doctrine: ScriptedParameters | None = None,
    ) -> None:
        self.slots, self.slot_rows = 2 * blocks, block_environments
        self.rows, self.ships_per_team = self.slots * block_environments, ships_per_team
        row = torch.arange(self.rows, device=device)
        slot = row // block_environments
        self.first_seat = slot % 2 == 0  # the rows that tally their block's games
        self.block_of_row = slot // 2
        self.columns, self.mode_count = blocks, 1
        # the scripted pass over every row, each row at its slot's play (written in place)
        self.temperature = torch.ones(self.rows, device=device)
        self.share = torch.ones(self.rows, device=device)
        self.floor = torch.zeros(self.rows, device=device)
        self.scripted = torch.ones(self.rows, dtype=torch.bool, device=device)
        # each network row's sampling temperature (written in place; 1 samples as without)
        self.network_temperature = torch.ones(self.rows, device=device)
        # with a doctrine, each row plays its slot's parameter set (a table written in place:
        # the slot's player's set, else the doctrine); else every row the mode's set
        self.doctrine, self.doctrine_table = doctrine, None
        parameters = MODE_DEFAULTS
        if doctrine is not None:
            self.doctrine_table = table([doctrine] * self.rows, device)
            parameters = unpack(self.doctrine_table)
        self.agent = Scripted(
            characteristics,
            ships_per_team,
            parameters,
            temperature=self.temperature,
            share=self.share,
        )
        self.lanes = (1,) * len(controllers) if lanes is None else tuple(lanes)
        if len(self.lanes) != len(controllers) or not all(
            1 <= count <= self.slots for count in self.lanes
        ):
            raise ValueError(
                f"lanes {self.lanes}: one count per network group, from 1 to the {self.slots} slots"
            )
        self.decisions = [_Decision(controller) for controller in controllers]
        # per group: each lane's slot (a group's lanes on different slots, an idle lane's
        # any) and whether it plays this batch (both written in place), and the lanes'
        # stacked parameters
        self.lane_slot = [torch.arange(count, device=device) for count in self.lanes]
        self.lane_active = [
            torch.zeros(count, dtype=torch.bool, device=device) for count in self.lanes
        ]
        self.lane_of: dict[int, tuple[int, int]] = {}  # this batch's network slots' lanes
        self.parameters: list[dict[str, torch.Tensor]] = []
        reads = Scripted.reads
        for decision, count in zip(self.decisions, self.lanes, strict=True):
            decision.requires_grad_(False)
            reads = decision.controller.reads + reads
            self.parameters.append(
                {
                    name: value.detach()[None].repeat((count,) + (1,) * value.dim()).clone()
                    for name, value in decision.state_dict().items()
                }
            )
        self.reads = tuple(dict.fromkeys(reads))
        self._offsets = torch.arange(block_environments, device=device)
        # each row's ablations (written in place), or None: the networks are called without;
        # only a Yemong group reads them
        self.ablations = None
        self.ablatable = [isinstance(d.controller.model, Yemong) for d in self.decisions]
        if ablations:
            if not any(self.ablatable):
                raise ValueError("only a Yemong network plays ablated")
            self.ablations = Ablations(
                **{
                    name: torch.zeros(self.rows, dtype=torch.bool, device=device)
                    for name in ABLATIONS
                }
            )

    def _slot_rows(self, slot: int) -> slice:
        return slice(slot * self.slot_rows, (slot + 1) * self.slot_rows)

    # ---- the controller

    def init_carry(self, views: int, device: torch.device) -> MatchCarry:
        """Every row's network carry (a row only its own group's lane reads and writes) and
        scripted carry, nothing finished."""
        model = {}
        for group, decision in enumerate(self.decisions):
            one = decision.controller.init_carry(views, device)
            if one is not None:  # a memoryless network has none
                model[str(group)] = one
        return MatchCarry(
            model=model,
            scripted=self.agent.init_carry(views, device),
            finished=torch.zeros(views, dtype=torch.bool, device=device),
            started=torch.zeros(1, dtype=torch.bool, device=device),
            statistics=torch.zeros((views, len(STATISTICS)), device=device),
        )

    def _network(
        self,
        group: int,
        network: torch.Tensor,
        entropy: torch.Tensor,
        carry,
        obs: Obs,
        reset: torch.Tensor,
    ):
        """Group `group`'s network on its lanes, as one forward over the lanes' stacked
        weights (`bnb.runtime.stacked`), each lane on its slot's rows at their temperatures:
        `network`, the packed actions (R, S, 3), and `entropy`, the sampled distributions'
        entropies (R, S), with the active lanes' rows written, and the group's network carry
        with theirs (an idle lane's rows keep their values; a group's lanes are always on
        different slots)."""
        arguments = (carry, obs, reset, self.network_temperature)
        if self.ablations is not None and self.ablatable[group]:
            arguments += (self.ablations,)
        lanes, slots = self.lanes[group], self.lane_slot[group]
        rows = (slots[:, None] * self.slot_rows + self._offsets).reshape(-1)
        active = self.lane_active[group].repeat_interleave(self.slot_rows)
        inputs = _pytree.tree_map_only(
            torch.Tensor,
            lambda x: x.index_select(0, rows).unflatten(0, (lanes, self.slot_rows)),
            arguments,
        )
        action, lane_carry, lane_entropy = stacked.call(
            self.decisions[group], self.parameters[group], inputs
        )

        def written(full: torch.Tensor | None, part: torch.Tensor | None) -> torch.Tensor | None:
            if full is None:  # a memoryless network's carry
                return None
            kept = full.index_select(0, rows)
            mask = active.reshape((-1,) + (1,) * (kept.dim() - 1))
            return full.index_copy(0, rows, torch.where(mask, part.flatten(0, 1), kept))

        network = written(network, action)
        entropy = written(entropy, lane_entropy.to(entropy.dtype))
        updated = None if lane_carry is None else _pytree.tree_map(written, carry, lane_carry)
        return network, entropy, updated

    def step(self, carry: MatchCarry, obs: Obs, reset: torch.Tensor):
        """One decision for every row; see `Controller.step`. The rollout probes its
        learner's outputs on one row when it is built; the seat gives none, and answers a
        view count other than its own rows' with idle actions."""
        views = reset.shape[0]
        if views != self.rows:
            idle = torch.zeros((views, self.ships_per_team, 3), dtype=torch.int32)
            return Action.unpack(idle.to(reset.device)), carry, Outputs()
        scripted, scripted_carry, _ = self.agent.step(carry.scripted, obs, reset)
        action = scripted.packed()
        uniform = torch.rand(action.shape[:-1], device=action.device) < self.floor[:, None]
        action = torch.where(
            uniform[..., None], uniform_actions(action.shape[:-1], action.device), action
        )
        model_carry = dict(carry.model)
        entropy = torch.zeros(action.shape[:-1], device=action.device)
        if self.decisions:
            network = torch.zeros_like(action)
            for group in range(len(self.decisions)):
                network, entropy, updated = self._network(
                    group, network, entropy, carry.model.get(str(group)), obs, reset
                )
                if updated is not None:
                    model_carry[str(group)] = updated
            action = torch.where(self.scripted[:, None, None], action, network)
            entropy = torch.where(self.scripted[:, None], 0.0, entropy)
        finished = carry.finished | (reset & carry.started)
        own = obs.ships[:, : self.ships_per_team]
        living = (own[..., SHIP["alive"]] > 0.5).float()
        playing = (~finished).float()[:, None]
        observed = torch.stack(
            [
                living.sum(-1),
                (living * own[..., SHIP["proper_speed"]]).sum(-1),
                (living * entropy).sum(-1),
            ],
            -1,
        )
        statistics = carry.statistics.clone()
        statistics[:, STATISTICS.index("ship_ticks") : ASPECT_STATISTICS] += playing * observed
        carry = MatchCarry(
            model=model_carry,
            scripted=scripted_carry,
            finished=finished,
            started=torch.ones_like(carry.started),
            statistics=statistics,
        )
        return Action.unpack(action), carry, Outputs()

    def tally(self, carry: MatchCarry, done: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        """The first games that ended this tick, by (block, outcome) from the block's first
        player's side (`result` (R,): the row's own, +1 a win, 0 a tie, -1 a loss): flat
        (blocks * 3,) int64, as `bnb.runtime.opponents.OUTCOMES`."""
        outcome = (1 - result).to(torch.int64)  # win 0, tie 1, loss 2
        counted = (done & ~carry.finished & self.first_seat).to(torch.int64)
        counts = torch.zeros(self.columns * len(OUTCOMES), dtype=torch.int64, device=done.device)
        return counts.index_add(0, self.block_of_row * len(OUTCOMES) + outcome, counted)

    def count_events(
        self,
        carry: MatchCarry,
        events,
        environments: torch.Tensor,
        team: torch.Tensor,
    ) -> None:
        """Add a tick's events to each row's first-game statistics (`STATISTICS`), in place:
        `events` the game's (sim order, every environment), `environments` and `team` (R,)
        each row's environment and team. The tick a game ends is its last counted (the
        row's `finished` turns at the next decision)."""
        S = self.ships_per_team
        ship_team = torch.arange(2 * S, device=team.device) // S  # (N,)
        own = ship_team[None, :] == team[:, None]  # (R, N)
        died = events.died[environments]
        killer = events.killer[environments]
        killer_team = torch.where(killer >= 0, killer // S, -1)
        kills = (died & ~own & (killer_team == team[:, None])).sum(-1)
        deaths = (died & own).sum(-1)
        shots = (events.fired[environments] & own).sum(-1)
        damaged = events.damage[environments] > 0.0  # (R, shooter, target)
        hits = (damaged & own[:, :, None] & ~own[:, None, :]).sum((-1, -2))
        tick = torch.stack([kills, deaths, shots, hits], -1).float()
        playing = (~carry.finished).float()[:, None]
        carry.statistics[:, : STATISTICS.index("ship_ticks")] += playing * tick
        # by aspect: the enemy-only damage its enemies took, and its kills by class
        dealt = (events.enemy_damage_by_aspect[environments] * (~own).float()[..., None]).sum(1)
        kill_aspect = events.kill_aspect[environments]
        own_kill = died & ~own & (killer_team == team[:, None])
        kills_by_aspect = torch.stack(
            [(own_kill & (kill_aspect == index)).sum(-1) for index in range(len(ASPECTS))], -1
        ).float()
        carry.statistics[:, ASPECT_STATISTICS:] += playing * torch.cat([dealt, kills_by_aspect], -1)

    # ---- the host's side, between batches

    @torch.no_grad()
    def clear(self) -> None:
        """A new batch's slots: every lane idle (every slot then loads its player)."""
        for active in self.lane_active:
            active.zero_()
        self.lane_of = {}

    @torch.no_grad()
    def load_network(
        self,
        slot: int,
        group: int,
        state: dict[str, torch.Tensor],
        ablations: tuple[str, ...] = (),
        temperature: float = 1.0,
    ) -> None:
        """Slot `slot` plays a network of group `group`: on the group's first idle lane, its
        state (its model's state dict) copied into the lane in place, with `ablations` (names
        of `ABLATIONS`) on its rows, sampling at `temperature`. Raises ValueError when every
        lane of the group plays (`clear` idles them)."""
        if not 0 <= group < len(self.decisions):
            raise ValueError(f"the seat was built with {len(self.decisions)} network groups")
        if ablations and (self.ablations is None or not self.ablatable[group]):
            raise ValueError("only a Yemong network of a seat built with ablations plays ablated")
        taken = [lane for g, lane in self.lane_of.values() if g == group]
        if len(taken) >= self.lanes[group]:
            raise ValueError(
                f"network group {group} has {self.lanes[group]} lanes, and a batch holds more "
                "of its slots"
            )
        lane, slots = len(taken), self.lane_slot[group]
        for name, value in state.items():
            self.parameters[group][f"model.{name}"][lane].copy_(value)
        # the idle lane on `slot`, if any, takes this lane's slot: the lanes stay apart
        holder = (slots == slot).nonzero()
        if len(holder):
            slots[holder[0, 0]] = slots[lane].clone()
        slots[lane] = slot
        self.lane_active[group][lane] = True
        self.lane_of[slot] = (group, lane)
        rows = self._slot_rows(slot)
        self.scripted[rows] = False
        self.network_temperature[rows] = temperature
        if self.ablations is not None:
            for name in ABLATIONS:
                getattr(self.ablations, name)[rows] = name in ablations

    @torch.no_grad()
    def load_scripted(self, slot: int, play: ScriptedPlay) -> None:
        """Slot `slot` plays the scripted agent as `play` says, written in place."""
        rows = self._slot_rows(slot)
        self.temperature[rows] = play.temperature
        self.share[rows] = play.share
        self.floor[rows] = play.floor
        self.scripted[rows] = True
        self.network_temperature[rows] = 1.0
        if self.doctrine_table is not None:
            own = self.doctrine if play.parameters is None else play.parameters
            self.doctrine_table[rows] = table([own], self.doctrine_table.device)
        elif play.parameters is not None:
            raise ValueError("a player with its own parameters needs an arena with a doctrine")
        if self.ablations is not None:
            for name in ABLATIONS:
                getattr(self.ablations, name)[rows] = False

    def finished(self, carry: MatchCarry) -> bool:
        """Whether every row's first game has ended (a host read)."""
        return bool(carry.finished.all())


@dataclasses.dataclass(frozen=True)
class ArenaConfig:
    """blocks: matchup blocks per batch; block_environments: games per block (even, so they
    split evenly between the seats); length: ticks per rollout segment (the host checks
    whether every game has ended once per segment); compiled: compile the rollout's stages
    (else eager); graph_ticks: the most ticks in one captured graph on a GPU (as training's:
    a graph holds every slot's kernels for each of its ticks)."""

    blocks: int = 8
    block_environments: int = 128
    length: int = 32
    compiled: bool = True
    graph_ticks: int = 8

    def __post_init__(self) -> None:
        if self.blocks < 1:
            raise ValueError(f"{self.blocks} blocks: need at least one")
        if self.block_environments < 2 or self.block_environments % 2:
            raise ValueError(
                f"{self.block_environments} environments per block must be even (split "
                "evenly between the seats)"
            )

    @property
    def games(self) -> int:
        """Games per batch: every environment's first."""
        return self.blocks * self.block_environments


def network_groups(players: list[Player]) -> list[int | None]:
    """Each player's network group, by its policy's build record (the model and its build;
    numbered in the order the builds first appear), or None for a scripted player."""
    builds: list[tuple[str, object]] = []
    groups: list[int | None] = []
    for one in players:
        if scripted_play(one) is not None:
            groups.append(None)
            continue
        record = checkpoint.read_policy(one.policy)
        build = (record.name, record.build)
        if build not in builds:
            builds.append(build)
        groups.append(builds.index(build))
    return groups


def default_lanes(groups: list[int | None], slots: int) -> tuple[int, ...]:
    """Each network group's lanes (`MatchSeat`) out of `slots` in all: one each, then each
    further lane to the group with the most players per lane (the field's network players
    by group, `network_groups`), so a batch costs one network pass over every row and its
    slots are spread over the groups as their players are. A group gets no more lanes than
    a batch can fill: a block's two slots hold different players, so a group with one
    player fills at most one slot per block (`slots // 2`), and every lane runs its forward
    whether it plays or not. Raises ValueError for more groups than slots."""
    count = 1 + max((group for group in groups if group is not None), default=-1)
    players = [groups.count(group) for group in range(count)]
    if len(players) > slots:
        raise ValueError(f"{len(players)} network builds need a lane each, more than {slots} slots")
    most = [slots // 2 * min(held, 2) for held in players]
    lanes = [1] * len(players)
    for _ in range(slots - count if count else 0):
        open_groups = [group for group in range(count) if lanes[group] < most[group]]
        if not open_groups:
            break
        richest = max(open_groups, key=lambda group: players[group] / lanes[group])
        lanes[richest] += 1
    return tuple(lanes)


def card_of(players: list[Player], game: GameConfig, device: torch.device) -> api.ShipTypes:
    """The ship card a match of these players plays: the one their policies were trained
    with (each policy record's `ship_card`), else (only scripted players, or records
    without a card) the default card of the game's ship set. Raises ValueError when two
    policies were trained with different cards: one match plays one card."""
    cards = {}
    for player in players:
        if player.policy is None:
            continue
        card = checkpoint.read_record(player.policy).get("ship_card")
        if card is not None:
            cards[player.policy] = card
    found = list(cards.values())
    if not found:
        return api.ship_types(device, ship_set=game.ship_set)
    if any(card != found[0] for card in found[1:]):
        raise ValueError(
            "these policies were trained with different ship cards: "
            + ", ".join(str(policy) for policy in cards)
        )
    return api.ship_types(device, ship_set=game.ship_set, **found[0])


def team_lineup(player: Player, rules) -> tuple[int, ...]:
    """A player's team lineup, one type per ship: its `ship_type` on every ship (or each
    ship's own, for a tuple), else the game's."""
    if player.ship_type is None:
        return rules.team_types
    if isinstance(player.ship_type, tuple):
        if len(player.ship_type) != rules.ships_per_team:
            raise ValueError(
                f"{player.name}: a lineup of {len(player.ship_type)} ships for a team of "
                f"{rules.ships_per_team}"
            )
        return player.ship_type
    return (player.ship_type,) * rules.ships_per_team


class Arena:
    """The match engine over `players` in `game` (a game preset's config), built once (see
    the module docstring). Each network player's weights are read once and kept on the
    host (an ablated or tempered copy shares its policy's, and its group); a batch copies
    them into its slots' lanes, with the player's ablations and temperature. `groups`: a
    template model per build, in the order the builds first appear among the players
    (`group_of` maps a network player's index to its group, as `network_groups`); lanes:
    each group's lanes, the most of its slots a batch may hold (`default_lanes` by default);
    card: the card to play (the players' by default, `card_of`); doctrine: the scripted
    agent's set for scripted players without their own (`Player.parameters`; None: the
    mode's set, and no player may bring one); benching: the batches may give the players
    active ship counts (`play`). `set_card` writes a new card in place."""

    def __init__(
        self,
        players: list[Player],
        game: GameConfig,
        device: torch.device,
        config: ArenaConfig = ArenaConfig(),
        lanes: tuple[int, ...] | None = None,
        card: api.ShipTypes | None = None,
        doctrine: ScriptedParameters | None = None,
        benching: bool = False,
    ) -> None:
        self.players, self.config, self.device = players, config, device
        self.ships_per_team = game.ships_per_team
        self.states: dict[int, dict[str, torch.Tensor]] = {}
        self.groups: list[torch.nn.Module] = []
        groups = network_groups(players)
        self.group_of = {index: group for index, group in enumerate(groups) if group is not None}
        self.lanes = default_lanes(groups, 2 * config.blocks) if lanes is None else tuple(lanes)
        loaded: dict[pathlib.Path, int] = {}  # a policy's first player
        for index, group in self.group_of.items():
            one = players[index]
            if one.policy in loaded:  # an ablated or tempered copy (or a repeat): the same weights
                self.states[index] = self.states[loaded[one.policy]]
            else:
                loaded[one.policy] = index
                model = checkpoint.load_policy(one.policy, device).requires_grad_(False)
                if group == len(self.groups):
                    self.groups.append(model)
                self.states[index] = {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                }
            if one.ablations and not isinstance(self.groups[group], Yemong):
                raise ValueError(f"{one.name}: only a Yemong network plays ablated")
        self.types = card_of(players, game, device) if card is None else card
        rules = api.rules_of(game)
        self.rules = rules
        agent = characteristics(self.types, rules)
        # each player's team lineup: its ship type on every ship, else the game's
        self.team_types = [team_lineup(one, rules) for one in players]
        ship_type = None
        if any(one.ship_type is not None for one in players):
            ship_type = torch.tensor(rules.slot_types, dtype=torch.int32).repeat(config.games, 1)
        controllers = [model_controller(template, game) for template in self.groups]
        self.seat = MatchSeat(
            controllers,
            config.blocks,
            config.block_environments,
            agent,
            game.ships_per_team,
            device,
            ablations=any(one.ablations for one in players),
            lanes=self.lanes,
            doctrine=doctrine,
        )
        pools = tuple(
            Pool(f"block_{index}", 1.0 / config.blocks, (SEAT, SEAT))
            for index in range(config.blocks)
        )
        self.rollout = Rollout(
            game,
            self.types,
            pools,
            {SEAT: self.seat},
            SEAT,
            config.games,
            config.length,
            device,
            compiled=config.compiled,
            graph_ticks=config.graph_ticks,
            staggered=False,
            ship_type=ship_type,
            benching=benching,
        )
        # environment i seats its block's first player on team i mod 2
        self.seats = torch.arange(config.games, device=device) % 2
        # every first game ends by its clock; the margin covers the segment it ends in
        self.tick_limit = int(self.rollout.episode_ticks.max()) + 2 * config.length

    def set_card(self, types: api.ShipTypes) -> None:
        """Play the following batches with this card ((num_types,) fields, as the arena's):
        written in place into the rollout's card and the scripted agent's characteristics,
        so nothing recompiles (`docs/explanation/game/ship-design.md`). Raises ValueError
        for another number of types, or a fire rate that needs more bullet slots than the
        arena was built with (build it with the fastest fire rate the cards will use)."""
        held = self.rollout.carry["types"]
        if types.max_energy.shape != held.max_energy.shape:
            raise ValueError(
                f"a card of {tuple(types.max_energy.shape)} types for an arena built with "
                f"{tuple(held.max_energy.shape)}"
            )
        rules = self.rollout.rules
        if api.bullet_slots_for(types.fire_rate, types.bullet_life, rules.tick_length) > (
            rules.bullet_slots
        ):
            raise ValueError(
                f"fire rates {types.fire_rate.tolist()} need more than the arena's "
                f"{rules.bullet_slots} bullet slots"
            )
        for field in dataclasses.fields(held):
            getattr(held, field.name).copy_(getattr(types, field.name))
        measured = characteristics(types, self.rules)
        agent = self.seat.agent.characteristics
        for field in dataclasses.fields(agent):
            getattr(agent, field.name).copy_(getattr(measured, field.name))
        self.types = types

    def _load(self, slot: int, index: int) -> None:
        play = scripted_play(self.players[index])
        if play is None:
            one = self.players[index]
            self.seat.load_network(
                slot, self.group_of[index], self.states[index], one.ablations, one.temperature
            )
        else:
            self.seat.load_scripted(slot, play)

    def _ship_types(self, matchups: list[tuple[int, int]]) -> torch.Tensor:
        """(games, N) int32 each ship's type for a batch: each environment's teams fly their
        players' lineups (`Player.ship_type`), the block's first player on team i mod 2."""
        rows = []
        for first, second in matchups:
            for environment in range(self.config.block_environments):
                teams = (first, second) if environment % 2 == 0 else (second, first)
                rows.append(self.team_types[teams[0]] + self.team_types[teams[1]])
        return torch.tensor(rows, dtype=torch.int32)

    def statistics(self) -> torch.Tensor:
        """(blocks, 2, len(STATISTICS)) float64 the last batch's first games' statistics per
        block, its first player's then its second's (`MatchSeat.count_events`), summed over
        the block's games (a host read)."""
        rows = self.rollout.carry["controllers"][SEAT].statistics.double().cpu()
        return rows.reshape(self.config.blocks, 2, self.config.block_environments, -1).sum(2)

    def kill_shares(self) -> torch.Tensor:
        """(blocks, 3) float64 the last batch's first games scored by kill share, as
        `play`'s (wins, ties, losses) are: (the sum of the first player's shares, 0, the
        block's games less that sum). A game's share is its kills over both sides' kills
        (`STATISTICS`' "kills": enemy ships killed by the side's hits), 0.5 in a game with no
        kill. Team deathmatch's rating score (`bnb.league.tournament.kill_share_scored`)."""
        rows = self.rollout.carry["controllers"][SEAT].statistics.double().cpu()
        kills = rows[:, STATISTICS.index("kills")].reshape(
            self.config.blocks, 2, self.config.block_environments
        )
        first, both = kills[:, 0], kills.sum(1)
        share = torch.where(both > 0.0, first / both.clamp_min(1.0), 0.5)
        score = share.sum(-1)
        games = torch.full_like(score, float(self.config.block_environments))
        return torch.stack([score, torch.zeros_like(score), games - score], -1)

    def _active(self, counts: list[tuple[int, int]]) -> torch.Tensor:
        """(B, 2) int32 each environment's active ships by team, from each block's (first
        player's, second player's) counts and the environment's seat."""
        for block, pair in enumerate(counts):
            if len(pair) != 2 or not all(1 <= count <= self.ships_per_team for count in pair):
                raise ValueError(
                    f"block {block}: active ships {tuple(pair)}, each from 1 to the game's "
                    f"{self.ships_per_team} ships per team"
                )
        by_block = torch.tensor(counts, dtype=torch.int32)  # (blocks, 2) first, second
        rows = by_block.repeat_interleave(self.config.block_environments, 0)  # (B, 2)
        first_on_team0 = (self.seats.cpu() == 0)[:, None]
        return torch.where(first_on_team0, rows, rows.flip(-1)).to(self.device)

    def play(
        self, matchups: list[tuple[int, int]], active: list[tuple[int, int]] | None = None
    ) -> torch.Tensor:
        """One batch: block b plays `matchups[b]` (its first player, its second; indices into
        the players), every block's games to their end; active: block b's players' active
        ship counts (first, second), for an engine built with benching (None: every ship).
        (blocks, 3) int64: each block's first player's wins, ties and losses (`statistics`
        then has the games' statistics). Raises RuntimeError if a game runs past the game
        clock (plus a margin) without ending."""
        if len(matchups) != self.config.blocks:
            raise ValueError(f"{len(matchups)} matchups for {self.config.blocks} blocks")
        if active is not None and len(active) != self.config.blocks:
            raise ValueError(f"{len(active)} active counts for {self.config.blocks} blocks")
        for block, (first, second) in enumerate(matchups):
            if first == second:
                raise ValueError(f"block {block}: {self.players[first].name} against itself")
        held = [self.group_of.get(index) for pair in matchups for index in pair]
        for group, count in enumerate(self.lanes):
            if held.count(group) > count:
                raise ValueError(
                    f"the batch holds {held.count(group)} slots of network group {group}, "
                    f"which has {count} lanes"
                )
        self.seat.clear()
        for block, (first, second) in enumerate(matchups):
            self._load(2 * block, first)
            self._load(2 * block + 1, second)
        if self.rollout.ship_type is not None:
            self.rollout.ship_type.copy_(self._ship_types(matchups))
        self.rollout.restart(None if active is None else self._active(active))
        self.rollout.carry["seat"].copy_(self.seats)
        counts = torch.zeros(self.config.blocks * len(OUTCOMES), dtype=torch.int64)
        ticks = 0
        while not self.seat.finished(self.rollout.carry["controllers"][SEAT]):
            if ticks > self.tick_limit:
                raise RuntimeError(
                    f"the batch's games have run {ticks} ticks, past the game clock's "
                    f"{int(self.rollout.episode_ticks.max())}: a game that never ends"
                )
            self.rollout.run()
            ticks += self.config.length
            counts += self.rollout.take_tally(SEAT)
        return counts.reshape(self.config.blocks, len(OUTCOMES))
