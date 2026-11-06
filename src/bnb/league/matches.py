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

The engine (`Arena`) is one rollout of one game preset, built once: `blocks` matchup blocks
of `block_environments` environments each. Each block is a pool of its own whose two seats
are both the one controller `MatchSeat`; block b's first seat is slot 2b and its second
slot 2b + 1, so each slot is a fixed, contiguous run of rows. A slot holds a network or a
scripted player. A network slot belongs to a group of its player's build. Each group has a
template of its own and the slots' stacked parameters, and runs its forward on every slot's rows
(`torch.func.functional_call` on the slot's slice, as the rating lanes'
`bnb.runtime.lanes.LaneSeat` does) with that slot's weights; a slot's action is its own
group's and the other groups' are discarded (a slot's player changes between batches and the
captured graph's shapes never do, so every group runs over every slot). A scripted slot's
rows take the scripted pass, one over every row, at each row's temperature, share and floor.
Each extra group therefore costs about one more network pass over every row.
The shapes never change, so the rollout's graph is captured once; a batch only copies the
chosen players into the slots, in place.

A batch: every environment starts a fresh game (`Rollout.restart`), environment i of a
block seating the block's first player on team i mod 2, so a block's games split evenly
between the seats. Each environment's first game is played to its end and tallied once,
from the block's first player's side (`MatchSeat.tally`).
"""

import dataclasses
import pathlib

import torch
from torch.utils import _pytree

from bnb.contracts import Action, GameConfig, Obs, Outputs, Pool
from bnb.contracts.tree import contract
from bnb.controllers.scripted import Scripted, uniform_actions
from bnb.controllers.scripted_memory import ScriptedCarry
from bnb.game import api
from bnb.game.handling.characteristics import Characteristics, characteristics
from bnb.learn.bc import TeacherConfig
from bnb.models.yemong.model import ABLATIONS, Ablations, Yemong
from bnb.runs import checkpoint
from bnb.runs.records import Run
from bnb.runtime.opponents import OUTCOMES, Rung, ladder_rung
from bnb.runtime.rollout import Rollout
from bnb.runtime.train import model_controller

SEAT = "matches"  # the match seat's controller name


@dataclasses.dataclass(frozen=True)
class Player:
    """One player: its name, and either a policy directory, a ladder entry's rung
    (`bnb.runtime.opponents.Rung`; +inf random), or a teacher (the scripted agent at its
    temperature with a floor); ablations: a network's inference-time ablations (names of
    `ABLATIONS`, sorted)."""

    name: str
    policy: pathlib.Path | None = None
    rung: Rung | None = None
    teacher: TeacherConfig | None = None
    ablations: tuple[str, ...] = ()


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


def player(name: str, runs_directory: pathlib.Path | None = None) -> Player:
    """The player a command-line name means: a ladder entry, a policy directory (one with a
    checkpoint.json), or a run's newest checkpoint; either of the last two with ablations
    (`ablations_of`)."""
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
    if runs_directory is not None and (pathlib.Path(runs_directory) / name).is_dir():
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


def scripted_play(player: Player) -> ScriptedPlay | None:
    """A scripted player's play (a rung's or a teacher's), or None for a network."""
    if player.rung is not None:
        return ScriptedPlay(player.rung.temperature, player.rung.share)
    if player.teacher is not None:
        return ScriptedPlay(player.teacher.temperature, floor=player.teacher.floor)
    return None


@contract
class MatchCarry:
    """Per row (R leading): model, the networks' carry by group (the group's controller's; a
    memoryless network has none); scripted, the scripted agent's; finished (R,) bool, the
    row's first game has ended; started (1,) bool, a decision has been made (the first
    observation's reset is the batch's start, not a game's end)."""

    model: object
    scripted: ScriptedCarry
    finished: torch.Tensor
    started: torch.Tensor


class _Decision(torch.nn.Module):
    """A network controller's decision as a module call, for `functional_call`."""

    def __init__(self, controller) -> None:
        super().__init__()
        self.model = controller.model
        self.controller = controller

    def forward(self, carry, obs: Obs, reset: torch.Tensor, ablations: Ablations | None = None):
        if ablations is None:
            action, carry, _ = self.controller.step(carry, obs, reset)
        else:
            action, carry, _ = self.controller.step(carry, obs, reset, ablations)
        return action.packed(), carry


class MatchSeat:
    """Every row's player, in one controller (see the module docstring). controllers: one
    network controller per group, over a model of its own (the forward swaps each slot's
    weights in), none when every player is scripted; blocks and block_environments: the layout (rows
    are block by block, each block's first seat's rows then its second's, as the lineup
    lists them); characteristics and ships_per_team: the scripted agent's; ablations: a
    network slot may play ablated (each row's ablations held in place and passed to every
    Yemong group's network as `Ablations`; else the networks are called without them).
    Every slot starts as the scripted agent."""

    def __init__(
        self,
        controllers: list,
        blocks: int,
        block_environments: int,
        characteristics: Characteristics,
        ships_per_team: int,
        device: torch.device,
        ablations: bool = False,
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
        self.agent = Scripted(
            characteristics, ships_per_team, temperature=self.temperature, share=self.share
        )
        self.group = torch.zeros(self.rows, dtype=torch.int64, device=device)  # a network row's
        self.decisions = [_Decision(controller) for controller in controllers]
        self.parameters: list[dict[str, torch.Tensor]] = []  # per group, stacked by slot
        reads = Scripted.reads
        for decision in self.decisions:
            decision.requires_grad_(False)
            reads = decision.controller.reads + reads
            self.parameters.append(
                {
                    name: value.detach()[None].repeat((self.slots,) + (1,) * value.dim()).clone()
                    for name, value in decision.state_dict().items()
                }
            )
        self.reads = tuple(dict.fromkeys(reads))
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
        """Every row's network carry (a scripted row's is never read) and scripted carry,
        nothing finished."""
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
        )

    def _network(self, group: int, carry, obs: Obs, reset: torch.Tensor):
        """Group `group`'s network on each slot's own rows, the slots one after another:
        (packed actions (R, S, 3), the group's network carry)."""
        outputs = []
        for slot in range(self.slots):
            rows = self._slot_rows(slot)
            parameters = {name: value[slot] for name, value in self.parameters[group].items()}
            arguments = (carry, obs, reset)
            if self.ablations is not None and self.ablatable[group]:
                arguments += (self.ablations,)
            inputs = _pytree.tree_map_only(torch.Tensor, lambda x: x[rows], arguments)
            outputs.append(torch.func.functional_call(self.decisions[group], parameters, inputs))

        def joined(*leaves: torch.Tensor | None) -> torch.Tensor | None:
            return None if leaves[0] is None else torch.cat(leaves)  # None: a memoryless network

        return _pytree.tree_map(joined, outputs[0], *outputs[1:])

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
        model_carry = carry.model
        if self.decisions:
            results = [
                self._network(group, carry.model.get(str(group)), obs, reset)
                for group in range(len(self.decisions))
            ]
            network = results[0][0]
            for group in range(1, len(results)):
                mine = (self.group == group)[:, None, None]
                network = torch.where(mine, results[group][0], network)
            model_carry = {
                str(group): result[1]
                for group, result in enumerate(results)
                if result[1] is not None
            }
            action = torch.where(self.scripted[:, None, None], action, network)
        carry = MatchCarry(
            model=model_carry,
            scripted=scripted_carry,
            finished=carry.finished | (reset & carry.started),
            started=torch.ones_like(carry.started),
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

    # ---- the host's side, between batches

    @torch.no_grad()
    def load_network(
        self,
        slot: int,
        group: int,
        state: dict[str, torch.Tensor],
        ablations: tuple[str, ...] = (),
    ) -> None:
        """Slot `slot` plays a network of group `group`: its state (its model's state dict),
        copied in place, with `ablations` (names of `ABLATIONS`) on its rows."""
        if not 0 <= group < len(self.decisions):
            raise ValueError(f"the seat was built with {len(self.decisions)} network groups")
        if ablations and (self.ablations is None or not self.ablatable[group]):
            raise ValueError("only a Yemong network of a seat built with ablations plays ablated")
        for name, value in state.items():
            self.parameters[group][f"model.{name}"][slot].copy_(value)
        rows = self._slot_rows(slot)
        self.group[rows] = group
        self.scripted[rows] = False
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


class Arena:
    """The match engine over `players` in `game` (a game preset's config), built once (see
    the module docstring). Each network player's weights are read once and kept on the
    host (an ablated copy shares its policy's, and its group); a batch copies them into its
    slots, with the player's ablations. `groups`: a template model per build, in the
    order the builds first appear among the players (`group_of` maps a network player's index
    to its group)."""

    def __init__(
        self,
        players: list[Player],
        game: GameConfig,
        device: torch.device,
        config: ArenaConfig = ArenaConfig(),
    ) -> None:
        self.players, self.config, self.device = players, config, device
        self.states: dict[int, dict[str, torch.Tensor]] = {}
        self.group_of: dict[int, int] = {}
        self.groups: list[torch.nn.Module] = []
        loaded: dict[pathlib.Path, int] = {}  # a policy's first player
        for index, one in enumerate(players):
            if scripted_play(one) is not None:
                continue
            if one.policy in loaded:  # an ablated copy (or a repeat): the same weights
                first = loaded[one.policy]
                self.group_of[index], self.states[index] = self.group_of[first], self.states[first]
            else:
                loaded[one.policy] = index
                model = checkpoint.load_policy(one.policy, device).requires_grad_(False)
                for group, template in enumerate(self.groups):
                    if type(model) is type(template) and model.build == template.build:
                        break
                else:
                    group = len(self.groups)
                    self.groups.append(model)
                self.group_of[index] = group
                self.states[index] = {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                }
            if one.ablations and not isinstance(self.groups[self.group_of[index]], Yemong):
                raise ValueError(f"{one.name}: only a Yemong network plays ablated")
        agent = characteristics(api.ship_types(device), api.rules_of(game))
        controllers = [model_controller(template, game) for template in self.groups]
        self.seat = MatchSeat(
            controllers,
            config.blocks,
            config.block_environments,
            agent,
            game.ships_per_team,
            device,
            ablations=any(one.ablations for one in players),
        )
        pools = tuple(
            Pool(f"block_{index}", 1.0 / config.blocks, (SEAT, SEAT))
            for index in range(config.blocks)
        )
        self.rollout = Rollout(
            game,
            api.ship_types(device),
            pools,
            {SEAT: self.seat},
            SEAT,
            config.games,
            config.length,
            device,
            compiled=config.compiled,
            graph_ticks=config.graph_ticks,
            staggered=False,
        )
        # environment i seats its block's first player on team i mod 2
        self.seats = torch.arange(config.games, device=device) % 2
        # every first game ends by its clock; the margin covers the segment it ends in
        self.tick_limit = int(self.rollout.episode_ticks.max()) + 2 * config.length

    def _load(self, slot: int, index: int) -> None:
        play = scripted_play(self.players[index])
        if play is None:
            self.seat.load_network(
                slot, self.group_of[index], self.states[index], self.players[index].ablations
            )
        else:
            self.seat.load_scripted(slot, play)

    def play(self, matchups: list[tuple[int, int]]) -> torch.Tensor:
        """One batch: block b plays `matchups[b]` (its first player, its second; indices into
        the players), every block's games to their end. (blocks, 3) int64: each block's first
        player's wins, ties and losses. Raises RuntimeError if a game runs past the game
        clock (plus a margin) without ending."""
        if len(matchups) != self.config.blocks:
            raise ValueError(f"{len(matchups)} matchups for {self.config.blocks} blocks")
        for block, (first, second) in enumerate(matchups):
            if first == second:
                raise ValueError(f"block {block}: {self.players[first].name} against itself")
            self._load(2 * block, first)
            self._load(2 * block + 1, second)
        self.rollout.restart()
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
