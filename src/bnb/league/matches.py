"""The offline match engine (league spec 6): players named on the command line, and batches
of games between pairs of them on fixed slots.

A player is a policy file (or a training checkpoint, whose policy part plays), the newest
checkpoint of a run, a ladder entry ("scripted", "random" or "rung_T<t>"), or a teacher:
the scripted agent at its own temperature T with a uniform floor on its actions,
`scripted@T=<temperature>,floor=<floor>` (either key optional: T defaults to 1, floor to 0;
BC's teacher is `scripted`, T = 1 and floor 0). A ladder entry and a teacher are scripted
players; the others are networks, and the networks of one engine share one architecture
(their weights are stacked).

The engine (`Arena`) is one rollout of one game preset, built once: `blocks` matchup blocks
of `block_environments` environments each. Each block is a pool of its own whose two seats
are both the one controller `MatchSeat`; block b's first seat is slot 2b and its second
slot 2b + 1, so each slot is a fixed, contiguous run of rows. A slot holds a network or a
scripted player. A network slot's weights are one set of the stacked parameters, run on the
slot's own rows only (`torch.func.functional_call` on its slice, as the rating lanes'
`bnb.runtime.lanes.LaneSeat` does); a scripted slot's rows take the scripted pass, one over
every row, at each row's temperature, share and floor. The shapes never change, so the
rollout's graph is captured once; a batch only copies the chosen players into the slots, in
place.

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
    temperature with a floor)."""

    name: str
    policy: pathlib.Path | None = None
    rung: Rung | None = None
    teacher: TeacherConfig | None = None


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


def player(name: str, runs_directory: pathlib.Path | None = None) -> Player:
    """The player a command-line name means: a ladder entry, a policy directory (one with a
    checkpoint.json), or a run's newest checkpoint."""
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
    """Per row (R leading): model, the networks' carry (their controller's); scripted, the
    scripted agent's; finished (R,) bool, the row's first game has ended; started (1,) bool,
    a decision has been made (the first observation's reset is the batch's start, not a
    game's end)."""

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

    def forward(self, carry, obs: Obs, reset: torch.Tensor):
        action, carry, _ = self.controller.step(carry, obs, reset)
        return action.packed(), carry


class MatchSeat:
    """Every row's player, in one controller (see the module docstring). controller: a
    network controller over a model of its own (the forward swaps each slot's weights in),
    or None when every player is scripted; blocks and block_environments: the layout (rows
    are block by block, each block's first seat's rows then its second's, as the lineup
    lists them); characteristics and ships_per_team: the scripted agent's. Every slot starts
    as the scripted agent."""

    def __init__(
        self,
        controller,
        blocks: int,
        block_environments: int,
        characteristics: Characteristics,
        ships_per_team: int,
        device: torch.device,
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
        self.decision = None if controller is None else _Decision(controller)
        self.parameters: dict[str, torch.Tensor] = {}
        reads = Scripted.reads
        if self.decision is not None:
            self.decision.requires_grad_(False)
            reads = controller.reads + reads
            self.parameters = {
                name: value.detach()[None].repeat((self.slots,) + (1,) * value.dim()).clone()
                for name, value in self.decision.state_dict().items()
            }
        self.reads = tuple(dict.fromkeys(reads))

    def _slot_rows(self, slot: int) -> slice:
        return slice(slot * self.slot_rows, (slot + 1) * self.slot_rows)

    # ---- the controller

    def init_carry(self, views: int, device: torch.device) -> MatchCarry:
        """Every row's network carry (a scripted row's is never read) and scripted carry,
        nothing finished."""
        model = (
            None if self.decision is None else self.decision.controller.init_carry(views, device)
        )
        return MatchCarry(
            model=model,
            scripted=self.agent.init_carry(views, device),
            finished=torch.zeros(views, dtype=torch.bool, device=device),
            started=torch.zeros(1, dtype=torch.bool, device=device),
        )

    def _network(self, carry, obs: Obs, reset: torch.Tensor):
        """Each slot's network on the slot's own rows, the slots one after another: (packed
        actions (R, S, 3), the network carry)."""
        outputs = []
        for slot in range(self.slots):
            rows = self._slot_rows(slot)
            parameters = {name: value[slot] for name, value in self.parameters.items()}
            inputs = _pytree.tree_map_only(torch.Tensor, lambda x: x[rows], (carry, obs, reset))
            outputs.append(torch.func.functional_call(self.decision, parameters, inputs))

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
        if self.decision is not None:
            network, model_carry = self._network(carry.model, obs, reset)
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
    def load_network(self, slot: int, state: dict[str, torch.Tensor]) -> None:
        """Slot `slot` plays a network: its state (its model's state dict), copied in place."""
        if self.decision is None:
            raise ValueError("the seat was built without a network")
        for name, value in state.items():
            self.parameters[f"model.{name}"][slot].copy_(value)
        self.scripted[self._slot_rows(slot)] = False

    @torch.no_grad()
    def load_scripted(self, slot: int, play: ScriptedPlay) -> None:
        """Slot `slot` plays the scripted agent as `play` says, written in place."""
        rows = self._slot_rows(slot)
        self.temperature[rows] = play.temperature
        self.share[rows] = play.share
        self.floor[rows] = play.floor
        self.scripted[rows] = True

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
    host; a batch copies them into its slots."""

    def __init__(
        self,
        players: list[Player],
        game: GameConfig,
        device: torch.device,
        config: ArenaConfig = ArenaConfig(),
    ) -> None:
        self.players, self.config, self.device = players, config, device
        self.states: dict[int, dict[str, torch.Tensor]] = {}
        template = None
        for index, one in enumerate(players):
            if scripted_play(one) is not None:
                continue
            model = checkpoint.load_policy(one.policy, device).requires_grad_(False)
            if template is None:
                template = model
            elif type(model) is not type(template) or model.build != template.build:
                raise ValueError(
                    f"{one.name}: the networks of one tournament share one architecture "
                    f"(their weights are stacked in the slots); it differs from "
                    f"{players[min(self.states)].name}'s"
                )
            self.states[index] = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
        agent = characteristics(api.ship_types(device), api.rules_of(game))
        controller = None if template is None else model_controller(template, game)
        self.seat = MatchSeat(
            controller,
            config.blocks,
            config.block_environments,
            agent,
            game.ships_per_team,
            device,
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

    def _load(self, slot: int, index: int) -> None:
        play = scripted_play(self.players[index])
        if play is None:
            self.seat.load_network(slot, self.states[index])
        else:
            self.seat.load_scripted(slot, play)

    def play(self, matchups: list[tuple[int, int]]) -> torch.Tensor:
        """One batch: block b plays `matchups[b]` (its first player, its second; indices into
        the players), every block's games to their end. (blocks, 3) int64: each block's first
        player's wins, ties and losses."""
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
        while True:
            self.rollout.run()
            counts += self.rollout.take_tally(SEAT)
            if self.seat.finished(self.rollout.carry["controllers"][SEAT]):
                break
        return counts.reshape(self.config.blocks, len(OUTCOMES))
