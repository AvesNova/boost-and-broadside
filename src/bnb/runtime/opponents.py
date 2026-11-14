"""The league's opponent seat inside the rollout (league spec 2): network slots with stacked
parameters, the scripted agent and the clean cut.

One controller, `OpponentSlots`, plays every opponent row of the batch:
- `slots` network slots, with their parameters stacked `(slots, ...)`, run as one forward
  over the stacked sets (`bnb.runtime.stacked`, `torch.vmap`), each set on its own rows:
  slot k plays rows k, k + slots, k + 2 slots, ... (so every slot plays every mode).
- the full-strength scripted agent (temperature 1): one scripted pass over every row. The
  training pool holds no other scripted player, no weaker rung and no random player (the
  author, 2026-11-12); the rating lanes and the offline tournament still play the ladder.
- per row, the kind it plays (a network set or the scripted agent) is selected with `where`.

A slot holds a network entry or the scripted agent.

The clean cut (`cut`): the host loads a new entry into a slot's set in place, at an update
boundary. The games in progress on its rows play on against the new entry; their results
are left out of the tally (the learner's ticks still train). The opponent's model carry on
those rows is reset (its trunk; the belief part is kept) or kept. Every host-side change is
an in-place copy between segments, so a captured graph stays valid.

Each row's column (`OpponentCarry.column`) names what it plays for the record: a slot's
set, or the scripted agent (the column after the slots'). The rollout tallies each
episode's result from the learner's side by column and mode (`tally`).
"""

import dataclasses
import math

import torch
from torch.utils import _pytree

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.tree import contract
from bnb.controllers.scripted import Scripted
from bnb.controllers.scripted_memory import ScriptedCarry
from bnb.game.handling.characteristics import Characteristics

from . import stacked

OPPONENTS = "opponents"  # the controller name of the league's seat


@dataclasses.dataclass(frozen=True)
class Rung:
    """A ladder entry: the scripted agent at `temperature` (+inf random), each ship playing
    it with probability `share` per game and uniformly otherwise (a mixture rung)."""

    temperature: float
    share: float = 1.0


RANDOM = Rung(math.inf)
# The stationary ladder (league spec 1), weakest first: random (temperature +inf), mixture
# rungs (each ship the agent at T = 2 with probability share, else random), the scripted
# agent at falling temperatures, the scripted agent itself (T = 1). The section notes give
# the temperature curve and the mixtures' scores the rungs were chosen from.
LADDER = (
    RANDOM,
    Rung(2.0, 0.1),
    Rung(2.0, 0.3),
    Rung(2.0, 0.5),
    Rung(8.0),
    Rung(4.0),
    Rung(3.0),
    Rung(2.5),
    Rung(2.0),
    Rung(1.5),
    Rung(1.0),
)
RUNG_PREFIX = "rung_T"
SHARE_PREFIX = "_p"
SCRIPTED_PLAYER = "scripted"


def ladder_name(rung: Rung) -> str:
    """A ladder entry's name: "random" (+inf), "scripted" (1), "rung_T<temperature>", or
    "rung_T<temperature>_p<share>" for a mixture rung."""
    if rung.temperature == math.inf:
        return "random"
    if rung.share != 1.0:
        return f"{RUNG_PREFIX}{rung.temperature:g}{SHARE_PREFIX}{rung.share:g}"
    if rung.temperature == 1.0:
        return SCRIPTED_PLAYER
    return f"{RUNG_PREFIX}{rung.temperature:g}"


def ladder_rung(name: str) -> Rung | None:
    """A ladder entry's rung from its name (`ladder_name`'s inverse), or None for another
    name."""
    if name == "random":
        return RANDOM
    if name == SCRIPTED_PLAYER:
        return Rung(1.0)
    if name.startswith(RUNG_PREFIX):
        temperature, _, share = name[len(RUNG_PREFIX) :].partition(SHARE_PREFIX)
        try:
            rung = Rung(float(temperature), float(share) if share else 1.0)
        except ValueError:
            return None
        valid = 1.0 <= rung.temperature < math.inf and 0.0 < rung.share <= 1.0
        if valid and ladder_name(rung) == name:
            return rung
    return None


NETWORK, SCRIPTED_KIND = 0, 1  # what a slot holds
OUTCOMES = ("wins", "ties", "losses")  # the tally's last axis, from the learner's side


CARRY_AT_CUT = ("reset", "keep")


@dataclasses.dataclass(frozen=True)
class LeagueConfig:
    """slots: network slots loaded at once; cut_every: updates between the clean cuts;
    cut_slots: the slots a cut draws anew (round robin); carry_at_cut: the opponent's model
    carry on a cut slot's rows, "keep" (the old occupant's, when both are networks of the
    one build; else reset) or "reset" (the trunk zeroed, the belief kept); floor: the
    share of the pool's sampling weight spread evenly over the entries; exponential_decay:
    the exponential average's decay per update; uniform_from: the environment steps after
    which the uniform average starts (before them it is the live weights: a warm start's
    optimizer, critic and advantages settle first); None, the default, is the run's
    learning-rate warm-up (`LearnerConfig.warmup_steps`, `uniform_start`), so a fine-tune's
    shorter warm-up starts its average sooner; snapshot_every: environment steps between
    snapshots (None: measured once, about 30 minutes of the run's throughput, and
    recorded)."""

    slots: int = 4
    # the author's clean cut, 2026-11-06 (decisions: round four)
    cut_every: int = 2
    cut_slots: int = 1
    carry_at_cut: str = "keep"  # the hand-over test's verdict (author, 2026-11-06)
    floor: float = 0.05
    exponential_decay: float = 0.95
    # the run's warm-up (author, 2026-11-07: decisions, run control)
    uniform_from: int | None = None
    snapshot_every: int | None = None

    def __post_init__(self) -> None:
        if self.slots < 1:
            raise ValueError(f"league slots {self.slots} must be at least 1")
        if self.cut_every < 1 or not 1 <= self.cut_slots <= self.slots:
            raise ValueError(
                f"league cut_every {self.cut_every} must be at least 1 and cut_slots "
                f"{self.cut_slots} in [1, slots]"
            )
        if self.carry_at_cut not in CARRY_AT_CUT:
            raise ValueError(f"league carry_at_cut {self.carry_at_cut!r}; one of {CARRY_AT_CUT}")
        if not 0.0 <= self.floor < 1.0:
            raise ValueError(f"league floor {self.floor} must be in [0, 1)")
        if not 0.0 < self.exponential_decay < 1.0:
            raise ValueError(f"league exponential_decay {self.exponential_decay} must be in (0, 1)")
        if self.uniform_from is not None and self.uniform_from < 0:
            raise ValueError(f"league uniform_from {self.uniform_from} must not be negative")
        if self.snapshot_every is not None and self.snapshot_every < 1:
            raise ValueError(f"league snapshot_every {self.snapshot_every} must be positive")

    def uniform_start(self, warmup_steps: int) -> int:
        """The environment steps the uniform average starts after: `uniform_from`, or the
        run's warm-up when it is None."""
        return warmup_steps if self.uniform_from is None else self.uniform_from


def whole_game_begins(reset: torch.Tensor, stepped: torch.Tensor) -> torch.Tensor:
    """Whether each row begins a whole game at this tick (`reset` (R,), `stepped` (R,), the
    row has played a tick before): every game but the row's first, which the rollout's
    staggered clocks cut short. Cut-short games lean to ties and read as a rating near the
    scripted agent's, so neither the lanes nor the league tally a row's first game."""
    return reset & stepped


@contract
class OpponentCarry:
    """Per opponent row (R leading): model, the network's carry (its controller's);
    column (R,) int64, what the row plays, for the tally; stepped (R,) bool, the row has
    played a tick; whole (R,) bool, the row's game began after its first tick
    (`whole_game_begins`) and has not been cut, so the tally counts it; scripted, the
    scripted agent's carry."""

    model: object
    column: torch.Tensor
    stepped: torch.Tensor
    whole: torch.Tensor
    scripted: ScriptedCarry


class _Decision(torch.nn.Module):
    """A network controller's decision as a module call, for `functional_call`: (the packed
    action (R, S, 3), the next carry)."""

    def __init__(self, controller) -> None:
        super().__init__()
        self.model = controller.model
        self.controller = controller

    def forward(self, carry, obs: Obs, reset: torch.Tensor, decision_key: torch.Tensor):
        action, carry, _ = self.controller.step(carry, obs, reset, decision_key)
        return action.packed(), carry


class OpponentSlots:
    """The league's seat (see the module docstring). controller: a network controller of the
    live policy's architecture over a model of its own (never the learner's: the forward
    swaps the sets' parameters in); modes (R,) int64: each opponent row's mode index;
    mode_count: how many modes; characteristics and ships_per_team: the scripted
    agent's."""

    def __init__(
        self,
        controller,
        modes: torch.Tensor,
        mode_count: int,
        characteristics: Characteristics,
        ships_per_team: int,
        config: LeagueConfig,
    ) -> None:
        rows, slots = modes.shape[0], config.slots
        if rows % slots:
            raise ValueError(f"{rows} opponent rows do not split evenly over {slots} slots")
        device = modes.device
        self.config, self.slots = config, slots
        self.mode_count, self.modes = mode_count, modes
        self.decision = _Decision(controller)
        self.decision.requires_grad_(False)
        self.reads = tuple(dict.fromkeys(controller.reads + Scripted.reads))
        self.scripted_agent = Scripted(characteristics, ships_per_team)  # full strength
        state = self.decision.state_dict()
        # The stacked sets, one per slot.
        self.parameters = {
            name: value.detach()[None].repeat((slots,) + (1,) * value.dim()).clone()
            for name, value in state.items()
        }
        self.rows = torch.arange(rows, device=device)
        self.slot_of_row = self.rows % slots
        self.kind = torch.full((slots,), NETWORK, dtype=torch.int64, device=device)
        self.scripted_column = slots  # the tally's column for the scripted agent
        self.columns = slots + 1

    # ---- the controller

    def init_carry(self, views: int, device: torch.device) -> OpponentCarry:
        """Every row on its slot's set."""
        return OpponentCarry(
            model=self.decision.controller.init_carry(views, device),
            column=self.slot_of_row.clone(),
            stepped=torch.zeros(views, dtype=torch.bool, device=device),
            whole=torch.zeros(views, dtype=torch.bool, device=device),
            scripted=self.scripted_agent.init_carry(views, device),
        )

    def _network(self, carry, obs: Obs, reset: torch.Tensor, decision_key: torch.Tensor):
        """Every set's decision on its own rows, as one forward over the sets' stacked
        parameters (`bnb.runtime.stacked`): (packed actions (R, S, 3), the network
        carry)."""
        slots = self.slots
        by_set = self.rows.reshape(-1, slots).T  # (slots, R / slots): row k is slot k's rows
        inputs = _pytree.tree_map_only(
            torch.Tensor,
            lambda x: x[by_set.reshape(-1)].unflatten(0, by_set.shape),
            (carry, obs, reset, decision_key),
        )
        outputs = stacked.call(self.decision, self.parameters, inputs)
        own_rows = by_set.reshape(-1)

        def assemble(stacked_leaf: torch.Tensor | None) -> torch.Tensor | None:
            if stacked_leaf is None:  # a memoryless network's carry
                return None
            flat = stacked_leaf.flatten(0, 1)
            return torch.empty_like(flat).index_copy(0, own_rows, flat)

        return _pytree.tree_map(assemble, outputs, is_leaf=lambda x: x is None)

    def step(self, carry: OpponentCarry, obs: Obs, reset: torch.Tensor, decision_key: torch.Tensor):
        """One decision for every opponent row; see `Controller.step`."""
        network_action, model_carry = self._network(carry.model, obs, reset, decision_key)
        scripted = self.kind[self.slot_of_row] == SCRIPTED_KIND  # (R,)
        scripted_action, scripted_carry, _ = self.scripted_agent.step(
            carry.scripted, obs, reset, decision_key
        )
        action = torch.where(scripted[:, None, None], scripted_action.packed(), network_action)
        column = torch.where(scripted, self.scripted_column, self.slot_of_row)
        carry = OpponentCarry(
            model=model_carry,
            column=column,
            stepped=torch.ones_like(carry.stepped),
            whole=carry.whole | whole_game_begins(reset, carry.stepped),
            scripted=scripted_carry,
        )
        return Action.unpack(action), carry, Outputs()

    def tally(self, carry: OpponentCarry, done: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        """Each ended episode (`done` (R,)), by (column, mode, outcome) with the outcome from
        the learner's side (`result` (R,) int32 is the opponent row's own: +1 won, 0 tied,
        -1 lost): flat (columns * modes * 3,) int64. A row's first game, cut short, is left
        out (`whole_game_begins`)."""
        outcome = (1 + result).to(torch.int64)  # the learner's win 0, tie 1, loss 2
        flat = (carry.column * self.mode_count + self.modes) * len(OUTCOMES) + outcome
        counts = torch.zeros(
            self.columns * self.mode_count * len(OUTCOMES), dtype=torch.int64, device=done.device
        )
        return counts.index_add(0, flat, (done & carry.whole).to(torch.int64))

    # ---- the host's side, between segments

    @torch.no_grad()
    def load(self, slot: int, state: dict[str, torch.Tensor]) -> None:
        """Copy a network's state (its model's state dict) into slot `slot`'s set, in place."""
        for name, value in state.items():
            self.parameters[f"model.{name}"][slot].copy_(value)

    def fill(self, state: dict[str, torch.Tensor]) -> None:
        """Every set holds `state`, as a network: the slots before a league draws them."""
        for index in range(self.slots):
            self.load(index, state)
        self.kind.fill_(NETWORK)

    def set_kind(self, slot: int, kind: int) -> None:
        """What slot `slot` holds: NETWORK or SCRIPTED_KIND."""
        self.kind[slot] = kind

    @torch.no_grad()
    def cut(self, carry: OpponentCarry, slot: int, keep_memory: bool) -> int:
        """The clean cut of `slot`, after its new entry is loaded (`load`, `set_kind`): the
        games in progress on its rows are no longer tallied (until each row's next reset),
        and unless `keep_memory` the network carry's memory on those rows is reset (the
        controller's `forget`: Yemong's trunk; its belief is kept). Returns the games left
        uncredited: the slot's rows whose game the tally would have counted (a host read)."""
        rows = self.slot_of_row == slot
        uncredited = int((carry.whole & rows).sum())
        carry.whole.copy_(carry.whole & ~rows)
        forget = getattr(self.decision.controller, "forget", None)
        if not keep_memory and forget is not None and carry.model is not None:
            forgotten = forget(carry.model, rows)
            for target, source in zip(
                _pytree.tree_leaves(carry.model), _pytree.tree_leaves(forgotten), strict=True
            ):
                if isinstance(target, torch.Tensor) and target is not source:
                    target.copy_(source)
        return uncredited

    def state_dict(self) -> dict[str, torch.Tensor]:
        """The sets and what each holds (a resume restores them with `load_state_dict`)."""
        state = {f"parameters.{name}": value for name, value in self.parameters.items()}
        state.update(kind=self.kind)
        return state

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Copy a `state_dict` back in place."""
        for name, value in self.state_dict().items():
            value.copy_(state[name])
