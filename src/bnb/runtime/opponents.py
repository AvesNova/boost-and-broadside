"""The league's opponent seat inside the rollout (league spec 2): network slots with stacked
parameters, the stationary ladder, the clean cut and the in-graph draws.

One controller, `OpponentSlots`, plays every opponent row of the batch:
- `slots` network slots, with their parameters stacked `(slots, ...)`. Each set runs on its
  own rows in the one compiled graph (`torch.func.functional_call` on the set's slice): slot
  k plays rows k, k + slots, k + 2 slots, ... (so every slot plays every mode). (Not
  `torch.vmap`: under vmap, scaled-dot-product attention skips autocast's casts, so Yemong's
  plain bf16 path fails.)
- the stationary ladder, when it is in the pool (`LeagueConfig.ladder_in_pool`): one
  scripted pass over every row, each row at its rung's temperature and share (scripted spec
  A6; +inf is random, a share below 1 a mixture rung). Without it the seat runs no scripted
  pass.
- per row, the kind it plays (a network set or the ladder) is selected with `where`.

A slot holds a network entry or a ladder entry. A ladder row draws a new rung at each of its
resets from its mode's ladder weights, by inverse CDF on a counter-based uniform (a hash of
the stream's key, the tick counter and the row), so the draws need no generator state and
repeat exactly.

The clean cut (`cut`): the host loads a new entry into a slot's set in place, at an update
boundary. The games in progress on its rows play on against the new entry; their results
are left out of the tally (the learner's ticks still train). The opponent's model carry on
those rows is reset (its trunk; the belief part is kept) or kept. Every host-side change is
an in-place copy between segments, so a captured graph stays valid.

Each row's column (`OpponentCarry.column`) names what it plays for the record: a slot's
set or a ladder entry. The rollout tallies each episode's result from the learner's side by
column and mode (`tally`).
"""

import dataclasses
import math
import zlib

import torch
from torch.utils import _pytree

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.tree import contract
from bnb.controllers.scripted import Scripted
from bnb.controllers.scripted_memory import ScriptedCarry
from bnb.game.handling.characteristics import Characteristics

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


def strength_order(rung: Rung) -> tuple[float, float]:
    """A key that sorts rungs weakest first: random, the mixtures by share, then the
    temperature rungs by falling temperature."""
    if rung.temperature == math.inf:
        return (0.0, 0.0)
    return (rung.share, -rung.temperature)


NETWORK, LADDER_KIND = 0, 1  # what a slot holds
OUTCOMES = ("wins", "ties", "losses")  # the tally's last axis, from the learner's side
MASK = (1 << 32) - 1


CARRY_AT_CUT = ("reset", "keep")


@dataclasses.dataclass(frozen=True)
class LeagueConfig:
    """slots: network slots loaded at once; cut_every: updates between the clean cuts;
    cut_slots: the slots a cut draws anew (round robin); carry_at_cut: the opponent's model
    carry on a cut slot's rows, "keep" (the old occupant's, when both are networks of the
    one build; else reset) or "reset" (the trunk zeroed, the belief kept); ladder_in_pool:
    the stationary ladder is in the training pool (the rating lanes play it either way);
    ladder: the stationary ladder's rungs, weakest
    first (`strength_order`; +inf random, T = 1 at share 1 the scripted agent); floor: the
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
    ladder_in_pool: bool = False
    ladder: tuple[Rung, ...] = LADDER
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
        rungs = list(self.ladder)
        if not rungs or any(
            not (rung.temperature >= 1.0 and 0.0 < rung.share <= 1.0) for rung in rungs
        ):
            raise ValueError(
                f"league ladder {self.ladder} must be rungs of temperature at least 1 and "
                "share in (0, 1]"
            )
        if rungs != sorted(set(rungs), key=strength_order) or len(
            set(map(strength_order, rungs))
        ) != len(rungs):
            raise ValueError(f"league ladder {self.ladder} must be distinct, weakest first")
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


def stream_key(seed: int, name: str) -> int:
    """A counter-based stream's 32-bit key, from the run's seed and the stream's name."""
    return zlib.crc32(f"{seed}/{name}".encode()) & MASK


def _multiply(value: torch.Tensor, constant: int) -> torch.Tensor:
    """(value * constant) mod 2^32 for int64 values below 2^32, without int64 overflow."""
    high, low = constant >> 16, constant & 0xFFFF
    return (value * low + (((value * high) & 0xFFFF) << 16)) & MASK


def _mix(value: torch.Tensor) -> torch.Tensor:
    """A 32-bit integer hash (lowbias32), on int64 values below 2^32."""
    value = value ^ (value >> 16)
    value = _multiply(value, 0x7FEB352D)
    value = value ^ (value >> 15)
    value = _multiply(value, 0x846CA68B)
    return value ^ (value >> 16)


def counter_uniform(key: int, counter: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Uniform draws in [0, 1) float32, one per `index` (int64), a pure function of the key,
    the counter (int64, broadcast) and the index: the same arguments give the same draws on
    every device."""
    value = _mix((index & MASK) ^ key)
    value = _mix(value ^ (counter & MASK))
    value = _mix(value ^ ((counter >> 32) & MASK))
    return (value >> 8).to(torch.float32) * (1.0 / (1 << 24))


def whole_game_begins(reset: torch.Tensor, stepped: torch.Tensor) -> torch.Tensor:
    """Whether each row begins a whole game at this tick (`reset` (R,), `stepped` (R,), the
    row has played a tick before): every game but the row's first, which the rollout's
    staggered clocks cut short. Cut-short games lean to ties and read as a rating near the
    scripted agent's, so neither the lanes nor the league tally a row's first game."""
    return reset & stepped


def draw(weights: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """One category per row of non-negative `weights` (..., E), by the inverse CDF of
    `uniform` (...): int64. A category of weight 0 is never drawn."""
    cumulative = weights.cumsum(-1)
    target = uniform[..., None] * cumulative[..., -1:]
    return (target >= cumulative[..., :-1]).sum(-1)


@contract
class OpponentCarry:
    """Per opponent row (R leading): model, the network's carry (its controller's);
    ladder_index (R,) int64, the ladder entry the row plays (when its slot holds the ladder);
    column (R,) int64, what the row plays, for the tally; stepped (R,) bool, the row has
    played a tick; whole (R,) bool, the row's game began after its first tick
    (`whole_game_begins`) and has not been cut, so the tally counts it; counter (1,) int64,
    the decisions so far (the ladder draws' counter); ladder, the ladder agent's carry."""

    model: object
    ladder_index: torch.Tensor
    column: torch.Tensor
    stepped: torch.Tensor
    whole: torch.Tensor
    counter: torch.Tensor
    ladder: ScriptedCarry


class _Decision(torch.nn.Module):
    """A network controller's decision as a module call, for `functional_call`: (the packed
    action (R, S, 3), the next carry)."""

    def __init__(self, controller) -> None:
        super().__init__()
        self.model = controller.model
        self.controller = controller

    def forward(self, carry, obs: Obs, reset: torch.Tensor):
        action, carry, _ = self.controller.step(carry, obs, reset)
        return action.packed(), carry


class OpponentSlots:
    """The league's seat (see the module docstring). controller: a network controller of the
    live policy's architecture over a model of its own (never the learner's: the forward
    swaps the sets' parameters in); modes (R,) int64: each opponent row's mode index;
    mode_count: how many modes; characteristics and ships_per_team: the scripted agent's,
    for the ladder; key: the ladder draws' stream key."""

    def __init__(
        self,
        controller,
        modes: torch.Tensor,
        mode_count: int,
        characteristics: Characteristics,
        ships_per_team: int,
        config: LeagueConfig,
        key: int,
    ) -> None:
        rows, slots = modes.shape[0], config.slots
        if rows % slots:
            raise ValueError(f"{rows} opponent rows do not split evenly over {slots} slots")
        device = modes.device
        self.config, self.slots, self.key = config, slots, key
        self.mode_count, self.modes = mode_count, modes
        self.decision = _Decision(controller)
        self.decision.requires_grad_(False)
        self.reads = tuple(dict.fromkeys(controller.reads + Scripted.reads))
        # One scripted pass for the whole ladder: each row's temperature and share are
        # written in place.
        self.ladder_temperature = torch.ones(rows, device=device)
        self.ladder_share = torch.ones(rows, device=device)
        self.ladder_agent = Scripted(
            characteristics,
            ships_per_team,
            temperature=self.ladder_temperature,
            share=self.ladder_share,
        )
        state = self.decision.state_dict()
        # The stacked sets, one per slot.
        self.parameters = {
            name: value.detach()[None].repeat((slots,) + (1,) * value.dim()).clone()
            for name, value in state.items()
        }
        self.rows = torch.arange(rows, device=device)
        self.slot_of_row = self.rows % slots
        self.kind = torch.full((slots,), NETWORK, dtype=torch.int64, device=device)
        ladder = len(config.ladder)
        rungs = list(config.ladder)
        self.temperatures = torch.tensor(
            [rung.temperature for rung in rungs], dtype=torch.float32, device=device
        )
        self.shares = torch.tensor([rung.share for rung in rungs], device=device)
        self.ladder_weights = torch.full((mode_count, ladder), 1.0 / ladder, device=device)
        self.columns = slots + ladder

    # ---- the controller

    def init_carry(self, views: int, device: torch.device) -> OpponentCarry:
        """Every row on its slot's set, the first ladder entry."""
        return OpponentCarry(
            model=self.decision.controller.init_carry(views, device),
            ladder_index=torch.zeros(views, dtype=torch.int64, device=device),
            column=self.slot_of_row.clone(),
            stepped=torch.zeros(views, dtype=torch.bool, device=device),
            whole=torch.zeros(views, dtype=torch.bool, device=device),
            counter=torch.zeros(1, dtype=torch.int64, device=device),
            ladder=self.ladder_agent.init_carry(views, device),
        )

    def _network(self, carry, obs: Obs, reset: torch.Tensor):
        """Every set's decision on its own rows, the sets one after another in the one
        compiled graph: (packed actions (R, S, 3), the network carry)."""
        slots = self.slots
        by_slot = self.rows.reshape(-1, slots)  # (R / slots, slots): column k is slot k's rows
        row_sets = [by_slot[:, slot] for slot in range(slots)]
        outputs = []
        for index, rows in enumerate(row_sets):
            parameters = {name: value[index] for name, value in self.parameters.items()}
            inputs = _pytree.tree_map_only(torch.Tensor, lambda x: x[rows], (carry, obs, reset))
            outputs.append(torch.func.functional_call(self.decision, parameters, inputs))
        own_rows = torch.cat(row_sets)

        def assemble(*leaves: torch.Tensor | None) -> torch.Tensor | None:
            if leaves[0] is None:  # a memoryless network's carry
                return None
            stacked = torch.cat(leaves)
            return torch.empty_like(stacked).index_copy(0, own_rows, stacked)

        return _pytree.tree_map(assemble, outputs[0], *outputs[1:])

    def step(self, carry: OpponentCarry, obs: Obs, reset: torch.Tensor):
        """One decision for every opponent row; see `Controller.step`."""
        network_action, model_carry = self._network(carry.model, obs, reset)
        action, column = network_action, self.slot_of_row
        ladder_index, ladder_carry = carry.ladder_index, carry.ladder
        if self.config.ladder_in_pool:
            ladder = self.kind[self.slot_of_row] == LADDER_KIND  # (R,)
            uniform = counter_uniform(self.key, carry.counter, self.rows)
            drawn = draw(self.ladder_weights[self.modes], uniform)
            ladder_index = torch.where(reset & ladder, drawn, carry.ladder_index)
            self.ladder_temperature.copy_(self.temperatures[ladder_index])
            self.ladder_share.copy_(self.shares[ladder_index])
            ladder_action, ladder_carry, _ = self.ladder_agent.step(carry.ladder, obs, reset)
            action = torch.where(ladder[:, None, None], ladder_action.packed(), network_action)
            column = torch.where(ladder, self.slots + ladder_index, self.slot_of_row)
        carry = OpponentCarry(
            model=model_carry,
            ladder_index=ladder_index,
            column=column,
            stepped=torch.ones_like(carry.stepped),
            whole=carry.whole | whole_game_begins(reset, carry.stepped),
            counter=carry.counter + 1,
            ladder=ladder_carry,
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
        """What slot `slot` holds: NETWORK or LADDER_KIND."""
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

    @torch.no_grad()
    def draw_ladder(self, carry: OpponentCarry, slot: int, uniform: torch.Tensor) -> None:
        """A ladder entry for each row of `slot` (now holding the ladder), from its mode's
        weights and `uniform` (R,): rows that play the ladder before their first reset."""
        rows = self.slot_of_row == slot
        drawn = draw(self.ladder_weights[self.modes], uniform.to(self.modes.device))
        carry.ladder_index.copy_(torch.where(rows, drawn, carry.ladder_index))
        column = self.slots + carry.ladder_index
        carry.column.copy_(torch.where(rows, column, carry.column))

    @torch.no_grad()
    def set_ladder_weights(self, weights: torch.Tensor) -> None:
        """Each mode's sampling weights over the ladder's entries, (modes, ladder)."""
        self.ladder_weights.copy_(weights)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """The sets, what each holds and the ladder weights (a resume restores them with
        `load_state_dict`)."""
        state = {f"parameters.{name}": value for name, value in self.parameters.items()}
        state.update(kind=self.kind, ladder_weights=self.ladder_weights)
        return state

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Copy a `state_dict` back in place."""
        for name, value in self.state_dict().items():
            value.copy_(state[name])
