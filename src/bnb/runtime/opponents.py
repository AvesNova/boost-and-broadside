"""The league's opponent seat inside the rollout (league spec 2): network slots with stacked
parameters, the stationary ladder, the drain and the in-graph draws.

One controller, `OpponentSlots`, plays every opponent row of the batch:
- `slots` network slots, with their parameters stacked `(slots + 1, ...)`; the extra set
  holds the incoming entry of the slot being drained. Each set runs on its own rows in the
  one compiled graph (`torch.func.functional_call` on the set's slice): slot k plays rows
  k, k + slots, k + 2 slots, ... (so every slot plays every mode), and the extra set plays
  the draining slot's rows. (Not `torch.vmap`: under vmap, scaled-dot-product attention
  skips autocast's casts, so Yemong's plain bf16 path fails.)
- the stationary ladder: one scripted pass and one uniform draw over every row, then each
  row's scripted probability `p` decides (scripted spec A6).
- per row, the kind it plays (a network set or the ladder) is selected with `where`.

A slot holds a network entry or a ladder entry. A ladder row draws a new `p` at each of its
resets from its mode's ladder weights, by inverse CDF on a counter-based uniform (a hash of
the stream's key, the tick counter and the row), so the draws need no generator state and
repeat exactly.

The drain (review 4b B1): the host loads a new entry for one slot into the extra set and
marks that slot draining. Each of its rows switches to the incoming entry at its own next
reset, so no opponent changes mid-episode. When every row has switched, the host copies the
extra set into the slot's set and the drain ends; at most one slot drains at a time. Every
host-side change is an in-place copy between segments, so a captured graph stays valid.

Each row's column (`OpponentCarry.column`) names what it plays for the record: a slot's
set, the incoming set, or a ladder entry. The rollout tallies each episode's result from
the learner's side by column and mode (`tally`).
"""

import dataclasses
import zlib

import torch
from torch.utils import _pytree

from bnb.contracts import Action, Obs, Outputs
from bnb.contracts.tree import contract
from bnb.controllers.scripted import Scripted
from bnb.game.handling.characteristics import Characteristics

OPPONENTS = "opponents"  # the controller name of the league's seat
# The stationary ladder (league spec 1): random, the rungs in tenths plus 0.95, the scripted
# agent.
LADDER = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
NETWORK, LADDER_KIND = 0, 1  # what a slot holds
OUTCOMES = ("wins", "ties", "losses")  # the tally's last axis, from the learner's side
MASK = (1 << 32) - 1


@dataclasses.dataclass(frozen=True)
class LeagueConfig:
    """slots: network slots loaded at once; ladder: the stationary ladder's scripted
    probabilities; floor: the share of the pool's sampling weight spread evenly over the
    entries; exponential_decay: the exponential average's decay per update;
    snapshot_every: environment steps between snapshots (None: measured once, about 30
    minutes of the run's throughput, and recorded)."""

    slots: int = 4
    ladder: tuple[float, ...] = LADDER
    floor: float = 0.05
    exponential_decay: float = 0.95
    snapshot_every: int | None = None

    def __post_init__(self) -> None:
        if self.slots < 1:
            raise ValueError(f"league slots {self.slots} must be at least 1")
        if not self.ladder or any(not 0.0 <= p <= 1.0 for p in self.ladder):
            raise ValueError(f"league ladder {self.ladder} must be probabilities in [0, 1]")
        if list(self.ladder) != sorted(set(self.ladder)):
            raise ValueError(f"league ladder {self.ladder} must be increasing")
        if not 0.0 <= self.floor < 1.0:
            raise ValueError(f"league floor {self.floor} must be in [0, 1)")
        if not 0.0 < self.exponential_decay < 1.0:
            raise ValueError(f"league exponential_decay {self.exponential_decay} must be in (0, 1)")
        if self.snapshot_every is not None and self.snapshot_every < 1:
            raise ValueError(f"league snapshot_every {self.snapshot_every} must be positive")


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


def draw(weights: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """One category per row of non-negative `weights` (..., E), by the inverse CDF of
    `uniform` (...): int64. A category of weight 0 is never drawn."""
    cumulative = weights.cumsum(-1)
    target = uniform[..., None] * cumulative[..., -1:]
    return (target >= cumulative[..., :-1]).sum(-1)


@contract
class OpponentCarry:
    """Per opponent row (R leading): model, the network's carry (its controller's); incoming
    (R,) bool, the row has switched to the draining slot's incoming entry; ladder_index (R,)
    int64, the ladder entry the row plays (when its slot holds the ladder); column (R,)
    int64, what the row plays, for the tally; counter (1,) int64, the decisions so far (the
    ladder draws' counter)."""

    model: object
    incoming: torch.Tensor
    ladder_index: torch.Tensor
    column: torch.Tensor
    counter: torch.Tensor


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
    mode_count: how many modes; characteristics: the scripted agent's, for the ladder; key:
    the ladder draws' stream key."""

    def __init__(
        self,
        controller,
        modes: torch.Tensor,
        mode_count: int,
        characteristics: Characteristics,
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
        # One scripted pass for the whole ladder: each row's p is written in place.
        self.ladder_probability = torch.ones(rows, device=device)
        self.ladder_agent = Scripted(characteristics, scripted_probability=self.ladder_probability)
        state = self.decision.state_dict()
        # The stacked sets: slots, then the incoming entry of the draining slot.
        self.parameters = {
            name: value.detach()[None].repeat((slots + 1,) + (1,) * value.dim()).clone()
            for name, value in state.items()
        }
        self.rows = torch.arange(rows, device=device)
        self.slot_of_row = self.rows % slots
        self.kind = torch.full((slots + 1,), NETWORK, dtype=torch.int64, device=device)
        self.draining = torch.full((1,), -1, dtype=torch.int64, device=device)
        ladder = len(config.ladder)
        self.probabilities = torch.tensor(config.ladder, dtype=torch.float32, device=device)
        self.ladder_weights = torch.full((mode_count, ladder), 1.0 / ladder, device=device)
        self.columns = slots + 1 + ladder

    # ---- the controller

    def init_carry(self, views: int, device: torch.device) -> OpponentCarry:
        """Every row on its slot's set, no drain, the first ladder entry."""
        return OpponentCarry(
            model=self.decision.controller.init_carry(views, device),
            incoming=torch.zeros(views, dtype=torch.bool, device=device),
            ladder_index=torch.zeros(views, dtype=torch.int64, device=device),
            column=self.slot_of_row.clone(),
            counter=torch.zeros(1, dtype=torch.int64, device=device),
        )

    def _network(self, carry, obs: Obs, reset: torch.Tensor, incoming: torch.Tensor):
        """Every set's decision on its own rows, the sets one after another in the one
        compiled graph: (packed actions (R, S, 3), the network carry), each row taking the
        incoming set's output where it is `incoming`."""
        slots = self.slots
        by_slot = self.rows.reshape(-1, slots)  # (R / slots, slots): column k is slot k's rows
        spare = by_slot[:, 0] + self.draining.clamp_min(0)  # the draining slot's rows
        row_sets = [by_slot[:, slot] for slot in range(slots)] + [spare]
        outputs = []
        for index, rows in enumerate(row_sets):
            parameters = {name: value[index] for name, value in self.parameters.items()}
            inputs = _pytree.tree_map_only(torch.Tensor, lambda x: x[rows], (carry, obs, reset))
            outputs.append(torch.func.functional_call(self.decision, parameters, inputs))
        own_rows = torch.cat(row_sets[:slots])

        def assemble(*leaves: torch.Tensor | None) -> torch.Tensor | None:
            if leaves[0] is None:  # a memoryless network's carry
                return None
            stacked = torch.cat(leaves[:slots])
            own = torch.empty_like(stacked).index_copy(0, own_rows, stacked)
            switched = own.index_copy(0, spare, leaves[slots])
            chosen = incoming.reshape(incoming.shape + (1,) * (own.dim() - 1))
            return torch.where(chosen, switched, own)

        return _pytree.tree_map(assemble, outputs[0], *outputs[1:])

    def step(self, carry: OpponentCarry, obs: Obs, reset: torch.Tensor):
        """One decision for every opponent row; see `Controller.step`."""
        slots = self.slots
        draining = (self.slot_of_row == self.draining) & (self.draining >= 0)
        incoming = carry.incoming | (reset & draining)
        kind = torch.where(incoming, self.kind[slots], self.kind[self.slot_of_row])  # (R,)
        ladder = kind == LADDER_KIND
        network_action, model_carry = self._network(carry.model, obs, reset, incoming)
        uniform = counter_uniform(self.key, carry.counter, self.rows)
        drawn = draw(self.ladder_weights[self.modes], uniform)
        ladder_index = torch.where(reset & ladder, drawn, carry.ladder_index)
        self.ladder_probability.copy_(self.probabilities[ladder_index])
        ladder_action, _, _ = self.ladder_agent.step(None, obs, reset)
        action = torch.where(ladder[:, None, None], ladder_action.packed(), network_action)
        column = torch.where(
            ladder,
            slots + 1 + ladder_index,
            torch.where(incoming, torch.full_like(self.slot_of_row, slots), self.slot_of_row),
        )
        carry = OpponentCarry(
            model=model_carry,
            incoming=incoming,
            ladder_index=ladder_index,
            column=column,
            counter=carry.counter + 1,
        )
        return Action.unpack(action), carry, Outputs()

    def tally(self, carry: OpponentCarry, done: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        """Each ended episode's result from the learner's side (`result` (R,) int32: +1 win,
        0 tie, -1 loss, read only where `done` (R,)) counted by (column, mode, outcome):
        flat (columns * modes * 3,) int64."""
        outcome = (1 - result).to(torch.int64)  # win 0, tie 1, loss 2
        flat = (carry.column * self.mode_count + self.modes) * len(OUTCOMES) + outcome
        counts = torch.zeros(
            self.columns * self.mode_count * len(OUTCOMES), dtype=torch.int64, device=done.device
        )
        return counts.index_add(0, flat, done.to(torch.int64))

    # ---- the host's side, between segments

    @torch.no_grad()
    def load(self, set_index: int, state: dict[str, torch.Tensor]) -> None:
        """Copy a network's state (its model's state dict) into set `set_index`, in place."""
        for name, value in state.items():
            self.parameters[f"model.{name}"][set_index].copy_(value)

    def fill(self, state: dict[str, torch.Tensor]) -> None:
        """Every set holds `state`, as a network: the slots before a league draws them."""
        for index in range(self.slots + 1):
            self.load(index, state)
        self.kind.fill_(NETWORK)

    def set_kind(self, set_index: int, kind: int) -> None:
        """What set `set_index` holds: NETWORK or LADDER_KIND."""
        self.kind[set_index] = kind

    def begin_drain(self, slot: int) -> None:
        """Start draining `slot` onto the incoming set (loaded with `load` and `set_kind`
        first)."""
        self.draining.fill_(slot)

    @property
    def draining_slot(self) -> int | None:
        """The slot being drained, or None (a host read)."""
        slot = int(self.draining)
        return None if slot < 0 else slot

    def drained(self, carry: OpponentCarry) -> bool:
        """Whether every row of the draining slot has switched (a host read)."""
        slot = self.draining_slot
        return slot is not None and bool(carry.incoming[self.slot_of_row == slot].all())

    @torch.no_grad()
    def finish_drain(self, carry: OpponentCarry) -> None:
        """The incoming set becomes the drained slot's own and the drain ends."""
        slot = self.draining_slot
        for value in self.parameters.values():
            value[slot].copy_(value[self.slots])
        self.kind[slot] = self.kind[self.slots]
        carry.column.copy_(torch.where(carry.column == self.slots, slot, carry.column))
        carry.incoming.zero_()
        self.draining.fill_(-1)

    @torch.no_grad()
    def draw_ladder(self, carry: OpponentCarry, slot: int, uniform: torch.Tensor) -> None:
        """A ladder entry for each row of `slot` (now holding the ladder), from its mode's
        weights and `uniform` (R,): rows that play the ladder before their first reset."""
        rows = self.slot_of_row == slot
        drawn = draw(self.ladder_weights[self.modes], uniform.to(self.modes.device))
        carry.ladder_index.copy_(torch.where(rows, drawn, carry.ladder_index))
        column = self.slots + 1 + carry.ladder_index
        carry.column.copy_(torch.where(rows, column, carry.column))

    @torch.no_grad()
    def set_ladder_weights(self, weights: torch.Tensor) -> None:
        """Each mode's sampling weights over the ladder's entries, (modes, ladder)."""
        self.ladder_weights.copy_(weights)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """The sets, what each holds, the draining slot and the ladder weights (a resume
        restores them with `load_state_dict`)."""
        state = {f"parameters.{name}": value for name, value in self.parameters.items()}
        state.update(kind=self.kind, draining=self.draining, ladder_weights=self.ladder_weights)
        return state

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Copy a `state_dict` back in place."""
        for name, value in self.state_dict().items():
            value.copy_(state[name])
