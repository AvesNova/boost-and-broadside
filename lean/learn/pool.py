"""The league: its members, the network slots that play them, PFSP draws within row
capacities, decaying win-rate tallies, v2's snapshot schedule and retention.

Members (fixed indices, `League.kinds`): the scripted agent (member 0), the EMA and the
uniform average of the live weights (lean.learn.weights), and up to `snapshot_members`
snapshots. A member is present (drawable) when it exists: the scripted agent always; the
snapshots from their step on; the EMA from the first snapshot (before it every league game
is against the scripted agent, decided 8.1 (5)); the uniform average from the first
snapshot once it holds an update (its warm-up has ended). The cheese agents (step C) go
after the snapshots as more members of kind "scripted": no slot (member_slot -1), a
capacity and tallies like any member; their rows will need their own scripted call (today
every scripted row takes the one scripted agent's action).

Draws: each league environment whose game ended draws its next member from the
probabilities (1 - floor) * (1 - p)^2 / sum + floor / present (AlphaStar's `squared`, floor
1/8), p the live policy's win rate against the member, over the members present. Every
member has a row capacity, ceil(capacity_factor * L / present) of the L league
environments (1.25 times an even share). The capacity rule (`League.draw`): each ended
game draws independently from the probabilities over the members with a free row; on a
member, the draws past its free rows (the later environments) are drawn again the same way
over the members still free after the accepted ones; what is still left takes the free rows
in member order (the scripted agent first). Present members' capacities add up to at least
L, so a row is always found; when no member is full, the draws are plain PFSP draws.

Win rates are kept as in AlphaStar's pseudocode (`Payoff`): per member, decaying tallies
of games, wins and draws; each game first decays that member's tallies by `decay` (0.99)
and then adds itself; p = (wins + draws / 2) / games, and an unplayed member reads 0.5.
A game whose member left the league (member RETIRED) is not tallied.

Network slots: each network member (the averages, the snapshots) is played by a slot of
weights; there are 2 + snapshot_members + retiring_slots of them. A snapshot that
retention drops gives up its member index at once, but its slot keeps its weights and plays
on the games it is in (`retiring`) until they end; then the slot is free. A slot is freed,
and the next segment's network table built (`prepare`), between segments, with one host
read of the rows per slot. The averages' slots take the averages' new weights after every
update: an average is one moving player, its running games continue with the new weights
(as the live policy's do).

Snapshot schedule (`SnapshotSchedule`, v2's src/bnb/runtime/snapshots.py): v2's
LEAGUE_SNAPSHOTS, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130 and 160M environment
steps, then every 30M (the last gap). Retention (`retained`, v2's src/bnb/league/pool.py):
the newest `recent_snapshots` (8) snapshots, plus of the older ones the snapshot nearest
each age 2^k snapshots back for k >= `snapshot_history_from` (3); then at most
`snapshot_members` (12), the oldest dropped past it. Run on v2's schedule the rule keeps at
most 11 snapshots to 500M steps (25 taken) and 12 to about 1.5B; it drops at most 2
snapshots at one addition.
"""

import copy
import dataclasses
import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from lean.contract import random_keywords
from lean.learn.config import LearnConfig

SCRIPTED = 0
EMA = 1
UNIFORM = 2
FIRST_SNAPSHOT = 3
RETIRED = -1  # the member of a league game whose member has left the league (not tallied)
SCRIPTED_KIND, EMA_KIND, UNIFORM_KIND, SNAPSHOT_KIND = "scripted", "ema", "uniform", "snapshot"


@dataclasses.dataclass(frozen=True)
class SnapshotSchedule:
    """The environment steps at which a snapshot is due: each of `steps` (ascending,
    positive), then every `every` steps after the last (None: the last gap between `steps`).
    v2's SnapshotSchedule."""

    steps: tuple[int, ...]
    every: int | None = None

    def __post_init__(self) -> None:
        if not self.steps or any(step < 1 for step in self.steps) or any(
            later <= earlier for earlier, later in zip(self.steps, self.steps[1:], strict=False)
        ):
            raise ValueError(f"snapshot steps {list(self.steps)} must be positive and ascending")

    @property
    def period(self) -> int:
        if self.every is not None:
            return self.every
        return self.steps[-1] if len(self.steps) == 1 else self.steps[-1] - self.steps[-2]

    def count(self, environment_steps: int) -> int:
        """The snapshots due at or before `environment_steps`."""
        listed = sum(step <= environment_steps for step in self.steps)
        return listed + max(0, environment_steps - self.steps[-1]) // self.period

    def due(self, before: int, after: int) -> bool:
        """Whether a snapshot falls in (before, after]: the update that took the run from
        `before` to `after` environment steps keeps one (at most one per update)."""
        return self.count(after) > self.count(before)

    def points(self, until: int) -> list[int]:
        """Every snapshot step at or before `until`, in order."""
        tail = range(self.steps[-1] + self.period, until + 1, self.period)
        return [step for step in self.steps if step <= until] + list(tail)


def retained(ordinals, recent: int, history_from: int, cap: int) -> set[int]:
    """The snapshots retention keeps, by ordinal (v2's bnb.league.pool.retained, then the
    cap): the newest `recent`, plus of the older ones the snapshot nearest each age 2^k
    snapshots back for k >= history_from (an equal distance keeps the older); ages counted
    from the newest. Past `cap` the oldest go."""
    ordinals = set(ordinals)
    if not ordinals:
        return set()
    newest = max(ordinals)
    keep = {ordinal for ordinal in ordinals if newest - ordinal < recent}
    older = sorted(newest - ordinal for ordinal in ordinals if newest - ordinal >= recent)
    if older:
        power = history_from
        while True:
            target = 1 << power
            nearest = min(older, key=lambda age: (abs(age - target), -age))
            keep.add(newest - nearest)
            if target >= older[-1]:
                break
            power += 1
    return set(sorted(keep)[-cap:])


class Stepper(nn.Module):
    """A policy whose forward is its `step`, so that torch.func.functional_call can run
    the step with a slot's weights."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs, carry, position_offset):
        return self.policy.step(obs, carry, position_offset)


def stepper_weights(policy, parameters: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
    """A slot's weights: the policy's parameters (or `parameters`, by the policy's own names)
    and buffers, named as Stepper's, detached copies."""
    weights = {}
    for name, tensor in policy.named_parameters():
        source = tensor if parameters is None else parameters[name].to(tensor.dtype)
        weights[f"policy.{name}"] = source.detach().clone()
    for name, tensor in policy.named_buffers():
        weights[f"policy.{name}"] = tensor.detach().clone()
    return weights


@dataclasses.dataclass
class NetworkTable:
    """One segment's network table (League.prepare): the K slots holding weights, each as
    (slot, parameters, buffers) with `slot` a 0-d long tensor (a tensor, so that one compiled
    graph serves every slot), and `positions` (W,) arange of the rows a slot runs on (W's
    shape carries it into the compiled graph as a symbolic size: no recompile when it
    changes)."""

    slots: list[tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]]
    positions: Tensor


class League:
    """The opponents of league games (see the module's docstring).

    Device tensors read in the tick (no host reads there): `present` (members,) bool,
    `games`, `wins`, `draws` (members,) the tallies, `member_slot` (members,) long the slot
    playing each member (-1: none, the scripted agent), `capacity` () long. Host-side
    bookkeeping: `kinds`, `slot_member` (the member each slot plays, None when retiring or
    free), `slot_weights`, `retiring`, the snapshots' ordinals and steps.
    """

    def __init__(self, config: LearnConfig, device=None):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.floor = config.league_floor
        self.decay = config.league_decay
        self.capacity_factor = config.league_capacity_factor
        self.schedule = SnapshotSchedule(config.league_snapshot_steps, config.league_snapshot_period)
        self.recent, self.history_from = config.recent_snapshots, config.snapshot_history_from
        self.snapshot_members = config.snapshot_members
        self.kinds = [SCRIPTED_KIND, EMA_KIND, UNIFORM_KIND] + [SNAPSHOT_KIND] * config.snapshot_members
        self.members = len(self.kinds)
        self.slot_count = 2 + config.snapshot_members + config.retiring_slots
        self.present = torch.zeros(self.members, dtype=torch.bool, device=self.device)
        self.present[SCRIPTED] = True
        self.games = torch.zeros(self.members, device=self.device)
        self.wins = torch.zeros(self.members, device=self.device)
        self.draws = torch.zeros(self.members, device=self.device)
        self.member_slot = torch.full((self.members,), -1, dtype=torch.long, device=self.device)
        self.capacity = torch.zeros((), dtype=torch.long, device=self.device)
        self.slot_member: list[int | None] = [None] * self.slot_count
        self.slot_weights: list[dict[str, Tensor] | None] = [None] * self.slot_count
        self.retiring: set[int] = set()
        self.ordinal: dict[int, int] = {}  # snapshot member -> its ordinal (0 the first snapshot)
        self.snapshot_step: dict[int, int] = {}  # snapshot member -> its environment steps
        self.taken = 0  # snapshots taken so far
        self.abandoned = 0  # games re-dealt because no slot was free (should stay 0)
        self.template: nn.Module | None = None
        self.table: NetworkTable | None = None

    # ---- members

    @property
    def snapshots(self) -> int:
        """Snapshot members present."""
        return len(self.ordinal)

    @property
    def networks(self) -> int:
        """Slots holding weights (members and retiring): the next table's K at most."""
        return sum(weights is not None for weights in self.slot_weights)

    def present_count(self) -> int:
        return 1 + self.snapshots + (self.slot_member[0] is not None) + (self.slot_member[1] is not None)

    def win_rates(self) -> Tensor:
        """(members,) the live policy's win rate against each member, draws as half,
        0.5 where unplayed."""
        played = self.games > 0
        rate = (self.wins + 0.5 * self.draws) / self.games.clamp(min=1e-12)
        return torch.where(played, rate, torch.full_like(rate, 0.5))

    def scripted_win_rate(self) -> float:
        """Win rate against the scripted agent for the BC gate; 0 until a game is played
        (so the BC term starts at full weight), not the sampler's 0.5 prior."""
        if float(self.games[SCRIPTED]) <= 0.0:
            return 0.0
        return float(self.win_rates()[SCRIPTED])

    def probabilities(self) -> Tensor:
        """(members,) PFSP probabilities: (1 - floor) by (1 - p)^2, floor uniform, both over
        the members present. Uniform in place of the first part when its weights sum below
        1e-10 (AlphaStar's rule)."""
        return pfsp(self.win_rates(), self.present, self.floor)

    # ---- draws and tallies (in the tick: no host reads)

    def draw(self, member: Tensor, done: Tensor, generator) -> tuple[Tensor, Tensor]:
        """New members for the league environments in `done` (L,) bool, the rest keeping
        `member` (L,) long; returns (member, slot), slot the network slot playing it (-1 for
        the scripted agent). The capacity rule is in the module's docstring."""
        drawn = capacity_draws(
            self.probabilities(), self.present, self.capacity, member, done, generator, self.members
        )
        member = torch.where(done, drawn, member)
        return member, self.member_slot.index_select(0, member.clamp(min=0))

    def record(self, member: Tensor, finished: Tensor, won: Tensor, drawn: Tensor) -> None:
        """Add the league games that ended: member, finished, won, drawn are (L,) over the
        league environments. Each game decays its member's tallies by `decay` before it
        counts. Games of one member ending on the same tick are taken in every order
        equally: n games give games * d^n + (1 - d^n) / (1 - d), and each win counts
        (1 - d^n) / (n (1 - d)), which is exact for n = 1. RETIRED members are skipped."""
        counted = (finished & (member >= 0)).to(self.games.dtype)
        index = torch.where(member >= 0, member, self.members)  # RETIRED to a spare entry
        spare = torch.zeros(self.members + 1, device=self.games.device, dtype=self.games.dtype)
        count = spare.index_add(0, index, counted)[:-1]
        wins = spare.index_add(0, index, counted * won.to(counted.dtype))[:-1]
        draws = spare.index_add(0, index, counted * drawn.to(counted.dtype))[:-1]
        factor = self.decay**count
        added = (1.0 - factor) / (1.0 - self.decay)
        per_game = added / count.clamp(min=1.0)
        self.games = self.games * factor + added
        self.wins = self.wins * factor + wins * per_game
        self.draws = self.draws * factor + draws * per_game

    # ---- between updates and segments (host side)

    def _start_member(self, member: int) -> None:
        self.present[member] = True
        self.games[member] = 0.0
        self.wins[member] = 0.0
        self.draws[member] = 0.0

    def _free_slot(self) -> int | None:
        for slot in range(2, self.slot_count):
            if self.slot_weights[slot] is None:
                return slot
        return None

    def _place(self, member: int, slot: int, weights: dict[str, Tensor]) -> None:
        self.slot_member[slot] = member
        self.slot_weights[slot] = weights
        self.member_slot[member] = slot

    def slot_rows(self, slot: Tensor) -> list[int]:
        """Rows (league environments) per slot, from the carry's `slot` (L,): one host read."""
        valid = slot >= 0
        counts = torch.zeros(self.slot_count + 1, dtype=torch.long, device=slot.device)
        counts.index_add_(0, torch.where(valid, slot, self.slot_count), valid.long())
        return counts[:-1].tolist()

    def free_retired(self, slot: Tensor) -> list[int]:
        """Free the retiring slots no game plays any more; returns the rows per slot."""
        rows = self.slot_rows(slot)
        for retiring in sorted(self.retiring):
            if rows[retiring] == 0:
                self.retiring.discard(retiring)
                self.slot_weights[retiring] = None
        return rows

    def after_update(self, policy, averages, steps_before: int, steps_after: int, member: Tensor, slot: Tensor):
        """After an RL update that took the run from `steps_before` to `steps_after`
        environment steps: a snapshot of the live weights when the schedule has one due, with
        retention; the averages' slots take their new weights (the EMA from the first
        snapshot, the uniform average once it holds an update). `member`, `slot` the carry's
        (L,). Returns (member, abandon): member with the games of dropped members RETIRED,
        and the slot (or None) whose games the caller must re-deal (lean.learn.rollout.
        abandon) because no slot was free for the new snapshot (counted in `abandoned`)."""
        if self.template is None:
            self.template = copy.deepcopy(Stepper(policy)).to("meta").eval()
        abandon = None
        if self.schedule.due(steps_before, steps_after):
            rows = self.free_retired(slot)
            ordinals = [*self.ordinal.values(), self.taken]  # the new snapshot's ordinal last
            kept = retained(ordinals, self.recent, self.history_from, self.snapshot_members)
            for dropped in [m for m, ordinal in self.ordinal.items() if ordinal not in kept]:
                old = int(self.member_slot[dropped])
                self.present[dropped] = False
                self.member_slot[dropped] = -1
                del self.ordinal[dropped], self.snapshot_step[dropped]
                self.slot_member[old] = None
                if rows[old] > 0:
                    self.retiring.add(old)
                else:
                    self.slot_weights[old] = None
                member = torch.where(member == dropped, torch.full_like(member, RETIRED), member)
            new = self._free_slot()
            if new is None:  # every slot busy: the retiring slot with the fewest games goes
                new = min(self.retiring, key=lambda retiring: rows[retiring])
                self.retiring.discard(new)
                self.abandoned += rows[new]
                abandon = new
            index = next(m for m in range(FIRST_SNAPSHOT, self.members) if m not in self.ordinal)
            self._place(index, new, stepper_weights(policy))
            self.ordinal[index] = self.taken
            self.snapshot_step[index] = steps_after
            self.taken += 1
            self._start_member(index)
        if self.taken > 0:
            for which, index, ready in (("ema", EMA, True), ("uniform", UNIFORM, averages.count > 0)):
                if not ready:
                    continue
                weights = stepper_weights(policy, getattr(averages, which))
                slot_index = index - EMA  # slots 0 and 1 belong to the averages
                if self.slot_weights[slot_index] is None:
                    self._place(index, slot_index, weights)
                    self._start_member(index)
                else:
                    for name, tensor in weights.items():
                        self.slot_weights[slot_index][name].copy_(tensor)
        return member, abandon

    def prepare(self, league_envs: int, slot: Tensor) -> None:
        """Before a segment: free the retiring slots without games, set the capacity
        ceil(capacity_factor * L / present) and build the network table (NetworkTable) of
        the slots holding weights, W = the most rows a slot can hold this segment: the
        capacity, or more where a slot already holds more (members added since, or a
        retiring slot), since new draws never take a member past its capacity. One host
        read (the rows per slot)."""
        rows = self.free_retired(slot)
        capacity = math.ceil(self.capacity_factor * league_envs / self.present_count()) if league_envs else 0
        self.capacity.fill_(capacity)
        active = [index for index, weights in enumerate(self.slot_weights) if weights is not None]
        if not active:
            self.table = None
            return
        width = max([capacity] + [rows[index] for index in active])
        parameter_names = {name for name, _ in self.template.named_parameters()}
        slots = []
        for index in active:
            weights = self.slot_weights[index]
            slots.append((
                torch.tensor(index, device=self.device),
                {name: value for name, value in weights.items() if name in parameter_names},
                {name: value for name, value in weights.items() if name not in parameter_names},
            ))
        self.table = NetworkTable(slots, torch.arange(width, device=self.device))

    # ---- checkpoints

    def state(self) -> dict:
        names = ("present", "games", "wins", "draws", "member_slot", "capacity")
        return {
            **{name: getattr(self, name) for name in names},
            "slot_member": list(self.slot_member), "slot_weights": list(self.slot_weights),
            "retiring": sorted(self.retiring), "ordinal": dict(self.ordinal), "snapshot_step": dict(self.snapshot_step),
            "taken": self.taken, "abandoned": self.abandoned,
        }

    def load(self, state: dict, policy) -> None:
        for name in ("present", "games", "wins", "draws", "member_slot", "capacity"):
            setattr(self, name, state[name].to(self.device).clone())
        self.slot_member = list(state["slot_member"])
        self.slot_weights = [
            None if weights is None else {name: value.to(self.device) for name, value in weights.items()}
            for weights in state["slot_weights"]
        ]
        self.retiring = set(state["retiring"])
        self.ordinal, self.snapshot_step = dict(state["ordinal"]), dict(state["snapshot_step"])
        self.taken, self.abandoned = state["taken"], state["abandoned"]
        self.template = copy.deepcopy(Stepper(policy)).to("meta").eval()


def pfsp(win_rate: Tensor, present: Tensor, floor: float) -> Tensor:
    """(members,) (1 - floor) * (1 - p)^2 / sum + floor / present over the members present;
    uniform in place of the first part when its weights sum below 1e-10."""
    present = present.to(win_rate.dtype)
    uniform = present / present.sum()
    weight = (1.0 - win_rate) ** 2 * present
    total = weight.sum()
    prioritised = torch.where(total > 1e-10, weight / total.clamp(min=1e-10), uniform)
    return (1.0 - floor) * prioritised + floor * uniform


def sample_categorical(weight: Tensor, count: int, generator) -> Tensor:
    """`count` independent draws (long) from the categorical with unnormalised `weight`
    (K,) by the inverse CDF (no host read; a zero weight is never drawn)."""
    cumulative = weight.cumsum(0)
    uniform = torch.rand(count, dtype=weight.dtype, **random_keywords(generator, weight.device))
    point = uniform * cumulative[-1]
    return torch.searchsorted(cumulative, point, right=True).clamp(max=weight.shape[0] - 1)


def accept(draw: Tensor, wanted: Tensor, free: Tensor, members: int) -> tuple[Tensor, Tensor]:
    """Of the draws `draw` (L,) where `wanted` (L,) bool, those within their member's `free`
    rows (members,), earlier environments first: (accepted (L,) bool, rows taken (members,))."""
    one_hot = functional.one_hot(draw, members) * wanted[:, None].long()
    rank = one_hot.cumsum(0).gather(1, draw[:, None]).squeeze(1)  # 1 for a member's first draw
    accepted = wanted & (rank <= free.index_select(0, draw))
    taken = (functional.one_hot(draw, members) * accepted[:, None].long()).sum(0)
    return accepted, taken


def capacity_draws(probability: Tensor, present: Tensor, capacity: Tensor, member: Tensor, done: Tensor, generator,
                   members: int) -> Tensor:
    """(L,) a member for every league environment (meaningful where `done`) by the capacity
    rule (module docstring): PFSP `probability` over the members with a free row, the
    overflow drawn once more over those still free, then the free rows in member order. The
    rows held are the environments not `done` (RETIRED ones belong to no member)."""
    holding = ~done & (member >= 0)
    held = (functional.one_hot(member.clamp(min=0), members) * holding[:, None].long()).sum(0)
    free = ((capacity - held).clamp(min=0) * present.long())
    count = member.shape[0]
    chosen = torch.zeros_like(member)
    wanted = done
    for _ in range(2):
        draw = sample_categorical(probability * (free > 0).to(probability.dtype), count, generator)
        accepted, taken = accept(draw, wanted, free, members)
        chosen = torch.where(accepted, draw, chosen)
        free = free - taken
        wanted = wanted & ~accepted
    # What is left takes the free rows in order: the j-th left over (from 1) the member
    # holding free row j when the members' free rows are laid end to end.
    order = wanted.long().cumsum(0)
    fill = torch.searchsorted(free.cumsum(0), order, right=False).clamp(max=members - 1)
    return torch.where(wanted, fill, chosen)
