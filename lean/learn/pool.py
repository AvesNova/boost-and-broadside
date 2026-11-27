"""The league, in two levels: a pool of members with their tallies, the slots that play them
(level 1 loads members into slots between updates), and each game's draw of a slot (level 2).
Also v2's snapshot schedule and retention.

The pool (host memory). Members have fixed indices (`League.kinds`): the yardstick scripted
agent (member 0), the EMA and the uniform average of the live weights (lean.learn.weights),
up to `snapshot_members` snapshots, then the scripted personas (LearnConfig.league_personas:
named points of the scripted agent's parameters). Scripted members and networks are sampled
alike. A member is present (sampled) when it exists: the yardstick always; the personas and
the EMA from RL's start (`open`); the uniform average once it holds an update; each snapshot
from its step until retention drops it. A snapshot's weights stay in host memory (and on
disk once, `snapshot_directory`); the averages' come from lean.learn.weights.

Win rates are kept as in AlphaStar's pseudocode (`Payoff`): per member, decaying tallies of
games, wins and draws; each game first decays its member's tallies by `decay` (0.99) and then
adds itself; p = (wins + draws / 2) / games, an unplayed member reading 0.5. The target
shares are PFSP: (1 - floor) * (1 - p)^2 / sum + floor / present (AlphaStar's `squared`,
floor 1/8), over the members present. A game whose member left the pool (RETIRED) is not
tallied.

Slots. There are K + R of them (LearnConfig.league_slots, league_spare_slots) and an overflow
slot. A slot is free, loaded (takes new games) or draining (takes none; its running games
finish there). It holds a member: a network slot a copy of its weights on the device (weights
are graph inputs, so a load is a copy and never a compile), a scripted slot the index of its
parameter row (`persona_table`; row 0 the yardstick). A member may hold several slots.

Level 1 (`allocate`, between updates, host side): member m's target is q_m K slots (q the
PFSP shares). Its debt d_m carries the rounding: v_m = d_m + q_m K - (slots it holds loaded);
while a slot is free and some present member is short by more than one slot against another
(v_u - v_o > 1), one slot of the most served member o (its loaded slot with the fewest rows)
is unloaded and the most short member u is loaded into a free slot; then d = v. So each
update changes at most as many slots as are free, the loaded slots always number K, and over
many updates each member's slot share follows its target share (the debt stays bounded).
An unloaded slot drains (a game lasts at most about 4 updates) and frees once its last game
has ended (`free_drained`, at every segment's start and after every update). A member that
leaves the pool (retention) unloads every slot it holds; its running games are RETIRED.

Level 2 (`draw`, at every game's end, in the tick: no host reads): the game draws uniformly
over the loaded slots with a free row (capacity C = ceil(capacity_factor * L / K) per slot,
fixed for the run): one draw, accepted within the free rows (earlier environments first), a
second draw for the rest over the slots still free, then the free rows in slot order. So a
member's share of games is about its share of the loaded slots. Only if no loaded slot has a
free row does the game go to the overflow slot, a yardstick slot with no capacity
(`overflow_games` counts them; it should stay 0).

The tick: each network slot holding weights (loaded or draining) runs once on W = C rows (one
routing graph for every slot; empty slots skipped); every scripted slot's rows take the one
scripted call over all ships with per-row parameters (lean.learn.rollout.scripted_rows); the
merge takes K + R entries (a skipped slot's a zero filler no row reads). The shapes are fixed
for the run, and the EMA is present from RL's first tick, so the routing compiles there once.

Snapshot schedule (`SnapshotSchedule`, v2's src/bnb/runtime/snapshots.py): v2's
LEAGUE_SNAPSHOTS, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130 and 160M environment
steps, then every 30M (the last gap). Retention (`retained`, v2's src/bnb/league/pool.py):
the newest `recent_snapshots` (8) snapshots, plus of the older ones the snapshot nearest
each age 2^k snapshots back for k >= `snapshot_history_from` (3); then at most
`snapshot_members` (12), the oldest dropped past it.
"""

import dataclasses
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional

from lean.contract import random_keywords
from lean.learn.config import LearnConfig
from lean.model.agent import Stepper, split_weights, stepper_weights, template
from lean.scripted import ScriptedConfig
from lean.scripted.parameters import PARAMETERS, SAMPLED

SCRIPTED = 0
EMA = 1
UNIFORM = 2
FIRST_SNAPSHOT = 3
RETIRED = -1  # the member of a league game whose member has left the league (not tallied)
SCRIPTED_KIND, EMA_KIND, UNIFORM_KIND, SNAPSHOT_KIND, PERSONA_KIND = (
    "scripted",
    "ema",
    "uniform",
    "snapshot",
    "persona",
)
FREE, LOADED, DRAINING = "free", "loaded", "draining"


@dataclasses.dataclass(frozen=True)
class SnapshotSchedule:
    """The environment steps at which a snapshot is due: each of `steps` (ascending,
    positive), then every `every` steps after the last (None: the last gap between `steps`).
    v2's SnapshotSchedule."""

    steps: tuple[int, ...]
    every: int | None = None

    def __post_init__(self) -> None:
        if (
            not self.steps
            or any(step < 1 for step in self.steps)
            or any(
                later <= earlier for earlier, later in zip(self.steps, self.steps[1:], strict=False)
            )
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


def persona_rows(personas) -> tuple[str, ...]:
    """The scripted parameter rows the league's per-row table holds: the sampled rows and any
    other row a persona names, in the table's order (lean.scripted.PARAMETERS)."""
    named = {name for _, rows in personas for name, _ in rows}
    unknown = named - set(PARAMETERS)
    if unknown:
        raise ValueError(f"league personas name rows not in the table: {sorted(unknown)}")
    return tuple(name for name in PARAMETERS if name in SAMPLED or name in named)


@dataclasses.dataclass
class NetworkTable:
    """One segment's network table (League.prepare): the network slots holding weights, each
    as (index, slot, parameters, buffers) with `index` the slot's int and `slot` the same as a
    0-d long tensor (a tensor, so that one compiled graph serves every slot); `positions` (W,)
    arange of the rows a slot runs on (W = C, fixed for the run); `count` K + R, the merge's
    entries. `fillers` (built by the segment's first tick, lean.learn.rollout.network_actions):
    per slot not in `slots`, a zero action and carry shaped as a slot's output, so the merge
    always takes K + R entries of the same shapes (one graph) though only the listed slots
    run; no row reads a filler (the merge reads network slots only)."""

    slots: list[tuple[int, Tensor, dict[str, Tensor], dict[str, Tensor]]]
    positions: Tensor
    count: int
    fillers: dict[int, tuple[Tensor, Tensor]] | None = None


class League:
    """The opponents of league games (see the module's docstring).

    Device tensors read in the tick (no host reads there): `present` (members,) bool,
    `games`, `wins`, `draws` (members,) the tallies, `dealt` (members,) the games dealt to
    each member (not decayed: the realised shares), `overflow_games` () the games dealt to
    the overflow slot; per slot (K + R + 1, the overflow last) `slot_member` long,
    `slot_loaded` bool (takes new games), `slot_network` bool (a network plays it),
    `slot_persona` long (its scripted parameter row; 0 for networks); `capacity` () long (C,
    set by the first `prepare`); `persona_table` (1 + personas, rows) float32 (row 0 the
    yardstick; in the BC stage only row 0 plays).

    Host side: `kinds`, `slot_state`, `slot_holder` (the member a slot holds, None when free),
    `slot_source` (what its weights or parameters are: ("ema",), ("uniform",), ("snapshot",
    steps), ("scripted", row)), `slot_weights` (device buffers, allocated once per slot and
    reused), `debt` (per member), the snapshots' ordinals and steps, `stored` (snapshot steps
    -> host weights).
    """

    def __init__(self, config: LearnConfig, device=None):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.floor = config.league_floor
        self.decay = config.league_decay
        self.capacity_factor = config.league_capacity_factor
        self.schedule = SnapshotSchedule(
            config.league_snapshot_steps, config.league_snapshot_period
        )
        self.recent, self.history_from = config.recent_snapshots, config.snapshot_history_from
        self.snapshot_members = config.snapshot_members
        self.personas = tuple(config.league_personas)
        self.first_persona = FIRST_SNAPSHOT + config.snapshot_members
        self.kinds = (
            [SCRIPTED_KIND, EMA_KIND, UNIFORM_KIND]
            + [SNAPSHOT_KIND] * config.snapshot_members
            + [PERSONA_KIND] * len(self.personas)
        )
        self.labels = (
            ["yardstick", "ema", "uniform"]
            + [f"snapshot{index}" for index in range(config.snapshot_members)]
            + [label for label, _ in self.personas]
        )
        self.members = len(self.kinds)
        self.loaded_slots = config.league_slots
        self.slot_count = config.league_slots + config.league_spare_slots
        self.overflow = self.slot_count  # the overflow slot's index
        if self.loaded_slots < 1:
            raise ValueError("the league needs at least one slot")
        device = self.device
        self.present = torch.zeros(self.members, dtype=torch.bool, device=device)
        self.present[SCRIPTED] = True
        self.games = torch.zeros(self.members, device=device)
        self.wins = torch.zeros(self.members, device=device)
        self.draws = torch.zeros(self.members, device=device)
        self.dealt = torch.zeros(self.members, device=device)
        self.overflow_games = torch.zeros((), device=device)
        entries = self.slot_count + 1
        self.slot_member = torch.full((entries,), SCRIPTED, dtype=torch.long, device=device)
        self.slot_loaded = torch.zeros(entries, dtype=torch.bool, device=device)
        self.slot_network = torch.zeros(entries, dtype=torch.bool, device=device)
        self.slot_persona = torch.zeros(entries, dtype=torch.long, device=device)
        self.capacity = torch.zeros((), dtype=torch.long, device=device)
        # The scripted members' parameter rows, row 0 the yardstick: built for every stage (the
        # BC stage reads row 0 only), so both stages run one scripted graph.
        self.parameter_rows = persona_rows(self.personas)
        self.persona_table = self._persona_table(ScriptedConfig())
        self.width = 0  # W = C, the slots' row capacity (prepare)
        self.slot_state = [FREE] * self.slot_count
        self.slot_holder: list[int | None] = [None] * self.slot_count
        self.slot_source: list[tuple | None] = [None] * self.slot_count
        self.slot_weights: list[dict[str, Tensor] | None] = [None] * self.slot_count
        self.debt = [0.0] * self.members
        self.snapshot_directory: Path | None = None  # where each snapshot is written once
        self.ordinal: dict[int, int] = {}  # snapshot member -> its ordinal (0 the first)
        self.snapshot_step: dict[int, int] = {}  # snapshot member -> its environment steps
        self.stored: dict[int, dict[str, Tensor]] = {}  # snapshot steps -> host weights
        self.taken = 0  # snapshots taken so far
        self.loads = 0  # slot loads so far (level 1)
        self.opened = False
        self.template: Stepper | None = None  # the network the slots' weights fill
        self.table: NetworkTable | None = None
        self.allocate([0] * (self.slot_count + 1))  # every loaded slot the yardstick

    # ---- members

    @property
    def snapshots(self) -> int:
        """Snapshot members present."""
        return len(self.ordinal)

    def present_count(self) -> int:
        return int(self.present.sum())

    def win_rates(self) -> Tensor:
        """(members,) the live policy's win rate against each member, draws as half,
        0.5 where unplayed."""
        played = self.games > 0
        rate = (self.wins + 0.5 * self.draws) / self.games.clamp(min=1e-12)
        return torch.where(played, rate, torch.full_like(rate, 0.5))

    def scripted_win_rate(self) -> float:
        """Win rate against the yardstick for the BC gate; 0 until a game is played (so the BC
        term starts at full weight), not the sampler's 0.5 prior."""
        if float(self.games[SCRIPTED]) <= 0.0:
            return 0.0
        return float(self.win_rates()[SCRIPTED])

    def probabilities(self) -> Tensor:
        """(members,) the PFSP target shares: (1 - floor) by (1 - p)^2, floor uniform, both
        over the members present. Uniform in place of the first part when its weights sum
        below 1e-10 (AlphaStar's rule)."""
        return pfsp(self.win_rates(), self.present, self.floor)

    def held(self) -> list[int]:
        """Loaded slots per member (host)."""
        counts = [0] * self.members
        for slot, state in enumerate(self.slot_state):
            if state == LOADED:
                counts[self.slot_holder[slot]] += 1
        return counts

    # ---- level 2 and the tallies (in the tick: no host reads)

    def draw(self, member: Tensor, slot: Tensor, done: Tensor, generator) -> tuple[Tensor, Tensor]:
        """New slots, and their members, for the league environments in `done` (L,) bool; the
        rest keep `member` and `slot` (L,) long (a game never changes its slot: a draining
        slot plays its games to the end). Level 2 (module docstring)."""
        drawn = slot_draws(self.slot_loaded, self.capacity, slot, done, generator, self.slot_count)
        new_member = self.slot_member.index_select(0, drawn)
        counted = done.to(self.dealt.dtype)
        self.dealt = self.dealt.index_add(0, new_member, counted)
        self.overflow_games = self.overflow_games + (counted * (drawn == self.overflow)).sum()
        return torch.where(done, new_member, member), torch.where(done, drawn, slot)

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

    # ---- slots (host side)

    def slot_rows(self, slot: Tensor) -> list[int]:
        """Rows (league environments) per slot, the overflow last, from the carry's `slot`
        (L,): one host read."""
        counts = torch.zeros(self.slot_count + 1, dtype=torch.long, device=slot.device)
        counts.index_add_(0, slot.clamp(min=0), (slot >= 0).long())
        return counts.tolist()

    def _sync(self, slot: int) -> None:
        """The device entries of one slot from its host state."""
        holder, source = self.slot_holder[slot], self.slot_source[slot]
        network = source is not None and source[0] != "scripted"
        self.slot_member[slot] = SCRIPTED if holder is None else holder
        self.slot_loaded[slot] = self.slot_state[slot] == LOADED
        self.slot_network[slot] = network
        self.slot_persona[slot] = source[1] if source is not None and not network else 0

    def free_drained(self, rows: list[int]) -> None:
        """Free the draining slots no game plays any more."""
        for slot in range(self.slot_count):
            if self.slot_state[slot] == DRAINING and rows[slot] == 0:
                self._free(slot)
        self._collect_stored()

    def _free(self, slot: int) -> None:
        self.slot_state[slot] = FREE
        self.slot_holder[slot] = None
        self.slot_source[slot] = None
        self._sync(slot)

    def _unload(self, slot: int, rows: list[int]) -> None:
        """Stop a loaded slot taking games: it drains, or frees at once without games."""
        if rows[slot] == 0:
            self._free(slot)
        else:
            self.slot_state[slot] = DRAINING
            self._sync(slot)

    def _source(self, member: int) -> tuple:
        kind = self.kinds[member]
        if kind == SCRIPTED_KIND:
            return ("scripted", 0)
        if kind == PERSONA_KIND:
            return ("scripted", 1 + member - self.first_persona)
        if kind == SNAPSHOT_KIND:
            return ("snapshot", self.snapshot_step[member])
        return (kind,)

    def _fill(self, slot: int, weights: dict[str, Tensor]) -> None:
        """Copy `weights` into the slot's device buffers (allocated on its first network)."""
        if self.slot_weights[slot] is None:
            self.slot_weights[slot] = {
                name: value.to(self.device, copy=True) for name, value in weights.items()
            }
            return
        for name, value in weights.items():
            self.slot_weights[slot][name].copy_(value)

    def _load(self, member: int, slot: int, averages, policy) -> None:
        source = self._source(member)
        if source[0] == "snapshot":
            self._fill(slot, self.stored[source[1]])
        elif source[0] != "scripted":
            self._fill(slot, stepper_weights(policy, getattr(averages, source[0])))
        self.slot_state[slot] = LOADED
        self.slot_holder[slot] = member
        self.slot_source[slot] = source
        self.loads += 1
        self._sync(slot)

    def allocate(self, rows: list[int], averages=None, policy=None) -> None:
        """Level 1 (module docstring): the loaded slots follow the PFSP targets, at most one
        load per free slot. `rows` per slot (slot_rows); `averages` and `policy` give the
        averages' weights to a slot that loads one."""
        present = self.present.tolist()
        share = self.probabilities().tolist()
        members = [member for member in range(self.members) if present[member]]
        held = self.held()
        value = {
            member: self.debt[member] + share[member] * self.loaded_slots - held[member]
            for member in members
        }
        free = [slot for slot in range(self.slot_count) if self.slot_state[slot] == FREE]
        loaded = sum(held)
        while loaded < self.loaded_slots and free:
            short = max(members, key=lambda member: (value[member], -member))
            self._load(short, free.pop(0), averages, policy)
            value[short] -= 1.0
            held[short] += 1
            loaded += 1
        while free:
            short = max(members, key=lambda member: (value[member], -member))
            served = [member for member in members if held[member] > 0]
            most = min(served, key=lambda member: (value[member], -member))
            if value[short] - value[most] <= 1.0 + 1e-9:
                break
            slots = [
                slot
                for slot in range(self.slot_count)
                if self.slot_state[slot] == LOADED and self.slot_holder[slot] == most
            ]
            unloaded = min(slots, key=lambda slot: (rows[slot], slot))
            self._unload(unloaded, rows)
            if self.slot_state[unloaded] == FREE:
                free = sorted(free + [unloaded])
            self._load(short, free.pop(0), averages, policy)
            value[short] -= 1.0
            value[most] += 1.0
            held[short] += 1
            held[most] -= 1
        self.debt = [value.get(member, 0.0) for member in range(self.members)]

    def _start_member(self, member: int) -> None:
        self.present[member] = True
        self.games[member] = 0.0
        self.wins[member] = 0.0
        self.draws[member] = 0.0
        self.debt[member] = 0.0

    def _drop(self, member: int, rows: list[int]) -> None:
        """A member leaves the pool: its loaded slots unload (their games play on, RETIRED)."""
        self.present[member] = False
        self.debt[member] = 0.0
        for slot in range(self.slot_count):
            if self.slot_holder[slot] == member and self.slot_state[slot] == LOADED:
                self._unload(slot, rows)

    def _collect_stored(self) -> None:
        """Forget the host weights of snapshots neither in the pool nor held by a slot."""
        used = set(self.snapshot_step.values()) | {
            source[1]
            for source in self.slot_source
            if source is not None and source[0] == "snapshot"
        }
        for steps in [steps for steps in self.stored if steps not in used]:
            del self.stored[steps]

    def _persona_table(self, scripted_config: ScriptedConfig) -> Tensor:
        """(1 + personas, rows) float32: the yardstick's values, then each persona's (the rows
        it does not name the yardstick's)."""
        table = [[scripted_config.values[name] for name in self.parameter_rows]]
        for _, named in self.personas:
            values = {**scripted_config.values, **dict(named)}
            table.append([values[name] for name in self.parameter_rows])
        return torch.tensor(table, dtype=torch.float32, device=self.device)

    def open(self, policy, averages, scripted_config: ScriptedConfig | None = None, slot=None):
        """RL's start: the personas and the EMA join the pool (the uniform average too once it
        holds an update) and level 1 loads the slots. `scripted_config` the yardstick (default
        ScriptedConfig(): the parameter table's values are rebuilt around it, in place). `slot`
        the carry's (L,) when games are running (default: none, so every slot can be
        reloaded)."""
        if self.opened:
            raise RuntimeError("the league is already open")
        self.template = template(policy)
        if scripted_config is not None:
            self.persona_table.copy_(self._persona_table(scripted_config))
        for index in range(len(self.personas)):
            self._start_member(self.first_persona + index)
        self._start_member(EMA)
        if averages.count > 0:
            self._start_member(UNIFORM)
        self.opened = True
        rows = [0] * (self.slot_count + 1) if slot is None else self.slot_rows(slot)
        self.free_drained(rows)
        self.allocate(rows, averages, policy)

    def after_update(
        self, policy, averages, steps_before: int, steps_after: int, member: Tensor, slot: Tensor
    ) -> Tensor:
        """After an RL update that took the run from `steps_before` to `steps_after`
        environment steps: the drained slots free; a snapshot of the live weights joins the
        pool when the schedule has one due, with retention (a dropped member's games RETIRED
        in the returned `member`); the uniform average joins once it holds an update; every
        slot holding an average takes its new weights (an average is one moving player); then
        level 1. `member`, `slot` the carry's (L,). One host read of the rows per slot and one
        of the tallies."""
        if not self.opened:
            raise RuntimeError("League.open comes before the first after_update (RL's start)")
        rows = self.slot_rows(slot)
        self.free_drained(rows)
        if self.schedule.due(steps_before, steps_after):
            ordinals = [*self.ordinal.values(), self.taken]  # the new snapshot's ordinal last
            kept = retained(ordinals, self.recent, self.history_from, self.snapshot_members)
            for dropped in [m for m, ordinal in self.ordinal.items() if ordinal not in kept]:
                self._drop(dropped, rows)
                del self.ordinal[dropped], self.snapshot_step[dropped]
                member = torch.where(member == dropped, torch.full_like(member, RETIRED), member)
            index = next(
                m for m in range(FIRST_SNAPSHOT, self.first_persona) if m not in self.ordinal
            )
            weights = {
                name: value.to("cpu", copy=True) for name, value in stepper_weights(policy).items()
            }
            self.stored[steps_after] = weights
            if self.snapshot_directory is not None:
                save_snapshot(self.snapshot_directory, steps_after, weights)
            self.ordinal[index] = self.taken
            self.snapshot_step[index] = steps_after
            self.taken += 1
            self._start_member(index)
            self._collect_stored()
        if averages.count > 0 and not bool(self.present[UNIFORM]):
            self._start_member(UNIFORM)
        for which in ("ema", "uniform"):
            slots = [index for index, source in enumerate(self.slot_source) if source == (which,)]
            if slots:
                weights = stepper_weights(policy, getattr(averages, which))
                for held in slots:
                    self._fill(held, weights)
        self.allocate(rows, averages, policy)
        return member

    def prepare(self, league_envs: int, slot: Tensor) -> None:
        """Before a segment: free the drained slots, set the capacity (C = ceil(capacity_factor
        * L / K), the same every segment of a run) and build the network table (NetworkTable)
        of the network slots holding weights on W = C rows (None while there are none: every
        league game scripted). One host read (the rows per slot)."""
        rows = self.slot_rows(slot)
        self.free_drained(rows)
        width = min(league_envs, math.ceil(self.capacity_factor * league_envs / self.loaded_slots))
        self.width = width
        self.capacity.fill_(width)
        orphans = [
            index
            for index in range(self.slot_count)
            if rows[index] and self.slot_state[index] == FREE
        ]
        if orphans:
            raise RuntimeError(f"league slots {orphans} have games but are free")
        networks = [
            index
            for index in range(self.slot_count)
            if self.slot_state[index] != FREE and self.slot_source[index][0] != "scripted"
        ]
        if not networks or league_envs == 0:
            self.table = None
            return
        over = [index for index in networks if rows[index] > width]
        if over:
            raise RuntimeError(
                f"league slots {over} hold {[rows[index] for index in over]} rows, more than "
                f"the table width {width} (level 2 never deals past it)"
            )
        slots = [
            (
                index,
                torch.tensor(index, device=self.device),
                *split_weights(self.template, self.slot_weights[index]),
            )
            for index in networks
        ]
        self.table = NetworkTable(slots, torch.arange(width, device=self.device), self.slot_count)

    def summary(self) -> dict:
        """Host numbers for the update's line: loaded slots by kind, draining slots, the
        scripted (yardstick and personas) share of the loaded slots and of the games dealt
        so far, the overflow's games and the loads so far."""
        by_kind: dict[str, int] = {}
        for member, count in enumerate(self.held()):
            by_kind[self.kinds[member]] = by_kind.get(self.kinds[member], 0) + count
        dealt = self.dealt.tolist()
        scripted = sum(
            count
            for member, count in enumerate(dealt)
            if self.kinds[member] in (SCRIPTED_KIND, PERSONA_KIND)
        )
        return {
            "loaded": by_kind,
            "draining": sum(state == DRAINING for state in self.slot_state),
            "scripted_slots": (by_kind.get(SCRIPTED_KIND, 0) + by_kind.get(PERSONA_KIND, 0))
            / self.loaded_slots,
            "scripted_dealt": scripted / max(1.0, sum(dealt)),
            "overflow_games": float(self.overflow_games),
            "loads": self.loads,
        }

    # ---- checkpoints

    TENSORS = (
        "present",
        "games",
        "wins",
        "draws",
        "dealt",
        "overflow_games",
        "slot_member",
        "slot_loaded",
        "slot_network",
        "slot_persona",
        "capacity",
    )

    def state(self) -> dict:
        """The league's checkpoint state. Slot weights are not in it (rebuilt on load: the
        averages' from the averages, a snapshot's from its host weights); the snapshots'
        host weights are, unless written to `snapshot_directory` (then None: read back from
        there)."""
        return {
            **{name: getattr(self, name) for name in self.TENSORS},
            "persona_table": self.persona_table,
            "parameter_rows": self.parameter_rows,
            "slot_state": list(self.slot_state),
            "slot_holder": list(self.slot_holder),
            "slot_source": list(self.slot_source),
            "debt": list(self.debt),
            "ordinal": dict(self.ordinal),
            "snapshot_step": dict(self.snapshot_step),
            "stored": {
                steps: None if self.snapshot_directory is not None else weights
                for steps, weights in self.stored.items()
            },
            "taken": self.taken,
            "loads": self.loads,
            "opened": self.opened,
            "width": self.width,
        }

    def load(self, state: dict, policy, averages, directory: Path | None = None) -> None:
        """Restore a state(); the snapshots it names without weights are read from `directory`
        (and written to `snapshot_directory` when that lacks them: a resume into another run
        directory). `averages` (already loaded) give the averages' slots back."""
        for name in self.TENSORS:
            setattr(self, name, state[name].to(self.device).clone())
        self.persona_table = state["persona_table"].to(self.device).clone()
        self.parameter_rows = tuple(state["parameter_rows"])
        self.slot_state = list(state["slot_state"])
        self.slot_holder = list(state["slot_holder"])
        self.slot_source = [
            None if source is None else tuple(source) for source in state["slot_source"]
        ]
        self.debt = list(state["debt"])
        self.ordinal, self.snapshot_step = dict(state["ordinal"]), dict(state["snapshot_step"])
        self.taken, self.loads = state["taken"], state["loads"]
        self.opened, self.width = state["opened"], state["width"]
        self.template = template(policy)
        self.stored = {}
        for steps, weights in state["stored"].items():
            if weights is None:
                weights = load_snapshot(directory, steps, "cpu")
            if (
                self.snapshot_directory is not None
                and not snapshot_path(self.snapshot_directory, steps).exists()
            ):
                save_snapshot(self.snapshot_directory, steps, weights)
            self.stored[steps] = weights
        self.slot_weights = [None] * self.slot_count
        for slot, source in enumerate(self.slot_source):
            if source is None or source[0] == "scripted":
                continue
            if source[0] == "snapshot":
                self._fill(slot, self.stored[source[1]])
            else:
                self._fill(slot, stepper_weights(policy, getattr(averages, source[0])))


SNAPSHOTS = "snapshots"


def snapshot_path(directory: Path, steps: int) -> Path:
    return Path(directory) / f"{steps:012d}.pt"


def save_snapshot(directory: Path, steps: int, weights: dict[str, Tensor]) -> None:
    """A snapshot's weights (in their training dtype), written once, atomically."""
    path = snapshot_path(directory, steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(weights, temporary)
    temporary.replace(path)


def load_snapshot(directory: Path | None, steps: int, device) -> dict[str, Tensor]:
    if directory is None:
        raise FileNotFoundError(
            f"snapshot {steps} is not in the checkpoint and no snapshot directory is given"
        )
    return torch.load(snapshot_path(directory, steps), map_location=device, weights_only=True)


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


def accept(draw: Tensor, wanted: Tensor, free: Tensor, slots: int) -> tuple[Tensor, Tensor]:
    """Of the draws `draw` (L,) where `wanted` (L,) bool, those within their slot's `free`
    rows (slots,), earlier environments first: (accepted (L,) bool, rows taken (slots,))."""
    one_hot = functional.one_hot(draw, slots) * wanted[:, None].long()
    rank = one_hot.cumsum(0).gather(1, draw[:, None]).squeeze(1)  # 1 for a slot's first draw
    accepted = wanted & (rank <= free.index_select(0, draw))
    taken = (functional.one_hot(draw, slots) * accepted[:, None].long()).sum(0)
    return accepted, taken


def slot_draws(
    loaded: Tensor, capacity: Tensor, slot: Tensor, done: Tensor, generator, slots: int
) -> Tensor:
    """(L,) a slot for every league environment (meaningful where `done`), level 2: uniform
    over the `loaded` (slots + 1,) slots with a free row of their `capacity` (the rows held
    are the environments not `done`), the rest drawn once more over those still free, then
    the free rows in slot order; past every free row, the overflow slot (index `slots`)."""
    holding = ~done & (slot >= 0)
    held = (functional.one_hot(slot.clamp(min=0), slots + 1) * holding[:, None].long()).sum(0)
    free = (capacity - held[:slots]).clamp(min=0) * loaded[:slots].long()
    count = slot.shape[0]
    chosen = torch.zeros_like(slot)
    wanted = done
    for _ in range(2):
        draw = sample_categorical((free > 0).float(), count, generator)
        accepted, taken = accept(draw, wanted, free, slots)
        chosen = torch.where(accepted, draw, chosen)
        free = free - taken
        wanted = wanted & ~accepted
    # What is left takes the free rows in order: the j-th left over (from 1) the slot holding
    # free row j when the slots' free rows are laid end to end; past them all, the overflow.
    order = wanted.long().cumsum(0)
    fill = torch.searchsorted(free.cumsum(0), order, right=False)
    return torch.where(wanted, fill, chosen)
