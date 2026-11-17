"""The unified evaluator's engine: one compiled rollout per size bucket, run continuously.

**A bucket's batch.** One rollout of `environments` games at the bucket's ships per team
(every smaller game padded with benched slots: `bnb.game.api`'s `active` counts), its
environments split into one contiguous slice per game preset in play (the modes' rules are
the compiled graph's constants, so a slice keeps its mode; the split follows the planner's
demand at the start). Everything else is data the host writes between graph replays, for
each environment's NEXT game: its players (`RefillSeat.pending`), seat, active ship counts,
ship lineup and arena scale (`BucketRollout`'s pending buffers). When a game ends, the
environment starts its next game at once inside the graph, from those buffers.

**The host loop** (`BucketEngine.step`) never waits on the block it has just launched:
it replays block k, queues the game logs' copy to pinned memory behind it, then reads
block k - 1's logs (long done), tallies the games that ended there (`Planner.finished`),
asks the planner for each such environment's game after next, and queues those writes
behind block k. So the GPU always has the next block queued while the host plans. A game
lasts thousands of ticks, so the one-block lag never meets an environment twice.

**Sizing** (`size_environments`): the free memory over a measured per-environment
footprint (two eager probes at small batches separate the fixed cost from the
per-environment one), at `MEMORY_SHARE` of the free memory, rounded down to a multiple of
`ROUNDING` so a machine's size, and so its compiled shapes, repeat run to run (one warm-up
per bucket per machine; the compile caches export and import as the box's
`docs/rewrite/sections/launch/cache.sh` does). An explicit count overrides it.
"""

import dataclasses
import json
import math
import pathlib
import random
import time

import torch

from bnb.contracts import ArenaScale, GameConfig, Pool
from bnb.contracts.config import TEAM_DEATHMATCH
from bnb.contracts.pools import pool_game
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.game.modes.rules import PRESETS, mode_rules_of
from bnb.league.matches import Player, card_of, imugi_play, network_groups
from bnb.models.yemong.model import Yemong
from bnb.runs import checkpoint
from bnb.runtime.rollout import Rollout
from bnb.runtime.train import model_controller

from .cells import Cell, Condition, Outcome, arena_uniform, bucket_of
from .planner import Planner
from .seat import RefillSeat, SeatPlayer

SEAT = "unified"
MEMORY_SHARE = 0.7  # of the free device memory the batch may take
ROUNDING = 64  # environments: a sized batch is a multiple of this
# the compiled engine's bytes per environment over the eager probe's (the captured graph's
# pool, the warm-up's carry copy, the allocator's reserve): on the laptop 4070 at bucket 5,
# 1024 environments, the eager probe gave 1.12 MB an environment and the compiled engine
# 1.22 MB allocated, 1.57 MB reserved (1.4x); 1.5 leaves room (unified-eval/results.md)
COMPILED_FACTOR = 1.5
SCALE_SAMPLING = ArenaScale()  # the map-size draw a slice with scaled cells is built with
CAPACITY_SLACK = 1.5  # a lane's rows over its player's expected share of the network rows
MINIMUM_CAPACITY = 32
MAXIMUM_RUNS = 3  # stacked forwards per network group
# The resident set (`BucketEngine` residents): a field of more network players than this
# plays among this many resident ones a group, their lanes sized evenly (`RESIDENT_SLACK`
# times the rows over the lanes), a resident swapped out (once its games are done) for the
# most valuable player outside when that one's value is `SWAP_RATIO` times the least
# valuable resident's; `REBALANCE_BLOCKS` blocks between the checks.
RESIDENTS = 12
RESIDENT_SLACK = 1.15
SWAP_RATIO = 1.25
REBALANCE_BLOCKS = 4
# A resident set's run has `SPARE_LANES` lanes beyond its residents: a swap seats the
# incoming player in a spare lane at once, while the outgoing one drains, so no environment
# waits on a swap; the drained lane is the next spare.
SPARE_LANES = 1
# A leaving resident still takes games expected to last at most this many game seconds
# (elimination's, in that slice) while one of its longer games is in flight, so its lane
# keeps playing as it drains and the drain ends no later.
SHORT_SECONDS = 120.0
# An environment's next game keeps its players (no new lane rows) when that game is worth at
# least this share of the best one (`Planner.choose`'s prefer), with a resident set: an
# environment holds its current and next games' rows, so new players for every next game
# would need twice the lanes.
PREFER_RATIO = 0.5
TICKETS = ("kills_per_ship", "ticket_exponent", "ticket_rounding")


def base_game(presets: list[str], ships_per_team: int) -> GameConfig:
    """The batch's shared config: team deathmatch's preset when one plays (its tickets are
    config fields, not a pool's), else the first, at the bucket's size. Raises ValueError
    when a preset's game is not the one its pool would play (another size, set or rule)."""
    first = next((p for p in presets if PRESETS[p].mode == TEAM_DEATHMATCH), presets[0])
    base = dataclasses.replace(PRESETS[first], ships_per_team=ships_per_team)
    for preset in presets:
        game = dataclasses.replace(PRESETS[preset], ships_per_team=ships_per_team)
        pooled = pool_game(base, preset_pool(preset, 1.0))
        if (game.num_fields, game.ship_set) != (base.num_fields, base.ship_set):
            raise ValueError(f"{preset}: other fields or ship set than {first}: one batch")
        theirs, ours = mode_rules_of(game), mode_rules_of(pooled)
        if game.mode != TEAM_DEATHMATCH:  # the tickets are read in team deathmatch only
            ours = dataclasses.replace(ours, **{k: getattr(theirs, k) for k in TICKETS})
        if ours != theirs or pooled.episode_seconds != game.episode_seconds:
            raise ValueError(f"{preset}: its pool in a {first} batch would play other rules")
    return base


def preset_pool(preset: str, share: float, scaled: bool = False) -> Pool:
    """A preset's slice of the batch: its mode, arena and clocks, both seats the unified
    seat; scaled: built with map-size sampling (each game's scale then set by the host)."""
    game = PRESETS[preset]
    return Pool(
        preset,
        share,
        (SEAT, SEAT),
        mode=game.mode,
        arena_radius=game.arena_radius,
        spawn_groups=game.spawn_groups,
        episode_seconds=game.episode_seconds,
        respawn_seconds=game.respawn_seconds,
        arena_scale=SCALE_SAMPLING if scaled else None,
    )


class BucketRollout(Rollout):
    """The training rollout with each environment's next game read from pending buffers the
    host writes (seat, active counts, lineup, arena scale's uniform) instead of keyed
    draws. `start` begins every environment's first game from them."""

    def __init__(self, config, types, pools, controllers, environments, length, device, **kwargs):
        S, N = config.ships_per_team, 2 * config.ships_per_team
        self.pending_seat = torch.zeros(environments, dtype=torch.int64, device=device)
        self.pending_active = torch.full((environments, 2), S, dtype=torch.int32, device=device)
        self.pending_ship_type = torch.zeros((environments, N), dtype=torch.int32, device=device)
        self.pending_arena = torch.full((environments,), -1.0, device=device)
        self.arena_slot = None
        super().__init__(
            config,
            types,
            pools,
            controllers,
            SEAT,
            environments,
            length,
            device,
            staggered=False,
            benching=True,
            lineup_types=(0,),
            **kwargs,
        )
        if self.rules.samples_arena:
            self.arena_slot = self.rules.map_uniform_count + api.TEAMS + self.rules.num_ships

    def _seats(self, episode):
        return self.pending_seat

    def _lineups(self, episode):
        return self.pending_ship_type

    def _team_sizes(self, episode):
        return self.pending_active, None

    def _episode_uniforms(self, episode):
        uniforms = super()._episode_uniforms(episode)
        if getattr(self, "arena_slot", None) is None:
            return uniforms
        slot = self.arena_slot
        chosen = torch.where(self.pending_arena >= 0.0, self.pending_arena, uniforms[:, slot])
        return torch.cat([uniforms[:, :slot], chosen[:, None], uniforms[:, slot + 1 :]], 1)

    def _initial_carry(self, types, staggered, active=None, batch=0):
        if active is None:
            active = self.pending_active.clone()
        return super()._initial_carry(types, staggered, active, batch)

    @torch.no_grad()
    def start(self, batch: int = 0) -> None:
        """Every environment starts its first game from the pending buffers (a new carry,
        copied in place)."""
        from bnb.runtime.rollout import _assign

        _assign(self.carry, self._initial_carry(self.carry["types"], False, None, batch))

    @torch.no_grad()
    def play_block(self) -> None:
        """Advance `length` ticks (the records go to the store's first shard, unread)."""
        self.carry["segment_tick"].zero_()
        self._play(self.length, 0)


@dataclasses.dataclass
class Slot:
    """An environment's game: its cell, the seat its first player takes, and whether it is
    overtime (chosen for a met goal's further precision or as filler, `Planner.choose`),
    parked (no cell at all: Imugi against itself at 1 against 1, untallied) or short
    (expected to last at most `SHORT_SECONDS`)."""

    cell: Cell | None
    seat: int
    overtime: bool = False
    parked: bool = False
    short: bool = False


def seat_players(
    players: list[Player], rules: dict[str, tuple[bool, bool]] | None = None
) -> tuple[list[SeatPlayer], list[int | None]]:
    """The seat's view of the run's players, and each one's network group. rules: a
    scripted player's own Imugi rules by name (`SeatPlayer.rules`)."""
    rules = rules or {}
    groups = network_groups(players)
    seat = []
    for one, group in zip(players, groups, strict=True):
        play = imugi_play(one)
        if one.cheese is not None or one.ablations:
            raise ValueError(f"{one.name}: the unified evaluator plays no cheese or ablations")
        if play is not None:
            if play.parameters is not None:
                raise ValueError(f"{one.name}: no doctrine players")
            seat.append(
                SeatPlayer(
                    one.name, None, play.temperature, play.share, play.floor, rules.get(one.name)
                )
            )
        else:
            seat.append(SeatPlayer(one.name, group, temperature=one.temperature))
    return seat, groups


class BucketEngine:
    """One bucket's continuously running batch (module docstring). players: every player of
    the run (`bnb.league.matches.Player`, by name); shares: each preset's share of the
    environments; planner: the run's; lineup_types: ship type indices a game's lineup is
    drawn from (each slot uniformly, both teams flying it); scaled: the presets whose
    slice needs map-size sampling; player_share: each network player's expected share of
    the seats (`lane_runs` cuts the players into runs and sizes their lanes by it; None:
    equal shares); record: the games' callback (`record(cell, outcome, slot, bucket)`)."""

    def __init__(
        self,
        bucket: int,
        environments: int,
        shares: dict[str, float],
        players: list[Player],
        planner: Planner,
        device: torch.device,
        lineup_types: tuple[int, ...] = (0, 1),
        scaled: tuple[str, ...] = (),
        player_share: dict[str, float] | None = None,
        compiled: bool = True,
        length: int = 32,
        graph_ticks: int = 8,
        seed: int = 0,
        record=None,
        residents: int | None = RESIDENTS,
        rules: dict[str, tuple[bool, bool]] | None = None,
    ) -> None:
        self.bucket, self.planner, self.device = bucket, planner, device
        self.players = players
        self.length = length  # ticks a block
        self.index = {one.name: number for number, one in enumerate(players)}
        self.random = random.Random(seed * 1_000_003 + bucket)
        self.lineup_types = tuple(lineup_types)
        self.record = record
        presets = [p for p, share in shares.items() if share > 0.0]
        self.game = base_game(presets, bucket)
        counts = _split(environments, [shares[p] for p in presets])
        presets = [p for p, c in zip(presets, counts, strict=True) if c > 0]
        counts = [c for c in counts if c > 0]
        self.environments = sum(counts)
        self.presets = presets
        pools = tuple(
            preset_pool(p, c / self.environments, p in scaled)
            for p, c in zip(presets, counts, strict=True)
        )
        self.preset_of = [p for p, c in zip(presets, counts, strict=True) for _ in range(c)]
        seat_view, groups = seat_players(players, rules)
        self.network = [group is not None for group in groups]
        self.group_of = groups
        templates, states, loaded = [], {}, {}
        for index, group in enumerate(groups):
            if group is None:
                continue
            one = players[index]
            if one.policy in loaded:
                states[index] = states[loaded[one.policy]]
                continue
            loaded[one.policy] = index
            model = checkpoint.load_policy(one.policy, device).requires_grad_(False)
            if group == len(templates):
                templates.append(model)
            states[index] = {k: v.detach() for k, v in model.state_dict().items()}
        for template in templates:
            if isinstance(template, Yemong):
                template.mask_benched = True
        controllers = [model_controller(template, self.game) for template in templates]
        rows = 2 * self.environments
        networks = [one.name for one, g in zip(players, groups, strict=True) if g is not None]
        if player_share is None:
            player_share = {name: 1.0 / max(len(networks), 1) for name in networks}
        shares_of = [player_share.get(one.name, 0.0) for one in players]
        self.resident_set = residents is not None and len(networks) > residents
        if self.resident_set:
            runs = resident_runs(groups, shares_of, rows, residents)
        else:
            runs = lane_runs(groups, shares_of, rows)
        self.runs = runs
        self.capacity_of = {
            index: capacity
            for _, members, capacity in runs
            for index in members
            if index is not None
        }
        self.leaving: set[int] = set()  # residents draining before a swap
        self.swaps = 0
        self.types = card_of(players, self.game, device)
        rules = api.rules_of(self.game)
        self.seat = RefillSeat(
            controllers,
            seat_view,
            states,
            runs,
            rows,
            characteristics(self.types, rules),
            bucket,
            device,
        )
        self.rollout = BucketRollout(
            self.game,
            self.types,
            pools,
            {SEAT: self.seat},
            self.environments,
            length,
            device,
            compiled=compiled,
            graph_ticks=graph_ticks,
            seed=seed,
        )
        lineup = self.rollout.lineup
        environments_of, role = lineup.environments[SEAT].cpu(), lineup.role[SEAT].cpu()
        self.first_row = torch.zeros(self.environments, dtype=torch.int64)
        self.second_row = torch.zeros(self.environments, dtype=torch.int64)
        for row, (environment, r) in enumerate(zip(environments_of.tolist(), role.tolist())):
            (self.first_row if r == 0 else self.second_row)[environment] = row
        self.first_row_device = self.first_row.to(device)
        self.tick_length = self.rollout.rules.tick_length
        self.in_use = [0] * len(players)
        # environments whose current or next game is a longer than short one of each player
        self.long_held = [0] * len(players)
        self.current: list[Slot] = []
        self.next: list[Slot | None] = [None] * self.environments
        self.scripted = next(i for i, g in enumerate(groups) if g is None)
        self.finished_seen = torch.zeros(self.environments, dtype=torch.int64)
        self.games = 0
        self.blocks = 0
        self.missed = 0
        self._previous = None  # (pinned logs, event) of the last block
        self.parked = 0  # parked games ended
        self.parked_ticks = 0.0  # environment ticks of the parked games ended
        self.real_ticks = 0.0  # and of the tallied ones
        self.host_seconds = 0.0  # the host's tallying and planning (`step`)

    # ---- assignment

    def _rows(self, environment: int) -> set[tuple[int, int]]:
        """The (role, player) rows an environment's current and next games hold: a lane's
        rows in use, each row counted once however many of its games a player plays."""
        held = set()
        for slot in (self.current[environment], self.next[environment]):
            if slot is not None:
                held.add((0, self.index_of(slot, 0)))
                held.add((1, self.index_of(slot, 1)))
        return held

    def index_of(self, slot: Slot, role: int) -> int:
        """The player index of a slot's role (a parked slot: Imugi in both)."""
        if slot.cell is None:
            return self.scripted
        return self.index[slot.cell.first if role == 0 else slot.cell.second]

    def _count(self, environment: int, sign: int) -> None:
        for _, index in self._rows(environment):
            self.in_use[index] += sign
        held = set()
        for slot in (self.current[environment], self.next[environment]):
            if slot is not None and slot.cell is not None and not slot.short:
                held.update((self.index_of(slot, 0), self.index_of(slot, 1)))
        for index in held:
            self.long_held[index] += sign

    def parked_now(self) -> int:
        """The environments playing a parked game now."""
        return sum(1 for slot in self.current if slot is not None and slot.cell is None)

    def _fits(self, environment: int):
        """The planner's room check for an environment's next game: each network player's
        rows, with this game's, within its lane's capacity."""
        held = self._rows(environment)

        def fits(cell: Cell) -> bool:
            for role, name in ((0, cell.first), (1, cell.second)):
                index = self.index[name]
                capacity = self.capacity_of.get(index)
                if self.network[index] and capacity is None:
                    return False  # not resident
                if index in self.leaving and not (self.long_held[index] > 0 and self._short(cell)):
                    return False  # draining: short games only, while a longer one plays
                if capacity is None or (role, index) in held:
                    continue
                if self.in_use[index] + 1 > capacity:
                    return False
            return True

        return fits

    def _choose(self, environment: int) -> Slot:
        """An environment's next game: the planner's (overtime when its slice's goals are
        met); else parked (Imugi against itself, 1 against 1, untallied)."""
        prefer = None
        if self.resident_set:  # a next game on the rows the environment holds costs no lane room
            held = self._rows(environment)

            def prefer(cell: Cell) -> bool:
                return (0, self.index[cell.first]) in held and (
                    1,
                    self.index[cell.second],
                ) in held

        chosen = self.planner.choose(
            self.bucket, self.preset_of[environment], self._fits(environment), prefer, PREFER_RATIO
        )
        if chosen is None:
            return Slot(None, 0, parked=True)
        cell, seat, overtime = chosen
        return Slot(cell, seat, overtime=overtime, short=self._short(cell))

    def _short(self, cell: Cell) -> bool:
        return self.planner.expected_seconds(cell.condition) <= SHORT_SECONDS

    def _set_next(self, environment: int, slot: Slot) -> None:
        self._count(environment, -1)
        self.next[environment] = slot
        self._count(environment, 1)

    def _pending_values(self, slot: Slot) -> tuple:
        """(seat, active (2,), ship types (N,), arena uniform, first player, second player)
        of a slot, as the buffers hold them."""
        S = self.bucket
        active = [1, 1]
        scale = 1.0
        if slot.cell is not None:
            condition = slot.cell.condition
            active[slot.seat] = condition.first
            active[1 - slot.seat] = condition.second
            scale = condition.arena_scale
        team = [self.random.choice(self.lineup_types) for _ in range(S)]
        arena = -1.0
        if scale != 1.0 or self.rollout.arena_slot is not None:
            arena = arena_uniform(
                scale, SCALE_SAMPLING.sigma, SCALE_SAMPLING.low, SCALE_SAMPLING.high
            )
        first, second = self.index_of(slot, 0), self.index_of(slot, 1)
        return (slot.seat, active, team + team, arena, first, second)

    def _write(self, environments: list[int], slots: list[Slot]) -> None:
        """Queue the pending buffers' writes for these environments' next games."""
        if not environments:
            return
        values = [self._pending_values(slot) for slot in slots]
        index = torch.tensor(environments, dtype=torch.int64).pin_memory()
        seat = torch.tensor([v[0] for v in values], dtype=torch.int64).pin_memory()
        active = torch.tensor([v[1] for v in values], dtype=torch.int32).pin_memory()
        ship_type = torch.tensor([v[2] for v in values], dtype=torch.int32).pin_memory()
        arena = torch.tensor([v[3] for v in values], dtype=torch.float32).pin_memory()
        rows = torch.cat([self.first_row[index], self.second_row[index]]).pin_memory()
        players = torch.tensor(
            [v[4] for v in values] + [v[5] for v in values], dtype=torch.int64
        ).pin_memory()
        device, rollout = self.device, self.rollout
        index = index.to(device, non_blocking=True)
        rollout.pending_seat.index_copy_(0, index, seat.to(device, non_blocking=True))
        rollout.pending_active.index_copy_(0, index, active.to(device, non_blocking=True))
        rollout.pending_ship_type.index_copy_(0, index, ship_type.to(device, non_blocking=True))
        rollout.pending_arena.index_copy_(0, index, arena.to(device, non_blocking=True))
        self.seat.pending.index_copy_(
            0, rows.to(device, non_blocking=True), players.to(device, non_blocking=True)
        )

    def start(self) -> None:
        """Every environment's first game and the one after it, chosen and written; the
        rollout started from the first."""
        self.current = [None] * self.environments
        for environment in range(self.environments):
            self._set_next(environment, self._choose(environment))
        # the chosen become the current games
        for environment in range(self.environments):
            self._count(environment, -1)
            self.current[environment], self.next[environment] = self.next[environment], None
            self._count(environment, 1)
        self._write(list(range(self.environments)), self.current)
        self.rollout.start()
        for environment in range(self.environments):
            self._set_next(environment, self._choose(environment))
        self._write(list(range(self.environments)), self.next)

    # ---- the loop

    def _logs(self) -> torch.Tensor:
        """(B, 6) the first-role rows' game logs: finished games, the last game's result,
        its own and enemy deaths, its ticks; and the lanes' overflow count."""
        carry = self.rollout.carry["controllers"][SEAT]
        rows = self.first_row_device
        return torch.cat(
            [
                carry.finished[rows].double()[:, None],
                carry.last_result[rows].double()[:, None],
                carry.last_deaths[rows].double(),
                carry.last_ticks[rows].double()[:, None],
                self.seat.overflow.double().expand(rows.shape[0])[:, None],
            ],
            1,
        )

    def step(self) -> int:
        """One block: replay it, queue its logs' copy, tally the previous block's ended
        games and queue their environments' next writes. Returns the games tallied (parked
        games are not)."""
        self.rollout.play_block()
        self.blocks += 1
        logs = torch.empty((self.environments, 6), dtype=torch.float64, pin_memory=True)
        logs.copy_(self._logs(), non_blocking=True)
        event = torch.cuda.Event() if self.device.type == "cuda" else None
        if event is not None:
            event.record()
        previous, self._previous = self._previous, (logs, event)
        if previous is None:
            return 0
        held, held_event = previous
        if held_event is not None:
            held_event.synchronize()
        started = time.perf_counter()
        tallied = self._tally(held)
        self.host_seconds += time.perf_counter() - started  # planning, while the GPU plays
        return tallied

    def drain(self) -> int:
        """Tally the last block's logs (at the end)."""
        if self._previous is None:
            return 0
        held, event = self._previous
        if event is not None:
            event.synchronize()
        self._previous = None
        return self._tally(held)

    def _tally(self, logs: torch.Tensor) -> int:
        finished = logs[:, 0].long()
        delta = finished - self.finished_seen
        if int(logs[0, 5]) > 0:
            raise RuntimeError(f"bucket {self.bucket}: rows past a lane's capacity")
        ended = delta.nonzero().flatten().tolist()
        environments, slots = [], []
        tallied = 0
        for environment in ended:
            if int(delta[environment]) > 1:  # never at a game's length; counted, not tallied
                self.missed += int(delta[environment]) - 1
            result, own_deaths, enemy_deaths, ticks = logs[environment, 1:5].tolist()
            slot = self.current[environment]
            if slot.cell is not None:
                outcome = Outcome(
                    score=(result + 1.0) / 2.0,
                    first_kills=round(enemy_deaths),
                    second_kills=round(own_deaths),
                    seconds=ticks * self.tick_length,
                )
                self.planner.finished(slot.cell, outcome)
                if self.record is not None:
                    self.record(slot.cell, outcome, slot, self.bucket)
                self.games += 1
                tallied += 1
                self.real_ticks += ticks
            else:
                self.parked += 1
                self.parked_ticks += ticks
            environments.append(environment)
            slots.append(self._advance(environment))
        self.finished_seen = finished
        self._write(environments, slots)
        if self.resident_set and self.blocks % REBALANCE_BLOCKS == 0:
            self._rebalance()
        return tallied

    def _advance(self, environment: int) -> Slot:
        """An environment's game ended: its next game is under way and becomes the current
        one (the pending buffers already hold it), and the one after it is chosen."""
        self._count(environment, -1)
        self.current[environment], self.next[environment] = self.next[environment], None
        self._count(environment, 1)
        following = self._choose(environment)
        self._set_next(environment, following)
        return following

    # ---- the resident set

    def _player_values(self) -> dict[int, float]:
        """Each network player's value now: the planner's values of the cells it plays,
        those whose opponent is not resident at half (so a player comes in with its useful
        opponents, and a pair of outsiders still counts)."""
        values = self.planner.values() or self.planner.values(True)
        held: dict[int, float] = {}
        for cell, value in values.items():
            a, b = self.index.get(cell.first), self.index.get(cell.second)
            if a is None or b is None:
                continue
            here = [not self.network[i] or i in self.capacity_of for i in (a, b)]
            for one, other_here in ((a, here[1]), (b, here[0])):
                held[one] = held.get(one, 0.0) + value * (1.0 if other_here else 0.5)
        return held

    def _rebalance(self) -> None:
        """Swap residents (module constants): a drained leaving resident's lane is emptied
        (the next spare); while a run has an empty lane, the most valuable outsider is
        seated in it at once when it is worth `SWAP_RATIO` times the least valuable staying
        resident, who is marked leaving (no new games but short ones, `_fits`). Weights are
        copied on the device, between replays."""
        value = self._player_values()
        for run, (group, _, capacity) in enumerate(self.runs):
            lanes = self.seat.lanes[run]
            for lane, index in enumerate(lanes):
                if index in self.leaving and self.in_use[index] == 0:
                    self.leaving.discard(index)
                    self.capacity_of.pop(index, None)
                    self.seat.swap(run, lane, None)
            outside = [
                i for i, g in enumerate(self.group_of) if g == group and i not in self.capacity_of
            ]
            empty = [lane for lane, index in enumerate(lanes) if index is None]
            staying = [i for i in lanes if i is not None and i not in self.leaving]
            if not outside or not empty or not staying:
                continue
            incoming = max(outside, key=lambda i: value.get(i, 0.0))
            worst = min(staying, key=lambda i: value.get(i, 0.0))
            best = value.get(incoming, 0.0)
            if best > 0.0 and best > SWAP_RATIO * value.get(worst, 0.0):
                self.seat.swap(run, empty[0], incoming)
                self.capacity_of[incoming] = capacity
                self.leaving.add(worst)
                self.swaps += 1

    def wanted(self) -> bool:
        """Whether the bucket still has work: a target wants a game in it, or a game a target
        asked for (not overtime) is still being played (its result is wanted)."""
        if any(bucket == self.bucket for bucket, _ in self.planner.wanted()):
            return True
        return any(
            slot is not None and slot.cell is not None and not slot.overtime
            for slot in self.current + self.next
        )


def lane_runs(
    groups: list[int | None], shares: list[float], rows: int, slack: float = CAPACITY_SLACK
) -> list[tuple[int, list[int], int]]:
    """The network players cut into runs, each one stacked forward: within a group, the
    players by their expected rows (rows x their seat share) from the most, a run taking
    the next player while it expects at least half the run's first's; at most `MAXIMUM_RUNS`
    runs a group (each is a network copy in the compiled graph). Each run's lanes hold
    `slack` times its first player's expected rows (at least `MINIMUM_CAPACITY`, at most
    every row)."""
    runs = []
    for group in sorted({g for g in groups if g is not None}):
        members = sorted(
            (index for index, g in enumerate(groups) if g == group), key=lambda i: -shares[i]
        )
        cut: list[list[int]] = []
        for index in members:
            if cut and (shares[index] * 2 >= shares[cut[-1][0]] or len(cut) == MAXIMUM_RUNS):
                cut[-1].append(index)
            else:
                cut.append([index])
        for run in cut:
            expected = rows * shares[run[0]]
            capacity = min(rows, max(MINIMUM_CAPACITY, math.ceil(slack * expected)))
            runs.append((group, run, capacity))
    return runs


def resident_runs(
    groups: list[int | None], shares: list[float], rows: int, residents: int
) -> list[tuple[int, list[int | None], int]]:
    """A resident set's runs: per network group one run of at most `residents` lanes,
    seated with the group's players of the largest expected share, each lane holding
    `RESIDENT_SLACK` times an even part of the rows, and `SPARE_LANES` empty lanes more
    when the group has more players than lanes (`BucketEngine._rebalance` swaps)."""
    found = sorted({g for g in groups if g is not None})
    members = {g: [i for i, one in enumerate(groups) if one == g] for g in found}
    lanes = {g: min(residents, len(members[g])) for g in found}
    total = sum(lanes.values())
    runs = []
    for group in found:
        seated = sorted(members[group], key=lambda i: -shares[i])[: lanes[group]]
        capacity = min(rows, max(MINIMUM_CAPACITY, math.ceil(RESIDENT_SLACK * rows / total)))
        spare = SPARE_LANES if len(members[group]) > lanes[group] else 0
        runs.append((group, list(seated) + [None] * spare, capacity))
    return runs


def _split(total: int, weights: list[float]) -> list[int]:
    """`total` in whole parts by the weights' largest remainders (a weight of 0 gets 0)."""
    weight = sum(weights)
    shares = [total * w / weight for w in weights]
    parts = [math.floor(s) for s in shares]
    order = sorted(range(len(weights)), key=lambda i: shares[i] - parts[i], reverse=True)
    for index in order[: total - sum(parts)]:
        parts[index] += 1
    return parts


# ---- sizing


def free_memory(device: torch.device) -> int:
    """The device's free bytes (the caching allocator's reserve counted free)."""
    free, _ = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


def footprint(make, device: torch.device, small: int = 64, large: int = 256) -> tuple[float, float]:
    """(fixed bytes, bytes per environment) of an engine: `make(environments)` builds an
    uncompiled engine; after a first throwaway build (the device's one-time allocations),
    two of them (small and large) play a few eager ticks, and the peaks give the line.
    The compiled engine's footprint differs from the eager one's (`COMPILED_FACTOR`)."""
    peaks = []
    import gc

    for environments in (small, small, large):
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        engine = make(environments)
        engine.start()
        for _ in range(2):
            engine.rollout._tick()
        torch.cuda.synchronize(device)
        peaks.append(torch.cuda.max_memory_allocated(device) - before)
        del engine
        gc.collect()
    peaks = peaks[1:]
    per = max((peaks[1] - peaks[0]) / (large - small), 1.0)
    return max(peaks[0] - per * small, 0.0), per


def size_environments(fixed: float, per: float, free: int, share: float = MEMORY_SHARE) -> int:
    """The environments that fit `share` of `free` bytes at the compiled engine's footprint
    (the eager probe's per-environment bytes times `COMPILED_FACTOR`), as a multiple of
    `ROUNDING` (at least one multiple)."""
    count = int((share * free - fixed) // (per * COMPILED_FACTOR))
    return max(ROUNDING, count // ROUNDING * ROUNDING)


# ---- records


class GameLog:
    """The run's durable record: `games.jsonl`, one line per flush with every game finished
    since (each line written whole and synced), read back on a resume."""

    def __init__(self, directory: pathlib.Path) -> None:
        self.path = directory / "games.jsonl"
        directory.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def add(self, cell: Cell, outcome: Outcome, slot: Slot, bucket: int) -> None:
        """Add a finished game to the next flush."""
        c = cell.condition
        self.rows.append(
            dict(
                preset=c.preset,
                first_count=c.first,
                second_count=c.second,
                arena_scale=c.arena_scale,
                first=cell.first,
                second=cell.second,
                seat=slot.seat,
                overtime=slot.overtime,
                bucket=bucket,
                score=outcome.score,
                kills=[outcome.first_kills, outcome.second_kills],
                seconds=round(outcome.seconds, 3),
            )
        )

    def export_to(self, pool, players: list[Player], conditions: set[str]) -> None:
        """Also append the games of these condition keys (`Condition.key`, the Elo
        targets') to a rating pool (`bnb.league.rating_pool.RatingPool`), one batch per
        condition per flush, each game a block, the players registered under their spec
        names with their identities, so `bnb pool table` and `bnb pool curves` read them."""
        from bnb.league.rating_pool import PoolPlayer
        from bnb.league.tournament import identity

        held, _ = pool.players()
        identities = {one.name: identity(one) for one in players}
        fresh = [
            PoolPlayer(name, mark, paths=(str(one.policy),) if one.policy else ())
            for one in players
            for name, mark in ((one.name, identities[one.name]),)
            if name not in held
        ]
        if fresh:
            pool.register(fresh)
        self.pool, self.identities, self.conditions = pool, identities, conditions

    def _export(self) -> None:
        import datetime

        from bnb.game import VERSION

        by_key: dict[str, list[dict]] = {}
        for row in self.rows:
            key = Condition(
                row["preset"], row["first_count"], row["second_count"], row["arena_scale"]
            ).key
            if key in self.conditions:
                by_key.setdefault(key, []).append(row)
        for key, rows in by_key.items():
            results = [
                [1, 0, 0] if r["score"] == 1.0 else ([0, 1, 0] if r["score"] == 0.5 else [0, 0, 1])
                for r in rows
            ]
            kills = [r["kills"] for r in rows]
            self.pool.append_batch(
                key,
                dict(
                    game_version=VERSION,
                    matchups=[[r["first"], r["second"]] for r in rows],
                    identities={
                        n: self.identities[n] for r in rows for n in (r["first"], r["second"])
                    },
                    results=results,
                    kill_share=[a / (a + b) if a + b else 0.5 for a, b in kills],
                    kills=kills,
                    layout=dict(unified=True),
                    wall_time=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                ),
            )

    def flush(self) -> None:
        """Append the games since the last flush as one synced line (and to the rating pool)."""
        if not self.rows:
            return
        import os

        with self.path.open("a") as file:
            file.write(json.dumps(dict(time=time.time(), games=self.rows)) + "\n")
            file.flush()
            os.fsync(file.fileno())
        if getattr(self, "pool", None) is not None:
            self._export()
        self.rows = []

    def replay(self) -> list[tuple[Cell, Outcome]]:
        """Every recorded game (a last line cut short is skipped)."""
        if not self.path.exists():
            return []
        found = []
        for line in self.path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            for row in entry["games"]:
                condition = Condition(
                    row["preset"], row["first_count"], row["second_count"], row["arena_scale"]
                )
                outcome = Outcome(row["score"], row["kills"][0], row["kills"][1], row["seconds"])
                found.append((Cell(condition, row["first"], row["second"]), outcome))
        return found


def bucket_cells(planner: Planner) -> dict[int, dict[str, float]]:
    """Each bucket's presets and their demand shares (`Planner.demand`)."""
    out: dict[int, dict[str, float]] = {}
    for (bucket, preset), share in planner.demand().items():
        out.setdefault(bucket, {})[preset] = share
    return out


def scaled_presets(planner: Planner, bucket: int) -> tuple[str, ...]:
    """The presets of a bucket with a cell off the preset's own map."""
    found = set()
    for target in planner.targets:
        for cell in target.proposals():
            c = cell.condition
            if c.arena_scale != 1.0 and bucket_of(c, planner.buckets) == bucket:
                found.add(c.preset)
    return tuple(sorted(found))
