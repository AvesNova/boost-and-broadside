"""League sampling, round 2 (docs/rewrite/sections/league-sampling-sim.md, "Round 2"): the
slots' turnover, their number, the ladder's place in the pool and the weighting, compared on
round 1's reconstructed inputs (`bench.league_sampling`, run `2026-11-06-mixed-calm-fern`,
the snapshots' ratings repaired). CPU only, no games.

Round 1 simulated the slots at whole updates, with the drain taken as linear. This round
simulates every opponent row's game clock, so that drains end when their last row resets,
clean cuts cut real games and per-row draws happen at real resets:

- **Rows.** Each mode's opponent rows (512, or the nearest multiple of the slot count below),
  slot `row % slots`, as `bnb.runtime.opponents.OpponentSlots` lays them out. `SUBSTEPS`
  ticks of the simulation per update.
- **Games.** A game's length is drawn at its start: exponential, capped at the game clock
  (300 s, 14 updates), with its mean the round-1 fitted mean game for its mode and opponent
  class (ladder or network). The rows' first games are staggered (uniform phase), as the
  rollout's clocks are.
- **Turnover** (`Variant.rule`):
  - `drain`: `incoming` slots drain at once, round robin (1 is today's league; all slots is
    "draining together"). A draining slot's row switches to the incoming set at its next
    reset; the drain ends when the last row has switched, and the next begins with a draw
    from the weights after the previous update.
  - `cut`: every `every` updates, at the update boundary, `fraction` of the slots (round
    robin, at least one) draw anew. A slot whose draw differs from what it held is swapped in
    place: every row of it plays the new entry from that tick on, the opponent's carry reset,
    the game itself running on. That game is "cut": its ticks train, its result is lost.
  - `row`: every row draws its own entry from its own mode's weights at every reset (the
    limit).
  A slot holding the ladder plays the rungs: each row draws its rung at each reset (and at a
  cut onto the ladder) from its mode's weights over the ladder.
- **Ladder** (`Variant.ladder`): `entries`, 11 entries as today (a drawn rung makes the slot a
  ladder slot); `one`, one entry weighted by its best rung, its rows drawing rungs by the
  rungs' own weights; `none`, the ladder left out of the training pool (the RL case).
- **Weighting** (`Variant.shape`, `Variant.floor`): p (1 - p) or (1 - p)^2 on the gauge's
  predicted p, with or without the league's 5% floor spread evenly over the pool's entries.

Scored on the late updates (48-95) per seed against round 1's "true" p; see `Score`.

    uv run --no-sync python -m bench.league_sampling_2 /path/to/run [--seeds 200] [--json out.json]
"""

import argparse
import dataclasses
import json
import math
import pathlib
import time

import torch

from bench import league_sampling as one

MODES = one.MODES
SUBSTEPS = 4  # simulation ticks per update
CAP_UPDATES = 14.0  # the game clock (300 s) in updates: drains last exactly this in the run
ROWS_PER_MODE = 512  # the run's opponent rows per mode
LATE = one.LATE
WINDOW = 10  # the diversity window, updates
PRESENT = 0.01  # an entry counts as an opponent of a window from this share of its ticks


@dataclasses.dataclass(frozen=True)
class Variant:
    """One league configuration (module docstring)."""

    rule: str = "drain"  # "drain", "cut" or "row"
    incoming: int = 1  # drain: slots draining at once (0: all)
    every: int = 1  # cut: updates between cuts
    fraction: float = 1.0  # cut: share of the slots swapped at a cut
    slots: int = 4
    ladder: str = "entries"  # "entries", "one" or "none"
    shape: str = "p(1-p)"  # or "(1-p)^2"
    floor: bool = True

    @property
    def draining(self) -> int:
        """How many slots drain at once."""
        return self.slots if self.incoming == 0 else min(self.incoming, self.slots)

    @property
    def swapped(self) -> int:
        """How many slots a cut swaps."""
        return max(1, round(self.fraction * self.slots))

    def turnover(self) -> str:
        """The turnover rule, in words."""
        if self.rule == "row":
            return "draw per row"
        if self.rule == "cut":
            share = "all" if self.swapped == self.slots else f"{self.swapped}/{self.slots}"
            return f"cut k={self.every} {share}"
        if self.draining == self.slots:
            return "drain together"
        return "serial drain" if self.draining == 1 else f"{self.draining} incoming"

    def label(self) -> str:
        """The variant, in one line."""
        floor = "floor" if self.floor else "no floor"
        slots = "" if self.rule == "row" else f"{self.slots} slots, "
        return f"{self.turnover()}, {slots}ladder {self.ladder}, {self.shape}, {floor}"

    def stacked_sets(self, networks: int) -> int:
        """The stacked parameter sets (opponent memory): the slots and the incoming sets; a
        clean cut swaps in place between segments; the per-row limit stacks every network
        entry of the pool (`networks`)."""
        if self.rule == "row":
            return networks
        if self.rule == "cut":
            return self.slots
        return self.slots + self.draining

    def set_calls(self, networks: int) -> int:
        """The network calls per tick: every stacked set runs every tick on its own rows (an
        idle incoming set runs on slot 0's rows, as `OpponentSlots._network` does)."""
        return self.stacked_sets(networks)

    def compute(self, networks: int) -> float:
        """The opponent rows' network compute per tick, in units of every row once: a slot's
        set runs on its rows, an incoming set on one slot's rows; the per-row limit, its
        entries masked over every row (gathering instead would make it 1 at dynamic shapes)."""
        if self.rule == "row":
            return float(networks)
        return self.stacked_sets(networks) / self.slots


# ---- the weights


def unit_weights(
    p: torch.Tensor, ladder_mask: torch.Tensor, ladder: str, shape: str, floor: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """(the draw weights over the units (1 + networks,): the ladder first, then the network
    entries in pool order; the rows' weights over the rungs (rungs,)) from P(live beats each
    entry) `p` (K,) in pool order, `ladder_mask` (K,) marking the rungs."""
    value = p * (1 - p) if shape == "p(1-p)" else (1 - p) ** 2
    rungs, networks = value[ladder_mask], value[~ladder_mask]
    share = one.FLOOR if floor else 0.0

    def normal(x: torch.Tensor) -> torch.Tensor:
        total = float(x.sum())
        return x / total if total > 0 else torch.full_like(x, 1.0 / x.numel())

    if ladder == "entries":
        weight = (1 - share) * normal(value) + share / value.numel()
        units = torch.cat([weight[ladder_mask].sum()[None], weight[~ladder_mask]])
        split = normal(weight[ladder_mask])
    elif ladder == "one":
        units = (1 - share) * normal(torch.cat([rungs.max()[None], networks]))
        units = units + share / units.numel()
        split = normal(rungs)
    elif ladder == "none":
        nets = (1 - share) * normal(networks) + share / networks.numel()
        units = torch.cat([torch.zeros(1, dtype=nets.dtype), nets])
        split = normal(rungs)
    else:
        raise ValueError(ladder)
    return units, split


@dataclasses.dataclass
class Tables:
    """A variant's inputs per update: the units' draw weights mixed over the modes (U, 1 + N)
    and per mode (U, M, 1 + N), each mode's weights over the rungs (U, M, L); the "true" p
    per mode and entry (U, M, E), entries the rungs then the networks (NaN before an entry
    joins); each entry's class (E,) (0 ladder, 1 network) and whether it is a snapshot;
    each mode and class's mean game, in updates (M, 2)."""

    mixed: torch.Tensor
    per_mode: torch.Tensor
    split: torch.Tensor
    truth: torch.Tensor
    network: torch.Tensor
    snapshot: torch.Tensor
    mean_game: torch.Tensor
    names: list[str]


def tables(
    variant: Variant,
    updates: list[one.Update],
    ratings: list[dict[str, dict[str, float]]],
    truth: list[dict[str, dict[str, float]]],
    mean_game: dict[str, dict[str, float]],
) -> Tables:
    """`Tables` for `variant` from the round-1 inputs (`ratings` the gauge's, as repaired or
    as run; `truth` the scoring p; `mean_game` per mode and class, in updates)."""
    final = updates[-1].rows[MODES[0]]
    rung_names = [row.name for row in final if row.kind in one.LADDER_KINDS]
    network_names = [row.name for row in final if row.kind not in one.LADDER_KINDS]
    names = rung_names + network_names
    count_u, count_m = len(updates), len(MODES)
    mixed = torch.zeros(count_u, 1 + len(network_names), dtype=torch.float64)
    per_mode = torch.zeros(count_u, count_m, 1 + len(network_names), dtype=torch.float64)
    split = torch.zeros(count_u, count_m, len(rung_names), dtype=torch.float64)
    scored = torch.full((count_u, count_m, len(names)), math.nan, dtype=torch.float64)
    for update in updates:
        pool = update.names()
        mask = torch.tensor([update.row(MODES[0], n).kind in one.LADDER_KINDS for n in pool])
        columns = [1 + network_names.index(n) for n in pool if n in network_names]
        for m, mode in enumerate(MODES):
            p = one.probabilities(update, mode, ratings[update.index][mode])
            units, rungs = unit_weights(p, mask, variant.ladder, variant.shape, variant.floor)
            per_mode[update.index, m, 0] = units[0]
            per_mode[update.index, m, columns] = units[1:]
            split[update.index, m] = rungs
            for name in pool:
                scored[update.index, m, names.index(name)] = truth[update.index][mode][name]
        mixed[update.index] = per_mode[update.index].mean(0)
    kinds = {row.name: row.kind for row in final}
    return Tables(
        mixed=mixed,
        per_mode=per_mode,
        split=split,
        truth=scored,
        network=torch.tensor([kinds[n] not in one.LADDER_KINDS for n in names]),
        snapshot=torch.tensor([kinds[n] == "checkpoint" for n in names]),
        mean_game=torch.tensor(
            [[mean_game[mode]["ladder"], mean_game[mode]["network"]] for mode in MODES],
            dtype=torch.float64,
        ),
        names=names,
    )


# ---- the simulation


def hazard(mean: float, cap: float) -> float:
    """The exponential rate whose length capped at `cap` has mean `mean` (bisection on
    (1 - exp(-rate cap)) / rate, which falls with the rate)."""
    if mean >= cap:
        return 1e-9
    low, high = 1e-9, 100.0
    for _ in range(200):
        rate = (low + high) / 2
        if (1 - math.exp(-rate * cap)) / rate > mean:
            low = rate
        else:
            high = rate
    return (low + high) / 2


@dataclasses.dataclass
class Played:
    """Per seed and update (S, U, ...): the ticks and the games ended per mode and entry
    (S, U, M, E), the games among them that a cut touched and their ticks, the rows whose
    opponent a cut reset (S, U) and the natural game ends (S, U); the slots' final entries
    (S, slots) and the rows' final units (S, R), for the tests; the drains' lengths in
    updates."""

    ticks: torch.Tensor
    games: torch.Tensor
    cut_games: torch.Tensor
    cut_ticks: torch.Tensor
    cut_resets: torch.Tensor
    ends: torch.Tensor
    slot_unit: torch.Tensor
    row_unit: torch.Tensor
    drains: list[float]


def simulate(variant: Variant, table: Tables, seeds: int, seed: int = 0) -> Played:
    """The rows' games under `variant`, `seeds` runs at once (module docstring)."""
    generator = torch.Generator().manual_seed(seed)
    count_s, slots = seeds, variant.slots
    count_u, count_m, count_e = table.truth.shape
    rungs = table.split.shape[-1]
    per_mode = slots * (ROWS_PER_MODE // slots)
    count_r = count_m * per_mode
    row_mode = torch.arange(count_r) // per_mode
    row_slot = (torch.arange(count_r) % per_mode) % slots
    cap = int(round(CAP_UPDATES * SUBSTEPS))
    rates = torch.tensor(
        [
            [hazard(float(table.mean_game[m, c]) * SUBSTEPS, cap) for c in range(2)]
            for m in range(count_m)
        ],
        dtype=torch.float64,
    )

    def uniform(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=generator, dtype=torch.float64)

    def pick(weights: torch.Tensor, *shape: int) -> torch.Tensor:
        """Draws by inverse CDF: weights (..., C) broadcast against `shape`."""
        cumulative = weights.cumsum(-1)
        target = uniform(*shape)[..., None] * cumulative[..., -1:]
        return (target >= cumulative[..., :-1]).sum(-1)

    def entry_of(unit: torch.Tensor, update: int) -> torch.Tensor:
        rung = pick(table.split[update][row_mode], count_s, count_r)
        return torch.where(unit == 0, rung, unit - 1 + rungs)

    def length(entry: torch.Tensor) -> torch.Tensor:
        rate = rates[row_mode[None, :].expand_as(entry), table.network[entry].long()]
        drawn = torch.ceil(-torch.log1p(-uniform(*entry.shape)) / rate)
        return drawn.clamp(1, cap)

    draining_count = variant.draining
    slot_unit = pick(table.mixed[0], count_s, slots)
    incoming = pick(table.mixed[0], count_s, slots)
    draining = torch.zeros(count_s, slots, dtype=torch.bool)
    if variant.rule == "drain":
        draining[:, :draining_count] = True
    next_slot = torch.full((count_s,), draining_count % slots, dtype=torch.int64)
    began = torch.zeros(count_s, slots)
    cut_pointer = 0
    if variant.rule == "row":
        row_unit = pick(table.per_mode[0][row_mode], count_s, count_r)
    else:
        row_unit = slot_unit[:, row_slot]
    row_entry = entry_of(row_unit, 0)
    remaining = torch.ceil(uniform(count_s, count_r) * cap).clamp(1, cap)  # staggered
    elapsed = torch.zeros(count_s, count_r, dtype=torch.float64)
    switched = torch.zeros(count_s, count_r, dtype=torch.bool)
    touched = torch.zeros(count_s, count_r, dtype=torch.bool)

    shape = (count_s, count_u, count_m * count_e)
    ticks, games = torch.zeros(shape), torch.zeros(shape)
    cut_games, cut_ticks = torch.zeros(shape), torch.zeros(shape)
    cut_resets, ends = torch.zeros(count_s, count_u), torch.zeros(count_s, count_u)
    drains: list[float] = []
    ones = torch.ones(count_s, count_r)

    for step in range(count_u * SUBSTEPS):
        u = step // SUBSTEPS
        previous = max(u - 1, 0)
        if variant.rule == "cut" and step % SUBSTEPS == 0 and u > 0 and u % variant.every == 0:
            for j in range(variant.swapped):
                slot = (cut_pointer + j) % slots
                new = pick(table.mixed[previous], count_s)
                changed = new != slot_unit[:, slot]
                slot_unit[:, slot] = new
                rows = changed[:, None] & (row_slot == slot)[None, :]
                row_unit = torch.where(rows, new[:, None], row_unit)
                row_entry = torch.where(rows, entry_of(row_unit, previous), row_entry)
                touched |= rows
                cut_resets[:, u] += rows.sum(1)
            cut_pointer += variant.swapped
        index = row_mode[None, :] * count_e + row_entry
        ticks[:, u].scatter_add_(1, index, ones)
        elapsed += 1
        remaining -= 1
        ended = remaining <= 0
        games[:, u].scatter_add_(1, index, ended.float())
        cut = ended & touched
        cut_games[:, u].scatter_add_(1, index, cut.float())
        cut_ticks[:, u].scatter_add_(1, index, (cut * elapsed).float())
        ends[:, u] += ended.sum(1)
        if variant.rule == "row":
            new_unit = pick(table.per_mode[previous][row_mode], count_s, count_r)
        elif variant.rule == "drain":
            switched |= ended & draining.gather(1, row_slot[None, :].expand(count_s, -1))
            slot_index = row_slot[None, :].expand(count_s, -1)
            new_unit = torch.where(
                switched, incoming.gather(1, slot_index), slot_unit.gather(1, slot_index)
            )
        else:
            new_unit = slot_unit[:, row_slot]
        row_unit = torch.where(ended, new_unit, row_unit)
        row_entry = torch.where(ended, entry_of(row_unit, previous), row_entry)
        remaining = torch.where(ended, length(row_entry), remaining)
        elapsed = torch.where(ended, 0.0, elapsed)
        touched &= ~ended
        if variant.rule == "drain":
            done_all = switched.view(count_s, count_m, -1, slots).all(1).all(1)
            done = draining & done_all
            for slot in range(slots):
                finished = done[:, slot]
                if not bool(finished.any()):
                    continue
                now = (step + 1) / SUBSTEPS
                drains.extend((now - began[finished, slot]).tolist())
                slot_unit[finished, slot] = incoming[finished, slot]
                switched &= ~(finished[:, None] & (row_slot == slot)[None, :])
                if draining_count == slots:
                    target = torch.full_like(next_slot, slot)
                else:
                    target = next_slot.clone()
                    draining[finished, slot] = False
                    next_slot = torch.where(finished, (next_slot + 1) % slots, next_slot)
                seeds_done = finished.nonzero().squeeze(1)
                draws = pick(table.mixed[previous], len(seeds_done))
                draining[seeds_done, target[seeds_done]] = True
                incoming[seeds_done, target[seeds_done]] = draws
                began[seeds_done, target[seeds_done]] = now
    view = (count_s, count_u, count_m, count_e)
    return Played(
        ticks.view(view),
        games.view(view),
        cut_games.view(view),
        cut_ticks.view(view),
        cut_resets,
        ends,
        slot_unit,
        row_unit,
        drains,
    )


# ---- the scores


def spread(values: torch.Tensor) -> tuple[float, float, float]:
    """(mean, 10th, 90th percentile) over seeds."""
    values = values.double()
    return (
        float(values.mean()),
        float(values.quantile(0.1)),
        float(values.quantile(0.9)),
    )


@dataclasses.dataclass
class Score:
    """A variant's late-update scores, (mean, p10, p90) over seeds, pooled over the late
    updates within a seed. open_games, open_steps: the games' and the ticks' share against
    entries the live policy truly beats less than `one.BEATEN` of the time; signal: the ticks'
    mean p (1 - p); snapshot_games, ladder_games: those entries' game shares; distinct,
    distinct_networks: the entries (and the network entries) holding at least PRESENT of a
    10-update window's ticks, both modes pooled; entropy: the window's opponent mix, bits;
    cut_games, cut_ticks: the share of the games a cut touched and of the ticks in them;
    cut_resets: the rows per update whose opponent a cut reset, a share of all rows;
    natural_resets: the same for game ends; drain: the mean drain, updates (drain rules)."""

    open_games: tuple[float, float, float]
    open_steps: tuple[float, float, float]
    signal: tuple[float, float, float]
    snapshot_games: tuple[float, float, float]
    ladder_games: tuple[float, float, float]
    distinct: tuple[float, float, float]
    distinct_networks: tuple[float, float, float]
    entropy: tuple[float, float, float]
    cut_games: tuple[float, float, float]
    cut_ticks: tuple[float, float, float]
    cut_resets: tuple[float, float, float]
    natural_resets: tuple[float, float, float]
    drain: float
    stacked_sets: int
    set_calls: int
    compute: float


def score(variant: Variant, table: Tables, played: Played, late: int = LATE) -> Score:
    """`Score` of one simulation."""
    truth = torch.nan_to_num(table.truth[late:], nan=1.0)[None]  # (1, U', M, E)
    open_ = (truth < one.BEATEN).double()
    signal = truth * (1 - truth)
    ticks, games = played.ticks[:, late:].double(), played.games[:, late:].double()
    tick_total = ticks.sum((1, 2, 3))
    game_total = games.sum((1, 2, 3))
    ladder = ~table.network
    rows = played.ticks[0, 0].sum() / SUBSTEPS
    windows = []
    for start in range(late, table.truth.shape[0] - WINDOW + 1):
        mix = played.ticks[:, start : start + WINDOW].sum((1, 2)).double()
        mix = mix / mix.sum(1, keepdim=True)
        present = mix >= PRESENT
        entropy = -(mix * torch.log2(mix.clamp_min(1e-300))).sum(1)
        windows.append(
            torch.stack([present.sum(1).double(), present[:, table.network].sum(1), entropy])
        )
    window = torch.stack(windows).mean(0)
    networks = int(table.network.sum())
    late_count = table.truth.shape[0] - late
    return Score(
        open_games=spread((games * open_).sum((1, 2, 3)) / game_total),
        open_steps=spread((ticks * open_).sum((1, 2, 3)) / tick_total),
        signal=spread((ticks * signal).sum((1, 2, 3)) / tick_total),
        snapshot_games=spread(games[..., table.snapshot].sum((1, 2, 3)) / game_total),
        ladder_games=spread(games[..., ladder].sum((1, 2, 3)) / game_total),
        distinct=spread(window[0]),
        distinct_networks=spread(window[1]),
        entropy=spread(window[2]),
        cut_games=spread(played.cut_games[:, late:].sum((1, 2, 3)) / game_total),
        cut_ticks=spread(played.cut_ticks[:, late:].sum((1, 2, 3)) / tick_total),
        cut_resets=spread(played.cut_resets[:, late:].sum(1) / late_count / rows),
        natural_resets=spread(played.ends[:, late:].sum(1) / late_count / rows),
        drain=sum(played.drains) / len(played.drains) if played.drains else math.nan,
        stacked_sets=variant.stacked_sets(networks),
        set_calls=variant.set_calls(networks),
        compute=variant.compute(networks),
    )


# ---- the grid


def grid() -> list[Variant]:
    """The variants run (docs: "Round 2", the grid): every turnover rule at today's
    sampler; the slot counts under the drain and cut rules; the ladder's three places and the
    weightings under the main turnover rules; the promising pairs."""
    base = Variant()
    turnovers = [
        base,
        dataclasses.replace(base, incoming=2),
        dataclasses.replace(base, incoming=0),
        *[Variant(rule="cut", every=k, fraction=f) for k in (1, 2, 4) for f in (1.0, 0.5, 0.25)],
        Variant(rule="row"),
    ]
    out = list(turnovers)
    for slots in (6, 8):
        for variant in turnovers[:-1]:
            out.append(dataclasses.replace(variant, slots=slots))
    main = [
        base,
        dataclasses.replace(base, incoming=0),
        Variant(rule="cut", every=2, fraction=1.0),
        Variant(rule="cut", every=1, fraction=0.25),
        Variant(rule="row"),
    ]
    for variant in main:
        for ladder in ("entries", "one", "none"):
            for shape in ("p(1-p)", "(1-p)^2"):
                for floor in (True, False):
                    candidate = dataclasses.replace(
                        variant, ladder=ladder, shape=shape, floor=floor
                    )
                    if candidate not in out:
                        out.append(candidate)
    for slots in (6, 8):  # the promising pairs: more slots with the ladder as one entry
        for variant in main[1:4]:
            for ladder in ("one", "none"):
                candidate = dataclasses.replace(variant, slots=slots, ladder=ladder)
                if candidate not in out:
                    out.append(candidate)
    return out


def round_one_inputs(run: pathlib.Path):
    """Round 1's reconstruction: (the updates, the inputs, the mean game per mode and class
    in updates)."""
    updates = one.load(run)
    states = one.checkpoint_states(run, updates)
    one.check_weights(updates)  # fills the live ratings the fit had not yet logged
    timeline = one.replay_draws(updates, states)
    fits = one.replay_snapshot_fits(updates)
    playing = one.reconstructed_residency(updates, timeline)
    seconds = one.seconds_per_update(timeline)
    rows = one.opponent_rows(run)
    rates = one.fit_rates(updates, playing)
    for _ in range(5):
        lags = one.game_lags(rates, rows, seconds)
        resident = one.reconstructed_residency(updates, timeline, lags)
        rates = one.fit_rates(updates, resident)
    lags = one.game_lags(rates, rows, seconds)
    return updates, one.reconstruct_inputs(updates, fits), lags


def main() -> None:
    """Run the grid on a run directory and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    parser.add_argument("--as-run", action="store_true", help="the gauge's ratings as run")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    updates, inputs, lags = round_one_inputs(arguments.run)
    ratings = inputs.gauge if arguments.as_run else inputs.repaired
    print("mean game, updates:", lags)
    report = {"mean_game_updates": lags, "variants": {}}
    for variant in grid():
        began = time.time()
        table = tables(variant, updates, ratings, inputs.truth, lags)
        result = score(variant, table, simulate(variant, table, arguments.seeds))
        report["variants"][variant.label()] = dict(
            variant=dataclasses.asdict(variant), score=dataclasses.asdict(result)
        )
        cells = "  ".join(
            f"{key} {value[0]:.3f}" if isinstance(value, tuple) else f"{key} {value}"
            for key, value in dataclasses.asdict(result).items()
        )
        print(f"{variant.label():58s} {cells}  ({time.time() - began:.0f}s)", flush=True)
        if arguments.json is not None:
            arguments.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
