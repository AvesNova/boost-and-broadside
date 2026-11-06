"""The league's opponent sampling, replayed from a run's logs and compared across samplers
(docs/rewrite/sections/league-sampling-sim.md). CPU only, no games: it reads `metrics.jsonl`
and the training checkpoints' league state, nothing else.

Four parts, each driven by the league's own code (`bnb.league.pool.sampling_weights`,
`bnb.league.ratings`, `bnb.runtime.opponents.counter_uniform` and `draw`):

1. **The weights.** Every update's logged sampling share per mode and entry, recomputed from
   the logged ratings. Before the fit first rates the live policy its rating is not logged;
   it is solved for from the shares (one number per update and mode) and the residual shown.
2. **The draws.** The host's draws replayed with the run's swap stream and the logged weights
   of the update before each draw: which entry each slot held at every update, checked
   against the logged slot column and every training checkpoint's `held` and `swaps`.
3. **The snapshots' ratings.** Run commit 724153e7 rated snapshots with the gauge's ladder
   fit (`fit_ladder`, reproduced here as `fit_ladder_724153e7`, since replaced by the run's
   one fit): each update's fit is replayed from the snapshots' credited games and the logged
   ladder ratings and compared with the logged ratings, beside the same likelihood's true
   maximum (`snapshot_maximum`).
4. **Residency to games.** The rows each entry played at each update (a slot is a quarter of
   every mode's opponent rows; the draining slot's rows switch over the drain, taken as
   linear; a slot holding the ladder plays the rungs by its mode's ladder weights), times a
   game rate per mode and opponent class (ladder or network) fitted over the run, against
   the logged `pool/game_share/<kind>`.

Then the samplers (`SAMPLERS`), on the same reconstructed ratings with the snapshots' ratings
repaired, through a Monte Carlo of the slots and drains (`simulate`), scored on the late
updates by the share of games against entries the live policy beats less than 90% of the
time, the learning signal (sum of share x p (1 - p)) and the information the same games give
the live policy's rating. Probabilities "true" for the scoring: observed scores for the
ladder and the averages, the live rating at the snapshot's update for a snapshot (it was the
live weights).

    uv run --no-sync python -m bench.league_sampling /path/to/run [--seeds 400] [--json out.json]
"""

import argparse
import dataclasses
import json
import math
import pathlib
import statistics

import torch

from bnb.league.pool import sampling_weights
from bnb.league.ratings import SCALE, _maximise, win_probability
from bnb.runtime.opponents import counter_uniform, draw, stream_key

MODES = ("frontline", "deathmatch")
KINDS = ("scripted", "rung", "average", "checkpoint")  # `bnb.league.league.ENTRY_KINDS`
LADDER_KINDS = ("scripted", "rung")
FLOOR = 0.05  # the run's `LeagueConfig.floor`
SLOTS = 4  # the run's `LeagueConfig.slots`
BEATEN = 0.9  # an entry the live policy beats at least this often counts as beaten
LATE = 48  # the late updates the samplers are scored on: from this one to the end
NEWEST = 2  # the fixed-share sampler's newest snapshots
LADDER_WINDOW = 20  # the updates a rung's observed score is taken over


# ---- the run's logs


@dataclasses.dataclass
class Row:
    """One entry of one mode at one update, as `pool/entries` logs it: the gauge's rating and
    standard error, the sampling share, this update's games and score against it, and the
    first slot holding it (the logged column: a name held twice shows only its first)."""

    name: str
    kind: str
    rating: float
    stderr: float
    share: float
    games: float
    score: float | None
    slot: int | None


@dataclasses.dataclass
class Update:
    """One update's logs: its index, environment steps, the live rating per mode (None
    before the fit rates it), the entries per mode in pool order and the logged game share
    and score per entry kind."""

    index: int
    steps: int
    live: dict[str, float | None]
    rows: dict[str, list[Row]]
    game_share: dict[str, float]
    game_seconds: dict[str, float]  # the mean length of the league's games that ended
    lane_games: dict[str, dict[str, float]]  # per mode, the lanes' live games by opponent

    def names(self) -> list[str]:
        """The pool's entries, in pool order."""
        return [row.name for row in self.rows[MODES[0]]]

    def row(self, mode: str, name: str) -> Row | None:
        """`name`'s row in `mode`, or None when it is not in the pool."""
        return next((row for row in self.rows[mode] if row.name == name), None)


def load(run: pathlib.Path) -> list[Update]:
    """Every update of `run`'s `metrics.jsonl`."""
    updates = []
    for line in (run / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        rows: dict[str, list[Row]] = {mode: [] for mode in MODES}
        for mode, name, kind, rating, stderr, share, games, score, slot in record["pool/entries"][
            "rows"
        ]:
            rows[mode].append(Row(name, kind, rating, stderr, share, games, score, slot))
        updates.append(
            Update(
                index=len(updates),
                steps=record["counters/environment_steps"],
                live={mode: record.get(f"rating/live/{mode}") for mode in MODES},
                rows=rows,
                game_share={kind: record.get(f"pool/game_share/{kind}", 0.0) for kind in KINDS},
                game_seconds={
                    mode: record[f"episode/length_seconds/{mode}/league"] for mode in MODES
                },
                lane_games={
                    mode: {
                        key.split("/")[-1][len("live_vs_") :]: value
                        for key, value in record.items()
                        if key.startswith(f"lanes/games/{mode}/live_vs_")
                    }
                    for mode in MODES
                },
            )
        )
    return updates


def checkpoint_states(run: pathlib.Path, updates: list[Update]) -> dict[int, dict]:
    """{update index: the league state} of every training checkpoint of `run`."""
    per_update = updates[0].steps
    out = {}
    for path in sorted((run / "checkpoints").glob("step_*/checkpoint.json")):
        state = json.loads(path.read_text())["league"]
        out[int(path.parent.name[len("step_") :]) // per_update - 1] = state
    return out


# ---- 1. the weights


def live_from_shares(rows: list[Row], floor: float = FLOOR) -> float:
    """The live rating that best reproduces `rows`' logged shares (a golden-section search
    over the summed squared error)."""

    def error(live: float) -> float:
        return float(((weights_at(live, rows, floor) - shares(rows)) ** 2).sum())

    low, high = -1000.0, 4000.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(200):
        a, b = high - ratio * (high - low), low + ratio * (high - low)
        if error(a) < error(b):
            high = b
        else:
            low = a
    return (low + high) / 2.0


def shares(rows: list[Row]) -> torch.Tensor:
    """The logged sampling shares, (K,) float64."""
    return torch.tensor([row.share for row in rows], dtype=torch.float64)


def weights_at(live: float, rows: list[Row], floor: float = FLOOR) -> torch.Tensor:
    """The league's sampling weights (`sampling_weights`) at live rating `live` and the
    rows' ratings."""
    ratings = torch.tensor([row.rating for row in rows], dtype=torch.float64)
    probability = win_probability(torch.tensor(live, dtype=torch.float64), ratings)
    return sampling_weights(probability, floor)


def check_weights(updates: list[Update]) -> dict:
    """The largest difference between the logged shares and the recomputed ones, where the
    live rating is logged and where it is solved for; fills in the solved live ratings."""
    logged, solved = 0.0, 0.0
    for update in updates:
        for mode in MODES:
            rows = update.rows[mode]
            if update.live[mode] is None:
                update.live[mode] = live_from_shares(rows)
                difference = (weights_at(update.live[mode], rows) - shares(rows)).abs().max()
                solved = max(solved, float(difference))
            else:
                difference = (weights_at(update.live[mode], rows) - shares(rows)).abs().max()
                logged = max(logged, float(difference))
    return dict(max_difference_logged_live=logged, max_difference_solved_live=solved)


# ---- 2. the draws


def mixed_weights(update: Update) -> torch.Tensor:
    """The slots' draw weights after `update`: the modes' shares mixed by their shares of the
    opponent rows (equal in the mixed preset)."""
    return sum(shares(update.rows[mode]) for mode in MODES) / len(MODES)


def host_draw(key: int, swaps: int, weights: torch.Tensor) -> int:
    """The entry index the league's `_draw` takes at swap counter `swaps`."""
    uniform = counter_uniform(key, torch.tensor(swaps), torch.zeros(1, dtype=torch.int64))
    return int(draw(weights, uniform.double())[0])


def logged_view(update: Update) -> list[int | None]:
    """The logged slot column of `update` (frontline's; both modes log the same)."""
    return [row.slot for row in update.rows[MODES[0]]]


def view_of(held: list[str | None], names: list[str]) -> list[int | None]:
    """The slot column `League.metrics` logs for `held`: each entry's first slot."""
    return [held.index(name) if name in held else None for name in names]


@dataclasses.dataclass
class Timeline:
    """The replayed slots: `held[u]` the entries of the slots and the incoming set after
    update u; `drains` (begin, end, slot) per drain (end None while it runs at the end of the
    logs); `draws` (update, swap counter, entry) per replayed draw."""

    held: list[list[str | None]]
    drains: list[tuple[int, int | None, int]]
    draws: list[tuple[int, int, str]]
    matched_updates: int
    checked_checkpoints: int
    matched_checkpoints: int


def replay_draws(updates: list[Update], states: dict[int, dict], seed: int = 0) -> Timeline:
    """Replay the host's draws from the first checkpoint on (its `held` and `swaps` hold the
    start's draws, whose weights are not logged), finishing a drain at the update whose
    logged slot column needs it, and check every update after and every checkpoint."""
    key = stream_key(seed, "league.swap")
    first = min(states)
    held = list(states[first]["held"])
    swaps, next_slot = states[first]["swaps"], states[first]["next_slot"]
    draining = (next_slot - 1) % SLOTS
    timeline = Timeline([], [(0, None, draining)], [], 0, 0, 0)
    for update in updates:
        if update.index > first:
            names = update.names()
            if view_of(held, names) != logged_view(update):
                # the drain finished during this update and the next began, drawn from the
                # weights after the previous update
                held[draining], held[SLOTS] = held[SLOTS], None
                begin, _, slot = timeline.drains[-1]
                timeline.drains[-1] = (begin, update.index, slot)
                draining, next_slot = next_slot, (next_slot + 1) % SLOTS
                index = host_draw(key, swaps, mixed_weights(updates[update.index - 1]))
                held[SLOTS] = updates[update.index - 1].names()[index]
                timeline.draws.append((update.index, swaps, held[SLOTS]))
                timeline.drains.append((update.index, None, draining))
                swaps += 1
            timeline.matched_updates += view_of(held, names) == logged_view(update)
        if update.index in states:
            state = states[update.index]
            timeline.checked_checkpoints += 1
            timeline.matched_checkpoints += state["held"] == held and state["swaps"] == swaps
        timeline.held.append(list(held))
    # before the first checkpoint the slots held what it holds (the logged column agrees)
    for index in range(first):
        timeline.held[index] = list(states[first]["held"])
    return timeline


# ---- 3. the snapshots' ratings


def fit_ladder_724153e7(
    scored: torch.Tensor, fixed: torch.Tensor, fixed_ratings: torch.Tensor, prior_games: float
) -> torch.Tensor:
    """The gauge's ladder fit as run commit 724153e7 had it (`bnb.league.ratings.fit_ladder`
    there, gone since), on the scored matrix (K, K) (wins plus half the ties of row over
    column): the `fixed` players keep `fixed_ratings`, the others are Newton's maximum from 0
    with `prior_games` virtual games, half won, against the strongest fixed rating. Kept here
    only to reproduce that run's logs."""
    count = scored.shape[0]
    strongest = float(fixed_ratings[fixed].max())
    padded = torch.zeros((count + 1, count + 1), dtype=torch.float64)
    padded[:count, :count] = scored
    free = torch.cat([~fixed, torch.zeros(1, dtype=torch.bool)])
    padded[:count, count][~fixed] = prior_games / 2.0
    padded[count, :count][~fixed] = prior_games / 2.0
    theta = torch.zeros(count + 1, dtype=torch.float64)
    theta[:count][fixed] = fixed_ratings[fixed] / SCALE
    theta[count] = strongest / SCALE
    theta = _maximise(theta, padded, free)
    ratings = theta[:count] * SCALE
    ratings[fixed] = fixed_ratings[fixed]
    return ratings


def snapshot_maximum(
    won: torch.Tensor, lost: torch.Tensor, opponents: torch.Tensor, strongest: float
) -> float:
    """The true maximum of the same likelihood for one snapshot (its credited scored games
    `won`, `lost` (K,) against opponents rated `opponents` (K,), and the prior's half game
    each way at `strongest`), by bisection on the derivative, which falls monotonically."""
    opponents = torch.cat([opponents, torch.tensor([strongest], dtype=torch.float64)])
    won = torch.cat([won, torch.tensor([0.5], dtype=torch.float64)])
    lost = torch.cat([lost, torch.tensor([0.5], dtype=torch.float64)])

    def slope(rating: float) -> float:
        probability = win_probability(torch.tensor(rating, dtype=torch.float64), opponents)
        return float((won * (1 - probability) - lost * probability).sum())

    low, high = -5000.0, 50000.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if slope(middle) > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


@dataclasses.dataclass
class SnapshotFits:
    """Per (update, mode, snapshot): the logged rating, the replayed `fit_ladder_724153e7`
    rating (None where the fit held the last rating) and the true maximum."""

    logged: dict[tuple[int, str, str], float]
    replayed: dict[tuple[int, str, str], float | None]
    maximum: dict[tuple[int, str, str], float]
    information: dict[tuple[str, str], float]  # the credited games' Fisher information

    def worst_difference(self) -> float:
        """The largest |replayed - logged| where the replay updated the rating."""
        return max(
            (abs(value - self.logged[key]) for key, value in self.replayed.items() if value),
            default=0.0,
        )


def replay_snapshot_fits(updates: list[Update]) -> SnapshotFits:
    """Replay the run's snapshot ratings: a snapshot taken at update a is credited with
    update a's games against the ladder (`Gauge.credit`: the averages left out), then every
    later update fits every snapshot jointly at that update's ladder ratings."""
    taken = {}  # snapshot name -> the update it was taken at
    for update in updates:
        for row in update.rows[MODES[0]]:
            if row.kind == "checkpoint" and row.name not in taken:
                taken[row.name] = update.index
    fits = SnapshotFits({}, {}, {}, {})
    last: dict[tuple[str, str], float] = {}
    for update in updates:
        for mode in MODES:
            ladder = [row for row in update.rows[mode] if row.kind in LADDER_KINDS]
            snapshots = [
                row
                for row in update.rows[mode]
                if row.kind == "checkpoint" and taken[row.name] < update.index
            ]
            for row in update.rows[mode]:
                if row.kind == "checkpoint" and taken[row.name] == update.index:
                    last[(mode, row.name)] = row.rating  # at the live rating
            if not snapshots:
                continue
            players = [row.name for row in ladder] + [row.name for row in snapshots]
            count, rungs = len(players), len(ladder)
            scored = torch.zeros((count, count), dtype=torch.float64)
            for position, snapshot in enumerate(snapshots, start=rungs):
                credit = updates[taken[snapshot.name]]
                for column, rung in enumerate(ladder):
                    games = credit.row(mode, rung.name)
                    if games is not None and games.games:
                        scored[position, column] = games.games * games.score
                        scored[column, position] = games.games * (1 - games.score)
            fixed = torch.tensor([True] * rungs + [False] * len(snapshots))
            fixed_ratings = torch.tensor(
                [row.rating for row in ladder] + [0.0] * len(snapshots), dtype=torch.float64
            )
            played = (scored + scored.T).sum(1) > 0
            keep = played | fixed
            fitted = fit_ladder_724153e7(
                scored[keep][:, keep], fixed[keep], fixed_ratings[keep], prior_games=1.0
            )
            kept = [name for name, used in zip(players, keep.tolist()) if used]
            finite = bool(torch.isfinite(fitted).all())
            strongest = float(fixed_ratings[fixed].max())
            opponents = fixed_ratings[:rungs]
            for snapshot in snapshots:
                name, key = snapshot.name, (update.index, mode, snapshot.name)
                fits.logged[key] = snapshot.rating
                value = fitted[kept.index(name)].item() if finite and name in kept else None
                fits.replayed[key] = value
                if value is not None:
                    last[(mode, name)] = value
                position = players.index(name)
                won, lost = scored[position, :rungs], scored[:rungs, position]
                fits.maximum[key] = snapshot_maximum(won, lost, opponents, strongest)
                probability = win_probability(torch.tensor(fits.maximum[key]), opponents)
                fits.information[(mode, name)] = float(
                    ((won + lost) * probability * (1 - probability)).sum() + 0.25
                )
    return fits


# ---- 4. residency to games


def place(
    out: dict[str, dict[str, float]],
    name: str | None,
    share: float | dict[str, float],
    ladder_weights: dict[str, dict[str, float]],
) -> None:
    """Add a set holding `name` on `share` of every mode's opponent rows (or per mode) to
    `out`: a ladder entry plays the rungs by its mode's `ladder_weights`."""
    if name is None:
        return
    for mode in MODES:
        part = share[mode] if isinstance(share, dict) else share
        if part == 0.0:
            continue
        if name in ladder_weights[mode]:
            for rung, weight in ladder_weights[mode].items():
                out[mode][rung] = out[mode].get(rung, 0.0) + part * weight
        else:
            out[mode][name] = out[mode].get(name, 0.0) + part


def residency(
    held: list[str | None],
    drain_fraction: float,
    draining: int,
    ladder_weights: dict[str, dict[str, float]],
    slots: int = SLOTS,
    incoming_fraction: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Per mode, each entry's share of the opponent rows: each slot a 1/slots share, the
    draining slot's `drain_fraction` moved to the incoming set (`held[slots]`); a set
    holding a ladder entry plays the rungs by its mode's `ladder_weights` (each mode's
    weights over the ladder, normalised). incoming_fraction: the incoming set's share per
    mode where it is not `drain_fraction` (the tally's: rows in their first game against it
    have ended none)."""
    out: dict[str, dict[str, float]] = {mode: {} for mode in MODES}
    for slot in range(slots):
        own = (1.0 - drain_fraction) if slot == draining else 1.0
        place(out, held[slot], own / slots, ladder_weights)
    incoming = incoming_fraction or dict.fromkeys(MODES, drain_fraction)
    place(out, held[slots], {mode: incoming[mode] / slots for mode in MODES}, ladder_weights)
    return out


def ladder_weights_of(weights: dict[str, dict[str, float]], ladder: list[str]):
    """Each mode's weights over the ladder entries, normalised."""
    out = {}
    for mode in MODES:
        total = sum(weights[mode][name] for name in ladder)
        out[mode] = {name: weights[mode][name] / total for name in ladder}
    return out


def kind_of(update: Update, name: str) -> str:
    """The logged kind of entry `name`."""
    return update.row(MODES[0], name).kind


def drain_fraction(update: float, begin: int, end: int | None, length: float) -> float:
    """The share of the draining slot's rows switched by the middle of `update`, taken as
    rising linearly over the drain (`length` updates when it has not ended); 0 before it."""
    span = (end - begin) if end is not None else length
    return min(1.0, max(0.0, (update - begin + 0.5) / span))


def reconstructed_residency(
    updates: list[Update], timeline: Timeline, lags: dict[str, dict[str, float]] | None = None
) -> list[dict]:
    """Per update u, `residency` of the replayed slots after update u - 1 with the drain
    running in u (a drain that ends in u is taken as complete), the rungs by the weights
    after u - 1. lags: each mode and class's mean game in updates, for the rows the tally
    reflects: a row's games against the incoming entry are tallied from one game after it
    switched, its outgoing entry's up to the switch."""
    lengths = [end - begin for begin, end, _ in timeline.drains if end is not None]
    length = statistics.mean(lengths[1:]) if len(lengths) > 1 else 14.0
    ladder = [row.name for row in updates[0].rows[MODES[0]] if row.kind in LADDER_KINDS]
    out = []
    for update in updates:
        u = update.index
        begin, end, slot = next(
            drain
            for drain in timeline.drains
            if (drain[0] < u or drain[0] == u == 0) and (drain[1] is None or u <= drain[1])
        )
        previous = updates[max(u - 1, 0)]
        weights = {mode: {row.name: row.share for row in previous.rows[mode]} for mode in MODES}
        held = timeline.held[max(u - 1, 0)]
        incoming = None
        if lags is not None:
            kind = class_of(update, held[SLOTS])
            incoming = {
                mode: drain_fraction(u - lags[mode][kind], begin, end, length) for mode in MODES
            }
        out.append(
            residency(
                held,
                drain_fraction(u, begin, end, length),
                slot,
                ladder_weights_of(weights, ladder),
                incoming_fraction=incoming,
            )
        )
    return out


EPISODE_CAP_SECONDS = 300.0  # the run's game clock: no game outlasts it


def seconds_per_update(timeline: Timeline) -> float:
    """Game seconds per update: a drain ends when the last of its rows resets, about one
    game clock after it began, so the cap over the completed drains' mean length (the first
    left out: it began with the run)."""
    lengths = [end - begin for begin, end, _ in timeline.drains if end is not None][1:]
    return EPISODE_CAP_SECONDS / statistics.mean(lengths)


def class_of(update: Update, name: str) -> str:
    """ "ladder" or "network": the opponent class an entry's games are timed by."""
    return "ladder" if kind_of(update, name) in LADDER_KINDS else "network"


def game_lags(
    rates: dict[str, dict[str, float]], rows: dict[str, int], seconds: float
) -> dict[str, dict[str, float]]:
    """Each class's mean game, in updates, from its rate: a mode's `rows` opponent rows on
    one class end rate games per update, so a game lasts rows / rate updates."""
    return {mode: {k: rows[mode] / rate for k, rate in rates[mode].items()} for mode in MODES}


def opponent_rows(run: pathlib.Path) -> dict[str, int]:
    """Each mode's opponent rows: the environments times the share of the pools whose second
    seat is the league's (`config.json`)."""
    config = json.loads((run / "config.json").read_text())
    rows = {}
    for pool in config["pools"]:
        if "opponents" in pool["seats"]:
            mode = pool["name"].split("/")[0]
            rows[mode] = rows.get(mode, 0) + round(config["environments"] * pool["share"])
    return rows


def fit_rates(updates: list[Update], rows_by_update: list[dict]) -> dict[str, dict[str, float]]:
    """Games per unit of row share per update, by mode and class ("ladder" or "network"),
    pooled over the run: the logged games over the reconstructed residency."""
    games = {mode: {"ladder": 0.0, "network": 0.0} for mode in MODES}
    rows = {mode: {"ladder": 0.0, "network": 0.0} for mode in MODES}
    for update, resident in zip(updates, rows_by_update):
        for mode in MODES:
            for row in update.rows[mode]:
                kind = "ladder" if row.kind in LADDER_KINDS else "network"
                games[mode][kind] += row.games
                rows[mode][kind] += resident[mode].get(row.name, 0.0)
    return {mode: {k: games[mode][k] / rows[mode][k] for k in games[mode]} for mode in MODES}


def kind_shares(update: Update, resident: dict, rates: dict) -> dict[str, float]:
    """The predicted game share by entry kind of one update (both modes pooled)."""
    totals = dict.fromkeys(KINDS, 0.0)
    for mode in MODES:
        for name, share in resident[mode].items():
            kind = kind_of(update, name)
            rate = rates[mode]["ladder" if kind in LADDER_KINDS else "network"]
            totals[kind] += share * rate
    total = sum(totals.values())
    return {kind: value / total for kind, value in totals.items()}


# ---- the samplers


def probabilities(update: Update, mode: str, ratings: dict[str, float]) -> torch.Tensor:
    """P(live beats each entry) in pool order, at the live rating and `ratings`."""
    values = torch.tensor([ratings[name] for name in update.names()], dtype=torch.float64)
    return win_probability(torch.tensor(update.live[mode], dtype=torch.float64), values)


def today(p: torch.Tensor, update: Update) -> torch.Tensor:
    """The league's rule: p (1 - p) plus the floor spread over every entry."""
    return sampling_weights(p, FLOOR)


def no_floor_over_beaten(p: torch.Tensor, update: Update) -> torch.Tensor:
    """p (1 - p), with the floor spread only over the entries the gauge does not call
    beaten (p < BEATEN)."""
    weight = p * (1 - p)
    open_ = (p < BEATEN).double()
    weight = (1 - FLOOR) * weight / weight.sum()
    if open_.sum() > 0:
        weight = weight + FLOOR * open_ / open_.sum()
    return weight / weight.sum()


def squared_loss_rate(p: torch.Tensor, update: Update) -> torch.Tensor:
    """(1 - p)^2: the entries the live policy loses to most."""
    weight = (1 - p) ** 2
    return weight / weight.sum()


def fixed_share(p: torch.Tensor, update: Update, share: float = 0.4) -> torch.Tensor:
    """`share` of the weight spread evenly over the NEWEST snapshots and the two averages,
    the rest by today's rule over the other entries."""
    rows = update.rows[MODES[0]]
    snapshots = [i for i, row in enumerate(rows) if row.kind == "checkpoint"][-NEWEST:]
    chosen = snapshots + [i for i, row in enumerate(rows) if row.kind == "average"]
    mask = torch.zeros(len(rows), dtype=torch.bool)
    mask[chosen] = True
    out = torch.zeros(len(rows), dtype=torch.float64)
    out[mask] = share / int(mask.sum())
    out[~mask] = (1 - share) * sampling_weights(p[~mask], FLOOR)
    return out


def ladder_as_one(p: torch.Tensor, update: Update) -> torch.Tensor:
    """Today's rule with the ladder as one entry: its weight the largest p (1 - p) among the
    rungs, split over the rungs by their own p (1 - p) (a ladder slot's rows draw by it)."""
    rows = update.rows[MODES[0]]
    ladder = torch.tensor([row.kind in LADDER_KINDS for row in rows])
    weight = p * (1 - p)
    others = weight[~ladder]
    one = weight[ladder].max()
    count = int((~ladder).sum()) + 1
    total = float(others.sum() + one)
    out = torch.zeros(len(rows), dtype=torch.float64)
    out[~ladder] = (1 - FLOOR) * others / total + FLOOR / count
    out[ladder] = (
        ((1 - FLOOR) * one / total + FLOOR / count) * weight[ladder] / weight[ladder].sum()
    )
    return out


def nash(p: torch.Tensor, update: Update, ratings_of=None, iterations: int = 4000) -> torch.Tensor:
    """The pool's Nash mixture: the symmetric zero-sum game among the entries with payoff
    P(i beats j) - 1/2 from their ratings (fitted to the recorded games), solved by
    multiplicative weights (its averaged play)."""
    ratings = ratings_of
    payoff = torch.sigmoid((ratings[:, None] - ratings[None, :]) / SCALE) - 0.5
    count = ratings.shape[0]
    mixture = torch.full((count,), 1.0 / count, dtype=torch.float64)
    average = torch.zeros(count, dtype=torch.float64)
    rate = 0.5
    for _ in range(iterations):
        gain = payoff @ mixture
        mixture = mixture * torch.exp(rate * gain)
        mixture = mixture / mixture.sum()
        average += mixture
    return average / iterations


SAMPLERS = {
    "today": today,
    "no floor over beaten": no_floor_over_beaten,
    "(1 - p)^2": squared_loss_rate,
    "fixed 40% newest + averages": fixed_share,
    "ladder as one entry": ladder_as_one,
    "Nash over the pool": nash,
}


@dataclasses.dataclass
class Inputs:
    """The reconstructed inputs per update: the gauge's ratings per mode (as run, and with
    the snapshots repaired), the "true" probability that live beats each entry, each entry's
    rating error (natural units) and the games per update per mode."""

    gauge: list[dict[str, dict[str, float]]]
    repaired: list[dict[str, dict[str, float]]]
    truth: list[dict[str, dict[str, float]]]
    sigma: list[dict[str, dict[str, float]]]
    games: dict[str, float]


def reconstruct_inputs(updates: list[Update], fits: SnapshotFits, window: int = 10) -> Inputs:
    """`Inputs` from the logs: the repaired snapshot ratings are the likelihood's true maxima
    (the newest snapshot at its live rating until its first fit); the truth is the observed
    score over the last LADDER_WINDOW updates (a ladder entry; the whole run's when they
    hold fewer than `window` games) or the last `window` updates (an average, at least
    `window` games), with half a game each way, else the gauge's prediction; a snapshot's
    is the live rating at its update against the live rating now."""
    taken: dict[str, tuple[int, dict[str, float]]] = {}
    pooled: dict[tuple[str, str], list[float]] = {}
    for update in updates:
        for mode in MODES:
            for row in update.rows[mode]:
                if row.kind in LADDER_KINDS and row.games:
                    total = pooled.setdefault((mode, row.name), [0.0, 0.0])
                    total[0] += row.games
                    total[1] += row.games * row.score
    gauge, repaired, truth, sigma = [], [], [], []
    for update in updates:
        g, r, t, s = {}, {}, {}, {}
        for mode in MODES:
            g[mode], r[mode], t[mode], s[mode] = {}, {}, {}, {}
            live = update.live[mode]
            for row in update.rows[mode]:
                g[mode][row.name] = row.rating
                r[mode][row.name] = row.rating
                predicted = float(win_probability(torch.tensor(live), torch.tensor(row.rating)))
                t[mode][row.name] = predicted
                s[mode][row.name] = row.stderr / SCALE
                if row.kind == "checkpoint":
                    if row.name not in taken:
                        taken[row.name] = (update.index, dict(update.live))
                    key = (update.index, mode, row.name)
                    r[mode][row.name] = fits.maximum.get(key, taken[row.name][1][mode])
                    then = taken[row.name][1][mode]
                    t[mode][row.name] = float(
                        win_probability(torch.tensor(live), torch.tensor(then))
                    )
                    information = fits.information.get((mode, row.name), 0.25)
                    s[mode][row.name] = 1.0 / math.sqrt(information)
                elif row.kind in LADDER_KINDS and (mode, row.name) in pooled:
                    recent = updates[max(0, update.index - LADDER_WINDOW + 1) : update.index + 1]
                    games = sum(u.row(mode, row.name).games for u in recent)
                    won = sum(
                        u.row(mode, row.name).games * (u.row(mode, row.name).score or 0.0)
                        for u in recent
                    )
                    if games < window:  # too few lately: the whole run's
                        games, won = pooled[(mode, row.name)]
                    t[mode][row.name] = (won + 0.5) / (games + 1.0)
                elif row.kind == "average":
                    recent = updates[max(0, update.index - window + 1) : update.index + 1]
                    games = sum(u.row(mode, row.name).games for u in recent)
                    won = sum(
                        u.row(mode, row.name).games * (u.row(mode, row.name).score or 0.0)
                        for u in recent
                    )
                    if games >= window:
                        t[mode][row.name] = (won + 0.5) / (games + 1.0)
        gauge.append(g)
        repaired.append(r)
        truth.append(t)
        sigma.append(s)
    games = {
        mode: statistics.mean(sum(row.games for row in u.rows[mode]) for u in updates)
        for mode in MODES
    }
    return Inputs(gauge, repaired, truth, sigma, games)


def sampler_weights(
    sampler, updates: list[Update], ratings: list[dict[str, dict[str, float]]]
) -> list[dict[str, dict[str, float]]]:
    """Per update and mode, the sampler's weights by entry from the given gauge ratings."""
    out = []
    for update, rated in zip(updates, ratings):
        per_mode = {}
        for mode in MODES:
            p = probabilities(update, mode, rated[mode])
            if sampler is nash:
                values = torch.tensor(
                    [rated[mode][name] for name in update.names()], dtype=torch.float64
                )
                weight = nash(p, update, ratings_of=values)
            else:
                weight = sampler(p, update)
            per_mode[mode] = dict(zip(update.names(), weight.tolist()))
        out.append(per_mode)
    return out


def simulate(
    updates: list[Update],
    weights: list[dict[str, dict[str, float]]],
    seeds: int,
    slots: int = SLOTS,
    drain: int = 14,
    concurrent: bool = False,
    instant: bool = False,
) -> list[list[dict[str, dict[str, float]]]]:
    """Per seed and update, each mode's row share by entry under the league's slots and
    drains: `slots` slots, each drawn from the modes' mixed weights of the previous update,
    drained one at a time round robin over `drain` updates (all at once when `concurrent`),
    the draining rows switching linearly; a ladder slot plays its mode's ladder weights.
    `instant`: every row draws its own entry at every
    reset from its own mode's weights (the limit of many slots), one seed."""
    ladder = [row.name for row in updates[0].rows[MODES[0]] if row.kind in LADDER_KINDS]
    results = []
    for seed in range(1 if instant else seeds):
        generator = torch.Generator().manual_seed(seed)

        def pick(update_weights: dict[str, dict[str, float]]) -> str:
            names = list(update_weights[MODES[0]])
            mixed = torch.tensor(
                [sum(update_weights[m][n] for m in MODES) / len(MODES) for n in names],
                dtype=torch.float64,
            )
            uniform = torch.rand(1, generator=generator, dtype=torch.float64)
            return names[int(draw(mixed, uniform)[0])]

        held = [pick(weights[0]) for _ in range(slots)]
        incoming = [pick(weights[0]) for _ in range(slots if concurrent else 1)]
        next_slot, begin = 0, 0
        per_update = []
        for update in updates:
            previous = weights[max(update.index - 1, 0)]
            if update.index - begin >= drain:  # the drain ends, the next begins
                if concurrent:
                    held = incoming
                    incoming = [pick(previous) for _ in range(slots)]
                else:
                    held[next_slot] = incoming[0]
                    next_slot = (next_slot + 1) % slots
                    incoming = [pick(previous)]
                begin = update.index
            fraction = (update.index - begin + 0.5) / drain
            ladder_weights = ladder_weights_of(previous, ladder)
            if instant:
                resident = {mode: dict(previous[mode]) for mode in MODES}
            elif concurrent:
                resident = {mode: {} for mode in MODES}
                for slot in range(slots):
                    place(resident, held[slot], (1.0 - fraction) / slots, ladder_weights)
                    place(resident, incoming[slot], fraction / slots, ladder_weights)
            else:
                resident = residency(held + incoming, fraction, next_slot, ladder_weights, slots)
            per_update.append(resident)
        results.append(per_update)
    return results


@dataclasses.dataclass
class Score:
    """A sampler's late-run scores, (mean, 10th, 90th percentile) over seeds: the game and
    the step (row) share against open entries, the learning signal, the rating information,
    the checkpoints' and the ladder's game shares (`score`)."""

    open_games: tuple[float, float, float]
    open_steps: tuple[float, float, float]
    signal: tuple[float, float, float]
    rating_games: tuple[float, float, float]
    checkpoint_games: tuple[float, float, float]
    ladder_games: tuple[float, float, float]


def games_of(rows: dict, rates: dict, update: Update) -> dict:
    """Each mode's games per update by entry: row share times its class's rate."""
    return {
        mode: {
            name: share * rates[mode][class_of(update, name)] for name, share in rows[mode].items()
        }
        for mode in MODES
    }


def score(
    updates: list[Update],
    inputs: Inputs,
    rows: list[list[dict]],
    rates: dict,
    games: list[list[dict]] | None = None,
    late: int = LATE,
) -> Score:
    """Per seed, over the late updates. An entry is open when the live policy beats it less
    than BEATEN of the time (the "true" p). The open share of the games (rows times the
    class's rate, or `games` as logged) and of the steps (rows: what the learner trains on);
    the learning signal, sum over the steps of p (1 - p) (at most 1/4); the information on
    live's rating from the training games, as even games against a known opponent per update
    and mode (sum over entries of n w / (1 + n w sigma^2), n the games, w = p (1 - p), sigma
    the entry's rating error); the checkpoints' and the ladder's game shares."""
    keys = ("open_games", "open_steps", "signal", "rating", "checkpoint", "ladder")
    values = {key: [] for key in keys}
    for seed, per_update in enumerate(rows):
        series = {key: [] for key in keys}
        for update, by_mode in zip(updates, per_update):
            if update.index < late:
                continue
            played = games[seed][update.index] if games else games_of(by_mode, rates, update)
            game_total = sum(sum(played[mode].values()) for mode in MODES)
            row_total = sum(sum(by_mode[mode].values()) for mode in MODES)
            point = dict.fromkeys(keys, 0.0)
            for mode in MODES:
                for name, value in by_mode[mode].items():
                    p = inputs.truth[update.index][mode][name]
                    point["open_steps"] += value / row_total * (p < BEATEN)
                    point["signal"] += value / row_total * p * (1 - p)
                for name, n in played[mode].items():
                    p = inputs.truth[update.index][mode][name]
                    w, kind = p * (1 - p), kind_of(update, name)
                    point["open_games"] += n / game_total * (p < BEATEN)
                    point["checkpoint"] += n / game_total * (kind == "checkpoint")
                    point["ladder"] += n / game_total * (kind in LADDER_KINDS)
                    sigma = inputs.sigma[update.index][mode][name]
                    if math.isfinite(sigma):
                        point["rating"] += n * w / (1 + n * w * sigma**2) / len(MODES) / 0.25
            for key in keys:
                series[key].append(point[key])
        for key in keys:
            values[key].append(statistics.mean(series[key]))

    def spread(series: list[float]) -> tuple[float, float, float]:
        ordered = sorted(series)
        low = ordered[int(0.1 * (len(ordered) - 1))]
        high = ordered[int(0.9 * (len(ordered) - 1))]
        return (statistics.mean(series), low, high)

    return Score(*(spread(values[key]) for key in keys))


def lane_information(
    updates: list[Update], inputs: Inputs, timeline: Timeline, late: int = LATE
) -> float:
    """The rating lanes' information on live's rating per update and mode over the late
    updates, in `score`'s units: their live games against the scripted agent, the
    exponential average and a league slot holding a snapshot (the run's lanes left a slot
    holding anything else unrated)."""
    values = []
    for update in updates[late:]:
        information = 0.0
        held = timeline.held[update.index]
        for mode in MODES:
            for opponent, n in update.lane_games[mode].items():
                if opponent.startswith("league_"):
                    opponent = held[int(opponent[len("league_") :])]
                    if update.row(mode, opponent).kind != "checkpoint":
                        continue
                if opponent == "scripted":
                    sigma = 0.0
                else:
                    sigma = inputs.sigma[update.index][mode][opponent]
                p = inputs.truth[update.index][mode][opponent]
                w = p * (1 - p)
                if math.isfinite(sigma):
                    information += n * w / (1 + n * w * sigma**2) / len(MODES) / 0.25
        values.append(information)
    return statistics.mean(values)


def logged_games(updates: list[Update]) -> list[list[dict]]:
    """The logged games per update, mode and entry, in `simulate`'s layout (one "seed")."""
    return [
        [
            {mode: {row.name: row.games for row in u.rows[mode] if row.games} for mode in MODES}
            for u in updates
        ]
    ]


# ---- the report


def window_shares(updates, predicted, start, stop) -> dict[str, tuple[float, float]]:
    """(predicted, logged) game share by kind over updates [start, stop), pooled by games."""
    out = {}
    for kind in KINDS:
        logged = statistics.mean(u.game_share[kind] for u in updates[start:stop])
        guess = statistics.mean(p[kind] for p in predicted[start:stop])
        out[kind] = (guess, logged)
    return out


def main() -> None:
    """Run every part on a run directory and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    updates = load(arguments.run)
    states = checkpoint_states(arguments.run, updates)
    report: dict = {}

    report["weights"] = check_weights(updates)
    print("1. weights:", report["weights"])

    timeline = replay_draws(updates, states)
    report["draws"] = dict(
        draws=timeline.draws,
        drains=timeline.drains,
        matched_updates=timeline.matched_updates,
        updates_after_first_checkpoint=len(updates) - 1 - min(states),
        checkpoints=(timeline.matched_checkpoints, timeline.checked_checkpoints),
    )
    print("2. draws:", report["draws"])

    fits = replay_snapshot_fits(updates)
    report["snapshot_fits"] = dict(
        worst_difference=fits.worst_difference(),
        fits=len(fits.replayed),
        held=sum(value is None for value in fits.replayed.values()),
        logged_median=statistics.median(fits.logged.values()),
        maximum_median=statistics.median(fits.maximum.values()),
        logged_range=(min(fits.logged.values()), max(fits.logged.values())),
        maximum_range=(min(fits.maximum.values()), max(fits.maximum.values())),
    )
    print("3. snapshot fits:", report["snapshot_fits"])

    playing = reconstructed_residency(updates, timeline)
    seconds = seconds_per_update(timeline)
    rows = opponent_rows(arguments.run)
    rates = fit_rates(updates, playing)
    for _ in range(5):  # the lags from the rates, the rates on the lagged rows
        lags = game_lags(rates, rows, seconds)
        resident = reconstructed_residency(updates, timeline, lags)
        rates = fit_rates(updates, resident)
    predicted = [kind_shares(u, r, rates) for u, r in zip(updates, resident)]
    windows = {
        f"{start}-{stop - 1}": window_shares(updates, predicted, start, stop)
        for start, stop in ((0, 32), (32, 64), (64, 96), (LATE, len(updates)))
    }
    absolute = statistics.mean(
        abs(p[kind] - u.game_share[kind]) for u, p in zip(updates, predicted) for kind in KINDS
    )
    report["games"] = dict(
        seconds_per_update=seconds,
        game_seconds={m: {k: v * seconds for k, v in lags[m].items()} for m in MODES},
        rates=rates,
        windows=windows,
        mean_absolute_error=absolute,
    )
    print("4. games: rates", rates, "per-update MAE", round(absolute, 3))
    for window, values in windows.items():
        print("  ", window, {k: (round(a, 3), round(b, 3)) for k, (a, b) in values.items()})

    # the decomposition over the late updates
    late = updates[LATE:]
    weight_kind = {kind: 0.0 for kind in KINDS}
    repaired_kind = {kind: 0.0 for kind in KINDS}
    inputs = reconstruct_inputs(updates, fits)
    repaired_today = sampler_weights(today, updates, inputs.repaired)
    for update in late:
        previous = updates[update.index - 1]
        mixed = mixed_weights(previous)
        for name, value in zip(previous.names(), mixed.tolist()):
            weight_kind[kind_of(previous, name)] += value / len(late)
        for name in previous.names():
            value = sum(repaired_today[previous.index][m][name] for m in MODES) / len(MODES)
            repaired_kind[kind_of(previous, name)] += value / len(late)
    rows_kind = {kind: 0.0 for kind in KINDS}
    for update, r in zip(late, playing[LATE:]):
        for mode in MODES:
            for name, value in r[mode].items():
                rows_kind[kind_of(update, name)] += value / len(MODES) / len(late)
    floor_ladder = (
        FLOOR
        * sum(1 for row in late[-1].rows[MODES[0]] if row.kind in LADDER_KINDS)
        / len(late[-1].rows[MODES[0]])
    )
    report["decomposition"] = dict(
        draw_weight=weight_kind,
        draw_weight_repaired=repaired_kind,
        row_share=rows_kind,
        games_predicted=window_shares(updates, predicted, LATE, len(updates)),
        floor_on_ladder=floor_ladder,
    )
    print("decomposition (late):", report["decomposition"])

    # the samplers, each on the run's ratings and on the repaired ones
    table = {}
    table["logged"] = dataclasses.asdict(
        score(updates, inputs, [playing], rates, games=logged_games(updates))
    )
    variants = [(name, sampler, {}) for name, sampler in SAMPLERS.items()]
    variants[1:1] = [
        ("today, 8 slots", today, dict(slots=8)),
        ("today, 4 slots draining together", today, dict(concurrent=True)),
        ("today, a draw per row (limit)", today, dict(instant=True)),
    ]
    variants.append(
        ("ladder as one entry, draining together", ladder_as_one, dict(concurrent=True))
    )
    for name, sampler, options in variants:
        for label, ratings in (("as run", inputs.gauge), ("repaired", inputs.repaired)):
            weights = sampler_weights(sampler, updates, ratings)
            rows_played = simulate(updates, weights, arguments.seeds, **options)
            table[f"{name} | {label}"] = dataclasses.asdict(
                score(updates, inputs, rows_played, rates)
            )
    report["samplers"] = table
    report["lane_information"] = lane_information(updates, inputs, timeline)
    print("lanes' information on live (late, per update and mode):", report["lane_information"])
    print("samplers (late; mean [p10, p90]):")
    for name, values in table.items():
        cells = "  ".join(
            f"{key} {mean:.3f} [{low:.3f}, {high:.3f}]" for key, (mean, low, high) in values.items()
        )
        print(f"  {name:48s} {cells}")
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
