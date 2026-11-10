"""The real-scale ship-type tournament: which type wins most as the scripted agent flies it.

    uv run --no-sync python -m bench.ship_type_tournament play --record DIR/batches.jsonl
    uv run --no-sync python -m bench.ship_type_tournament report --record DIR/batches.jsonl \
        --out DIR

`play`: a full round robin over the real-scale set's types in real_scale_deathmatch, each
player one type flown by the scripted agent (T = 1, its real-scale deathmatch set) with
every ship of its team (`bnb.league.matches.ship_type_players`): each pair plays
`--games` games, half with each type on each side, in blocks of `--block-games` games,
`--blocks` blocks a batch, on the league's match engine with the balanced allocation
(`bnb.league.tournament`). Each batch is appended to the record (its matchups, results,
the fit so far and each block's first-game statistics) as it finishes, and a record found
is resumed; the run is bounded by the budget (every pair's games) and `--hours`.

`report`: from a record, the Bradley-Terry ratings (draws half a win, one prior game
against the anchor, the P-51D at 1000) with their standard errors and 95% intervals
(1.96 standard errors, relative to the anchor), each type's score, wins, draws and losses,
kills and deaths a game, kill ratio, shots a game, hit share and mean speed, and every
pair's results: `ratings.csv`, `pairs.csv` and `tournament.json` in `--out`, and the table
as Markdown (printed).

`analyse`: from `--out`'s `tournament.json` and the set's ship files, each handle's and a
few dimensionless groups' correlation with the Elo (Pearson and Spearman, over the types),
and how well a small linear model predicts the Elo: ridge regression on every feature
(standardised; its penalty chosen by leave-one-out) and every one-, two- and
three-feature least-squares model, each scored by its leave-one-out error against the
mean's, and the same searches on the Elo shuffled across the types (what they find in
noise); `analysis.json` in `--out` (with `--exclude`, a suffix naming the types left out),
and the tables as Markdown (printed).

It rates a type as the scripted agent, tuned on the P-51D, flies it: not the type's
ceiling.
"""

import argparse
import csv
import datetime
import itertools
import json
import math
import pathlib
import time

import torch

from bench.lock import bench_lock
from bnb.game import api, ships
from bnb.league import tournament
from bnb.league.matches import STATISTICS, ArenaConfig, ship_type_players
from bnb.league.ratings import fit_ratings

PRESET = "real_scale_deathmatch"
ANCHOR = "P-51D"
PRIOR_GAMES = 1.0
CONFIDENCE = 1.96  # standard errors in a 95% interval


def _clock(seconds: float) -> str:
    later = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=seconds)
    return later.strftime("%H:%M")


def play(args) -> None:
    """The round robin the module describes, appended to the record batch by batch."""
    game = api.PRESETS[PRESET]
    players = ship_type_players(game.ship_set)
    count = len(players)
    pairs = count * (count - 1) // 2
    if args.games % args.block_games or args.block_games % 2:
        raise SystemExit("--games must be whole blocks of an even --block-games")
    blocks = pairs * args.games // args.block_games
    if blocks % args.blocks:
        raise SystemExit(f"{blocks} blocks do not split into batches of {args.blocks}")
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=args.blocks, block_environments=args.block_games),
        target_stderr=1e-9,  # never met: the budget ends it, a full round robin
        budget=pairs * args.games,
        allocation="balanced",
    )
    record = pathlib.Path(args.record)
    started = time.time()
    deadline = started + 3600.0 * args.hours

    def log(line: str) -> None:
        print(f"{datetime.datetime.now():%H:%M:%S} {line}", flush=True)
        if time.time() > deadline:
            raise SystemExit(f"past the {args.hours} h bound: resume from {record}")

    log(
        f"{count} types, {pairs} pairs x {args.games} games = {pairs * args.games} games in "
        f"{blocks // args.blocks} batches of {config.arena.games}; "
        f"bound {_clock(deadline - time.time())}"
    )
    with bench_lock():
        tournament.rate(
            players,
            PRESET,
            torch.device("cuda"),
            config,
            log,
            game,
            record=record,
            anchor=ANCHOR,
            statistics=True,
        )


def _summaries(record: pathlib.Path):
    """From a record: the names, the pair counts (K, K, 3) of row's wins, ties and losses
    against column, and the statistics (K, len(STATISTICS)) and games (K,) per type."""
    lines = [json.loads(line) for line in record.read_text().splitlines()]
    names = [row["player"] for row in lines[0]["ratings"]]
    index = {name: number for number, name in enumerate(names)}
    count = len(names)
    counts = torch.zeros((count, count, 3), dtype=torch.float64)
    statistics = torch.zeros((count, len(STATISTICS)), dtype=torch.float64)
    for line in lines:
        for (first, second), (won, tied, lost), (mine, theirs) in zip(
            line["matchups"], line["results"], line["statistics"], strict=True
        ):
            i, j = index[first], index[second]
            counts[i, j] += torch.tensor([won, tied, lost], dtype=torch.float64)
            counts[j, i] += torch.tensor([lost, tied, won], dtype=torch.float64)
            statistics[i] += torch.tensor(mine, dtype=torch.float64)
            statistics[j] += torch.tensor(theirs, dtype=torch.float64)
    return names, counts, statistics, lines


def report(args) -> None:
    """The tables the module describes, from a record."""
    record, out = pathlib.Path(args.record), pathlib.Path(args.out)
    names, counts, statistics, lines = _summaries(record)
    anchor = names.index(ANCHOR)
    wins = counts[..., 0] + counts[..., 1] / 2
    ratings, stderr = fit_ratings(wins, anchor, PRIOR_GAMES)
    games = counts.sum((1, 2))
    column = {name: statistics[:, number] for number, name in enumerate(STATISTICS)}
    rows = []
    for k, name in enumerate(names):
        won, tied, lost = (float(counts[k, :, outcome].sum()) for outcome in range(3))
        played = float(games[k])
        rows.append(
            dict(
                type=name,
                elo=1000.0 + float(ratings[k]),
                stderr=float(stderr[k]),
                low=1000.0 + float(ratings[k]) - CONFIDENCE * float(stderr[k]),
                high=1000.0 + float(ratings[k]) + CONFIDENCE * float(stderr[k]),
                games=int(played),
                score=(won + tied / 2) / played,
                wins=int(won),
                draws=int(tied),
                losses=int(lost),
                kills_per_game=float(column["kills"][k]) / played,
                deaths_per_game=float(column["deaths"][k]) / played,
                kill_ratio=float(column["kills"][k]) / max(float(column["deaths"][k]), 1.0),
                shots_per_game=float(column["shots"][k]) / played,
                hit_share=float(column["hits"][k]) / max(float(column["shots"][k]), 1.0),
                mean_speed=float(column["speed"][k]) / max(float(column["ship_ticks"][k]), 1.0),
            )
        )
    rows.sort(key=lambda row: -row["elo"])
    pairs = []
    for i, first in enumerate(names):
        for j, second in enumerate(names):
            if i != j:
                won, tied, lost = (int(value) for value in counts[i, j].tolist())
                total = won + tied + lost
                pairs.append(
                    dict(
                        first=first,
                        second=second,
                        wins=won,
                        draws=tied,
                        losses=lost,
                        score=(won + tied / 2) / total if total else math.nan,
                    )
                )
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (("ratings.csv", rows), ("pairs.csv", pairs)):
        with open(out / name, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    total = int(games.sum()) // 2
    meta = dict(
        preset=PRESET,
        anchor=ANCHOR,
        games=total,
        batches=len(lines),
        draws=int(counts[..., 1].sum()) // 2,
        prior_games=PRIOR_GAMES,
        confidence=CONFIDENCE,
    )
    (out / "tournament.json").write_text(
        json.dumps(dict(meta=meta, ratings=rows, pairs=pairs), indent=1) + "\n"
    )
    header = (
        "| Rank | Type | Elo (95% interval) | Score | W / D / L | Kills / deaths a game "
        "| Kill ratio | Shots a game | Hit share | Mean speed m/s |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {rank} | {row['type']} | {row['elo']:.0f} ({row['low']:.0f} to {row['high']:.0f}) "
        f"| {row['score']:.3f} | {row['wins']} / {row['draws']} / {row['losses']} "
        f"| {row['kills_per_game']:.2f} / {row['deaths_per_game']:.2f} "
        f"| {row['kill_ratio']:.2f} | {row['shots_per_game']:.0f} | {row['hit_share']:.3f} "
        f"| {row['mean_speed']:.0f} |\n"
        for rank, row in enumerate(rows, 1)
    )
    print(json.dumps(meta))
    print(header + body, end="")


GRAVITY = 9.81  # m/s^2, the real-scale set's


def features(ship_set: str) -> dict[str, dict[str, float]]:
    """Per type name: the handles that differ between types, and the groups."""
    from bnb.game.real_scale import combat_constants

    fire_over_charge = combat_constants()[0]  # continuous fire's power over the climb charge
    out = {}
    for ship in ships.load(ship_set):
        h = ship.handles
        # the climb rate, from the combat economy: continuous fire costs fire_over_charge
        # times the initial climb charge g x climb_rate
        climb_rate = ship.card["bullet_energy_cost"] * h["fire_rate"] / fire_over_charge / GRAVITY
        weight_of_fire = h["fire_rate"] * h["hit_damage"]  # damage per second
        spread_angle = h["bullet_spread"] / h["muzzle_speed"]  # rad, per axis
        row = dict(
            stall_speed=h["stall_speed"],
            g_limit=h["g_limit"],
            cruise_speed=h["cruise_speed"],
            climb_rate=climb_rate,
            store=h["ceiling"],
            low_speed_acceleration=h["low_speed_acceleration"],
            sustained_turn_rate=math.degrees(h["sustained_turn_rate"]),
            nose_reversal_time=h["nose_reversal_time"],
            stall_angle=h["stall_angle"],
            zero_lift_drag=h["zero_lift_drag"],
            hit_radius=h["hit_radius"],
            max_shield=h["max_shield"],
            fire_rate=h["fire_rate"],
            hit_damage=h["hit_damage"],
            weight_of_fire=weight_of_fire,
            bullet_spread=h["bullet_spread"],
            muzzle_speed=h["muzzle_speed"],
            gun_range=h["gun_range"],
            # dimensionless groups
            speed_range=h["cruise_speed"] / h["stall_speed"],
            turn_load=h["sustained_turn_rate"] * h["cruise_speed"] / GRAVITY,
            climb_over_cruise=climb_rate / h["cruise_speed"],
            shots_to_kill_p51=math.ceil(100.0 / h["hit_damage"]),
            kill_time_ratio=weight_of_fire / 360.0 * 100.0 / h["max_shield"],
            target_over_cone=h["hit_radius"] / (spread_angle * h["gun_range"]),
            gun_range_over_turn_radius=h["gun_range"]
            * h["sustained_turn_rate"]
            / h["cruise_speed"],
        )
        out[ship.name] = row
    return out


def _ranks(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort()
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(len(values), dtype=values.dtype)
    return ranks


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-30))


def _loo_error(x: torch.Tensor, y: torch.Tensor, penalty: float = 0.0) -> float:
    """Leave-one-out root-mean-square error of a linear model with an intercept (ridge
    `penalty` on the slopes, the columns standardised on the training rows)."""
    errors = []
    for left in range(len(y)):
        keep = torch.arange(len(y)) != left
        train, target = x[keep], y[keep]
        if x.shape[1] == 0:  # the mean alone
            errors.append(float(target.mean() - y[left]))
            continue
        mean, scale = train.mean(0), train.std(0).clamp_min(1e-12)
        design = (train - mean) / scale
        slopes = torch.linalg.solve(
            design.T @ design + penalty * torch.eye(x.shape[1], dtype=x.dtype),
            design.T @ (target - target.mean()),
        )
        guess = target.mean() + ((x[left] - mean) / scale) @ slopes
        errors.append(float(guess - y[left]))
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def analyse(args) -> None:
    """The correlations and models the module describes."""
    out = pathlib.Path(args.out)
    rated = json.loads((out / "tournament.json").read_text())["ratings"]
    rated = [row for row in rated if row["type"] not in args.exclude]
    table = features(api.PRESETS[PRESET].ship_set)
    names = [row["type"] for row in rated]
    elo = torch.tensor([row["elo"] for row in rated], dtype=torch.float64)
    keys = list(next(iter(table.values())))
    x = torch.tensor([[table[name][key] for key in keys] for name in names], dtype=torch.float64)
    correlations = {
        key: dict(
            pearson=_pearson(x[:, k], elo),
            spearman=_pearson(_ranks(x[:, k]), _ranks(elo)),
        )
        for k, key in enumerate(keys)
    }
    null = _loo_error(torch.zeros((len(elo), 0), dtype=torch.float64), elo)
    penalties = [0.1 * 2**k for k in range(16)]
    ridge = min((_loo_error(x, elo, p), p) for p in penalties)
    subsets = []
    for size in (1, 2, 3):
        for chosen in itertools.combinations(range(len(keys)), size):
            error = _loo_error(x[:, list(chosen)], elo)
            subsets.append((error, [keys[k] for k in chosen]))
    subsets.sort()
    best = {size: next(s for s in subsets if len(s[1]) == size) for size in (1, 2, 3)}
    # what the best-of-many subset search finds in pure noise: the same search over the Elo
    # shuffled across the types
    generator = torch.Generator().manual_seed(0)
    shuffled, ridge_shuffled = [], []
    for _ in range(args.shuffles):
        noise = elo[torch.randperm(len(elo), generator=generator)]
        ridge_shuffled.append(
            min(_loo_error(x, noise, p) for p in penalties) / _loo_error(x[:, :0], noise)
        )
        shuffled.append(
            min(
                _loo_error(x[:, list(chosen)], noise) / _loo_error(x[:, :0], noise)
                for size in (1, 2)
                for chosen in itertools.combinations(range(len(keys)), size)
            )
        )
    result = dict(
        types=len(names),
        elo_spread=float(elo.std()),
        null_loo_rmse=null,
        ridge=dict(loo_rmse=ridge[0], penalty=ridge[1]),
        best_subsets={
            str(size): dict(loo_rmse=error, features=chosen)
            for size, (error, chosen) in best.items()
        },
        excluded=list(args.exclude),
        shuffled_best_ratio=sorted(shuffled),
        shuffled_ridge_ratio=sorted(ridge_shuffled),
        correlations=correlations,
    )
    suffix = "-without-" + "-".join(args.exclude).lower().replace(" ", "") if args.exclude else ""
    (out / f"analysis{suffix}.json").write_text(json.dumps(result, indent=1) + "\n")
    lines = [
        "| Feature | Pearson r | Spearman rho |",
        "|---|---|---|",
        *(
            f"| {key} | {value['pearson']:+.2f} | {value['spearman']:+.2f} |"
            for key, value in sorted(correlations.items(), key=lambda kv: -abs(kv[1]["spearman"]))
        ),
        "",
        f"Leave-one-out RMSE (Elo): the mean {null:.0f}; ridge on all {len(keys)} features "
        f"{ridge[0]:.0f} (penalty {ridge[1]:g}); "
        + "; ".join(
            f"best {size}-feature {error:.0f} ({', '.join(chosen)})"
            for size, (error, chosen) in best.items()
        )
        + f". The best one- or two-feature model on shuffled Elo ({args.shuffles} shuffles): "
        f"median {sorted(shuffled)[len(shuffled) // 2]:.2f} of the mean's error, "
        f"5th percentile {sorted(shuffled)[len(shuffled) // 20]:.2f}; ridge on shuffled Elo: "
        f"median {sorted(ridge_shuffled)[len(ridge_shuffled) // 2]:.2f}, 5th percentile "
        f"{sorted(ridge_shuffled)[len(ridge_shuffled) // 20]:.2f}"
        + (f" (without {', '.join(args.exclude)})." if args.exclude else "."),
    ]
    print("\n".join(lines))


def main() -> None:
    """`play` or `report`, as the module describes."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    playing = commands.add_parser("play")
    playing.add_argument("--record", required=True)
    playing.add_argument("--games", type=int, default=128, help="per pair")
    playing.add_argument("--block-games", type=int, default=64)
    playing.add_argument("--blocks", type=int, default=18, help="per batch")
    playing.add_argument("--hours", type=float, default=1.5, help="the run's bound")
    reporting = commands.add_parser("report")
    reporting.add_argument("--record", required=True)
    reporting.add_argument("--out", required=True)
    analysing = commands.add_parser("analyse")
    analysing.add_argument("--out", required=True)
    analysing.add_argument("--shuffles", type=int, default=200)
    analysing.add_argument("--exclude", nargs="*", default=[], help="types left out")
    args = parser.parse_args()
    dict(play=play, report=report, analyse=analyse)[args.command](args)


if __name__ == "__main__":
    main()
