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
as Markdown (`table.md`, also printed).

It rates a type as the scripted agent, tuned on the P-51D, flies it: not the type's
ceiling.
"""

import argparse
import csv
import datetime
import json
import math
import pathlib
import time

import torch

from bench.lock import bench_lock
from bnb.game import api
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
    (out / "table.md").write_text(header + body)
    print(json.dumps(meta))
    print(header + body, end="")


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
    args = parser.parse_args()
    (play if args.command == "play" else report)(args)


if __name__ == "__main__":
    main()
