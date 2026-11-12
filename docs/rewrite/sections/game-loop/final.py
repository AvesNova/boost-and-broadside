"""The final measurement: on the final card, the league's Elo (mixed lineups, every pair,
`--elo-games` a pair and order, 120 s games), the champion's self-play in every lineup and
against off in 300 s games (the pace), its distinctness, and the card's head-on TTK in
passes. Writes final/.

    PYTHONPATH=src:. python docs/rewrite/sections/game-loop/final.py --card CARD.json \
        --champion NAME
"""

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import league  # noqa: E402

from bench.game_loop import (  # noqa: E402
    Group,
    Job,
    Loop,
    card_passes,
    distinctness,
    job_summary,
    markdown,
    ratings,
    run,
)

HERE = pathlib.Path(__file__).parent
LINEUPS = ("energy v turn", "turn mirror", "energy mirror", "mixed", "mixed reversed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--elo-games", type=int, default=2048)
    parser.add_argument("--self-games", type=int, default=8192)
    parser.add_argument("--environments", type=int, default=8192)
    args = parser.parse_args()
    started = time.time()
    out_dir = HERE / "final"
    out_dir.mkdir(exist_ok=True)
    card = json.loads(pathlib.Path(args.card).read_text())
    players = league.load()
    by = {p.name: p for p in players}
    champion = by[args.champion]
    log = lambda line: print(line, flush=True)  # noqa: E731

    loop = Loop(args.environments, 1, 120.0)
    loop.set_types([card["energy"]], [card["turn"]])
    passes = card_passes(loop)[0]
    log(f"card {json.dumps(passes)}")
    jobs = [
        Job(0, lineup, a, b, args.elo_games)
        for i, a in enumerate(players)
        for b in players[i + 1 :]
        for lineup in ("mixed", "mixed reversed")
    ]
    rows = [job_summary(j, o, 120.0) for j, o in zip(jobs, run(loop, jobs, log), strict=True)]
    elo = ratings(rows)
    log(
        "Elo "
        + json.dumps(
            {
                k: [round(v[0]), round(v[1])]
                for k, v in sorted(elo.items(), key=lambda kv: -kv[1][0])
            }
        )
    )
    distinct = distinctness(
        loop,
        [Group(0, "mixed", champion, champion, args.environments)],
        champion.energy,
        champion.turn,
    )
    log("distinct " + json.dumps(distinct))
    (out_dir / "result.json").write_text(
        json.dumps(
            dict(card=card, passes=passes, elo=elo, distinct=distinct, league_rows=rows), indent=1
        )
    )
    del loop

    pace_keys = (
        "kills_per_minute_mean",
        "kills_per_minute_median",
        "kills_per_minute_per_ship",
        "life_seconds",
        "death_gap_mean",
        "death_gap_median",
        "spawn_to_engagement_median",
        "first_hit_median",
        "first_kill_median",
        "trade_fraction",
    )
    self_rows = []
    for size in (5, 2, 1):
        pace_loop = Loop(args.environments, 1, 300.0, ships_per_team=size)
        pace_loop.set_types([card["energy"]], [card["turn"]])
        lineups = LINEUPS if size > 1 else LINEUPS[:3]
        games = args.self_games if size == 5 else args.self_games // 2
        jobs = [Job(0, lineup, champion, champion, games) for lineup in lineups]
        jobs += [Job(0, lineup, champion, by["off"], games // 4) for lineup in lineups]
        jobs += [Job(0, lineup, by["off"], by["off"], games // 4) for lineup in lineups]
        rows_size = [
            dict(job_summary(j, o, 300.0), size=size)
            for j, o in zip(jobs, run(pace_loop, jobs, log), strict=True)
        ]
        self_rows += rows_size
        log(f"## {size}v{size}")
        log(markdown(rows_size))
        for r in rows_size:
            log(
                f"{size}v{size} {r['lineup']} {r['a']} v {r['b']}: "
                + ", ".join(f"{k} {r[k]}" for k in pace_keys)
                + " engagement "
                + json.dumps(r["engagement"])
            )
        (out_dir / "result.json").write_text(
            json.dumps(
                dict(
                    card=card,
                    passes=passes,
                    elo=elo,
                    distinct=distinct,
                    league_rows=rows,
                    self_rows=self_rows,
                ),
                indent=1,
            )
        )
        del pace_loop
    log(f"{time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
