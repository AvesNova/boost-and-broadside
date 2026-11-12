"""A doctrine round: the search (`bench.game_loop.search`) of both types' doctrines from each
start against the league, then the league's Elo (mixed lineups, every pair), the
distinctness of each result and the self-play table of the champion; the champion (the
best Elo among the results) joins the league as `r<round>`. Writes round<N>/.

    PYTHONPATH=src:. python docs/rewrite/sections/game-loop/doctrine_round.py --round 1 \
        --starts hand off --generations 12
"""

import argparse
import dataclasses
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
    clock,
    distinctness,
    es_search,
    job_summary,
    markdown,
    ratings,
    run,
    search,
)

HERE = pathlib.Path(__file__).parent
SELF_LINEUPS = ("energy v turn", "turn mirror", "energy mirror", "mixed", "mixed reversed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--starts", nargs="+", default=["hand", "off"])
    parser.add_argument("--opponents", nargs="+", default=None)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--children", type=int, default=12)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--environments", type=int, default=8192)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--elo-games", type=int, default=512)
    parser.add_argument("--minutes", type=float, default=40.0, help="the search's bound")
    parser.add_argument("--method", choices=("es", "mutation"), default="es")
    parser.add_argument("--pairs", type=int, default=16, help="the ES's antithetic pairs")
    parser.add_argument("--card", help="a JSON file of card factors {energy: {}, turn: {}}")
    args = parser.parse_args()
    out_dir = HERE / f"round{args.round}"
    out_dir.mkdir(exist_ok=True)
    started = time.time()
    deadline = started + 60 * args.minutes
    print(f"round {args.round}: bound {clock(60 * args.minutes)} KST", flush=True)
    loop = Loop(args.environments, 1, args.seconds)
    if args.card:
        card = json.loads(pathlib.Path(args.card).read_text())
        loop.set_types([card["energy"]], [card["turn"]])
    players = league.load()
    by = {p.name: p for p in players}
    opponents = [
        by[n]
        for n in (args.opponents or [p.name for p in players if p.name not in ("random", "hot")])
    ]
    results = []
    for start in args.starts:
        log_path = out_dir / f"search-{start}.jsonl"
        log_path.unlink(missing_ok=True)
        print(f"search from {start} against {[o.name for o in opponents]}", flush=True)
        if args.method == "es":
            found = es_search(
                loop,
                by[start],
                opponents,
                log_path,
                args.generations,
                args.pairs,
                args.games,
                args.sigma,
                seed=args.round * 100 + len(results),
                deadline=deadline,
                log=lambda line: print(line, flush=True),
            )
        else:
            found = search(
                loop,
                by[start],
                opponents,
                log_path,
                args.generations,
                args.children,
                args.games,
                args.sigma,
                margin=args.margin,
                seed=args.round * 100 + len(results),
                deadline=deadline,
                log=lambda line: print(line, flush=True),
            )
        results.append(dataclasses.replace(found, name=f"r{args.round}-{start}"))
    # the league's Elo, every pair, the mixed lineups
    field = players + results
    jobs = [
        Job(0, lineup, a, b, args.elo_games)
        for i, a in enumerate(field)
        for b in field[i + 1 :]
        for lineup in ("mixed", "mixed reversed")
    ]
    rows = [job_summary(j, o, args.seconds) for j, o in zip(jobs, run(loop, jobs), strict=True)]
    elo = ratings(rows)
    champion = max(results, key=lambda p: elo[p.name][0])
    print(
        "Elo",
        json.dumps(
            {
                k: [round(v[0]), round(v[1])]
                for k, v in sorted(elo.items(), key=lambda kv: -kv[1][0])
            }
        ),
        flush=True,
    )
    distinct = {
        p.name: distinctness(loop, [Group(0, "mixed", p, p, args.environments)], p.energy, p.turn)
        for p in results
    }
    print("distinct", json.dumps(distinct), flush=True)
    self_jobs = [Job(0, lineup, champion, champion, 2048) for lineup in SELF_LINEUPS]
    self_jobs += [Job(0, lineup, champion, by["off"], 1024) for lineup in SELF_LINEUPS]
    self_rows = [
        job_summary(j, o, args.seconds)
        for j, o in zip(self_jobs, run(loop, self_jobs), strict=True)
    ]
    print(markdown(self_rows), flush=True)
    (out_dir / "result.json").write_text(
        json.dumps(
            dict(
                elo=elo,
                distinct=distinct,
                champion=champion.named(),
                results=[p.named() for p in results],
                league_rows=rows,
                self_rows=self_rows,
            ),
            indent=1,
        )
    )
    league.add(dataclasses.replace(champion, name=f"r{args.round}"))
    (out_dir / "self-table.md").write_text(markdown(self_rows) + "\n")
    print(
        f"champion {champion.name} -> league as r{args.round}; {time.time() - started:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
