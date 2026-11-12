"""Balance across fleet sizes: candidates (a card variant and a player, the player a league
champion with doctrine fields overridden) in self-play at each fleet size, energy v turn
and optionally the mirrors; per candidate and size the score (Yongma's side), the kill mix,
trades, pace and the fight shape. Every size is one compiled loop; the candidates' card
variants are the rows of its type table, their doctrines per group. Each size's rows are
written as soon as it finishes (`<tag>.json`), the table at the end (`<tag>.md`).

    PYTHONPATH=src:. uv run --no-sync --active python \
        docs/rewrite/sections/tank-armour/sizes.py --candidates search/candidates.json \
        --tag search/sizes --sizes 1 2 5 --lineups "energy v turn" "turn mirror" "energy mirror" \
        --mirror-sizes 1 --versus r9

A candidate: {"name", "energy": {card factors}, "turn": {card factors}, "player": league
name (default r9), "doctrine": {"energy": {field: value}, "turn": {...}}, "parameters":
{index: value}}. Card factors as `bench.game_loop.variant_types`.
"""

import argparse
import dataclasses
import json
import math
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "game-loop"))

import league  # noqa: E402

from bench.game_loop import Job, Loop, Player, card_passes, clock, job_summary, run  # noqa: E402
from bnb.controllers.scripted_doctrine import NAMES, Doctrine  # noqa: E402
from bnb.controllers.scripted_parameters import Logistic  # noqa: E402

ENVIRONMENTS = {1: 8192, 2: 8192, 5: 4096, 10: 2048, 20: 1024, 50: 256}


def overridden(doctrine: Doctrine, fields: dict) -> Doctrine:
    """The doctrine with fields replaced (a logistic as [centre, scale])."""
    values = {}
    for name, value in fields.items():
        if name not in NAMES:
            raise ValueError(f"no doctrine field {name}")
        kind = getattr(doctrine, name)
        values[name] = Logistic(*value, kind.rising) if isinstance(kind, Logistic) else value
    return dataclasses.replace(doctrine, **values)


def player_of(candidate: dict, by: dict[str, Player], number: int) -> Player:
    """The candidate's player, under its own name."""
    base = by[candidate.get("player", "r9")]
    changes = candidate.get("doctrine", {})
    parameters = base.parameters
    if candidate.get("parameters"):
        parameters = list(parameters)
        for index, value in candidate["parameters"].items():
            parameters[int(index)] = value
        parameters = tuple(parameters)
    return dataclasses.replace(
        base,
        name=f"c{number}",
        energy=overridden(base.energy, changes.get("energy", {})),
        turn=overridden(base.turn, changes.get("turn", {})),
        parameters=parameters,
    )


def cells(row: dict) -> dict:
    """The numbers a table shows of a job summary."""
    by_definition = row.get("engagement", {}).get("two turn diameters", {})
    return dict(
        score=row["score"],
        kill_share=row["kill_share"],
        trade_fraction=row["trade_fraction"],
        life_seconds=row["life_seconds"],
        kills_per_minute_per_ship=row["kills_per_minute_per_ship"],
        kills_per_game=row["kills_per_game"],
        merges_mean=by_definition.get("merges_per_kill_mean"),
        merges_median=by_definition.get("merges_per_kill_median"),
        first_pass=by_definition.get("first_pass_kill_share"),
        merges_p90=by_definition.get("merges_per_kill_p90"),
        circles_turned=by_definition.get("circles_turned_mean"),
        one_two_scissors=by_definition.get("one_two_scissors"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--games", type=int, default=512, help="a candidate, size and lineup")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--lineups", nargs="+", default=["energy v turn"])
    parser.add_argument(
        "--versus", default=None, help="also each candidate against this player, mixed, 5v5"
    )
    parser.add_argument(
        "--mirror-sizes",
        type=int,
        nargs="*",
        default=None,
        help="play the lineups other than energy v turn only at these sizes (default: all)",
    )
    parser.add_argument(
        "--versus-lineups",
        nargs="+",
        default=["mixed", "mixed reversed"],
        help="the lineups against --versus",
    )
    parser.add_argument("--versus-sizes", type=int, nargs="+", default=[5])
    parser.add_argument("--arena-scale", type=float, default=1.0, help="off the density rule")
    parser.add_argument("--fog", type=int, default=1)
    args = parser.parse_args()
    started = time.time()
    candidates = json.loads((HERE / args.candidates).read_text())
    by = {p.name: p for p in league.load()}
    players = [player_of(c, by, n) for n, c in enumerate(candidates)]
    cards, card_of = [], []
    for c in candidates:
        key = json.dumps([c.get("energy", {}), c.get("turn", {})], sort_keys=True)
        if key not in cards:
            cards.append(key)
        card_of.append(cards.index(key))
    K = len(cards)
    energy = [json.loads(k)[0] for k in cards]
    turn = [json.loads(k)[1] for k in cards]
    out_path = HERE / f"{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(candidates=candidates, sizes={}, versus=[])
    log = lambda line: print(line, flush=True)  # noqa: E731
    for size in args.sizes:
        B = ENVIRONMENTS[size]
        loop = Loop(
            B,
            K,
            args.seconds,
            ships_per_team=size,
            fog=bool(args.fog),
            arena_scale=args.arena_scale,
        )
        loop.set_types(energy, turn)
        if size == args.sizes[0]:
            result["card_passes"] = card_passes(loop)
        games = args.games if size <= 5 else max(args.games // 2, 128)
        if size >= 20:
            games = max(args.games // 4, 64)
        jobs = [
            Job(card_of[n], lineup, player, player, games)
            for n, player in enumerate(players)
            for lineup in args.lineups
            if lineup == "energy v turn" or args.mirror_sizes is None or size in args.mirror_sizes
        ]
        if args.versus and size in args.versus_sizes:
            versus = by[args.versus]
            jobs += [
                Job(card_of[n], lineup, player, versus, args.games)
                for n, player in enumerate(players)
                for lineup in args.versus_lineups
            ]
        log(f"## {size}v{size}: {len(jobs)} jobs, B {B}, {clock(0)} KST")
        outs = run(loop, jobs, log)
        rows = []
        for job, o in zip(jobs, outs, strict=True):
            row = job_summary(job, o, args.seconds)
            number = int(job.a.name[1:])
            row["candidate"] = candidates[number]["name"]
            rows.append(row)
            log(
                f"{size}v{size} {row['candidate']:>28} {job.lineup:>15} v {job.b.name}: "
                f"score {row['score'][0]:.3f} ± {row['score'][1]:.3f}  kills m/o/f/t "
                + " / ".join(f"{v:.2f}" for v in row["kill_share"])
                + f"  life {row['life_seconds']:.0f} s  kpm/ship "
                f"{row['kills_per_minute_per_ship']:.2f}  trades {row['trade_fraction']:.2f}"
            )
        result["sizes"][str(size)] = [r for r in rows if r["b"] == r["a"]]
        result["versus"] += [dict(r, size=size) for r in rows if r["b"] != r["a"]]
        out_path.write_text(json.dumps(result, indent=1))
        del loop
    table = markdown(result, args.sizes)
    (HERE / f"{args.tag}.md").write_text(table)
    log(table)
    log(f"{time.time() - started:.0f} s")


def markdown(result: dict, sizes: list[int]) -> str:
    """Per candidate: energy v turn by size, the spread, and the 5v5 shape."""
    lines = [
        "| Candidate | "
        + " | ".join(f"E v T {s}v{s}" for s in sizes)
        + " | Worst |dev| | Kills m / o / f / t (5v5) | Trades | Life s by size "
        "| Kills/min/ship by size | Merges a kill mean (1v1) |",
        "|---" * (len(sizes) + 7) + "|",
    ]
    for c in result["candidates"]:
        by_size = {}
        for s in sizes:
            rows = result["sizes"].get(str(s), [])
            row = next(
                (r for r in rows if r["candidate"] == c["name"] and r["lineup"] == "energy v turn"),
                None,
            )
            by_size[s] = row
        scores = [r["score"] for r in by_size.values() if r]
        worst = max(abs(s[0] - 0.5) for s in scores) if scores else math.nan
        five = by_size.get(5) or next(r for r in by_size.values() if r)
        one = by_size.get(1)
        merges = one["engagement"]["two turn diameters"]["merges_per_kill_mean"] if one else None
        lines.append(
            f"| {c['name']} | "
            + " | ".join(
                f"{r['score'][0]:.3f} ± {r['score'][1]:.3f}" if r else "" for r in by_size.values()
            )
            + f" | {worst:.3f} | "
            + " / ".join(f"{v:.2f}" for v in five["kill_share"])
            + f" | {five['trade_fraction']:.2f} | "
            + " / ".join(f"{r['life_seconds']:.0f}" if r else "" for r in by_size.values())
            + " | "
            + " / ".join(
                f"{r['kills_per_minute_per_ship']:.2f}" if r else "" for r in by_size.values()
            )
            + f" | {merges if merges is None else round(merges, 2)} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
