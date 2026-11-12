"""The three-ship set: Yongma, Jujak and a middle ship in one card table, flown by r9m2.

    PYTHONPATH=src:. python docs/rewrite/sections/ships-r9m2/threeship.py \
        --set threeship/set.json --tag threeship/run1 --games 512

The set file: {"yongma": {card factors}, "jujak": {...}, "middle_energy": {...},
"middle_turn": {...}, "gate": Jujak-slot gate}. The variant table has two rows a slot:
row 0 Yongma and row 1 the middle ship (the Yongma slot: they fly r9m2's Yongma doctrine),
row 2 Jujak and row 3 the middle ship again (the Jujak slot's doctrine), so the middle ship
is flown under either doctrine ("Me", "Mt"). Lineups (`bench.game_loop.ROW_LINEUPS`, side
A's rows v side B's) at each size: every pair (A the first named), the middle ship's
mirrors, and at 5v5 / 10v10 one mixed lineup (2 Y + 2 J + 1 M a side, doubled at 10v10).
Per lineup the `job_summary` (A's score, the fight shape, the choices) and the kills by type
row. Results in `<tag>.json` (written as each size finishes).
"""

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "game-loop"))
sys.path.insert(0, str(HERE.parent / "maneuvers-2"))

import league  # noqa: E402
from sizes import cells, player_of  # noqa: E402

from bench.game_loop import ROW_LINEUPS, Job, Loop, clock, job_summary, player_from, run  # noqa: E402

ENVIRONMENTS = {1: 8192, 5: 4096, 10: 2048}
NAMES = ("Y", "Me", "J", "Mt")  # the rows
PAIRS = (("Y", "J"), ("Y", "Me"), ("J", "Me"), ("Y", "Mt"), ("J", "Mt"), ("Me", "Me"), ("Mt", "Mt"))


def lineups(size: int) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    row = {n: i for i, n in enumerate(NAMES)}
    out = {f"{a} v {b}": ((row[a],) * size, (row[b],) * size) for a, b in PAIRS}
    if size in (5, 10):
        for m in ("Me", "Mt"):
            team = (row["Y"], row["Y"], row["J"], row["J"], row[m]) * (size // 5)
            out[f"mixed ({m})"] = (team, team)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--environments", type=int, default=None, help="B at every size")
    args = parser.parse_args()
    started = time.time()
    card = json.loads((HERE / args.set).read_text())
    by = {p.name: p for p in league.load()}
    champion = player_from(json.loads((HERE.parent / "maneuvers-2/champion2.json").read_text()))
    by[champion.name] = champion
    player = player_of(dict(player="r9m2", **card.get("player", {})), by, 0)
    energy = [card["yongma"], card["middle_energy"]]
    turn = [card["jujak"], card["middle_turn"]]
    out_path = HERE / f"{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(set=card, sizes={})
    log = lambda line: print(line, flush=True)  # noqa: E731
    for size in args.sizes:
        ROW_LINEUPS.clear()
        ROW_LINEUPS.update(lineups(size))
        loop = Loop(args.environments or ENVIRONMENTS[size], 2, args.seconds, ships_per_team=size)
        loop.set_types(energy, turn)
        games = args.games if size <= 5 else max(args.games // 2, 128)
        jobs = [Job(0, lineup, player, player, games) for lineup in ROW_LINEUPS]
        log(f"## {size}v{size}: {len(jobs)} jobs, {clock(0)} KST")
        outs = run(loop, jobs, log)
        rows = []
        for job, o in zip(jobs, outs, strict=True):
            row = job_summary(job, o, args.seconds)
            kills = o["kills_by_row"].sum(0)  # killer side, killer row, victim row
            row["kills_by_row"] = kills.tolist()
            row["cells"] = cells(row)
            rows.append(row)
            log(
                f"{size}v{size} {job.lineup:>12}: A {row['score'][0]:.3f} ± {row['score'][1]:.3f}"
                f"  life {row['life_seconds']:.0f} s"
                f"  kpm/ship {row['kills_per_minute_per_ship']:.2f}"
            )
        result["sizes"][str(size)] = rows
        out_path.write_text(json.dumps(result, indent=1))
        del loop
    log(f"{time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
