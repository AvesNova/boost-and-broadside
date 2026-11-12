"""The four-ship set (standard, brawler, small, close fighter) in mixed 4v4 games: every team
of four ships drawn from one card table of the four types, flown by the scripted agent.

    PYTHONPATH=src:. uv run --no-sync python docs/rewrite/sections/morning/fourship.py \
        --players r9 r9m2 --games 1024 --tag fourship

The types are the play maps' cards (`src/bnb/game/maps`): standard and brawler from
`arch-standard-brawler`, the close fighter from `arch-close-standard` (Yongma slot), small
from `arch-standard-small` (Jujak slot); each map's card fields as absolute values on the
bench's base card, as `r9m2-maps/make.py` builds them (every field the maps leave unlisted
is the same in both ship files, so a type is the same ship in either slot). The variant
table has two rows a slot: rows 0 standard and 1 close fighter (the Yongma slot: they fly
the player's Yongma doctrine, as on their maps), rows 2 brawler and 3 small (the Jujak
slot's doctrine; on `arch-brawler-small` the brawler flies the Yongma slot instead).

Lineups (`bench.game_loop.ROW_LINEUPS`, side A's team v side B's): one of each v one of
each; the three two-and-two splits (A two of two types, B two of the other two); and each
type four-strong against one of each. Per lineup: A's score (more kills wins, draws half)
and per type and side the kills, deaths, the share of all kills, kills / deaths and kills a
minute a ship. Results in `<tag>.json` (written as each player finishes) and `<tag>.md`.
"""

import argparse
import json
import math
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "game-loop"))

import league  # noqa: E402

from bench.game_loop import ROW_LINEUPS, Job, Loop, clock, player_from, run  # noqa: E402

TYPES = ("standard", "close", "brawler", "small")  # the rows: Yongma slot 0-1, Jujak 2-3
SOURCES = {
    "standard": ("arch-standard-brawler", "yongma"),
    "close": ("arch-close-standard", "yongma"),
    "brawler": ("arch-standard-brawler", "jujak"),
    "small": ("arch-standard-small", "jujak"),
}
ROW = {name: row for row, name in enumerate(TYPES)}
ALL = tuple(ROW[t] for t in ("standard", "brawler", "small", "close"))


def lineups() -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    out = {"one of each v one of each": (ALL, ALL)}
    for pair in (("standard", "brawler"), ("standard", "small"), ("standard", "close")):
        rest = tuple(t for t in ("standard", "brawler", "small", "close") if t not in pair)
        team_a = tuple(ROW[t] for t in pair for _ in range(2))
        team_b = tuple(ROW[t] for t in rest for _ in range(2))
        out[f"2 {pair[0]} + 2 {pair[1]} v 2 {rest[0]} + 2 {rest[1]}"] = (team_a, team_b)
    for name in ("standard", "brawler", "small", "close"):
        out[f"4 {name} v one of each"] = ((ROW[name],) * 4, ALL)
    return out


def card(map_name: str, slot: str) -> dict:
    data = json.loads((ROOT / "src/bnb/game/maps" / f"{map_name}.json").read_text())
    return {k: {"set": float(v)} for k, v in data["ships"][slot].items()}


def players() -> dict:
    by = {p.name: p for p in league.load()}
    for file in ("champion.json", "champion2.json"):  # maneuvers-2's r9m and r9m2
        path = HERE.parent / "maneuvers-2" / file
        champion = player_from(json.loads(path.read_text()))
        by[champion.name] = champion
    return by


def per_type(out: dict, teams: tuple, seconds: float) -> dict:
    """Per side and type: kills, deaths, share of all kills, K/D, kills a minute a ship."""
    kills = out["kills_by_row"].sum(0)  # (killer side, killer row, victim row)
    games = out["kills_side"].shape[0]
    total = float(kills.sum())
    rows = {}
    for side, team in enumerate(teams):
        for row in sorted(set(team)):
            ships = team.count(row)
            k = float(kills[side, row].sum())
            d = float(kills[1 - side, :, row].sum())
            rows[f"{'AB'[side]} {TYPES[row]}"] = dict(
                ships=ships,
                kills=k,
                deaths=d,
                kill_share=k / max(total, 1.0),
                kd=k / max(d, 1.0),
                kills_per_minute_per_ship=k / (games * ships * seconds / 60.0),
            )
    # by type over both sides (the symmetric lineups: one of each on both)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", nargs="+", default=["r9"])
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--environments", type=int, default=4096)
    parser.add_argument("--tag", default="fourship")
    args = parser.parse_args()
    started = time.time()
    ROW_LINEUPS.update(lineups())
    by = players()
    energy = [card(*SOURCES["standard"]), card(*SOURCES["close"])]
    turn = [card(*SOURCES["brawler"]), card(*SOURCES["small"])]
    loop = Loop(args.environments, 2, args.seconds, ships_per_team=4)
    loop.set_types(energy, turn)
    result = dict(types=TYPES, sources=SOURCES, lineups=ROW_LINEUPS, players={})
    out_path = HERE / f"{args.tag}.json"
    log = lambda line: print(line, flush=True)  # noqa: E731
    for name in args.players:
        player = by[name]
        jobs = [Job(0, lineup, player, player, args.games) for lineup in ROW_LINEUPS]
        log(f"## {name}: {len(jobs)} jobs, B {args.environments}, {clock(0)} KST")
        outs = run(loop, jobs, log)
        rows = []
        for job, o in zip(jobs, outs, strict=True):
            side = o["kills_side"]
            n = side.shape[0]
            # the per-type tally against the per-side one (the same kills)
            if not (o["kills_by_row"].sum((2, 3)) == side).all():
                raise SystemExit(f"{job.lineup}: kills by type differ from kills by side")
            wins = float((side[:, 0] > side[:, 1]).sum())
            draws = float((side[:, 0] == side[:, 1]).sum())
            score = (wins + 0.5 * draws) / n
            row = dict(
                lineup=job.lineup,
                games=n,
                wins=wins,
                draws=draws,
                score=score,
                stderr=math.sqrt(score * (1 - score) / n),
                kills_per_game=float(side.sum()) / n,
                types=per_type(o, ROW_LINEUPS[job.lineup], args.seconds),
            )
            rows.append(row)
            log(
                f"{name} {job.lineup:>40}: A {score:.3f} ± {row['stderr']:.3f}  "
                + "  ".join(
                    f"{t} {v['kill_share']:.2f} kd {v['kd']:.2f}" for t, v in row["types"].items()
                )
            )
        result["players"][name] = rows
        out_path.write_text(json.dumps(result, indent=1))
    (HERE / f"{args.tag}.md").write_text(markdown(result))
    log(markdown(result))
    log(f"{time.time() - started:.0f} s")


def markdown(result: dict) -> str:
    lines = []
    for name, rows in result["players"].items():
        lines += [
            f"### {name}",
            "",
            "| Lineup (A v B) | A score | Per side and type: kill share, K/D, kills/min/ship |",
            "|---|---|---|",
        ]
        for r in rows:
            cells = "; ".join(
                f"{t} {v['kill_share']:.2f}, {v['kd']:.2f}, {v['kills_per_minute_per_ship']:.2f}"
                for t, v in r["types"].items()
            )
            lines.append(f"| {r['lineup']} | {r['score']:.3f} ± {r['stderr']:.3f} | {cells} |")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
