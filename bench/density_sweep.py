"""The density sweep: does the scripted agent's edge over a weaker rung (the same agent at a
higher temperature) hold as the fleets grow, or collapse toward 50% past a critical
density? (The author's hypothesis from v1: past it fights become a spawn lottery.)

    PYTHONPATH=src python -m bench.density_sweep --out DIR [--mode team_deathmatch]
        [--sizes 5,10,20,35,50] [--radii fixed,density] [--temperature 2]
        [--stderr 0.05] [--hours 2]

Each cell is one game: `--base` (real_scale_deathmatch) with no fields, `--mode`, n ships a
side, and an arena radius: "fixed" keeps the base's (7800 m), "density" grows it as
sqrt(n / 5) (the fleet presets' `_density` rule), or a number in metres. The scripted agent
(T = 1) plays `rung_T<temperature>` on the league's match engine (`bnb.league.matches.Arena`,
compiled, both seats evenly); every batch's wins, ties and losses for the scripted agent
are appended to DIR/density.jsonl (with the batch's kills, deaths and ship-seconds), and a
cell stops when its score's standard error is under `--stderr` (score: wins plus half the
ties over games) or after `--max-games`. A cell already in the record resumes. `--hours`
bounds the whole run. Prints a progress line per batch with a KST clock ETA.
"""

import argparse
import dataclasses
import datetime
import json
import math
import pathlib
import time
import zoneinfo

import torch

from bnb.cli import compile_limits
from bnb.game import ships as ship_sets
from bnb.game.api import PRESETS
from bnb.league.matches import STATISTICS, Arena, ArenaConfig, player
from bnb.runtime.opponents import Rung, ladder_name

KST = zoneinfo.ZoneInfo("Asia/Seoul")


def clock(seconds: float) -> str:
    """The KST clock time `seconds` from now."""
    return (datetime.datetime.now(KST) + datetime.timedelta(seconds=seconds)).strftime("%H:%M")


def radius_of(kind: str, base_radius: float, base_ships: int, ships: int) -> float:
    """A cell's arena radius (m)."""
    if kind == "fixed":
        return base_radius
    if kind == "density":
        return round(base_radius * math.sqrt(ships / base_ships), 1)
    return float(kind)


def lineup_of(lineup: str, game) -> tuple[int, ...]:
    """A team's ship types: the keys cycled over its slots."""
    keys = [ship.key for ship in ship_sets.load(game.ship_set)]
    chosen = [keys.index(key) for key in lineup.split(",")]
    return tuple(chosen[slot % len(chosen)] for slot in range(game.ships_per_team))


def score(counts: list[int]) -> tuple[float, float]:
    """(score, standard error) of wins, ties, losses."""
    wins, ties, losses = counts
    games = wins + ties + losses
    if games == 0:
        return math.nan, math.inf
    values = [1.0] * wins + [0.5] * ties + [0.0] * losses
    mean = sum(values) / games
    variance = sum((v - mean) ** 2 for v in values) / max(games - 1, 1)
    return mean, math.sqrt(variance / games) if games > 1 else math.inf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--base", default="real_scale_deathmatch")
    parser.add_argument("--mode", default=None, help="the base's mode by default")
    parser.add_argument("--sizes", default="5,10,20,35,50")
    parser.add_argument("--radii", default="fixed,density")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--stderr", type=float, default=0.05)
    parser.add_argument("--max-games", type=int, default=2048)
    parser.add_argument("--environments", type=int, default=256, help="games per batch")
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--fields", type=int, default=0)
    parser.add_argument(
        "--lineup", default=None, help="type keys cycled over each team's slots (both teams)"
    )
    arguments = parser.parse_args()
    compile_limits()
    arguments.out.mkdir(parents=True, exist_ok=True)
    record = arguments.out / "density.jsonl"
    done: dict[str, list[int]] = {}
    if record.exists():
        for line in record.read_text().splitlines():
            entry = json.loads(line)
            held = done.setdefault(entry["cell"], [0, 0, 0])
            for index in range(3):
                held[index] += entry["counts"][index]
    base = PRESETS[arguments.base]
    weaker = ladder_name(Rung(arguments.temperature))
    players = [player("scripted"), player(weaker)]
    device = torch.device("cuda")
    deadline = time.time() + 3600.0 * arguments.hours
    cells = [
        (ships, kind)
        for kind in arguments.radii.split(",")
        for ships in (int(size) for size in arguments.sizes.split(","))
    ]
    for ships, kind in cells:
        radius = radius_of(kind, base.arena_radius, base.ships_per_team, ships)
        mode = arguments.mode or base.mode
        cell = f"{mode}_{ships}v{ships}_R{radius:.0f}_T{arguments.temperature:g}"
        if arguments.lineup:
            cell += f"_{arguments.lineup.replace(',', '-')}"
        counts = done.setdefault(cell, [0, 0, 0])
        if score(counts)[1] < arguments.stderr or sum(counts) >= arguments.max_games:
            print(f"{cell}: done {score(counts)}", flush=True)
            continue
        if time.time() > deadline:
            break
        changes = dict(ships_per_team=ships, arena_radius=radius, num_fields=arguments.fields)
        if mode != base.mode:
            changes.update(mode=mode, respawn_seconds=None)
        game = dataclasses.replace(base, **changes)
        environments = arguments.environments
        built = time.time()
        arena = Arena(players, game, device, ArenaConfig(blocks=1, block_environments=environments))
        if arguments.lineup:
            team = lineup_of(arguments.lineup, game)
            arena.team_types = [team, team]
            arena.rollout.ship_type = torch.tensor(
                team + team, dtype=torch.int32, device=device
            ).repeat(environments, 1)
        print(f"{cell}: built in {time.time() - built:.0f} s", flush=True)
        while score(counts)[1] >= arguments.stderr and sum(counts) < arguments.max_games:
            started = time.time()
            result = arena.play([(0, 1)])[0].tolist()
            statistics = arena.statistics()[0]  # (2, columns) scripted's, the rung's
            seconds = time.time() - started
            for index in range(3):
                counts[index] += result[index]
            columns = {
                name: [float(statistics[side, index]) for side in range(2)]
                for index, name in enumerate(STATISTICS)
                if index < statistics.shape[-1]
            }
            entry = dict(
                cell=cell,
                mode=mode,
                ships=ships,
                radius=radius,
                kind=kind,
                temperature=arguments.temperature,
                lineup=arguments.lineup,
                counts=result,
                seconds=seconds,
                statistics=columns,
                at=datetime.datetime.now(KST).isoformat(timespec="seconds"),
            )
            with record.open("a") as file:
                file.write(json.dumps(entry) + "\n")
            mean, error = score(counts)
            needed = (
                max(0.0, (error / arguments.stderr) ** 2 - 1.0) * sum(counts)
                if math.isfinite(error)
                else environments
            )
            batches = math.ceil(min(needed, arguments.max_games - sum(counts)) / environments)
            print(
                f"{cell}: {counts} score {mean:.3f} +- {error:.3f}, batch {seconds:.0f} s, "
                f"cell ETA {clock(batches * seconds)}",
                flush=True,
            )
            if time.time() > deadline:
                break
        del arena
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
