"""Two scripted parameter sets head to head in one preset, and each against itself, with the
games' statistics: the check of a parameter search's choice (the real-scale variant's
light retune, docs/explanation/game/real-scale.md).

    uv run --no-sync python -m bench.scripted_compare --preset real_scale_deathmatch \\
        --ship-type 0 --first real_scale_deathmatch --second deathmatch --games 512

A set is a mode's in `MODE_DEFAULTS` (its name) or a search log's candidate
(`LOG.jsonl:GENERATION:INDEX`). Every match is `--games` fresh games at T = 1, fog on, the
seat drawn per game (`bnb.runtime.evaluate.first_results`'s rollout), seeded with
`--seed`: the first set against the second (its score, a draw half a win, with its standard
error), then each against itself. Per match: the games' kills (combat deaths), shots and
hits a game over the episodes ended, and kills, front moves (Frontline's captures) and
shots a minute over every tick played; the share of games ended by the clock. One JSON
line per match is appended to `--out` as it finishes. Holds the benchmark lock.
"""

import argparse
import json
import math
import time

import torch

from bench.lock import bench_lock
from bnb.contracts import Pool
from bnb.controllers.scripted import Scripted
from bnb.controllers.scripted_defaults import MODE_DEFAULTS
from bnb.controllers.scripted_parameters import from_row
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.runtime.analytics import EPISODE_COLUMNS, PERCEIVE_COLUMNS, RESOLVE_COLUMNS
from bnb.runtime.evaluate import score
from bnb.runtime.rollout import Rollout


def parameters_of(name: str):
    """A mode's default set, or a search log's candidate (`LOG:GENERATION:INDEX`)."""
    if name in MODE_DEFAULTS.sets:
        return MODE_DEFAULTS.sets[name]
    path, generation, index = name.rsplit(":", 2)
    for line in open(path):
        row = json.loads(line)
        if row["generation"] == int(generation) and row["index"] == int(index):
            return from_row(row["parameters"])
    raise SystemExit(f"{name}: no such candidate")


def match(game, types, card, first, second, games, ship_type, device) -> dict:
    """`games` fresh games of `first` against `second` to every first game's end, and the
    games' statistics (the module docstring)."""
    S = game.ships_per_team
    lineup = None
    if ship_type is not None:
        lineup = torch.full((games, game.num_ships), ship_type, dtype=torch.int32)
    rollout = Rollout(
        game,
        types,
        (Pool("compare", 1.0, ("first", "second")),),
        {"first": Scripted(card, S, first), "second": Scripted(card, S, second)},
        "first",
        games,
        32,
        device,
        staggered=False,
        ship_type=lineup,
    )
    views = rollout.lineup.rows("first")
    result = torch.zeros(views, dtype=torch.int32, device=device)
    finished = torch.zeros(views, dtype=torch.bool, device=device)
    episodes = torch.zeros(len(EPISODE_COLUMNS), device=device)
    while not bool(finished.all()):
        trajectory = rollout.run()
        for tick in range(trajectory.result.shape[0]):
            first_end = trajectory.done[tick] & ~finished
            result = torch.where(first_end, trajectory.result[tick], result)
            finished = finished | trajectory.done[tick]
        episodes += rollout.take_episodes().sum(0)
    sums = rollout.carry["game_sums"]
    resolve = dict(zip(RESOLVE_COLUMNS, sums["resolve"].sum(0).tolist(), strict=True))
    perceive = dict(zip(PERCEIVE_COLUMNS, sums["perceive"].sum(0).tolist(), strict=True))
    ends, timeouts, _ = episodes.tolist()
    minutes = resolve["environment_ticks"] * rollout.rules.tick_length / 60.0
    results = result.cpu()
    won = (results > 0).double()
    points = won + 0.5 * (results == 0).double()
    return dict(
        games=games,
        score=score(results),
        score_stderr=float(points.std() / math.sqrt(games)),
        wins=int((results > 0).sum()),
        draws=int((results == 0).sum()),
        losses=int((results < 0).sum()),
        kills_per_game=resolve["episode_event/combat_deaths"] / ends,
        shots_per_game=resolve["episode_event/shots_fired"] / ends,
        hits_per_game=resolve["episode_event/hits"] / ends,
        kills_per_minute=resolve["combat_deaths"] / minutes,
        front_moves_per_minute=perceive["front_moves"] / minutes,
        clock_share=timeouts / ends,
        episodes=int(ends),
    )


def main() -> None:
    """The matches the module describes."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preset", required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--ship-type", type=int, help="every ship flies this type")
    parser.add_argument("--seed", type=int, default=1_000_000, help="above the search's seeds")
    parser.add_argument("--self-play", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    game = api.PRESETS[args.preset]
    types = api.ship_types(device, ship_set=game.ship_set)
    card = characteristics(types, api.rules_of(game))
    first, second = parameters_of(args.first), parameters_of(args.second)
    pairs = [(args.first, first, args.second, second)]
    if args.self_play:
        pairs += [
            (args.first, first, args.first, first),
            (args.second, second, args.second, second),
        ]
    with bench_lock(), open(args.out, "a") as out:
        for first_name, one, second_name, other in pairs:
            started = time.perf_counter()
            torch.manual_seed(args.seed)
            torch._dynamo.reset()
            row = match(game, types, card, one, other, args.games, args.ship_type, device)
            row = dict(preset=args.preset, first=first_name, second=second_name, **row)
            row["seconds"] = time.perf_counter() - started
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
