"""The scripted agent's parameter search (scripted spec A3), kept so it can be rerun.

    uv run --no-sync python -m bench.scripted_search [--rounds 3] [--games 128] [--sigma 0.3]
    uv run --no-sync python -m bench.scripted_search --resume search.jsonl --rounds 8 \\
        --games 256 --sigma 0.2 --decay 0.8

Each round plays 32 candidates side by side (one parameter-table row per view, so one
compiled rollout per match): the round's best so far and 31 log-normal perturbations of it.
Each candidate plays fresh 5v5 games against the chaser and against `STARTING_PARAMETERS`,
in the team deathmatch and Frontline presets, fog on, the seat drawn per game. Its objective
is the mean of those four scores. The next round perturbs the best, with sigma times decay.
Every candidate's scores and parameters are appended to --log as a JSON line.

Seeds: the perturbations draw from a generator seeded with --seed; round r's games seed the
default generators with `game_seed(seed, r)`, below `SEED_LIMIT` for every seed and round
this tool accepts. The acceptance tests (`tests/sections/test_scripted_acceptance.py`) seed
their games at or above it.

Holds the benchmark lock. The section notes record the run that chose the defaults.
"""

import argparse
import dataclasses
import json
import math
import os
import time

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch  # noqa: E402

from bench.lock import bench_lock  # noqa: E402
from bnb.contracts import Pool  # noqa: E402
from bnb.controllers.scripted import Chaser, Scripted  # noqa: E402
from bnb.controllers.scripted_parameters import (  # noqa: E402
    Ramp,
    ScriptedParameters,
    pack,
    table,
    unpack,
)
from bnb.game import api  # noqa: E402
from bnb.game.handling.characteristics import characteristics  # noqa: E402
from bnb.game.modes.rules import PRESETS  # noqa: E402
from bnb.runtime.evaluate import first_results  # noqa: E402

# Where the search began: my starting values (v1's code was out of bounds; the shoot ramp,
# 0.5-1.0 gun ranges, is v1's own, converted through its 200 px range).
STARTING_PARAMETERS = ScriptedParameters(
    turn_ramp_in_stall_angle=Ramp(0.03, 0.15, 0.0, 1.0),
    level_two_ramp_in_stall_angle=Ramp(0.2, 1.0, 0.0, 1.0),
    level_three_ramp_in_stall_angle=Ramp(1.0, 3.0, 0.0, 1.0),
    boost_slow_ramp_in_corner_speed=Ramp(0.8, 1.1, 1.0, 0.0),
    boost_far_ramp_in_store_gun_ranges=Ramp(1.5, 3.0, 0.0, 1.0),
    reverse_ramp_in_gun_range=Ramp(0.25, 0.5, 1.0, 0.0),
    aim_ramp_in_target_size=Ramp(0.5, 1.5, 1.0, 0.0),
    shoot_ramp_in_gun_range=Ramp(0.5, 1.0, 1.0, 0.0),
    lead_cap_in_gun_range=2.0,
    boundary_margin_in_turn_radius=3.0,
    boundary_steer_ramp_in_margin=Ramp(0.0, 0.5, 0.0, 1.0),
    centre_hold_in_arena_radius=0.3,
    combat_radius_in_gun_range=3.0,
    support_radius_in_zone_radius=2.0,
    defensive_floor=0.5,
    defensive_margin=0.5,
    aggression=0.0,
    defense_urgency=1.0,
    separation_radius_in_gun_range=0.5,
    separation_weight=0.5,
    recovery_shield_fraction=0.3,
    recovery_threat=0.5,
)
PRESET_NAMES = ("deathmatch", "frontline")
OPPONENTS = ("chaser", "start")
CANDIDATES = 32
MAXIMUM_SEED, MAXIMUM_ROUNDS = 999, 1000
SEED_LIMIT = MAXIMUM_ROUNDS * (MAXIMUM_SEED + 1)  # every search game seed is below this


def game_seed(seed: int, round_index: int) -> int:
    """The default generators' seed for a round's games, below SEED_LIMIT for every seed and round
    the tool accepts."""
    return MAXIMUM_ROUNDS * seed + round_index


def perturb(
    parameters: ScriptedParameters, sigma: float, generator: torch.Generator
) -> ScriptedParameters:
    """Every scalar and ramp end times exp(sigma N(0, 1)) (a zero ramp end becomes a small
    positive one), aggression plus 2 sigma N(0, 1); the ramps' probabilities are kept."""

    def normal() -> float:
        return float(torch.randn(1, generator=generator))

    values = {}
    for field in dataclasses.fields(parameters):
        value = getattr(parameters, field.name)
        if isinstance(value, Ramp):
            ends = [
                end * math.exp(sigma * normal())
                if end != 0.0
                else 0.1 * sigma * float(torch.rand(1, generator=generator))
                for end in (value.low, value.high)
            ]
            low, high = sorted(ends)
            values[field.name] = Ramp(low, max(high, low + 1e-3), value.at_low, value.at_high)
        elif field.name == "aggression":
            values[field.name] = value + 2.0 * sigma * normal()
        else:
            values[field.name] = value * math.exp(sigma * normal())
    values["recovery_shield_fraction"] = min(values["recovery_shield_fraction"], 0.9)
    return ScriptedParameters(**values)


def scores(
    candidates: list[ScriptedParameters], opponent: str, games: int, device: torch.device
) -> dict[str, list[float]]:
    """Per preset, each candidate's score over `games` fresh games against the opponent."""
    out = {}
    for name in PRESET_NAMES:
        game = PRESETS[name]
        card = characteristics(api.ship_types(device), api.rules_of(game))
        rows = table([parameters for parameters in candidates for _ in range(games)], device)
        rival = (
            Chaser(game.ships_per_team)
            if opponent == "chaser"
            else Scripted(card, STARTING_PARAMETERS)
        )
        torch._dynamo.reset()  # a fresh rollout per match; the compile cache is not reused
        results = first_results(
            game,
            (Pool("search", 1.0, ("player", "opponent")),),
            {"player": Scripted(card, unpack(rows)), "opponent": rival},
            "player",
            len(candidates) * games,
            device,
        )
        points = (results > 0).float() + 0.5 * (results == 0).float()
        out[name] = points.view(len(candidates), games).mean(1).tolist()
    return out


def best_of_last_round(path: str) -> ScriptedParameters:
    """The parameters of the last round's best candidate (by combined score) in the search log at
    path."""
    rows = [json.loads(line) for line in open(path)]
    last = max(row["round"] for row in rows)
    top = max((row for row in rows if row["round"] == last), key=lambda row: row["combined"])
    values = unpack(torch.tensor([top["parameters"]]))
    return ScriptedParameters(
        **{
            field.name: (
                Ramp(*(float(end) for end in dataclasses.astuple(value)))
                if isinstance(value := getattr(values, field.name), Ramp)
                else float(value)
            )
            for field in dataclasses.fields(values)
        }
    )


def main() -> None:
    """Run the search the module describes, appending every candidate's scores and parameters to the
    log."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--games", type=int, default=128, help="per candidate and match")
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--decay", type=float, default=0.6, help="sigma's factor per round")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", help="a log to continue from (its last round's best)")
    parser.add_argument("--log", default="scripted_search.jsonl")
    args = parser.parse_args()
    if not 0 <= args.seed <= MAXIMUM_SEED or args.rounds > MAXIMUM_ROUNDS:
        raise SystemExit(f"seed in [0, {MAXIMUM_SEED}] and at most {MAXIMUM_ROUNDS} rounds")
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(args.seed)
    best = STARTING_PARAMETERS if args.resume is None else best_of_last_round(args.resume)
    sigma, start = args.sigma, time.perf_counter()
    with bench_lock(), open(args.log, "a") as log:
        for round_index in range(args.rounds):
            torch.manual_seed(game_seed(args.seed, round_index))
            candidates = [best] + [perturb(best, sigma, generator) for _ in range(CANDIDATES - 1)]
            matches = {
                opponent: scores(candidates, opponent, args.games, device) for opponent in OPPONENTS
            }
            combined = [
                sum(
                    matches[opponent][name][index]
                    for opponent in OPPONENTS
                    for name in PRESET_NAMES
                )
                / 4
                for index in range(CANDIDATES)
            ]
            for index, parameters in enumerate(candidates):
                log.write(
                    json.dumps(
                        dict(
                            round=round_index,
                            index=index,
                            combined=combined[index],
                            **{
                                opponent: [matches[opponent][name][index] for name in PRESET_NAMES]
                                for opponent in OPPONENTS
                            },
                            parameters=pack(parameters),
                        )
                    )
                    + "\n"
                )
            log.flush()
            winner = max(range(CANDIDATES), key=lambda index: combined[index])
            best, sigma = candidates[winner], sigma * args.decay
            print(
                f"round {round_index}: best {combined[winner]:.3f} (candidate {winner}) at "
                f"{time.perf_counter() - start:.0f} s",
                flush=True,
            )
    print("best", repr(best))


if __name__ == "__main__":
    main()
