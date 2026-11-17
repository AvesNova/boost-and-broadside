"""The static Elo-curve design (`bnb.league.unified.static_target`) offline (CPU), against
the adaptive planner's numbers (`bench/elo_curve_planner.py`, results.md).

Ground truth: dry run 1's fitted curves table (`elo_curve_planner.read_table`: a bound
placed 300 Elo past its value, an undetermined player between its neighbours), one mode.
The field: the anchor and the run's chain (BC's snapshots, then RL's; the champion is an
alias of the last RL snapshot, so not a player). Games are Bradley-Terry draws from the
truth, fed through the `Planner`: `--environments` games in flight, a random `--finish`
share of them ending per round, as the engine's refills. Prints each stage's designed
games, the final errors (the fit's), the errors against the truth and every WARNING.

    uv run --no-sync python bench/elo_static_design.py --mode elimination
"""

import argparse
import json
import math
import random

import numpy

from bench.elo_curve_planner import PRESETS, TABLE, field, read_table
from bnb.league.ratings import SCALE
from bnb.league.unified import static_design
from bnb.league.unified.cells import Condition, Outcome
from bnb.league.unified.planner import Planner
from bnb.league.unified.static_target import StaticEloTarget


def target_for(
    order: list[str], mode: str, stderr: float, reach: int = 2, pilot: float = 1.0
) -> StaticEloTarget:
    found = [
        static_design.Checkpoint(name, "dry", name.split("_")[0], int(name.split("_")[1]))
        for name in order
    ]
    chains = static_design.chains_of(found)
    names = static_design.design_order(chains)
    chain = chains["dry"]
    return StaticEloTarget(
        f"curves/{mode}",
        Condition(PRESETS[mode], 5, 5),
        ["scripted"] + names,
        stderr,
        names,
        static_design.candidate_pairs(chains, "scripted", neighbours=reach),
        static_design.prior_ratings(chains),
        pilot,
        neighbours=[(a.name, b.name) for a, b in zip(chain, chain[1:])],
    )


def simulate(
    mode: str, environments: int, finish: float, seed: int, stderr: float, reach=2, pilot=1.0
) -> dict:
    players, truth = field(read_table(TABLE, mode))
    order = [p for p in players if p not in ("scripted", "champion")]
    target = target_for(order, mode, stderr, reach, pilot)
    planner = Planner([target])
    rng = random.Random(seed)
    flight: list = []
    stage_games = {}

    def refill(count: int) -> None:
        for _ in range(count):
            chosen = planner.choose(5, target.condition.preset)
            if chosen is None:
                return
            flight.append(chosen[0])

    refill(environments)
    games = 0
    while flight:
        ending = max(1, int(len(flight) * finish))
        rng.shuffle(flight)
        ended, flight[:] = flight[:ending], flight[ending:]
        for cell in ended:
            p = 1.0 / (1.0 + math.exp(-(truth[cell.first] - truth[cell.second]) / SCALE))
            stage = target.stage
            planner.finished(cell, Outcome(1.0 if rng.random() < p else 0.0, seconds=60.0))
            games += 1
            stage_games[stage] = stage_games.get(stage, 0) + 1
        refill(environments - len(flight))  # a parked environment asks again
    report = target.report()
    errors = {n: report["errors"].get(n) for n in target.players[1:]}
    finite = [e for e in errors.values() if e is not None]
    deviations = [
        (report["ratings"][n] - truth[n]) / errors[n]
        for n in target.players[1:]
        if errors[n] is not None and report["ratings"][n] is not None
    ]
    over = {
        n: (round(e, 1) if e is not None else None)
        for n, e in errors.items()
        if e is None or e > stderr
    }
    return dict(
        mode=mode,
        reach=reach,
        pilot=pilot,
        games=games,
        designed={k: v["games"] for k, v in report["design"]["stages"].items()},
        played_by_stage={str(k): v for k, v in stage_games.items()},
        worst_stderr=max(finite) if finite else None,
        mean_stderr=float(numpy.mean(finite)) if finite else None,
        above_goal=over,
        error_over_stderr_rms=float(numpy.sqrt(numpy.mean(numpy.square(deviations)))),
        stopped=report["stopped"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=sorted(PRESETS), default="elimination")
    parser.add_argument("--environments", type=int, default=2048)
    parser.add_argument("--finish", type=float, default=0.02)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--stderr", type=float, default=60.0)
    parser.add_argument("--reach", type=int, default=2, help="chain neighbours each side")
    parser.add_argument("--pilot", type=float, default=1.0, help="stage 1's goal over the goal")
    args = parser.parse_args()
    for seed in range(args.seeds):
        held = simulate(
            args.mode, args.environments, args.finish, seed, args.stderr, args.reach, args.pilot
        )
        print(json.dumps(held))
        for name, error in held["above_goal"].items():
            print(f"WARNING {held['mode']} seed {seed}: {name} at {error} (goal {args.stderr})")


if __name__ == "__main__":
    main()
