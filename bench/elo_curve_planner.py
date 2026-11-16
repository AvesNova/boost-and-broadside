"""The unified evaluator's Elo planner on a learning curve's field, offline (CPU).

Ground truth: a fitted table (`bnb pool table` output, default dry run 1's curves,
docs/rewrite/sections/unified-eval/dry-v24-curves-table.txt), one mode; a bound is placed
`BOUND_STEP` Elo past its value and an undetermined player midway between its list
neighbours. The field is the box check's (`box/unified_check.sh`): the anchor, every BC
snapshot, every RL snapshot in step order, and the champion (the last RL snapshot's
policy) last. Games are drawn from Bradley-Terry (draws optional, half a win each) and
fed to `EloTarget` through the `Planner`, `--environments` games in flight, a random
`--finish` share of them ending per round (the engine's refills come in bursts).

Prints and writes (--out, JSON per mode, rewritten every `--every` games): the games, the
worst standard error, the players without a rating, the true expected score of every game
played (its distribution), and the errors against the truth.

    uv run --no-sync python bench/elo_curve_planner.py --mode frontline --out OUT.json
"""

import argparse
import json
import math
import pathlib
import random
import re
import time

import numpy

from bnb.league.ratings import SCALE
from bnb.league.unified.cells import Condition, Outcome
from bnb.league.unified.planner import Planner
from bnb.league.unified.targets import EloTarget

TABLE = pathlib.Path(__file__).parent.parent / "docs/rewrite/sections/unified-eval"
TABLE = TABLE / "dry-v24-curves-table.txt"
PRESETS = {
    "frontline": "real_scale_frontline",
    "deathmatch": "real_scale_deathmatch",
    "elimination": "real_scale_elimination",
}
BOUND_STEP = 300.0  # Elo past a one-sided bound (it held at 5%)


def read_table(path: pathlib.Path, mode: str) -> dict[str, float]:
    """name -> Elo of the mode's section; bounds stepped out, undetermined imputed (nan)."""
    ratings, section = {}, None
    for line in path.read_text().splitlines():
        header = re.match(r"^(\w+)_5v5 ", line)
        if header:
            section = header.group(1)
            continue
        if section != mode:
            continue
        found = re.match(r"^\s+(\S+)\s+(.*?)\s+(\d+) games$", line)
        if not found:
            continue
        name, value = found.group(1), found.group(2)
        short = (
            name
            if name == "scripted"
            else re.sub(r"dry-v24-s1-(\w+)/step_(\d+)/.*", r"\1_\2", name)
        )
        number = re.search(r"-?\d+\.\d+", value)
        if "undetermined" in value:
            ratings[short] = math.nan
        elif value.startswith(">"):
            ratings[short] = float(number.group()) + BOUND_STEP
        elif value.startswith("<"):
            ratings[short] = float(number.group()) - BOUND_STEP
        else:
            ratings[short] = float(number.group())
    return ratings


def field(ratings: dict[str, float]) -> tuple[list[str], dict[str, float]]:
    """The box check's field order and the truth (the champion = the last RL snapshot)."""

    def steps(stage):
        return sorted(
            (n for n in ratings if n.startswith(stage + "_")), key=lambda n: int(n.split("_")[1])
        )

    order = ["scripted"] + steps("bc") + steps("rl")
    truth = dict(ratings)
    for index, name in enumerate(order):  # undetermined: midway between list neighbours
        if math.isnan(truth[name]):
            known = [truth[n] for n in order[max(index - 1, 1) : index + 2] if n != name]
            truth[name] = float(numpy.nanmean(known))
    truth["champion"] = truth[order[-1]]
    return order + ["champion"], truth


def simulate(mode, environments, finish, draws, seed, budget, out, every, log=print):
    ratings = read_table(TABLE, mode)
    players, truth = field(ratings)
    condition = Condition(PRESETS[mode], 5, 5)
    target = EloTarget(f"curves/{mode}", condition, players, stderr=60.0, budget=budget)
    planner = Planner([target])
    rng = random.Random(seed)
    flight: list = []
    expected: list[float] = []
    games, started, last = 0, time.time(), 0

    def refill(count):
        for _ in range(count):
            chosen = planner.choose(5, condition.preset)
            if chosen is None:
                return
            flight.append(chosen[0])

    def summary(final=False):
        errors = target.errors(fresh=True)
        report = target.report()
        unrated = {n: s for n, s in report["status"].items() if s != "rated"}
        p = numpy.array(expected)
        far = numpy.minimum(p, 1 - p)
        late = far[environments:]
        fitted = numpy.array(
            [report["ratings"][n] - truth[n] for n in players[1:] if report["ratings"][n]]
        )
        held = dict(
            mode=mode,
            games=games,
            done=target.done(),
            met=target.met(),
            worst_stderr=float(numpy.max(errors)),
            unresolved=unrated,
            seconds=time.time() - started,
            expected_score_quantiles={
                str(q): float(numpy.quantile(far, q)) for q in (0.01, 0.05, 0.1, 0.25, 0.5)
            }
            if len(far)
            else {},
            share_beyond={
                "0.95": float((far < 0.05).mean()) if len(far) else 0.0,
                "0.99": float((far < 0.01).mean()) if len(far) else 0.0,
            },
            share_beyond_after_first_wave={  # the games planned with a fit in hand
                "0.95": float((late < 0.05).mean()) if len(late) else 0.0,
                "0.99": float((late < 0.01).mean()) if len(late) else 0.0,
            },
            mean_information_per_game=float((p * (1 - p)).mean()) if len(p) else 0.0,
            error_over_stderr_rms=float(
                numpy.sqrt(
                    numpy.mean(
                        [
                            ((report["ratings"][n] - truth[n]) / report["errors"][n]) ** 2
                            for n in players[1:]
                            if report["errors"].get(n)
                        ]
                    )
                )
            )
            if len(fitted)
            else None,
            final=final,
        )
        out.write_text(json.dumps(held, indent=1))
        return held

    refill(environments)
    while flight and not target.done():
        ending = max(1, int(len(flight) * finish))
        rng.shuffle(flight)
        ended, flight[:] = flight[:ending], flight[ending:]
        for cell in ended:
            gap = (truth[cell.first] - truth[cell.second]) / SCALE
            p = 1.0 / (1.0 + math.exp(-gap))
            expected.append(p)
            u = rng.random()
            if draws and u < draws * 4 * p * (1 - p):  # draws likeliest when even
                score = 0.5
            else:
                score = 1.0 if rng.random() < p else 0.0
            planner.finished(cell, Outcome(score, seconds=60.0))
            games += 1
        refill(len(ended))
        if games - last >= every:
            last = games
            held = summary()
            log(
                f"{mode}: {games} games, worst SE {held['worst_stderr']:.0f}, "
                f"unresolved {len(held['unresolved'])}, {held['seconds']:.0f} s"
            )
    return summary(final=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=sorted(PRESETS), required=True)
    parser.add_argument("--environments", type=int, default=2048)
    parser.add_argument("--finish", type=float, default=0.02)
    parser.add_argument("--draws", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--every", type=int, default=2000)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    held = simulate(
        args.mode,
        args.environments,
        args.finish,
        args.draws,
        args.seed,
        args.budget,
        args.out,
        args.every,
    )
    print(json.dumps(held, indent=1))


if __name__ == "__main__":
    main()
