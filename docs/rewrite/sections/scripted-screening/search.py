"""A short CMA-ES search of the yardstick (docs/rewrite/scripted-screening.md "CMA-ES search").

Ten rows (`ROWS`) in the table's own units (log rows in log space), each scaled by its box
width, so a coordinate of 1 is one box half-width (one whole box for an upward row); the
search starts at the yardstick (every coordinate 0). Every other row stays at its default.
Fitness is the screening's skill (screen.py: mean score over the panel and the 18
conditions, equal weight per cell), played into the screening's store. A floor on the step
size keeps the population a spread of about a quarter box, so the distribution's mean is
pushed toward a plateau rather than a peak; the result is that mean, not the best candidate.

    search.py STORE search [--population 24 --games 6 --minutes 42]
    search.py STORE confirm [--confirm-games 64]      # the mean and the yardstick, many games; local25 around both
    search.py STORE held-out                  # both against 8 personas outside the panel, 32 games a cell

Deterministic given the store: pycma is seeded and every fitness is read back from the store,
so a rerun replays the same generations and plays only the missing games (the resume).
Each generation is appended to search.jsonl (the durable record) after its games are stored.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cma
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen  # noqa: E402  (the panel, the conditions, play)

from lean.eval import ScriptedAgent, Store  # noqa: E402
from lean.eval.store import conditions_of  # noqa: E402
from lean.scripted import ScriptedConfig  # noqa: E402
from lean.scripted.parameters import LOG, PARAMETERS, UPWARD  # noqa: E402

ROWS = ("lead_turn_scale", "lead_gain", "yo_yo_time", "fight_speed", "closing_time", "plan_threshold",
        "in_range", "nose_tail", "aligned", "temperature")
# Search bounds in box half-widths (upward rows: whole boxes). The screening's boxes, except
# lead_gain and fight_speed widened upward to three half-widths: lead_gain 0.8 to 1.6 (box
# 0.8 to 1.2), fight_speed e^-0.2 to e^0.6 = 0.82 to 1.82 (box 0.82 to 1.22).
BOUNDS = {name: (0.0, 1.0) if PARAMETERS[name].law == UPWARD else (-1.0, 1.0) for name in ROWS}
BOUNDS["lead_gain"] = (-1.0, 3.0)
BOUNDS["fight_speed"] = (-1.0, 3.0)
SIGMA0 = 0.3  # box half-widths
MINIMUM_STEP = 0.1  # the step floor, box half-widths (a quarter box's uniform has sd 0.144)
SEED = 20261124
PATIENCE = 4  # generations without the best population mean rising beyond its error
MINIMUM_GENERATIONS = 8
RECORD = HERE / "search.jsonl"


def value_of(name: str, coordinate: float) -> float:
    row = PARAMETERS[name]
    if row.law == LOG:
        return row.default * math.exp(row.width * coordinate)
    return row.default + row.width * coordinate


def config_of(coordinates) -> ScriptedConfig:
    return ScriptedConfig().replace(**{name: value_of(name, float(x)) for name, x in zip(ROWS, coordinates, strict=True)})


def skill(store: Store, agent: ScriptedAgent) -> tuple[float, float, list[float], int]:
    """(skill, its standard error, the mean score per condition, games) of an agent with games
    in every (panel member, condition) cell, as screen.skill_of, from the store's counts."""
    members = screen.panel()
    scores, variance, per_condition, games = [], 0.0, [], 0
    for game_config in screen.game_configs():
        for sizes in screen.SIZES:
            condition = conditions_of(game_config, sizes)[0]
            row = []
            for member in members:
                count = store.counts(agent.id, member.id, condition)
                n = count["games"]
                if n < 2:
                    raise RuntimeError(f"{agent.id}: {n} games against {member.id}")
                mean = (count["wins"] + 0.5 * count["draws"]) / n
                square = (count["wins"] + 0.25 * count["draws"]) / n
                variance += max(square - mean * mean, 0.0) / (n - 1)
                row.append(mean)
                games += n
            scores += row
            per_condition.append(float(np.mean(row)))
    return float(np.mean(scores)), math.sqrt(variance) / len(scores), per_condition, games


def search(store: Store, population: int, games: int, minutes: float) -> None:
    started = time.perf_counter()
    lower, upper = zip(*(BOUNDS[name] for name in ROWS), strict=True)
    options = {"popsize": population, "seed": SEED, "bounds": [list(lower), list(upper)],
               "minstd": MINIMUM_STEP, "verbose": -9, "CMA_mirrors": 0}
    es = cma.CMAEvolutionStrategy([0.0] * len(ROWS), SIGMA0, options)
    done = {}
    if RECORD.exists():
        for line in RECORD.read_text().splitlines():
            entry = json.loads(line)
            done[entry["generation"]] = entry
    best, best_error, best_generation = -math.inf, 0.0, 0
    generation = 0
    while True:
        generation += 1
        if generation not in done and (time.perf_counter() - started) / 60.0 > minutes:
            screen.log(f"time cap: {minutes} min reached before generation {generation}")
            break
        solutions = es.ask()
        agents = [ScriptedAgent.of(config_of(x)) for x in solutions]
        gen_started = time.perf_counter()
        screen.play(store, list(dict.fromkeys(agents)), games, f"cma{generation}")
        results = [skill(store, agent) for agent in agents]
        fitness = np.array([item[0] for item in results])
        es.tell(solutions, [-item for item in fitness])
        mean_fitness = float(fitness.mean())
        mean_error = float(fitness.std(ddof=1) / math.sqrt(len(fitness)))
        if mean_fitness > best + best_error:
            best, best_error, best_generation = mean_fitness, mean_error, generation
        favourite = es.result.xfavorite
        entry = {
            "generation": generation, "seconds": time.perf_counter() - gen_started,
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
            "population_mean_skill": mean_fitness, "population_mean_error": mean_error,
            "best_candidate_skill": float(fitness.max()), "candidate_error": float(np.median([r[1] for r in results])),
            "sigma": float(es.sigma), "step": [float(item) for item in es.stds],
            "mean": {name: float(x) for name, x in zip(ROWS, favourite, strict=True)},
            "mean_values": {name: value_of(name, x) for name, x in zip(ROWS, favourite, strict=True)},
            "solutions": [[float(item) for item in x] for x in solutions], "skills": fitness.tolist(),
        }
        if generation not in done:
            with RECORD.open("a") as file:
                file.write(json.dumps(entry) + "\n")
        screen.log(f"generation {generation}: population mean {mean_fitness:.4f} +- {mean_error:.4f}, best "
                   f"candidate {fitness.max():.4f}, sigma {es.sigma:.3f}, {entry['seconds']:.0f} s, "
                   f"elapsed {entry['elapsed_minutes']:.1f} min; mean "
                   + " ".join(f"{name}={value_of(name, x):.3g}" for name, x in zip(ROWS, favourite, strict=True)))
        if generation >= MINIMUM_GENERATIONS and generation - best_generation >= PATIENCE:
            screen.log(f"stop: no rise beyond the error since generation {best_generation} ({best:.4f} +- {best_error:.4f})")
            break
    final = es.result.xfavorite
    candidate = {
        "generations": generation,
        "minimum_step": MINIMUM_STEP, "bounds": BOUNDS,
        "coordinates": {name: float(x) for name, x in zip(ROWS, final, strict=True)},
        "searched": {name: value_of(name, x) for name, x in zip(ROWS, final, strict=True)},
        "values": config_of(final).values,
    }
    (HERE / "cma_candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
    screen.log(f"wrote {HERE / 'cma_candidate.json'}")


def marginals(per_condition: list[float]) -> dict:
    """Mean score by radius, team size and spawn rule (the 18 conditions in screen's order)."""
    conditions = [(game_config.arena_radius, game_config.spawn, sizes) for game_config in screen.game_configs()
                  for sizes in screen.SIZES]
    out = {}
    for key, position in (("radius", 0), ("spawn", 1), ("sizes", 2)):
        levels = dict.fromkeys(str(condition[position]) for condition in conditions)
        out[key] = {level: float(np.mean([score for condition, score in zip(conditions, per_condition, strict=True)
                                          if str(condition[position]) == level])) for level in levels}
    return out


def confirm(store: Store, games: int) -> None:
    record = json.loads((HERE / "cma_candidate.json").read_text())
    candidate = ScriptedAgent.of(ScriptedConfig(record["values"]))
    yardstick = ScriptedAgent.of()
    screen.play(store, [candidate, yardstick], games, "confirm")
    points = screen.sobol_points(store, "local25", 256, 0)
    around = [screen.agent_at(row, 0.25, candidate.config()) for row in points]
    screen.play(store, around, 2, "confirm-local25")
    out = {}
    for name, agent in (("candidate", candidate), ("yardstick", yardstick)):
        value, error, per_condition, played = skill(store, agent)
        out[name] = {"skill": value, "error": error, "games": played, "by_condition": per_condition,
                     "marginals": marginals(per_condition)}
    for name, centre in (("candidate", candidate), ("yardstick", yardstick)):
        local = [skill(store, screen.agent_at(row, 0.25, centre.config())) for row in points]
        values = np.array([item[0] for item in local])
        out[name]["local25"] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                                "error_of_mean": float(values.std(ddof=1) / math.sqrt(len(values))),
                                "median_noise": float(np.median([item[1] for item in local])),
                                "quantile5": float(np.quantile(values, 0.05)),
                                "share_above_yardstick": float(np.mean(values > out["yardstick"]["skill"])),
                                "skills": values.tolist()}
    paired = np.array(out["candidate"]["local25"]["skills"]) - np.array(out["yardstick"]["local25"]["skills"])
    out["local25_paired_difference"] = {"mean": float(paired.mean()),
                                        "error": float(paired.std(ddof=1) / math.sqrt(len(paired)))}
    record["confirm"] = out
    (HERE / "cma_candidate.json").write_text(json.dumps(record, indent=1) + "\n")
    for name in ("candidate", "yardstick"):
        entry = out[name]
        screen.log(f"{name}: {entry['skill']:.4f} +- {entry['error']:.4f} ({entry['games']} games); local25 mean "
                   f"{entry['local25']['mean']:.4f} sd {entry['local25']['sd']:.4f}; {entry['marginals']}")
    screen.log(f"local25 paired difference {out['local25_paired_difference']}")


HELD_OUT_SEED = 20261125


def held_out(store: Store, games: int) -> None:
    """The candidate and the yardstick against 8 personas drawn (seeded) from the full boxes,
    none in the search's panel, at the 18 conditions: whether the gain is the panel's. Also
    each one's mean score against every member of the search's panel."""
    import torch

    from lean.eval import Evaluator
    from lean.scripted.parameters import SAMPLED

    record = json.loads((HERE / "cma_candidate.json").read_text())
    candidate, yardstick = ScriptedAgent.of(ScriptedConfig(record["values"])), ScriptedAgent.of()
    unit = torch.rand((8, len(SAMPLED)), generator=torch.Generator().manual_seed(HELD_OUT_SEED), dtype=torch.float64)
    members = [screen.agent_at(row) for row in unit]
    for number, game_config in enumerate(screen.game_configs()):
        requests = [(agent, member, sizes, games) for agent in (candidate, yardstick) for member in members
                    for sizes in screen.SIZES]
        Evaluator(store, game_config, envs=4096, seed=900 + 7 * number).ensure(requests)
        screen.log(f"held-out call {number + 1}/6; store {store.total_games()}")
    out = {"seed": HELD_OUT_SEED, "games_per_cell": games}
    for name, agent in (("candidate", candidate), ("yardstick", yardstick)):
        for panel_name, panel in (("held_out", members), ("panel", screen.panel())):
            scores, variance = [], 0.0
            per_member = []
            for member in panel:
                row = []
                for game_config in screen.game_configs():
                    for sizes in screen.SIZES:
                        count = store.counts(agent.id, member.id, conditions_of(game_config, sizes)[0])
                        n = count["games"]
                        mean = (count["wins"] + 0.5 * count["draws"]) / n
                        variance += max((count["wins"] + 0.25 * count["draws"]) / n - mean * mean, 0.0) / (n - 1)
                        row.append(mean)
                scores += row
                per_member.append(float(np.mean(row)))
            out.setdefault(name, {})[panel_name] = {"skill": float(np.mean(scores)),
                                                    "error": math.sqrt(variance) / len(scores),
                                                    "per_member": per_member}
            screen.log(f"{name} {panel_name}: {np.mean(scores):.4f} +- {math.sqrt(variance) / len(scores):.4f}; "
                       f"per member {[round(item, 3) for item in per_member]}")
    record["held_out"] = out
    (HERE / "cma_candidate.json").write_text(json.dumps(record, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store")
    parser.add_argument("command", choices=("search", "confirm", "held-out"))
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--games", type=int, default=6, help="games per (panel member, condition) cell")
    parser.add_argument("--minutes", type=float, default=42.0, help="no new generation after this")
    parser.add_argument("--confirm-games", type=int, default=64, help="confirm: games per cell")
    arguments = parser.parse_args()
    store = Store(arguments.store)
    if arguments.command == "search":
        search(store, arguments.population, arguments.games, arguments.minutes)
    elif arguments.command == "confirm":
        confirm(store, arguments.confirm_games)
    else:
        held_out(store, 32)


if __name__ == "__main__":
    main()
