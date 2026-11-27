"""Round 2, step 4: a CMA-ES search of the yardstick's rows and the crowd endpoints for the
mean Elo against the old yardstick over 1v1, 4v4 and 16v16 (equal weight), each size at its
study clock (skill2.CLOCKS), fields and fog.

    tune.py DATA cma --minutes 12 [--games 256]   # resumable: DATA/cma.pickle, DATA/cma.jsonl
    tune.py DATA verify NAME=JSON ... --sizes 1 2 4 8 16 --target-error 0.012
    tune.py DATA report NAME=JSON ... --sizes 1 2 4 8 16 --out OUT.json

NAME=JSON: a config as rows over the defaults (a JSON object). The crowd measure screen
(step 3) is a `verify` of the same endpoints under each measure.

The search coordinates (`Space`): each tuned sampled row's unit coordinate u (its spread-1
box at [0, 1], allowed to [-0.5, 1.5]: half a box past either edge), and each crowd row's
log ratio in units of its row's box (v: the crowd value is the duel value x e^(2 width v),
v in [-1.5, 1.5], 0 off). A candidate's Elo per size is from its score against the old
yardstick (the score shrunk half a game towards 0.5). Every game goes into the SQLite store of
its job (`DATA/tune.sqlite`; verification `DATA/verify.sqlite`, separate, so its games are
fresh), and the CMA state is pickled after each generation.
"""

import argparse
import dataclasses
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill2  # noqa: E402

from lean import game  # noqa: E402
from lean.eval import Evaluator, ScriptedAgent, Store  # noqa: E402
from lean.eval.rating import ELO_PER_NAT  # noqa: E402
from lean.scripted import PARAMETERS, ScriptedConfig  # noqa: E402
from lean.scripted.parameters import SAMPLED  # noqa: E402

SIZES = (1, 4, 16)
# Inert in round 1 (boundary, seek, disengaged energy) and in round 2's fits: not searched.
INERT = ("boundary_weight", "seek_weight", "disengaged_energy_weight")
U_LOW, U_HIGH = -0.5, 1.5
V_LIMIT = 1.5


def log(text):
    print(f"{time.strftime('%H:%M:%S')} {text}", file=sys.stderr, flush=True)


def elo_of(score, games):
    rate = (score + 0.5) / (games + 1.0)
    return ELO_PER_NAT * math.log(rate / (1.0 - rate))


def row_value(name, u, centre=None):
    """A sampled row at unit coordinate u of its spread-1 box around `centre` (default the
    table's default), any u (the box's laws extended)."""
    row = PARAMETERS[name]
    value = row.default if centre is None else centre
    if row.law == "log":
        return value * math.exp(row.width * (2.0 * u - 1.0))
    return value + row.width * (2.0 * u - 1.0)


def unit_of(name, value):
    row = PARAMETERS[name]
    if row.law == "log":
        return 0.5 + math.log(value / row.default) / (2.0 * row.width)
    return 0.5 + (value - row.default) / (2.0 * row.width)


@dataclasses.dataclass
class Space:
    rows: tuple  # tuned sampled rows (unit coordinates)
    crowded: tuple  # rows with a crowd endpoint (v)
    measure: float = 0.0
    half: float = 1.0

    @property
    def dimension(self):
        return len(self.rows) + len(self.crowded)

    def config(self, z) -> ScriptedConfig:
        z = np.asarray(z, dtype=float)
        values = {}
        for name, u in zip(self.rows, z[: len(self.rows)], strict=True):
            values[name] = row_value(name, float(np.clip(u, U_LOW, U_HIGH)))
        for name, v in zip(self.crowded, z[len(self.rows) :], strict=True):
            v = float(np.clip(v, -V_LIMIT, V_LIMIT))
            values[f"{name}_crowd"] = math.exp(2.0 * PARAMETERS[name].width * v)
        if self.crowded:
            values["crowd_measure"] = self.measure
            values["crowd_half"] = self.half
        return ScriptedConfig().replace(**values)

    def centre(self):
        return np.array([0.5] * len(self.rows) + [0.0] * len(self.crowded))


def evaluators(store, envs):
    return {
        size: Evaluator(
            store, skill2.config_of(size), envs=envs, minimum_envs=envs, seed=4242 + size
        )
        for size in SIZES
    }


def rate(evaluators_, agents, games, sizes=SIZES):
    """Elo per size of each agent against the old yardstick from `games` games each."""
    yard = ScriptedAgent.of()
    out = [{} for _ in agents]
    for size in sizes:
        rows = evaluators_[size].ensure([(yard, agent, (size, size), games) for agent in agents])
        for index, row in enumerate(rows):
            score = row["losses"] + 0.5 * row["draws"]  # the candidate is seat B
            out[index][size] = elo_of(score, row["games"]) + 1000.0
    return out


def cma_search(data, minutes, games, envs, space, popsize, sigma, x0):
    import cma

    started = time.perf_counter()
    state = data / "cma.pickle"
    journal = data / "cma.jsonl"
    if state.exists():
        es = pickle.loads(state.read_bytes())
        log(f"resumed CMA at generation {es.countiter}")
    else:
        es = cma.CMAEvolutionStrategy(
            list(x0),
            sigma,
            {
                "popsize": popsize,
                "seed": 1128,
                "bounds": [
                    [U_LOW] * len(space.rows) + [-V_LIMIT] * len(space.crowded),
                    [U_HIGH] * len(space.rows) + [V_LIMIT] * len(space.crowded),
                ],
                "verbose": -9,
            },
        )
    store = Store(data / "tune.sqlite")
    evaluating = evaluators(store, envs)
    while (time.perf_counter() - started) / 60.0 < minutes:
        solutions = es.ask()
        agents = [ScriptedAgent.of(space.config(z)) for z in solutions]
        elos = rate(evaluating, agents, games)
        means = [float(np.mean([e[s] for s in SIZES])) for e in elos]
        es.tell(solutions, [-m for m in means])
        mean_z = es.mean
        with journal.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "generation": es.countiter,
                        "time": time.strftime("%H:%M:%S"),
                        "solutions": [list(map(float, z)) for z in solutions],
                        "elo": [{str(k): v for k, v in e.items()} for e in elos],
                        "mean": means,
                        "cma_mean": list(map(float, mean_z)),
                        "sigma": float(es.sigma),
                        "space": dataclasses.asdict(space),
                    }
                )
                + "\n"
            )
        state.write_bytes(pickle.dumps(es))
        best = int(np.argmax(means))
        log(
            f"generation {es.countiter}: mean Elo {np.mean(means):.0f}, best {means[best]:.0f} "
            f"({', '.join(f'{s}v{s} {elos[best][s]:.0f}' for s in SIZES)}), sigma {es.sigma:.3f}"
        )
    log("time bound")


def verify(data, named, sizes, target, minutes, envs, opponents):
    """Fresh games (a store of its own): each named config against the old yardstick (and
    `opponents`, other named configs, as extra pairings) at every size to a target error."""
    started = time.perf_counter()
    store = Store(data / "verify.sqlite")
    yard = ScriptedAgent.of()
    agents = {name: ScriptedAgent.of(config) for name, config in named.items()}
    for size in sizes:
        evaluator = Evaluator(
            store, skill2.config_of(size), envs=envs, minimum_envs=envs, seed=777 + size
        )
        pairings = [(yard, agent, (size, size)) for agent in agents.values()]
        names = list(agents)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                if first in opponents or second in opponents:
                    pairings.append((agents[first], agents[second], (size, size)))
        while (time.perf_counter() - started) / 60.0 < minutes:
            rows = evaluator.adaptive(
                pairings, target_error=target, budget=16384, round_games=16384
            )
            worst = max(row["error"] for row in rows)
            log(f"verify {size}v{size}: worst error {worst:.4f}")
            if worst <= target:
                break
        else:
            log("time bound")
            return


def report_verify(data, named, sizes, out):
    store = Store(data / "verify.sqlite")
    agents = {"yardstick": ScriptedAgent.of()} | {
        name: ScriptedAgent.of(config) for name, config in named.items()
    }
    result = {}
    for size in sizes:
        evaluator = Evaluator(store, skill2.config_of(size))
        ratings = {
            identity: (r, e)
            for identity, r, e in evaluator.elo((size, size), agents=list(agents.values()))
        }
        result[f"{size}v{size}"] = {
            name: ratings.get(agent.id, (None, None)) for name, agent in agents.items()
        }
        conditions, _ = evaluator.conditions((size, size))
        result[f"{size}v{size}"]["_games"] = store.connection.execute(
            "SELECT COUNT(*) FROM games WHERE conditions = ?", (conditions,)
        ).fetchone()[0]
    Path(out).write_text(json.dumps(result, indent=1))
    for key, entry in result.items():
        log(
            f"{key}: "
            + ", ".join(
                f"{name} {value[0]:.0f}+-{value[1]:.0f}"
                for name, value in entry.items()
                if name != "_games" and value[0] is not None
            )
        )


def load_named(items):
    named = {}
    for item in items:
        name, path = item.split("=", 1)
        named[name] = ScriptedConfig().replace(**json.loads(Path(path).read_text()))
    return named


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("command", choices=("cma", "verify", "report"))
    parser.add_argument("named", nargs="*")
    parser.add_argument("--space", type=Path, help="JSON: rows, crowded, measure, half, x0")
    parser.add_argument("--minutes", type=float, default=12.0)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--popsize", type=int, default=14)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--target-error", type=float, default=0.012)
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "cma":
        spec = json.loads(arguments.space.read_text())
        space = Space(tuple(spec["rows"]), tuple(spec["crowded"]), spec["measure"], spec["half"])
        x0 = spec.get("x0") or space.centre()
        cma_search(
            arguments.data,
            arguments.minutes,
            arguments.games,
            arguments.envs,
            space,
            arguments.popsize,
            arguments.sigma,
            x0,
        )
    elif arguments.command == "verify":
        verify(
            arguments.data,
            load_named(arguments.named),
            arguments.sizes,
            arguments.target_error,
            arguments.minutes,
            arguments.envs,
            set(arguments.opponents),
        )
    else:
        report_verify(arguments.data, load_named(arguments.named), arguments.sizes, arguments.out)


if __name__ == "__main__":
    main()
