"""Evaluation from the command line, every game kept in one SQLite store.

    python -m lean.eval play STORE AGENT AGENT [AGENT ...] [--sizes 4 4] [--against-first]
        (--games N | --target-error 0.02 [--budget N]) [--sobol NAME --count N]
    python -m lean.eval report STORE AGENT AGENT [...] [--sizes 4 4] [--against-first]
    python -m lean.eval elo STORE [--sizes 4 4] [--anchor AGENT]
    python -m lean.eval crossover STORE DIR/rl.pt [--weights ema] [--sizes 4 8] [--games 64]

AGENT: `yardstick` (the scripted agent's defaults), `scripted:FILE.json` (a JSON object of
parameter rows, the rest at their defaults), `net:CHECKPOINT[:live|ema|uniform]` (ema by
default), or an agent id already in the store. Pairings: every pair of the agents (round
robin), or each against the first (`--against-first`). `--sobol NAME --count N` adds the next
N points of the Sobol stream NAME (the persona box around the yardstick) as agents, each
against the first. Games are the core game (`GameConfig()`, the standard spawn) unless
changed by the game options; the stored games of the same conditions are reused.
"""

import argparse
import dataclasses
import itertools
import json
import sys
import time

import torch

from lean import game
from lean.eval import Evaluator, NetworkAgent, ScriptedAgent, Store, crossover, evaluator_round, games_for
from lean.scripted import ScriptedConfig


def parse_agent(text: str, store: Store):
    if text == "yardstick":
        return ScriptedAgent.of()
    if text.startswith("scripted:"):
        with open(text.removeprefix("scripted:")) as file:
            return ScriptedAgent.of(ScriptedConfig().replace(**json.load(file)))
    if text.startswith("net:"):
        path, _, weights = text.removeprefix("net:").partition(":")
        return NetworkAgent(path, weights or "ema")
    agent = store.agent(text)
    if agent is None:
        raise SystemExit(f"unknown agent {text!r}")
    return agent


def name_of(agent) -> str:
    if isinstance(agent, NetworkAgent):
        return f"{agent.id} net {agent.checkpoint} ({agent.weights})"
    return f"{agent.id} scripted{' (yardstick)' if agent == ScriptedAgent.of() else ''}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("play", "report", "elo", "crossover"))
    parser.add_argument("store")
    parser.add_argument("agents", nargs="*")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="seat A's and seat B's ships (play, report, elo; default 4 4); network sizes (crossover; default 4)")
    parser.add_argument("--against-first", action="store_true")
    parser.add_argument("--games", type=int, default=None, help="games per pairing (crossover: per match, 0 for v1's)")
    parser.add_argument("--target-error", type=float, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--sobol", default=None)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--anchor", default=None)
    parser.add_argument("--weights", choices=("ema", "uniform", "live"), default="ema")
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--seconds", type=float, default=None, help="cutoff per runner call: no game starts after it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clock-ticks", type=int, default=None)
    parser.add_argument("--arena-radius", type=float, default=None)
    parser.add_argument("--spawn", choices=("mixed", "sides"), default=None)
    parser.add_argument("--fields", type=int, default=None)
    parser.add_argument("--no-fog", action="store_true")
    arguments = parser.parse_args()

    changes = {name: value for name, value in (
        ("clock_ticks", arguments.clock_ticks), ("arena_radius", arguments.arena_radius),
        ("spawn", arguments.spawn), ("field_count", arguments.fields)) if value is not None}
    if arguments.no_fog:
        changes["fog"] = False
    game_config = dataclasses.replace(game.GameConfig(), **changes)
    store = Store(arguments.store)
    started = time.perf_counter()

    def progress(games, busy, seconds):
        print(f"  {games} games started, {busy} playing, {seconds:.0f} s", file=sys.stderr, flush=True)

    evaluator = Evaluator(store, game_config, device=arguments.device, envs=arguments.envs, seed=arguments.seed,
                          seconds=arguments.seconds, progress=progress)

    if arguments.command == "crossover":
        network = NetworkAgent(arguments.agents[0], arguments.weights)
        games = games_for if arguments.games == 0 else (arguments.games or 64)
        report = crossover(evaluator_round(evaluator, network), arguments.sizes or [4], games)
        for size, entry in report["sizes"].items():
            matches = " ".join(f"{scripted}:{match['wins']}/{match['games']}" for scripted, match in entry["matches"].items())
            print(f"crossover {size}: beats_up_to {entry['beats_up_to']} ratio {entry['ratio']:.2f} "
                  f"(rounds {report['rounds']}; wins by scripted size {matches})")
        return

    sizes = tuple(arguments.sizes or (4, 4))
    if arguments.command == "elo":
        anchor = parse_agent(arguments.anchor, store) if arguments.anchor else None
        for identity, rating, error in evaluator.elo(sizes, anchor):
            print(f"{rating:8.1f} +- {error:5.1f}  {name_of(store.agent(identity))}")
        return

    agents = [parse_agent(text, store) for text in arguments.agents]
    pairs = [(agents[0], agent) for agent in agents[1:]] if arguments.against_first else list(itertools.combinations(agents, 2))
    if arguments.sobol:
        candidates = evaluator.sobol_candidates(arguments.sobol, arguments.count, seed=arguments.seed)
        pairs += [(agents[0] if agents else ScriptedAgent.of(), candidate) for candidate in candidates]
    pairings = [(a, b, sizes) for a, b in pairs]
    if arguments.command == "play":
        if arguments.games is not None:
            evaluator.ensure([(a, b, sizes, arguments.games) for a, b, sizes in pairings])
        else:
            evaluator.adaptive(pairings, target_error=arguments.target_error, budget=arguments.budget)
    for (a, b, _), row in zip(pairings, evaluator.report(pairings), strict=True):
        print(f"{row['score']:.3f} +- {row['error']:.3f} ({row['wins']}-{row['draws']}-{row['losses']} of "
              f"{row['games']})  {name_of(a)}  vs  {name_of(b)}")
    if arguments.command == "play":
        print(f"{time.perf_counter() - started:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
