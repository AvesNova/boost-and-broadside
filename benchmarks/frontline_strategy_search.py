"""Successive batched tournaments for Frontline scripted strategy parameters.

Every round places the incumbent and all proposed configurations in one
``Tournament``. With the default 127 proposals and two games per pairing, all
16,256 side-balanced games run in a single TensorEnv batch. The highest-rated
configuration becomes the next round's center when its fitted improvement is
large enough. A final 256-game batch compares the winner with the original.

This is a rough search tool rather than a claim of statistical optimality. It
stores seeds, configurations, outcome matrices, and ratings so a promising
result can be inspected and reproduced.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config.defaults import SHIP_CONFIG
from boost_and_broadside.env.frontline import frontline_ship_config
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.evaluation.tournament import (
    Player,
    Tournament,
    effective_wins,
)
from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG
from boost_and_broadside.train.rl.bradley_terry import fit_bradley_terry

PARAMETERS = (
    "frontline_aggression",
    "frontline_combat_radius",
    "frontline_zone_radius",
    "frontline_zone_margin",
    "frontline_separation_radius",
    "frontline_recovery_health",
)
BOUNDS = np.asarray(
    [
        (0.35, 1.65),
        (250.0, 850.0),
        (650.0, 1200.0),
        (0.6, 1.4),
        (70.0, 200.0),
        (0.3, 0.75),
    ]
)


def _vector(config: dict[str, float]) -> np.ndarray:
    return np.asarray([config[name] for name in PARAMETERS], dtype=np.float64)


def _config(vector: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAMETERS, vector, strict=True)}


def _sample_round(
    rng: np.random.Generator,
    incumbent: dict[str, float],
    count: int,
    radius: float,
) -> list[dict[str, float]]:
    center = _vector(incumbent)
    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    candidates = []
    # Antithetic pairs cover both directions around the center in a small field.
    while len(candidates) < count:
        direction = rng.normal(size=len(PARAMETERS))
        direction /= np.linalg.norm(direction).clip(min=1e-9)
        for sign in (1.0, -1.0):
            proposal = np.clip(
                center + sign * radius * span * direction,
                BOUNDS[:, 0],
                BOUNDS[:, 1],
            )
            candidates.append(_config(proposal))
            if len(candidates) == count:
                break
    return candidates


def _players(configs: list[dict[str, float]], ship_config) -> list[Player]:
    defaults = asdict(StochasticAgentConfig())
    players = []
    for index, values in enumerate(configs):
        configured = dict(defaults)
        configured.update(values)
        agent = StochasticScriptedAgent(ship_config, StochasticAgentConfig(**configured))
        players.append(Player(f"p{index:02d}", ResolvedAgent("scripted", agent), None, None))
    return players


def _allocation(size: int, games_per_pair: int) -> np.ndarray:
    allocation = np.zeros((size, size), dtype=np.int64)
    allocation[np.triu_indices(size, 1)] = games_per_pair
    return allocation


def _play_field(
    configs: list[dict[str, float]],
    *,
    games_per_pair: int,
    seconds: float,
    seed: int,
    device: str,
) -> dict:
    ship_config = frontline_ship_config(SHIP_CONFIG)
    max_steps = round(seconds / (ship_config.dt * PLAY_ENV_CONFIG.action_repeat))
    env_config = replace(PLAY_ENV_CONFIG, max_episode_steps=max_steps)
    allocation = _allocation(len(configs), games_per_pair)
    games = int(allocation.sum())

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    tournament = Tournament(
        _players(configs, ship_config),
        ship_config,
        env_config,
        "shared_pass",
        games,
        device,
    )
    started = time.perf_counter()
    tournament.play_batch(allocation)
    elapsed = time.perf_counter() - started
    fit = fit_bradley_terry(
        effective_wins(tournament.wins, tournament.ties, "half_win"),
        anchor=0,
        prior_games=1.0,
    )
    ratings = fit.ratings.tolist()
    ranking = sorted(range(len(configs)), key=lambda index: ratings[index], reverse=True)
    return {
        "games": games,
        "seconds_per_episode": seconds,
        "wall_seconds": elapsed,
        "games_per_second": games / elapsed,
        "ratings": ratings,
        "ranking": ranking,
        "wins_matrix": tournament.wins.tolist(),
        "ties_matrix": tournament.ties.tolist(),
        "directed_outcomes": tournament.directed_outcomes.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ideas", type=int, default=127)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--games-per-pair", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=150.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--min-gain-elo", type=float, default=15.0)
    parser.add_argument("--initial-json", help="JSON object overriding the initial incumbent")
    parser.add_argument("--radius-start", type=float, default=0.55)
    parser.add_argument("--radius-end", type=float, default=0.08)
    parser.add_argument("--final-games", type=int, default=4096)
    parser.add_argument("--final-seconds", type=float, default=300.0)
    args = parser.parse_args()
    if args.ideas < 2 or args.games_per_pair < 2 or args.games_per_pair % 2:
        raise ValueError("ideas >= 2 and games-per-pair must be a positive even number")
    if args.final_games < 2 or args.final_games % 2:
        raise ValueError("final-games must be a positive even number")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for the scripted search but is unavailable. "
            "Restore GPU visibility, or pass --device cpu only for a functional smoke test."
        )

    defaults = asdict(StochasticAgentConfig())
    incumbent = {name: float(defaults[name]) for name in PARAMETERS}
    if args.initial_json:
        supplied = json.loads(args.initial_json)
        incumbent.update({name: float(supplied[name]) for name in PARAMETERS})
    initial = dict(incumbent)
    rng = np.random.default_rng(args.seed)
    record: dict = {
        "method": "successive all-pairs tournaments in one TensorEnv batch per round",
        "seed": args.seed,
        "parameters": list(PARAMETERS),
        "bounds": {name: list(bound) for name, bound in zip(PARAMETERS, BOUNDS.tolist())},
        "initial": initial,
        "rounds": [],
    }

    radii = np.geomspace(args.radius_start, args.radius_end, args.rounds)
    for round_index, radius in enumerate(radii):
        candidates = _sample_round(rng, incumbent, args.ideas, float(radius))
        configs = [incumbent, *candidates]
        result = _play_field(
            configs,
            games_per_pair=args.games_per_pair,
            seconds=args.seconds,
            seed=args.seed + round_index,
            device=args.device,
        )
        best_index = result["ranking"][0]
        gain = float(result["ratings"][best_index])
        accepted = best_index != 0 and gain >= args.min_gain_elo
        record["rounds"].append(
            {
                "index": round_index,
                "radius": float(radius),
                "incumbent": incumbent,
                "configs": configs,
                "best_index": best_index,
                "gain_elo_vs_incumbent": gain,
                "accepted": accepted,
                **result,
            }
        )
        print(
            f"round {round_index + 1}: {result['games']} games in "
            f"{result['wall_seconds']:.1f}s; best=p{best_index:02d}, "
            f"gain={gain:+.1f} Elo, accepted={accepted}",
            flush=True,
        )
        if accepted:
            incumbent = dict(configs[best_index])
        elif round_index >= 1:
            break
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    verification = _play_field(
        [initial, incumbent],
        games_per_pair=args.final_games,
        seconds=args.final_seconds,
        seed=args.seed + 99_000,
        device=args.device,
    )
    record["winner"] = incumbent
    record["verification_vs_initial"] = verification
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"winner": incumbent, "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
