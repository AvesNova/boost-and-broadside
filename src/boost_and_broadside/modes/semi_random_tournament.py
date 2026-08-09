"""Evaluate a controlled ladder from random to scripted play at each fleet size."""

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from boost_and_broadside.agents.semi_random_scripted import SemiRandomScriptedAgent
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import ShipConfig, TrainConfig
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.evaluation.environment import create_evaluation_field_map
from boost_and_broadside.evaluation.run_catalog import resolve_legacy_elo_run
from boost_and_broadside.evaluation.tournament import (
    Player,
    Progress,
    Tournament,
    load_run_config,
    parallel_envs_for,
    rating_views,
    semi_random_label,
)
from boost_and_broadside.train.rl.bradley_terry import fit_bradley_terry

_SCHEMA_VERSION = 1
_SEED_BASE = 683_000


def parse_probabilities(text: str) -> list[float]:
    """Parse and normalize a random-to-scripted probability ladder."""
    try:
        probabilities = sorted({float(value.strip()) for value in text.split(",")})
    except ValueError as error:
        raise ValueError("scripted probabilities must be comma-separated numbers") from error
    if any(not math.isfinite(value) for value in probabilities):
        raise ValueError("scripted probabilities must be finite")
    if not probabilities or probabilities[0] != 0.0 or probabilities[-1] != 1.0:
        raise ValueError("scripted probabilities must include both 0 and 1")
    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("scripted probabilities must lie in [0, 1]")
    return probabilities


# Player naming is shared with the calibration tournament so a rung carries the
# same label whichever mode rated it.
_label = semi_random_label


def _greatest_divisor_at_most(value: int, maximum: int) -> int:
    """Choose a batch chunk that divides the per-pair game target exactly."""
    if value <= 0 or maximum <= 0:
        raise ValueError("game target and batch capacity must be positive")
    return max(
        candidate for candidate in range(1, min(value, maximum) + 1) if value % candidate == 0
    )


def _players(ship_config: ShipConfig, probabilities: list[float]) -> list[Player]:
    shared_scripted = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
    return [
        Player(
            _label(probability),
            ResolvedAgent(
                "semi_random",
                SemiRandomScriptedAgent(ship_config, probability, scripted_agent=shared_scripted),
            ),
            None,
            None,
        )
        for probability in probabilities
    ]


def _scale_result(
    team_size: int,
    probabilities: list[float],
    tournament: Tournament,
    batches: list[dict],
    games_per_pair: int,
) -> dict:
    labels = [player.label for player in tournament.players]
    scripted_index = labels.index("scripted")
    scored = tournament.scored_wins("half_win")
    fit = fit_bradley_terry(scored, anchor=scripted_index, prior_games=1.0)
    pair_games = tournament.pair_games("half_win")

    adjacent = []
    for lower in range(len(labels) - 1):
        higher = lower + 1
        decisive = tournament.wins[lower, higher] + tournament.wins[higher, lower]
        ties = tournament.ties[lower, higher] + tournament.ties[higher, lower]
        total = decisive + ties
        higher_score = tournament.wins[higher, lower] + 0.5 * ties
        adjacent.append(
            {
                "lower": labels[lower],
                "higher": labels[higher],
                "games": int(total),
                "higher_expected_score": float(higher_score / total) if total else None,
            }
        )

    return {
        "team_size": team_size,
        "total_ships": 2 * team_size,
        "probabilities": probabilities,
        "labels": labels,
        "games_per_pair": games_per_pair,
        "parallel_envs": tournament.num_envs,
        "seed_base": _SEED_BASE + team_size * 100,
        "complete": bool(batches and batches[-1]["cumulative_games_per_pair"] >= games_per_pair),
        "ratings": rating_views(fit.ratings, pair_games, labels),
        "adjacent_matchups": adjacent,
        "wins_matrix": tournament.wins.tolist(),
        "ties_matrix": tournament.ties.tolist(),
        "directed_outcomes": tournament.directed_outcomes.tolist(),
        "batches": batches,
    }


def _write(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(path)


def run_semi_random_tournament(
    run_spec: str,
    team_sizes: list[int],
    probabilities: list[float],
    games_per_pair: int,
    max_parallel_envs: int,
    ship_config: ShipConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    plot_dir: str = "docs/results",
    plot: bool = True,
    train_config: TrainConfig | None = None,
) -> dict:
    """Run or resume a side-balanced round robin among semi-random agents.

    Args:
        run_spec: Finished run to read the environment config from, and the
            directory results are written to.
        train_config: Rate the ladder under a training *profile* instead of a
            finished run. The rungs feed that profile's roster as fixed
            references, and their ratings are a property of the environment they
            play in — tick rate, field count, ship config — so they have to be
            fitted under the config the run will actually use, which may not
            exist as a run yet. Results land in ``<checkpoint_dir>/<run_spec>/``.
    """
    from boost_and_broadside.modes.semi_random_tournament_plots import write_plots

    if games_per_pair <= 0:
        raise ValueError("games_per_pair must be positive")
    field_map = None
    if train_config is not None:
        run_dir = Path(checkpoint_dir) / run_spec
        run_dir.mkdir(parents=True, exist_ok=True)
        base_env = train_config.scales[0].env_config
        paradigm = train_config.paradigm
        if base_env.num_fields > 0:
            if train_config.field_map is None:
                raise ValueError(f"profile {run_spec!r} has fields but no field_map config")
            print(f"  generating field map cache ({train_config.field_map.cache_size} maps)...")
            field_map = create_evaluation_field_map(
                ship_config, base_env, train_config.field_map, torch.device(device)
            )
    else:
        run_dir = resolve_legacy_elo_run(run_spec, checkpoint_dir).path
        base_env, _, paradigm = load_run_config(run_dir)
    labels = [_label(probability) for probability in probabilities]
    output = run_dir / "semi_random_tournament.json"

    if output.exists():
        result = json.loads(output.read_text())
        if result.get("probabilities") != probabilities:
            raise ValueError("stored tournament uses a different probability ladder")
        if result.get("games_per_pair") != games_per_pair:
            raise ValueError("stored tournament uses a different per-pair game target")
    else:
        result = {
            "schema_version": _SCHEMA_VERSION,
            "run": run_dir.name,
            "probabilities": probabilities,
            "labels": labels,
            "games_per_pair": games_per_pair,
            "team_sizes": sorted(set(team_sizes)),
            "seed_base": _SEED_BASE,
            "scales": {},
        }
    result["team_sizes"] = sorted(set(result.get("team_sizes", []) + team_sizes))
    _write(output, result)

    pair_count = len(probabilities) * (len(probabilities) - 1) // 2
    for team_size in sorted(set(team_sizes)):
        if team_size <= 0:
            raise ValueError("team sizes must be positive")
        stored = result["scales"].get(str(team_size))
        if stored and stored.get("complete"):
            print(f"  {team_size}v{team_size}: already complete, skipping")
            continue

        capacity = parallel_envs_for(2 * team_size, max_parallel_envs)
        if capacity < pair_count:
            raise ValueError(
                f"parallel capacity {capacity} cannot cover all {pair_count} agent pairs; "
                "reduce the ladder or increase --calib-envs"
            )
        games_per_pair_per_batch = _greatest_divisor_at_most(games_per_pair, capacity // pair_count)
        num_envs = pair_count * games_per_pair_per_batch
        total_batches = games_per_pair // games_per_pair_per_batch
        print(
            f"\n=== semi-random ladder: {team_size}v{team_size}  "
            f"{games_per_pair} games/pair in {total_batches} batch(es) "
            f"({num_envs} envs, {device}) ==="
        )

        players = _players(ship_config, probabilities)
        env_config = replace(base_env, num_ships=2 * team_size)
        tournament = Tournament(
            players, ship_config, env_config, paradigm, num_envs, device, field_map=field_map
        )
        batches = list(stored.get("batches", [])) if stored else []
        if stored:
            tournament.wins[:] = np.asarray(stored["wins_matrix"], dtype=float)
            tournament.ties[:] = np.asarray(stored["ties_matrix"], dtype=float)
            tournament.directed_outcomes[:] = np.asarray(stored["directed_outcomes"], dtype=float)

        allocation = np.zeros((len(players), len(players)), dtype=np.int64)
        allocation[np.triu_indices(len(players), k=1)] = games_per_pair_per_batch
        allocation += allocation.T
        progress = Progress()
        for batch_index in range(len(batches), total_batches):
            torch.manual_seed(_SEED_BASE + team_size * 100 + batch_index + 1)
            started = time.perf_counter()
            games = tournament.play_batch(allocation, progress)
            elapsed = time.perf_counter() - started
            batch = {
                "batch": batch_index + 1,
                "games": games,
                "cumulative_games": int((batch_index + 1) * num_envs),
                "games_per_pair": games_per_pair_per_batch,
                "cumulative_games_per_pair": int((batch_index + 1) * games_per_pair_per_batch),
                "seconds": elapsed,
            }
            batches.append(batch)
            result["scales"][str(team_size)] = _scale_result(
                team_size, probabilities, tournament, batches, games_per_pair
            )
            _write(output, result)
            print(
                f"  batch {batch_index + 1:2d}/{total_batches}  "
                f"games={games:5d}  pair total="
                f"{batch['cumulative_games_per_pair']:3d}/{games_per_pair}  "
                f"({elapsed:.0f}s)",
                flush=True,
            )

        del tournament, players
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if plot:
            write_plots(result, Path(plot_dir))

    if plot:
        paths = write_plots(result, Path(plot_dir))
        print(f"\n  wrote {len(paths)} semi-random ladder charts to {plot_dir}")
    print(f"  wrote {output}")
    return result
