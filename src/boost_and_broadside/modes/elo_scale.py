"""Rate frozen checkpoints across symmetric fleet sizes.

Each fleet size gets an independent stationary tournament containing random,
scripted, every preserved ladder checkpoint, and the final checkpoint. Raw
outcomes are saved after every adaptive batch; reporting anchors are pure
post-processing and never require replaying a match.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from boost_and_broadside.config import EloCalibrateConfig, ShipConfig
from boost_and_broadside.modes.agent_factory import ResolvedAgent
from boost_and_broadside.modes.elo_calibrate import (
    BatchStat,
    Player,
    Progress,
    Tournament,
    _load_ladder_policy,
    _load_run_config,
    build_players,
    run_tournament,
)
from boost_and_broadside.train.rl.bradley_terry import (
    fit_bradley_terry,
    rating_covariance,
    rating_stderr,
)

_COLLISION_BUDGET = 4_000_000  # B*N², tuned for an 8 GB GPU
_SCHEMA_VERSION = 1
_SEED_BASE = 682_000


def parallel_envs_for(total_ships: int, maximum: int) -> int:
    """Largest parallel batch under the collision-memory budget."""
    if total_ships <= 0 or maximum <= 0:
        raise ValueError("ship and environment counts must be positive")
    return max(1, min(maximum, _COLLISION_BUDGET // (total_ships * total_ships)))


def rating_views(
    ratings: np.ndarray, pair_games: np.ndarray, labels: list[str]
) -> dict[str, dict[str, list[float]]]:
    """Transform one fitted rating vector into the three reporting conventions."""
    random_index = labels.index("random")
    scripted_index = labels.index("scripted")

    random_zero = ratings - ratings[random_index]
    random_error = rating_stderr(pair_games, ratings, anchor=random_index)

    scripted_1000 = ratings - ratings[scripted_index] + 1000.0
    scripted_error = rating_stderr(pair_games, ratings, anchor=scripted_index)

    gap = float(random_zero[scripted_index])
    if abs(gap) < 1e-9:
        dual = np.full_like(ratings, np.nan)
        dual_error = np.full_like(ratings, np.inf)
    else:
        dual = 1000.0 * random_zero / gap
        covariance = rating_covariance(pair_games, ratings, anchor=random_index)
        dual_error = np.zeros_like(ratings)
        for index in range(ratings.size):
            gradient = np.zeros_like(ratings)
            gradient[index] += 1000.0 / gap
            gradient[scripted_index] -= 1000.0 * random_zero[index] / gap**2
            variance = float(gradient @ covariance @ gradient)
            dual_error[index] = np.sqrt(max(variance, 0.0))

    return {
        "random_zero": {
            "ratings": random_zero.tolist(),
            "stderr": random_error.tolist(),
        },
        "scripted_1000": {
            "ratings": scripted_1000.tolist(),
            "stderr": scripted_error.tolist(),
        },
        "random_zero_scripted_1000": {
            "ratings": dual.tolist(),
            "stderr": dual_error.tolist(),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path(run_dir: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    return run_dir / path.name


def _player_metadata(run_dir: Path, roster: dict, final_path: Path) -> list[dict]:
    records = [
        {"label": "random", "kind": "random", "global_step": 0},
        {"label": "scripted", "kind": "scripted", "global_step": None},
    ]
    for entry in sorted(
        (item for item in roster["entries"] if item["kind"] == "checkpoint"),
        key=lambda item: item["global_step"],
    ):
        path = _checkpoint_path(run_dir, entry["path"])
        if not path.exists():
            continue
        records.append(
            {
                "label": entry["label"],
                "kind": "checkpoint",
                "global_step": entry["global_step"],
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    final_checkpoint = torch.load(str(final_path), map_location="cpu", weights_only=False)
    records.append(
        {
            "label": "final",
            "kind": "checkpoint",
            "global_step": int(final_checkpoint.get("global_step", 0)),
            "path": str(final_path),
            "sha256": _sha256(final_path),
        }
    )
    return records


def _build_scale_players(
    run_dir: Path,
    roster: dict,
    model_config,
    ship_config: ShipConfig,
    total_ships: int,
    device: str,
    final_path: Path,
) -> list[Player]:
    players = build_players(
        run_dir, roster, model_config, ship_config, total_ships, device
    )
    checkpoint = torch.load(str(final_path), map_location="cpu", weights_only=False)
    final_policy = _load_ladder_policy(
        final_path, model_config, ship_config, total_ships, device
    )
    players.append(
        Player(
            "final",
            ResolvedAgent("policy", final_policy),
            None,
            int(checkpoint.get("global_step", 0)),
        )
    )
    return players


def _restore_tournament(tournament: Tournament, stored: dict | None) -> list[BatchStat]:
    if stored is None:
        return []
    shape = (tournament.size, tournament.size)
    wins = np.asarray(stored["wins_matrix"], dtype=np.float64)
    ties = np.asarray(stored["ties_matrix"], dtype=np.float64)
    if wins.shape != shape or ties.shape != shape:
        raise ValueError("stored scale tournament does not match the current player field")
    tournament.wins[:] = wins
    tournament.ties[:] = ties
    directed = stored.get("directed_outcomes")
    if directed is not None:
        outcomes = np.asarray(directed, dtype=np.float64)
        if outcomes.shape != (*shape, 3):
            raise ValueError("stored directed outcomes have the wrong shape")
        tournament.directed_outcomes[:] = outcomes
    return [BatchStat(**row) for row in stored.get("batches", [])]


def _scale_result(
    team_size: int,
    tournament: Tournament,
    stats: list[BatchStat],
    reference: int,
    config: EloCalibrateConfig,
) -> dict:
    labels = [player.label for player in tournament.players]
    primary_wins = tournament.scored_wins(config.tie_mode)
    primary = fit_bradley_terry(
        primary_wins, anchor=reference, prior_games=config.prior_games
    )
    primary_views = rating_views(
        primary.ratings, tournament.pair_games(config.tie_mode), labels
    )

    alternate_mode = "decisive" if config.tie_mode == "half_win" else "half_win"
    alternate = fit_bradley_terry(
        tournament.scored_wins(alternate_mode),
        anchor=reference,
        prior_games=config.prior_games,
    )
    alternate_views = rating_views(
        alternate.ratings, tournament.pair_games(alternate_mode), labels
    )
    return {
        "team_size": team_size,
        "total_ships": 2 * team_size,
        "parallel_envs": tournament.num_envs,
        "seed_base": _SEED_BASE + team_size * 100,
        "reference": labels[reference],
        "tie_mode": config.tie_mode,
        "tie_mode_alt": alternate_mode,
        "converged": bool(stats and stats[-1].max_stderr <= config.target_stderr),
        "ratings": primary_views,
        "ratings_alt": alternate_views,
        "games_by_player": primary.games.tolist(),
        "wins_matrix": tournament.wins.tolist(),
        "ties_matrix": tournament.ties.tolist(),
        "directed_outcomes": tournament.directed_outcomes.tolist(),
        "batches": [vars(stat) for stat in stats],
    }


def _write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(path)


def run_elo_scale_mode(
    run_spec: str,
    team_sizes: list[int],
    ship_config: ShipConfig,
    device: str,
    config: EloCalibrateConfig,
    checkpoint_dir: str = "checkpoints",
    plot_dir: str = "docs/results",
    plot: bool = True,
) -> dict:
    """Run or resume checkpoint tournaments across symmetric team sizes."""
    from boost_and_broadside.modes.elo_scale_plots import write_scale_plots
    from boost_and_broadside.modes.elo_stats import find_run_dir

    run_dir = find_run_dir(run_spec, checkpoint_dir)
    roster = json.loads((run_dir / "roster.json").read_text())
    base_env, model_config, paradigm = _load_run_config(run_dir)
    final_candidates = sorted(run_dir.glob("step_*.pt"))
    if not final_candidates:
        raise FileNotFoundError(f"no final step checkpoint in {run_dir}")
    final_path = final_candidates[-1]
    metadata = _player_metadata(run_dir, roster, final_path)
    labels = [record["label"] for record in metadata]

    output = run_dir / "elo_scale.json"
    if output.exists():
        result = json.loads(output.read_text())
        if result.get("player_labels") != labels:
            raise ValueError("stored scale result uses a different checkpoint field")
    else:
        result = {
            "schema_version": _SCHEMA_VERSION,
            "run": run_dir.name,
            "player_labels": labels,
            "players": metadata,
            "team_sizes": sorted(set(team_sizes)),
            "target_stderr": config.target_stderr,
            "max_batches": config.max_batches,
            "max_parallel_envs": config.num_envs,
            "seed_base": _SEED_BASE,
            "scales": {},
        }

    result["team_sizes"] = sorted(set(result.get("team_sizes", []) + team_sizes))
    result["target_stderr"] = config.target_stderr
    result["max_batches"] = config.max_batches
    result["max_parallel_envs"] = config.num_envs
    _write_result(output, result)

    for team_size in sorted(set(team_sizes)):
        if team_size <= 0:
            raise ValueError("team sizes must be positive")
        stored = result["scales"].get(str(team_size))
        stored_batches = stored.get("batches", []) if stored else []
        if stored and (stored.get("converged") or len(stored_batches) >= config.max_batches):
            print(f"  {team_size}v{team_size}: already complete, skipping")
            continue

        total_ships = 2 * team_size
        num_envs = parallel_envs_for(total_ships, config.num_envs)
        print(
            f"\n=== scale ELO: {run_dir.name}  {team_size}v{team_size}  "
            f"({num_envs} games/batch, {device}) ==="
        )
        players = _build_scale_players(
            run_dir,
            roster,
            model_config,
            ship_config,
            total_ships,
            device,
            final_path,
        )
        if [player.label for player in players] != labels:
            raise ValueError("loaded tournament field does not match stored metadata")
        env_config = replace(base_env, num_ships=total_ships)
        tournament = Tournament(
            players, ship_config, env_config, paradigm, num_envs, device
        )
        initial_stats = _restore_tournament(tournament, stored)
        random_index = labels.index("random")
        progress = Progress()

        def save_batch(
            current: Tournament,
            _fit,
            stats: list[BatchStat],
            reference: int,
        ) -> None:
            result["scales"][str(team_size)] = _scale_result(
                team_size, current, stats, reference, config
            )
            _write_result(output, result)
            if plot:
                write_scale_plots(result, Path(plot_dir))

        fit, stats, reference = run_tournament(
            tournament,
            random_index,
            config,
            progress,
            initial_stats=initial_stats,
            on_batch=save_batch,
            seed_base=_SEED_BASE + team_size * 100,
        )
        del fit
        save_batch(tournament, None, stats, reference)
        del tournament, players
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if plot:
        written = write_scale_plots(result, Path(plot_dir))
        print(f"\n  wrote {len(written)} scale charts to {plot_dir}")
    print(f"  wrote {output}")
    return result
