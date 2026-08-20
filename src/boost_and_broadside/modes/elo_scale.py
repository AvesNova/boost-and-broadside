"""Rate frozen checkpoints across symmetric fleet sizes.

Each fleet size gets an independent stationary tournament containing random,
scripted, every preserved ladder checkpoint, and the final checkpoint. Raw
outcomes are saved into the run-owned ``elo-scale`` artifact after every
adaptive batch, so an interrupted sweep resumes where it stopped.

Reporting anchors are pure post-processing and never require replaying a match.
The published fleet-scale figure — including the join through an independently
measured semi-random reference ladder — is rendered by ``bnb publish`` from this
artifact, not written here.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from boost_and_broadside.artifacts import ArtifactRecipe, ArtifactStore, file_sha256
from boost_and_broadside.config import EloCalibrateConfig, ShipConfig
from boost_and_broadside.evaluation.environment import run_field_map
from boost_and_broadside.evaluation.run_catalog import (
    InvalidCheckpointError,
    resolve_exact_run,
    select_final_training_checkpoint,
    select_tournament_ladder_policies,
)
from boost_and_broadside.evaluation.subjects import (
    describe_agent,
    describe_checkpoint_configuration,
    describe_environment,
)
from boost_and_broadside.evaluation.tournament import (
    BatchStat,
    Player,
    Progress,
    Tournament,
    build_players,
    load_run_config,
    parallel_envs_for,
    rating_views,
    run_tournament,
)
from boost_and_broadside.train.rl.bradley_terry import fit_bradley_terry
from boost_and_broadside.train.rl.checkpoint_schema import load_checkpoint_payload

_SCHEMA_VERSION = 1
_SEED_BASE = 682_000


def _player_metadata(run_dir: Path, roster: dict, final_path: Path) -> list[dict]:
    records = [
        {"label": "random", "kind": "random", "global_step": 0},
        {"label": "scripted", "kind": "scripted", "global_step": None},
    ]
    final_checkpoint = load_checkpoint_payload(final_path, map_location="cpu")
    final_step = int(final_checkpoint.get("global_step", 0))
    selected_final = select_final_training_checkpoint(run_dir)
    if selected_final.step != final_step:
        raise InvalidCheckpointError(
            f"final checkpoint {final_path} records global_step={final_step}; "
            f"filename records {selected_final.step}"
        )
    for policy_ref in select_tournament_ladder_policies(run_dir, roster):
        if policy_ref.global_step == final_step:
            continue
        path = policy_ref.checkpoint.path
        records.append(
            {
                "label": policy_ref.label,
                "kind": "checkpoint",
                "global_step": policy_ref.global_step,
                "path": str(path),
                "sha256": file_sha256(path),
            }
        )
    records.append(
        {
            "label": "final",
            "kind": "checkpoint",
            "global_step": final_step,
            "path": str(final_path),
            "sha256": file_sha256(final_path),
            "training_config": describe_checkpoint_configuration(final_checkpoint),
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
) -> list[Player]:
    return build_players(
        run_dir,
        roster,
        model_config,
        ship_config,
        total_ships,
        device,
        final_label="final",
    )


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


def _scale_recipe(
    run: str, metadata: list[dict], team_sizes: list[int], config: EloCalibrateConfig, base_env
) -> ArtifactRecipe:
    """Identify this sweep by its exact player field and stopping rule."""

    return ArtifactRecipe(
        artifact_type="elo-scale",
        result_schema_version=_SCHEMA_VERSION,
        subjects={
            "run": run,
            "players": [
                {
                    key: record.get(key)
                    for key in ("label", "kind", "global_step", "sha256", "training_config")
                }
                for record in metadata
            ],
            "scripted": describe_agent("scripted"),
        },
        parameters={
            "team_sizes": sorted(set(team_sizes)),
            "target_stderr": config.target_stderr,
            "max_batches": config.max_batches,
            "max_parallel_envs": config.num_envs,
            "tie_mode": config.tie_mode,
            "prior_games": config.prior_games,
            "seed_base": _SEED_BASE,
            "environment": describe_environment(base_env),
        },
    )


def run_elo_scale_mode(
    run_spec: str,
    team_sizes: list[int],
    ship_config: ShipConfig,
    device: str,
    config: EloCalibrateConfig,
    checkpoint_dir: str = "checkpoints",
    store: ArtifactStore | None = None,
) -> dict:
    """Run or resume checkpoint tournaments across symmetric team sizes."""
    run_dir = resolve_exact_run(run_spec, checkpoint_dir).path
    roster = json.loads((run_dir / "roster.json").read_text())
    base_env, model_config, paradigm, field_map_config = load_run_config(run_dir)
    final_path = select_final_training_checkpoint(run_dir).path
    metadata = _player_metadata(run_dir, roster, final_path)
    labels = [record["label"] for record in metadata]

    store = store or ArtifactStore(checkpoint_root=checkpoint_dir)
    artifact, resumed = store.open_resumable(
        _scale_recipe(run_dir.name, metadata, team_sizes, config, base_env),
        store.run_owner(run_dir.name),
    )
    if resumed and artifact.has("result.json"):
        result = artifact.read_json()
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
    artifact.write_json(result)

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
            f"\n=== scale Elo: {run_dir.name}  {team_size}v{team_size}  "
            f"({num_envs} games/batch, {device}) ==="
        )
        players = _build_scale_players(
            run_dir,
            roster,
            model_config,
            ship_config,
            total_ships,
            device,
        )
        if [player.label for player in players] != labels:
            raise ValueError("loaded tournament field does not match stored metadata")
        env_config = replace(base_env, num_ships=total_ships)
        tournament = Tournament(
            players,
            ship_config,
            env_config,
            paradigm,
            num_envs,
            device,
            field_map=run_field_map(ship_config, env_config, field_map_config, device),
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
            artifact.write_json(result)

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

    artifact.complete()
    print(f"\n  wrote {artifact.path}")
    return result
