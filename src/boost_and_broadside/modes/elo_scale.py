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
from boost_and_broadside.evaluation.run_catalog import (
    InvalidCheckpointError,
    resolve_exact_run,
    select_final_training_checkpoint,
    select_tournament_ladder_policies,
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


def combine_reference_ladder(result: dict, reference_result: dict) -> dict:
    """Refit scale ratings after joining an independently measured reference ladder.

    The checkpoint and reference tournaments share the same random and scripted
    controllers. Joining their outcome matrices at those players adds intermediate
    comparisons without replaying checkpoint matches. The returned object is a derived
    reporting view; both input artifacts remain the sources of raw outcomes.
    """
    if result.get("run") != reference_result.get("run"):
        raise ValueError("checkpoint and reference tournaments belong to different runs")

    checkpoint_labels = list(result["player_labels"])
    reference_labels = list(reference_result["labels"])
    for endpoint in ("random", "scripted"):
        if endpoint not in checkpoint_labels or endpoint not in reference_labels:
            raise ValueError(f"both tournaments must contain {endpoint!r}")

    labels = checkpoint_labels + [
        label for label in reference_labels if label not in checkpoint_labels
    ]
    label_indices = {label: index for index, label in enumerate(labels)}

    def add_matrix(target: np.ndarray, values: list[list[float]], source_labels: list[str]) -> None:
        matrix = np.asarray(values, dtype=np.float64)
        expected = (len(source_labels), len(source_labels))
        if matrix.shape != expected:
            raise ValueError("stored tournament matrix does not match its player labels")
        indices = [label_indices[label] for label in source_labels]
        target[np.ix_(indices, indices)] += matrix

    scales = {}
    for key, checkpoint_scale in result.get("scales", {}).items():
        reference_scale = reference_result.get("scales", {}).get(key)
        if reference_scale is None:
            continue
        if checkpoint_scale["team_size"] != reference_scale["team_size"]:
            raise ValueError(f"team-size mismatch for scale {key}")
        if checkpoint_scale.get("tie_mode", "half_win") != "half_win":
            raise ValueError("reference-ladder reporting requires half-win tie scoring")

        shape = (len(labels), len(labels))
        wins = np.zeros(shape, dtype=np.float64)
        ties = np.zeros(shape, dtype=np.float64)
        add_matrix(wins, checkpoint_scale["wins_matrix"], checkpoint_labels)
        add_matrix(ties, checkpoint_scale["ties_matrix"], checkpoint_labels)
        add_matrix(wins, reference_scale["wins_matrix"], reference_labels)
        add_matrix(ties, reference_scale["ties_matrix"], reference_labels)

        scored_wins = wins + 0.5 * ties
        pair_games = wins + wins.T + ties + ties.T
        fit = fit_bradley_terry(
            scored_wins,
            anchor=labels.index("scripted"),
            prior_games=1.0,
        )
        scale = dict(checkpoint_scale)
        scale["ratings"] = rating_views(fit.ratings, pair_games, labels)
        scale["reference_ladder_games"] = int(
            np.asarray(reference_scale["wins_matrix"], dtype=float).sum()
            + np.asarray(reference_scale["ties_matrix"], dtype=float).sum()
        )
        scales[key] = scale

    return {
        "run": result["run"],
        "player_labels": labels,
        "team_sizes": sorted(int(key) for key in scales),
        "reference_ladder": {
            "probabilities": reference_result["probabilities"],
            "games_per_pair": reference_result["games_per_pair"],
        },
        "scales": scales,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                "sha256": _sha256(path),
            }
        )
    records.append(
        {
            "label": "final",
            "kind": "checkpoint",
            "global_step": final_step,
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

    run_dir = resolve_exact_run(run_spec, checkpoint_dir).path
    roster = json.loads((run_dir / "roster.json").read_text())
    base_env, model_config, paradigm = load_run_config(run_dir)
    final_path = select_final_training_checkpoint(run_dir).path
    metadata = _player_metadata(run_dir, roster, final_path)
    labels = [record["label"] for record in metadata]

    output = run_dir / "elo_scale.json"
    reference_output = run_dir / "semi_random_tournament.json"

    def reporting_result(raw_result: dict) -> dict:
        if not reference_output.exists():
            return raw_result
        reference_result = json.loads(reference_output.read_text())
        return combine_reference_ladder(raw_result, reference_result)

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
                write_scale_plots(reporting_result(result), Path(plot_dir))

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
        written = write_scale_plots(reporting_result(result), Path(plot_dir))
        print(f"\n  wrote {len(written)} scale charts to {plot_dir}")
    print(f"  wrote {output}")
    return result
