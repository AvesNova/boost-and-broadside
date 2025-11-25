import multiprocessing as mp
import pickle
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.data_collector import DataCollector
from src.game_coordinator import GameCoordinator


def collect_worker(
    worker_id: int, cfg_dict: dict[str, Any], run_timestamp: str
) -> None:
    """
    Worker function for collecting data in a separate process.

    This function initializes a DataCollector and GameCoordinator, runs the
    specified number of episodes for each game mode, and saves the collected
    data to disk.

    Args:
        worker_id: Unique identifier for this worker (0 to num_workers-1).
        cfg_dict: Configuration dictionary (serialized for multiprocessing).
        run_timestamp: Timestamp string for this run, used for directory naming.
    """
    cfg = OmegaConf.create(cfg_dict)

    print(f"Worker {worker_id}: Starting data collection...")

    collector = DataCollector(cfg, worker_id=worker_id, run_timestamp=run_timestamp)
    coordinator = GameCoordinator(cfg, render_mode=cfg.collect.render_mode)

    episodes_per_mode = cfg.collect.episodes_per_mode
    total_episodes = sum(episodes_per_mode.values())

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Worker {worker_id}: Collecting episodes...",
                total=total_episodes,
            )

            for game_mode, num_episodes in episodes_per_mode.items():
                for episode_idx in range(num_episodes):
                    coordinator.reset(game_mode=game_mode)
                    episode_sim_time, terminated = coordinator.step()

                    collector.add_episode(
                        tokens_team_0=coordinator.all_tokens[0],
                        tokens_team_1=coordinator.all_tokens[1],
                        actions=coordinator.all_actions,
                        rewards=coordinator.all_rewards,
                        sim_time=episode_sim_time,
                    )

                    progress.update(task, advance=1)

                    progress.console.print(
                        f"[green]Worker {worker_id}: Completed {game_mode} episode "
                        f"{episode_idx + 1}/{num_episodes} (sim_time: {episode_sim_time:.2f}s)"
                    )

    except KeyboardInterrupt:
        print(f"\nWorker {worker_id}: Data collection interrupted by user")
    except Exception as e:
        print(f"Worker {worker_id}: Error during collection: {e}")
        raise
    finally:
        collector.finalize()
        coordinator.close()
        print(f"Worker {worker_id}: Data collection complete")


def aggregate_worker_data(cfg: DictConfig, run_timestamp: str) -> Path | None:
    """
    Aggregate data from all workers into a single dataset file.

    Reads the individual data files produced by each worker, concatenates
    the tensors, and saves a unified `aggregated_data.pkl` and `metadata.yaml`.

    Args:
        cfg: Configuration dictionary.
        run_timestamp: Timestamp string for this run.

    Returns:
        Path to the aggregated data file, or None if no data was found.
    """
    run_dir = Path(cfg.collect.output_dir) / run_timestamp
    worker_dirs = sorted(run_dir.glob("worker_*"))

    if not worker_dirs:
        print("No worker data found to aggregate")
        return None

    print(f"\nAggregating data from {len(worker_dirs)} workers...")

    all_team_0_tokens = []
    all_team_0_actions = []
    all_team_0_rewards = []
    all_team_1_tokens = []
    all_team_1_actions = []
    all_team_1_rewards = []
    all_episode_ids = []
    all_episode_lengths = []

    total_episodes = 0
    total_timesteps = 0
    total_sim_time = 0.0
    worker_metadata = []

    for worker_dir in worker_dirs:
        final_data_path = worker_dir / "data_final.pkl"
        if not final_data_path.exists():
            print(f"Warning: {final_data_path} not found, skipping")
            continue

        with open(final_data_path, "rb") as f:
            data = pickle.load(f)

        episode_offset = total_episodes
        adjusted_episode_ids = data["episode_ids"] + episode_offset
        
        all_team_0_tokens.append(data["team_0"]["tokens"])
        all_team_0_actions.append(data["team_0"]["actions"])
        all_team_0_rewards.append(data["team_0"]["rewards"])
        all_team_1_tokens.append(data["team_1"]["tokens"])
        all_team_1_actions.append(data["team_1"]["actions"])
        all_team_1_rewards.append(data["team_1"]["rewards"])
        all_episode_ids.append(adjusted_episode_ids)
        all_episode_lengths.append(data["episode_lengths"])

        total_episodes += data["metadata"]["num_episodes"]
        total_timesteps += data["metadata"]["total_timesteps"]
        total_sim_time += data["metadata"]["total_sim_time"]
        worker_metadata.append(data["metadata"])

    if total_episodes == 0:
        print("No episodes found in worker data.")
        return None

    aggregated_data = {
        "team_0": {
            "tokens": torch.cat(all_team_0_tokens, dim=0),
            "actions": torch.cat(all_team_0_actions, dim=0),
            "rewards": torch.cat(all_team_0_rewards, dim=0),
        },
        "team_1": {
            "tokens": torch.cat(all_team_1_tokens, dim=0),
            "actions": torch.cat(all_team_1_actions, dim=0),
            "rewards": torch.cat(all_team_1_rewards, dim=0),
        },
        "episode_ids": torch.cat(all_episode_ids, dim=0),
        "episode_lengths": torch.cat(all_episode_lengths, dim=0),
        "metadata": {
            "num_episodes": total_episodes,
            "total_timesteps": total_timesteps,
            "total_sim_time": total_sim_time,
            "num_workers": len(worker_dirs),
            "run_timestamp": run_timestamp,
            "max_ships": worker_metadata[0]["max_ships"],
            "token_dim": worker_metadata[0]["token_dim"],
            "num_actions": worker_metadata[0]["num_actions"],
            "worker_metadata": worker_metadata,
        },
    }

    aggregated_pkl_path = run_dir / "aggregated_data.pkl"
    with open(aggregated_pkl_path, "wb") as f:
        pickle.dump(aggregated_data, f)

    print(f"Saved aggregated data to {aggregated_pkl_path}")

    metadata_yaml = {
        "num_episodes": int(total_episodes),
        "total_timesteps": int(total_timesteps),
        "total_sim_time": float(total_sim_time),
        "num_workers": len(worker_dirs),
        "run_timestamp": run_timestamp,
        "max_ships": int(worker_metadata[0]["max_ships"]),
        "token_dim": int(worker_metadata[0]["token_dim"]),
        "num_actions": int(worker_metadata[0]["num_actions"]),
        "data_shapes": {
            "team_0_tokens": list(aggregated_data["team_0"]["tokens"].shape),
            "team_0_actions": list(aggregated_data["team_0"]["actions"].shape),
            "team_0_rewards": list(aggregated_data["team_0"]["rewards"].shape),
            "team_1_tokens": list(aggregated_data["team_1"]["tokens"].shape),
            "team_1_actions": list(aggregated_data["team_1"]["actions"].shape),
            "team_1_rewards": list(aggregated_data["team_1"]["rewards"].shape),
            "episode_ids": list(aggregated_data["episode_ids"].shape),
            "episode_lengths": list(aggregated_data["episode_lengths"].shape),
        },
    }

    metadata_yaml_path = run_dir / "metadata.yaml"
    with open(metadata_yaml_path, "w") as f:
        yaml.dump(metadata_yaml, f, default_flow_style=False)

    print(f"Saved metadata to {metadata_yaml_path}")
    print(
        f"\nAggregation complete: {total_episodes} episodes, {total_timesteps} timesteps"
    )
    return aggregated_pkl_path


def collect(cfg: DictConfig) -> Path | None:
    """
    Execute the data collection pipeline.

    Orchestrates the data collection process by spawning worker processes
    and then aggregating the results.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Path to the final aggregated data file, or None if failed.
    """
    num_workers = cfg.collect.num_workers
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if num_workers == 1:
        print(f"Starting single-worker data collection (run: {run_timestamp})...")
        collect_worker(0, OmegaConf.to_container(cfg, resolve=True), run_timestamp)
    else:
        print(
            f"Starting parallel data collection with {num_workers} workers "
            f"(run: {run_timestamp})..."
        )

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        with mp.Pool(processes=num_workers) as pool:
            try:
                pool.starmap(
                    collect_worker,
                    [
                        (worker_id, cfg_dict, run_timestamp)
                        for worker_id in range(num_workers)
                    ],
                )
            except KeyboardInterrupt:
                print("\nTerminating all workers...")
                pool.terminate()
                pool.join()
                raise

        print(f"\nAll {num_workers} workers completed!")

    return aggregate_worker_data(cfg, run_timestamp)
