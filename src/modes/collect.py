import multiprocessing as mp
import pickle
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from data_collector import DataCollector
from game_coordinator import GameCoordinator


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
                        action_masks=coordinator.all_action_masks,
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


def _compute_discounted_returns(
    rewards: torch.Tensor, episode_lengths: torch.Tensor, gamma: float = 0.99
) -> torch.Tensor:
    """
    Compute discounted returns (on-the-fly during aggregation).
    """
    device = rewards.device
    max_len = episode_lengths.max().item()
    num_episodes = episode_lengths.shape[0]

    # Create padded episode tensor
    episodes = torch.zeros(num_episodes, max_len, device=device)

    # Fill in episodes
    start_idx = 0
    for i, length in enumerate(episode_lengths):
        ep_len = length.item()
        episodes[i, :ep_len] = rewards[start_idx : start_idx + ep_len]
        start_idx += ep_len

    # Create discount matrix: [1, gamma, gamma^2, ..., gamma^(max_len-1)]
    discounts = gamma ** torch.arange(max_len, device=device)

    # Compute returns using convolution-like operation
    returns_padded = torch.zeros_like(episodes)
    for i in range(max_len):
        # For position i, sum rewards[i:] * discounts[:len-i]
        remaining = max_len - i
        returns_padded[:, i] = (episodes[:, i:] * discounts[:remaining]).sum(dim=1)

    # Flatten back to original shape
    returns = torch.zeros_like(rewards)
    start_idx = 0
    for i, length in enumerate(episode_lengths):
        ep_len = length.item()
        returns[start_idx : start_idx + ep_len] = returns_padded[i, :ep_len]
        start_idx += ep_len

    return returns


def aggregate_worker_data(cfg: DictConfig, run_timestamp: str) -> Path | None:
    """
    Aggregate data from all workers into a single HDF5 dataset file.

    Reads the individual pickle files produced by each worker, aggregates them
    into a single HDF5 file, precomputes returns, and cleans up the pickles.

    Args:
        cfg: Configuration dictionary.
        run_timestamp: Timestamp string for this run.

    Returns:
        Path to the aggregated HDF5 file, or None if no data was found.
    """
    import h5py

    run_dir = Path(cfg.collect.output_dir) / run_timestamp
    worker_dirs = sorted(run_dir.glob("worker_*"))

    if not worker_dirs:
        print("No worker data found to aggregate")
        return None

    print(f"\nAggregating data from {len(worker_dirs)} workers into HDF5...")

    aggregated_h5_path = run_dir / "aggregated_data.h5"

    total_episodes = 0
    total_timesteps = 0
    total_sim_time = 0.0
    worker_metadata_list = []

    # Initialize HDF5 file
    with h5py.File(aggregated_h5_path, "w") as f:
        # Datasets will be created on first write
        dsets = {}

        for worker_dir in worker_dirs:
            # Find all checkpoint files
            checkpoint_files = sorted(
                worker_dir.glob("data_checkpoint_*.pkl"),
                key=lambda p: int(p.stem.split("_")[-1]),
            )

            if not checkpoint_files:
                print(f"Warning: No data checkpoints found in {worker_dir}, skipping")
                continue

            for checkpoint_path in checkpoint_files:
                with open(checkpoint_path, "rb") as pf:
                    data = pickle.load(pf)

                # Extract Team 0 data ONLY
                team_0 = data["team_0"]
                tokens = team_0["tokens"]
                actions = team_0["actions"]
                rewards = team_0["rewards"]
                action_masks = team_0.get("action_masks")
                
                # Check for action_masks backward compat
                if action_masks is None:
                    # Create default ones if missing
                    action_masks = torch.ones(actions.shape[0], actions.shape[1], dtype=torch.float32)

                episode_lengths = data["episode_lengths"]
                episode_ids = data["episode_ids"] + total_episodes # Adjust IDs

                # Compute Returns immediately
                gamma = cfg.train.rl.gamma if "rl" in cfg.train else 0.99
                returns = _compute_discounted_returns(rewards, episode_lengths, gamma=gamma)

                # Prepare batch dict
                batch_data = {
                    "tokens": tokens,
                    "actions": actions,
                    "action_masks": action_masks,
                    "rewards": rewards,
                    "returns": returns,
                    "episode_ids": episode_ids,
                    "episode_lengths": episode_lengths, # 1D array
                }

                # Write to HDF5
                for key, tensor in batch_data.items():
                    np_data = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
                    
                    if key not in dsets:
                        # Create dataset with maxshape for resizing
                        shape = list(np_data.shape)
                        maxshape = list(np_data.shape)
                        maxshape[0] = None # Allow first dim refactor
                        
                        dsets[key] = f.create_dataset(
                            key, 
                            data=np_data, 
                            maxshape=tuple(maxshape),
                            chunks=True # Enable chunking for resize
                        )
                    else:
                        # Resize and append
                        dset = dsets[key]
                        old_len = dset.shape[0]
                        new_len = old_len + np_data.shape[0]
                        
                        dset.resize(new_len, axis=0)
                        dset[old_len:] = np_data

                # Update stats
                total_episodes += data["metadata"]["num_episodes"]
                total_timesteps += data["metadata"]["total_timesteps"]
                total_sim_time += data["metadata"]["total_sim_time"]
                worker_metadata_list.append(data["metadata"])
                
                # Cleanup Pickle
                try:
                    checkpoint_path.unlink()
                except OSError as e:
                    print(f"Warning: Failed to delete {checkpoint_path}: {e}")

        if total_episodes == 0:
            print("No episodes found in worker data.")
            return None
        
        # Save Metadata as Attributes
        f.attrs["num_episodes"] = total_episodes
        f.attrs["total_timesteps"] = total_timesteps
        f.attrs["total_sim_time"] = total_sim_time
        f.attrs["run_timestamp"] = run_timestamp
        f.attrs["num_workers"] = len(worker_dirs)
        
        # Save constant config from first worker
        if worker_metadata_list:
            first_meta = worker_metadata_list[0]
            f.attrs["max_ships"] = first_meta["max_ships"]
            f.attrs["token_dim"] = first_meta["token_dim"]
            f.attrs["num_actions"] = first_meta["num_actions"]

    print(f"Saved aggregated data to {aggregated_h5_path}")
    print(f"Aggregation complete: {total_episodes} episodes, {total_timesteps} timesteps")
    print("Cleanup complete: All worker pickle files deleted.")
    
    return aggregated_h5_path


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
