import multiprocessing as mp
import pickle
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

    # Calculate episodes distribution
    total_episodes_target = cfg.collect.total_episodes
    num_workers = cfg.collect.num_workers

    # Base episodes per worker (floor division)
    episodes_per_worker = total_episodes_target // num_workers

    # Distribute remainder
    if worker_id < (total_episodes_target % num_workers):
        episodes_per_worker += 1

    print(f"Worker {worker_id}: Target episodes = {episodes_per_worker}")

    # Parse Ratios
    type_ratios = cfg.collect.type_ratios
    ship_count_ratios = cfg.collect.ship_count_ratios

    # Calculate episode counts for this worker's share
    tasks = []  # List of (game_mode, scenario_type, team_skills, team_ids)

    import numpy as np

    rng = np.random.default_rng(worker_id)  # Seeding with worker_id

    # Generate task list based on ratios
    # We process ratios to exact counts

    # First, split by ship counts
    ship_counts_map = {}
    remaining_eps = episodes_per_worker

    ship_keys = list(ship_count_ratios.keys())
    ship_probs = np.array([ship_count_ratios[k] for k in ship_keys])
    ship_probs /= ship_probs.sum()  # Normalize

    # Deterministic assignment would be better but probabilistic is fine for large N
    # Let's iterate and accumulate

    assigned_total = 0
    for i, (k, p) in enumerate(zip(ship_keys, ship_probs)):
        if i == len(ship_keys) - 1:
            count = remaining_eps  # Assign remainder
        else:
            count = int(episodes_per_worker * p)
            remaining_eps -= count
        ship_counts_map[k] = count
        assigned_total += count

    # Now for each ship count, split by type
    type_keys = list(type_ratios.keys())
    type_probs = np.array([type_ratios[k] for k in type_keys])
    type_probs /= type_probs.sum()

    for mode, count in ship_counts_map.items():
        if count == 0:
            continue

        rem_mode_eps = count
        for i, (t_key, t_prob) in enumerate(zip(type_keys, type_probs)):
            if i == len(type_keys) - 1:
                t_count = rem_mode_eps
            else:
                t_count = int(count * t_prob)
                rem_mode_eps -= t_count

            for _ in range(t_count):
                tasks.append((mode, t_key))

    # Shuffle tasks to avoid blocks of same type
    rng.shuffle(tasks)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Worker {worker_id}: Collecting episodes...",
                total=len(tasks),
            )

            for mode, scenario_type in tasks:
                # Skill Sampling Logic
                X = float(rng.uniform(0.0, 1.0))

                # Default skills
                skill_0 = 1.0
                skill_1 = 1.0

                # Logic from user request
                # Type 1: E(1.0) vs E(1.0)
                # Type 2: E(1.0) vs E(X)
                # Type 3: E(X) vs E(1.0)
                # Type 4: E(X) vs E(X')
                X2 = float(rng.uniform(0.0, 1.0))

                team_ids_map = {0: 0, 1: 1}  # Default

                if scenario_type == "type1":
                    skill_0 = 1.0
                    skill_1 = 1.0
                elif scenario_type == "type2":
                    skill_0 = 1.0
                    skill_1 = X
                elif scenario_type == "type3":
                    skill_0 = X
                    skill_1 = 1.0
                elif scenario_type == "type4":
                    skill_0 = X
                    skill_1 = X2

                team_skills = {0: skill_0, 1: skill_1}

                coordinator.reset(game_mode=mode, team_skills=team_skills)
                episode_sim_time, terminated = coordinator.step()

                collector.add_episode(
                    tokens_team_0=coordinator.all_tokens[0],
                    tokens_team_1=coordinator.all_tokens[1],
                    actions=coordinator.all_actions,
                    expert_actions=coordinator.all_expert_actions,
                    action_masks=coordinator.all_action_masks,
                    rewards=coordinator.all_rewards,
                    sim_time=episode_sim_time,
                    agent_skills=team_skills,
                    team_ids=team_ids_map,
                )

                progress.update(task, advance=1)

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
    
    world_size = tuple(cfg.environment.world_size)

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

                # Extract Team 0 data
                team_0 = data["team_0"]
                t0_tokens = team_0["tokens"]
                t0_actions = team_0["actions"]
                t0_rewards = team_0["rewards"]
                t0_expert_actions = team_0.get("expert_actions")
                if t0_expert_actions is None:
                     # Fallback if not present (shouldn't happen with new code, but safe)
                     t0_expert_actions = t0_actions.clone()
                
                # Handle missing keys for backward compatibility
                t0_masks = team_0.get("action_masks")
                if t0_masks is None:
                    t0_masks = torch.ones(
                        t0_actions.shape[0], t0_actions.shape[1], dtype=torch.float32
                    )
                t0_skills = team_0.get("agent_skills")
                if t0_skills is None:
                    t0_skills = torch.ones(t0_actions.shape[0], dtype=torch.float32)
                t0_ids = team_0.get("team_ids")
                if t0_ids is None:
                    t0_ids = torch.zeros(t0_actions.shape[0], dtype=torch.int64)

                # Extract Team 1 data
                team_1 = data["team_1"]
                t1_tokens = team_1["tokens"]
                t1_actions = team_1["actions"]
                t1_rewards = team_1["rewards"]
                t1_expert_actions = team_1.get("expert_actions")
                if t1_expert_actions is None:
                     t1_expert_actions = t1_actions.clone()

                t1_masks = team_1.get("action_masks")
                if t1_masks is None:
                    t1_masks = torch.ones(
                        t1_actions.shape[0], t1_actions.shape[1], dtype=torch.float32
                    )
                t1_skills = team_1.get("agent_skills")
                if t1_skills is None:
                    t1_skills = torch.ones(t1_actions.shape[0], dtype=torch.float32)
                t1_ids = team_1.get("team_ids")
                if t1_ids is None:
                    t1_ids = torch.ones(t1_actions.shape[0], dtype=torch.int64)

                episode_lengths = data["episode_lengths"]
                episode_ids = data["episode_ids"] + total_episodes

                # Compute Returns for both
                gamma = cfg.train.rl.gamma if "rl" in cfg.train else 0.99
                t0_returns = _compute_discounted_returns(
                    t0_rewards, episode_lengths, gamma=gamma
                )
                t1_returns = _compute_discounted_returns(
                    t1_rewards, episode_lengths, gamma=gamma
                )
                

                # Concatenate Data
                tokens = torch.cat([t0_tokens, t1_tokens], dim=0)
                actions = torch.cat([t0_actions, t1_actions], dim=0)
                expert_actions = torch.cat([t0_expert_actions, t1_expert_actions], dim=0)
                
                action_masks = torch.cat([t0_masks, t1_masks], dim=0)
                rewards = torch.cat([t0_rewards, t1_rewards], dim=0)
                returns = torch.cat([t0_returns, t1_returns], dim=0)

                # Duplicate episode IDs for the second batch
                batch_episode_ids = torch.cat([episode_ids, episode_ids], dim=0)
                batch_episode_lengths = torch.cat(
                    [episode_lengths, episode_lengths], dim=0
                )

                agent_skills = torch.cat([t0_skills, t1_skills], dim=0)
                team_ids_tensor = torch.cat([t0_ids, t1_ids], dim=0)

                # Prepare batch dict
                batch_data = {
                    "tokens": tokens,
                    "actions": actions,
                    "expert_actions": expert_actions,
                    "action_masks": action_masks,
                    "rewards": rewards,
                    "returns": returns,
                    "episode_ids": batch_episode_ids,
                    "episode_lengths": batch_episode_lengths,
                    "agent_skills": agent_skills,
                    "team_ids": team_ids_tensor,
                }
                

                # Write to HDF5
                for key, tensor in batch_data.items():
                    np_data = (
                        tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
                    )

                    if key not in dsets:
                        # Create dataset with maxshape for resizing
                        shape = list(np_data.shape)
                        maxshape = list(np_data.shape)
                        maxshape[0] = None  # Allow first dim refactor

                        dsets[key] = f.create_dataset(
                            key,
                            data=np_data,
                            maxshape=tuple(maxshape),
                            chunks=True,  # Enable chunking for resize
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
                total_timesteps += data["metadata"]["total_timesteps"] * 2  # Both teams
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
            # Save world size if available, else optional
            if "world_size" in cfg.environment:
                f.attrs["world_size"] = tuple(cfg.environment.world_size)

    print(f"Saved aggregated data to {aggregated_h5_path}")
    print(
        f"Aggregation complete: {total_episodes} episodes, {total_timesteps} timesteps"
    )
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
