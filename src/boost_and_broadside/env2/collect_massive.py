"""
Script for massive GPU-accelerated data collection using vectorized environments.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.env2.agents.scripted import VectorScriptedAgent
from boost_and_broadside.env2.agents.sticky_wrapper import VectorStickyAgent
from boost_and_broadside.env2.collector import AsyncCollector
from boost_and_broadside.env2.env import TensorEnv


@dataclass
class CollectionArgs:
    """Arguments for data collection."""

    num_envs: int
    total_steps: int
    output_dir: str
    seed: int
    device: str
    min_skill: float
    max_skill: float
    expert_ratio: float
    random_dist: str
    buffer_steps: int = 512
    num_workers: int = 4
    random_speed: bool = False
    min_speed: float = 1.0
    max_speed: float = 180.0
    default_speed: float = 100.0


def run_collection(args: CollectionArgs) -> None:
    """Runs the massive data collection process.

    Args:
        args: Configuration arguments for collection.
    """
    device = torch.device(args.device)
    print(f"Running on {device} with {args.num_envs} envs.")

    # Create Output Directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = ShipConfig(
        random_speed=args.random_speed,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        default_speed=args.default_speed,
    )

    # Initialize Environment
    env = TensorEnv(args.num_envs, config, device=device)
    obs = env.reset(seed=args.seed)

    # Initialize with low health to ensure episodes finish quickly for verification
    env.state.ship_health[:] = 10.0

    # Initialize Agent
    base_agent = VectorScriptedAgent(config)

    # Wrap with Sticky/Skill Logic
    print(
        f"Agent Skill Range: [{args.min_skill}, {args.max_skill}] | "
        f"Expert Ratio: {args.expert_ratio} | Noise: {args.random_dist}"
    )

    agent = VectorStickyAgent(
        base_agent=base_agent,
        num_envs=args.num_envs,
        max_ships=env.max_ships,
        device=device,
        min_skill=args.min_skill,
        max_skill=args.max_skill,
        expert_ratio=args.expert_ratio,
        random_dist=args.random_dist,  # type: ignore
    )

    # Initialize Collector
    hdf5_path = output_path / "aggregated_data.h5"
    collector = AsyncCollector(
        str(hdf5_path),
        args.num_envs,
        env.max_ships,
        device,
        max_steps=args.buffer_steps,
        num_workers=args.num_workers,
        total_steps=args.total_steps,
    )

    # Collection Loop
    progress_bar = tqdm(total=args.total_steps, desc="Collecting", unit="step")
    start_time = time.time()

    try:
        last_reported_steps = 0
        while collector.total_transitions_completed < args.total_steps:
            # 1. Compute Actions
            taken_actions, expert_actions, skills = agent.get_actions(env.state)

            # 2. Step Environment
            prev_obs = obs
            obs, rewards, dones, _, _ = env.step(taken_actions)

            # 3. Check for infinite loops (max steps)
            # If an env exceeds buffer_steps, force its 'dones' to True to trigger reset/capture
            if (collector.step_counts >= args.buffer_steps).any():
                over_limit = (collector.step_counts >= args.buffer_steps).to(dones.device)
                # Inject 'done' into the collector's view for these envs
                force_dones = dones | over_limit
                
                # We ALSO need to reset these in the environment so they don't stay in inf loop
                if over_limit.any():
                    env._reset_envs(over_limit)
                    # Update obs for reset envs
                    obs = env._get_obs()
            else:
                force_dones = dones

            # 4. Collect Data
            collector.step(
                prev_obs,
                taken_actions,
                rewards,
                force_dones,
                expert_actions=expert_actions,
                agent_skills=skills,
            )

            current_completed = collector.total_transitions_completed
            if current_completed > last_reported_steps:
                progress_bar.update(current_completed - last_reported_steps)
                last_reported_steps = current_completed

    except KeyboardInterrupt:
        print("Stopping collection early...")
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        total_transitions = collector.total_transitions_completed
        
        print("\nFinished collection.")
        print(f"Total Completed Transitions: {total_transitions}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Throughput: {total_transitions / elapsed if elapsed > 0 else 0:.2f} transitions/sec")

        collector.close()
        progress_bar.close()


def collect_massive(cfg: Any) -> None:
    """Hydra entry point for massive collection.

    Args:
        cfg: The Hydra configuration object.
    """
    collect_cfg = cfg.collect
    massive_cfg = collect_cfg.massive

    args = CollectionArgs(
        num_envs=massive_cfg.num_envs,
        total_steps=massive_cfg.steps,
        output_dir=collect_cfg.output_dir,
        seed=cfg.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        min_skill=collect_cfg.min_skill,
        max_skill=collect_cfg.max_skill,
        expert_ratio=collect_cfg.expert_ratio,
        random_dist=collect_cfg.random_dist,
        buffer_steps=massive_cfg.buffer_steps,
        num_workers=collect_cfg.num_workers,
    )

    run_collection(args)


if __name__ == "__main__":
    # If called directly, we don't have Hydra cfg, but we can't easily mixed them
    # Projects guideline: Entry point is main.py.
    # But for backward compatibility with the user command from earlier:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--output_dir", type=str, default="data/massive_collection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min_skill", type=float, default=1.0)
    parser.add_argument("--max_skill", type=float, default=1.0)
    parser.add_argument("--expert_ratio", type=float, default=0.0)
    parser.add_argument("--random_dist", type=str, default="beta")
    parser.add_argument("--buffer_steps", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    pa = parser.parse_args()

    args = CollectionArgs(
        num_envs=pa.num_envs,
        total_steps=pa.total_steps,
        output_dir=pa.output_dir,
        seed=pa.seed,
        device=pa.device,
        min_skill=pa.min_skill,
        max_skill=pa.max_skill,
        expert_ratio=pa.expert_ratio,
        random_dist=pa.random_dist,
        buffer_steps=pa.buffer_steps,
        num_workers=pa.num_workers,
    )
    run_collection(args)
