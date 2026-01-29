
import os
import torch
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Any

from env2.env import TensorEnv
from core.config import ShipConfig
from env2.agents.scripted import VectorScriptedAgent
from env2.collector import AsyncCollector

def run_collection(args: Any) -> None:
    """
    Runs the massive data collection process.
    
    Args:
        args: Parsed arguments containing configuration like num_envs, total_steps, etc.
    """
    device = torch.device(args.device)
    print(f"Running on {device} with {args.num_envs} envs.")
    
    # Create Output Directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = ShipConfig()
    
    # Initialize Environment
    env = TensorEnv(args.num_envs, config, device=device)
    obs = env.reset(seed=args.seed)
    
    # Initialize with low health to ensure episodes finish quickly for verification
    # This allows us to collect completed episodes faster in the start
    env.state.ship_health[:] = 10.0
    
    # Initialize Agent
    agent = VectorScriptedAgent(config)
    
    # Initialize Collector
    hdf5_path = output_path / "aggregated_data.h5"
    collector = AsyncCollector(str(hdf5_path), args.num_envs, env.max_ships, device)
    
    # Collection Loop
    progress_bar = tqdm(total=args.total_steps, desc="Collecting", unit="step")
    start_time = time.time()
    
    try:
        current_steps = 0
        while current_steps < args.total_steps:
             # 1. Compute Actions
            actions = agent.get_actions(env.state)
            
            # 2. Step Environment
            prev_obs = obs
            
            obs, rewards, dones, _, _ = env.step(actions)
            
            # 3. Collect Data
            collector.step(prev_obs, actions, rewards, dones)
            
            current_steps += 1 
            
            if current_steps % 100 == 0:
                progress_bar.update(100)
                
    except KeyboardInterrupt:
        print("Stopping collection early...")
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        total_transitions = current_steps * args.num_envs
        fps = total_transitions / elapsed if elapsed > 0 else 0
        
        print(f"Finished {current_steps} steps.")
        print(f"Total Transitions: {total_transitions}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Throughput: {fps:.2f} transitions/sec")
        
        collector.close()
        progress_bar.close()

def main() -> None:
    """Main function to parse arguments and run collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--total_steps", type=int, default=100_000, help="Total steps to simulate")
    parser.add_argument("--output_dir", type=str, default="data/bc_pretraining/massive_collection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    run_collection(args)

def collect_massive(cfg: Any) -> None:
    """
    Hydra entry point for massive collection.
    
    Args:
        cfg: The Hydra configuration object.
    """
    class Args:
        pass
    
    args = Args()
    # Extract arguments from Hydra config
    collect_cfg = cfg.get("collect", {})
    args.num_envs = collect_cfg.get("num_envs", 1024)
    args.total_steps = collect_cfg.get("total_steps", 100000)
    args.output_dir = "data/massive_collection"
    args.seed = cfg.get("seed", 42)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_collection(args)

if __name__ == "__main__":
    main()
