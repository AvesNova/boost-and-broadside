
import os
import torch
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# Adjust path if needed
import sys
# sys.path.append("src") # Usually not needed if run via uv/module

from env2.env import TensorEnv
from env2.state import ShipConfig
from env2.agents.scripted import VectorScriptedAgent
from env2.collector import AsyncCollector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--total_steps", type=int, default=100_000, help="Total steps to simulate")
    parser.add_argument("--output_dir", type=str, default="data/bc_pretraining/massive_collection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Running on {device} with {args.num_envs} envs.")
    
    # Create Output Dir
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Config
    config = ShipConfig() # Default config
    
    # Init Env
    env = TensorEnv(args.num_envs, config, device=device)
    obs = env.reset(seed=args.seed)
    
    # Init Init with low health to ensure episodes finish quickly for verification
    env.state.ship_health[:] = 10.0
    
    # Init Agent
    agent = VectorScriptedAgent(config)
    
    # Init Collector
    hdf5_path = output_path / "aggregated_data.h5"
    collector = AsyncCollector(str(hdf5_path), args.num_envs, env.max_ships, device)
    
    # Loop
    pbar = tqdm(total=args.total_steps, desc="Collecting", unit="step")
    start_time = time.time()
    
    try:
        current_steps = 0
        while current_steps < args.total_steps:
            # 1. Action
            # obs is dict of tensors (B, N)
            # agent expects TensorState?
            # VectorScriptedAgent expected TensorState in my implementation.
            # But Env.step returns obs_dict.
            # I should either modifying Agent to use obs_dict OR pass env.state to agent.
            # Passing env.state is Cheat/Oracle but ScriptedAgent IS an oracle (uses perfect state).
            # env.state is available.
            
            actions = agent.get_actions(env.state)
            
            # 2. Step
            # Env step returns new_obs, rewards, dones...
            # But collector needs PREVIOUS obs?
            # Standard: Store (obs_t, a_t, r_t+1, d_t+1).
            # Collector step signature: obs, actions, rewards, dones.
            # If I pass `obs` (from before step), `actions` (just computed), `rewards` (from step), `dones` (from step).
            # This is (s_t, a_t, r_t+1, d_t+1). Correct for BC (Learning mapping s_t -> a_t).
            
            prev_obs = obs # dict
            
            obs, rewards, dones, _, _ = env.step(actions)
            
            # 3. Collect
            collector.step(prev_obs, actions, rewards, dones)
            
            current_steps += 1 # 1 step = B transitions?
            # Usually total_steps means global simulation steps.
            # Total transitions = steps * B.
            
            if current_steps % 100 == 0:
                pbar.update(100)
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        total_transitions = current_steps * args.num_envs
        fps = total_transitions / elapsed
        
        print(f"Finished {current_steps} steps.")
        print(f"Total Transitions: {total_transitions}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Throughput: {fps:.2f} transitions/sec")
        
        collector.close()
        pbar.close()

if __name__ == "__main__":
    main()
