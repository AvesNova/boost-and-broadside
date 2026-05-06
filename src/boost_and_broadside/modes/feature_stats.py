"""feature_stats mode: run parallel games to collect feature delta distributions."""

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.modes.agent_factory import (
    get_actions,
    init_hidden,
    reset_done_envs,
    resolve_agent_spec,
)
from boost_and_broadside.modes.collect import _obs_from_state

def run_feature_stats_mode(
    team0_spec: str,
    team1_spec: str,
    num_envs: int,
    num_steps: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    model_config: ModelConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "/home/vizia/.gemini/antigravity/artifacts",
) -> None:
    """Run parallel games and calculate theoretical and empirical delta state distributions."""
    B = num_envs
    N = env_config.num_ships
    num_tokens = N + env_config.num_obstacles
    dev = torch.device(device)

    agent0 = resolve_agent_spec(team0_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N)
    agent1 = resolve_agent_spec(team1_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N)

    env = TensorEnv(B, ship_config, env_config, device)
    init_hidden(agent0, B, num_tokens, dev)
    init_hidden(agent1, B, num_tokens, dev)
    env.reset()

    delta_pos = []
    delta_vel = []
    delta_angle = []
    delta_health = []
    delta_power = []

    prev_pos = env.state.ship_pos.clone()
    prev_vel = env.state.ship_vel.clone()
    prev_att = env.state.ship_attitude.clone()
    prev_health = env.state.ship_health.clone()
    prev_power = env.state.ship_power.clone()
    prev_alive = env.state.ship_alive.clone()

    t0 = time.perf_counter()
    print(f"Collecting features for {num_steps} steps in {B} envs...")

    for step in range(num_steps):
        obs = _obs_from_state(env.state, ship_config)
        action0 = get_actions(agent0, obs, env.state, B, N, dev)
        action1 = get_actions(agent1, obs, env.state, B, N, dev)
        team_id = env.state.ship_team_id
        action = torch.where((team_id == 0).unsqueeze(-1), action0, action1)

        dones, truncated = env.step(action)
        
        valid = prev_alive & env.state.ship_alive & ~(dones | truncated).unsqueeze(-1)
        if valid.any():
            w, h = ship_config.world_size
            d_pos = env.state.ship_pos[valid] - prev_pos[valid]
            d_pos.real = (d_pos.real + w/2) % w - w/2
            d_pos.imag = (d_pos.imag + h/2) % h - h/2
            delta_pos.append(d_pos.abs().cpu().numpy())
            
            d_vel = env.state.ship_vel[valid] - prev_vel[valid]
            delta_vel.append(d_vel.abs().cpu().numpy())
            
            d_att = env.state.ship_attitude[valid] * torch.conj(prev_att[valid])
            delta_angle.append(torch.angle(d_att).abs().cpu().numpy())
            
            delta_health.append((env.state.ship_health[valid] - prev_health[valid]).cpu().numpy())
            delta_power.append((env.state.ship_power[valid] - prev_power[valid]).cpu().numpy())

        prev_pos = env.state.ship_pos.clone()
        prev_vel = env.state.ship_vel.clone()
        prev_att = env.state.ship_attitude.clone()
        prev_health = env.state.ship_health.clone()
        prev_power = env.state.ship_power.clone()
        prev_alive = env.state.ship_alive.clone()
        
        done_any = dones | truncated
        if done_any.any():
            env.reset_envs(done_any)
            reset_done_envs(agent0, done_any, num_tokens)
            reset_done_envs(agent1, done_any, num_tokens)
            prev_pos[done_any] = env.state.ship_pos[done_any]
            prev_vel[done_any] = env.state.ship_vel[done_any]
            prev_att[done_any] = env.state.ship_attitude[done_any]
            prev_health[done_any] = env.state.ship_health[done_any]
            prev_power[done_any] = env.state.ship_power[done_any]
            prev_alive[done_any] = env.state.ship_alive[done_any]

    elapsed = time.perf_counter() - t0
    print(f"Collection done in {elapsed:.2f}s.")

    all_d_pos = np.concatenate(delta_pos)
    all_d_vel = np.concatenate(delta_vel)
    all_d_angle = np.concatenate(delta_angle)
    all_d_health = np.concatenate(delta_health)
    all_d_power = np.concatenate(delta_power)

    # Calculate theoretical maxes
    dt = ship_config.dt
    max_speed = np.sqrt(ship_config.boost_thrust / ship_config.no_turn_drag_coeff)
    max_d_pos = max_speed * dt
    max_lift_force = ship_config.sharp_turn_lift_coeff * (max_speed**2)
    max_d_vel = (ship_config.boost_thrust + max_lift_force) * dt
    max_d_angle = ship_config.sharp_turn_angle
    max_d_power = ((ship_config.boost_thrust / ship_config.power_speed_constant) * max_speed) * dt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    features = {
        "Delta Position (px per step)": (all_d_pos, max_d_pos),
        "Delta Velocity (px per step)": (all_d_vel, max_d_vel),
        "Delta Angle (rad per step)": (all_d_angle, max_d_angle),
        "Delta Health per step": (all_d_health, None),
        "Delta Power per step": (all_d_power, max_d_power),
    }

    md_content = ["# Feature Delta Distributions & Theoretical Maxes\n"]
    md_content.append("This document analyzes the per-step changes (deltas) in the environment's state features.")
    md_content.append("It includes both empirical distributions from simulated gameplay and theoretical maximum values based on the physics engine.\n")
    
    md_content.append("## Theoretical Max Deltas\n")
    md_content.append(f"- **Max Speed (Terminal Velocity):** `{max_speed:.2f}` px/s")
    md_content.append(f"- **Max $\\Delta$ Position:** `{max_d_pos:.4f}` px/step")
    md_content.append(f"- **Max $\\Delta$ Velocity (thrust + lift at max speed):** `{max_d_vel:.4f}` px/step")
    md_content.append(f"- **Max $\\Delta$ Angle:** `{max_d_angle:.4f}` rad/step")
    md_content.append(f"- **Max $\\Delta$ Power Drain:** `{max_d_power:.4f}` power/step\n")
    
    md_content.append("## Empirical Distributions\n")
    md_content.append("| Feature | Mean $\\Delta$ | MSE (Variance) | Empirical Max | Theoretical Max |")
    md_content.append("|---------|---------------|----------------|---------------|-----------------|")

    for name, (data, max_val) in features.items():
        plt.figure(figsize=(8, 4))
        plt.hist(data, bins=100, alpha=0.7, color='blue', edgecolor='black')
        if max_val is not None:
            plt.axvline(max_val, color='red', linestyle='dashed', linewidth=2, label=f'Theoretical Max: {max_val:.2f}')
        plt.axvline(np.max(data), color='green', linestyle='dashed', linewidth=2, label=f'Empirical Max: {np.max(data):.2f}')
        plt.title(f"Distribution of {name}")
        plt.xlabel(name)
        plt.ylabel("Frequency")
        plt.yscale('log')
        plt.legend()
        
        filename = f"{name.split(' ')[1].lower()}_dist.png"
        filepath = out_dir / filename
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

        mean_val = np.mean(data)
        variance = np.var(data)  # This is the MSE of predicting the mean
        emp_max = np.max(data)
        
        max_str = f"{max_val:.4f}" if max_val is not None else "N/A"
        md_content.append(f"| {name} | {mean_val:.4f} | {variance:.4f} | {emp_max:.4f} | {max_str} |")
        
        md_content.append(f"\n### {name}\n")
        md_content.append(f"![{name}](file://{filepath.resolve()})\n")

    md_path = out_dir / "feature_distributions.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_content))
    print(f"Done! Report generated at {md_path}")
