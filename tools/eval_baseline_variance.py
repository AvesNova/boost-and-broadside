import torch
import numpy as np
import hydra
from omegaconf import DictConfig

from boost_and_broadside.env2.env import TensorEnv
from boost_and_broadside.env2.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.env2.agents.evolvable_agent import BatchedEvolvableAgent

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    num_envs = 10000
    max_episode_steps = cfg.train.evolve.get("max_episode_steps", 512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Initializing {num_envs} environments for Baseline vs Baseline...")
    
    from boost_and_broadside.core.config import ShipConfig
    from omegaconf import OmegaConf

    env_cfg = OmegaConf.to_container(cfg.environment, resolve=True)
    valid_keys = ShipConfig.__annotations__.keys()
    ship_cfg_dict = {k: v for k, v in env_cfg.items() if k in valid_keys}
    ship_config = ShipConfig(**ship_cfg_dict)
    
    if "world_size" in env_cfg:
        ship_config.world_size = tuple(env_cfg["world_size"])
        
    env = TensorEnv(
        num_envs=num_envs,
        config=ship_config,
        device=device,
        max_ships=8,
        max_bullets=env_cfg.get('max_bullets', 20),
        max_episode_steps=max_episode_steps,
    )
    
    agent_logic = BatchedEvolvableAgent(env.config)
    
    default_v = torch.tensor(StochasticAgentConfig.default_vector(), device=device, dtype=torch.float32)
    baseline_agent = default_v.view(1, 1, 24).repeat(1, 4, 1)[0]
    
    obs = env.reset()
    
    config_tensor = torch.zeros(num_envs, env.max_ships, 24, device=device)
    team_ids = env.state.ship_team_id
    
    t0_mask = (team_ids == 0)
    t1_mask = (team_ids == 1)
    
    # Both teams are the exact same baseline agent
    baseline_repeated = baseline_agent.unsqueeze(0).repeat(num_envs, 1, 1)
    config_tensor[t0_mask] = baseline_repeated.reshape(-1, 24)
    config_tensor[t1_mask] = baseline_repeated.reshape(-1, 24)
    
    env_wins_team_0 = torch.zeros(num_envs, device=device)
    
    step = 0
    alive = torch.ones(num_envs, device=device, dtype=torch.bool)
    
    print(f"Running simulation for {max_episode_steps} steps max...")
    while alive.any() and step < max_episode_steps:
        actions = agent_logic.get_actions(env.state, config_tensor)
        _, _, terminated, truncated, info = env.step(actions)
        
        dones = terminated | truncated
        just_finished = alive & dones
        
        if just_finished.any():
            final_alive = info["final_observation"]["alive"]
            t0_ships_alive = (final_alive & t0_mask).sum(dim=1)
            t1_ships_alive = (final_alive & t1_mask).sum(dim=1)
            
            t0_won = just_finished & (t0_ships_alive > t1_ships_alive)
            t1_won = just_finished & (t1_ships_alive > t0_ships_alive)
            draws = just_finished & (t0_ships_alive == t1_ships_alive)
            
            env_wins_team_0 += t0_won.float() * 1.0
            env_wins_team_0 += draws.float() * 0.5
            env_wins_team_0 += t1_won.float() * 0.0
            
            alive[just_finished] = False
        step += 1
        
    print("Simulation complete.")
    results = env_wins_team_0.cpu().numpy()
    
    mean_win = np.mean(results)
    print(f"Overall Mean Win Rate (Team 0): {mean_win:.4f}")
    
    print("\nStatistical Analysis (Bootstrap over 10000 iterations):")
    print(f"{'Matches/Eval':>15} | {'Mean':>8} | {'Std Dev of Mean':>16} | {'95% CI Half-Width'}")
    print("-" * 65)
    for m in [1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 64, 128]:
        means = []
        for _ in range(10000):
            sample = np.random.choice(results, size=m, replace=True)
            means.append(np.mean(sample))
        std_eval = np.std(means)
        ci_95 = 1.96 * std_eval
        print(f"{m:15d} | {np.mean(means):8.4f} | +/- {std_eval:12.4f} | +/- {ci_95:.4f}")

if __name__ == "__main__":
    main()
