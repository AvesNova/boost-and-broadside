
import torch
import numpy as np
import logging
from env.env import Environment
from agents.world_model_agent import WorldModelAgent
from agents.scripted import ScriptedAgent
from agents.tokenizer import observation_to_tokens

log = logging.getLogger(__name__)

def compute_rollout_metrics(
    model, 
    env_config, 
    device, 
    num_scenarios=1, 
    max_steps=128, 
    step_intervals=None
):
    """
    Computes Rollout MSE metrics comparing Expert (Scripted) vs Agent (WorldModel).
    
    Generates three trajectories for each scenario (seed):
    1. Expert (ScriptedAgent) in Real Env ("expert")
    2. Agent (WorldModelAgent) in Real Env ("sim") -> Closed Loop
    3. Agent (WorldModelAgent) in Dream ("dream") -> Open Loop
    
    Args:
        model: WorldModel instance.
        env_config: Environment config dict.
        device: Torch device.
        num_scenarios: Number of random scenarios to evaluate per epoch.
        max_steps: Maximum steps per scenario.
        step_intervals: List of steps to aggregate MSE for (e.g. [1, 2, 4, ...]).
                        If None, computes for all steps up to max_steps.
                        
    Returns:
        Dictionary containing:
            - "mse_sim": Average MSE(Sim vs Expert)
            - "mse_dream": Average MSE(Dream vs Expert)
            - "step_mse_sim": {step: mse}
            - "step_mse_dream": {step: mse}
            - "full_mse_sim": (steps,) array of average MSE per step
            - "full_mse_dream": (steps,) array of average MSE per step
    """
    
    if step_intervals is None:
        step_intervals = [1, 2, 4, 8, 16, 32, 64, 128]
        step_intervals = [s for s in step_intervals if s <= max_steps]
        
    model.eval()
    
    # Initialize accumulators for average MSE per step
    accum_sq_err_sim = torch.zeros(max_steps, device=device)
    accum_sq_err_dream = torch.zeros(max_steps, device=device)
    counts = torch.zeros(max_steps, device=device)
    
    # Prepare Env
    eval_env_config = env_config.copy()
    eval_env_config["render_mode"] = "none"
    if "memory_size" not in eval_env_config:
         eval_env_config["memory_size"] = 2
         
    env = Environment(**eval_env_config)
    
    # Prepare Agents
    expert_config = {
         "max_shooting_range": 500.0,
         "angle_threshold": 5.0,
         "bullet_speed": 500.0,
         "target_radius": 10.0,
         "radius_multiplier": 1.5,
         "world_size": eval_env_config["world_size"]
    }
    expert = ScriptedAgent(**expert_config)
    
    wm_agent = WorldModelAgent(
        agent_id="wm_agent",
        team_id=0,
        squad=[0],
        model=model,
        device=str(device),
        max_ships=eval_env_config.get("max_ships", 8),
        action_dim=12,
        state_dim=10,
        embed_dim=model.config.embed_dim,
        n_layers=model.config.n_layers,
        n_heads=model.config.n_heads,
        context_len=model.config.max_context_len
    )
    
    rng = np.random.default_rng(42) 
    
    for i in range(num_scenarios):
        scenario_seed = int(rng.integers(0, 1000000))
        
        # --- 1. Expert Trajectory ---
        obs, _ = env.reset(seed=scenario_seed, game_mode="1v1")
        expert_tokens = []
        
        # We record State at t=1..max_steps (Result of actions)
        # Assuming initial state error is 0 (same seed)
        
        curr_obs = obs
        for t in range(max_steps):
            actions = expert(curr_obs, ship_ids=[0])
            full_actions = actions.copy()
            full_actions[1] = torch.tensor([0.0, 0.0, 0.0])
            
            next_obs, _, _, _, _ = env.step(full_actions)
            
            next_token = observation_to_tokens(next_obs, perspective=0).to(device)
            expert_tokens.append(next_token)
            
            curr_obs = next_obs
            
        expert_tensor = torch.stack(expert_tokens) # (Steps, 1, N, F)
        
        # --- 2. Agent Closed Loop (Sim) ---
        env.reset(seed=scenario_seed, game_mode="1v1")
        wm_agent.reset()
        
        curr_obs, _ = env.reset(seed=scenario_seed, game_mode="1v1") 
        
        sim_sq_errs = []
        
        for t in range(max_steps):
            actions = wm_agent(curr_obs, ship_ids=[0])
            full_actions = actions.copy()
            full_actions[1] = torch.tensor([0.0, 0.0, 0.0])
            
            next_obs, _, _, _, _ = env.step(full_actions)
            
            next_token = observation_to_tokens(next_obs, perspective=0).to(device)
            
            sq_err = (next_token - expert_tensor[t]).pow(2).mean()
            sim_sq_errs.append(sq_err)
            
            curr_obs = next_obs
            
        for t, err in enumerate(sim_sq_errs):
            accum_sq_err_sim[t] += err
        
        # --- 3. Agent Open Loop (Dream) ---
        obs, _ = env.reset(seed=scenario_seed, game_mode="1v1") 
        initial_token = observation_to_tokens(obs, perspective=0).to(device) # (1, N, F)
        
        initial_action = torch.zeros(1, wm_agent.max_ships, 12, device=device)
        
        with torch.no_grad():
             dream_states, _ = model.generate(
                initial_token, 
                initial_action, 
                steps=max_steps, 
                n_ships=wm_agent.max_ships
            )
        # dream_states: (1, Steps, N, F) -> permute to (Steps, 1, N, F)
        # model.generate returns state at t+1, t+2... matching our expert log loop.
        dream_states = dream_states.permute(1, 0, 2, 3)
        
        sq_errs_dream = (dream_states - expert_tensor).pow(2)
        step_sq_errs_dream = sq_errs_dream.mean(dim=(1, 2, 3)) 
        
        accum_sq_err_dream += step_sq_errs_dream
        counts += 1.0
        
    env.close()
    
    avg_full_mse_sim = accum_sq_err_sim / counts
    avg_full_mse_dream = accum_sq_err_dream / counts
    
    step_mse_sim = {}
    step_mse_dream = {}
    
    for s in step_intervals:
        idx = s - 1
        if idx < len(avg_full_mse_sim):
            step_mse_sim[s] = avg_full_mse_sim[idx].item()
            step_mse_dream[s] = avg_full_mse_dream[idx].item()
            
    overall_mse_sim = avg_full_mse_sim.mean().item()
    overall_mse_dream = avg_full_mse_dream.mean().item()
    
    return {
        "mse_sim": overall_mse_sim,
        "mse_dream": overall_mse_dream,
        "step_mse_sim": step_mse_sim,
        "step_mse_dream": step_mse_dream,
        "full_mse_sim": avg_full_mse_sim.cpu().numpy(),
        "full_mse_dream": avg_full_mse_dream.cpu().numpy()
    }
