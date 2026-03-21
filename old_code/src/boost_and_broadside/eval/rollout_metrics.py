import torch
import numpy as np
import logging
from boost_and_broadside.env2.coordinator_wrapper import TensorEnvWrapper
from boost_and_broadside.agents.world_model_agent import WorldModelAgent
from boost_and_broadside.agents.scripted import ScriptedAgent
from boost_and_broadside.agents.tokenizer import observation_to_tokens

log = logging.getLogger(__name__)


def compute_rollout_metrics(
    model, env_config, device, num_scenarios=1, max_steps=128, step_intervals=None
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
            - "error_sim_power": Average Power action error (Sim)
            - "error_sim_turn": Average Turn action error (Sim)
            - "error_sim_shoot": Average Shoot action error (Sim)
            - "error_dream_power": Average Power action error (Dream)
            - "error_dream_turn": Average Turn action error (Dream)
            - "error_dream_shoot": Average Shoot action error (Dream)
    """

    if step_intervals is None:
        step_intervals = [1, 2, 4, 8, 16, 32, 64, 128]
        step_intervals = [s for s in step_intervals if s <= max_steps]

    model.eval()

    # Initialize accumulators for average MSE per step
    accum_sq_err_sim = torch.zeros(max_steps, device=device)
    accum_sq_err_dream = torch.zeros(max_steps, device=device)

    # Accumulators for Action Errors (Scalar sum over all steps/scenarios)
    # We will average by total_steps_count at the end
    total_error_sim = {"power": 0.0, "turn": 0.0, "shoot": 0.0}
    total_error_dream = {"power": 0.0, "turn": 0.0, "shoot": 0.0}

    counts = torch.zeros(max_steps, device=device)
    total_steps_count = 0

    # Prepare Env
    eval_env_config = env_config.copy()
    eval_env_config["render_mode"] = "none"
    if "memory_size" not in eval_env_config:
        eval_env_config["memory_size"] = 2

    env = TensorEnvWrapper(**eval_env_config)

    # Prepare Agents
    expert_config = {
        "max_shooting_range": 500.0,
        "angle_threshold": 5.0,
        "bullet_speed": 500.0,
        "target_radius": 10.0,
        "radius_multiplier": 1.5,
        "world_size": eval_env_config["world_size"],
    }
    expert = ScriptedAgent(**expert_config)

    wm_agent = WorldModelAgent(
        agent_id="wm_agent",
        team_id=0,
        squad=[0],
        model=model,
        device=str(device),
        max_ships=eval_env_config.get("max_ships", eval_env_config.get("environment", {}).get("max_ships", 8)),
        world_size=eval_env_config.get("world_size", (1024.0, 1024.0)),
    )

    rng = np.random.default_rng(42)

    for i in range(num_scenarios):
        scenario_seed = int(rng.integers(0, 1000000))

        # --- 1. Expert Trajectory ---
        obs, _ = env.reset(seed=scenario_seed, game_mode="1v1")

        # Get active ship IDs from env state
        # TensorEnv approach: state is env.env.state
        state = env.env.state
        team_ids = state.ship_team_id[0] # (N,)
        ship_alive = state.ship_alive[0] # (N,)
        
        team_0_ships = []
        team_1_ships = []
        
        for i in range(env.max_ships):
             if not ship_alive[i]:
                  continue
             tid = int(team_ids[i].item())
             if tid == 0:
                  team_0_ships.append(i)
             elif tid == 1:
                  team_1_ships.append(i)

        if not team_0_ships or not team_1_ships:
            log.warning("Skipping scenario with missing teams.")
            continue

        agent_id = team_0_ships[0]
        opponent_id = team_1_ships[0]

        expert_tokens = []
        expert_actions_list = []  # Store expert actions (indices) for comparison

        # We record State at t=1..max_steps (Result of actions)
        # Assuming initial state error is 0 (same seed)

        curr_obs = obs
        for t in range(max_steps):
            actions = expert(curr_obs, ship_ids=[agent_id])
            full_actions = actions.copy()
            # No-op for opponent
            full_actions[opponent_id] = torch.tensor([0.0, 0.0, 0.0])

            next_obs, _, _, _, _ = env.step(full_actions)

            next_token = observation_to_tokens(
                next_obs, perspective=0, world_size=eval_env_config["world_size"]
            ).to(device)
            expert_tokens.append(next_token)

            # Store Expert Action for this step (t)
            # actions[agent_id] is (3,) tensor of floats (indices)
            expert_actions_list.append(actions[agent_id].to(device))

            curr_obs = next_obs

        expert_tensor = torch.stack(expert_tokens)  # (Steps, 1, N, F)
        expert_actions_tensor = torch.stack(expert_actions_list)  # (Steps, 3)

        # --- 2. Agent Closed Loop (Sim) ---
        env.reset(seed=scenario_seed, game_mode="1v1")

        # Update WorldModelAgent squad
        wm_agent.squad = [agent_id]
        wm_agent.reset()

        curr_obs, _ = env.reset(seed=scenario_seed, game_mode="1v1")

        sim_sq_errs = []

        for t in range(max_steps):
            actions = wm_agent(curr_obs, ship_ids=[agent_id])
            full_actions = actions.copy()
            full_actions[opponent_id] = torch.tensor([0.0, 0.0, 0.0])

            next_obs, _, _, _, _ = env.step(full_actions)

            next_token = observation_to_tokens(
                next_obs, perspective=0, world_size=eval_env_config["world_size"]
            ).to(device)

            sq_err = (next_token - expert_tensor[t]).pow(2).mean()
            sim_sq_errs.append(sq_err)

            # --- Action Error (Sim) ---
            # actions[agent_id] is what the Agent chose at this step
            sim_action = actions[agent_id].to(device)  # (3,)
            exp_action = expert_actions_tensor[t]  # (3,)

            # Compare indices (Power=0, Turn=1, Shoot=2)
            total_error_sim["power"] += (sim_action[0] != exp_action[0]).float().item()
            total_error_sim["turn"] += (sim_action[1] != exp_action[1]).float().item()
            total_error_sim["shoot"] += (sim_action[2] != exp_action[2]).float().item()

            curr_obs = next_obs

        for t, err in enumerate(sim_sq_errs):
            accum_sq_err_sim[t] += err

        # --- 3. Agent Open Loop (Dream) ---
        obs, _ = env.reset(seed=scenario_seed, game_mode="1v1")
        initial_token = observation_to_tokens(
            obs, perspective=0, world_size=eval_env_config["world_size"]
        ).to(device)  # (1, N, F)

        initial_action = torch.zeros(1, wm_agent.max_ships, 12, device=device)

        # Observation from TensorEnvWrapper is a dict of tensors (N,). Position is complex.
        pos_complex = obs["position"].to(device) # (N,)
        initial_pos = torch.view_as_real(pos_complex).unsqueeze(0) # (1, N, 2)
        # Normalize/Scale if needed? MambaBB expects World Units (0-1024).
        # Obs is usually normalized or raw?
        # TensorEnvWrapper usually returns RAW (or check config).
        # Let's check `TensorEnvWrapper`. It likely returns raw pos.
        
        # Dream Rollout
        dream_tokens = [initial_token]
        dream_actions = []
        
        curr_dream_state = initial_token
        curr_dream_pos = initial_pos
        curr_dream_actions = initial_action # (1, N, 12) - wait, initial_action is 0s
        
        # We need a proper history for the model
        dream_history_states = [initial_token]
        dream_history_pos = [initial_pos]
        dream_history_actions = [torch.zeros(1, 1, wm_agent.max_ships, 3, device=device)]
        
        with torch.no_grad():
            for t in range(max_steps):
                # Prepare context
                s_in = torch.stack(dream_history_states, dim=1) # (1, t+1, N, F)
                p_in = torch.stack(dream_history_pos, dim=1)
                a_in = torch.stack(dream_history_actions, dim=1) # (1, t+1, N, 3)
                
                # Predict
                outputs = model(
                    state=s_in,
                    prev_action=a_in,
                    pos=p_in,
                    vel=s_in[:, -1, :, StateFeature.VX:StateFeature.VY+1],
                    team_ids=torch.zeros(1, s_in.shape[1], wm_agent.max_ships, device=device, dtype=torch.long),
                    world_size=eval_env_config.get("world_size", (1024.0, 1024.0))
                )
                pred_s, pred_a_logits, _, _, _ = outputs
                
                # Action
                last_logits = pred_a_logits[:, -1]
                p_idx = last_logits[..., 0:3].argmax(dim=-1)
                t_idx = last_logits[..., 3:10].argmax(dim=-1)
                s_idx = last_logits[..., 10:12].argmax(dim=-1)
                next_act_idx = torch.stack([p_idx, t_idx, s_idx], dim=-1).float()
                
                # Dream State Update (Delta)
                if pred_s is not None:
                    delta = pred_s[:, -1]
                    last_s = s_in[:, -1]
                    next_s = last_s.clone()
                    next_s[..., StateFeature.HEALTH] = (last_s[..., StateFeature.HEALTH] + delta[..., TargetFeature.DHEALTH]).clamp(0, 1)
                    next_s[..., StateFeature.POWER] = (last_s[..., StateFeature.POWER] + delta[..., TargetFeature.DPOWER]).clamp(0, 1)
                    next_s[..., StateFeature.VX] = (last_s[..., StateFeature.VX] + delta[..., TargetFeature.DVX])
                    next_s[..., StateFeature.VY] = (last_s[..., StateFeature.VY] + delta[..., TargetFeature.DVY])
                    next_s[..., StateFeature.ANG_VEL] = (last_s[..., StateFeature.ANG_VEL] + delta[..., TargetFeature.DANG_VEL])
                    
                    # Pos
                    last_p = p_in[:, -1]
                    next_p = last_p + delta[..., [TargetFeature.DX, TargetFeature.DY]]
                    # Toroidal wrap
                    ws = eval_env_config.get("world_size", (1024.0, 1024.0))
                    next_p[..., 0] %= ws[0]
                    next_p[..., 1] %= ws[1]
                else:
                    # Fallback for spatial models: no state update
                    next_s = last_s
                    next_p = p_in[:, -1]

                dream_history_states.append(next_s.unsqueeze(1))
                dream_history_pos.append(next_p.unsqueeze(1))
                dream_history_actions.append(next_act_idx.unsqueeze(1))
                
                dream_tokens.append(next_s)
                # Convert next_act_idx to one-hot for existing metric logic
                oh_p = torch.zeros(1, wm_agent.max_ships, 3, device=device).scatter_(-1, p_idx.unsqueeze(-1), 1.0)
                oh_t = torch.zeros(1, wm_agent.max_ships, 7, device=device).scatter_(-1, t_idx.unsqueeze(-1), 1.0)
                oh_s = torch.zeros(1, wm_agent.max_ships, 2, device=device).scatter_(-1, s_idx.unsqueeze(-1), 1.0)
                dream_actions.append(torch.cat([oh_p, oh_t, oh_s], dim=-1))

        # dream_states: (Steps, 1, N, F)
        dream_states = torch.stack(dream_tokens[1:]) # (Steps, 1, N, F)
        # gen_actions: (Steps, 1, N, 12)
        gen_actions = torch.stack(dream_actions).permute(1, 0, 2, 3) # (1, Steps, N, 12)

        # --- Action Error (Dream) ---
        # 1. Convert One-Hot to Indices for our agent (Index 0)
        # gen_actions shape: (1, Steps, N, 12)
        dream_actions_oh = gen_actions[0, :, 0, :]  # (Steps, 12) for agent ship 0

        # Split and Argmax
        p_idx = dream_actions_oh[:, 0:3].argmax(dim=-1)
        t_idx = dream_actions_oh[:, 3:10].argmax(dim=-1)
        s_idx = dream_actions_oh[:, 10:12].argmax(dim=-1)

        # Compare with Expert (Steps, 3)
        # Expert col 0=Power, 1=Turn, 2=Shoot
        err_p = (p_idx != expert_actions_tensor[:, 0]).float().sum().item()
        err_t = (t_idx != expert_actions_tensor[:, 1]).float().sum().item()
        err_s = (s_idx != expert_actions_tensor[:, 2]).float().sum().item()

        total_error_dream["power"] += err_p
        total_error_dream["turn"] += err_t
        total_error_dream["shoot"] += err_s

        dream_states = dream_states.permute(1, 0, 2, 3)

        sq_errs_dream = (dream_states - expert_tensor).pow(2)
        step_sq_errs_dream = sq_errs_dream.mean(dim=(1, 2, 3))

        accum_sq_err_dream += step_sq_errs_dream
        counts += 1.0
        total_steps_count += max_steps

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
        "full_mse_dream": avg_full_mse_dream.cpu().numpy(),
        # Action Errors
        "error_sim_power": total_error_sim["power"] / total_steps_count
        if total_steps_count > 0
        else 0.0,
        "error_sim_turn": total_error_sim["turn"] / total_steps_count
        if total_steps_count > 0
        else 0.0,
        "error_sim_shoot": total_error_sim["shoot"] / total_steps_count
        if total_steps_count > 0
        else 0.0,
        "error_dream_power": total_error_dream["power"] / total_steps_count
        if total_steps_count > 0
        else 0.0,
        "error_dream_turn": total_error_dream["turn"] / total_steps_count
        if total_steps_count > 0
        else 0.0,
        "error_dream_shoot": total_error_dream["shoot"] / total_steps_count
        if total_steps_count > 0
        else 0.0,
    }
