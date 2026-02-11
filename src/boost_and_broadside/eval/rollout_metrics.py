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
        max_ships=eval_env_config.get("max_ships", 8),
        world_size=eval_env_config.get("world_size", (1024.0, 1024.0)),
        action_dim=12,
        state_dim=10,
        embed_dim=getattr(model.config, "embed_dim", getattr(model.config, "d_model", 128)),
        n_layers=model.config.n_layers,
        n_heads=model.config.n_heads,
        context_len=model.config.max_context_len,
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
        
        with torch.no_grad():
            dream_states, gen_actions = model.generate(
                initial_token,
                initial_action,
                initial_pos=initial_pos,
                steps=max_steps,
                n_ships=wm_agent.max_ships,
            )
        # dream_states: (1, Steps, N, F)
        # gen_actions: (1, Steps, N, 12) - One Hot

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
