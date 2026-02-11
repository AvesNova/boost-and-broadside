import logging
import torch
from boost_and_broadside.env2.coordinator_wrapper import TensorEnvWrapper
from boost_and_broadside.agents.world_model_agent import WorldModelAgent
from boost_and_broadside.agents.tokenizer import observation_to_tokens
from boost_and_broadside.utils.tensor_utils import to_one_hot

log = logging.getLogger(__name__)


def compute_dreaming_error(model, val_loader, device, max_steps=19, num_batches=10):
    """
    Computes autoregressive rollout MSE (Open Loop).

    Args:
        model: WorldModel instance.
        val_loader: DataLoader for validation data.
        device: Torch device.
        max_steps: Maximum rollout steps (default 19, matching training config).
        num_batches: Number of batches to evaluate.

    Returns:
        Average MSE over all timesteps.
    """
    model.eval()
    total_mse = 0.0
    num_samples = 0

    with torch.no_grad():
        for i, (states, actions, returns, mask, action_masks) in enumerate(val_loader):
            if i >= num_batches:
                break

            states = states.to(device)
            # Use raw actions for now, or convert if needed by generate?
            # model.generate expects keys (initial_state, initial_action)
            # initial_action needs to be one-hot.

            actions_oh = to_one_hot(actions).to(device)

            initial_state = states[:, 0, :, :]  # (B, N, F)
            initial_action = actions_oh[:, 0, :, :]  # (B, N, 12)

            # Ground truth for comparison
            # We want to match length.
            steps_available = states.shape[1] - 1
            steps_to_gen = min(steps_available, max_steps)

            if steps_to_gen <= 0:
                continue

            # Generate rollout
            gen_states, _ = model.generate(
                initial_state,
                initial_action,
                steps=steps_to_gen,
                n_ships=states.shape[2],
            )
            # gen_states: (B, Steps, N, F)

            ground_truth = states[:, 1 : steps_to_gen + 1, :, :]

            # Compute MSE
            mse = (gen_states - ground_truth).pow(2).mean()
            total_mse += mse.item()
            num_samples += 1

    return total_mse / num_samples if num_samples > 0 else 0.0


def compute_controlling_error(
    model, env_config, device, max_episode_length=200, num_episodes=1
):
    """
    Computes one-step prediction error during live control (Closed Loop).
    Also returns average reward.

    Args:
        model: WorldModel instance.
        env_config: Environment configuration dict.
        device: Torch device.
        max_episode_length: Max steps per episode.
        num_episodes: Number of episodes to run.

    Returns:
        (avg_mse, avg_reward)
    """
    model.eval()
    total_mse = 0.0
    total_reward = 0.0
    total_steps = 0

    # Initialize Agent with the EXISTING model instance
    # We use a dummy squad [0] just for initialization,
    # but we'll manually handle the loop or use the agent properly 1v1.
    # Let's run a simple 1v1 against a dummy or static opponent for stability?
    # Or just 1v1 self-play if possible?
    # WorldModelAgent is designed to control a squad.

    # Let's run 1v1 where Agent 0 is WorldModel and Agent 1 is passive/random.
    # Actually, let's just use the agent to control team 0.

    agent = WorldModelAgent(
        agent_id="eval_agent",
        team_id=0,
        squad=[0],  # Assuming 1v1 with 1 ship
        model=model,
        device=str(device),
        max_ships=env_config.get("max_ships", 8),
        world_size=env_config.get("world_size", (1024.0, 1024.0)),
        action_dim=12,  # Hardcoded for now
        state_dim=10,  # Hardcoded
        embed_dim=model.config.embed_dim,  # Match model
        n_layers=model.config.n_layers,
        n_heads=model.config.n_heads,
        context_len=model.config.max_context_len,
    )

    # Create Env
    # Ensure config allows headless
    eval_env_config = env_config.copy()
    eval_env_config["render_mode"] = "none"

    env = TensorEnvWrapper(**eval_env_config)

    for _ in range(num_episodes):
        obs, _ = env.reset(game_mode="1v1")  # 1v1 single ship per team
        agent.reset()

        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < max_episode_length:
            # 1. Agent predicts next action AND we can peek at its state prediction?
            # WorldModelAgent.__call__ runs the model.
            # We want to check the model's STATE prediction for the NEXT step.

            # To do this non-intrusively, we might need to manually invoke the model here
            # mirroring what the agent does, OR trust the agent.
            # But the agent only returns actions.

            # Let's manually perform the prediction step here to get the state error.
            # The agent.history has the past.
            # We can replicate the agent's forward pass slightly to get pred_states.

            # Prepare inputs from agent history + current observation
            # Get current token
            current_token = observation_to_tokens(
                obs, 0, world_size=env_config["world_size"]
            ).to(device)  # (1, N, F)

            # Reconstruct history input
            hist_tokens = [t for t, a in agent.history]
            hist_actions = [a for t, a in agent.history]

            input_tokens_list = hist_tokens + [current_token]
            input_actions_list = hist_actions + [agent.last_action]

            # Append MASK for next step prediction
            mask_token = torch.zeros_like(current_token)
            mask_action = torch.zeros_like(agent.last_action)

            input_tokens_list.append(mask_token)
            input_actions_list.append(mask_action)

            input_tokens = torch.cat(input_tokens_list, dim=0).unsqueeze(
                0
            )  # (1, Seq, N, F)
            input_actions = torch.cat(input_actions_list, dim=0).unsqueeze(
                0
            )  # (1, Seq, N, A)

            seq_len = input_tokens.shape[1]
            mask = torch.zeros(
                1, seq_len, agent.max_ships, dtype=torch.bool, device=device
            )
            mask[:, -1, :] = True

            with torch.no_grad():
                pred_states, pred_actions_logits, _, _, _ = model(
                    input_tokens, input_actions, mask=mask
                )

            # The model predicts the state at masked position (last one)
            predicted_next_state = pred_states[:, -1, :, :]  # (1, N, F)

            # Get action from agent (it updates its history)
            actions_dict = agent(obs, ship_ids=[0])

            # Convert to full env actions (dummy opponent)
            # We need to supply actions for all ships to avoid crash?
            # Env expects dict[int, Tensor].
            # 1v1 has ships 0 and 1.
            full_actions = actions_dict.copy()
            # Add dummy action for opponent (ship 1)
            full_actions[1] = torch.tensor([0.0, 0.0, 0.0])  # Coast, Straight, NoShoot

            # Step Env
            next_obs, rewards, term, trunc, _ = env.step(full_actions)

            # Calculate control error: MSE between predicted_next_state and ACTUAL next state token
            # We need to tokenize next_obs
            next_token = observation_to_tokens(
                next_obs, 0, world_size=env_config["world_size"]
            ).to(device)

            mse = (predicted_next_state - next_token).pow(2).mean()
            total_mse += mse.item()
            total_steps += 1

            episode_reward += rewards[0]

            obs = next_obs
            done = term or trunc
            steps += 1

        total_reward += episode_reward

    env.close()

    avg_mse = total_mse / total_steps if total_steps > 0 else 0.0
    avg_reward = total_reward / num_episodes if num_episodes > 0 else 0.0

    return avg_mse, avg_reward
