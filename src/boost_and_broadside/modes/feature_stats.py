"""feature_stats mode: collect label null-model MSE to validate/calibrate label_scale values.

For each consecutive obs pair (excluding episode boundaries and dead ships), computes
coordinator.compute_labels() then squares it. Since labels are pre-scaled by label_scale,
the null-model MSE in scaled space should be ~1.0 if label_scale is well calibrated
(label_scale = 1/std(raw_label) → scaled_label has std ≈ 1 → null MSE ≈ 1).

Reports per-prediction-dim stats and suggested label_scale corrections:
  suggested_scale = current_scale / sqrt(mean_sq)
"""

import time

import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import observation_from_state
from boost_and_broadside.modes.agent_factory import (
    get_actions,
    init_hidden,
    reset_done_envs,
    resolve_agent_spec,
)
from boost_and_broadside.train.rl.features import build_standard_coordinator


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
    B = num_envs
    N = env_config.num_ships
    num_tokens = N + env_config.num_fields
    dev = torch.device(device)

    coordinator = build_standard_coordinator(ship_config)
    feat_names = coordinator.get_feature_names()
    P = coordinator.total_prediction_dimension
    curr_scale = coordinator.label_scale_vector(dev)

    agent0 = resolve_agent_spec(
        team0_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N
    )
    agent1 = resolve_agent_spec(
        team1_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N
    )

    env = TensorEnv(B, ship_config, env_config, device)
    init_hidden(agent0, B, num_tokens, dev)
    init_hidden(agent1, B, num_tokens, dev)
    env.reset()

    sq_err_sum = torch.zeros(P, device=dev)
    count = torch.zeros(1, device=dev)

    obs = observation_from_state(env.state, ship_config)
    prev_targets = coordinator.get_target_vector(obs)[:, :N]  # (B, N, target_dim)
    prev_alive = env.state.ship_alive.clone()

    t0 = time.perf_counter()
    print(f"Collecting label null-model MSE for {num_steps} steps across {B} envs...")

    for step in range(num_steps):
        obs = observation_from_state(env.state, ship_config)
        action0 = get_actions(agent0, obs, env.state, B, N, dev)
        action1 = get_actions(agent1, obs, env.state, B, N, dev)
        team_id = env.state.ship_team_id
        action = torch.where((team_id == 0).unsqueeze(-1), action0, action1)

        dones, truncated = env.step(action)

        next_obs = observation_from_state(env.state, ship_config)
        next_targets = coordinator.get_target_vector(next_obs)[:, :N]
        next_alive = env.state.ship_alive.clone()

        # Valid: both ships alive this step and no episode boundary
        episode_end = (dones | truncated).unsqueeze(-1)  # (B, 1)
        valid = prev_alive & next_alive & ~episode_end  # (B, N)

        if valid.any():
            v_curr = prev_targets[valid]  # (K, target_dim)
            v_next = next_targets[valid]  # (K, target_dim)
            labels = coordinator.compute_labels(v_curr, v_next)  # (K, P) scaled
            sq_err_sum += labels.pow(2).sum(0)
            count += valid.sum().float()

        done_any = dones | truncated
        if done_any.any():
            env.reset_envs(done_any)
            reset_done_envs(agent0, done_any, num_tokens)
            reset_done_envs(agent1, done_any, num_tokens)

        next_obs_after_reset = observation_from_state(env.state, ship_config)
        prev_targets = coordinator.get_target_vector(next_obs_after_reset)[:, :N]
        prev_alive = env.state.ship_alive.clone()

        if (step + 1) % 500 == 0:
            print(f"  step {step + 1}/{num_steps}  valid samples: {int(count.item()) * N:,}")

    elapsed = time.perf_counter() - t0
    n = count.item()
    print(f"\nDone in {elapsed:.1f}s — {int(n * N):,} valid (ship, step) pairs.\n")

    mean_sq = (sq_err_sum / max(n, 1.0)).cpu()
    curr_scale_cpu = curr_scale.cpu()
    suggested = curr_scale_cpu / mean_sq.sqrt().clamp(min=1e-9)

    print("=" * 72)
    print("Null-model MSE in scaled label space (target ≈ 1.0 if well-calibrated)")
    print(
        f"{'Feature':<24}  {'Mean sq (scaled)':>18}  {'Current scale':>14}  {'Suggested scale':>16}"
    )
    print("-" * 72)
    for i, name in enumerate(feat_names):
        print(
            f"{name:<24}  {mean_sq[i].item():>18.4f}  "
            f"{curr_scale_cpu[i].item():>14.2f}  {suggested[i].item():>16.2f}"
        )
    print("=" * 72)

    print("\nSuggested label_scale values for build_standard_coordinator:")
    for i, name in enumerate(feat_names):
        print(f"  {name}: {suggested[i].item():.1f}")
