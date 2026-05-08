"""feature_stats mode: collect encoded AUX feature null-model MSE for weight calibration.

For each consecutive obs pair (excluding episode boundaries and dead ships), computes the
squared error a null model (predicting zero delta for everything) would make per AUX target
feature. The inverse of the mean squared error is the calibrated static weight.

Feature layout (AUX_TARGET, 16 dims):
  [0:4]   pos_x(sin,cos), pos_y(sin,cos)   — toroidal phase
  [4:6]   vel_dir(cos,sin)                  — velocity direction unit vec
  [6]     symlog_speed                       — symlog |v|
  [7:9]   att(cos,sin)                      — heading unit vec
  [9]     ang_vel                            — symlog ω  (absolute target)
  [10:12] health(sin,cos)                   — quarter-wave
  [12:14] power(sin,cos)                    — quarter-wave
  [14:16] cooldown(sin,cos)                 — quarter-wave

Null-model squared error per feature:
  All delta features [0:9, 10:16]: (next - curr)^2  (null predicts no change)
  Absolute feature  [9]:           next^2            (null predicts 0)
"""

import time
import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.modes.agent_factory import (
    _extract_aux_cont,
    get_actions,
    init_hidden,
    reset_done_envs,
    resolve_agent_spec,
)
from boost_and_broadside.modes.collect import _obs_from_state

_FEAT_NAMES = (
    "pos_sin_x", "pos_cos_x", "pos_sin_y", "pos_cos_y",
    "vel_dir_cos", "vel_dir_sin",
    "symlog_speed",
    "att_cos", "att_sin",
    "ang_vel_abs",
    "health_sin", "health_cos",
    "power_sin", "power_cos",
    "cooldown_sin", "cooldown_cos",
)
_ANG_VEL_IDX = 9  # absolute feature — null MSE = E[target^2], not E[delta^2]


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
    num_tokens = N + env_config.num_obstacles
    dev = torch.device(device)

    agent0 = resolve_agent_spec(team0_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N)
    agent1 = resolve_agent_spec(team1_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N)

    env = TensorEnv(B, ship_config, env_config, device)
    init_hidden(agent0, B, num_tokens, dev)
    init_hidden(agent1, B, num_tokens, dev)
    env.reset()

    n_feat = len(_FEAT_NAMES)
    sq_err_sum = torch.zeros(n_feat, device=dev)
    count = torch.zeros(1, device=dev)

    obs = _obs_from_state(env.state, ship_config)
    prev_aux   = _extract_aux_cont(obs, N, ship_config)   # (B, N, 16)
    prev_alive = env.state.ship_alive.clone()             # (B, N)

    t0 = time.perf_counter()
    print(f"Collecting AUX null-model MSE for {num_steps} steps across {B} envs...")

    for step in range(num_steps):
        obs = _obs_from_state(env.state, ship_config)
        action0 = get_actions(agent0, obs, env.state, B, N, dev)
        action1 = get_actions(agent1, obs, env.state, B, N, dev)
        team_id = env.state.ship_team_id
        action = torch.where((team_id == 0).unsqueeze(-1), action0, action1)

        dones, truncated = env.step(action)

        next_obs   = _obs_from_state(env.state, ship_config)
        next_aux   = _extract_aux_cont(next_obs, N, ship_config)   # (B, N, 16)
        next_alive = env.state.ship_alive.clone()                  # (B, N)

        # Valid: both ships alive this step and no episode boundary
        episode_end = (dones | truncated).unsqueeze(-1)            # (B, 1)
        valid = prev_alive & next_alive & ~episode_end             # (B, N)

        if valid.any():
            v_curr = prev_aux[valid]   # (K, 16)
            v_next = next_aux[valid]   # (K, 16)

            # Null-model squared error per feature
            null_sq = (v_next - v_curr).pow(2)                    # (K, 16)  delta features
            null_sq[:, _ANG_VEL_IDX] = v_next[:, _ANG_VEL_IDX].pow(2)  # absolute feature

            sq_err_sum += null_sq.sum(0)
            count       += valid.sum().float()

        # Handle episode resets
        done_any = dones | truncated
        if done_any.any():
            env.reset_envs(done_any)
            reset_done_envs(agent0, done_any, num_tokens)
            reset_done_envs(agent1, done_any, num_tokens)

        next_obs_after_reset = _obs_from_state(env.state, ship_config)
        prev_aux             = _extract_aux_cont(next_obs_after_reset, N, ship_config)
        prev_alive           = env.state.ship_alive.clone()

        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{num_steps}  valid samples so far: {int(count.item()) * N}")

    elapsed = time.perf_counter() - t0
    n = count.item()
    print(f"\nDone in {elapsed:.1f}s — {int(n * N):,} valid (ship, step) pairs collected.\n")

    mean_sq = (sq_err_sum / max(n, 1.0)).cpu()
    weights  = 1.0 / mean_sq.clamp(min=1e-9)

    print("=" * 60)
    print("Per-feature null-model MSE and calibrated weights:")
    print(f"{'Feature':<20}  {'Mean sq err':>14}  {'Weight (1/MSE)':>16}")
    print("-" * 60)
    for i, name in enumerate(_FEAT_NAMES):
        print(f"{name:<20}  {mean_sq[i].item():>14.6f}  {weights[i].item():>16.1f}")
    print("=" * 60)

    print("\nPaste into _STATIC_AUX_WEIGHTS in ppo.py:\n")
    vals = [f"{weights[i].item():.1f}" for i in range(n_feat)]
    print("_STATIC_AUX_WEIGHTS = (")
    print(f"    {vals[0]}, {vals[1]}, {vals[2]}, {vals[3]},  # pos_sin_x, pos_cos_x, pos_sin_y, pos_cos_y")
    print(f"    {vals[4]}, {vals[5]},  # vel_dir_cos, vel_dir_sin")
    print(f"    {vals[6]},  # symlog_speed")
    print(f"    {vals[7]}, {vals[8]},  # att_cos, att_sin")
    print(f"    {vals[9]},  # ang_vel_abs")
    print(f"    {vals[10]}, {vals[11]},  # health_sin, health_cos")
    print(f"    {vals[12]}, {vals[13]},  # power_sin, power_cos")
    print(f"    {vals[14]}, {vals[15]},  # cooldown_sin, cooldown_cos")
    print(")")
