"""``feature-stats`` mode: collect label null-model MSE for label-scale calibration.

For each consecutive obs pair (excluding episode boundaries and dead ships), computes
coordinator.compute_labels() then squares it. Since labels are pre-scaled by label_scale,
the null-model MSE in scaled space should be ~1.0 if label_scale is well calibrated
(label_scale = 1/std(raw_label) → scaled_label has std ≈ 1 → null MSE ≈ 1).

Reports per-prediction-dim stats and suggested label_scale corrections:
  suggested_scale = current_scale / sqrt(mean_sq)

The measurement depends on both acting agents, the environment, and the sample
budget, so it is not a property of the profile alone: it writes a
``feature-stats`` artifact owned by the single run behind its checkpoints, or by
nothing at all.
"""

import time

import torch

from boost_and_broadside.artifacts import ArtifactRecipe, ArtifactStore
from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.observation import observation_from_state
from boost_and_broadside.evaluation.agents import (
    agents_read_bullets,
    get_actions,
    init_hidden,
    reset_done_envs,
    resolve_agent_spec,
)
from boost_and_broadside.evaluation.environment import create_evaluation_env
from boost_and_broadside.evaluation.match import merge_team_actions
from boost_and_broadside.evaluation.subjects import describe_agents, describe_environment
from boost_and_broadside.train.rl.features import build_standard_coordinator

_SCHEMA_VERSION = 1


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
    store: ArtifactStore | None = None,
) -> dict:
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

    include_bullets = agents_read_bullets(agent0, agent1)

    env = create_evaluation_env(B, ship_config, env_config, device)
    init_hidden(agent0, B, num_tokens, dev)
    init_hidden(agent1, B, num_tokens, dev)
    env.reset()

    sq_err_sum = torch.zeros(P, device=dev)
    count = torch.zeros(1, device=dev)

    obs = observation_from_state(env.state, ship_config, include_bullets=include_bullets)
    prev_targets = coordinator.get_target_vector(obs)[:, :N]  # (B, N, target_dim)
    prev_alive = env.state.ship_alive.clone()

    t0 = time.perf_counter()
    print(f"Collecting label null-model MSE for {num_steps} steps across {B} envs...")

    for step in range(num_steps):
        obs = observation_from_state(env.state, ship_config, include_bullets=include_bullets)
        action0 = get_actions(agent0, obs, env.state, B, N, dev)
        action1 = get_actions(agent1, obs, env.state, B, N, dev)
        team_id = env.state.ship_team_id
        action = merge_team_actions(action0, action1, team_id)

        dones, truncated = env.step(action)

        next_obs = observation_from_state(env.state, ship_config, include_bullets=include_bullets)
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

        next_obs_after_reset = observation_from_state(
            env.state, ship_config, include_bullets=include_bullets
        )
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

    result = {
        "schema_version": _SCHEMA_VERSION,
        "valid_pairs": int(n * N),
        "seconds": elapsed,
        "features": [
            {
                "name": name,
                "mean_sq_scaled": float(mean_sq[i].item()),
                "current_scale": float(curr_scale_cpu[i].item()),
                "suggested_scale": float(suggested[i].item()),
            }
            for i, name in enumerate(feat_names)
        ],
    }
    store = store or ArtifactStore(checkpoint_root=checkpoint_dir)
    recipe = ArtifactRecipe(
        artifact_type="feature-stats",
        result_schema_version=_SCHEMA_VERSION,
        subjects=describe_agents(
            checkpoint_root=checkpoint_dir, team0=team0_spec, team1=team1_spec
        ),
        parameters={
            "num_envs": num_envs,
            "decision_steps": num_steps,
            "environment": describe_environment(env_config),
        },
    )
    owner = store.owner_for(
        store.owning_run_for_paths(
            [spec for spec in (team0_spec, team1_spec) if spec.endswith(".pt")]
        )
    )
    artifact = store.create(recipe, owner)
    artifact.write_json(result)
    artifact.complete()
    print(f"\nWrote {artifact.path}")
    return result
