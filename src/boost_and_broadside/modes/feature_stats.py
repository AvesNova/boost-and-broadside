"""``feature-stats`` mode: null-model MSE for the fixed physical delta scales.

For each consecutive truth pair (excluding episode boundaries, dead ships and
respawn teleports), normalizes the eleven physical deltas by the Phase-1 scales
and squares them. A channel whose scale still conditions its labels reads ~1.0,
and the null model scores 1.0.

Reports per-channel stats and the scale that would restore that:
  suggested_scale = current_scale * sqrt(mean_sq)

This is a *check* on the one-time calibration in
``benchmarks/physical_delta_calibration.py``, not a fitting loop. The constants
are fixed by contract; a channel far from 1.0 here is a reason to re-run that
benchmark deliberately, not to nudge a number.

The measurement depends on both acting agents, the environment, and the sample
budget, so it is not a property of the profile alone: it writes a
``feature-stats`` artifact owned by the single run behind its checkpoints, or by
nothing at all.

Caveat: state is read directly -- unoccluded truth -- while controller decisions
use the ordinary perceived/belief view through ``MatchRunner``. The statistic
therefore measures truth-to-truth labels, while training re-bases the label on
the *believed* current state (see ``PPOTrainer._precompute_ns_labels``). The two
agree exactly for a visible ship and diverge for a hidden one, where the label
carries the belief correction and has a much wider distribution. So the scales
suggested here are a lower bound on what training sees, and the gap grows with
time spent out of contact.
"""

import time

import torch

from boost_and_broadside.artifacts import ArtifactRecipe, ArtifactStore
from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.evaluation.agents import resolve_agent_spec
from boost_and_broadside.evaluation.environment import (
    create_evaluation_env,
    resolve_evaluation_environment,
)
from boost_and_broadside.evaluation.match import MatchRunner
from boost_and_broadside.evaluation.subjects import describe_agents, describe_environment
from boost_and_broadside.train.rl.physical_belief import (
    PHYSICAL_MEAN_DIM,
    PHYSICAL_MEAN_NAMES,
    PhysicalNextState,
    physical_means_from_state,
)

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
    dev = torch.device(device)

    next_state = PhysicalNextState.from_ship_config(ship_config)
    feat_names = list(PHYSICAL_MEAN_NAMES)
    P = PHYSICAL_MEAN_DIM
    curr_scale = next_state.scale_vector(dev)

    agent0 = resolve_agent_spec(
        team0_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N
    )
    agent1 = resolve_agent_spec(
        team1_spec, ship_config, model_config, device, checkpoint_dir, num_ships=N
    )
    env_config = resolve_evaluation_environment(
        env_config, (agent0, agent1), ship_config=ship_config
    )

    env = create_evaluation_env(
        B,
        ship_config,
        env_config,
        device,
    )
    env.reset()
    runner = MatchRunner(
        env,
        [agent0, agent1],
        torch.zeros(B, dtype=torch.long, device=dev),
        torch.ones(B, dtype=torch.long, device=dev),
        ship_config,
        N,
    )
    runner.init_hidden()

    sq_err_sum = torch.zeros(P, device=dev)
    count = torch.zeros(1, device=dev)

    prev_means = physical_means_from_state(env.state, ship_config)[:, :N]  # (B, N, 11)
    prev_alive = env.state.ship_alive.clone()

    t0 = time.perf_counter()
    print(f"Collecting label null-model MSE for {num_steps} steps across {B} envs...")

    for step in range(num_steps):
        runner.observe()
        dones, truncated = runner.advance(runner.actions())

        next_means = physical_means_from_state(env.state, ship_config)[:, :N]
        next_alive = env.state.ship_alive.clone()

        # Valid: both ships alive this step and no episode boundary
        episode_end = (dones | truncated).unsqueeze(-1)  # (B, 1)
        valid = prev_alive & next_alive & ~episode_end & ~env.state.ship_respawned

        if valid.any():
            labels = next_state.labels(prev_means[valid], next_means[valid])  # (K, 11)
            sq_err_sum += labels.pow(2).sum(0)
            count += valid.sum().float()

        done_any = dones | truncated
        runner.reset_finished(done_any)

        prev_means = physical_means_from_state(env.state, ship_config)[:, :N]
        prev_alive = env.state.ship_alive.clone()

        if (step + 1) % 500 == 0:
            print(f"  step {step + 1}/{num_steps}  valid samples: {int(count.item()) * N:,}")

    elapsed = time.perf_counter() - t0
    n = count.item()
    print(f"\nDone in {elapsed:.1f}s — {int(n * N):,} valid (ship, step) pairs.\n")

    mean_sq = (sq_err_sum / max(n, 1.0)).cpu()
    curr_scale_cpu = curr_scale.cpu()
    suggested = curr_scale_cpu * mean_sq.sqrt()

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

    print("\nSuggested PHYSICAL_DELTA_SCALES values:")
    for i, name in enumerate(feat_names):
        print(f"  {name}: {suggested[i].item():.4g}")

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
            "environment": describe_environment(env_config, ship_config=ship_config),
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
