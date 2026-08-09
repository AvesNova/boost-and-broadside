"""Autoregressive-rollout diagnostic measurement.

`bnb ar-report` runs one ground-truth episode, then replays it two ways through the
policy's learned next-state predictor: a *closed-loop* rollout (the recorded actions are
forced, so only the imagined dynamics drift) and an *open-loop* rollout (the policy also
imagines its own actions).

The three rollouts are the measurement, and they are what the artifact stores: one
``result.npz`` holding every recorded field for every ship at every step, plus the
metadata a renderer needs to read it. The report itself — trajectory maps, per-metric
divergence charts, and the markdown that links them — is a rendering contract owned by
``bnb publish``, so a changed figure never means a replayed episode.
"""

import numpy as np
import torch

from boost_and_broadside.artifacts import ArtifactRecipe, ArtifactStore
from boost_and_broadside.config import EnvConfig, ModelConfig, RewardConfig, ShipConfig
from boost_and_broadside.constants import DEFAULT_MAX_BULLETS_PER_SHIP
from boost_and_broadside.env.observation import YemongObservation
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.evaluation.agents import (
    ResolvedAgent,
    agents_read_bullets,
    get_actions,
    init_hidden,
    resolve_agent_spec,
)
from boost_and_broadside.evaluation.match import merge_team_actions
from boost_and_broadside.evaluation.next_state import decode_targets_to_observation
from boost_and_broadside.evaluation.subjects import describe_agents, describe_environment

History = list[dict[str, torch.Tensor]]

# Every field recorded per step, and the shape the renderer expects it in.
_ROLLOUT_FIELDS = (
    "pos",
    "vel",
    "att",
    "ang_vel",
    "health",
    "power",
    "cooldown",
    "alive",
    "alive_prob",
)
_SCHEMA_VERSION = 1


def run_canonical_ar_report_mode(
    team0_spec: str,
    team1_spec: str,
    num_steps: int,
    ship_config: ShipConfig,
    rewards: RewardConfig,
    model_config: ModelConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    store: ArtifactStore | None = None,
) -> dict:
    """Run the one canonical AR report: a 4v4 diagnostic.

    The CLI owns subjects and budget only. Keeping the scenario here prevents
    adapters from silently restoring the retired 2v2/1v1 report pair.
    """

    return run_ar_report_mode(
        team0_spec=team0_spec,
        team1_spec=team1_spec,
        num_steps=num_steps,
        ship_config=ship_config,
        env_config=EnvConfig(
            num_ships=8,
            max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
            max_episode_steps=num_steps,
        ),
        rewards=rewards,
        model_config=model_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        store=store,
    )


def run_ar_report_mode(
    team0_spec: str,
    team1_spec: str,
    num_steps: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    rewards: RewardConfig,
    model_config: ModelConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    store: ArtifactStore | None = None,
) -> dict:
    print("Initializing agents...")
    agent0 = resolve_agent_spec(
        team0_spec,
        ship_config,
        model_config,
        device,
        checkpoint_dir,
        num_ships=env_config.num_ships,
    )
    agent1 = resolve_agent_spec(
        team1_spec,
        ship_config,
        model_config,
        device,
        checkpoint_dir,
        num_ships=env_config.num_ships,
    )

    wrapper = YemongEnvWrapper(
        num_envs=1,
        ship_config=ship_config,
        env_config=env_config,
        rewards=rewards,
        device=device,
        field_map=None,
        include_bullets=agents_read_bullets(agent0, agent1),
    )

    N = wrapper.num_ships
    num_tokens = N + env_config.num_fields

    print("Running ground truth simulation...")
    obs = wrapper.reset()
    init_hidden(agent0, 1, num_tokens, device)
    init_hidden(agent1, 1, num_tokens, device)

    # Save initial state for AR
    init_obs = YemongObservation(data={k: v.clone() for k, v in obs.items()})
    init_hidden0 = agent0.hidden.clone() if agent0.hidden is not None else None
    init_hidden1 = agent1.hidden.clone() if agent1.hidden is not None else None

    history_sim: History = []
    actions_sim = []

    for _ in range(num_steps):
        state = wrapper.state
        action0 = get_actions(agent0, obs, state, 1, N, device, return_pred_next=False)
        action1 = get_actions(agent1, obs, state, 1, N, device, return_pred_next=False)

        team_id = obs["team_id"][:, :N]
        action = merge_team_actions(action0, action1, team_id)
        actions_sim.append(action.clone())

        history_sim.append(
            {
                "pos": obs["pos"][:, :N].clone(),
                "vel": obs["vel"][:, :N].clone(),
                "att": obs["att"][:, :N].clone(),
                "ang_vel": obs["ang_vel"][:, :N].clone(),
                "health": obs["health"][:, :N].clone(),
                "power": obs["power"][:, :N].clone(),
                "cooldown": obs["cooldown"][:, :N].clone(),
                "alive": obs["alive"][:, :N].clone(),
                "alive_prob": torch.ones_like(obs["alive"][:, :N], dtype=torch.float32),
            }
        )

        obs, _, terminated, truncated, _ = wrapper.step(action)

        if terminated or truncated:
            print(f"Episode finished early at step {_}. Truncating rollout.")
            break

    actual_steps = len(history_sim)

    print("Running AR Rollout (Closed Loop)...")
    history_closed = _run_ar(
        agent0,
        agent1,
        init_obs,
        init_hidden0,
        init_hidden1,
        actual_steps,
        N,
        actions_sim,
        True,
    )

    print("Running AR Rollout (Open Loop)...")
    history_open = _run_ar(
        agent0,
        agent1,
        init_obs,
        init_hidden0,
        init_hidden1,
        actual_steps,
        N,
        None,
        False,
    )

    store = store or ArtifactStore(checkpoint_root=checkpoint_dir)
    recipe = ArtifactRecipe(
        artifact_type="ar-report",
        result_schema_version=_SCHEMA_VERSION,
        subjects=describe_agents(
            checkpoint_root=checkpoint_dir, team0=team0_spec, team1=team1_spec
        ),
        parameters={
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
    result = {
        "schema_version": _SCHEMA_VERSION,
        "num_ships": N,
        "num_steps": actual_steps,
        "world_size": list(ship_config.world_size),
        "agents": {"team0": team0_spec, "team1": team1_spec},
        "rollouts": ["gt", "cl", "ol"],
        "fields": list(_ROLLOUT_FIELDS),
    }
    artifact.write_json(result)
    artifact.write_npz(
        {
            **_rollout_arrays("gt", history_sim),
            **_rollout_arrays("cl", history_closed),
            **_rollout_arrays("ol", history_open),
        }
    )
    artifact.complete()
    print(f"Done! Wrote {artifact.path}.")
    return result


def _rollout_arrays(prefix: str, history: History) -> dict[str, np.ndarray]:
    """Stack one rollout's recorded fields into ``<prefix>_<field>`` arrays.

    The leading batch dimension is squeezed out: an AR report is one episode, so
    every array is (steps, ships, ...) and reads the same for all three rollouts.
    """

    return {
        f"{prefix}_{field}": np.stack(
            [step[field].squeeze(0).cpu().numpy() for step in history]
        ).astype(np.float32)
        for field in _ROLLOUT_FIELDS
    }


def _run_ar(
    agent0: ResolvedAgent,
    agent1: ResolvedAgent,
    init_obs: YemongObservation,
    init_hidden0: torch.Tensor | None,
    init_hidden1: torch.Tensor | None,
    num_steps: int,
    N: int,
    forced_actions: list[torch.Tensor] | None,
    is_closed_loop: bool,
) -> History:
    obs = YemongObservation(data={k: v.clone() for k, v in init_obs.items()})
    if agent0.hidden is not None:
        agent0.hidden = init_hidden0.clone()
    if agent1.hidden is not None:
        agent1.hidden = init_hidden1.clone()

    # Get coordinator from whichever agent is a policy (prefer agent0)
    coordinator = None
    if agent0.kind == "policy":
        coordinator = agent0.agent.coordinator
    elif agent1.kind == "policy":
        coordinator = agent1.agent.coordinator

    history: History = []
    curr_ship_targets = coordinator.get_target_vector(obs)[:, :N] if coordinator else None

    for step in range(num_steps):
        action0, pred_next0 = get_actions(
            agent0, obs, None, 1, N, obs["pos"].device, return_pred_next=True
        )
        action1, pred_next1 = get_actions(
            agent1, obs, None, 1, N, obs["pos"].device, return_pred_next=True
        )

        team_id = obs["team_id"][:, :N]
        mask = (team_id == 0).unsqueeze(-1)

        # Merge imagined actions and predictions
        imag_action = torch.where(mask, action0, action1)
        pred_next = None
        if pred_next0 is not None and pred_next1 is not None:
            pred_next = torch.where(mask, pred_next0, pred_next1)
        elif pred_next0 is not None:
            pred_next = torch.where(mask, pred_next0, torch.zeros_like(pred_next0))
        elif pred_next1 is not None:
            pred_next = torch.where(mask, torch.zeros_like(pred_next1), pred_next1)

        # Use forced action if closed loop
        action_to_apply = (
            forced_actions[step] if is_closed_loop and forced_actions is not None else imag_action
        )

        # Record state before applying delta
        history.append(
            {
                "pos": obs["pos"][:, :N].clone(),
                "vel": obs["vel"][:, :N].clone(),
                "att": obs["att"][:, :N].clone(),
                "ang_vel": obs["ang_vel"][:, :N].clone(),
                "health": obs["health"][:, :N].clone(),
                "power": obs["power"][:, :N].clone(),
                "cooldown": obs["cooldown"][:, :N].clone(),
                "alive": obs["alive"][:, :N].clone(),
                "alive_prob": obs["alive"][:, :N].float().clone(),
            }
        )

        if pred_next is not None and coordinator is not None:
            next_ship_targets = coordinator.apply_scaled_predictions(curr_ship_targets, pred_next)
            obs = decode_targets_to_observation(
                next_ship_targets, obs, action_to_apply, N, coordinator
            )
            curr_ship_targets = coordinator.get_target_vector(obs)[:, :N]

    return history
