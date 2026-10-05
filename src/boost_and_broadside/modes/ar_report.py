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

import math

import numpy as np
import torch

from boost_and_broadside.artifacts import ArtifactRecipe, ArtifactStore
from boost_and_broadside.config import EnvConfig, ModelConfig, RewardConfig, ShipConfig
from boost_and_broadside.constants import DEFAULT_MAX_BULLETS_PER_SHIP
from boost_and_broadside.env.observation import YemongObservation, observation_from_state
from boost_and_broadside.evaluation.agents import (
    ResolvedAgent,
    get_actions,
    resolve_agent_spec,
)
from boost_and_broadside.evaluation.environment import (
    create_evaluation_env,
    resolve_evaluation_environment,
)
from boost_and_broadside.evaluation.match import MatchRunner
from boost_and_broadside.evaluation.next_state import means_to_observation
from boost_and_broadside.evaluation.subjects import describe_agents, describe_environment
from boost_and_broadside.train.rl.physical_belief import (
    predicted_means,
    predicted_uncertainty,
)

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

    # A fields policy reports on the map distribution it was trained on; without
    # this it would be diagnosed in an empty arena it never saw.
    env_config = resolve_evaluation_environment(
        env_config, (agent0, agent1), ship_config=ship_config
    )

    del rewards  # The shared evaluation runtime does not compute training rewards.
    env = create_evaluation_env(1, ship_config, env_config, device)
    N = env_config.num_ships

    print("Running ground truth simulation...")
    env.reset()
    runner = MatchRunner(
        env,
        [agent0, agent1],
        torch.zeros(1, dtype=torch.long, device=device),
        torch.ones(1, dtype=torch.long, device=device),
        ship_config,
        N,
    )
    runner.init_hidden()
    index_log_scale = 2.0 * math.log(ship_config.field_index_step)

    history_sim: History = []
    actions_sim = []
    init_obs = None
    init_hidden0 = None
    init_hidden1 = None

    for step in range(num_steps):
        truth_obs = observation_from_state(
            env.state, ship_config, include_bullets=runner.include_bullets
        )
        hidden0_before = agent0.hidden.clone() if agent0.hidden is not None else None
        hidden1_before = agent1.hidden.clone() if agent1.hidden is not None else None
        runner.observe()
        selection = runner.select_actions(trace_agents=frozenset({0, 1}))
        actions_sim.append(selection.action.clone())

        if init_obs is None:
            # AR replay is anchored to the exact model input used for the first
            # real decision, including belief and the neutral pending queue.
            first_view = selection.observations[0]
            init_obs = _clone_observation(first_view)
            init_hidden0 = hidden0_before
            init_hidden1 = hidden1_before

        history_sim.append(
            {
                "pos": truth_obs["pos"][:, :N].clone(),
                "vel": truth_obs["vel"][:, :N].clone(),
                "att": truth_obs["att"][:, :N].clone(),
                "ang_vel": truth_obs["ang_vel"][:, :N].clone(),
                "health": truth_obs["health"][:, :N].clone(),
                "power": truth_obs["power"][:, :N].clone(),
                "cooldown": truth_obs["cooldown"][:, :N].clone(),
                "alive": truth_obs["alive"][:, :N].clone(),
                "alive_prob": torch.ones_like(truth_obs["alive"][:, :N], dtype=torch.float32),
            }
        )

        terminated, truncated = runner.advance(selection.action)

        if bool((terminated | truncated).any()):
            print(f"Episode finished early at step {step}. Truncating rollout.")
            break

    actual_steps = len(history_sim)
    if init_obs is None:
        raise ValueError("ar-report requires at least one decision step")

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
        index_log_scale,
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
        actions_sim,
        False,
        index_log_scale,
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
            "environment": describe_environment(env_config, ship_config=ship_config),
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
    # The rollouts above are what every report reads. The inputs that produced
    # them — the forced action sequence and the state both replays started from —
    # are what a later predictor needs to re-imagine this same episode without
    # replaying the environment, so they are retained as an ignored local payload.
    artifact.write_samples_npz(
        {
            "actions": np.stack([action.squeeze(0).cpu().numpy() for action in actions_sim]).astype(
                np.float32
            ),
            **{
                f"init_{key}": value.squeeze(0).cpu().numpy().astype(np.float32)
                for key, value in init_obs.items()
            },
        },
        "rollout_inputs.npz",
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


def _clone_observation(observation: YemongObservation) -> YemongObservation:
    """Detach a diagnostic snapshot from the runtime's mutable buffers."""
    return YemongObservation(
        data={key: value.clone() for key, value in observation.items()},
        bullets=(
            None
            if observation.bullets is None
            else {key: value.clone() for key, value in observation.bullets.items()}
        ),
    )


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
    index_log_scale: float,
) -> History:
    obs = _clone_observation(init_obs)
    if agent0.hidden is not None:
        agent0.hidden = init_hidden0.clone()
    if agent1.hidden is not None:
        agent1.hidden = init_hidden1.clone()

    history: History = []

    for step in range(num_steps):
        recorded = (
            forced_actions[step]
            if forced_actions is not None
            else torch.zeros((1, N, 3), dtype=torch.int32, device=obs["pos"].device)
        )
        if agent0.kind == "policy":
            action0, pred_next0 = get_actions(
                agent0, obs, None, 1, N, obs["pos"].device, return_pred_next=True
            )
        else:
            # Scripted/random controllers require authoritative physics state,
            # which an imagined observation cannot reconstruct. Reuse the real
            # episode's decision for their side while neural policies remain AR.
            action0, pred_next0 = recorded, None
        if agent1.kind == "policy":
            action1, pred_next1 = get_actions(
                agent1, obs, None, 1, N, obs["pos"].device, return_pred_next=True
            )
        else:
            action1, pred_next1 = recorded, None

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

        if pred_next is not None:
            obs = means_to_observation(
                predicted_means(pred_next.float()),
                obs,
                action_to_apply,
                N,
                index_log_scale,
                uncertainty=predicted_uncertainty(pred_next.float()),
            )

    return history
