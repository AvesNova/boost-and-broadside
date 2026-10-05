"""Shared physical next-state composition and autoregressive imagination harness."""

import torch

from boost_and_broadside.constants import NUM_JOINT_ACTIONS
from boost_and_broadside.env.observation import GameMode, ObsKey, YemongObservation
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.runtime.actions import write_pending_action_view
from boost_and_broadside.train.rl.physical_belief import (
    ALIVE_HEALTH_EPS,
    ANGULAR_VELOCITY,
    ATTITUDE,
    COOLDOWN,
    HEALTH,
    LOCAL_LOG_INDEX,
    POSITION_X,
    POWER,
    SHIELD_DELAY,
    VELOCITY_X,
    predicted_means,
    predicted_uncertainty,
)


def means_to_observation(
    means: torch.Tensor,
    prev_obs: YemongObservation,
    action: torch.Tensor,
    num_ships: int,
    index_log_scale: float,
    observer_team: int = 0,
    enemy_action_logits: torch.Tensor | None = None,
    uncertainty: torch.Tensor | None = None,
) -> YemongObservation:
    """Write physical ship means into an observation, retaining the map tokens.

    The imagined counterpart of legal-view composition: the same eleven physical
    quantities, written into the same channels, with the same normalization the
    environment's builder applies. ``uncertainty``, the ``(B, N, 14)`` spreads
    that came with the means, is written to the belief channel when given, so
    the next forward reads the smoothed code the forecast stated.

    Bullets are intentionally absent: next-state prediction does not model them,
    so an imagined rollout is blind to fire in flight.
    """
    attitude = means[..., ATTITUDE : ATTITUDE + 1]
    alive = torch.where(
        prev_obs[ObsKey.GAME_MODE][:, num_ships, GameMode.FRONTLINE : GameMode.FRONTLINE + 1] > 0,
        prev_obs.alive[:, :num_ships],
        means[..., HEALTH] > ALIVE_HEALTH_EPS,
    )
    ship_values = {
        ObsKey.POS: means[..., POSITION_X : POSITION_X + 2],
        ObsKey.VEL: means[..., VELOCITY_X : VELOCITY_X + 2],
        ObsKey.ATT: torch.cat([torch.cos(attitude), torch.sin(attitude)], dim=-1),
        ObsKey.ANG_VEL: means[..., ANGULAR_VELOCITY : ANGULAR_VELOCITY + 1],
        ObsKey.HEALTH: means[..., HEALTH : HEALTH + 1],
        ObsKey.SHIELD_DELAY: means[..., SHIELD_DELAY : SHIELD_DELAY + 1],
        ObsKey.POWER: means[..., POWER : POWER + 1],
        ObsKey.COOLDOWN: means[..., COOLDOWN : COOLDOWN + 1],
        ObsKey.LOCAL_LOG_INDEX: means[..., LOCAL_LOG_INDEX : LOCAL_LOG_INDEX + 1] / index_log_scale,
        ObsKey.ALIVE: alive,
    }
    if uncertainty is not None:
        ship_values[ObsKey.BELIEF_UNCERTAINTY] = uncertainty
    data = {key: value.clone() for key, value in prev_obs.items()}
    if uncertainty is not None and ObsKey.BELIEF_UNCERTAINTY not in data:
        data[ObsKey.BELIEF_UNCERTAINTY] = torch.zeros(
            (*prev_obs[ObsKey.POS].shape[:-1], uncertainty.shape[-1]),
            device=uncertainty.device,
            dtype=uncertainty.dtype,
        )
    for key, values in ship_values.items():
        data[key] = torch.cat([values, prev_obs[key][:, num_ships:]], dim=1)
    enemy_probabilities = (
        torch.full_like(data[ObsKey.PREVIOUS_ACTION][:, :num_ships], 1.0 / NUM_JOINT_ACTIONS)
        if enemy_action_logits is None
        else enemy_action_logits.float().softmax(-1)
    )
    write_pending_action_view(
        data[ObsKey.PREVIOUS_ACTION][:, :num_ships],
        action,
        prev_obs[ObsKey.TEAM_ID][:, :num_ships],
        observer_team,
        torch.zeros_like(prev_obs[ObsKey.TEAM_ID][:, :num_ships], dtype=torch.bool),
        belief_action=enemy_probabilities,
    )
    return YemongObservation(data=data)


def imagine_trajectory(
    agent: ResolvedAgent,
    observation: YemongObservation,
    n_steps: int,
    num_ships: int,
    device,
    observer_team: int = 0,
    index_log_scale: float | None = None,
) -> list[torch.Tensor]:
    """Roll a policy's prediction head forward without mutating live hidden state.

    Each step's forecast, decoded to moments, becomes the next step's input:
    means and spreads both, so an imagined ship blurs as the head says it should.

    Returns one ``(B, num_ships, 4)`` *pose* per imagined step -- world x, world
    y, and the Cartesian heading ``(cos, sin)`` -- rather than the raw prediction
    vectors, because the caller is a renderer.

    ``index_log_scale`` comes from the ship configuration and is required
    whenever ``n_steps`` is positive.
    """
    if agent.kind != "policy" or agent.hidden is None or n_steps <= 0:
        return []
    if index_log_scale is None:
        raise ValueError("imagining a trajectory needs the observation's log-index scale")

    hidden = agent.hidden.clone()
    imagined = YemongObservation(data={key: value.clone() for key, value in observation.items()})

    poses: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n_steps):
            action, _, _, prediction, enemy_logits, hidden = agent.agent.get_action_and_value(
                imagined, hidden, return_enemy_action=True
            )
            imagined = means_to_observation(
                predicted_means(prediction.float()),
                imagined,
                action,
                num_ships,
                index_log_scale,
                observer_team,
                enemy_action_logits=enemy_logits,
                uncertainty=predicted_uncertainty(prediction.float()),
            )
            poses.append(
                torch.cat(
                    [
                        imagined[ObsKey.POS][:, :num_ships],
                        imagined[ObsKey.ATT][:, :num_ships],
                    ],
                    dim=-1,
                )
            )
    return poses
