"""Shared next-state decoding and autoregressive imagination harness."""

import torch

from boost_and_broadside.env.observation import ObsKey, YemongObservation
from boost_and_broadside.evaluation.agents import ResolvedAgent

ALIVE_HEALTH_EPS = 1.0


def decode_targets_to_observation(
    targets: torch.Tensor,
    prev_obs: YemongObservation,
    action: torch.Tensor,
    num_ships: int,
    coordinator,
) -> YemongObservation:
    """Decode coordinator targets and retain non-predicted field tokens.

    Bullets are intentionally absent: next-state prediction does not model them,
    so an imagined rollout is blind to fire in flight.
    """
    raw = coordinator.decode_targets(targets)
    pos = torch.cat([raw["position_x"], raw["position_y"]], dim=-1)
    alive = raw["health"].squeeze(-1) > ALIVE_HEALTH_EPS
    ship_values = {
        ObsKey.POS: pos,
        ObsKey.VEL: raw["velocity"],
        ObsKey.ATT: raw["attitude"],
        ObsKey.ANG_VEL: raw["angular_velocity"],
        ObsKey.HEALTH: raw["health"],
        ObsKey.POWER: raw["power"],
        ObsKey.COOLDOWN: raw["cooldown"],
        ObsKey.LOCAL_LOG_INDEX: raw["local_log_index"],
        ObsKey.ALIVE: alive,
        ObsKey.PREVIOUS_ACTION: action,
    }
    data = {key: value.clone() for key, value in prev_obs.items()}
    for key, values in ship_values.items():
        data[key] = torch.cat([values, prev_obs[key][:, num_ships:]], dim=1)
    return YemongObservation(data=data)


def imagine_trajectory(
    agent: ResolvedAgent,
    observation: YemongObservation,
    n_steps: int,
    num_ships: int,
    device,
) -> list[torch.Tensor]:
    """Roll a policy's prediction head forward without mutating live hidden state."""
    if agent.kind != "policy" or agent.hidden is None or n_steps <= 0:
        return []

    coordinator = agent.agent.coordinator
    label_scale = coordinator.label_scale_vector(device)
    hidden = agent.hidden.clone()
    imagined = YemongObservation(data={key: value.clone() for key, value in observation.items()})
    ship_targets = coordinator.get_target_vector(imagined)[:, :num_ships]

    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n_steps):
            action, _, _, scaled_prediction, hidden = agent.agent.get_action_and_value(
                imagined, hidden, return_state_prediction=True
            )
            predictions.append(scaled_prediction / label_scale)
            ship_targets = coordinator.apply_scaled_predictions(ship_targets, scaled_prediction)
            imagined = decode_targets_to_observation(
                ship_targets, imagined, action, num_ships, coordinator
            )
            ship_targets = coordinator.get_target_vector(imagined)[:, :num_ships]
    return predictions
