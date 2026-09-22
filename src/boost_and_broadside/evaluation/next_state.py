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
    alive = torch.where(
        prev_obs[ObsKey.GAME_MODE][:, -1, 0:1] > 0,
        prev_obs.alive[:, :num_ships],
        raw["health"].squeeze(-1) > ALIVE_HEALTH_EPS,
    )
    ship_values = {
        ObsKey.POS: pos,
        ObsKey.VEL: raw["velocity"],
        ObsKey.ATT: raw["attitude"],
        ObsKey.ANG_VEL: raw["angular_velocity"],
        ObsKey.HEALTH: raw["health"],
        ObsKey.SHIELD_DELAY: raw["shield_delay"].clamp_min(0),
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
    """Roll a policy's prediction head forward without mutating live hidden state.

    Returns one ``(B, num_ships, 4)`` *pose* per imagined step -- world x, world
    y, and the Cartesian heading ``(cos, sin)`` -- rather than the raw prediction
    vectors.

    Poses rather than predictions because the caller is a renderer, and a
    prediction vector can only be read by something that knows the feature
    layout. That layout is not stable: position is ten Fourier harmonics whose
    count follows the world size, so a fixed channel index into it is wrong on
    every map but one. The decode already happens here, one line further down,
    to build the next step's input -- so handing back what it produced costs
    nothing and leaves the prediction layout entirely inside this module.
    """
    if agent.kind != "policy" or agent.hidden is None or n_steps <= 0:
        return []

    coordinator = agent.agent.coordinator
    hidden = agent.hidden.clone()
    imagined = YemongObservation(data={key: value.clone() for key, value in observation.items()})
    ship_targets = coordinator.get_target_vector(imagined)[:, :num_ships]

    poses: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n_steps):
            action, _, _, scaled_prediction, hidden = agent.agent.get_action_and_value(
                imagined, hidden
            )
            ship_targets = coordinator.apply_scaled_predictions(ship_targets, scaled_prediction)
            imagined = decode_targets_to_observation(
                ship_targets, imagined, action, num_ships, coordinator
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
            ship_targets = coordinator.get_target_vector(imagined)[:, :num_ships]
    return poses
