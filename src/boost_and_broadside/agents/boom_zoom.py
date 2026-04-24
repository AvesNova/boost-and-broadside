import math
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import PowerActions, TurnActions, ShootActions
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.agents.scripted_utils import select_targets

_1DEG   = math.radians(1)
_15DEG  = math.radians(15)
_MAX_SHOOT_RANGE = 500.0


class BoomZoomAgent:
    """Maintain high speed; only turn toward enemies within 15° of the flight path; shoot on the pass."""

    def __init__(self, ship_config: ShipConfig):
        self.ship_config = ship_config

    def get_actions(self, state: TensorState) -> torch.Tensor:
        B, N = state.ship_pos.shape
        device = state.device

        closest_dist, target_idx, has_target, bearing = select_targets(state, self.ship_config)
        active = state.ship_alive & has_target

        att = state.ship_attitude
        rel_angle = torch.angle(bearing * torch.conj(att))  # signed angle from heading to enemy
        abs_angle = rel_angle.abs()
        in_cone = abs_angle <= _15DEG

        # Turn: only nudge when enemy is inside the 15° cone
        turn = torch.full((B, N), TurnActions.GO_STRAIGHT, dtype=torch.int32, device=device)
        turn = torch.where(in_cone & (rel_angle > 0), torch.tensor(TurnActions.TURN_RIGHT, device=device), turn)
        turn = torch.where(in_cone & (rel_angle < 0), torch.tensor(TurnActions.TURN_LEFT,  device=device), turn)

        # Power: boost when power reserves are sufficient
        power_ratio = state.ship_power / self.ship_config.max_power
        power = torch.where(
            power_ratio >= 0.5,
            torch.tensor(PowerActions.BOOST, device=device),
            torch.tensor(PowerActions.COAST, device=device),
        )

        # Shoot: fire on the pass when aligned and in range
        target_angular_size = 2.0 * torch.atan(
            self.ship_config.collision_radius / (closest_dist + 1e-8)
        )
        shoot_threshold = target_angular_size.clamp(min=_1DEG)
        in_range = closest_dist <= _MAX_SHOOT_RANGE
        aligned = abs_angle <= shoot_threshold
        shoot = torch.where(
            aligned & in_range,
            torch.tensor(ShootActions.SHOOT,    device=device),
            torch.tensor(ShootActions.NO_SHOOT, device=device),
        )

        # Inactive ships: coast / straight / no shoot
        coast    = torch.tensor(PowerActions.COAST,       device=device)
        straight = torch.tensor(TurnActions.GO_STRAIGHT,  device=device)
        no_shoot = torch.tensor(ShootActions.NO_SHOOT,    device=device)
        power = torch.where(active, power, coast)
        turn  = torch.where(active, turn,  straight)
        shoot = torch.where(active, shoot, no_shoot)

        return torch.stack([power, turn, shoot], dim=-1)
