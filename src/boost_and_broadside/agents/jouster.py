import math
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import PowerActions, TurnActions, ShootActions
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.agents.scripted_utils import select_targets, predict_interception

_5DEG  = math.radians(5)
_15DEG = math.radians(15)
_1DEG  = math.radians(1)


class JousterAgent:
    """Charge at the nearest enemy using lead-corrected aim; boost when power allows; fire when aligned."""

    def __init__(self, ship_config: ShipConfig):
        self.ship_config = ship_config

    def get_actions(self, state: TensorState) -> torch.Tensor:
        B, N = state.ship_pos.shape
        device = state.device

        closest_dist, target_idx, has_target, _ = select_targets(state, self.ship_config)
        dir_pred = predict_interception(state, self.ship_config, target_idx, closest_dist)
        active = state.ship_alive & has_target

        att = state.ship_attitude
        rel_angle = torch.angle(dir_pred * torch.conj(att))
        abs_angle = rel_angle.abs()

        # Turn hard toward intercept point
        turn = torch.full((B, N), TurnActions.GO_STRAIGHT, dtype=torch.int32, device=device)
        normal = (abs_angle >= _5DEG) & (abs_angle < _15DEG)
        sharp  = abs_angle >= _15DEG
        turn = torch.where(normal & (rel_angle > 0), torch.tensor(TurnActions.TURN_RIGHT,  device=device), turn)
        turn = torch.where(normal & (rel_angle < 0), torch.tensor(TurnActions.TURN_LEFT,   device=device), turn)
        turn = torch.where(sharp  & (rel_angle > 0), torch.tensor(TurnActions.SHARP_RIGHT, device=device), turn)
        turn = torch.where(sharp  & (rel_angle < 0), torch.tensor(TurnActions.SHARP_LEFT,  device=device), turn)

        # Boost while power reserves allow shooting; coast otherwise
        power_ratio = state.ship_power / self.ship_config.max_power
        power = torch.where(
            power_ratio >= 0.5,
            torch.tensor(PowerActions.BOOST, device=device),
            torch.tensor(PowerActions.COAST, device=device),
        )

        # Shoot when heading is within the angular size of the target
        target_angular_size = 2.0 * torch.atan(
            self.ship_config.collision_radius / (closest_dist + 1e-8)
        )
        shoot_threshold = target_angular_size.clamp(min=_1DEG)
        shoot = torch.where(
            abs_angle <= shoot_threshold,
            torch.tensor(ShootActions.SHOOT,    device=device),
            torch.tensor(ShootActions.NO_SHOOT, device=device),
        )

        coast    = torch.tensor(PowerActions.COAST,      device=device)
        straight = torch.tensor(TurnActions.GO_STRAIGHT, device=device)
        no_shoot = torch.tensor(ShootActions.NO_SHOOT,   device=device)
        power = torch.where(active, power, coast)
        turn  = torch.where(active, turn,  straight)
        shoot = torch.where(active, shoot, no_shoot)

        return torch.stack([power, turn, shoot], dim=-1)
