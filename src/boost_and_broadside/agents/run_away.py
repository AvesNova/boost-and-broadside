import math
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import PowerActions, TurnActions, ShootActions
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.agents.scripted_utils import select_targets

_5DEG = math.radians(5)
_15DEG = math.radians(15)


class RunAwayAgent:
    """Turn away from the nearest enemy and boost."""

    def __init__(self, ship_config: ShipConfig):
        self.ship_config = ship_config

    def get_actions(self, state: TensorState) -> torch.Tensor:
        B, N = state.ship_pos.shape
        device = state.device

        closest_dist, target_idx, has_target, bearing = select_targets(state, self.ship_config)
        active = state.ship_alive & has_target

        att = state.ship_attitude
        dir_away = -bearing
        rel_angle = torch.angle(dir_away * torch.conj(att))  # (B, N)
        abs_angle = rel_angle.abs()

        # Turn: aim heading away from enemy
        turn = torch.full((B, N), TurnActions.GO_STRAIGHT, dtype=torch.int32, device=device)
        normal = (abs_angle >= _5DEG) & (abs_angle < _15DEG)
        sharp = abs_angle >= _15DEG
        turn = torch.where(normal & (rel_angle > 0), torch.tensor(TurnActions.TURN_RIGHT, device=device), turn)
        turn = torch.where(normal & (rel_angle < 0), torch.tensor(TurnActions.TURN_LEFT,  device=device), turn)
        turn = torch.where(sharp  & (rel_angle > 0), torch.tensor(TurnActions.SHARP_RIGHT, device=device), turn)
        turn = torch.where(sharp  & (rel_angle < 0), torch.tensor(TurnActions.SHARP_LEFT,  device=device), turn)

        power = torch.full((B, N), PowerActions.BOOST, dtype=torch.int32, device=device)
        shoot = torch.full((B, N), ShootActions.NO_SHOOT, dtype=torch.int32, device=device)

        # Inactive ships: coast / straight / no shoot
        coast = torch.tensor(PowerActions.COAST, device=device)
        straight = torch.tensor(TurnActions.GO_STRAIGHT, device=device)
        power = torch.where(active, power, coast)
        turn  = torch.where(active, turn,  straight)

        return torch.stack([power, turn, shoot], dim=-1)
