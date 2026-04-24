import math
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import PowerActions, TurnActions, ShootActions
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.agents.scripted_utils import compute_team_target_bearings, predict_interception

_5DEG  = math.radians(5)
_15DEG = math.radians(15)
_1DEG  = math.radians(1)


class TeamJousterAgent:
    """All ships on a team charge the enemy closest to their team's center of mass."""

    def __init__(self, ship_config: ShipConfig):
        self.ship_config = ship_config

    def get_actions(self, state: TensorState) -> torch.Tensor:
        B, N = state.ship_pos.shape
        device = state.device

        _, team_dist, team_target_idx, has_target = compute_team_target_bearings(state, self.ship_config)
        dir_pred = predict_interception(state, self.ship_config, team_target_idx, team_dist)
        active = state.ship_alive & has_target

        att = state.ship_attitude
        rel_angle = torch.angle(dir_pred * torch.conj(att))
        abs_angle = rel_angle.abs()

        # Turn hard toward lead-corrected intercept of team target
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

        # Shoot when aligned to the angular size of the team target
        target_angular_size = 2.0 * torch.atan(
            self.ship_config.collision_radius / (team_dist + 1e-8)
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
