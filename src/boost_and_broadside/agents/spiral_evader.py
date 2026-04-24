import math
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import PowerActions, TurnActions, ShootActions
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.agents.scripted_utils import select_targets

_5DEG  = math.radians(5)
_15DEG = math.radians(15)

# Unit complex for ±45° rotation
_COS45 = math.cos(math.radians(45))
_SIN45 = math.sin(math.radians(45))


class SpiralEvaderAgent:
    """Boost away from the nearest enemy at a 45° offset, creating a curving escape."""

    def __init__(self, ship_config: ShipConfig):
        self.ship_config = ship_config

    def get_actions(self, state: TensorState) -> torch.Tensor:
        B, N = state.ship_pos.shape
        device = state.device

        closest_dist, target_idx, has_target, bearing = select_targets(state, self.ship_config)
        active = state.ship_alive & has_target

        att = state.ship_attitude
        away = -bearing  # unit vector directly away from enemy

        # Rotate away-bearing by ±45°; pick whichever requires less turn from current heading
        rot_ccw = torch.complex(
            away.real * _COS45 - away.imag * _SIN45,
            away.real * _SIN45 + away.imag * _COS45,
        )
        rot_cw = torch.complex(
            away.real * _COS45 + away.imag * _SIN45,
            -away.real * _SIN45 + away.imag * _COS45,
        )

        rel_ccw = torch.angle(rot_ccw * torch.conj(att))
        rel_cw  = torch.angle(rot_cw  * torch.conj(att))

        use_ccw = rel_ccw.abs() <= rel_cw.abs()
        target_rel = torch.where(use_ccw, rel_ccw, rel_cw)
        abs_target = target_rel.abs()

        # Turn toward the chosen spiral heading
        turn = torch.full((B, N), TurnActions.GO_STRAIGHT, dtype=torch.int32, device=device)
        normal = (abs_target >= _5DEG) & (abs_target < _15DEG)
        sharp  = abs_target >= _15DEG
        turn = torch.where(normal & (target_rel > 0), torch.tensor(TurnActions.TURN_RIGHT,  device=device), turn)
        turn = torch.where(normal & (target_rel < 0), torch.tensor(TurnActions.TURN_LEFT,   device=device), turn)
        turn = torch.where(sharp  & (target_rel > 0), torch.tensor(TurnActions.SHARP_RIGHT, device=device), turn)
        turn = torch.where(sharp  & (target_rel < 0), torch.tensor(TurnActions.SHARP_LEFT,  device=device), turn)

        power = torch.full((B, N), PowerActions.BOOST, dtype=torch.int32, device=device)
        shoot = torch.full((B, N), ShootActions.NO_SHOOT, dtype=torch.int32, device=device)

        coast    = torch.tensor(PowerActions.COAST,      device=device)
        straight = torch.tensor(TurnActions.GO_STRAIGHT, device=device)
        power = torch.where(active, power, coast)
        turn  = torch.where(active, turn,  straight)

        return torch.stack([power, turn, shoot], dim=-1)
