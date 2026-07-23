import math

import torch

from boost_and_broadside.agents.scripted_utils import select_targets, turn_toward
from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import PowerActions, ShootActions, TurnActions
from boost_and_broadside.env.state import TensorState

_15DEG = math.radians(15)


class AbreastAgent:
    """Stay perpendicular to the nearest enemy to ruin their shot.

    Boost when the enemy is aimed at us.
    """

    def __init__(self, ship_config: ShipConfig):
        self.ship_config = ship_config

    def get_actions(self, state: TensorState) -> torch.Tensor:
        B, N = state.ship_pos.shape
        device = state.device

        closest_dist, target_idx, has_target, bearing = select_targets(state, self.ship_config)
        active = state.ship_alive & has_target

        att = state.ship_attitude

        # Two perpendicular headings relative to bearing-to-enemy
        # CW  rotation of bearing by -90°: multiply by -i → (imag, -real)
        # CCW rotation of bearing by +90°: multiply by  i → (-imag, real)
        dir_cw = torch.complex(bearing.imag, -bearing.real)
        dir_ccw = torch.complex(-bearing.imag, bearing.real)

        rel_cw = torch.angle(dir_cw * torch.conj(att))
        rel_ccw = torch.angle(dir_ccw * torch.conj(att))

        # Pick whichever perpendicular requires the smaller turn
        use_cw = rel_cw.abs() <= rel_ccw.abs()
        target_rel = torch.where(use_cw, rel_cw, rel_ccw)

        # Turn toward the chosen perpendicular heading
        turn = turn_toward(target_rel)

        # Power: boost when enemy nose is within 15° of us (we are in their sights)
        enemy_att = torch.gather(state.ship_attitude, 1, target_idx)  # (B, N) complex
        # bearing is from me to enemy; -bearing is from enemy toward me
        enemy_rel = torch.angle(-bearing * torch.conj(enemy_att))
        power = torch.where(
            enemy_rel.abs() <= _15DEG,
            torch.tensor(PowerActions.BOOST, device=device),
            torch.tensor(PowerActions.COAST, device=device),
        )

        shoot = torch.full((B, N), ShootActions.NO_SHOOT, dtype=torch.int32, device=device)

        # Inactive ships: coast / straight / no shoot
        coast = torch.tensor(PowerActions.COAST, device=device)
        straight = torch.tensor(TurnActions.GO_STRAIGHT, device=device)
        power = torch.where(active, power, coast)
        turn = torch.where(active, turn, straight)

        return torch.stack([power, turn, shoot], dim=-1)
