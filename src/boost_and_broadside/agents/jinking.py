import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import PowerActions, TurnActions, ShootActions
from boost_and_broadside.env.state import TensorState

_JINK_HALF_PERIOD = 20  # steps per direction (~0.33s at 60Hz)


class JinkingAgent:
    """Alternates sharp left/right turns on a fixed timer. Pure chaos baseline."""

    def __init__(self, ship_config: ShipConfig):
        self.ship_config = ship_config

    def get_actions(self, state: TensorState) -> torch.Tensor:
        B, N = state.ship_pos.shape
        device = state.device

        # step_count is (B,); broadcast to (B, N)
        phase = (state.step_count // _JINK_HALF_PERIOD) % 2  # (B,) — 0 or 1
        phase = phase.unsqueeze(1).expand(B, N)               # (B, N)

        turn = torch.where(
            phase == 0,
            torch.tensor(TurnActions.SHARP_LEFT,  device=device),
            torch.tensor(TurnActions.SHARP_RIGHT, device=device),
        ).int()

        power = torch.full((B, N), PowerActions.BOOST,    dtype=torch.int32, device=device)
        shoot = torch.full((B, N), ShootActions.NO_SHOOT, dtype=torch.int32, device=device)

        alive = state.ship_alive
        coast    = torch.tensor(PowerActions.COAST,      device=device)
        straight = torch.tensor(TurnActions.GO_STRAIGHT, device=device)
        power = torch.where(alive, power, coast)
        turn  = torch.where(alive, turn,  straight)

        return torch.stack([power, turn, shoot], dim=-1)
