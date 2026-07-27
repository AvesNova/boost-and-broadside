"""Tests for watch-mode helpers in modes/agent_factory.py."""

import math

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import ObsKey, YemongObservation
from boost_and_broadside.modes.agent_factory import _decode_targets_to_obs
from boost_and_broadside.train.rl.features import build_standard_coordinator


def _make_prev_obs(B: int, N: int) -> YemongObservation:
    """Minimal ship-only observation with every key _decode_targets_to_obs touches."""
    return YemongObservation(
        data={
            ObsKey.POS: torch.zeros(B, N, 2),
            ObsKey.VEL: torch.zeros(B, N, 2),
            ObsKey.ATT: torch.zeros(B, N, 2),
            ObsKey.ANG_VEL: torch.zeros(B, N, 1),
            ObsKey.HEALTH: torch.zeros(B, N, 1),
            ObsKey.POWER: torch.zeros(B, N, 1),
            ObsKey.COOLDOWN: torch.zeros(B, N, 1),
            ObsKey.ALIVE: torch.ones(B, N, dtype=torch.bool),
            ObsKey.PREVIOUS_ACTION: torch.zeros(B, N, 3, dtype=torch.long),
        }
    )


class TestDecodeTargetsToObs:
    def test_position_decodes_each_axis_with_its_own_world_extent(self):
        """Regression (audit §1.4): pos_y must decode with world height, not width.

        Uses a rectangular world (W != H) so decoding y with W produces the
        wrong coordinate; encode (x, y) exactly as the feature pipeline does
        (Fourier(1, period) → (sin, cos)) and expect the round-trip identity.
        """
        ship_config = ShipConfig(world_size=(1024.0, 512.0))
        coordinator = build_standard_coordinator(ship_config)
        target_slices = coordinator.target_slices()
        W, H = ship_config.world_size
        x, y = 700.0, 300.0

        targets = torch.zeros(1, 1, coordinator.total_target_dimension)
        targets[0, 0, target_slices["position_x"]] = torch.tensor(
            [math.sin(2 * math.pi * x / W), math.cos(2 * math.pi * x / W)]
        )
        targets[0, 0, target_slices["position_y"]] = torch.tensor(
            [math.sin(2 * math.pi * y / H), math.cos(2 * math.pi * y / H)]
        )
        targets[0, 0, target_slices["attitude"]] = torch.tensor([0.0, 1.0])

        obs = _decode_targets_to_obs(
            targets,
            prev_obs=_make_prev_obs(B=1, N=1),
            action=torch.zeros(1, 1, 3, dtype=torch.long),
            N=1,
            coordinator=coordinator,
        )

        decoded = obs[ObsKey.POS][0, 0]
        assert torch.allclose(decoded, torch.tensor([x, y]), atol=1e-3)
