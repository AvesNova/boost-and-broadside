import math

import pytest
import torch

from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.train.rl.physical_deltas import (
    PHYSICAL_DELTA_NAMES,
    PhysicalShipSnapshot,
    normalize_physical_deltas,
    physical_ship_deltas,
)


def _snapshot(**overrides) -> PhysicalShipSnapshot:
    values = {
        "position": torch.tensor([[complex(99.0, 1.0)]]),
        "velocity": torch.tensor([[complex(2.0, -3.0)]]),
        "attitude": torch.polar(torch.ones(1, 1), torch.tensor([[math.pi - 0.1]])),
        "angular_velocity": torch.tensor([[0.2]]),
        "shield_delay": torch.tensor([[1.0]]),
        "health": torch.tensor([[80.0]]),
        "power": torch.tensor([[60.0]]),
        "cooldown": torch.tensor([[0.1]]),
        "local_index": torch.tensor([[2.0]]),
    }
    values.update(overrides)
    return PhysicalShipSnapshot(**values)


def test_physical_delta_definition_wraps_position_and_attitude() -> None:
    current = _snapshot()
    next_ = _snapshot(
        position=torch.tensor([[complex(1.0, 99.0)]]),
        velocity=torch.tensor([[complex(5.0, 1.0)]]),
        attitude=torch.polar(torch.ones(1, 1), torch.tensor([[-math.pi + 0.2]])),
        angular_velocity=torch.tensor([[-0.1]]),
        shield_delay=torch.tensor([[0.5]]),
        health=torch.tensor([[70.0]]),
        power=torch.tensor([[62.0]]),
        cooldown=torch.tensor([[0.0]]),
        local_index=torch.tensor([[1.0]]),
    )

    deltas = physical_ship_deltas(current, next_, (100.0, 100.0))[0, 0]

    assert PHYSICAL_DELTA_NAMES == (
        "position_x",
        "position_y",
        "velocity_x",
        "velocity_y",
        "attitude",
        "angular_velocity",
        "shield_delay",
        "health",
        "power",
        "cooldown",
        "local_log_index",
    )
    torch.testing.assert_close(
        deltas,
        torch.tensor([2.0, -2.0, 3.0, 4.0, 0.3, -0.3, -0.5, -10.0, 2.0, -0.1, -math.log(2)]),
    )


def test_snapshot_aliases_survive_tensor_env_reassignment(ship_config, env_config) -> None:
    env = TensorEnv(1, ship_config, env_config, "cpu")
    env.reset(seed=7)
    snapshot = PhysicalShipSnapshot.from_state(env.state)
    original = snapshot.position.clone()

    env.step(torch.zeros((1, env_config.num_ships, 3), dtype=torch.int32))

    torch.testing.assert_close(snapshot.position, original)
    assert snapshot.position.data_ptr() != env.state.ship_pos.data_ptr()


def test_normalization_preserves_exact_zero_and_rejects_bad_scales() -> None:
    deltas = torch.zeros(2, len(PHYSICAL_DELTA_NAMES))
    scales = torch.arange(1, len(PHYSICAL_DELTA_NAMES) + 1, dtype=torch.float32)
    normalized = normalize_physical_deltas(deltas, scales)

    assert torch.equal(normalized, deltas)
    with pytest.raises(ValueError, match="finite and positive"):
        normalize_physical_deltas(deltas, scales.masked_fill(scales == 3, 0))
