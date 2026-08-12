"""Tests for the AR measurement's scenario and stored rollout contract.

The report's numeric display helpers — toroidal unwrapping, center of mass,
error masking — moved to the publication renderer with the plots they serve and
are covered there. What remains here is the measurement itself: the one
canonical scenario, and the array layout every renderer reads.
"""

import numpy as np
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.config.defaults import MODEL_CONFIG, REWARDS
from boost_and_broadside.modes import ar_report
from boost_and_broadside.modes.ar_report import _ROLLOUT_FIELDS, _rollout_arrays


def test_canonical_ar_mode_owns_one_4v4_scenario(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(ar_report, "run_ar_report_mode", lambda **kwargs: captured.update(kwargs))

    ar_report.run_canonical_ar_report_mode(
        "scripted",
        "random",
        12,
        ShipConfig(),
        REWARDS,
        MODEL_CONFIG,
        "cpu",
    )
    assert captured["env_config"].num_ships == 8
    assert captured["env_config"].max_episode_steps == 12


def test_rollout_arrays_drop_the_batch_axis_and_keep_every_field() -> None:
    history = [
        {
            "pos": torch.zeros(1, 4, 2),
            "vel": torch.zeros(1, 4, 2),
            "att": torch.zeros(1, 4, 2),
            "ang_vel": torch.zeros(1, 4, 1),
            "health": torch.zeros(1, 4, 1),
            "power": torch.zeros(1, 4, 1),
            "cooldown": torch.zeros(1, 4, 1),
            "alive": torch.ones(1, 4, dtype=torch.bool),
            "alive_prob": torch.ones(1, 4),
        }
        for _ in range(3)
    ]

    arrays = _rollout_arrays("gt", history)

    assert set(arrays) == {f"gt_{field}" for field in _ROLLOUT_FIELDS}
    assert arrays["gt_pos"].shape == (3, 4, 2)
    assert arrays["gt_alive"].shape == (3, 4)
    assert all(array.dtype == np.float32 for array in arrays.values())
