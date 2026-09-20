"""CPU-only guards for the isolated CUDA graph tick harness."""

import pytest
import torch

from benchmarks.cuda_graph_tick import RNGParityUnsupported, check_rng_support
from boost_and_broadside.env.cuda_graph import CapturedTick


def test_rng_guard_rejects_shooting_with_nonzero_spread():
    env = type("Env", (), {"ship_config": type("Config", (), {"bullet_spread": 1.0})()})()
    action = torch.zeros((1, 100, 3), dtype=torch.int32)
    action[..., 2] = 1
    with pytest.raises(RNGParityUnsupported, match="torch.randn_like"):
        check_rng_support(env, action)


def test_rng_guard_allows_fixed_no_shoot_dispatch_measurement():
    env = type("Env", (), {"ship_config": type("Config", (), {"bullet_spread": 1.0})()})()
    check_rng_support(env, torch.zeros((1, 100, 3), dtype=torch.int32))


def test_captured_tick_rejects_cpu_environment():
    env = type(
        "Env",
        (),
        {
            "device": torch.device("cpu"),
            "state": type("State", (), {"prev_action": torch.zeros((1, 100, 3))})(),
        },
    )()
    with pytest.raises(ValueError, match="CUDA TensorEnv"):
        CapturedTick(env, torch.zeros((1, 100, 3), dtype=torch.int32))
