"""Fix K: collect_massive partial reset only updates obs for reset envs."""
import pytest
import torch


def apply_fix_k(obs, over_limit, fresh_obs):
    """Replicates the fixed obs update logic from collect_massive.run_collection()."""
    for k in obs:
        obs[k][over_limit] = fresh_obs[k][over_limit]
    return obs


def apply_old_behavior(obs, over_limit, fresh_obs):
    """Old (buggy) behavior: replaces entire obs dict."""
    obs = fresh_obs  # full replacement — loses non-reset env data
    return obs


def test_fix_k_non_reset_envs_unaffected():
    """Non-reset envs should keep their original obs after partial reset."""
    B, N = 8, 4
    obs = {
        "health": torch.arange(B * N, dtype=torch.float).reshape(B, N),
        "position": torch.randn(B, N, 2)
    }
    fresh_obs = {
        "health": torch.zeros(B, N),
        "position": torch.zeros(B, N, 2)
    }
    over_limit = torch.zeros(B, dtype=torch.bool)
    over_limit[[0, 2, 5]] = True  # Only envs 0, 2, 5 reset

    original_non_reset_health = obs["health"][[1, 3, 4, 6, 7]].clone()

    obs = apply_fix_k(obs, over_limit, fresh_obs)

    # Non-reset envs should be unchanged
    assert torch.allclose(obs["health"][[1, 3, 4, 6, 7]], original_non_reset_health), \
        "Non-reset envs should keep their original obs"


def test_fix_k_reset_envs_get_fresh_obs():
    """Reset envs should get the fresh obs values."""
    B, N = 6, 4
    obs = {
        "health": torch.ones(B, N) * 99.0,
    }
    fresh_obs = {
        "health": torch.zeros(B, N),
    }
    over_limit = torch.tensor([True, False, True, False, False, True])

    obs = apply_fix_k(obs, over_limit, fresh_obs)

    reset_indices = [0, 2, 5]
    for idx in reset_indices:
        assert (obs["health"][idx] == 0.0).all(), \
            f"Env {idx} (reset) should have fresh obs"


def test_fix_k_old_behavior_corrupts_active_envs():
    """Old behavior replaces entire obs dict, losing non-reset env data."""
    B, N = 4, 4
    sentinel = 42.0
    obs = {
        "health": torch.full((B, N), sentinel),
    }
    fresh_obs = {
        "health": torch.zeros(B, N),
    }
    over_limit = torch.tensor([True, False, False, False])  # Only env 0 resets

    obs_old = apply_old_behavior(obs, over_limit, fresh_obs)

    # Old behavior: non-reset envs lost their data
    assert (obs_old["health"][1] == 0.0).all(), \
        "Old behavior should corrupt non-reset envs (demonstrating the bug)"


def test_fix_k_handles_multiple_keys():
    """Fix K should update all obs keys for reset envs."""
    B, N = 4, 4
    obs = {
        "health": torch.ones(B, N) * 10.0,
        "power": torch.ones(B, N) * 20.0,
        "state": torch.ones(B, N, 5) * 30.0,
    }
    fresh = {
        "health": torch.zeros(B, N),
        "power": torch.zeros(B, N),
        "state": torch.zeros(B, N, 5),
    }
    over_limit = torch.tensor([False, True, False, True])

    obs = apply_fix_k(obs, over_limit, fresh)

    for k in ["health", "power", "state"]:
        assert (obs[k][1] == 0.0).all(), f"Key '{k}' env 1 should be fresh"
        assert (obs[k][3] == 0.0).all(), f"Key '{k}' env 3 should be fresh"
        # Non-reset envs preserved
        assert (obs["health"][0] == 10.0).all()
        assert (obs["power"][2] == 20.0).all()
