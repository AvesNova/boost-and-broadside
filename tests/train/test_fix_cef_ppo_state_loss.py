"""Fixes C, E, F: PPO auxiliary state loss improvements."""
import pytest
import torch
from boost_and_broadside.core.constants import StateFeature


# --- Pure logic helpers matching what ppo.py now does ---

def compute_valid_mask_and_delta(mb_obs_state, mb_next_obs_state, mb_dones_perm):
    """Replicate the fixed PPO loss mask + delta computation."""
    B, T, N, F = mb_next_obs_state.shape
    mb_curr_obs_perm = mb_obs_state
    mb_next_obs_perm = mb_next_obs_state
    mb_delta_obs = mb_next_obs_perm - mb_curr_obs_perm

    # Fix E: all terminal steps masked
    valid_mask = (1.0 - mb_dones_perm.float())  # (B, T)
    # Fix F: dead ships masked
    valid_mask = valid_mask.unsqueeze(-1).expand(B, T, N)
    alive_mask = (mb_next_obs_perm[..., StateFeature.HEALTH] > 0).float()
    valid_mask = valid_mask * alive_mask
    return mb_delta_obs, valid_mask


# --- Tests ---

def test_fix_c_delta_target():
    """delta obs = next_obs - curr_obs (not raw next obs)."""
    B, T, N, F = 2, 4, 4, 5
    curr = torch.ones(B, T, N, F) * 3.0
    nxt = torch.ones(B, T, N, F) * 5.0
    dones = torch.zeros(B, T, dtype=torch.long)

    delta, _ = compute_valid_mask_and_delta(curr, nxt, dones)

    expected = torch.ones(B, T, N, F) * 2.0  # 5 - 3
    assert torch.allclose(delta, expected), "Delta should be next - curr"


def test_fix_e_terminal_steps_masked():
    """All done steps should be zeroed in valid_mask."""
    B, T, N, F = 2, 6, 4, 5
    curr = torch.randn(B, T, N, F)
    nxt = torch.randn(B, T, N, F)
    # Make nxt health > 0 throughout so we isolate the done masking
    nxt[..., StateFeature.HEALTH] = 1.0

    dones = torch.zeros(B, T, dtype=torch.long)
    dones[:, 2] = 1  # step 2 is terminal for all envs
    dones[0, 4] = 1  # step 4 is also terminal for env 0

    _, valid_mask = compute_valid_mask_and_delta(curr, nxt, dones)

    # All entries at done steps should be 0
    assert valid_mask[:, 2].sum() == 0.0, "Step 2 (done) should be fully masked"
    assert valid_mask[0, 4].sum() == 0.0, "Step 4 env 0 (done) should be masked"
    # Non-terminal steps should be unmasked (all alive)
    assert valid_mask[0, 0].sum() > 0.0, "Non-terminal, alive ships should be unmasked"
    assert valid_mask[1, 5].sum() > 0.0, "Non-terminal, alive ships should be unmasked"


def test_fix_e_only_last_step_not_sufficient():
    """Old mask (only last step) would miss intermediate terminal steps."""
    B, T, N, F = 2, 6, 4, 5
    curr = torch.randn(B, T, N, F)
    nxt = torch.randn(B, T, N, F)
    nxt[..., StateFeature.HEALTH] = 1.0

    dones = torch.zeros(B, T, dtype=torch.long)
    dones[:, 2] = 1  # step 2 is terminal — old mask would miss this

    _, valid_mask = compute_valid_mask_and_delta(curr, nxt, dones)

    # The new mask correctly zeros step 2
    assert valid_mask[:, 2].sum() == 0.0, "Fix E must catch intermediate terminal steps"


def test_fix_f_dead_ships_masked():
    """Ships with health=0 in next_obs should be masked from state loss."""
    B, T, N, F = 2, 4, 4, 5
    curr = torch.randn(B, T, N, F)
    nxt = torch.randn(B, T, N, F)
    nxt[..., StateFeature.HEALTH] = 1.0  # all alive baseline

    # Kill ship index 0 at timestep 1 for env 0
    nxt[0, 1, 0, StateFeature.HEALTH] = 0.0

    dones = torch.zeros(B, T, dtype=torch.long)
    _, valid_mask = compute_valid_mask_and_delta(curr, nxt, dones)

    assert valid_mask[0, 1, 0] == 0.0, "Dead ship should be masked (health=0)"
    assert valid_mask[0, 1, 1] > 0.0, "Alive ship should not be masked"


def test_fix_f_no_alive_ships_step_is_zero():
    """If all ships dead in a step, that step's contribution should be zero."""
    B, T, N, F = 1, 4, 4, 5
    curr = torch.randn(B, T, N, F)
    nxt = torch.randn(B, T, N, F)
    nxt[..., StateFeature.HEALTH] = 1.0
    nxt[0, 2, :, StateFeature.HEALTH] = 0.0  # all ships dead at step 2

    dones = torch.zeros(B, T, dtype=torch.long)
    _, valid_mask = compute_valid_mask_and_delta(curr, nxt, dones)

    assert valid_mask[0, 2].sum() == 0.0, "All-dead step should have zero valid_mask"
