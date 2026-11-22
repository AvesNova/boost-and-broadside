"""Invariants of the policy model. Each test names the failure it catches."""

from types import SimpleNamespace

import pytest
import torch

from lean.model import ModelConfig, Policy, decode_value, hl_gauss_targets, value_bin_centres

SHIP_FEATURES, GLOBAL_FEATURES = 27, 3


def make_policy(**overrides) -> Policy:
    torch.manual_seed(0)
    policy = Policy(ModelConfig(**overrides), SHIP_FEATURES, GLOBAL_FEATURES)
    # The heads start at (near) zero; perturb them so the tests see the trunk's output.
    with torch.no_grad():
        policy.policy_head.weight.normal_(std=0.1)
        policy.value_head.weight.normal_(std=0.1)
    return policy.eval()


def make_obs(lead: tuple[int, ...], slots: int, seed: int = 1, live_fraction: float = 0.8):
    generator = torch.Generator().manual_seed(seed)
    key_mask = torch.rand(*lead, slots, generator=generator) < live_fraction
    return SimpleNamespace(
        ship=torch.randn(*lead, slots, SHIP_FEATURES, generator=generator),
        pos=(torch.rand(*lead, slots, 2, generator=generator) - 0.5) * 6000.0,
        heading=(torch.rand(*lead, slots, generator=generator) - 0.5) * 6.28,
        key_mask=key_mask,
        own=key_mask.clone(),
        globals=torch.randn(*lead, GLOBAL_FEATURES, generator=generator),
    )


def index_obs(obs, index):
    return SimpleNamespace(**{name: value[index] for name, value in vars(obs).items()})


def select_slots(obs, order):
    """Reorder (or subset) the ship slots of a single-tick observation."""
    fields = {name: value[:, order] for name, value in vars(obs).items() if name != "globals"}
    return SimpleNamespace(globals=obs.globals, **fields)


@pytest.mark.parametrize("rotate_values", [False, True])
def test_sequence_equals_repeated_steps(rotate_values):
    """Catches a learner forward that differs from the rollout forward: carry, reset, masks."""
    ticks, envs, slots = 6, 3, 5
    policy = make_policy(rotate_values=rotate_values)
    obs = make_obs((ticks, envs), slots)
    reset = torch.zeros(ticks, envs, dtype=torch.bool)
    reset[0, 1] = reset[3, 0] = reset[4, 2] = True
    offset = torch.randn(envs, 2) * 1000.0
    carry = torch.randn(policy.initial_carry(envs, slots, "cpu").shape)
    with torch.no_grad():
        logits, values, next_state, final = policy.sequence(obs, carry, reset, offset)
        for tick in range(ticks):
            step_logits, step_values, carry = policy.step(index_obs(obs, tick), carry, offset, reset[tick])
            live = obs.key_mask[tick]
            assert torch.allclose(step_logits[live], logits[tick][live], atol=1e-5)
            assert torch.allclose(step_values[live], values[tick][live], atol=1e-5)
    assert torch.allclose(carry, final, atol=1e-5)
    assert next_state.shape == (ticks, envs, slots, policy.config.next_state_size)


def test_masked_rows_keep_a_zero_carry():
    """Catches state leaking through a dead or absent slot."""
    policy = make_policy()
    obs = make_obs((4,), 6)
    with torch.no_grad():
        _, _, carry = policy.step(obs, torch.randn(policy.initial_carry(4, 6, "cpu").shape))
    assert (carry[:, :, 1:][:, ~obs.key_mask] == 0).all()
    assert (carry[:, :, 1:][:, obs.key_mask] != 0).any()


@pytest.mark.parametrize("positionless_rotary", ["zero", "identity"])
def test_padding_and_permutation_leave_real_ships_unchanged(positionless_rotary):
    """Catches any dependence on slot index, slot count or the content of masked rows."""
    envs, real, padded = 3, 7, 12
    policy = make_policy(positionless_rotary=positionless_rotary)
    obs = make_obs((envs,), padded, live_fraction=2.0)
    obs.key_mask[:, real:] = False
    small = select_slots(obs, torch.arange(real))
    order = torch.randperm(padded, generator=torch.Generator().manual_seed(3))
    shuffled = select_slots(obs, order)
    with torch.no_grad():
        base_logits, base_values, base_carry = policy.step(small, policy.initial_carry(envs, real, "cpu"))
        pad_logits, pad_values, pad_carry = policy.step(obs, policy.initial_carry(envs, padded, "cpu"))
        mix_logits, mix_values, mix_carry = policy.step(shuffled, policy.initial_carry(envs, padded, "cpu"))
    assert torch.allclose(pad_logits[:, :real], base_logits, atol=1e-5)
    assert torch.allclose(pad_values[:, :real], base_values, atol=1e-5)
    assert torch.allclose(pad_carry[:, :, : 1 + real], base_carry, atol=1e-5)
    live = shuffled.key_mask
    assert torch.allclose(mix_logits[live], pad_logits[:, order][live], atol=1e-5)
    assert torch.allclose(mix_values[live], pad_values[:, order][live], atol=1e-5)
    assert torch.allclose(mix_carry[:, :, 1:][:, live], pad_carry[:, :, 1:][:, order][:, live], atol=1e-5)
    assert torch.allclose(mix_carry[:, :, 0], pad_carry[:, :, 0], atol=1e-5)


def test_ship_scores_ignore_a_common_translation():
    """Catches absolute position entering attention (a positionless token acting as a ship
    at the origin). The token code is held fixed by an opposite position offset."""
    envs, slots = 2, 6
    policy = make_policy()
    obs = make_obs((envs,), slots)
    shift = torch.tensor([1234.5, -876.5])
    moved = SimpleNamespace(**{**vars(obs), "pos": obs.pos + shift})
    carry = policy.initial_carry(envs, slots, "cpu")
    with torch.no_grad():
        base_logits, _, _ = policy.step(obs, carry, torch.zeros(envs, 2))
        moved_logits, _, _ = policy.step(moved, carry, -shift.expand(envs, 2))
    live = obs.key_mask
    assert torch.allclose(base_logits[live], moved_logits[live], atol=2e-3)


def test_same_weights_run_at_any_ship_count():
    """Catches a parameter or buffer sized by the number of slots."""
    policy = make_policy()
    for slots in (10, 40, 100):
        obs = make_obs((2,), slots)
        with torch.no_grad():
            logits, values, carry = policy.step(obs, policy.initial_carry(2, slots, "cpu"))
        assert logits.shape == (2, slots, 42)
        assert values.shape == (2, slots, 4, 77)
        assert carry.shape == (2, 2, 1 + slots, 128)
        assert torch.isfinite(logits).all() and torch.isfinite(values).all()


def test_decay_initialisation_is_in_the_papers_range():
    """Catches a retention range below Griffin's a^c in [0.9, 0.999]."""
    policy = make_policy()
    for block in policy.blocks:
        retention = block.temporal.retention_at_open_gate()
        assert retention.min() >= 0.9 - 1e-4 and retention.max() <= 0.999 + 1e-4
        assert retention.max() - retention.min() > 0.05


def test_coarsest_period_and_ladder():
    """Catches a rotary ladder that is a power of two or too short for the arena."""
    periods = ModelConfig().position_periods
    assert len(periods) == 6 and abs(periods[0] - 64.0) < 1e-2 and periods[-1] > 16000.0
    ratio = periods[1] / periods[0]
    assert abs(ratio - round(ratio)) > 0.1


def test_hl_gauss_targets_sum_to_one_and_decode_to_the_value():
    """Catches a mis-normalised or shifted value target."""
    centres = value_bin_centres(torch.tensor([-2.0, -10.0]), torch.tensor([2.0, 30.0]), 77)
    width = centres[:, 1] - centres[:, 0]
    inside = torch.rand(500, 2, generator=torch.Generator().manual_seed(0))
    values = centres[:, 0] + 3.0 * width + inside * (centres[:, -1] - centres[:, 0] - 6.0 * width)
    targets = hl_gauss_targets(values, centres)
    assert targets.shape == (500, 2, 77)
    assert torch.allclose(targets.sum(-1), torch.ones(500, 2), atol=1e-5)
    assert (targets >= 0).all()
    decoded = decode_value(targets.clamp_min(1e-30).log(), centres)
    assert ((decoded - values).abs() < 1e-2 * width).all()
    outside = hl_gauss_targets(torch.tensor([100.0, -100.0]), centres)
    assert torch.allclose(outside.sum(-1), torch.ones(2), atol=1e-5)
