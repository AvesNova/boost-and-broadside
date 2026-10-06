"""The categorical critic: value bins, outcome targets, and the return scale."""

import pytest
import torch

from boost_and_broadside.constants import (
    NUM_OUTCOME_CLASSES,
    OUTCOME_LOSS_INDEX,
    OUTCOME_TIE_INDEX,
    OUTCOME_UNRESOLVED_INDEX,
    OUTCOME_WIN_INDEX,
)
from boost_and_broadside.env.observation import YemongObservation
from boost_and_broadside.train.rl.buffer import RolloutBuffer
from boost_and_broadside.train.rl.critic import (
    ReturnNormalizer,
    expectation,
    outcome_targets,
    outcome_values,
    symexp,
    symlog,
    two_hot,
    value_bins,
)

BINS = value_bins(51, 5.0)


def test_bins_are_symmetric_and_include_zero():
    assert torch.allclose(BINS, -BINS.flip(0), atol=1e-5)
    assert abs(BINS[25].item()) < 1e-6


def test_symexp_inverts_symlog():
    x = torch.linspace(-200.0, 200.0, 41)
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("value", [0.0, 0.13, -0.6, 3.7, -42.0, 140.0])
def test_a_two_hot_target_has_the_value_as_its_mean(value):
    target = two_hot(torch.tensor([value]), BINS)
    assert target.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert (target * BINS).sum().item() == pytest.approx(value, rel=1e-5, abs=1e-5)


def test_a_two_hot_target_lights_at_most_two_neighbouring_bins():
    target = two_hot(torch.tensor([2.5]), BINS)[0]
    lit = torch.nonzero(target).flatten()
    assert len(lit) <= 2 and (len(lit) < 2 or lit[1] - lit[0] == 1)


def test_values_beyond_the_support_land_on_the_edge_bin():
    target = two_hot(torch.tensor([1e6]), BINS)[0]
    assert target[-1] == 1.0


def test_the_expectation_of_a_sharpened_two_hot_recovers_the_value():
    target = two_hot(torch.tensor([7.3]), BINS)
    logits = torch.log(target.clamp(min=1e-12))
    assert expectation(logits, BINS).item() == pytest.approx(7.3, rel=1e-3)


def _random_rollout(steps: int, envs: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    probabilities = torch.rand(steps, envs, NUM_OUTCOME_CLASSES, generator=generator)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    terminated = torch.rand(steps, envs, generator=generator) < 0.2
    result = torch.where(
        terminated,
        torch.randint(-1, 2, (steps, envs), generator=generator).float(),
        torch.zeros(steps, envs),
    )
    following = torch.rand(envs, NUM_OUTCOME_CLASSES, generator=generator)
    following = following / following.sum(-1, keepdim=True)
    return probabilities, result, terminated, following


def test_outcome_targets_are_distributions():
    probabilities, result, terminated, following = _random_rollout(12, 5)
    targets = outcome_targets(
        probabilities, result, terminated, following, terminated[-1], 0.99, 0.9
    )
    assert torch.all(targets >= 0.0)
    assert torch.allclose(targets.sum(-1), torch.ones(12, 5))


def test_the_outcome_targets_mean_is_the_scalar_lambda_return():
    """The categorical target and GAE's scalar return agree exactly (plan 10.2)."""
    steps, envs, gamma, lam = 16, 6, 0.97, 0.9
    probabilities, result, terminated, following = _random_rollout(steps, envs, seed=3)
    targets = outcome_targets(
        probabilities, result, terminated, following, terminated[-1], gamma, lam
    )

    buffer = RolloutBuffer(
        num_steps=steps,
        num_envs=envs,
        num_ships=1,
        num_components=1,
        obs_sample=YemongObservation(data={"pos": torch.zeros(envs, 1, 2)}),
        gamma=torch.tensor([gamma]),
        gae_lambda=torch.tensor([lam]),
        device=torch.device("cpu"),
    )
    for t in range(steps):
        buffer.add(
            {"pos": torch.zeros(envs, 1, 2)},
            torch.zeros(envs, 1, 3),
            torch.zeros(envs, 1),
            result[t].view(envs, 1, 1),
            outcome_values(probabilities[t]).view(envs, 1, 1),
            torch.ones(envs, 1, dtype=torch.bool),
            terminated=terminated[t],
        )
    buffer.compute_gae(outcome_values(following).view(envs, 1, 1), terminated[-1].float())

    assert torch.allclose(outcome_values(targets), buffer.returns[..., 0, 0], atol=1e-5)


def test_a_terminal_transition_targets_its_realised_class():
    probabilities, result, terminated, following = _random_rollout(4, 3)
    terminated[2] = True
    result[2] = torch.tensor([1.0, 0.0, -1.0])
    targets = outcome_targets(
        probabilities, result, terminated, following, terminated[-1], 0.99, 0.9
    )
    expected = [OUTCOME_WIN_INDEX, OUTCOME_TIE_INDEX, OUTCOME_LOSS_INDEX]
    assert targets[2].argmax(-1).tolist() == expected
    assert torch.allclose(targets[2].max(-1).values, torch.ones(3))


def test_the_discount_leaks_into_unresolved():
    probabilities = torch.zeros(1, 1, NUM_OUTCOME_CLASSES)
    probabilities[..., OUTCOME_WIN_INDEX] = 1.0
    following = probabilities[0].clone()
    targets = outcome_targets(
        probabilities,
        torch.zeros(1, 1),
        torch.zeros(1, 1, dtype=torch.bool),
        following,
        torch.zeros(1, dtype=torch.bool),
        0.9,
        0.95,
    )
    assert targets[0, 0, OUTCOME_UNRESOLVED_INDEX].item() == pytest.approx(0.1)
    assert targets[0, 0, OUTCOME_WIN_INDEX].item() == pytest.approx(0.9)


def test_the_return_scale_is_floored_at_one_win():
    normalizer = ReturnNormalizer(decay=0.0, floor=1.0, device=torch.device("cpu"))
    normalizer.update(torch.linspace(-0.1, 0.1, 100), torch.ones(100, dtype=torch.bool))
    assert normalizer.scale.item() == 1.0


def test_the_return_scale_is_the_percentile_spread():
    normalizer = ReturnNormalizer(decay=0.0, floor=1.0, device=torch.device("cpu"))
    returns = torch.linspace(0.0, 100.0, 1001)
    normalizer.update(returns, torch.ones_like(returns, dtype=torch.bool))
    assert normalizer.scale.item() == pytest.approx(90.0, rel=1e-3)


def test_the_return_scale_ignores_masked_entries():
    normalizer = ReturnNormalizer(decay=0.0, floor=1.0, device=torch.device("cpu"))
    returns = torch.cat([torch.linspace(0.0, 10.0, 101), torch.full((50,), 1e6)])
    mask = torch.cat([torch.ones(101, dtype=torch.bool), torch.zeros(50, dtype=torch.bool)])
    normalizer.update(returns, mask)
    assert normalizer.scale.item() == pytest.approx(9.0, rel=1e-3)


def test_the_percentiles_are_an_ema():
    normalizer = ReturnNormalizer(decay=0.99, floor=0.0, device=torch.device("cpu"))
    returns = torch.linspace(0.0, 100.0, 1001)
    normalizer.update(returns, torch.ones_like(returns, dtype=torch.bool))
    assert normalizer.high.item() == pytest.approx(0.01 * 95.0, rel=1e-3)


def test_the_normaliser_state_round_trips():
    normalizer = ReturnNormalizer(decay=0.5, floor=1.0, device=torch.device("cpu"))
    returns = torch.linspace(-30.0, 30.0, 101)
    normalizer.update(returns, torch.ones_like(returns, dtype=torch.bool))
    restored = ReturnNormalizer(decay=0.5, floor=1.0, device=torch.device("cpu"))
    restored.load_state_dict(normalizer.state_dict())
    assert restored.scale.item() == normalizer.scale.item()


def test_outcome_calibration_scores_each_step_against_its_episodes_result():
    """Steps are scored against the result their episode ends with, in the rollout."""
    from boost_and_broadside.train.rl.critic import outcome_calibration

    # Two environments, three steps. Env 0 wins at t=1; env 1 never ends.
    probabilities = torch.tensor(
        [
            [[0.6, 0.1, 0.1, 0.2], [0.5, 0.0, 0.5, 0.0]],
            [[0.8, 0.0, 0.0, 0.2], [0.5, 0.0, 0.5, 0.0]],
            [[0.2, 0.0, 0.6, 0.2], [0.5, 0.0, 0.5, 0.0]],
        ]
    )
    result = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    terminated = torch.tensor([[False, False], [True, False], [False, False]])
    diagnostics = outcome_calibration(probabilities, result, terminated)
    total, count = diagnostics["outcome/realised_win"]
    assert count.item() == 2.0, "only env 0's two steps up to its terminal one resolve"
    assert total.item() == 2.0
    predicted, _ = diagnostics["outcome/predicted_win"]
    # P(win | resolved) = 0.6 / 0.8 and 0.8 / 0.8.
    assert predicted.item() == pytest.approx(0.75 + 1.0)
    brier, _ = diagnostics["outcome/calibration_brier"]
    assert brier.item() == pytest.approx(0.25**2)
    top, top_count = diagnostics["outcome/calibration_bin_9/realised"]
    assert (top.item(), top_count.item()) == (1.0, 1.0)
