"""The policy term's trust regions (`LearnerConfig.trust_region`): PPO's clip as before, DAPO's
clip-higher, and DPPO's mask (Qi et al. 2026) with the exact total variation of the joint
action distribution or its binary approximation."""

import copy
import math

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.contracts.action import NUM_JOINT
from bnb.controllers.scripted import Chaser
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.learn import losses
from bnb.learn.ppo import DIVERGENCE_BINS, LearnerConfig, divergence_quantiles


def _old_surrogate(log_probability, old, advantage, scale, clip):
    """The clipped surrogate exactly as it was written before the trust-region option."""
    scaled = advantage / scale
    ratio = torch.exp(log_probability - old)
    return torch.maximum(-scaled * ratio, -scaled * torch.clamp(ratio, 1.0 - clip, 1.0 + clip))


def test_the_clip_is_todays():
    generator = torch.Generator().manual_seed(0)
    log_probability = -3 * torch.rand(4096, generator=generator)
    old = log_probability + 0.4 * torch.randn(4096, generator=generator)
    advantage = torch.randn(4096, generator=generator)
    got = losses.surrogate(log_probability, old, advantage, 1.7, 0.2)
    # the same few float32 operations in another order: 1e-6 is about 10 ulps of a value near 1
    torch.testing.assert_close(
        got, _old_surrogate(log_probability, old, advantage, 1.7, 0.2), rtol=1e-6, atol=1e-6
    )
    assert LearnerConfig().trust_region == "clip"


def test_clip_higher_bounds_the_ratio_asymmetrically():
    old = torch.zeros(4)
    ratio = torch.tensor([0.5, 0.9, 1.25, 2.0])
    advantage = torch.ones(4)
    got = losses.surrogate(ratio.log(), old, advantage, 1.0, 0.2, 0.28)
    # A > 0: min(r, clip(r)) = r below 1 + 0.28, the bound above it
    assert torch.allclose(-got, torch.tensor([0.5, 0.9, 1.25, 1.28]))
    got = losses.surrogate(ratio.log(), old, -advantage, 1.0, 0.2, 0.28)
    # A < 0: -max(-r, -clip(r)) bounds the ratio below at 1 - 0.2
    assert torch.allclose(got, torch.tensor([0.8, 0.9, 1.25, 2.0]))


def test_the_dppo_mask_blocks_only_moves_further_away_past_delta():
    ratio = torch.tensor([1.5, 0.5, 0.5, 1.5, 1.5, 0.5])
    advantage = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
    divergence = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.1, 0.1])
    keep = losses.dppo_keep(divergence, ratio, advantage, 0.15)
    # A > 0, r > 1 past delta: masked; A > 0, r < 1: back toward 1, kept; A < 0, r < 1 past
    # delta: masked; A < 0, r > 1: kept; under delta: kept either way
    assert keep.tolist() == [False, True, False, True, True, True]
    log_probability = ratio.log().requires_grad_()
    loss = losses.dppo_surrogate(log_probability, torch.zeros(6), advantage, 1.0, keep).sum()
    loss.backward()
    # a kept sample's gradient is the unclipped -A r; a masked one's is zero
    expected = torch.where(keep, -advantage * ratio, 0.0)
    assert torch.allclose(log_probability.grad, expected)


def test_the_exact_total_variation_matches_a_hand_computation():
    old = torch.zeros(NUM_JOINT)
    old[:3] = torch.tensor([0.5, 0.3, 0.2])
    current = torch.zeros(NUM_JOINT)
    current[:4] = torch.tensor([0.4, 0.4, 0.1, 0.1])
    logits = torch.where(current > 0, current.log(), -1e4)
    # 1/2 (0.1 + 0.1 + 0.1 + 0.1)
    assert losses.total_variation(old, logits).item() == pytest.approx(0.2, abs=1e-6)
    same = losses.total_variation(old, old.clamp_min(1e-30).log()).item()
    assert same == pytest.approx(0.0, abs=1e-6)


def test_the_binary_total_variation_is_a_lower_bound_of_the_exact():
    generator = torch.Generator().manual_seed(1)
    old_logits = 2 * torch.randn((2000, NUM_JOINT), generator=generator)
    logits = old_logits + 0.5 * torch.randn((2000, NUM_JOINT), generator=generator)
    old = torch.softmax(old_logits, -1)
    action = torch.multinomial(old, 1, generator=generator)[:, 0]
    pick = lambda values: values.gather(-1, action[:, None])[:, 0]  # noqa: E731
    binary = losses.binary_total_variation(pick(torch.log_softmax(logits, -1)), pick(old.log()))
    exact = losses.total_variation(old, logits)
    assert (binary <= exact + 1e-6).all() and (binary < exact).float().mean() > 0.9


def test_the_divergence_quantiles_read_the_log_histogram():
    counts = torch.zeros(DIVERGENCE_BINS)
    counts[DIVERGENCE_BINS - 4] = 10.0  # [10^-0.5, 10^-0.375): every value in one bin
    got = divergence_quantiles(counts, torch.tensor([0.25, 0.5, 1.0]))
    assert torch.allclose(got, 10 ** torch.tensor([-0.46875, -0.4375, -0.375]))
    assert torch.isnan(divergence_quantiles(torch.zeros(DIVERGENCE_BINS), torch.tensor([0.5])))


def test_dppo_tv_needs_the_rollouts_distribution():
    with pytest.raises(ValueError, match="rollout"):
        LearnerConfig(trust_region="dppo_tv", old_log_probabilities="recomputed")
    with pytest.raises(ValueError, match="trust_region"):
        LearnerConfig(trust_region="kl")


@pytest.fixture(scope="module")
def recorded(module_global_generator):
    """A rollout whose learner records its joint action distribution."""
    net = cases.model()
    controller = cases.controller(net, cases.SHORT_SQUAD)
    controller.distribution = True
    runner = lc.rollout(
        net,
        controllers={
            "learner": controller,
            "chaser": Chaser(
                characteristics(api.ship_types(), api.rules_of(cases.SHORT_SQUAD)),
                cases.SHORT_SQUAD.ships_per_team,
            ),
        },
    )
    return net, runner


def test_the_rollout_records_the_distribution_of_the_sampled_action(recorded, global_generator):
    _, runner = recorded
    trajectory = lc.whole(runner)
    probabilities = trajectory.probabilities
    assert probabilities.dtype == torch.float16
    assert probabilities.shape == trajectory.log_probability.shape + (NUM_JOINT,)
    from bnb.learn.ppo import _own_joint  # noqa: PLC0415

    sampled = probabilities.float().gather(-1, _own_joint(trajectory).long()[..., None])[..., 0]
    alive = trajectory.alive[..., : trajectory.log_probability.shape[-1]]
    assert torch.allclose(
        sampled[alive], trajectory.log_probability.exp()[alive], rtol=2e-3, atol=1e-4
    )
    assert torch.allclose(probabilities.float().sum(-1), torch.ones(()), atol=2e-2)


@pytest.mark.parametrize("region", ["clip", "clip_higher", "dppo_tv", "dppo_binary_tv"])
def test_every_trust_region_updates_and_reports_its_rate(recorded, region, global_generator):
    net, runner = recorded
    model = copy.deepcopy(net)
    learner = lc.learner(model, runner, epochs=2, minibatches=2, trust_region=region)
    out = learner.update(runner.store, runner.bootstrap(), 0)
    read = {
        name: value.total / value.count if hasattr(value, "total") else value
        for name, value in out.items()
    }
    for name in ("policy/clip_fraction", "policy/divergence/mean", "policy/divergence/p50"):
        assert math.isfinite(float(read[name])), name
    assert 0.0 <= float(read["policy/clip_fraction"]) <= 1.0
    assert float(read["policy/divergence/p50"]) <= float(read["policy/divergence/p99"])
    rates = [name for name in read if name.startswith("policy/clip_fraction/entropy_")]
    assert rates  # the buckets that held tokens
    moved = any(
        not torch.equal(a, b) for a, b in zip(model.parameters(), net.parameters(), strict=True)
    )
    assert moved
