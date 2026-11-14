"""The keyed random streams (`bnb.streams`) and the samplers that read them: draws are pure
functions of their keys, distributed as they should be, and an action sampled by the inverse
CDF from the same uniforms is the same action wherever the distributions agree."""

import pytest
import torch

from bnb.contracts.action import NUM_JOINT
from bnb.controllers.scripted import Marginals, sample
from bnb.models import heads
from bnb.models.yemong import policy
from bnb.streams import (
    action_uniforms,
    inverse_cdf,
    keyed_normal,
    row_uniforms,
    stream_generator,
    stream_key,
)

DRAWS = 200_000


def test_keyed_normals_are_standard_and_pure():
    key, index = stream_key(0, "test"), torch.arange(DRAWS)
    normal = keyed_normal(key, index)
    assert abs(float(normal.mean())) < 0.01 and abs(float(normal.std()) - 1.0) < 0.01
    assert torch.isfinite(normal).all()
    assert torch.equal(normal, keyed_normal(key, index))
    assert not torch.equal(normal, keyed_normal(stream_key(1, "test"), index))


def test_a_rows_uniforms_are_keyed_by_the_row_and_the_element():
    rows = torch.tensor([3, 3, 4])
    uniforms = row_uniforms(7, rows, (5, 3))
    assert uniforms.shape == (3, 5, 3)
    assert torch.equal(uniforms[0], uniforms[1]) and not torch.equal(uniforms[0], uniforms[2])
    assert uniforms[0].unique().numel() == 15  # every element its own draw


def test_a_stream_generator_depends_on_the_seed_on_the_cpu():
    """The CPU generator reads a seed's low 32 bits only: the stream key is all it gets."""
    first = torch.rand(8, generator=stream_generator(0, "minibatch_order"))
    second = torch.rand(8, generator=stream_generator(1, "minibatch_order"))
    assert not torch.equal(first, second)
    again = torch.rand(8, generator=stream_generator(0, "minibatch_order"))
    assert torch.equal(first, again)


def test_the_inverse_cdf_never_draws_a_category_of_weight_zero():
    weights = torch.tensor([0.0, 2.0, 0.0, 1.0, 0.0])
    uniform = torch.rand(DRAWS, generator=torch.Generator().manual_seed(0))
    drawn = inverse_cdf(weights.expand(DRAWS, -1), uniform)
    counts = torch.bincount(drawn, minlength=5).float() / DRAWS
    assert counts[0] == counts[2] == counts[4] == 0
    torch.testing.assert_close(counts[[1, 3]], torch.tensor([2 / 3, 1 / 3]), atol=0.01, rtol=0)


def test_the_joint_sampler_draws_the_policys_distribution_and_reports_its_log_probability():
    logits = torch.randn(4, 1, NUM_JOINT, generator=torch.Generator().manual_seed(0)) * 2.0
    many = logits.expand(4, DRAWS // 4, NUM_JOINT).reshape(DRAWS, 1, NUM_JOINT)
    uniforms = action_uniforms(torch.arange(DRAWS), 1)
    joint, log_probability, entropy = policy.sample(many, uniforms)
    expected_log, expected_entropy = policy.evaluate(many, joint)
    torch.testing.assert_close(log_probability, expected_log)
    torch.testing.assert_close(entropy, expected_entropy)
    rows = joint.reshape(4, DRAWS // 4).long()
    for row in range(4):
        frequency = torch.bincount(rows[row], minlength=NUM_JOINT).float() / rows.shape[1]
        expected = torch.softmax(logits[row, 0], -1)
        assert (frequency - expected).abs().max() < 0.01


def test_the_same_uniforms_pick_the_same_action_from_close_policies():
    """Identical logits give identical actions; slightly different logits mostly the same
    (independent draws would agree only as often as two samples of one distribution)."""
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(DRAWS // 10, 5, NUM_JOINT, generator=generator)
    uniforms = action_uniforms(torch.arange(logits.shape[0]), 5)
    first, _, _ = policy.sample(logits, uniforms)
    same, _, _ = policy.sample(logits.clone(), uniforms)
    assert torch.equal(first, same)
    nearby, _, _ = policy.sample(
        logits + 0.05 * torch.randn(logits.shape, generator=generator), uniforms
    )
    agreement = float((first == nearby).float().mean())
    independent = float((torch.softmax(logits, -1) ** 2).sum(-1).mean())
    assert agreement > 0.9 > independent


@pytest.mark.parametrize("sampler", ["factored", "scripted"])
def test_the_factored_samplers_draw_each_factor_by_its_own_uniform(sampler):
    generator = torch.Generator().manual_seed(2)
    sizes = (3, 7, 2)
    logits = torch.randn(DRAWS // 10, 5, sum(sizes), generator=generator)
    uniforms = action_uniforms(torch.arange(logits.shape[0]), 5)
    parts = [torch.softmax(part, -1) for part in logits.split(sizes, -1)]
    if sampler == "factored":
        action, _, _ = heads.sample(logits, uniforms)
        picks = torch.stack([action.power, action.stick, action.shoot], -1)
    else:
        picks = sample(Marginals(*parts), torch.arange(logits.shape[0]))
    for index, part in enumerate(parts):
        assert torch.equal(picks[..., index].long(), inverse_cdf(part, uniforms[..., index]))
