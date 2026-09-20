"""Small CPU correctness tests for the two-policy vmap prototype."""

import torch

from benchmarks.two_policy_vmap import (
    small_cpu_fixture,
    vmap_deterministic_step,
    vmap_sampled_step,
)
from boost_and_broadside.constants import NUM_POWER_ACTIONS, NUM_SHOOT_ACTIONS, NUM_TURN_ACTIONS


def test_vmapped_distinct_weight_outputs_match_separate_team_view_forwards():
    policies, observations, hidden = small_cpu_fixture()
    expected = []
    from benchmarks.two_policy_vmap import deterministic_sampling

    with deterministic_sampling():
        for policy, observation, policy_hidden in zip(policies, observations, hidden, strict=True):
            expected.append(policy.get_action_and_value(observation, policy_hidden))
    actual = vmap_deterministic_step(policies, observations, hidden)
    for actual_tensor, result_index in (
        (actual.action, 0),
        (actual.logprob, 1),
        (actual.value, 2),
        (actual.prediction, 3),
        (actual.hidden, 4),
    ):
        assert torch.allclose(actual_tensor[0], expected[0][result_index])
        assert torch.allclose(actual_tensor[1], expected[1][result_index])
    assert not torch.allclose(actual.value[0], actual.value[1])


def test_vmapped_sampling_uses_supported_distinct_randomness_mode():
    policies, observations, hidden = small_cpu_fixture()
    output = vmap_sampled_step(policies, observations, hidden)
    assert output.action.shape == (2, 1, 2, 3)
    assert output.logprob.shape == (2, 1, 2)
    assert torch.all((output.action[..., 0] >= 0) & (output.action[..., 0] < NUM_POWER_ACTIONS))
    assert torch.all((output.action[..., 1] >= 0) & (output.action[..., 1] < NUM_TURN_ACTIONS))
    assert torch.all((output.action[..., 2] >= 0) & (output.action[..., 2] < NUM_SHOOT_ACTIONS))
