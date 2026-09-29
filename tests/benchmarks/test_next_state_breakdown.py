"""CPU tests for the next-state breakdown's bucketing.

The split into ally / visible enemy / hidden enemy is the one piece of real
logic in the probe, and reading it off the wrong channel is silent: it produces
a plausible table of the wrong thing.
"""

import torch

from benchmarks.next_state_breakdown import OBSERVER_TEAM, _bucket_masks
from boost_and_broadside.env.observation import ObsKey


def _observation(team, visible):
    """A minimal ``(T, B, N)`` stand-in for the channels the bucketing reads."""
    return {
        ObsKey.TEAM_ID: torch.tensor(team, dtype=torch.uint8)[None, None],
        ObsKey.VISIBLE: torch.tensor(visible, dtype=torch.bool)[None, None],
    }


def test_buckets_split_by_team_and_sight():
    #                 ally  ally  enemy-seen  enemy-hidden
    observation = _observation([0, 0, 1, 1], [True, True, True, False])
    ns_mask = torch.ones(1, 1, 4, dtype=torch.bool)
    buckets = _bucket_masks(observation, ns_mask, num_ships=4)

    assert buckets["ally"].squeeze().tolist() == [True, True, False, False]
    assert buckets["visible_enemy"].squeeze().tolist() == [False, False, True, False]
    assert buckets["hidden_enemy"].squeeze().tolist() == [False, False, False, True]


def test_every_masked_token_lands_in_exactly_one_bucket():
    observation = _observation([0, 1, 1, 0], [True, False, True, True])
    ns_mask = torch.tensor([True, True, True, False])[None, None]
    buckets = _bucket_masks(observation, ns_mask, num_ships=4)

    disjoint = ["ally", "visible_enemy", "hidden_enemy"]
    stacked = torch.stack([buckets[name] for name in disjoint])
    assert torch.equal(stacked.sum(0), ns_mask.long())


def test_tokens_outside_the_training_mask_are_in_no_bucket():
    observation = _observation([0, 1], [True, False])
    buckets = _bucket_masks(observation, torch.zeros(1, 1, 2, dtype=torch.bool), num_ships=2)
    assert all(not mask.any() for mask in buckets.values())


def test_a_trailing_channel_axis_is_squeezed_not_broadcast():
    """The ship channels are ``(T, B, N)`` here and ``(T, B, N, 1)`` elsewhere."""
    flat = _observation([0, 1], [True, False])
    shaped = {key: value.unsqueeze(-1) for key, value in flat.items()}
    ns_mask = torch.ones(1, 1, 2, dtype=torch.bool)
    for name, mask in _bucket_masks(shaped, ns_mask, num_ships=2).items():
        assert mask.shape == (1, 1, 2)
        assert torch.equal(mask, _bucket_masks(flat, ns_mask, num_ships=2)[name])


def test_hidden_ally_is_reported_so_an_empty_class_is_proven_not_assumed():
    observation = _observation([0, 0], [True, False])
    buckets = _bucket_masks(observation, torch.ones(1, 1, 2, dtype=torch.bool), num_ships=2)
    assert buckets["hidden_ally"].squeeze().tolist() == [False, True]


def test_the_observer_is_team_zero_the_seat_the_live_policy_holds():
    assert OBSERVER_TEAM == 0
