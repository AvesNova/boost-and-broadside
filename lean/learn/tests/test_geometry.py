"""The geometry losses' rows, pairs and draw (lean.learn.geometry), and the visible-ship
next-state and hit rows (author 2026-11-28)."""

import torch

from lean import game
from lean.learn import collect
from lean.learn.geometry import (
    draw_keys,
    geometry_losses,
    observed_rows,
    pair_counts,
    pair_query,
    sample,
    ship_pairs,
)
from lean.learn.tests.setting import small
from lean.learn.training import hit_loss
from lean.model.geometry import column_cross_entropy

torch.set_num_threads(2)


def test_sampling_is_unbiased_and_exact_when_k_covers_the_valid_pairs():
    """Catches a biased pair estimate: over many draws of K valid pairs, the weighted sum over
    the sample divided by the valid count averages to the mean over every valid pair, and with
    K at or above the valid count each draw gives it exactly. Invalid pairs are never chosen
    as valid."""
    generator = torch.Generator().manual_seed(0)
    pairs, draws = 28, 40000
    valid = torch.rand(pairs, generator=generator) < 0.6
    losses = torch.rand(pairs, generator=generator) * 3.0
    exact = losses[valid].mean()
    count = valid.sum()
    for samples in (4, 8, int(count), 28):
        keys = torch.rand((1, draws, pairs), generator=generator)
        picked = sample(valid.expand(1, draws, pairs), keys, samples)
        assert not (picked.valid & ~valid[picked.index]).any()
        estimate = (losses[picked.index] * picked.weight).sum(-1) / count
        if samples >= count:
            assert torch.allclose(estimate, exact.expand_as(estimate), atol=1e-5)
        else:
            error = estimate.std() / draws**0.5
            assert (estimate.mean() - exact).abs() < 4.0 * error
            # Without replacement: K distinct valid pairs when enough are valid.
            assert int(picked.valid.sum(-1).min()) == samples
            assert (picked.index.sort(-1).values.diff(dim=-1) > 0).all()


def test_every_pair_with_no_draw_and_the_upper_triangle():
    """Catches pairs outside the upper triangle (symmetric targets train on first < second
    only) and a draw where K covers every pair."""
    first, second = ship_pairs(6, "cpu")
    assert (first < second).all() and first.numel() == 15
    keys, field_keys = draw_keys(15, 4, 3, 2, 6, 5, None, "cpu")
    assert keys is None and field_keys.shape == (3, 2, 30)
    observed = torch.ones(3, 2, 6, dtype=torch.bool)
    query, ships, fields = pair_query(observed, 5, None, field_keys, 15, 4)
    assert torch.equal(query.first[0, 0], first) and ships.valid.all()
    assert fields.index.shape == (3, 2, 4) and (fields.weight == 30 / 4).all()


def test_the_learner_rows_are_the_visible_ships():
    """Catches hidden enemies or dead ships in the trained rows: in a real fogged rollout the
    observed rows are the seat's living ships and the enemies it sees; next_valid rows are
    observed, alive after the step and visible on the next tick; some visible enemies train
    and no hidden one does; pairs with a hidden ship are never valid."""
    config, game_config, policy, opponent, state, carry, generator = small(
        envs=8, pair_samples=3, field_pair_samples=5
    )
    rollout = collect(config, game_config, policy, opponent, state, carry, 48, generator)
    observation = rollout.observation
    observed = rollout.observed
    assert torch.equal(observed, observed_rows(observation))
    assert (observed >= observation.own).all()
    enemy = observation.key_mask & ~observation.own
    assert (observed & enemy).any() and (enemy & ~observed).any()  # some seen, some hidden
    assert not (rollout.next_valid & ~observed).any()
    following = rollout.next_valid[:-1] & ~rollout.reset[1:, :, None]
    assert (rollout.next_valid & ~observation.own).any()
    assert not (following & ~observation.visible[1:]).any()
    keys, field_keys = draw_keys(
        3, 5, *observed.shape, observation.field_pos.shape[-2], None, "cpu"
    )
    query, ships, _ = pair_query(observed, observation.field_pos.shape[-2], keys, field_keys, 3, 5)
    both = observed.gather(-1, query.first) & observed.gather(-1, query.second)
    assert torch.equal(ships.valid, both & ships.valid)
    assert (both == ships.valid).all()
    # The hit loss's rows are the observed ones.
    _, labelled = hit_loss(torch.zeros(observed.shape + (3,)), rollout.hit_label, observed)
    assert torch.equal(labelled, observed & (rollout.hit_label >= 0))


def test_the_pair_loss_with_every_pair_is_the_mean_over_the_valid_pairs():
    """Catches a pair loss that is not the mean cross-entropy over the valid pairs and
    columns, or targets that are not the game's: with K covering every pair the loss equals a
    direct sum over every valid pair of lean.game.pair_geometry's targets."""
    config, game_config, policy, opponent, state, carry, generator = small(envs=4)
    rollout = collect(config, game_config, policy, opponent, state, carry, 12, generator)
    observation, observed = rollout.observation, rollout.observed
    fields = observation.field_pos.shape[-2]
    query, ships, field_sample = pair_query(observed, fields, None, None, 10**6, 10**6)
    with torch.no_grad():
        outputs = policy.sequence(
            observation, rollout.initial_carry, rollout.reset, rollout.position_offset, query
        )
    pairs, field_pairs = pair_counts(observed, fields)
    denominators = torch.stack([observed.sum().float(), pairs.sum(), field_pairs.sum()])
    pair, field, edge = geometry_losses(
        policy, outputs, observation, rollout.geometry, query, ships, field_sample, denominators
    )
    total, slots = 0.0, observed.shape[-1]
    first, second = ship_pairs(slots, "cpu")
    for index, (one, two) in enumerate(zip(first.tolist(), second.tolist(), strict=True)):
        valid = observed[..., one] & observed[..., two]
        if not valid.any():
            continue
        forward = game.pair_geometry(
            observation.pos[..., one, :],
            observation.heading[..., one],
            rollout.geometry[..., one, :],
            observation.pos[..., two, :],
            rollout.geometry[..., two, :],
        )
        reverse = game.pair_geometry(
            observation.pos[..., two, :],
            observation.heading[..., two],
            rollout.geometry[..., two, :],
            observation.pos[..., one, :],
            rollout.geometry[..., one, :],
        )
        for column, logits in zip(policy.config.pair_columns, outputs.pairs, strict=True):
            centres = policy.geometry_centres(column)
            sigma = policy.config.next_state_sigma
            cross = column_cross_entropy(
                column, logits[:, :, index], forward[column.name], centres, sigma
            )
            if column.both_orders:
                back = logits[:, :, first.numel() + index]
                cross = 0.5 * (
                    cross + column_cross_entropy(column, back, reverse[column.name], centres, sigma)
                )
            total = total + cross[valid].sum()
    expected = total / (pairs.sum() * len(policy.config.pair_columns))
    assert torch.allclose(pair, expected, rtol=1e-5)
    assert torch.isfinite(field) and torch.isfinite(edge) and field > 0 and edge > 0
