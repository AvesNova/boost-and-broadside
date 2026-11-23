"""Invariants of the game step: finite values, inert padding, and team symmetry."""

import dataclasses

import pytest
import torch

torch.set_num_threads(1)

from lean.game import GameConfig, draw_noise, observe, reset, reset_done, step

CLOSE = GameConfig(arena_radius=700.0, spawn="mixed", clock_ticks=400)


def random_action(envs, slots, generator):
    return torch.stack(
        [torch.randint(0, n, (envs, slots), generator=generator) for n in (7, 3, 2)], -1
    )


def floats(tree):
    for field in dataclasses.fields(tree):
        value = getattr(tree, field.name)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            yield field.name, value


@pytest.mark.parametrize("sizes, ticks", [((3, 5), 300), ((20, 20), 30)])
def test_finite_and_padding_inert(sizes, ticks):
    """Catches a division by zero on a padded or dead row, and padding that acts or is hit."""
    generator = torch.Generator().manual_seed(1)
    envs = 4
    team_sizes = torch.tensor([sizes, (2, 2), sizes, (1, 3)])
    state = reset(CLOSE, envs, team_sizes, generator, "cpu")
    slots = state.x.shape[1]
    assert slots == sum(sizes)
    deaths = 0
    for _ in range(ticks):
        action = random_action(envs, slots, generator)
        state, out = step(CLOSE, state, action, draw_noise(CLOSE, envs, slots, generator, "cpu"))
        for name, value in list(floats(state)) + list(floats(out)):
            assert torch.isfinite(value).all(), name
        observation = observe(CLOSE, state, 0)
        assert torch.isfinite(observation.ship).all() and torch.isfinite(observation.globals).all()
        absent = ~state.exists
        assert not state.fired[absent].any() and not out.died[absent].any()
        assert (out.damage_taken[absent] == 0).all() and (out.hit_damage[absent] == 0).all()
        assert not observation.key_mask[absent].any() and not observation.own[absent].any()
        assert not state.alive[absent].any()
        assert (state.bullet_moves_left[absent] == 0).all()
        assert ((out.result >= 0) == out.done).all()
        deaths += int(out.died.sum())
        state = reset_done(CLOSE, state, out.done, team_sizes, generator)
    assert deaths > 0


def test_swapping_teams_mirrors_the_outcome():
    """Catches any dependence on slot order or team label: the slots reversed and the team
    labels exchanged must give the same game with the result mirrored."""
    generator = torch.Generator().manual_seed(2)
    envs = 8
    team_sizes = torch.tensor([(4, 4)] * envs)
    state = reset(CLOSE, envs, team_sizes, generator, "cpu")
    slots = state.x.shape[1]

    # The slot axis of each packed field: 2 in the blocks (E, rows, N[, K]), 1 in (E, N[, K]).
    slot_axis = {
        "ships": 2, "flags": 2, "identity": 2, "bullets": 2, "bullet_counts": 2, "bullet_real": 1,
        "seen": 2, "ticks_since_seen": 1, "pending": 1,
    }
    mirror = {name: getattr(state, name).flip(axis) for name, axis in slot_axis.items()}
    mirror["constants"] = dataclasses.replace(state.constants, table=state.constants.table.flip(2))
    other = dataclasses.replace(state, **mirror).replace(team=1 - state.team.flip(1))
    finished = torch.zeros(envs, dtype=torch.bool)
    damage = 0.0
    for _ in range(120):
        action = random_action(envs, slots, generator)
        noise = draw_noise(CLOSE, envs, slots, generator, "cpu")
        state, out = step(CLOSE, state, action, noise)
        other, mirrored = step(CLOSE, other, action.flip(1), {"spread": noise["spread"].flip(1)})
        live = ~finished
        assert (out.done == mirrored.done)[live].all()
        assert (out.died == mirrored.died.flip(1))[live].all()
        expected = torch.where(out.result < 2, torch.where(out.result >= 0, 1 - out.result, -1), 2)
        assert (expected == mirrored.result)[live].all()
        assert torch.allclose(out.hit_damage[live], mirrored.hit_damage.flip(1, 2)[live], atol=1e-4)
        assert torch.allclose(state.x[live], other.x.flip(1)[live], atol=1e-3)
        damage += float(out.hit_damage.sum())
        finished = finished | out.done
    assert damage > 0.0
