"""Fields drawn per environment at every reset (author 2026-11-22; v2's random map)."""

import math

import torch

from lean.game import card, core

torch.set_num_threads(1)
CONFIG = core.GameConfig()
DEVICE = torch.device("cpu")


def fresh(envs=64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.tensor([[2, 2]]).expand(envs, 2)
    return core.reset(CONFIG, envs, sizes, generator, DEVICE), generator, sizes


def test_fields_are_point_symmetric_pairs_within_v2s_bounds():
    """v2's map: pairs mirrored through the centre, radius in the scaled bounds, centres inside
    the disc, levels in -2, -1, +1, +2 half-octaves."""
    state, _, _ = fresh()
    fields = state.fields
    count, pairs = CONFIG.field_count, CONFIG.field_count // 2
    x, y = fields[:, core.FIELD_X], fields[:, core.FIELD_Y]
    assert x.shape == (64, count)
    assert torch.allclose(x[:, pairs : 2 * pairs], -x[:, :pairs]) and torch.allclose(
        y[:, pairs : 2 * pairs], -y[:, :pairs]
    )
    radius = fields[:, core.FIELD_RADIUS]
    assert torch.equal(radius[:, :pairs], radius[:, pairs : 2 * pairs])
    scale = CONFIG.arena_radius / core.MAP_SCALE_RADIUS
    assert radius.min() >= scale * CONFIG.field_radius_min - 1e-3
    assert radius.max() <= scale * CONFIG.field_radius_max + 1e-3
    assert (
        torch.sqrt(x * x + y * y).max() <= CONFIG.field_disc_fraction * CONFIG.arena_radius + 1e-3
    )
    levels = torch.round(fields[:, core.FIELD_TARGET] / card.LN_SQRT2)
    assert set(levels.unique().tolist()) <= {-2.0, -1.0, 1.0, 2.0}


def test_each_environment_has_its_own_fields_and_sees_them():
    """Fields differ between environments, and the medium at the same point differs with them."""
    state, _, _ = fresh()
    x = state.fields[:, core.FIELD_X]
    assert (x[0] != x[1]).any()
    centre_of_first = state.fields[:, core.FIELD_X, :1], state.fields[:, core.FIELD_Y, :1]
    log_index, _, _ = core.medium(state.fields, *centre_of_first)
    assert log_index.abs().max() > 0.1  # each environment sees its own first field at its centre


def test_a_re_deal_redraws_fields_only_where_a_game_ended():
    """A finished environment gets a new map; the others keep theirs."""
    state, generator, sizes = fresh()
    done = torch.zeros(64, dtype=torch.bool)
    done[::2] = True
    after = core.take_deal(state, done, core.deal(CONFIG, 64, state.x.shape[1], sizes, generator, DEVICE))
    before_x, after_x = state.fields[:, core.FIELD_X], after.fields[:, core.FIELD_X]
    assert torch.equal(after_x[1::2], before_x[1::2])
    assert (after_x[::2] != before_x[::2]).any(-1).all()


def test_the_observation_carries_the_fields():
    """The observation has one row per field: centre, radius, band and level (v2's scales)."""
    state, _, _ = fresh(envs=4)
    observation = core.observe(CONFIG, state, 0)
    assert observation.field.shape == (4, CONFIG.field_count, 3)
    assert torch.allclose(observation.field_pos[..., 0], state.fields[:, core.FIELD_X])
    assert torch.allclose(
        observation.field[..., 0] * core.FIELD_RADIUS_SCALE, state.fields[:, core.FIELD_RADIUS]
    )


def test_the_core_game_runs_with_fields():
    """Smoke: the core game steps with per-environment fields and stays finite."""
    state, generator, sizes = fresh(envs=8)
    for _ in range(60):
        action = torch.randint(0, 3, (8, 4, 3), generator=generator) % torch.tensor([7, 3, 2])
        noise = core.draw_noise(CONFIG, 8, 4, generator, DEVICE)
        state, out = core.step(CONFIG, state, action, noise)
        state = core.take_deal(state, out.done, core.deal(CONFIG, 8, 4, sizes, generator, DEVICE))
    for name in ("x", "y", "speed", "energy", "shield"):
        assert torch.isfinite(getattr(state, name)).all(), name
    assert math.isfinite(float(state.energy.max()))
