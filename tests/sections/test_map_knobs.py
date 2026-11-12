"""The random map's knobs (`Rules.field_*`): radius range, centre disc and level shares."""

import dataclasses
import math

import torch

from bnb.game import api
from bnb.game.modes.rules import PRESETS
from bnb.game.physics.spawning import new_map

B = 4000
RULES = api.rules_of(PRESETS["deathmatch"])
M = RULES.num_fields
RADIUS = torch.full((B,), 2600.0)
SCALE = torch.full((B,), 2.0)
UNIFORMS = torch.rand(B, RULES.map_uniform_count, generator=torch.Generator().manual_seed(3))


def _fields(**changes):
    return new_map(dataclasses.replace(RULES, **changes), RADIUS, SCALE, UNIFORMS)[0]


def test_the_defaults_are_the_old_draws():
    fields = _fields()
    assert fields.radius.min() >= 2.0 * 30.0 and fields.radius.max() <= 2.0 * 490.0
    expected = torch.tensor([-2, -1, 1, 2])[
        (4.0 * UNIFORMS[:, : 4 * M].reshape(B, M, 4)[..., 3]).long()
    ]
    P = M // 2
    assert torch.equal(fields.level[:, :P], expected[:, :P])


def test_the_radius_range_and_disc_follow_the_rules():
    fields = _fields(field_radius_min=100.0, field_radius_max=150.0, field_disc_fraction=0.5)
    assert fields.radius.min() >= 200.0 and fields.radius.max() <= 300.0
    assert torch.hypot(fields.x, fields.y).max() <= 0.5 * 2600.0 + 1e-3


def test_the_level_shares():
    level = _fields(field_high_share=0.8, field_strong_share=0.25).level.float()
    assert math.isclose((level > 0).float().mean().item(), 0.8, abs_tol=0.03)
    assert math.isclose((level.abs() == 2).float().mean().item(), 0.25, abs_tol=0.03)
    assert set(level.unique().tolist()) <= {-2.0, -1.0, 1.0, 2.0}
    assert (_fields(field_high_share=1.0).level > 0).all()


def _radii(**changes):
    return _fields(**changes).radius / SCALE[:, None]


def test_the_default_distribution_is_the_old_uniform_radius():
    drawn = UNIFORMS[:, : 4 * M].reshape(B, M, 4)
    P = M // 2
    u = torch.cat([drawn[:, :P], drawn[:, :P], drawn[:, 2 * P :]], 1)[..., 2]
    old = SCALE[:, None] * (RULES.field_radius_min + (490.0 - 30.0) * u)
    assert torch.equal(_fields().radius, old)
    assert torch.equal(_fields(field_radius_distribution=0).radius, old)


def test_the_triangular_radius_peaks_at_the_smallest():
    radii = _radii(field_radius_distribution=1)
    assert radii.min() >= 30.0 - 1e-3 and radii.max() <= 490.0 + 1e-3
    assert math.isclose(radii.mean().item(), 30.0 + 460.0 / 3.0, rel_tol=0.01)
    assert (radii < 260.0).float().mean() > 0.7  # 1 - (1/2)^2 = 0.75 below the midpoint


def test_the_lognormal_radius_has_the_median_and_stays_in_range():
    radii = _radii(
        field_radius_distribution=2, field_radius_median=200.0, field_radius_log_sigma=0.3
    )
    assert radii.min() >= 30.0 - 1e-3 and radii.max() <= 490.0 + 1e-3
    assert math.isclose(radii.median().item(), 200.0, rel_tol=0.02)
    wide = _radii(
        field_radius_distribution=2, field_radius_median=200.0, field_radius_log_sigma=2.0
    )
    assert wide.min() >= 30.0 - 1e-3 and wide.max() <= 490.0 + 1e-3 and torch.isfinite(wide).all()


def test_the_summary_matches_the_draws():
    from bnb.game.physics.spawning import field_radius_summary

    for code in (0, 1, 2):
        rules = dataclasses.replace(RULES, field_radius_distribution=code)
        mean, low, high = field_radius_summary(rules)
        radii = _radii(field_radius_distribution=code).flatten()
        assert math.isclose(mean, radii.mean().item(), rel_tol=0.02)
        assert math.isclose(low, radii.quantile(0.1).item(), rel_tol=0.03)
        assert math.isclose(high, radii.quantile(0.9).item(), rel_tol=0.03)
