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
