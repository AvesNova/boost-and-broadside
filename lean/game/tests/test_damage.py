"""Damage: the nose armour, the aspect curve and StepOut's damage detail (the reward ledger)."""

import math

import pytest
import torch

from lean.contract import POWERS, STICKS
from lean.game import card, core

torch.set_num_threads(1)
CPU = torch.device("cpu")


@pytest.mark.parametrize("pair", ["nose", "blame"])
def test_the_aspect_curve_is_0_inside_the_inner_angle_1_beyond_the_outer(pair):
    """The aspect curve (core.aspect_factor) on the sine of the angle off the nose, for both
    angle pairs of the card (the nose armour, the collision blame): 0 up to the inner angle, 1
    from the outer angle on, the smootherstep in the sine between (half-way in the sine, one
    half); behind the beam (a negative cosine) 1 even where the sine is small; the same on
    either side (the sine's sign)."""
    c = core.ship_constants(core.GameConfig(), torch.tensor([[0]]))
    inner, outer = card.CARD[f"{pair}_inner_angle"][0], card.CARD[f"{pair}_outer_angle"][0]
    middle = math.asin(0.5 * (math.sin(inner) + math.sin(outer)))
    angles = [0.0, 0.999 * inner, middle, 1.001 * outer, 0.5 * math.pi, 0.75 * math.pi, math.pi]
    for side in (1.0, -1.0):
        hit_cos = torch.tensor([math.cos(angle) for angle in angles])
        hit_sin = torch.tensor([abs(side * math.sin(angle)) for angle in angles])
        aspect = core.aspect_factor(
            hit_cos,
            hit_sin,
            torch.as_tensor(c[f"{pair}_inner_sin"]).flatten(),
            torch.as_tensor(c[f"inverse_{pair}_width"]).flatten(),
        )
        assert aspect.tolist()[:2] == [0.0, 0.0]
        assert abs(float(aspect[2]) - 0.5) < 1e-5
        assert aspect.tolist()[3:] == [1.0] * 4


def shot_at(heading_off_nose: float, side: float = 1.0, deflector: str = "off", nose_cone=1.0):
    """A shooter at the origin flying +x with its gun firing this tick (no spread) at a nearly
    still target 40 m ahead whose nose is `heading_off_nose` radians off the bullet's path
    (toward the shooter, to the left for side +1): the bullet's damage on the target."""
    config = core.GameConfig(field_count=0, fog=False, deflector=deflector)
    state = core.reset(config, 1, torch.tensor([[1, 1]]), torch.Generator().manual_seed(0), CPU)
    state = state.replace(constants=core.with_hit_model(state.constants, nose_cone=nose_cone))
    nose = math.pi - side * heading_off_nose
    state = core.with_velocity(
        state.replace(
            x=torch.tensor([[0.0, 40.0]]),
            y=torch.tensor([[0.0, 0.0]]),
            heading_cos=torch.tensor([[1.0, math.cos(nose)]]),
            heading_sin=torch.tensor([[0.0, math.sin(nose)]]),
            speed=torch.tensor([[200.0, 1.0]]),
            angle=torch.zeros(1, 2),
            angle_cos=torch.ones(1, 2),
            angle_sin=torch.zeros(1, 2),
            nose_rate=torch.zeros(1, 2),
        )
    )
    fire = torch.tensor([[[0, 0, 1], [0, 0, 0]]])
    state = state.replace(pending=fire)
    _, out = core.step(config, state, fire, {"spread": torch.zeros(1, 2, 2)})
    return float(out.hit_damage[0, 0, 1])


@pytest.mark.parametrize("deflector", list(card.DEFLECTOR_PRESETS))
def test_a_bullet_inside_the_nose_cone_does_nothing_and_beyond_it_full_damage(deflector):
    """Through the step (author 2026-11-25; on by default again 2026-11-28), with the nose
    armour's rule on (nose_cone 1, the default): a bullet 29 degrees off the target's nose does
    no damage, at 46 degrees as much as from dead astern, on either side, with the deflector on
    or off (a centred shot is never turned). Off (nose_cone 0), the same bullets all do the
    full damage."""
    full = shot_at(math.pi, deflector=deflector)
    for angle in (0.0, 29.0, 46.0):
        assert abs(shot_at(math.radians(angle), 1.0, deflector, 0.0) - full) < 1e-5 * full
    assert full > 0.0
    for side in (1.0, -1.0):
        assert shot_at(math.radians(29.0), side, deflector) == 0.0
        assert abs(shot_at(math.radians(46.0), side, deflector) - full) < 1e-5 * full


def test_hit_damage_matrix_matches_the_totals():
    """Column sums of hit_damage plus the boundary damage are each target's damage taken; no ship
    damages itself; recharges are not negative."""
    config = core.GameConfig(arena_radius=1200.0, spawn="mixed", clock_ticks=10**6)
    generator = torch.Generator().manual_seed(4)
    sizes = torch.tensor([[4, 4]]).expand(32, 2)
    state = core.reset(config, 32, sizes, generator, torch.device("cpu"))
    seen_hits = 0
    for _ in range(200):
        n = state.x.shape[1]
        action = torch.stack(
            [
                torch.randint(0, STICKS, (32, n), generator=generator),
                torch.randint(0, POWERS, (32, n), generator=generator),
                torch.ones((32, n), dtype=torch.long),
            ],
            -1,
        )
        noise = core.draw_noise(config, 32, n, generator, torch.device("cpu"))
        state, out = core.step(config, state, action, noise)
        matrix = out.hit_damage
        assert torch.allclose(matrix.sum(-2) + out.boundary_damage, out.damage_taken, atol=1e-4)
        assert float(torch.diagonal(matrix, dim1=-2, dim2=-1).abs().max()) == 0.0
        assert (out.shield_restored >= 0.0).all()
        seen_hits += int((matrix > 0).sum())
        state = core.take_deal(state, out.done, core.deal(config, 32, n, sizes, generator, "cpu"))
    assert seen_hits > 0


def test_only_straight_keeps_the_shield_quiet():
    """The shield's quiet (author 2026-11-26, card.QUIET_STICK_MAX): coasting, no trigger, not
    hurt, with its wait run out, a ship flying straight recharges and its wait stays run out;
    every turn level (Sustain, Max, Snap, either side) restarts the wait and recharges nothing."""
    config = core.GameConfig()
    state = core.reset(config, 1, torch.tensor([[1, 1]]), torch.Generator().manual_seed(0), CPU)
    c = state.constants
    state = state.replace(
        shield=0.5 * c["max_shield"] * torch.ones(1, 2), shield_wait=torch.zeros(1, 2)
    )
    no_damage, coast = torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.long)
    no_trigger = torch.zeros(1, 2, dtype=torch.bool)
    assert card.QUIET_STICK_MAX == card.STICK_LEVELS.index(0.0)
    for stick in range(len(card.STICK_LEVELS)):
        sticks = torch.full((1, 2), stick)
        _, _, restored, wait = core._shields(config, state, no_damage, sticks, coast, no_trigger)
        if card.STICK_LEVELS[stick] == 0.0:
            assert bool((restored > 0.0).all()) and bool((wait == 0.0).all()), stick
        else:
            assert bool((restored == 0.0).all()), stick
            assert torch.allclose(wait, torch.full_like(wait, c["shield_delay"])), stick
