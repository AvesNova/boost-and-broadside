"""The core damage form (agreed: the same effects, cheaper). Expected: similar to v2's damage,
not equal; the head-on curve within 0.024 of v2's, the same half-damage angle; total damage
within 2% on the same hits (the closeness is v2's)."""

import dataclasses
import math

import torch

from lean.game import card, core

torch.set_num_threads(1)


def head_on_curves(degrees):
    c = core.ship_constants(core.GameConfig(), torch.tensor([[2]]))
    theta = torch.tensor(degrees, dtype=torch.float64).deg2rad()
    sharpness = float(c["head_on_sharpness"])
    v2 = 1.0 - torch.exp(-card.LN_2 * (theta * float(c["inverse_head_on_angle"])) ** sharpness)
    ratio = (1.0 - torch.cos(theta)) * float(c["inverse_one_minus_cos_head_on"])
    cheap = 1.0 - torch.exp(-card.LN_2 * ratio ** (0.5 * sharpness))
    return v2, cheap


def test_head_on_curve_keeps_the_half_damage_angle_and_stays_within_0_024():
    """Same effect: no damage head-on, half at the same angle, full from behind; at most 0.024
    from v2's curve anywhere."""
    degrees = [i * 0.05 for i in range(3601)]
    v2, cheap = head_on_curves(degrees)
    assert float((v2 - cheap).abs().max()) < 0.024
    half = card.HEAD_ON_ANGLE * card.CARD["head_on_angle_scale"][2]
    v2_half, cheap_half = head_on_curves([math.degrees(half)])
    assert abs(float(cheap_half[0]) - 0.5) < 1e-6
    assert float(cheap[0]) == 0.0 and abs(float(cheap[-1]) - 1.0) < 1e-5


def test_damage_per_tick_is_similar_to_v2s_on_the_same_hits():
    """The same states and actions with each damage form: the same hits and deaths are not
    required (damage feeds back), so compare one tick from many states; total damage per tick
    within 2%, and every hit that does damage in one does damage in the other."""
    generator = torch.Generator().manual_seed(3)
    v2_config = core.GameConfig(arena_radius=1200.0)
    core_config = dataclasses.replace(v2_config, damage_model="core")
    sizes = torch.tensor([[5, 5]]).expand(64, 2)
    state = core.reset(v2_config, 64, sizes, generator, torch.device("cpu"))
    totals = {"v2": 0.0, "core": 0.0}
    for _ in range(150):
        action = torch.randint(0, 42, (64, 10), generator=generator)
        action = torch.stack([action // 6, (action // 2) % 3, torch.ones_like(action)], -1)
        noise = core.draw_noise(v2_config, 64, 10, generator, torch.device("cpu"))
        after_v2, out_v2 = core.step(v2_config, state, action, noise)
        _, out_core = core.step(core_config, state, action, noise)
        totals["v2"] += float(out_v2.damage_taken.sum())
        totals["core"] += float(out_core.damage_taken.sum())
        assert torch.equal(out_v2.damage_taken > 0, out_core.damage_taken > 0)
        state = core.reset_done(v2_config, after_v2, out_v2.done, sizes, generator)
    assert totals["v2"] > 0.0
    assert abs(totals["core"] - totals["v2"]) < 0.02 * totals["v2"], totals
