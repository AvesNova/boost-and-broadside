"""Damage: the head-on term's shape, and StepOut's damage detail (for the reward ledger)."""

import math

import torch

from lean.game import card, core

torch.set_num_threads(1)


def test_head_on_term_is_none_head_on_half_at_the_half_angle_and_full_behind():
    """The aspect's head-on term in the hit angle's cosine (core._damage): no damage head-on,
    half at the card's half-damage angle, full from behind."""
    c = core.ship_constants(core.GameConfig(), torch.tensor([[0]]))
    half = card.HEAD_ON_ANGLE * card.CARD["head_on_angle_scale"][0]
    theta = torch.tensor([0.0, half, math.pi], dtype=torch.float64)
    ratio = (1.0 - torch.cos(theta)) * float(c["inverse_one_minus_cos_head_on"])
    curve = 1.0 - torch.exp(-card.LN_2 * ratio ** (0.5 * float(c["head_on_sharpness"])))
    assert float(curve[0]) == 0.0
    assert abs(float(curve[1]) - 0.5) < 1e-6 and abs(float(curve[2]) - 1.0) < 1e-5


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
                torch.randint(0, 7, (32, n), generator=generator),
                torch.randint(0, 3, (32, n), generator=generator),
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
