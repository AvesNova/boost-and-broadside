"""StepOut's damage detail (for v2's reward ledger): the attacker-by-target matrix agrees with
the per-ship totals, and the boundary and recharge parts are the step's own."""

import torch

from lean.game import core

torch.set_num_threads(1)


def test_hit_damage_matrix_matches_the_totals():
    """Row sums of hit_damage are each attacker's damage to enemies plus team-mates; column sums
    plus the boundary damage are each target's damage taken; no ship damages itself."""
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
        assert torch.allclose(matrix.sum(-1), out.damage_dealt + out.friendly_damage, atol=1e-4)
        assert torch.allclose(matrix.sum(-2) + out.boundary_damage, out.damage_taken, atol=1e-4)
        assert float(torch.diagonal(matrix, dim1=-2, dim2=-1).abs().max()) == 0.0
        assert (out.shield_restored >= 0.0).all()
        seen_hits += int((matrix > 0).sum())
        state = core.reset_done(config, state, out.done, sizes, generator)
    assert seen_hits > 0
