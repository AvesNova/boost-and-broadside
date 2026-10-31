"""The sight test against a brute-force closest-point check (game spec 10.1)."""

import torch

from bnb.game.perception.sight import core_membership, sighted


def _brute_clear(ax, ay, px, py, cx, cy, rho2):
    """(Q, O) clear lines from observers a (O,) to targets p (Q,), float64 closest points."""
    a = torch.stack([ax, ay], -1).double()[None, :, None, :]  # (1, O, 1, 2)
    p = torch.stack([px, py], -1).double()[:, None, None, :]  # (Q, 1, 1, 2)
    c = torch.stack([cx, cy], -1).double()[None, None, :, :]  # (1, 1, M, 2)
    d = p - a
    t = (((c - a) * d).sum(-1) / (d * d).sum(-1).clamp_min(1e-300)).clamp(0.0, 1.0)
    closest = a + t[..., None] * d
    dist2 = ((closest - c) ** 2).sum(-1)
    r2 = rho2.double()[None, None, :]
    both_in = (((a - c) ** 2).sum(-1) < r2) & (((p - c) ** 2).sum(-1) < r2)
    blocked = (dist2 < r2) & ~both_in
    # Margin: pairs whose closest distance lies within float32 rounding of a core edge.
    edge = ((dist2 - r2).abs() < 1e-3 * r2).any(-1) | \
        ((((a - c) ** 2).sum(-1) - r2).abs() < 1e-3 * r2).any(-1) | \
        ((((p - c) ** 2).sum(-1) - r2).abs() < 1e-3 * r2).any(-1)
    return ~blocked.any(-1), edge


def test_sight_lines_match_a_brute_force_closest_point_check():
    g = torch.Generator().manual_seed(3)
    B, T, O, Q, M = 6, 2, 9, 11, 13  # noqa: E741

    def u(*shape, lo=-1000.0, hi=1000.0):
        return lo + (hi - lo) * torch.rand(*shape, generator=g)

    cx, cy = u(B, M), u(B, M)
    rho2 = u(B, M, lo=0.0, hi=300.0) ** 2
    rho2[:, 0] = 0.0  # a field without a core blocks nothing
    ox, oy, tx, ty = u(B, T, O), u(B, T, O), u(B, T, Q), u(B, T, Q)
    # Put some ends inside cores, together and apart, to exercise the exceptions.
    ox[:, :, 0], oy[:, :, 0] = cx[:, 1, None] + 5.0, cy[:, 1, None]
    tx[:, :, 0], ty[:, :, 0] = cx[:, 1, None] - 5.0, cy[:, 1, None]
    tx[:, :, 1], ty[:, :, 1] = cx[:, 2, None], cy[:, 2, None] + 3.0
    cores = (cx, cy, rho2)
    alive = torch.ones(B, T, O, dtype=torch.bool)
    total = checked = 0
    for o in range(O):  # one observer at a time: seen == that line is clear
        only = torch.zeros_like(alive)
        only[..., o] = True
        observers = (ox, oy, core_membership(cores, ox, oy), only, torch.full((B, T, O), 1e9))
        seen = sighted(cores, observers, (tx, ty, core_membership(cores, tx, ty)))
        for b in range(B):
            for t in range(T):
                want, edge = _brute_clear(ox[b, t, o:o + 1], oy[b, t, o:o + 1], tx[b, t], ty[b, t],
                                          cx[b], cy[b], rho2[b])
                keep = ~edge[:, 0]
                assert torch.equal(seen[b, t][keep], want[:, 0][keep]), (b, t, o)
                total += Q
                checked += int(keep.sum())
    assert checked > 0.95 * total
    # The scenario is not trivial: some lines are blocked, some clear.
    assert 0 < int(seen.sum()) < seen.numel()


def test_range_is_the_observers():
    """A per-observer range: the far observer sees only with the long range."""
    cores = (torch.full((1, 1), 1e5), torch.full((1, 1), 1e5), torch.zeros(1, 1))
    ox, oy = torch.tensor([[[0.0, 500.0]]]).expand(1, 2, 2), torch.zeros(1, 2, 2)
    tx, ty = torch.full((1, 2, 1), 1500.0), torch.zeros(1, 2, 1)
    alive = torch.tensor([[[True, False]]]).expand(1, 2, 2)  # only the far observer looks
    for far_range, want in ((1024.0, False), (1600.0, True)):
        reach = torch.tensor([[[1024.0, 1024.0]]]).expand(1, 2, 2).clone()
        reach[..., 0] = far_range
        observers = (ox, oy, core_membership(cores, ox, oy), alive, reach)
        seen = sighted(cores, observers, (tx, ty, core_membership(cores, tx, ty)))
        assert bool(seen.all()) == want
