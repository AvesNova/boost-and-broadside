"""The sight test (game spec 10.1): the core and band cases enumerated, and random lines
against a brute-force closest-point check."""

import pytest
import torch

from bnb.game.perception.sight import membership, sighted

# One field at the origin: core radius 100, outer radius 200 (radius 150, band 100).
CORE, BAND, OUTSIDE = (10.0, 0.0), (150.0, 0.0), (300.0, 0.0)


def _clear(fields: list, a: tuple, b: tuple) -> bool:
    """A sight line from a to b, among fields (x, y, core radius, outer radius), with an
    unlimited range."""
    x, y, core, outer = (torch.tensor([[f[k] for f in fields]]) for k in range(4))
    discs = (x, y, core * core, outer * outer)

    def points(p):
        px, py = torch.tensor(p[0]).view(1, 1, 1), torch.tensor(p[1]).view(1, 1, 1)
        return px.expand(1, 2, 1), py.expand(1, 2, 1)

    ox, oy = points(a)
    tx, ty = points(b)
    alive, reach = torch.ones(1, 2, 1, dtype=torch.bool), torch.full((1, 2, 1), 1e9)
    observers = (ox, oy, *membership(discs, ox, oy), alive, reach)
    seen = sighted(discs, observers, (tx, ty, *membership(discs, tx, ty)))
    assert bool(seen[0, 0]) == bool(seen[0, 1])
    return bool(seen[0, 0])


FIELD = [(0.0, 0.0, 100.0, 200.0)]
CASES = [
    # (a, b, clear): each is tested both ways round
    (CORE, (-10.0, 5.0), True),  # core to core of the same field
    (CORE, BAND, True),  # core to band
    (CORE, (0.0, -190.0), True),  # core to band on the far side
    (CORE, OUTSIDE, False),  # core to outside
    (BAND, OUTSIDE, True),  # band to outside, clear of the core
    ((0.0, 150.0), (0.0, -300.0), False),  # band to outside through the core
    (BAND, (0.0, 150.0), True),  # band to band, clear of the core
    (BAND, (-150.0, 0.0), False),  # band to band through the core
    (OUTSIDE, (0.0, 300.0), True),  # outside to outside, clear of the core
    (OUTSIDE, (-300.0, 0.0), False),  # outside to outside through the core
    ((210.0, 0.0), (0.0, 210.0), True),  # outside to outside, crossing only the band
]


@pytest.mark.parametrize("a, b, want", CASES)
def test_the_core_band_and_outside_cases(a, b, want):
    assert _clear(FIELD, a, b) == want
    assert _clear(FIELD, b, a) == want


def test_a_field_without_a_band_blocks_core_to_outside():
    assert not _clear([(0.0, 0.0, 100.0, 100.0)], CORE, (100.5, 0.0))
    assert _clear([(0.0, 0.0, 100.0, 100.0)], CORE, (99.5, 0.0))


def test_a_field_without_a_core_blocks_nothing():
    assert _clear([(0.0, 0.0, 0.0, 50.0)], (-100.0, 0.0), (100.0, 0.0))
    assert _clear([(0.0, 0.0, 0.0, 50.0)], (0.0, 0.0), (100.0, 0.0))


def test_each_field_applies_its_own_rule():
    """Two fields: the core of one to the core of the other is blocked when each end is
    outside the other's outer disc, and clear when each core lies in the other's band."""
    apart = [(0.0, 0.0, 100.0, 200.0), (1000.0, 0.0, 100.0, 200.0)]
    assert not _clear(apart, CORE, (990.0, 0.0))
    overlapping = [(0.0, 0.0, 100.0, 200.0), (250.0, 0.0, 100.0, 200.0)]
    assert _clear(overlapping, (90.0, 0.0), (160.0, 0.0))
    # from both bands, across the second core to outside both: blocked
    assert not _clear(overlapping, (150.0, 100.0), (500.0, -100.0))


def _brute_clear(ax, ay, px, py, cx, cy, rho2, outer2):
    """(Q, O) clear lines from observers a (O,) to targets p (Q,), float64 closest points."""
    a = torch.stack([ax, ay], -1).double()[None, :, None, :]  # (1, O, 1, 2)
    p = torch.stack([px, py], -1).double()[:, None, None, :]  # (Q, 1, 1, 2)
    c = torch.stack([cx, cy], -1).double()[None, None, :, :]  # (1, 1, M, 2)
    d = p - a
    t = (((c - a) * d).sum(-1) / (d * d).sum(-1).clamp_min(1e-300)).clamp(0.0, 1.0)
    closest = a + t[..., None] * d
    dist2 = ((closest - c) ** 2).sum(-1)
    r2, o2 = rho2.double()[None, None, :], outer2.double()[None, None, :]
    a2, p2 = ((a - c) ** 2).sum(-1), ((p - c) ** 2).sum(-1)
    a_core, p_core = a2 < r2, p2 < r2
    both_field = (a2 < o2) & (p2 < o2)
    one_core = a_core ^ p_core
    blocked = (one_core & ~both_field) | (~a_core & ~p_core & (dist2 < r2))
    # Margin: pairs whose distances lie within float32 rounding of a core or outer edge.
    edge = torch.zeros_like(blocked)
    for q in (dist2, a2, p2):
        for bound in (r2, o2):
            edge = edge | ((q - bound).abs() < 1e-3 * bound)
    return ~blocked.any(-1), edge.any(-1)


def test_sight_lines_match_a_brute_force_closest_point_check():
    g = torch.Generator().manual_seed(3)
    B, T, O, Q, M = 6, 2, 9, 11, 13  # noqa: E741

    def u(*shape, lo=-1000.0, hi=1000.0):
        return lo + (hi - lo) * torch.rand(*shape, generator=g)

    cx, cy = u(B, M), u(B, M)
    rho = u(B, M, lo=0.0, hi=300.0)
    outer = rho + u(B, M, lo=0.0, hi=150.0)
    rho[:, 0] = 0.0  # a field without a core blocks nothing
    rho2, outer2 = rho**2, outer**2
    ox, oy, tx, ty = u(B, T, O), u(B, T, O), u(B, T, Q), u(B, T, Q)
    # Put some ends inside cores and bands, together and apart, to exercise the cases.
    ox[:, :, 0], oy[:, :, 0] = cx[:, 1, None] + 5.0, cy[:, 1, None]
    tx[:, :, 0], ty[:, :, 0] = cx[:, 1, None] - 5.0, cy[:, 1, None]
    tx[:, :, 1], ty[:, :, 1] = cx[:, 2, None], cy[:, 2, None] + 3.0
    band = 0.5 * (rho + outer)
    tx[:, :, 2], ty[:, :, 2] = cx[:, 1, None] + band[:, 1, None], cy[:, 1, None]
    ox[:, :, 1], oy[:, :, 1] = cx[:, 1, None], cy[:, 1, None] - band[:, 1, None]
    discs = (cx, cy, rho2, outer2)
    alive = torch.ones(B, T, O, dtype=torch.bool)
    total = checked = 0
    for o in range(O):  # one observer at a time: seen == that line is clear
        only = torch.zeros_like(alive)
        only[..., o] = True
        observers = (ox, oy, *membership(discs, ox, oy), only, torch.full((B, T, O), 1e9))
        seen = sighted(discs, observers, (tx, ty, *membership(discs, tx, ty)))
        for b in range(B):
            for t in range(T):
                want, edge = _brute_clear(
                    ox[b, t, o : o + 1],
                    oy[b, t, o : o + 1],
                    tx[b, t],
                    ty[b, t],
                    cx[b],
                    cy[b],
                    rho2[b],
                    outer2[b],
                )
                keep = ~edge[:, 0]
                assert torch.equal(seen[b, t][keep], want[:, 0][keep]), (b, t, o)
                total += Q
                checked += int(keep.sum())
    assert checked > 0.95 * total
    # The scenario is not trivial: some lines are blocked, some clear.
    assert 0 < int(seen.sum()) < seen.numel()


def test_range_is_the_observers():
    """A per-observer range: the far observer sees only with the long range."""
    far = torch.full((1, 1), 1e5)
    discs = (far, far, torch.zeros(1, 1), torch.zeros(1, 1))
    ox, oy = torch.tensor([[[0.0, 500.0]]]).expand(1, 2, 2), torch.zeros(1, 2, 2)
    tx, ty = torch.full((1, 2, 1), 1500.0), torch.zeros(1, 2, 1)
    alive = torch.tensor([[[True, False]]]).expand(1, 2, 2)  # only the far observer looks
    for far_range, want in ((1024.0, False), (1600.0, True)):
        reach = torch.tensor([[[1024.0, 1024.0]]]).expand(1, 2, 2).clone()
        reach[..., 0] = far_range
        observers = (ox, oy, *membership(discs, ox, oy), alive, reach)
        seen = sighted(discs, observers, (tx, ty, *membership(discs, tx, ty)))
        assert bool(seen.all()) == want


def test_the_sixty_fourth_field_uses_the_sign_bit():
    """With 64 fields, field 63's bit is the int64 sign bit: its rule still holds."""
    fields = [(1e6 + 1000.0 * i, 0.0, 100.0, 200.0) for i in range(63)] + FIELD
    assert _clear(fields, CORE, BAND)
    assert not _clear(fields, CORE, OUTSIDE)
    assert not _clear(fields, BAND, (-150.0, 0.0))
