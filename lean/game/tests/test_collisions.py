"""Collisions (author 2026-11-25): every pair of living ships, any teams; scored once per overlap
on the tick of first touch (author 2026-11-26); blame by the nose cone at first touch; damage
linear in the closing speed, a full shield's worth head-on at cruise with full blame, closeness
by the pass's predicted closest approach; credited to the other ship like a hit."""

import math

import torch

from lean.game import card, core

torch.set_num_threads(1)
CPU = torch.device("cpu")
CONFIG = core.GameConfig(field_count=0, fog=False)
H = CONFIG.tick_length


def placed(positions, headings, speeds=None, shields=None, sizes=None):
    """One environment, ship i at positions[i] flying headings[i] (radians) at speeds[i]
    (default the spawn speed), straight and level; team sizes `sizes` (default an even split,
    the first half team 0)."""
    count = len(positions)
    sizes = sizes or [count // 2, count - count // 2]
    state = core.reset(CONFIG, 1, torch.tensor([sizes]), torch.Generator().manual_seed(0), CPU)
    angle = torch.tensor([headings], dtype=torch.float32)
    changes = dict(
        x=torch.tensor([[p[0] for p in positions]], dtype=torch.float32),
        y=torch.tensor([[p[1] for p in positions]], dtype=torch.float32),
        heading_cos=angle.cos(),
        heading_sin=angle.sin(),
        angle=torch.zeros(1, count),
        angle_cos=torch.ones(1, count),
        angle_sin=torch.zeros(1, count),
        nose_rate=torch.zeros(1, count),
    )
    if speeds is not None:
        changes["speed"] = torch.tensor([speeds], dtype=torch.float32)
    if shields is not None:
        changes["shield"] = torch.tensor([shields], dtype=torch.float32)
    return core.with_velocity(state.replace(**changes))


def swept(start, end, nose):
    """core.collisions on straight moves from `start` to `end` (one tick) with noses `nose`."""
    state = placed(start, [0.0] * len(start))
    tensor = lambda values: torch.tensor([values], dtype=torch.float32)  # noqa: E731
    return core.collisions(
        state,
        state.alive,
        H,
        tensor([p[0] for p in end]),
        tensor([p[1] for p in end]),
        tensor([math.cos(a) for a in nose]),
        tensor([math.sin(a) for a in nose]),
    )


def constant(name):
    return core.ship_constants(CONFIG, torch.tensor([[0]]))[name]


def smootherstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def blame_of(cosine):
    """The rule written out: full blame inside the card's inner nose angle, 0.5 beyond the
    outer, a smootherstep in the cosine between."""
    inner = math.cos(card.CARD["nose_inner_angle"][0])
    outer = math.cos(card.CARD["nose_outer_angle"][0])
    return 0.5 + 0.5 * smootherstep((cosine - outer) / (inner - outer))


def first_touch(start, end, me, other):
    """The line of centres (me to the other) when the gap first equals the two hit radii, by
    solving |p + w t| = R for the smaller root."""
    reach = 2.0 * constant("hit_radius")
    p = [start[other][k] - start[me][k] for k in range(2)]
    w = [(end[other][k] - start[other][k]) - (end[me][k] - start[me][k]) for k in range(2)]
    a = w[0] ** 2 + w[1] ** 2
    b = 2.0 * (p[0] * w[0] + p[1] * w[1])
    c = p[0] ** 2 + p[1] ** 2 - reach**2
    t = (-b - math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    return [(p[k] + w[k] * t) / reach for k in range(2)]


def reach():
    return 2.0 * constant("hit_radius")


def core_reach():
    return 2.0 * constant("core_radius")


def test_equal_speed_crossing_at_ninety_degrees_is_half_blame_each():
    """(a) A along +x and B along +y at equal speeds reach the origin together, starting just
    outside the two hit radii: at first touch each nose is 45 degrees off the line of centres
    (the outer edge of the zone), blame 0.5 each; the centres meet (closeness 1); the closing
    speed is sqrt(2) v."""
    half = reach() / math.sqrt(2.0) + 1.0
    step = 2.0 * half
    start, end = [(-half, 0.0), (0.0, -half)], [(half, 0.0), (0.0, half)]
    out = swept(start, end, [0.0, math.pi / 2])
    assert bool(out.contact[0, 0, 1]) and bool(out.contact[0, 1, 0])
    for me, other in ((0, 1), (1, 0)):
        assert abs(float(out.nose_cos[0, me, other]) - math.cos(math.pi / 4)) < 1e-6
        assert abs(float(out.blame[0, me, other]) - 0.5) < 1e-5
    closing = math.sqrt(2.0) * step / H
    assert abs(float(out.closing[0, 0, 1]) - closing) < 1e-5 * closing
    expected = constant("max_shield") * closing * constant("inverse_collision_speed") * 0.5
    assert abs(float(out.damage[0, 0, 1]) - expected) < 1e-5 * expected
    assert abs(float(out.damage[0, 1, 0]) - expected) < 1e-5 * expected


def test_the_t_bone_blames_the_nose_that_hits_the_side():
    """(b) A at the origin flying +x slowly; B from below flying +y into A's side: A is about 90
    degrees off the line (blame 0.5), B about 0 (blame 1); the cosines are the first touch's,
    solved independently."""
    start, end = [(0.0, 0.0), (0.5, -30.0)], [(1.0, 0.0), (0.5, 0.0)]
    nose = [0.0, math.pi / 2]
    out = swept(start, end, nose)
    assert bool(out.contact[0, 0, 1])
    for me, other in ((0, 1), (1, 0)):
        line = first_touch(start, end, me, other)
        cosine = math.cos(nose[me]) * line[0] + math.sin(nose[me]) * line[1]
        assert abs(float(out.nose_cos[0, me, other]) - cosine) < 1e-5
        assert abs(float(out.blame[0, me, other]) - blame_of(cosine)) < 1e-5
    assert abs(float(out.nose_cos[0, 0, 1])) < math.sin(math.radians(5.0))
    assert float(out.nose_cos[0, 1, 0]) > math.cos(math.radians(5.0))
    assert float(out.blame[0, 0, 1]) == 0.5 and float(out.blame[0, 1, 0]) == 1.0


def run(state, sizes=None):
    action = torch.zeros(state.exists.shape + (3,), dtype=torch.long)
    return core.step(CONFIG, state, action, {"spread": torch.zeros(state.x.shape + (2,))})


def test_head_on_at_cruise_costs_a_full_shield_each_before_the_clamp():
    """(c) Head-on at cruise, the centres meeting inside the tick: each ship takes a full
    shield's worth (full blame, closeness 1, closing at twice cruise), credited to the other;
    the shields go to 0 and neither dies (each had a shield)."""
    cruise = 0.5 / constant("inverse_collision_speed")
    step = cruise * H
    half = 0.5 * reach() + 0.1 * step  # outside the two hit radii, touching early in the tick
    state = placed([(-half, 0.0), (half, 0.0)], [0.0, math.pi], speeds=[cruise, cruise])
    new, out = run(state)
    shield = constant("max_shield")
    for me, other in ((0, 1), (1, 0)):
        assert abs(float(out.damage_taken[0, me]) - shield) < 2e-3 * shield
        assert float(out.hit_damage[0, other, me]) == float(out.damage_taken[0, me])
    assert new.shield[0].tolist() == [0.0, 0.0]
    assert out.died[0].tolist() == [False, False]


def scores(path_me, path_other):
    """Ordered-pair contacts per tick for straight moves between successive positions."""
    counted = []
    for tick in range(len(path_me) - 1):
        out = swept(
            [path_me[tick], path_other[tick]],
            [path_me[tick + 1], path_other[tick + 1]],
            [0.0, math.pi],
        )
        counted.append(int(out.contact[0, 0, 1]) + int(out.contact[0, 1, 0]))
    return counted


def test_a_straight_pass_is_scored_once_whatever_its_phase():
    """(d) A flies +x past B (still, 5 m off the line) over five ticks, the pass at a fraction f
    of a tick from a tick boundary (f = 0, inside, exactly on a boundary): one first touch in
    all, never two and never none."""
    step = 20.0
    for fraction in (0.0, 0.25, 0.999, 1.0, 1.001, 1.5, 1.999):
        a = [(-(2.0 + fraction) * step + k * step, 0.0) for k in range(6)]
        counted = scores(a, [(0.0, 5.0)] * 6)
        assert sum(counted) == 2, (fraction, counted)  # one pass, both ordered pairs


def old_rule_scores(gap, move):
    """The rule before 2026-11-26, written out: scored when the closest approach of the tick's
    straight relative move falls in it (0 <= tau < 1) inside the two hit radii."""
    tau = -(gap[0] * move[0] + gap[1] * move[1]) / (move[0] ** 2 + move[1] ** 2)
    s = min(max(tau, 0.0), 1.0)
    return 0.0 <= tau < 1.0 and math.hypot(gap[0] + s * move[0], gap[1] + s * move[1]) < reach()


def test_a_fast_pass_with_both_ships_turning_is_scored_exactly_once():
    """Both ships turn between the two ticks of a fast pass (the relative velocity rotates), so
    the closest approach of the straight moves falls in both ticks (the old rule scores twice)
    or in neither (the old rule misses). First touch scores it once, on the first tick."""
    for bend in (1.0, -1.0):  # +1: the old rule scores twice; -1: it misses
        me_moves = [(-15.0, -2.5 * bend), (-15.0, 2.5 * bend)]
        other_moves = [(15.0, 2.5 * bend), (15.0, -2.5 * bend)]
        middle = (0.0, 10.0)  # the gap after the first tick, inside the two radii
        relative = [(o[0] - m[0], o[1] - m[1]) for m, o in zip(me_moves, other_moves)]
        gaps = [(middle[0] - relative[0][0], middle[1] - relative[0][1]), middle]
        old = [old_rule_scores(gap, move) for gap, move in zip(gaps, relative)]
        assert old == ([True, True] if bend > 0 else [False, False])
        me = [(0.0, 0.0)]
        for move in me_moves:
            me.append((me[-1][0] + move[0], me[-1][1] + move[1]))
        other = [(me[0][0] + gaps[0][0], me[0][1] + gaps[0][1])]
        for move in other_moves:
            other.append((other[-1][0] + move[0], other[-1][1] + move[1]))
        closing = math.hypot(*relative[0]) / H
        assert closing > 100.0  # a fast pass
        assert scores(me, other) == [2, 0], bend


def test_no_score_while_overlapping_and_a_new_one_only_after_separating():
    """Ships that start a tick inside the two hit radii are never scored; a pair is scored
    again only after a tick that starts at or beyond them. Gaps along x at the tick starts."""
    R = reach()

    def along(gaps):
        return scores([(0.0, 0.0)] * len(gaps), [(g, 0.0) for g in gaps])

    assert along([0.5 * R, 0.2 * R, 0.6 * R]) == [0, 0]  # spawned overlapping
    assert along([R + 5.0, 0.4 * R, 0.8 * R, 0.4 * R]) == [2, 0, 0]  # no separation
    assert along([R + 5.0, 0.4 * R, R + 5.0, 0.4 * R]) == [2, 0, 2]  # separated, then again
    assert along([R, 0.5 * R]) == [2]  # at the radii at tick start counts as outside
    assert along([0.5 * R, R, 0.5 * R]) == [0, 2]  # ending at the radii is a separation


def test_closeness_full_when_the_cores_would_meet_partial_for_a_graze():
    """Damage is max shield x closing / (2 x cruise) x graze(b) x blame, b the closest distance
    of the tick's relative line extended past the tick: inside the two core radii full, between
    them and the two hit radii a smoothstep (written out here), the line's b and not the tick's
    own nearest distance."""
    R, R_core = reach(), core_reach()
    for offset in (0.5 * R_core, R_core + 0.3 * (R - R_core), R_core + 0.8 * (R - R_core)):
        # A flies +x towards B (still, `offset` off the line) and the tick ends before the
        # closest approach: the tick's own nearest distance is above `offset`.
        start, end = [(-R - 5.0, 0.0), (0.0, offset)], [(-0.25 * R, 0.0), (0.0, offset)]
        out = swept(start, end, [0.0, math.pi])
        assert bool(out.contact[0, 0, 1]) and bool(out.contact[0, 1, 0])
        assert math.hypot(end[0][0], offset) > offset + 0.5
        band = min(max((offset - R_core) / (R - R_core), 0.0), 1.0)
        closeness = 1.0 - band * band * (3.0 - 2.0 * band)
        assert abs(float(out.miss[0, 0, 1]) - offset) < 1e-4
        if offset < R_core:
            assert closeness == 1.0
        else:
            assert 0.0 < closeness < 1.0
        for me, other in ((0, 1), (1, 0)):
            scale = constant("max_shield") * constant("inverse_collision_speed")
            expected = scale * float(out.closing[0, me, other]) * closeness
            expected *= float(out.blame[0, me, other])
            assert abs(float(out.damage[0, me, other]) - expected) < 1e-5 * expected


def test_collision_damage_is_credited_to_the_other_ship_team_mates_included():
    """(e) Two team-mates collide (the enemy far away): each one's damage is in hit_damage as
    dealt by the other, and a ship already at 0 shield dies of it, as of a bullet."""
    speed = 200.0
    half = 0.5 * reach() + 0.5
    state = placed(
        [(-half, 3.0), (half, -3.0), (3000.0, 0.0)],
        [0.0, math.pi, 0.0],
        speeds=[speed] * 3,
        shields=[100.0, 0.0, 100.0],
        sizes=[2, 1],
    )
    assert int(state.team[0, 0]) == int(state.team[0, 1])
    new, out = run(state)
    rule = core.collisions(state, state.alive, H, new.x, new.y, new.heading_cos, new.heading_sin)
    taken = rule.damage[0].sum(-1)
    assert float(taken[0]) > 0.0 and float(taken[1]) > 0.0 and float(taken[2]) == 0.0
    assert torch.allclose(out.damage_taken[0], taken)
    assert float(out.hit_damage[0, 1, 0]) == float(rule.damage[0, 0, 1])
    assert float(out.hit_damage[0, 0, 1]) == float(rule.damage[0, 1, 0])
    assert out.died[0].tolist() == [False, True, False]
    # Zero sum: every point of collision damage is credited to some ship.
    assert abs(float(out.hit_damage.sum()) - float(out.damage_taken.sum())) < 1e-4


def test_no_contact_for_a_wide_pass_a_later_one_or_a_dead_ship():
    """A pass wider than the two hit radii, one that touches only after the tick, and one with
    a ship not alive at tick start score nothing; a ship never collides with itself."""
    reach = 2.0 * constant("hit_radius")
    wide = swept([(-10, 0), (10, reach + 1)], [(10, 0), (-10, reach + 1)], [0.0, math.pi])
    later = swept([(-30, 0), (30, 0)], [(-20, 0), (20, 0)], [0.0, math.pi])
    assert not bool(wide.contact.any()) and not bool(later.contact.any())
    state = placed([(-10.0, 0.0), (10.0, 0.0)], [0.0, math.pi])
    dead = state.alive.clone()
    dead[0, 1] = False
    end = torch.tensor([[10.0, -10.0]])
    out = core.collisions(state, dead, H, end, torch.zeros(1, 2), *torch.ones(2, 1, 2))
    assert not bool(out.contact.any())
    out = core.collisions(state, state.alive, H, end, torch.zeros(1, 2), *torch.ones(2, 1, 2))
    assert not bool(torch.diagonal(out.contact, dim1=-2, dim2=-1).any())
