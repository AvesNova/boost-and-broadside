"""The bullet deflector (proposal 2026-11-27, core.deflection, core._deflect) and the hit model's
runtime inputs (core.HIT_MODEL, core.with_hit_model). Every number is read from the code."""

import dataclasses
import math

import pytest
import torch

from lean.contract import BULLET_SLOTS
from lean.game import card, core

torch.set_num_threads(1)
CPU = torch.device("cpu")
SLOT = 3  # the test bullet's slot in the shooter's ring


def crossing(config, offsets, aspects, start_x=-30.0, real=True, moves=5, target_speed=1.0):
    """E = len(offsets) environments: a target at the origin with its nose at `aspects` (the
    bullet's path off the nose, radians; 0 head-on), nearly still, and the shooter's bullet at
    (start_x, offset) relative to it flying +x at the bullet launch speed: the bullet passes the
    target at `offset` this tick. The shooter sits far away, not firing."""
    offsets = torch.as_tensor(offsets, dtype=torch.float32)
    aspects = torch.as_tensor(aspects, dtype=torch.float32)
    envs = len(offsets)
    state = core.reset(
        config, envs, torch.tensor([[1, 1]]).expand(envs, 2), torch.Generator().manual_seed(0), CPU
    )
    # The bullet flies +x; the target's nose points back along it, turned by the aspect.
    nose = math.pi - aspects
    zeros, ones = torch.zeros(envs, 2), torch.ones(envs, 2)
    state = core.with_velocity(
        state.replace(
            x=torch.stack([torch.full((envs,), -3000.0), torch.zeros(envs)], 1),
            y=torch.zeros(envs, 2),
            heading_cos=torch.stack([torch.ones(envs), torch.cos(nose)], 1),
            heading_sin=torch.stack([torch.zeros(envs), torch.sin(nose)], 1),
            speed=torch.stack([torch.full((envs,), 200.0), torch.full((envs,), target_speed)], 1),
            angle=zeros,
            angle_cos=ones,
            angle_sin=zeros,
            nose_rate=zeros,
        )
    )
    speed = float(torch.as_tensor(state.constants["bullet_launch_speed"]).flatten()[0])
    bullets = state.bullets.clone()
    bullets[:, :, 0, SLOT] = 0.0
    bullets[:, core.BULLET_X, 0, SLOT] = start_x
    bullets[:, core.BULLET_Y, 0, SLOT] = offsets
    bullets[:, core.BULLET_DIR_X, 0, SLOT] = 1.0
    bullets[:, core.BULLET_SPEED, 0, SLOT] = speed
    counts = state.bullet_counts.clone()
    counts[:, core.MOVES_LEFT, 0, SLOT] = moves
    bullet_real = state.bullet_real.clone()
    bullet_real[:, 0, SLOT] = real
    next_slot = torch.full((envs, 2), float(SLOT + 1))
    return state.replace(
        bullets=bullets, bullet_counts=counts, bullet_real=bullet_real, next_slot=next_slot
    )


def run(config, state, ticks=1):
    """Step coasting straight with no trigger and no spread; per tick (state, out)."""
    envs = state.x.shape[0]
    idle = torch.zeros(envs, 2, 3, dtype=torch.long)
    states = []
    for _ in range(ticks):
        state, out = core.step(
            config, state.replace(pending=idle), idle, {"spread": torch.zeros(envs, 2, 2)}
        )
        states.append((state, out))
    return states


def bullet(state):
    """The test bullet's x, y, direction x, y, speed (E,) each."""
    return [state.bullets[:, column, 0, SLOT] for column in range(core.BULLET_SPEED + 1)]


def model(config):
    return card.DEFLECTOR_PRESETS[config.deflector]


def card_value(name):
    return float(card.CARD[name][0])


OFF = core.GameConfig(field_count=0, fog=False, deflector="off")
SYMMETRIC = dataclasses.replace(OFF, deflector="symmetric")
FRONTAL = dataclasses.replace(OFF, deflector="frontal")


def test_presets_hold_the_frontal_rule():
    """The frontal preset's front D is at least the hit radius (every frontal bullet glances
    off) and replaces the nose cone's rule; the symmetric preset has one D at every aspect."""
    frontal, symmetric = card.DEFLECTOR_PRESETS["frontal"], card.DEFLECTOR_PRESETS["symmetric"]
    assert frontal["deflector_front"] >= card_value("hit_radius")
    assert frontal["nose_cone"] == 0.0
    assert symmetric["deflector_front"] == symmetric["deflector_side"] > 0.0
    off = card.DEFLECTOR_PRESETS["off"]
    assert off["deflector_front"] == off["deflector_side"] == 0.0 and off["nose_cone"] == 1.0


@pytest.mark.parametrize("config", [SYMMETRIC, FRONTAL], ids=lambda c: c.deflector)
def test_a_pass_is_pushed_out_by_d_and_damage_reads_the_pushed_distance(config):
    """Off the nose (from the side, from astern: D the side value): a bullet passing at b does
    the damage of a pass at b + D (the graze on the pushed distance, compared with the
    deflector off at b + D); a bullet that misses after the push is bent: its path's point at
    the pass lies b + D_eff from the target, on the side it was passing, D_eff = min(D, R + D
    - b)."""
    side = model(config)["deflector_side"]
    radius, core_radius = card_value("hit_radius"), card_value("core_radius")
    hits = [0.5 * core_radius, 0.5 * (core_radius + radius) - side, -(radius - side) + 0.5]
    aspect = [0.5 * math.pi] * len(hits)
    ((_, out_on),) = run(config, crossing(config, hits, aspect))
    pushed = [math.copysign(abs(b) + side, b) for b in hits]
    ((_, out_off),) = run(OFF, crossing(OFF, pushed, aspect))
    on, off = out_on.hit_damage[:, 0, 1], out_off.hit_damage[:, 0, 1]
    assert bool((off > 0.0).all())
    assert torch.allclose(on, off, rtol=1e-4, atol=1e-4)

    misses = [radius - side + 1.0, -(radius - side + 2.0), radius + 0.5 * side, 0.4 * radius]
    misses = [b for b in misses if abs(b) + side >= radius]
    aspect = [math.pi] * len(misses)  # from astern: the target's slow move is along the path
    start = crossing(config, misses, aspect)
    ((after, out),) = run(config, start)
    assert float(out.hit_damage.sum()) == 0.0
    x, y, direction_x, direction_y, _ = bullet(after)
    target_x, target_y = after.x[:, 1], after.y[:, 1]
    # Back from the end along the bent path by the rest of the move (the target barely moves).
    rest = x - target_x
    push_x = x - rest * direction_x / direction_x - target_x
    push_y = y - rest * direction_y / direction_x - target_y
    b = torch.tensor(misses)
    expected = torch.minimum(torch.full_like(b, side), radius + side - b.abs())
    assert torch.allclose(push_x, torch.zeros_like(push_x), atol=0.02)
    assert torch.allclose(push_y, b + torch.sign(b) * expected, atol=0.02)
    # Turned away from the ship by atan(D_eff / length).
    turn = torch.atan2(direction_y, direction_x)
    angle = torch.atan(expected / model(config)["deflector_length"])
    assert torch.allclose(turn, torch.sign(b) * angle, atol=1e-4)


@pytest.mark.parametrize("config", [SYMMETRIC, FRONTAL], ids=lambda c: c.deflector)
def test_the_bend_keeps_the_speed(config):
    """No energy added: a bent bullet's speed equals the unbent one's; its direction stays a
    unit vector."""
    offsets = torch.linspace(-20.0, 20.0, 41)
    aspects = torch.linspace(0.0, math.pi, 41)
    ((on, _),) = run(config, crossing(config, offsets, aspects))
    ((off, _),) = run(OFF, crossing(OFF, offsets, aspects))
    *_, direction_x, direction_y, speed = bullet(on)
    assert torch.equal(speed, bullet(off)[4])
    assert torch.allclose(direction_x**2 + direction_y**2, torch.ones_like(speed), atol=1e-6)
    assert bool((direction_y != 0.0).any())  # some were bent


@pytest.mark.parametrize("config", [SYMMETRIC, FRONTAL], ids=lambda c: c.deflector)
def test_exactly_one_bend_per_pass(config):
    """Over a whole pass flown tick by tick, whatever part of a tick the closest approach falls
    in, a bullet passing inside the field (and missing after the push) is bent on exactly one
    tick, and one passing outside it never."""
    radius, side = card_value("hit_radius"), model(config)["deflector_side"]
    starts = torch.linspace(-140.0, -80.0, 25)
    for offset, expected in ((radius - side + 0.5, 1), (radius + side + 5.0, 0)):
        state = crossing(
            config,
            [offset] * len(starts),
            [0.5 * math.pi] * len(starts),
            moves=8,
        )
        state = state.replace(bullets=_shifted(state.bullets, starts))
        bends = torch.zeros(len(starts), dtype=torch.long)
        before = bullet(state)
        for after, _ in run(config, state, ticks=6):
            now = bullet(after)
            bends += ((now[2] != before[2]) | (now[3] != before[3])).long()
            before = now
        assert bends.tolist() == [expected] * len(starts), offset


def _shifted(bullets, starts):
    bullets = bullets.clone()
    bullets[:, core.BULLET_X, 0, SLOT] = starts
    return bullets


@pytest.mark.parametrize("config", [SYMMETRIC, FRONTAL], ids=lambda c: c.deflector)
def test_ghosts_bend_like_real_bullets(config):
    """A ghost's flight is a real bullet's: the same bullet block after the tick."""
    offsets = torch.linspace(-20.0, 20.0, 41)
    aspects = torch.linspace(0.0, math.pi, 41)
    ((real, _),) = run(config, crossing(config, offsets, aspects, real=True))
    ((ghost, _),) = run(config, crossing(config, offsets, aspects, real=False))
    assert torch.equal(real.bullets, ghost.bullets)
    assert torch.equal(real.bullet_moves_left, ghost.bullet_moves_left)


def test_the_frontal_preset_takes_no_damage_from_the_front():
    """Frontal: every bullet within the front cone's angle, at any offset inside the hit radius,
    glances off: no damage and no contact (the bullet flies on); beyond the outer angle the
    same bullets do damage."""
    inner = model(FRONTAL)["deflector_inner_angle"]
    outer = model(FRONTAL)["deflector_outer_angle"]
    radius = card_value("hit_radius")
    generator = torch.Generator().manual_seed(3)
    count = 400
    offsets = (torch.rand(count, generator=generator) * 2.0 - 1.0) * radius
    for low, high, expect_damage in ((0.0, 0.999 * inner, False), (1.001 * outer, math.pi, True)):
        side = torch.where(torch.rand(count, generator=generator) < 0.5, 1.0, -1.0)
        aspects = side * (low + (high - low) * torch.rand(count, generator=generator))
        ((after, out),) = run(FRONTAL, crossing(FRONTAL, offsets, aspects))
        damage = out.hit_damage[:, 0, 1]
        if expect_damage:
            assert int((damage > 0.0).sum()) > count // 2
        else:
            assert float(damage.abs().max()) == 0.0
            assert bool((after.bullet_moves_left[:, 0, SLOT] > 0).all())  # flew on


def test_hit_model_round_trip_and_unknown_names():
    """with_hit_model writes the inputs; hit_model_of reads them back; an unknown name raises."""
    state = core.reset(SYMMETRIC, 2, torch.tensor([[1, 1]]).expand(2, 2), None, CPU)
    values = dict(
        hit_radius=14.0,
        core_radius=4.0,
        hit_damage=9.0,
        bullet_spread=2.0,
        deflector_front=11.0,
        deflector_side=2.5,
        deflector_inner_angle=0.4,
        deflector_outer_angle=0.9,
        deflector_length=60.0,
        nose_cone=0.0,
        nose_inner_angle=0.3,
        nose_outer_angle=0.7,
    )
    assert set(values) == {entry.name for entry in core.HIT_MODEL}
    changed = core.with_hit_model(state.constants, **values)
    read = core.hit_model_of(changed)
    for name, value in values.items():
        assert abs(read[name] - value) < 1e-5, name
    assert torch.equal(changed.table[:, changed.columns["hit_radius"][0]], torch.full((2, 2), 14.0))
    with pytest.raises(KeyError):
        core.with_hit_model(state.constants, gun_range=1.0)


def test_a_hit_model_change_needs_no_recompile():
    """The compiled step (whole graph) runs with a changed hit model (every input moved) on the
    same compiled graph, and the change takes effect: a bullet that hit no longer does."""
    from torch._dynamo.testing import CompileCounter

    torch.compiler.reset()
    counter = CompileCounter()
    compiled = torch.compile(core.step, backend=counter, fullgraph=True)
    radius, core_radius = card_value("hit_radius"), card_value("core_radius")
    state = crossing(SYMMETRIC, [0.5 * (radius + core_radius)] * 2, [0.5 * math.pi] * 2)
    idle = torch.zeros(2, 2, 3, dtype=torch.long)
    noise = {"spread": torch.zeros(2, 2, 2)}
    _, first = compiled(SYMMETRIC, state, idle, noise)
    assert float(first.hit_damage[:, 0, 1].min()) > 0.0
    moved = {entry.name: entry.low + 0.25 * (entry.high - entry.low) for entry in core.HIT_MODEL}
    moved.update(hit_radius=0.5 * (radius + core_radius) - 1.0, core_radius=1.0)
    changed = state.replace(constants=core.with_hit_model(state.constants, **moved))
    with torch._dynamo.config.patch(error_on_recompile=True):
        _, second = compiled(SYMMETRIC, changed, idle, noise)
    assert counter.frame_count == 1
    assert float(second.hit_damage.abs().max()) == 0.0
    eager = core.step(SYMMETRIC, changed, idle, noise)[1]
    assert torch.allclose(second.damage_taken, eager.damage_taken)
    assert BULLET_SLOTS > SLOT + 1
