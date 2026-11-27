"""The bullet deflector (proposal 2026-11-27, the egg: core.egg_entry, core.egg_turn,
core._deflect) and the hit model's runtime inputs (core.HIT_MODEL, core.with_hit_model).
Every number is read from the code."""

import dataclasses
import math

import pytest
import torch

from lean.contract import BULLET_SLOTS, CONTACTED, GHOST, REAL
from lean.game import card, core

torch.set_num_threads(1)
CPU = torch.device("cpu")
SLOT = 3  # the test bullet's slot in the shooter's ring


def crossing(config, offsets, aspects, start_x=-30.0, real=True, target_speed=1.0):
    """E = len(offsets) environments: a target at the origin with its nose at `aspects` (the
    bullet's path off the nose, radians; 0 head-on), nearly still, and the shooter's bullet at
    (start_x, offset) relative to it flying +x at the bullet launch speed: the bullet passes the
    target at `offset` this tick. The shooter sits far away, not firing. The tick is the one
    after the bullet's slot's, so the bullet has BULLET_SLOTS - 1 ticks to fly."""
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
    bullet_state = state.bullet_state.clone()
    bullet_state[:, 0, SLOT] = REAL if real else GHOST
    return state.replace(
        bullets=bullets,
        bullet_state=bullet_state,
        tick=torch.full((envs,), SLOT + 1, dtype=torch.long),
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


def heading(direction_x, direction_y):
    return torch.atan2(direction_y, direction_x)


OFF = core.GameConfig(field_count=0, fog=False, deflector="off")
EGG = dataclasses.replace(OFF, deflector="egg")
# A start well outside the egg on every aspect, the egg's entry within the first tick's move.
FAR = -(model(EGG)["egg_front"] + 10.0)


def sketch():
    """The design sketch's ray tracer (docs/rewrite/sections/bullet-deflector), set to the
    game's egg, turn curve and largest turn."""
    import sys
    from pathlib import Path

    folder = Path(__file__).resolve().parents[3] / "docs/rewrite/sections/bullet-deflector"
    sys.path.insert(0, str(folder))
    try:
        import deflector_humps_sketch as humps
        import deflector_profiles_sketch as profiles
    finally:
        sys.path.remove(str(folder))
    egg = model(EGG)
    profiles.EDGE = humps.EDGE = egg["egg_half_width"]
    shape = [(0.0, egg["egg_front"], egg["egg_half_width"], +1)]
    shape.append((0.0, egg["egg_rear"], egg["egg_half_width"], -1))
    act = profiles.profile_turn(
        humps.CURVES["early hump, 27/4 x (1 - x)^2"], math.degrees(egg["deflector_turn"])
    )
    return humps, shape, act


def test_the_turn_curve_is_the_early_hump():
    """hump: 0 at both ends and beyond them, its peak 1 at a third."""
    x = torch.tensor([-0.5, 0.0, 1.0 / 3.0, 1.0, 1.5])
    assert torch.allclose(core.hump(x), torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]), atol=1e-6)
    assert float(core.hump(torch.tensor(0.2))) < 1.0 and float(core.hump(torch.tensor(0.5))) < 1.0


def test_exactly_one_turn_per_pass():
    """Over a whole pass flown tick by tick, wherever in a tick the egg's entry falls, a bullet
    passing through the egg off centre is turned on exactly one tick; one passing outside the
    egg, or dead centre, never."""
    half_width = model(EGG)["egg_half_width"]
    starts = torch.linspace(FAR - 70.0, FAR, 25)
    for offset, aspect, expected in (
        (0.3 * half_width, 0.5 * math.pi, 1),
        (-0.2 * half_width, 0.0, 1),
        (0.4 * half_width, math.pi, 1),
        (half_width + 3.0, 0.5 * math.pi, 0),
        (0.0, 0.0, 0),  # head-on: the target's own move is along the line, b stays 0
        (0.0, math.pi, 0),
    ):
        state = crossing(EGG, [offset] * len(starts), [aspect] * len(starts), start_x=FAR)
        state = state.replace(bullets=_shifted(state.bullets, starts))
        turns = torch.zeros(len(starts), dtype=torch.long)
        before = bullet(state)
        for after, _ in run(EGG, state, ticks=6):
            now = bullet(after)
            alive = after.bullet_state[:, 0, SLOT] != CONTACTED
            changed = (now[2] - before[2]).abs() + (now[3] - before[3]).abs() > 1e-6
            turns += (changed & (alive | (turns == 0))).long()
            before = now
        assert turns.tolist() == [expected] * len(starts), (offset, aspect)


def _shifted(bullets, starts):
    bullets = bullets.clone()
    bullets[:, core.BULLET_X, 0, SLOT] = starts
    return bullets


def test_the_turn_keeps_the_speed():
    """No energy added in the relative frame: a turned bullet's speed relative to the ship
    (its average velocity over the tick) equals the unturned one's; its direction stays a unit
    vector."""
    offsets = torch.linspace(-30.0, 30.0, 41)
    aspects = torch.linspace(0.0, math.pi, 41)
    start = crossing(EGG, offsets, aspects, start_x=FAR, target_speed=150.0)
    ((on, _),) = run(EGG, start)
    ((off, _),) = run(OFF, crossing(OFF, offsets, aspects, start_x=FAR, target_speed=150.0))
    tick = EGG.tick_length
    ship_x = (on.x[:, 1] - start.x[:, 1]) / tick
    ship_y = (on.y[:, 1] - start.y[:, 1]) / tick

    def relative_speed(state):
        *_, direction_x, direction_y, speed = bullet(state)
        return torch.hypot(direction_x * speed - ship_x, direction_y * speed - ship_y)

    assert torch.allclose(relative_speed(on), relative_speed(off), rtol=1e-5)
    *_, direction_x, direction_y, _ = bullet(on)
    assert torch.allclose(direction_x**2 + direction_y**2, torch.ones_like(direction_x), atol=1e-6)
    assert bool((direction_y.abs() > 1e-3).any())  # some were turned


def test_dead_centre_goes_straight_and_hits():
    """A bullet aimed at the centre does the full damage (inside the core) at every aspect;
    head-on and from astern (the target's own move along the line) it is not turned at all,
    from the side by at most the hump's slope at the target's sideways move this tick."""
    aspects = torch.linspace(0.0, math.pi, 9)
    offsets = torch.zeros_like(aspects)
    states = run(EGG, crossing(EGG, offsets, aspects, start_x=FAR, target_speed=0.0), ticks=2)
    damage = sum(out.hit_damage[:, 0, 1] for _, out in states)
    plain = sum(
        out.hit_damage[:, 0, 1]
        for _, out in run(
            OFF, crossing(OFF, offsets, aspects, start_x=FAR, target_speed=0.0), ticks=2
        )
    )
    assert bool((plain > 0.0).all())
    assert torch.allclose(damage, plain, rtol=1e-5)
    first, _ = states[0]
    turn = bullet(first)[3].abs()
    assert float(turn[0]) < 1e-8 and float(turn[-1]) < 1e-8  # (float rounding of cos pi)
    sideways = float((first.x[:, 1] ** 2 + first.y[:, 1] ** 2).sqrt().max())
    egg = model(EGG)
    assert float(turn.max()) <= 6.75 * egg["deflector_turn"] * sideways / egg["egg_half_width"]


def test_a_bullet_from_inside_the_egg_is_not_turned():
    """A bullet starting inside the egg (fired from inside, or already turned) never enters it,
    so flies straight: the same block as with the deflector off."""
    offsets = torch.tensor([6.0, -6.0, 10.0])
    aspects = torch.tensor([0.5 * math.pi, 0.25 * math.pi, 0.0])
    start = -0.5 * model(EGG)["egg_rear"]
    ((on, _),) = run(EGG, crossing(EGG, offsets, aspects, start_x=start))
    ((off, _),) = run(OFF, crossing(OFF, offsets, aspects, start_x=start))
    assert torch.allclose(on.bullets, off.bullets)


def test_ghosts_turn_like_real_bullets():
    """A ghost's flight is a real bullet's: the same bullet block after the tick."""
    offsets = torch.linspace(-30.0, 30.0, 41)
    aspects = torch.linspace(0.0, math.pi, 41)
    ((real, _),) = run(EGG, crossing(EGG, offsets, aspects, real=True, start_x=FAR))
    ((ghost, _),) = run(EGG, crossing(EGG, offsets, aspects, real=False, start_x=FAR))
    assert torch.equal(real.bullets, ghost.bullets)
    assert torch.equal(real.bullet_state == CONTACTED, ghost.bullet_state == CONTACTED)


def test_the_turn_matches_the_design_sketch():
    """The game's turned direction and hit-or-miss on a nearly still ship (its own move this
    tick, a few centimetres, shifts the relative line: the turn agrees within 0.01 rad) equal
    the design sketch's ray tracer (trace_through, the early hump, the egg) for rays across
    the egg."""
    humps, shape, act = sketch()
    np = pytest.importorskip("numpy")
    rays = [
        (a, o)
        for a in (0.0, 0.25 * math.pi, 0.5 * math.pi, 0.75 * math.pi, math.pi)
        for o in (-17.0, -9.0, -4.0, -1.5, 2.0, 5.0, 8.0, 14.0, 21.0)
    ]
    aspects = torch.tensor([a for a, _ in rays])
    offsets = torch.tensor([o for _, o in rays])
    states = run(EGG, crossing(EGG, offsets, aspects, start_x=FAR, target_speed=0.0), ticks=3)
    after, _ = states[0]
    hit = sum(out.hit_damage[:, 0, 1] for _, out in states) > 0.0
    game_turn = heading(*bullet(after)[2:4])
    for i, (aspect, offset) in enumerate(rays):
        # The ray in the target's frame (nose +x): the bullet flies +x in the world, the nose at
        # pi - aspect, so in the ship's frame the bullet flies at angle aspect - pi.
        nose = math.pi - aspect
        cos, sin = math.cos(-nose), math.sin(-nose)
        d = np.array([cos, sin])
        p = np.array([FAR * cos - offset * sin, FAR * sin + offset * cos])
        points, sketch_hit = humps.trace_through(p, d, shape, act)
        out = points[-1] - points[-2]
        sketch_turn = math.atan2(out[1], out[0]) - math.atan2(d[1], d[0])
        sketch_turn = (sketch_turn + math.pi) % (2 * math.pi) - math.pi
        assert abs(float(game_turn[i]) - sketch_turn) < 1e-2, (aspect, offset)
        assert bool(hit[i]) == bool(sketch_hit), (aspect, offset)


def test_the_hit_widths_match_the_design_sketch():
    """The hit widths (core.hit_widths, the agent's table) equal the sketch's at head-on, side
    and from astern, within the scan's step; 2 x the hit radius with no turn."""
    humps, shape, act = sketch()
    radius = card_value("hit_radius")
    widths = core.hit_widths(torch.tensor([radius]), model(EGG))[0]
    for degrees in (0.0, 90.0, 180.0):
        hit_cos = torch.tensor(math.cos(math.radians(degrees)))
        game = float(core.hit_width(widths, hit_cos))
        assert abs(game - humps.hit_width(shape, act, degrees)) < 0.15, degrees
    plain = core.hit_widths(torch.tensor([radius]), model(OFF))[0]
    assert torch.allclose(plain, torch.full_like(plain, 2.0 * radius), atol=0.11)


def test_hit_model_round_trip_and_unknown_names():
    """with_hit_model writes the inputs (and rebuilds the hit widths); hit_model_of reads them
    back; an unknown name raises."""
    state = core.reset(EGG, 2, torch.tensor([[1, 1]]).expand(2, 2), None, CPU)
    values = dict(
        hit_radius=14.0,
        core_radius=4.0,
        hit_damage=9.0,
        bullet_spread=2.0,
        egg_front=60.0,
        egg_rear=20.0,
        egg_half_width=30.0,
        deflector_turn=0.6,
        nose_cone=1.0,
        nose_inner_angle=0.3,
        nose_outer_angle=0.7,
    )
    assert set(values) == {entry.name for entry in core.HIT_MODEL}
    changed = core.with_hit_model(state.constants, **values)
    read = core.hit_model_of(changed)
    for name, value in values.items():
        assert abs(read[name] - value) < 1e-5, name
    assert torch.equal(changed.table[:, changed.columns["hit_radius"][0]], torch.full((2, 2), 14.0))
    expected = core.hit_widths(torch.tensor([14.0]), values)[0]
    assert torch.allclose(changed["hit_widths"][0, :, 0], expected)
    with pytest.raises(KeyError):
        core.with_hit_model(state.constants, gun_range=1.0)


def test_a_hit_model_change_needs_no_recompile():
    """The compiled step (whole graph) runs with a changed hit model (every input moved) on the
    same compiled graph, and the change takes effect (the damage follows the new rows)."""
    from torch._dynamo.testing import CompileCounter

    torch.compiler.reset()
    counter = CompileCounter()
    compiled = torch.compile(core.step, backend=counter, fullgraph=True)
    state = crossing(EGG, [0.0, 0.0], [0.5 * math.pi] * 2, start_x=-30.0)
    idle = torch.zeros(2, 2, 3, dtype=torch.long)
    noise = {"spread": torch.zeros(2, 2, 2)}
    _, first = compiled(EGG, state, idle, noise)
    assert float(first.hit_damage[:, 0, 1].min()) > 0.0
    moved = {entry.name: entry.low + 0.25 * (entry.high - entry.low) for entry in core.HIT_MODEL}
    moved.update(nose_inner_angle=0.3, nose_outer_angle=0.6)
    changed = state.replace(constants=core.with_hit_model(state.constants, **moved))
    with torch._dynamo.config.patch(error_on_recompile=True):
        _, second = compiled(EGG, changed, idle, noise)
    assert counter.frame_count == 1
    assert not torch.allclose(second.hit_damage, first.hit_damage)
    eager = core.step(EGG, changed, idle, noise)[1]
    assert torch.allclose(second.damage_taken, eager.damage_taken)
    assert BULLET_SLOTS > SLOT + 1
