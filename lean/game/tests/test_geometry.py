"""The relative-geometry functions the learner's geometry heads take their targets from
(lean.game.core "relative geometry"): each reads the game the way the step and the observation
do, so the targets have one source."""

import math

import torch

from lean.contract import BULLET_SLOTS
from lean.game import core

torch.set_num_threads(1)
CONFIG = core.GameConfig()
DEVICE = torch.device("cpu")


def played(envs=16, ticks=40, seed=0, config=CONFIG):
    """A state some ticks into random play (2v2), so ships move, turn and see each other."""
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.tensor([[2, 2]]).expand(envs, 2)
    state = core.reset(config, envs, sizes, generator, DEVICE)
    for _ in range(ticks):
        action = torch.stack(
            [
                torch.randint(0, 7, (envs, 4), generator=generator),
                torch.randint(0, 3, (envs, 4), generator=generator),
                torch.randint(0, 2, (envs, 4), generator=generator),
            ],
            -1,
        )
        state, _ = core.tick(config, state, action, generator)
    return state


def rows(state, first, second):
    """pair_geometry of slot `first` on slot `second` from the true state."""
    record = core.geometry_record(CONFIG, state)
    position = torch.stack([state.x, state.y], -1)
    heading = torch.atan2(state.heading_sin, state.heading_cos)
    return core.pair_geometry(
        position[:, first],
        heading[:, first],
        record[:, first],
        position[:, second],
        record[:, second],
    )


def test_range_rates_are_symmetric_and_the_energy_difference_antisymmetric():
    """Catches an order-dependent symmetric target (the learner trains those on the upper
    triangle only) or an energy difference that does not change sign with the order."""
    state = played()
    forward, reverse = rows(state, 0, 2), rows(state, 2, 0)
    for name in ("range", "range_rate", "line_of_sight_rate"):
        assert torch.allclose(forward[name], reverse[name], rtol=1e-5, atol=1e-5), name
    assert torch.allclose(forward["energy_difference"], -reverse["energy_difference"])
    assert not torch.allclose(forward["train_angle"], reverse["train_angle"])


def test_range_rate_is_the_change_of_range():
    """Catches a range rate or line-of-sight rate off in sign or units: both match the change
    of the range and of the line of sight's bearing over one tick, within the tick's
    curvature."""
    state = played(ticks=30)
    before = rows(state, 0, 2)
    line = torch.stack([state.x[:, 2] - state.x[:, 0], state.y[:, 2] - state.y[:, 0]], -1)
    after_state, _ = core.tick(
        CONFIG, state, torch.zeros(state.x.shape + (3,), dtype=torch.long), None
    )
    after = rows(after_state, 0, 2)
    alive = state.alive[:, 0] & state.alive[:, 2] & after_state.alive[:, 0]
    alive &= after_state.alive[:, 2] & (before["range"] > 300.0)
    h = CONFIG.tick_length
    rate = (after["range"] - before["range"]) / h
    mean = 0.5 * (before["range_rate"] + after["range_rate"])
    assert alive.sum() > 4
    assert torch.allclose(rate[alive], mean[alive], rtol=0.02, atol=1.0)
    line_after = torch.stack(
        [after_state.x[:, 2] - after_state.x[:, 0], after_state.y[:, 2] - after_state.y[:, 0]], -1
    )
    turned = torch.atan2(
        line[:, 0] * line_after[:, 1] - line[:, 1] * line_after[:, 0],
        (line * line_after).sum(-1),
    )
    mean_rate = 0.5 * (before["line_of_sight_rate"] + after["line_of_sight_rate"])
    assert torch.allclose((turned / h)[alive], mean_rate[alive], rtol=0.05, atol=2e-3)


def test_train_angle_is_from_the_nose_to_the_line_of_sight():
    """Catches a train angle measured from the wrong axis or with the wrong sign: dead ahead
    0, a target to the left (counter-clockwise) +90 degrees, astern pi."""
    state = played(envs=1, ticks=1)
    record = core.geometry_record(CONFIG, state)[:, :2]
    heading = torch.tensor([[0.3, 0.0]])
    for offset, expected in ((0.0, 0.0), (0.5 * math.pi, 0.5 * math.pi), (math.pi, math.pi)):
        angle = 0.3 + offset
        position = torch.tensor([[[0.0, 0.0], [500.0 * math.cos(angle), 500.0 * math.sin(angle)]]])
        geometry = core.pair_geometry(
            position[:, 0], heading[:, 0], record[:, 0], position[:, 1], record[:, 1]
        )
        error = math.remainder(float(geometry["train_angle"][0]) - expected, 2.0 * math.pi)
        assert abs(error) < 1e-5


def test_a_bullet_fired_along_the_gun_solution_meets_its_target():
    """Catches a lead point that is not the game's bullet's: a bullet launched by the game's
    own motion (`bullet_motion`, no fields) along the launch direction turned by gun_error,
    at the launch speed, passes the target flying straight at its
    world velocity (within 0.5 m; the hit half width is 2.4 m), for targets inside the gun
    range (placed 150 to 550 m out)."""
    open_map = core.GameConfig(field_count=0)
    state = played(envs=64, ticks=20, seed=3, config=open_map)
    generator = torch.Generator().manual_seed(4)
    distance = 150.0 + 400.0 * torch.rand(64, generator=generator)
    bearing = 2.0 * math.pi * torch.rand(64, generator=generator)
    x, y = state.x.clone(), state.y.clone()
    x[:, 2] = x[:, 0] + distance * torch.cos(bearing)
    y[:, 2] = y[:, 0] + distance * torch.sin(bearing)
    state = state.replace(x=x, y=y)
    geometry = rows(state, 0, 2)
    record = core.geometry_record(CONFIG, state)
    column = {name: index for index, name in enumerate(core.GEOMETRY_COLUMNS)}
    launch_x = record[:, 0, column["launch_x"]]
    launch_y = record[:, 0, column["launch_y"]]
    speed = torch.sqrt(launch_x * launch_x + launch_y * launch_y)
    turn = geometry["gun_error"]
    direction_x = (launch_x * torch.cos(turn) - launch_y * torch.sin(turn)) / speed
    direction_y = (launch_x * torch.sin(turn) + launch_y * torch.cos(turn)) / speed
    x, y = state.x[:, 0].clone(), state.y[:, 0].clone()
    target_x, target_y = state.x[:, 2].clone(), state.y[:, 2].clone()
    velocity_x, velocity_y = record[:, 2, column["velocity_x"]], record[:, 2, column["velocity_y"]]
    none = torch.zeros(state.x.shape[0], 5, 0)
    zero = torch.zeros_like(x)
    closest = torch.full_like(x, float("inf"))
    h = CONFIG.tick_length
    drag = record[:, 0, column["bullet_drag"]]
    for _ in range(BULLET_SLOTS):
        start_x, start_y, start_tx, start_ty = x, y, target_x, target_y
        x, y, direction_x, direction_y, speed, *_ = core.bullet_motion(
            none, h, x, y, direction_x, direction_y, speed, drag, (zero, zero, zero)
        )
        target_x, target_y = target_x + velocity_x * h, target_y + velocity_y * h
        # The closest approach within the tick, both moving straight.
        relative_x, relative_y = start_x - start_tx, start_y - start_ty
        move_x = (x - start_x) - (target_x - start_tx)
        move_y = (y - start_y) - (target_y - start_ty)
        along = (
            -(relative_x * move_x + relative_y * move_y)
            / (move_x * move_x + move_y * move_y).clamp_min(1e-9)
        ).clamp(0.0, 1.0)
        gap = torch.hypot(relative_x + along * move_x, relative_y + along * move_y)
        closest = torch.minimum(closest, gap)
    assert closest.max() < 0.5  # 3 cm at most when written (2026-11-28)


def test_the_geometry_record_reads_the_step_and_the_observation():
    """Catches a record column that drifts from the game: the launch velocity is the fire
    stage's without the spread (over the map), the velocity is the observation's world
    velocity, the edge is minus the observation's boundary distance."""
    state = played()
    record = core.geometry_record(CONFIG, state)
    column = {name: index for index, name in enumerate(core.GEOMETRY_COLUMNS)}
    index = core.slot_indices(state.x.shape[1], DEVICE)
    shots = core._fire(
        CONFIG, state, torch.ones_like(state.alive), torch.zeros(state.x.shape + (2,)), index
    )
    slot = int(state.tick[0]) % BULLET_SLOTS
    bullets = shots.bullets[..., slot]
    launch_speed = bullets[:, core.BULLET_SPEED] * torch.exp(-state.log_index)
    assert torch.allclose(
        record[..., column["launch_x"]], bullets[:, core.BULLET_DIR_X] * launch_speed, atol=1e-2
    )
    assert torch.allclose(
        record[..., column["launch_y"]], bullets[:, core.BULLET_DIR_Y] * launch_speed, atol=1e-2
    )
    observation = core.observe(CONFIG, state, 0)
    own = observation.own
    seen = core.symlog(record[..., column["velocity_x"]] / core.SPEED_SCALE)
    assert torch.allclose(seen[own], observation.ship[..., 0][own], atol=1e-5)
    boundary = observation.ship[..., 12]  # the observation's clipped boundary scalar
    expected = (-record[..., column["edge"]] / core.BOUNDARY_CLIP).clamp(-1.0, 1.0)
    assert torch.allclose(boundary[own], expected[own], atol=1e-6)
    assert torch.allclose(
        record[..., column["bullet_life"]], torch.tensor(BULLET_SLOTS * CONFIG.tick_length)
    )


def test_a_fields_edge_is_the_middle_of_its_band():
    """Catches a field distance measured to the wrong circle: 0 where a lone field's medium is
    half its level (the band's middle), negative inside, positive outside."""
    fields = torch.tensor([[[100.0], [-50.0], [400.0], [160.0], [0.7]]])
    angle = torch.tensor([[0.0, 1.0, 2.0]])
    for radius, sign in ((400.0, 0.0), (300.0, -1.0), (600.0, 1.0)):
        x = 100.0 + radius * torch.cos(angle)
        y = -50.0 + radius * torch.sin(angle)
        distance = core.field_edge_distance(
            fields[:, 0, :1], fields[:, 1, :1], fields[:, 2, :1], x[..., None], y[..., None]
        )[..., 0]
        assert torch.allclose(distance, torch.full_like(distance, radius - 400.0), atol=1e-3)
        if sign == 0.0:
            log_index, _, _ = core.medium(fields, x, y)
            assert torch.allclose(log_index, torch.full_like(log_index, 0.35), atol=1e-5)


def test_visible_rows_are_the_observations():
    """Catches the learner's next-tick visibility differing from what the next view shows."""
    state = played(ticks=60, seed=5)
    for team in (0, 1):
        assert torch.equal(
            core.visible_rows(CONFIG, state, team), core.observe(CONFIG, state, team).visible
        )
    assert not core.visible_rows(CONFIG, state, 0).all()  # fog hides some enemy
