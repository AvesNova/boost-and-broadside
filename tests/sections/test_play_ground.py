"""What grounds the view where nothing else is near: the look-ahead camera's lead and spring
(`bnb.play.camera`), the dot grid's spacing levels (`bnb.play.render.grid`) and the options
that choose them. No GPU: the grid's drawing is tested in `test_play_render.py`, the
controls in `test_play_interface.py`."""

import math

import pytest
import tyro

from bnb.play.app import Options
from bnb.play.camera import (
    LEAD_CAP,
    LEAD_FREQUENCY,
    LEAD_SECONDS,
    Camera,
    LookAhead,
    lead_target,
    spring_step,
)
from bnb.play.render.grid import GRID_PIXELS, PATTERNS, grid_level

WIDTH, HEIGHT = 1280, 720
CAP = LEAD_CAP * 0.5 * HEIGHT  # pixels


def test_the_lead_is_along_the_velocity_and_none_at_rest():
    x, y = lead_target(30.0, -40.0, 2.0, WIDTH, HEIGHT)
    assert x > 0.0 and y < 0.0
    assert math.isclose(x * -40.0, y * 30.0, rel_tol=1e-12)  # parallel
    assert lead_target(0.0, 0.0, 2.0, WIDTH, HEIGHT) == (0.0, 0.0)


def test_the_lead_grows_with_speed_up_to_the_cap():
    """Slow: LEAD_SECONDS of travel on screen; fast: just under the cap, never over."""
    metres_per_pixel = 4.0
    slow = math.hypot(*lead_target(1.0, 0.0, metres_per_pixel, WIDTH, HEIGHT))
    assert math.isclose(slow, LEAD_SECONDS / metres_per_pixel, rel_tol=1e-3)
    lengths = [
        math.hypot(*lead_target(speed, 0.0, metres_per_pixel, WIDTH, HEIGHT))
        for speed in (10.0, 50.0, 100.0, 200.0, 400.0, 1e4)
    ]
    assert lengths == sorted(lengths) and all(length <= CAP for length in lengths)
    assert lengths[-1] > 0.999 * CAP


def _settle(target: float, seconds: float, frame_rate: float) -> list[float]:
    position, rate, path = 0.0, 0.0, []
    for _ in range(round(seconds * frame_rate)):
        position, rate = spring_step(position, rate, target, 1.0 / frame_rate)
        path.append(position)
    return path


def test_the_spring_converges_without_overshoot():
    path = _settle(100.0, 6.0, 60.0)
    assert all(later >= earlier for earlier, later in zip(path, path[1:], strict=False))
    assert max(path) <= 100.0 and path[-1] > 99.9
    assert 40.0 < path[59] < 70.0  # about 1 - 3 e^-2 of the way after 1 s


def test_the_spring_does_not_depend_on_the_frame_rate():
    """The same second at 30, 60 and 144 frames per second ends at the same place."""
    ends = {rate: _settle(100.0, 1.0, rate)[-1] for rate in (30.0, 60.0, 144.0)}
    assert max(ends.values()) - min(ends.values()) < 1e-9
    exact = 100.0 * (1.0 - (1.0 + LEAD_FREQUENCY) * math.exp(-LEAD_FREQUENCY))
    assert math.isclose(ends[60.0], exact, rel_tol=1e-9)


def _fly(lead: LookAhead, camera: Camera, frame_rate: float, seconds: float, on=True):
    """A ship flying at (150, 0) m/s from the origin; returns the last offset (m)."""
    offset = (0.0, 0.0)
    for index in range(round(seconds * frame_rate) + 1):
        time = index / frame_rate
        offset = lead.update(0, 150.0 * time, 0.0, time, 1.0 / frame_rate, camera, on)
    return offset


def test_the_look_ahead_leads_a_flying_ship_the_same_at_any_frame_rate():
    camera = Camera(WIDTH, HEIGHT, metres_per_pixel=4.0)
    expected = lead_target(150.0, 0.0, 4.0, WIDTH, HEIGHT)[0] * 4.0
    offsets = [_fly(LookAhead(), camera, rate, 8.0) for rate in (30.0, 60.0, 144.0)]
    for x, y in offsets:
        assert abs(y) < 1e-9 and math.isclose(x, expected, rel_tol=1e-3)
    assert max(x for x, _ in offsets) - min(x for x, _ in offsets) < 1e-3 * expected


def test_the_look_ahead_starts_over_for_a_new_ship_or_a_spawn_and_eases_back_when_off():
    camera = Camera(WIDTH, HEIGHT, metres_per_pixel=4.0)
    lead = LookAhead()
    led = _fly(lead, camera, 60.0, 4.0)[0]
    assert led > 0.0
    assert lead.update(1, 0.0, 0.0, 10.0, 1 / 60, camera, True) == (0.0, 0.0)  # a new ship
    _fly(lead, camera, 60.0, 4.0)
    assert lead.update(0, 9e3, 0.0, 4.0 + 1 / 60, 1 / 60, camera, True, spawned=True) == (0, 0)
    lead = LookAhead()
    _fly(lead, camera, 60.0, 4.0)
    released = _fly(lead, camera, 60.0, 6.0, on=False)[0]  # the same ship, now not leading
    assert 0.0 <= released < 0.01 * led


def test_a_zoom_keeps_the_lead_where_it_is_on_screen():
    camera = Camera(WIDTH, HEIGHT, metres_per_pixel=4.0)
    lead = LookAhead()
    x, _ = _fly(lead, camera, 60.0, 4.0)
    camera.metres_per_pixel = 2.0
    zoomed, _ = lead.update(0, 150.0 * 4.0, 0.0, 4.0, 0.0, camera, True)  # no time between
    assert math.isclose(zoomed / 2.0, x / 4.0, rel_tol=1e-9)


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
def test_the_grid_spacing_steps_by_the_patterns_factor_and_keeps_its_density(pattern):
    """The fine spacing is a power of the factor, between GRID_PIXELS / factor and
    GRID_PIXELS on screen, and fades from 0 to 1 across a level."""
    factor = PATTERNS[pattern][1]
    for metres_per_pixel in (0.05, 0.3, 1.0, 3.9, 17.0, 46.8):
        spacing, fade = grid_level(pattern, metres_per_pixel)
        exponent = math.log(spacing, factor)
        assert abs(exponent - round(exponent)) < 1e-9
        assert GRID_PIXELS / factor < spacing / metres_per_pixel <= GRID_PIXELS * (1 + 1e-9)
        assert 0.0 <= fade <= 1.0
    at_level = factor**3 / GRID_PIXELS  # the fine spacing exactly GRID_PIXELS on screen
    assert grid_level(pattern, at_level) == (pytest.approx(factor**3), pytest.approx(0.0))


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
def test_the_grid_crosses_a_level_without_a_jump(pattern):
    """Just before a level change the fine dots have faded out and the coarse ones show;
    just after, those coarse dots are the new fine level, at full strength."""
    factor = PATTERNS[pattern][1]
    boundary = factor**4 / GRID_PIXELS
    before_spacing, before_fade = grid_level(pattern, boundary * (1 - 1e-7))
    after_spacing, after_fade = grid_level(pattern, boundary * (1 + 1e-7))
    assert before_fade > 1 - 1e-6 and after_fade < 1e-6
    assert math.isclose(before_spacing * factor, after_spacing)


def test_the_grid_and_look_ahead_are_options():
    options = tyro.cli(Options, args=["--grid", "hex", "--look-ahead", "flying"])
    assert options.grid == "hex" and options.look_ahead == "flying"
    defaults = Options()
    assert defaults.grid in PATTERNS and defaults.look_ahead == "always"
    with pytest.raises(SystemExit):
        tyro.cli(Options, args=["--grid", "hexagon"])
