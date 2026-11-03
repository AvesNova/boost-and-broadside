"""Selection and the interface's geometry, on the CPU: the hit test picks the right
object and respects the team view; the minimap's camera mapping round-trips; the layout
keeps its cards inside the screen and apart."""

import dataclasses

import numpy as np
import pytest

from bnb.contracts import GameConfig
from bnb.play.camera import Camera
from bnb.play.interface.layout import CARDS, Layout
from bnb.play.interface.minimap import to_minimap, to_world, view_rect
from bnb.play.loop import MatchLoop, MatchSetup
from bnb.play.render.ships import ring_radius
from bnb.play.selection import SLACK_PIXELS, Selection, hit_test


@pytest.fixture(scope="module")
def frame():
    loop = MatchLoop(MatchSetup(GameConfig(3, 4, 2600.0, mode="frontline")), compiled=False)
    return loop.current


def _placed(frame, ships_x, ships_y, visible=None, alive=None):
    ships = frame.ships
    count = len(ships.x)
    ships = dataclasses.replace(
        ships,
        x=np.asarray(ships_x, np.float32),
        y=np.asarray(ships_y, np.float32),
        alive=np.ones(count, bool) if alive is None else np.asarray(alive),
        visible=np.ones((2, count), bool) if visible is None else np.asarray(visible),
    )
    return dataclasses.replace(frame, ships=ships)


def _far(frame) -> tuple[float, float]:
    """A point in no field and no zone."""
    rng = np.random.default_rng(0)
    for _ in range(10000):
        x, y = rng.uniform(-2500, 2500, 2)
        if hit_test(frame, -1, x, y, 1.0) is None:
            return x, y
    raise AssertionError("no empty sky")


def test_the_hit_test_picks_the_nearest_ship_within_its_drawn_radius(frame):
    count = len(frame.ships.x)
    x, y = _far(frame)
    xs = np.full(count, 99999.0)
    ys = np.full(count, 99999.0)
    xs[2], ys[2] = x, y
    xs[4], ys[4] = x + 30.0, y
    placed = _placed(frame, xs, ys)
    pixel = 1.0
    assert hit_test(placed, -1, x + 1.0, y, pixel) == Selection("ship", 2)
    assert hit_test(placed, -1, x + 25.0, y, pixel) == Selection("ship", 4)
    reach = ring_radius(placed.ships.hit_radius[2:3], pixel)[0] + SLACK_PIXELS * pixel
    assert hit_test(placed, -1, x - reach + 0.5, y, pixel) == Selection("ship", 2)
    assert hit_test(placed, -1, x - reach - 0.5, y, pixel) is None
    zoomed_out = 20.0  # m per pixel: the drawn ring (14 px) is far larger than the hitbox
    assert hit_test(placed, -1, x - 200.0, y, zoomed_out) == Selection("ship", 2)


def test_the_hit_test_respects_the_team_view(frame):
    count = len(frame.ships.x)
    x, y = _far(frame)
    xs, ys = np.full(count, 99999.0), np.full(count, 99999.0)
    enemy = count - 1  # a team-1 ship
    xs[enemy], ys[enemy] = x, y
    visible = np.ones((2, count), bool)
    visible[0, enemy] = False
    placed = _placed(frame, xs, ys, visible=visible)
    assert hit_test(placed, -1, x, y, 1.0) == Selection("ship", enemy)  # the spectator
    assert hit_test(placed, 1, x, y, 1.0) == Selection("ship", enemy)  # its own team
    assert hit_test(placed, 0, x, y, 1.0) is None  # team 0 cannot see it
    dead = _placed(frame, xs, ys, alive=np.zeros(count, bool))
    assert hit_test(dead, -1, x, y, 1.0) is None


def test_the_hit_test_falls_back_to_the_zone_then_the_field(frame):
    zones, fields = frame.zones, frame.fields
    away = _placed(
        frame, np.full(len(frame.ships.x), 99999.0), np.full(len(frame.ships.x), 99999.0)
    )
    for zone in np.flatnonzero(zones.present):
        assert hit_test(away, 0, zones.x[zone], zones.y[zone], 1.0) == Selection("zone", int(zone))
    for field in range(len(fields.x)):
        inside_zone = np.hypot(zones.x - fields.x[field], zones.y - fields.y[field]) <= zones.radius
        if not (inside_zone & zones.present).any():
            got = hit_test(away, 1, fields.x[field], fields.y[field], 1.0)
            assert got == Selection("field", field)


def test_the_minimap_mapping_round_trips():
    layout = Layout(1920, 1080)
    square = layout.minimap_map
    minimap = Camera(int(square.width), int(square.width))
    minimap.fit(2600.0)
    rng = np.random.default_rng(1)
    for px, py in rng.uniform(0, 1, (50, 2)) * square.width + (square.x, square.y):
        assert np.allclose(
            to_minimap(square, minimap, *to_world(square, minimap, px, py)), (px, py)
        )
    for x, y in rng.uniform(-2600, 2600, (50, 2)):
        assert np.allclose(
            to_world(square, minimap, *to_minimap(square, minimap, x, y)), (x, y), atol=1e-6
        )
    assert np.allclose(to_minimap(square, minimap, 0.0, 0.0), square.centre())
    camera = Camera(1920, 1080)
    camera.fit(2600.0)
    camera.follow(500.0, -300.0)
    view = view_rect(square, minimap, camera)
    assert np.allclose(to_world(square, minimap, *view.centre()), (500.0, -300.0), atol=1e-3)
    assert view.width == pytest.approx(1920 * camera.metres_per_pixel / minimap.metres_per_pixel)


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080), (2560, 1440)])
@pytest.mark.parametrize("scale", [1.0, 1.5])
@pytest.mark.parametrize("replay", [False, True])
def test_the_layout_keeps_its_cards_on_screen_and_apart(size, scale, replay):
    width, height = size
    for open_cards in (frozenset(CARDS), frozenset()):
        layout = Layout(width, height, scale, open_cards, replay=replay, flying=True)
        for rect in layout.rects():
            assert 0 <= rect.x and rect.right <= width and 0 <= rect.y and rect.bottom <= height
        side = [layout.fleets, layout.selection, layout.minimap, layout.top_bar]
        side += [layout.scrubber] if replay else []
        for index, first in enumerate(side):
            for second in side[index + 1 :]:
                apart = (
                    first.right <= second.x
                    or second.right <= first.x
                    or first.bottom <= second.y
                    or second.bottom <= first.y
                )
                assert apart, (first, second)
    centre = Layout(width, height, scale)
    assert not centre.covers(0.5 * width, 0.5 * height)  # the middle stays battlefield
