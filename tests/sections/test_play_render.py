"""The renderer offscreen (`bnb.play.render`): the index texture, the fog mask, and pixel
checks on rendered frames. Needs an OpenGL 3.3 context (EGL)."""

import dataclasses
import math

import numpy as np
import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game.modes.rules import PRESETS
from bnb.game.perception.sight import core_membership, sighted
from bnb.game.physics.medium import medium
from bnb.play import themes
from bnb.play.app import Options, read_pixels, run_offscreen
from bnb.play.camera import Camera
from bnb.play.index_map import build
from bnb.play.loop import MatchLoop
from bnb.play.render.renderer import SPECTATOR, Renderer
from bnb.play.render.ships import ring_radius
from bnb.play.theme_editor import with_colour
from bnb.play.themes import Colour
from bnb.play.trails import Trails

pytestmark = pytest.mark.gpu
WIDTH, HEIGHT = 320, 240


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """One CPU thread: B = 1 ticks gain nothing from more, and lose much on a busy machine."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def context():
    moderngl = pytest.importorskip("moderngl")
    try:
        context = moderngl.create_context(standalone=True, backend="egl", require=330)
    except Exception as error:  # no GPU or no EGL here
        pytest.skip(f"no OpenGL 3.3 context: {error}")
    yield context
    context.release()


@pytest.fixture(scope="module")
def imgui_context():
    from imgui_bundle import imgui

    created = imgui.create_context()
    imgui.set_current_context(created)
    yield
    imgui.destroy_context(created)


@pytest.fixture(scope="module")
def renderer(context, imgui_context):
    return Renderer(context, WIDTH, HEIGHT)


def _target(context):
    return context.framebuffer(color_attachments=[context.texture((WIDTH, HEIGHT), 4)])


def _loop(config, fogged=True, ticks=0, seed=0):
    torch.manual_seed(seed)
    loop = MatchLoop(config, fogged=fogged, compile=False, pilot=None)
    for _ in range(ticks):
        loop.step()
    return loop


def _draw(renderer, context, frame, view, camera, theme):
    grid, values = build(frame.fields, frame.arena_radius)
    renderer.set_map(frame, grid, values)
    target = _target(context)
    trails = Trails(len(frame.ships.x), frame.stall_angle)
    renderer.draw(frame, view, camera, theme, trails, 0.0, target)
    assert context.error == "GL_NO_ERROR"
    return read_pixels(target).astype(int)


def _pixel_of(camera: Camera, x: float, y: float) -> tuple[int, int]:
    """(row, column) of a world point."""
    column = (x - camera.centre_x) / camera.metres_per_pixel + 0.5 * camera.width
    row = 0.5 * camera.height - (y - camera.centre_y) / camera.metres_per_pixel
    return int(row), int(column)


def test_index_texture_equals_medium_at_sampled_points(renderer):
    loop = _loop(PRESETS["frontline"])
    frame = loop.current
    grid, values = build(frame.fields, frame.arena_radius)
    renderer.set_map(frame, grid, values)
    uploaded = np.frombuffer(renderer.fields.index_texture.read(), np.float32)
    uploaded = uploaded.reshape(grid.size, grid.size)
    generator = torch.Generator().manual_seed(4)
    column, row = torch.randint(0, grid.size, (2, 50_000), generator=generator)
    near = torch.from_numpy(values != 0.0).nonzero()[:5000]  # many inside the fields
    row, column = torch.cat([row, near[:, 0]]), torch.cat([column, near[:, 1]])
    x = (grid.origin + (column + 0.5) * grid.texel).float()
    y = (grid.origin + (row + 0.5) * grid.texel).float()
    want = medium(loop.realtime.state.fields, x[None], y[None])[0][0]
    got = torch.from_numpy(uploaded.copy())[row, column]
    assert torch.allclose(got, want, atol=1e-5)
    assert (want != 0).sum() > 1000


@pytest.mark.parametrize("config", [PRESETS["deathmatch"], GameConfig(12, 30, 5000.0)])
def test_fog_mask_agrees_with_sighted(renderer, config):
    loop = _loop(config, ticks=30, seed=5)
    frame, state = loop.current, loop.realtime.state
    camera = Camera(WIDTH, HEIGHT)
    camera.fit(frame.arena_radius)
    ships, fields = frame.ships, frame.fields
    team = 0
    own = ships.alive & (ships.team == team)
    observers = np.stack([ships.x[own], ships.y[own], ships.vision_range[own],
                          np.zeros(int(own.sum()))], -1)
    renderer.set_map(frame, *build(fields, frame.arena_radius))
    renderer.fog.draw_mask(camera.uniforms(), observers, renderer.cores)
    width, height = renderer.fog.size
    mask = np.frombuffer(renderer.fog.mask.read(), np.uint8).reshape(height, width) > 127

    # The game's sight test at the mask's texel centres (row 0 at the bottom).
    half_width = 0.5 * camera.width * camera.metres_per_pixel
    half_height = 0.5 * camera.height * camera.metres_per_pixel
    x = camera.centre_x + ((np.arange(width) + 0.5) / width * 2 - 1) * half_width
    y = camera.centre_y + ((np.arange(height) + 0.5) / height * 2 - 1) * half_height
    target_x = torch.from_numpy(np.tile(x, height)).float()[None, None].expand(1, 2, -1)
    target_y = torch.from_numpy(np.repeat(y, width)).float()[None, None].expand(1, 2, -1)
    radius = torch.from_numpy(fields.core_radius)[None]
    cores = (state.fields.x, state.fields.y, radius * radius)
    S = loop.rules.ships_per_team
    observer_x, observer_y = state.ships.x.reshape(1, 2, S), state.ships.y.reshape(1, 2, S)
    seen = sighted(cores, (observer_x, observer_y, core_membership(cores, observer_x, observer_y),
                           state.ships.alive.reshape(1, 2, S),
                           torch.from_numpy(ships.vision_range).reshape(1, 2, S)),
                   (target_x, target_y, core_membership(cores, target_x, target_y)))
    want = seen[0, team].numpy().reshape(height, width)
    # Compare away from the mask's edges: cells whose neighbours all agree with them.
    padded = np.pad(want, 1, mode="edge")
    neighbours = np.stack([padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    interior = (neighbours == want).all(0)
    assert interior.mean() > 0.8
    assert want[interior].any() and not want[interior].all()  # both sides are tested
    assert np.array_equal(mask[interior], want[interior])


def test_a_ship_draws_in_its_team_colour(renderer, context):
    loop = _loop(PRESETS["deathmatch"], ticks=10)
    frame = loop.current
    theme = themes.load(themes.DEFAULT)
    ships = frame.ships
    distance = np.hypot(ships.x[:, None] - ships.x, ships.y[:, None] - ships.y)
    distance[np.diag_indices_from(distance)] = np.inf
    chosen = int(np.argmax(distance.min(1) * ships.alive))  # the most isolated ship
    camera = Camera(WIDTH, HEIGHT, float(ships.x[chosen]), float(ships.y[chosen]), 0.2)
    pixels = _draw(renderer, context, frame, SPECTATOR, camera, theme)
    size = float(ring_radius(ships.hit_radius[chosen:chosen + 1], 0.2)[0])
    ahead = 0.2 * size  # inside the silhouette, forward of the centre
    row, column = _pixel_of(camera, ships.x[chosen] + ahead * math.cos(ships.heading[chosen]),
                            ships.y[chosen] + ahead * math.sin(ships.heading[chosen]))
    colour = np.array(theme.team[int(ships.team[chosen])].floats()[:3]) * 255
    assert np.abs(pixels[row, column] - colour).max() <= 1


def test_fog_darkens_what_the_team_cannot_see(renderer, context):
    loop = _loop(PRESETS["deathmatch"], ticks=10)
    frame = loop.current
    theme = themes.load(themes.DEFAULT)
    camera = Camera(WIDTH, HEIGHT)
    camera.fit(frame.arena_radius)
    spectator = _draw(renderer, context, frame, SPECTATOR, camera, theme)
    team = _draw(renderer, context, frame, 0, camera, theme)
    ships = frame.ships
    own = ships.alive & (ships.team == 0)
    # A point in the arena farther than any team-0 ship's vision range.
    for angle in np.linspace(0, 2 * math.pi, 64, endpoint=False):
        x, y = (0.9 * frame.arena_radius * f(angle) for f in (math.cos, math.sin))
        if np.hypot(ships.x[own] - x, ships.y[own] - y).min() > 1.2 * ships.vision_range[0]:
            break
    else:
        pytest.fail("no unseen point in the scenario")
    row, column = _pixel_of(camera, x, y)
    fog = np.array(theme.fog.floats()[:3])
    assert np.allclose(team[row, column], spectator[row, column] * fog, atol=2)
    assert team[row, column].sum() < spectator[row, column].sum()
    # A team-0 ship's own spot is not fogged.
    seen = int(np.flatnonzero(own)[0])
    beside = 3 * ring_radius(ships.hit_radius, camera.metres_per_pixel)[seen]
    row, column = _pixel_of(camera, ships.x[seen] + beside, ships.y[seen])
    assert np.abs(team[row, column] - spectator[row, column]).max() <= 2


def test_a_hidden_enemy_draws_nothing_in_its_enemys_view(renderer, context):
    loop = _loop(PRESETS["deathmatch"], ticks=0, seed=1)
    for _ in range(300):
        loop.step()
        ships = loop.current.ships
        hidden = ships.alive & (ships.team == 1) & ~ships.visible[0]
        if hidden.any():
            break
    frame = loop.current
    enemy = int(np.flatnonzero(hidden)[0])
    theme = themes.load(themes.DEFAULT)
    camera = Camera(WIDTH, HEIGHT, float(frame.ships.x[enemy]), float(frame.ships.y[enemy]),
                    0.5)
    seen = _draw(renderer, context, frame, 0, camera, theme)
    alive = frame.ships.alive.copy()
    alive[enemy] = False
    without = dataclasses.replace(frame, ships=dataclasses.replace(frame.ships, alive=alive))
    assert np.array_equal(seen, _draw(renderer, context, without, 0, camera, theme))
    spectator = _draw(renderer, context, frame, SPECTATOR, camera, theme)
    assert not np.array_equal(spectator, _draw(renderer, context, without, SPECTATOR, camera,
                                               theme))


def test_an_edited_colour_draws_on_the_next_frame(renderer, context):
    loop = _loop(PRESETS["deathmatch"], ticks=0)
    frame = loop.current
    camera = Camera(WIDTH, HEIGHT)
    camera.fit(frame.arena_radius)
    theme = themes.load(themes.DEFAULT)
    edited = with_colour(theme, "sky", Colour.from_hex("#204060"))
    before = _draw(renderer, context, frame, SPECTATOR, camera, theme)
    after = _draw(renderer, context, frame, SPECTATOR, camera, edited)
    # A sky pixel: inside the arena, away from every field and ship.
    fields, ships = frame.fields, frame.ships
    for angle in np.linspace(0, 2 * math.pi, 90, endpoint=False):
        for fraction in (0.3, 0.5, 0.7):
            x = fraction * frame.arena_radius * math.cos(angle)
            y = fraction * frame.arena_radius * math.sin(angle)
            clear_of_fields = (np.hypot(fields.x - x, fields.y - y)
                               > fields.radius + fields.band_width + 50).all()
            clear_of_ships = (np.hypot(ships.x - x, ships.y - y) > 200).all()
            if clear_of_fields and clear_of_ships:
                row, column = _pixel_of(camera, x, y)
                assert np.array_equal(before[row, column],
                                      np.array(theme.sky.floats()[:3]) * 255)
                assert np.array_equal(after[row, column], [0x20, 0x40, 0x60])
                return
    pytest.fail("no clear sky in the scenario")


def test_offscreen_frames_are_written(tmp_path):
    paths = run_offscreen(Options(offscreen=tmp_path, frames=3, width=WIDTH, height=HEIGHT,
                                  compile=False, preset="frontline", fog=True, view="team1"))
    assert [p.name for p in paths] == ["frame_0000.png", "frame_0001.png", "frame_0002.png"]
    assert all(p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in paths)


def test_a_hex_code_typed_into_the_editor_applies(context, tmp_path):
    """Click the sky's hex field, select it, type a code: the next frame draws with it."""
    from imgui_bundle import imgui

    from bnb.play.app import Session
    from bnb.play.theme_editor import PANEL_WIDTH

    width, height = 1280, 720
    target = context.framebuffer(color_attachments=[context.texture((width, height), 4)])
    session = Session(Options(compile=False), context, width, height)
    session.update(1.0 / 60.0)
    session.editor.open = True
    session.draw(target)
    session.draw(target)  # a new panel takes its size on its second frame
    io, style = imgui.get_io(), imgui.get_style()
    row = imgui.get_font_size() + 2 * style.frame_padding.y
    io.add_mouse_pos_event(width - PANEL_WIDTH - 10 + style.window_padding.x + 30,
                           10 + row + style.window_padding.y + 0.5 * row)
    for event in (lambda: io.add_mouse_button_event(0, True),
                  lambda: io.add_mouse_button_event(0, False),
                  lambda: (io.add_key_event(imgui.Key.mod_ctrl, True),
                           io.add_key_event(imgui.Key.a, True)),
                  lambda: (io.add_key_event(imgui.Key.a, False),
                           io.add_key_event(imgui.Key.mod_ctrl, False)),
                  lambda: io.add_input_characters_utf8("#204060"),
                  lambda: io.add_key_event(imgui.Key.enter, True),
                  lambda: io.add_key_event(imgui.Key.enter, False)):
        event()
        session.draw(target)
    assert session.theme.sky == Colour.from_hex("#204060")
    assert session.theme.thin_air == themes.load(themes.DEFAULT).thin_air
    session.close()
