"""The interface offscreen (`bnb.play.interface`), driven only through its controls:
synthetic mouse input on named controls, as a player would click them. Needs an OpenGL
3.3 context (EGL)."""

import numpy as np
import pytest
import torch
from play_cases import tiny_policy

from bnb.game import api
from bnb.game.api import PRESETS
from bnb.play import themes
from bnb.play.app import Options, Session, read_pixels
from bnb.play.interface.minimap import to_world
from bnb.play.loop import MatchEvent
from bnb.play.selection import Selection, shown_ships
from bnb.play.sides import Agent

pytestmark = pytest.mark.gpu
FRAME = 1.0 / 60.0


@pytest.fixture(autouse=True, scope="module")
def one_thread():
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


class Player:
    """Drives a session through ImGui's input, as the window would."""

    def __init__(self, session: Session, context, width: int, height: int) -> None:
        from imgui_bundle import imgui

        self.session, self.io = session, imgui.get_io()
        self.target = context.framebuffer(color_attachments=[context.texture((width, height), 4)])
        self.io.add_mouse_pos_event(-1.0, -1.0)
        self.frames(2)  # a new window takes its size on its second frame

    def frames(self, count: int = 1, elapsed: float = FRAME) -> None:
        for _ in range(count):
            self.session.update(elapsed)
            self.session.draw(self.target)

    def click_at(self, x: float, y: float) -> None:
        self.io.add_mouse_pos_event(x, y)
        self.frames()
        self.io.add_mouse_button_event(0, True)
        self.frames()
        self.io.add_mouse_button_event(0, False)
        self.frames()
        self.io.add_mouse_pos_event(-1.0, -1.0)
        self.frames(2)  # a window the click opened takes its size on its second frame

    def click(self, name: str) -> None:
        self.click_at(*self.session.interface.controls.rect(name).centre())

    def pixels(self) -> np.ndarray:
        return read_pixels(self.target).astype(int)


def _session(context, tmp_path, width=1280, height=720, **options) -> Session:
    options = Options(
        compile=False,
        width=width,
        height=height,
        replays=tmp_path / "replays",
        runs=tmp_path / "runs",
        **options,
    )
    return Session(options, context, width, height, library=themes.Library(tmp_path / "themes"))


def test_a_match_is_set_up_watched_flown_and_replayed_from_the_interface(context, tmp_path):
    """Spec 8.1: everything through the controls, no keys."""
    policy = tiny_policy(tmp_path / "runs" / "2026-11-03-tiny" / "policy", PRESETS["deathmatch"])
    session = _session(context, tmp_path)
    player = Player(session, context, 1280, 720)

    # Start a Frontline match.
    player.click("menu")
    player.click("menu/new match")
    player.click("new match/mode/frontline")
    player.click("new match/start")
    assert session.loop.setup.config.mode == "frontline" and not session.interface.menu.is_open
    player.frames(30)

    # Select a ship on the battlefield, take control, release.
    frame = session.frame
    ship = int(np.flatnonzero(shown_ships(frame, -1))[0])
    session.following = None
    player.click_at(*session.camera.pixel(float(frame.ships.x[ship]), float(frame.ships.y[ship])))
    assert session.selection == Selection("ship", ship)
    player.click("selection/take control")
    assert session.loop.pilot == ship
    player.frames(10)
    player.click("flying/release")
    assert session.loop.pilot is None

    # Pause, step, play, fast-forward.
    player.click("pause")
    assert session.loop.paused
    ticks = session.loop.ticks
    player.click("step")
    assert session.loop.ticks == ticks + 1
    player.click("play")
    player.click("speed/8")
    assert not session.loop.paused and session.loop.time_scale == 8.0

    # Change the view; the minimap's far corner moves the camera there.
    player.click("view/team0")
    assert session.view_name == "team0"
    square = session.interface.layout().minimap_map
    corner = (square.right - 3.0, square.bottom - 3.0)
    player.click_at(*corner)
    want = to_world(square, session.minimap_camera, *corner)
    assert (session.camera.centre_x, session.camera.centre_y) == pytest.approx(want, abs=5.0)

    # A new team deathmatch match with an agent on team 1.
    player.click("menu")
    player.click("menu/new match")
    player.click("new match/mode/team_deathmatch")
    player.click("new match/side1")
    player.click("new match/side1/2026-11-03-tiny/policy")
    player.click("new match/start")
    loop = session.loop
    assert loop.setup.config == PRESETS["deathmatch"]
    assert isinstance(loop.sides[1], Agent) and loop.setup.sides[1].policy == str(policy)
    player.click("speed/8")
    for _ in range(600):
        player.frames()
        if any(event.kind == "kill" for event in loop.events):
            break
    assert any(event.kind == "kill" for event in loop.events), "no kill in the time allowed"

    # Save its replay, open it, seek to a kill marker.
    player.click("menu")
    player.click("menu/save replay")
    player.click("menu/replays")
    player.click("replays/0")
    player.click("replays/open")
    assert session.loop.replaying
    markers = session.loop.markers()
    kill = next(index for index, event in enumerate(markers) if event.kind == "kill")
    player.click(f"replay/marker/{kill}")
    want = max(markers[kill].tick - 90, 0)  # three seconds before; it plays on from there
    assert want <= session.loop.position <= want + 8
    session.close()


def _uncovered_equal(player: Player, layout) -> tuple[int, int]:
    """(pixels differing outside the layout's rectangles, pixels differing inside)."""
    session = player.session
    session.show_interface = True
    player.frames(elapsed=0.0)
    with_interface = player.pixels()
    session.show_interface = False
    player.frames(elapsed=0.0)
    without = player.pixels()
    differs = np.any(with_interface != without, axis=-1)
    height, width = differs.shape
    rows, columns = np.mgrid[0:height, 0:width]
    covered = np.zeros_like(differs)
    for rect in layout.rects():
        covered |= (
            (columns >= np.floor(rect.x))
            & (columns < np.ceil(rect.right))
            & (rows >= np.floor(rect.y))
            & (rows < np.ceil(rect.bottom))
        )
    return int((differs & ~covered).sum()), int((differs & covered).sum())


@pytest.mark.parametrize(
    "width, height, scale, collapsed",
    [(1280, 720, 1.0, False), (1920, 1080, 1.5, True), (2560, 1440, 2.0, False)],
)
def test_the_interface_draws_only_inside_its_layout(
    context, tmp_path, width, height, scale, collapsed
):
    """Spec 8.4: every card, open or collapsed, the kill feed, a notice and the flying HUD,
    at each scale, leave the battlefield's pixels outside the layout's rectangles as they
    are."""
    session = _session(context, tmp_path, width, height, preset="frontline", scale=scale)
    player = Player(session, context, width, height)
    player.frames(20)
    session.loop.paused = True
    session.select(Selection("ship", 2))
    session.take_control(3)
    if collapsed:
        for card in ("fleets", "selection", "minimap"):
            session.interface.open.discard(card)
    else:
        session.interface.open |= {"fleets", "selection", "minimap"}
    session.say("a notice that is long enough to need some room " * 2)
    session.feed.append(
        (
            session.loop.ticks * session.loop.tick_length,
            MatchEvent(1, "kill", ship=0, other=width % 3 + 5),
        )
    )
    player.frames(2, elapsed=0.0)
    outside, inside = _uncovered_equal(player, session.interface.layout())
    assert outside == 0
    assert inside > 1000  # the interface did draw
    session.close()


def test_taking_control_puts_the_camera_on_the_flown_ship_and_keeps_it_there(context, tmp_path):
    """Spec 4: flying follows the flown ship by default, zoomed in from the fitted view:
    the ship is at the middle of the view, now and as it flies."""
    session = _session(context, tmp_path, 1920, 1080, preset="frontline")
    player = Player(session, context, 1920, 1080)
    player.frames(20)
    fitted = session.camera.metres_per_pixel
    ship = int(np.flatnonzero(session.frame.ships.alive)[3])
    session.take_control(ship)
    session.loop.inputs[:] = torch.tensor([True, False, True, False, True, False, False])
    for _ in range(4):
        frame = session.frame
        px, py = session.camera.pixel(float(frame.ships.x[ship]), float(frame.ships.y[ship]))
        assert abs(px - 960.0) < 3.0 and abs(py - 540.0) < 3.0
        player.frames(30)
    assert session.camera.metres_per_pixel < 0.5 * fitted
    session.close()


def test_the_tuning_panel_opens_from_the_menu_and_a_restart_keeps_the_tuning(context, tmp_path):
    """**Tune the game** opens the panel and starts the live match's tuner; a rule changed
    there is drawn on the next frame and stops the match's recording, and a restart starts
    from the tuned rules, unrecorded."""
    session = _session(context, tmp_path, preset="frontline")
    player = Player(session, context, 1280, 720)
    player.click("menu")
    player.click("menu/tune")
    assert session.tuning_panel.open and session.tuner is not None
    assert not session.interface.menu.is_open
    window = session.tuning_panel.window  # opened inside the display, scrolling within
    assert window.right <= 1280 and window.bottom <= 720 and window.width <= 420
    assert session.recorder is not None
    session.tuner.set_setting("mode.zone_radius", 500.0)
    assert session.recorder is None  # a tuned match is not recorded
    player.frames(2)
    assert session.frame.zones.radius == pytest.approx(np.full(5, 500.0))  # map scale 1
    player.click("menu")
    player.click("menu/restart")
    assert session.loop.rules.pools[0].mode.zone_radius == 500.0
    assert session.loop.setup.tuning.rules == {"mode.zone_radius": 500.0}
    assert session.tuner is not None and session.tuner.loop is session.loop
    assert session.recorder is None
    session.toggle_tuning()
    assert not session.tuning_panel.open
    player.frames(2)
    session.close()


def test_the_tuning_panel_adds_and_removes_ships(context, tmp_path, monkeypatch):
    """Adding a copy of a ship type writes its file and starts a match with three types;
    **Remove** asks first (Cancel keeps the type), and confirming goes back to two (the
    ship files here are a copy)."""
    import shutil
    import time

    from bnb.game import api, ships

    copy = tmp_path / "ships"
    shutil.copytree(ships.SHIPS, copy)
    monkeypatch.setattr(ships, "SHIPS", copy)
    try:
        session = _session(context, tmp_path, preset="frontline")
        player = Player(session, context, 1280, 720)
        session.toggle_tuning()
        deadline = time.monotonic() + 300.0
        while not session.tuner.ready() and time.monotonic() < deadline:
            player.frames()
            time.sleep(0.02)
        panel = session.tuning_panel
        panel.new_ship = "Scout"
        panel._add(session, session.tuner)
        assert api.type_names() == ("Fighter", "Heavy", "Scout")
        assert (copy / "scout.toml").is_file()
        assert session.loop.types.max_energy.shape == (3,)
        player.frames(3)
        panel.ship = 2
        player.frames(2)
        player.click("tuning/remove")
        assert session.interface.controls.shown("tuning/remove/confirm")
        player.click("tuning/remove/cancel")
        assert not session.interface.controls.shown("tuning/remove/confirm")
        assert api.type_names() == ("Fighter", "Heavy", "Scout")
        player.click("tuning/remove")
        player.click("tuning/remove/confirm")
        assert api.type_names() == ("Fighter", "Heavy") and not (copy / "scout.toml").exists()
        assert session.loop.types.max_energy.shape == (2,)
        player.frames(3)
        session.close()
    finally:
        monkeypatch.undo()
        api.reload_ships()


def test_a_real_scale_match_is_chosen_with_a_dropdown_lineup_and_sped_up(context, tmp_path):
    """New match: the Real scale map plays the real-scale ship set; a lineup slot is a
    dropdown of the set's type names; Make this speed 1x turns the playback speed into the
    game's own (the time scale) and sets playback back to x1."""
    session = _session(context, tmp_path)
    player = Player(session, context, 1280, 720)
    player.click("menu")
    player.click("menu/new match")
    player.click("new match/map/real_scale")
    player.click("new match/mode/frontline")
    player.click("new match/lineup/0")
    names = api.type_names("real-scale")
    player.click(f"new match/lineup/0/{names[1]}")
    player.click("new match/start")
    config = session.loop.setup.config
    assert config == PRESETS["real_scale_frontline"]
    assert session.loop.setup.lineup[0] == 1
    assert session.loop.types.max_energy.tolist() == list(
        api.default_card("real-scale")["max_energy"]
    )
    gravity = session.loop.rules.flight.gravity
    assert gravity == pytest.approx(9.81)
    player.click("speed/0.5")
    session.make_speed_normal()
    assert session.loop.time_scale == 1.0
    assert session.loop.rules.flight.gravity == pytest.approx(0.25 * gravity)
    assert session.loop.rules.pools[0].episode_ticks == 2 * 600 * 30
    session.close()
