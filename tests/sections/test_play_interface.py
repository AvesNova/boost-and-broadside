"""The interface offscreen (`bnb.play.interface`), driven only through its controls:
synthetic mouse input on named controls, as a player would click them. Needs an OpenGL
3.3 context (EGL)."""

import dataclasses

import numpy as np
import pytest
import torch
from play_cases import tiny_policy

from bnb.game import api
from bnb.game.api import PRESETS
from bnb.play import themes
from bnb.play.app import Options, Session, read_pixels
from bnb.play.interface.minimap import to_world
from bnb.play.loop import MatchEvent, SideSetup
from bnb.play.maps import Maps
from bnb.play.render.bullets import BulletLook
from bnb.play.selection import Selection, shown_ships
from bnb.play.sides import Agent
from bnb.play.tuning import Tuning

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

    def type(self, text: str) -> None:
        """Type `text` as the window passes keys on: each key's press (with Shift for a
        capital) and its character, then its release, a frame each."""
        import glfw

        from bnb.play.window import IMGUI_KEYS

        for character in text:
            key = IMGUI_KEYS.get(getattr(glfw, f"KEY_{character.upper()}", None))
            if key is None and character == " ":
                key = IMGUI_KEYS[glfw.KEY_SPACE]
            self.io.add_key_event(imgui_key("mod_shift"), character.isupper())
            if key is not None:
                self.io.add_key_event(key, True)
            self.io.add_input_character(ord(character))
            self.frames()
            if key is not None:
                self.io.add_key_event(key, False)
            self.io.add_key_event(imgui_key("mod_shift"), False)
            self.frames()

    def pixels(self) -> np.ndarray:
        return read_pixels(self.target).astype(int)


def imgui_key(name: str):
    from imgui_bundle import imgui

    return getattr(imgui.Key, name)


def _session(context, tmp_path, width=1280, height=720, **options) -> Session:
    options.setdefault("preset", "deathmatch")  # the standard game, as the tiny policy's
    options = Options(
        compile=False,
        width=width,
        height=height,
        replays=tmp_path / "replays",
        runs=tmp_path / "runs",
        **options,
    )
    return Session(
        options,
        context,
        width,
        height,
        library=themes.Library(tmp_path / "themes"),
        maps=Maps(tmp_path / "maps", tmp_path / "project maps"),
    )


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
    session.seek(0)
    assert not np.isfinite(session.last_shot).any()  # "last shot": none seen since the seek
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


class _StaleScreen:
    """A window's framebuffer as moderngl gives it: `size` stays what it was when the
    context was made, whatever size the window has since taken."""

    def __init__(self, framebuffer, stale_size: tuple[int, int]) -> None:
        self.framebuffer, self.size = framebuffer, stale_size

    def __getattr__(self, name: str):
        return getattr(self.framebuffer, name)  # everything but the size: the framebuffer's


def test_the_interface_fills_a_window_resized_after_it_opened(context, tmp_path):
    """A window opened at 1280 x 720 and then maximised (its framebuffer 1600 x 900, the
    screen's size still 1280 x 720): the interface draws over the whole new size, its
    right-hand controls past the old width included."""
    session = _session(context, tmp_path)
    session.resize(1600, 900)
    framebuffer = context.framebuffer(color_attachments=[context.texture((1600, 900), 4)])
    screen = _StaleScreen(framebuffer, (1280, 720))

    def drawn(interface: bool) -> np.ndarray:
        session.show_interface = interface
        for _ in range(3):
            session.update(FRAME)
            session.draw(screen)
        return read_pixels(framebuffer).astype(int)

    bare, with_interface = drawn(False), drawn(True)
    rect = session.interface.controls.rect("speed/8")
    assert rect.x > 1280  # laid out in the new width
    x, y = (int(v) for v in rect.centre())
    assert (
        np.abs(
            with_interface[y - 3 : y + 4, x - 3 : x + 4] - bare[y - 3 : y + 4, x - 3 : x + 4]
        ).sum()
        > 0
    )


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
    with the look-ahead off, the ship is at the middle of the view, now and as it flies."""
    session = _session(context, tmp_path, 1920, 1080, preset="frontline", look_ahead="off")
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


def test_the_map_panel_redraws_the_map_restarts_and_switches_the_mode(context, tmp_path):
    """**New map** is a fresh seed (new fields) with the tuning kept, **Restart match** the
    same map again, a mode button a new match in that mode; the panel's rule groups are
    the map's and the ship panel's are the rest."""
    session = _session(context, tmp_path, preset="frontline")
    player = Player(session, context, 1280, 720)
    session.toggle_map_panel()
    player.frames(3)
    assert session.map_panel.open and session.tuner is not None
    session.tuner.set_setting("rules.field_high_share", 1.0)
    fields = session.loop.current.fields.level.copy()
    player.click("map/new map")
    player.frames(2)
    assert session.loop.setup.seed != 0
    assert (session.loop.current.fields.level > 0).all() and not (fields > 0).all()
    seed, radius = session.loop.setup.seed, session.loop.current.fields.radius.copy()
    player.click("map/restart")
    assert session.loop.setup.seed == seed
    assert np.array_equal(session.loop.current.fields.radius, radius)
    player.click("map/mode/elimination")
    assert session.loop.setup.config.mode == "elimination"
    assert session.loop.setup.tuning.rules["rules.field_high_share"] == 1.0
    window = session.map_panel.window
    assert window.right <= 1280 and window.bottom <= 720
    session.close()


def test_typed_letters_reach_the_text_boxes_and_call_no_shortcut(context, tmp_path):
    """Letters typed into a text box (each a shortcut or a flight key elsewhere: Z, T, E,
    V, P, F, C, N, G, O, W, A, S, D, space) go into the box only: the view, theme, true
    size, grid, contours, pause, editor and the pilot's inputs stay as they were."""
    session = _session(context, tmp_path, preset="frontline")
    player = Player(session, context, 1280, 720)
    session.toggle_tuning()
    player.frames(3)

    def state():
        loop = session.loop
        return (
            session.true_size,
            session.grid,
            session.contours,
            session.theme.name,
            session.view_name,
            session.editor.open,
            session.show_keys,
            loop.paused,
            loop.time_scale,
            session.following,
            bool(loop.inputs.any()),
        )

    before = state()
    text = "Zeta Paw Fencer 1001 vwasdcntgo"
    player.click("tuning/new ship")
    player.type(text)
    assert session.tuning_panel.new_ship == text
    assert state() == before
    session.toggle_tuning()
    player.click("menu")
    player.click("menu/new match")
    before = state()  # the menu pauses the match
    player.click("new match/seed")
    player.type("12")
    player.io.add_key_event(imgui_key("enter"), True)  # a number box takes its text on Enter
    player.frames()
    player.io.add_key_event(imgui_key("enter"), False)
    player.frames()
    assert session.interface.menu.draft.seed == 12 and state() == before
    player.click("new match/side1")
    player.click("new match/side1/path")
    player.type("runs/zeta-1")
    assert session.interface.typed_policy == "runs/zeta-1"
    assert state() == before  # "-" is the slower key elsewhere
    session.close()


def test_the_tuning_panel_adds_and_removes_ships(context, tmp_path, monkeypatch):
    """Adding a copy of a ship type writes its file at once (under a time scale, the
    design's x1 values: the original's own) and starts a match with three types at the
    same time scale; **Remove** asks first (Cancel keeps the type), and confirming goes
    back to two (the ship files here are a copy)."""
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
        session.time_scale(2.0)
        player.frames(2)
        player.click("tuning/new ship")
        player.type("Scout 1001")
        player.click("tuning/add")
        assert api.type_names() == ("Fighter", "Heavy", "Scout 1001")
        scout, fighter = ships.load()[2], ships.load()[0]
        assert scout.key == "scout-1001" and (copy / "scout-1001.toml").is_file()
        assert scout.card == pytest.approx(fighter.card, rel=1e-6)  # the card is float32
        assert scout.handles == fighter.handles
        assert panel.message.startswith("added Scout 1001, a copy of Fighter: saved")
        assert session.loop.types.max_energy.shape == (3,)
        assert session.time_factor == 2.0
        player.frames(3)
        panel.ship = 2
        player.frames(2)
        player.click("tuning/remove")
        assert session.interface.controls.shown("tuning/remove/confirm")
        player.click("tuning/remove/cancel")
        assert not session.interface.controls.shown("tuning/remove/confirm")
        assert api.type_names() == ("Fighter", "Heavy", "Scout 1001")
        player.click("tuning/remove")
        player.click("tuning/remove/confirm")
        assert api.type_names() == ("Fighter", "Heavy") and not (copy / "scout-1001.toml").exists()
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
    assert gravity == pytest.approx(4.0 * 9.81)  # the set's native x2 (game v13)
    player.click("speed/0.5")
    session.make_speed_normal()
    assert session.loop.time_scale == 1.0
    assert session.loop.rules.flight.gravity == pytest.approx(0.25 * gravity)
    assert session.loop.rules.pools[0].episode_ticks == 2 * 600 * 30
    player.click("speed/2")  # x2 playback of the x0.5 game: the design's own speed
    session.make_speed_normal()  # relative to now, unlike Faster by
    assert session.time_factor == 1.0 and session.loop.time_scale == 1.0
    assert session.loop.rules.flight.gravity == pytest.approx(gravity)
    session.close()


def test_a_side_slow_to_make_plays_on_until_it_is_made(context, tmp_path, monkeypatch):
    """A scripted agent whose card is not measured yet: choosing it returns at once, the
    team plays on with its controller while the measurement runs aside, and the agent takes
    over when it is in."""
    import threading

    from bnb.play import app

    measured = threading.Event()
    monkeypatch.setattr(
        app, "cached_characteristics", lambda types, rules: object() if measured.is_set() else None
    )
    monkeypatch.setattr(app, "measure_aside", lambda types, rules: measured.wait(60))
    session = _session(context, tmp_path)
    player = Player(session, context, 1280, 720)
    assert session.set_side(1, SideSetup("scripted_agent"))
    player.frames(10)
    assert session.loop.sides[1].label == "waiting"
    assert "takes over" in session.notice
    measured.set()
    for _ in range(200):
        player.frames()
        if session.loop.sides[1].label == "scripted agent r9m2":
            break
    assert session.loop.sides[1].label == "scripted agent r9m2"
    assert session.loop.setup.sides[1] == SideSetup("scripted_agent")


def test_a_new_match_on_an_unmeasured_card_flies_the_last_matchs_characteristics(
    context, tmp_path, monkeypatch
):
    """A new match whose card no run has measured (a tuned card): its scripted sides fly
    on the last match's characteristics while the card is measured aside, not coasting
    straight, and take the card's own when they are in."""
    import threading

    from bnb.game.handling.characteristics import characteristics
    from bnb.play import app

    session = _session(context, tmp_path)
    old = session.loop
    traits = characteristics(old.types, old.rules)
    measured = threading.Event()
    monkeypatch.setattr(
        app,
        "cached_characteristics",
        lambda types, rules: traits if types is old.types or measured.is_set() else None,
    )
    monkeypatch.setattr(app, "measure_aside", lambda types, rules: measured.wait(60))
    session.restart()
    assert session.loop is not old
    assert all(side.label != "waiting" for side in session.loop.sides)
    measured.set()
    session.close()


def test_the_arena_is_chosen_flown_and_drawn_at_true_size(context, tmp_path):
    """New match offers the arena as a map (2v2, no fields, a P-51D and a Spitfire XIV a
    team); taking control of a ship keeps every card as it was, open or collapsed; the top
    bar's True size switches the ships' drawing."""
    session = _session(context, tmp_path)
    player = Player(session, context, 1280, 720)
    player.click("menu")
    player.click("menu/new match")
    player.click("new match/map/arena")
    player.click("new match/start")
    assert session.loop.setup.config == PRESETS["arena"]
    assert session.loop.rules.team_types == api.preset_lineup("arena")
    player.frames(10)

    player.click("fleet/ship/1")
    assert session.selection == Selection("ship", 1)
    assert {"fleets", "selection"} <= session.interface.open
    player.click("selection/take control")
    assert session.loop.pilot == 1
    assert {"fleets", "selection", "minimap"} <= session.interface.open
    player.click("flying/release")
    player.click("fleets/collapse")
    assert "fleets" not in session.interface.open
    player.click("selection/take control")
    assert session.loop.pilot == 1
    assert "fleets" not in session.interface.open and "selection" in session.interface.open

    assert not session.true_size
    player.click("true size")
    assert session.true_size
    player.click("true size")
    assert not session.true_size
    session.close()


def test_start_never_measures_in_the_windows_thread(context, tmp_path, monkeypatch):
    """The author's hang: with the tuning panel open on a tuned card, New match -> Arena ->
    Start measured the card's characteristics for the scripted sides in the window's
    thread (minutes for the real-scale set) and joined the tuner's workers. Now Start
    returns at once, both sides wait (coasting) while the card is measured aside, and the
    scripted agents take over when it is in."""
    import threading
    import time

    from bnb.play import app, sides

    measured = threading.Event()
    monkeypatch.setattr(
        app, "cached_characteristics", lambda types, rules: object() if measured.is_set() else None
    )
    monkeypatch.setattr(app, "measure_aside", lambda types, rules: measured.wait(60))
    measure = sides.characteristics

    def characteristics(types, rules):
        assert measured.is_set(), "a card measured in the window's thread"
        return measure(types, rules)

    monkeypatch.setattr(sides, "characteristics", characteristics)
    measured.set()  # the first match's card is measured
    session = _session(context, tmp_path, preset="arena")
    player = Player(session, context, 1280, 720)
    session.toggle_tuning()
    player.frames(5)
    session.take_control(1)
    measured.clear()  # a card no run has measured yet
    player.click("menu")
    player.click("menu/new match")
    player.click("new match/map/arena")
    assert session.interface.menu.draft.lineup == list(api.preset_lineup("arena"))
    player.click("new match/fog/off")
    started = time.perf_counter()
    player.click("new match/start")
    assert time.perf_counter() - started < 5.0
    assert session.loop.setup.config == PRESETS["arena"] and not session.interface.menu.is_open
    assert [side.label for side in session.loop.sides] == ["waiting", "waiting"]
    assert "takes over" in session.notice
    player.frames(10)
    measured.set()
    for _ in range(200):
        player.frames()
        if all(side.label == "scripted agent r9m2" for side in session.loop.sides):
            break
    assert [side.label for side in session.loop.sides] == ["scripted agent r9m2"] * 2
    session.close()


def test_the_camera_leads_the_flown_ship_in_its_direction_of_travel(context, tmp_path):
    """The look-ahead: flying, the view's middle moves ahead of the ship along its
    velocity, within the cap; set to Off in the settings, it eases back onto the ship."""
    from bnb.play.camera import LEAD_CAP

    session = _session(context, tmp_path, 1280, 720, preset="frontline")
    player = Player(session, context, 1280, 720)
    player.frames(10)
    ship = int(np.flatnonzero(session.frame.ships.alive)[3])
    session.take_control(ship)
    session.loop.inputs[:] = torch.tensor([True, False, False, False, False, False, False])
    player.frames(150)
    frame, camera = session.frame, session.camera
    x, y = float(frame.ships.x[ship]), float(frame.ships.y[ship])
    lead_x, lead_y = camera.centre_x - x, camera.centre_y - y  # m
    lead = np.hypot(lead_x, lead_y) / camera.metres_per_pixel
    assert 20.0 < lead <= LEAD_CAP * 360.0
    path = frame.ships.flight_path[ship]
    assert lead_x * np.cos(path) + lead_y * np.sin(path) > 0.95 * lead * camera.metres_per_pixel
    player.click("menu")
    player.click("menu/settings")
    player.click("settings/look ahead/off")
    assert session.look_ahead == "off"
    player.click("menu/back")
    player.click("menu/resume")
    player.frames(300)
    frame = session.frame
    px, py = camera.pixel(float(frame.ships.x[ship]), float(frame.ships.y[ship]))
    assert abs(px - 640.0) < 2.0 and abs(py - 360.0) < 2.0
    session.close()


def test_the_dot_grid_is_chosen_in_the_top_bar_the_settings_and_with_g(context, tmp_path):
    """The top bar's Grid button and G go to the next pattern (off, square, triangle, hex);
    the settings choose one; the battlefield changes with it."""
    session = _session(context, tmp_path, preset="arena", grid="off")
    player = Player(session, context, 1280, 720)
    seen = []
    for _ in range(4):
        player.click("grid")
        seen.append(session.grid)
    assert seen == ["square", "triangle", "hex", "off"]
    player.io.add_key_event(imgui_key("g"), True)
    player.frames()
    player.io.add_key_event(imgui_key("g"), False)
    player.frames()
    assert session.grid == "square"
    player.click("menu")
    player.click("menu/settings")
    player.click("settings/grid/hex")
    assert session.grid == "hex"
    player.click("settings/grid/off")
    assert session.grid == "off"

    def battlefield():  # the menu paused the match: only the grid changes it
        session.show_interface = False
        player.frames()
        pixels = player.pixels()
        session.show_interface = True
        player.frames(2)
        return pixels

    bare = battlefield()
    player.click("settings/grid/triangle")
    assert session.grid == "triangle"
    assert (battlefield() != bare).any()  # the dots
    session.close()


def test_the_bullets_look_is_chosen_in_the_settings(context, tmp_path):
    """Settings > Bullets: today's tracer by default; Trail only hides the head's size and
    changes how the battlefield draws; a slider sets a width."""
    session = _session(context, tmp_path, preset="arena", bullet_trail_head_width=3.0)
    assert session.bullet_look == BulletLook(trail_head_pixels=3.0)
    player = Player(session, context, 1280, 720)
    player.click("menu")
    player.click("menu/settings")
    controls = session.interface.controls
    assert "settings/bullets/head_pixels" in controls.last
    player.click("settings/bullets/trail")
    assert session.bullet_look.style == "trail"
    assert "settings/bullets/head_pixels" not in controls.last
    width = controls.rect("settings/bullets/trail_head_pixels")
    player.click_at(width.x + 0.95 * width.width, width.y + 0.5 * width.height)
    assert session.bullet_look.trail_head_pixels > 12.0
    player.click("settings/bullets/tracer")
    assert session.bullet_look.style == "tracer"
    session.close()


def test_the_looks_save_with_the_theme_and_flags_override_them(context, tmp_path):
    """The Settings menu edits the theme's looks; Save as theme keeps them, and the next
    session that loads the theme has them. A flag overrides the loaded theme's value."""
    session = _session(context, tmp_path, preset="arena")
    session.set_grid("hex")
    session.set_contours("zones")
    session.toggle_true_size()
    session.set_look_ahead("off")
    session.set_bullet_look(style="trail", trail_seconds=0.1)
    saved = session.editor.save_as(session.theme, "my-look")
    assert saved is not None
    session.close()
    again = _session(context, tmp_path, preset="arena", theme="my-look", grid="square")
    assert again.grid == "square"  # the flag
    assert (again.contours, again.true_size, again.look_ahead) == ("zones", True, "off")
    assert again.bullet_look == BulletLook(style="trail", trail_seconds=0.1)
    assert again.original.grid_pattern == "hex"  # Revert goes back to the file
    again.close()


def test_a_map_is_made_saved_played_edited_renamed_and_deleted_in_the_menu(context, tmp_path):
    """Menu > Maps: start from the arena, change it, save it as "duel" (yours); it shows on
    New match and plays from there and from `--preset duel`; edited and saved again, moved
    to the project, renamed and deleted (after the popup asks). A match's tuning goes with
    the map made from it."""
    session = _session(context, tmp_path, preset="arena")
    player = Player(session, context, 1280, 720)
    library = session.maps
    player.click("menu")
    player.click("menu/maps")
    player.click("maps/start from")
    player.click("maps/start from/arena")
    editor = session.interface.menu.editor
    assert editor.form.ships_per_team == 2 and editor.form.lineup == [0, 5]
    editor.form.num_fields = 3
    player.click("maps/name")
    player.type("duel")
    player.click("maps/save")
    duel = library.find("duel")
    assert duel.place == "user" and (tmp_path / "maps" / "duel.json").is_file()
    assert duel.config == dataclasses.replace(PRESETS["arena"], num_fields=3)
    assert duel.lineup == ("P-51D", "Spitfire XIV")

    player.click("menu/back")
    player.click("menu/new match")
    player.click("new match/map/duel")
    player.click("new match/start")
    assert session.loop.setup.config == duel.config
    assert session.loop.setup.lineup == (0, 5)

    player.click("menu")
    player.click("menu/maps")
    player.click("maps/start from")
    player.click("maps/start from/duel")
    assert editor.name == "duel"
    editor.form.kills_per_ship = 2
    player.click("maps/place/project")
    player.click("maps/save")  # the same name: the map edited, and moved to the project
    assert library.find("duel").config.kills_per_ship == 2
    assert library.place_of("duel") == "project" and not (tmp_path / "maps" / "duel.json").exists()
    player.click("maps/name")
    player.type("-two")
    player.click("maps/rename")
    assert library.place_of("duel") is None and library.find("duel-two").place == "project"
    player.click("maps/delete")
    assert session.interface.controls.shown("maps/delete/confirm")
    player.click("maps/delete/cancel")
    assert library.place_of("duel-two") == "project"

    # A tuned match's map keeps the tuning.
    setup = dataclasses.replace(session.loop.setup, tuning=Tuning(time_factor=2.0))
    assert session.new_match(setup, tuned=True)
    player.click("menu")
    player.click("menu/maps")
    assert editor.form.tuning == Tuning(time_factor=2.0)
    player.click("maps/name")
    player.type("fast-duel")
    player.click("maps/save")
    assert library.find("fast-duel").tuning.time_factor == 2.0
    session.close()

    again = _session(context, tmp_path, preset="duel-two")  # `bnb play --preset duel-two`
    assert again.loop.setup.config.kills_per_ship == 2
    player = Player(again, context, 1280, 720)
    player.click("menu")
    player.click("menu/maps")
    player.click("maps/start from")
    player.click("maps/start from/duel-two")
    player.click("maps/delete")
    player.click("maps/delete/confirm")
    assert library.place_of("duel-two") is None
    player.click("menu/back")
    player.click("menu/new match")
    assert not again.interface.controls.shown("new match/map/duel-two")
    assert again.interface.controls.shown("new match/map/fast-duel")
    again.close()


def test_the_contours_are_chosen_in_the_top_bar_the_settings_and_with_o(context, tmp_path):
    """The top bar's Contours button and O go to the next shape set (off, fields, edge,
    fields and edge, all, zones); the settings choose one; the battlefield changes with it."""
    session = _session(context, tmp_path, preset="frontline", contours="off")
    player = Player(session, context, 1280, 720)
    seen = []
    for _ in range(6):
        player.click("contours")
        seen.append(session.contours)
    assert seen == ["fields", "edge", "fields+edge", "fields+edge+zones", "zones", "off"]
    player.io.add_key_event(imgui_key("o"), True)
    player.frames()
    player.io.add_key_event(imgui_key("o"), False)
    player.frames()
    assert session.contours == "fields"
    player.click("menu")
    player.click("menu/settings")
    player.click("settings/contours/off")
    assert session.contours == "off"

    def battlefield():  # the menu paused the match: only the contours change it
        session.show_interface = False
        player.frames()
        pixels = player.pixels()
        session.show_interface = True
        player.frames(2)
        return pixels

    bare = battlefield()
    player.click("settings/contours/fields+edge+zones")
    assert session.contours == "fields+edge+zones"
    assert (battlefield() != bare).any()  # the lines
    session.close()


def test_the_frame_rate_shows_in_the_top_bar_and_f4_and_the_settings_toggle_it(context, tmp_path):
    """On by default; F4 and Settings > Frame rate turn it off and on; the readout is
    drawn in the top bar (its pixels change with it)."""
    session = _session(context, tmp_path, preset="arena")
    player = Player(session, context, 1280, 720)
    assert session.show_fps
    player.frames(70)
    assert session.frame_rate.per_second == pytest.approx(60.0)
    shown = player.pixels()[:40]
    player.io.add_key_event(imgui_key("f4"), True)
    player.frames()
    player.io.add_key_event(imgui_key("f4"), False)
    player.frames()
    assert not session.show_fps
    assert (player.pixels()[:40] != shown).any()
    player.click("menu")
    player.click("menu/settings")
    player.click("settings/fps/on")
    assert session.show_fps
    player.click("settings/fps/off")
    assert not session.show_fps
    session.close()


def test_the_fleets_combo_sets_the_scripted_agents_doctrine(context, tmp_path, monkeypatch):
    """The fleets card's controller combo offers the scripted agent with r9m2 (the default)
    and with r9; choosing r9 makes that side fly it."""
    from bnb.play import app

    monkeypatch.setattr(app, "cached_characteristics", lambda types, rules: object())
    session = _session(context, tmp_path)
    player = Player(session, context, 1280, 720)
    assert session.loop.setup.sides[1] == SideSetup("scripted_agent", doctrine="r9m2")
    player.click("fleet/team1/controller")
    player.click("fleet/team1/controller/scripted agent r9")
    assert session.loop.setup.sides[1] == SideSetup("scripted_agent", doctrine="r9")
    assert session.loop.sides[1].label == "scripted agent r9"
    session.close()


def test_the_fleets_card_shows_only_the_pilots_own_wave_clock():
    from bnb.play.interface.fleets import wave_shown

    assert wave_shown("frontline", None, 5, 0) and wave_shown("frontline", None, 5, 1)
    assert wave_shown("frontline", 7, 5, 1) and not wave_shown("frontline", 7, 5, 0)
    assert not wave_shown("team_deathmatch", None, 5, 0)


def test_a_locked_zones_dot_carries_its_countdown_in_the_held_teams_colour():
    from types import SimpleNamespace

    from bnb.play.interface.top_bar import _lock_label, _zone_tooltip

    zones = SimpleNamespace(
        lock=np.array([[0.0, 0.0], [44.2, 0.0]]),
        role=np.array([1, 2]),
        owner=np.array([0, -1]),
        attacker=np.array([1, -1]),
        meter=np.array([0.0, 0.0]),
    )
    assert _lock_label(zones, 0) == ("45s", 1)
    assert _lock_label(zones, 1) is None
    assert "44.2 s" in _zone_tooltip(zones, 0, 1)
