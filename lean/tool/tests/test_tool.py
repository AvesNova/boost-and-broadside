"""The web tool: the envelope function matches the game's quasi-steady maths, the server answers
the envelope endpoint and a handles change, and the sandbox socket advances a World."""

import math

import numpy as np
import pytest
import torch

from lean.game import card, core, tuning
from lean.tool import design, session

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from lean.tool.app import app  # noqa: E402


def test_envelope_equals_the_quasi_steady_maths_for_the_p51d():
    """Rate, Ps and the sustained curve of the envelope are the flight code's quasi-steady
    numbers (core.quasi_steady, the independent entry the tuning's performance uses)."""
    row = tuning.row_of("p51d")
    speeds, sticks = np.array([120.0, 200.0, 335.0, 420.0]), np.array([0.0, 0.4, 0.7, 1.0])
    grid = tuning.envelope(row, speeds, sticks)
    with tuning._with_row(row) as index:
        p = core.per_type_constants(core.GameConfig(), torch.full((16,), index))
        v = torch.tensor(np.tile(speeds, 4), dtype=torch.float32)
        s = torch.tensor(np.repeat(sticks, 4), dtype=torch.float32)
        rate, _, energy_rate = core.quasi_steady(p, v, s, 0, p.spawn_energy)
    assert grid["rate"].ravel() == pytest.approx(rate.numpy(), rel=1e-5)
    assert (grid["ps"] * card.GRAVITY).ravel() == pytest.approx(
        energy_rate.numpy(), rel=1e-4, abs=1e-3
    )


def test_envelope_key_numbers_agree_with_performance():
    """The corner is the full-stick peak: the settled-flight corner of tuning.performance, to
    the grid's resolution; the best sustained turn is held (Ps >= 0) and at most the corner's."""
    row = tuning.row_of("p51d")
    grid = tuning.envelope(row, np.linspace(20.0, 470.0, 226), np.linspace(0.0, 1.0, 101))
    flown = tuning.performance(row)
    assert grid["corner_speed"] == pytest.approx(flown["corner_speed_flown"], abs=2.5)
    assert grid["corner_rate"] == pytest.approx(
        flown["corner_rate"], rel=1e-2
    )  # grid, and spawn against mid store
    assert grid["best_sustained_rate"] <= grid["corner_rate"]
    assert (
        grid["tightest_sustained_radius"]
        <= grid["best_sustained_speed"] / grid["best_sustained_rate"] + 1e-6
    )


def test_handles_change_moves_the_design_and_the_reference_stays():
    handles = tuning.handles_of(tuning.row_of("p51d"))
    result = design.envelope(
        "p51d", {"g_limit": 1.3 * handles["g_limit"]}, None, tuning.TIME_FACTOR
    )
    assert result["design"]["key"]["corner_rate"] > result["reference"]["key"]["corner_rate"]
    assert result["reference"]["key"]["top_speed"] == pytest.approx(
        handles["cruise_speed"], rel=1e-4
    )


def test_server_answers_the_envelope_endpoint():
    client = TestClient(app)
    presets = client.get("/api/presets").json()
    assert "p51d" in presets["presets"]
    assert client.get("/").status_code == 200
    reply = client.post("/api/envelope", json={"preset": "p51d", "time_scale": 1.0})
    assert reply.status_code == 200
    data = reply.json()
    assert math.isfinite(data["design"]["key"]["corner_rate"])
    assert (
        data["design"]["lowest"]["full"] > data["design"]["key"]["stall_speed"]
    )  # lines start above stall
    # the same design at 1x real time has half the speeds and rates (to_real)
    at_game = client.post("/api/envelope", json={"preset": "p51d"}).json()
    assert data["design"]["key"]["top_speed"] == pytest.approx(
        at_game["design"]["key"]["top_speed"] / 2.0, rel=1e-3
    )


def test_sandbox_socket_advances_a_world():
    client = TestClient(app)
    with client.websocket_connect("/api/sandbox") as socket:
        socket.send_json({"preset": "p51d", "time_scale": 2.0, "opponent": True, "ticks": 5})
        setup = socket.receive_json()
        assert setup["type"] == "setup" and setup["arena_radius"] > 0
        frames = [socket.receive_json() for _ in range(5)]
    assert [f["tick"] for f in frames] == [1, 2, 3, 4, 5]
    assert frames[-1]["readout"]["speed"] > 0


def test_session_applies_controls():
    """session.run in-process: a held full left stick for some ticks turns the player."""

    class Pipe:
        def __init__(self):
            self.sent, self.queue = [], [dict(side=1, level=3, power=0, fire=False)]

        def send(self, message):
            self.sent.append(message)

        def poll(self):
            return bool(self.queue)

        def recv(self):
            return self.queue.pop(0)

    pipe = Pipe()
    session.run(
        pipe,
        tuning.row_of("p51d"),
        tuning.TIME_FACTOR,
        opponent=False,
        realtime=False,
        max_ticks=30,
    )
    frames = [m for m in pipe.sent if m["type"] == "frame"]
    assert len(frames) == 30
    assert abs(frames[-1]["ships"][0]["heading"] - math.pi / 2) > 0.05  # left of its start heading
    assert card.CARD is not None


def test_the_sandbox_opponent_spawns_by_the_games_rule():
    """Review B7: the sandbox places its ships through the game's spawn (core.spawn), so a
    respawned opponent has the standard speed over the map wherever it is (it had the
    proper speed of empty space, too slow or fast inside a field)."""
    from lean.sandbox.world import World

    for seed in range(20):
        world = World(core.GameConfig(), seed=seed)
        log_index = float(world.state.log_index[0, 1])
        if abs(log_index) > 1e-3:  # the opponent inside a field
            break
    else:
        pytest.fail("no seed placed the opponent in a field")
    s = world.state
    assert float(s.speed[0, 1]) == pytest.approx(card.SPAWN_SPEED * math.exp(log_index), rel=1e-6)
    assert float(s.energy[0, 1]) == pytest.approx(world.constant("spawn_energy", 1))
    assert bool(s.alive[0, 0]) and bool(s.alive[0, 1])


def test_the_doghouse_endpoint_returns_the_committed_charts_png():
    """The tool's doghouse is drawn by the committed charts' matplotlib code, as a PNG."""
    from fastapi.testclient import TestClient

    from lean.tool.app import app

    response = TestClient(app).post(
        "/api/doghouse.png", json={"preset": "p51d", "time_scale": 2.0, "units": "real"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_sustained_endpoint_returns_a_png_and_an_unsolvable_design_an_error():
    client = TestClient(app)
    response = client.post(
        "/api/sustained.png", json={"preset": "p51d", "values": {"g_limit": 9.0}, "units": "real"}
    )
    assert response.status_code == 200
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    bad = client.post(
        "/api/doghouse.png", json={"preset": "p51d", "values": {"stall_speed": 400.0}}
    )
    assert bad.status_code == 422 and bad.json()["error"]


def test_the_inputs_are_the_tunings_own():
    """The page's inputs are generated from the tuning route: every handle tuning.handles_of
    returns (but the fixed ones) and every card field handles_row takes from its base row rather
    than from the handles; every game term with an off value (tuning.POINT_MASS_TERMS) has a
    switch; the page keeps no list of its own and no Plotly."""
    from pathlib import Path

    row = tuning.row_of("p51d")
    handles = tuning.handles_of(row)
    set_by_handles = set(tuning.direct_fields(handles, handles["cruise_speed"])) | set(
        tuning.SOLVED
    )
    expected = (set(handles) - set(design.FIXED_HANDLES)) | (
        set(row) - set_by_handles - set(handles) - {"exchange_acceleration"}
    )
    entries = design.presets()["p51d"]["inputs"]
    names = [e["name"] for e in entries]
    assert sorted(names) == sorted(expected)
    assert set(card.FLIGHT_TERMS) <= set(names)
    assert {e["name"] for e in entries if "switch" in e} == (
        set(tuning.POINT_MASS_TERMS) & set(names)
    ) | {design.MATCH}
    for e in entries:
        assert e["low"] <= e["value"] <= e["high"] and e["group"] in design.GROUPS
    page = Path(design.__file__).with_name("index.html").read_text()
    assert "plotly" not in page.lower()
    assert not any(f'"{name}"' in page or f"'{name}'" in page for name in names)


def test_inputs_and_switches_reach_the_row():
    """A base-row input and a switch change the solved row; the handles still fly."""
    row = design.build_row(
        "p51d", {"fine_stick_fraction": 0.3}, ["low_speed_acceleration", "separation_drag"]
    )
    assert row["fine_stick_fraction"] == pytest.approx(0.3)
    assert row["separation_drag"] == tuning.POINT_MASS_TERMS["separation_drag"]
    assert tuning.performance(row)["top_speed"] == pytest.approx(
        tuning.performance(tuning.row_of("p51d"))["top_speed"], rel=1e-3
    )
    assert design.build_row("p51d", {}, None) == tuning.row_of("p51d")


@pytest.fixture(scope="module")
def p51d_flown():
    """The P-51D's lowest flown speeds, quasi-steady key points and flown points (game units)."""
    from lean.tool import flown

    with tuning.flying(tuning.row_of("p51d")):
        lowest = flown.achievable_speeds()
        key = design._chart().key_points(lowest)
        points = flown.critical_points(
            key["corner_speed"], key["best_sustained_speed"], key["tightest_sustained_speed"]
        )
    return lowest, key, points


def test_the_p51d_lines_start_where_the_game_flies_them(p51d_flown):
    """The lowest speeds the game flies each line (flown.achievable_speeds, measured here on the
    card as it stands, no stored numbers): every line is flown somewhere between the stall speed
    and the corner speed, and an independent random search of control sequences
    (flown.random_search) flies no line more than 5 m/s below its measured start, so the
    search the charts use finds the lowest states near enough (3 m/s before 2026-11-24; with
    the stick levels as fractions of the top line the sustained stick's start is 3.9 m/s above
    the random search's best)."""
    from lean.tool import flown

    lowest, key, _ = p51d_flown
    stall = tuning.performance(tuning.row_of("p51d"))["stall_speed"]
    for name, speed in lowest.items():
        assert stall < speed < key["corner_speed"], name
    with tuning.flying(tuning.row_of("p51d")):
        searched = flown.random_search(ships=1024, seconds=40.0, seed=1)
    for name, speed in searched.items():
        assert speed >= lowest[name] - 5.0, (name, speed, lowest[name])


def test_the_flown_points_agree_with_the_quasi_steady_ones(p51d_flown):
    """Flown in the game (flown.critical_points), the P-51D's corner is within 2% in speed and
    rate (the nose lags a little while full stick decelerates through it), the best sustained
    turn within 1% (held), the tightest sustained turn within 2% in speed and radius (passed:
    full stick gains energy there, so the speed cannot be held)."""
    _, key, points = p51d_flown
    assert points["corner"]["speed"] == pytest.approx(key["corner_speed"], rel=0.02)
    assert points["corner"]["rate"] == pytest.approx(key["corner_rate"], rel=0.02)
    assert points["best_sustained"]["speed"] == pytest.approx(key["best_sustained_speed"], rel=0.01)
    assert points["best_sustained"]["rate"] == pytest.approx(key["best_sustained_rate"], rel=0.01)
    assert points["tightest_sustained"]["speed"] == pytest.approx(
        key["tightest_sustained_speed"], rel=0.02
    )
    assert points["tightest_sustained"]["radius"] == pytest.approx(
        key["tightest_sustained_radius"], rel=0.02
    )


def test_no_line_is_drawn_below_its_lowest_flown_speed(p51d_flown):
    lowest = p51d_flown[0]
    chart = design._chart()
    for real in (False, True):
        with tuning.flying(tuning.row_of("p51d")):
            figure = chart.doghouse_figure("test", lowest, real)
        drawn = {line.get_gid(): line for line in figure.axes[0].lines if line.get_gid()}
        assert set(drawn) == set(lowest)
        for name, line in drawn.items():
            x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata(), dtype=float)
            finite = np.isfinite(y)
            assert finite.any()
            assert x[finite].min() >= lowest[name] / (tuning.TIME_FACTOR if real else 1.0) - 1e-6, (
                name
            )
        chart.plt.close(figure)
