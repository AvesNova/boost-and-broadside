"""The web tool: the envelope function matches the game's quasi-steady maths, the server answers
the envelope and ship endpoints and a design change, the sliders are the design inputs, saved
presets keep their row, and the sandbox socket advances a World."""

import math

import numpy as np
import pytest
import torch

from lean.game import card, core, tuning
from lean.tool import design, session, units

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


def test_a_design_change_moves_the_design_and_the_reference_stays():
    reference = tuning.reference("p51d").design
    result = design.envelope("p51d", {"g_limit": 1.3 * reference["g_limit"]}, tuning.TIME_FACTOR)
    assert result["design"]["key"]["corner_rate"] > result["reference"]["key"]["corner_rate"]
    assert result["reference"]["key"]["top_speed"] == pytest.approx(
        reference["top_speed"], rel=1e-6
    )


def test_the_sliders_are_the_design_inputs():
    """The page's sliders are tuning.DESIGN_INPUTS, in order, each starting at the preset's
    reference value with its provenance and inside its range; the page keeps no list of its own
    and no Plotly."""
    from pathlib import Path

    entries = design.presets()["p51d"]["inputs"]
    assert [e["name"] for e in entries] == [spec.name for spec in tuning.DESIGN_INPUTS]
    reference = tuning.reference("p51d")
    for e in entries:
        assert e["value"] == pytest.approx(reference.design[e["name"]])
        assert e["low"] <= e["value"] <= e["high"] and e["group"] in tuning.GROUPS
        assert e["source"] and e["method"] in ("closed", "solved", "short circuit")
    assert {e["kind"] for e in entries} - {None} <= set(units.KINDS)
    page = Path(design.__file__).with_name("index.html").read_text()
    assert "plotly" not in page.lower()
    assert not any(f'"{e["name"]}"' in page or f"'{e['name']}'" in page for e in entries)


def test_the_ship_panels_show_implied_physics_warnings_and_the_solved_row():
    """POST /api/ship: the implied physical quantities, the warnings (the P-51D's own: e and the
    structure), the solved card row with the solve's numbers; a moved design is solved and flies
    its input; an unknown input is refused."""
    client = TestClient(app)
    own = client.post("/api/ship", json={"preset": "p51d"}).json()
    assert set(own["row"]) == set(card.CARD_FIELDS)
    assert own["implied"]["max_lift_coefficient"] == pytest.approx(1.6)
    assert {w["quantity"] for w in own["warnings"]} == {"oswald", "structural_load"}
    assert own["solve"]["iterations"] == 0
    top = 1.05 * tuning.reference("p51d").design["top_speed"]
    moved = client.post("/api/ship", json={"preset": "p51d", "values": {"top_speed": top}}).json()
    assert moved["flies"]["top_speed"] == pytest.approx(top, rel=1e-4)
    assert moved["solve"]["iterations"] > 0 and moved["solve"]["condition"] > 0.0
    bad = client.post("/api/ship", json={"preset": "p51d", "values": {"cruise_speed": 300.0}})
    assert bad.status_code == 422


def test_a_slider_drag_solves_warm_and_agrees_with_a_cold_solve(monkeypatch):
    """Successive slider values solve warm from the preset's last solve (design._WARM: its row
    and Jacobian) and fly the same design as a cold solve from the preset's row."""
    monkeypatch.setattr(design, "_SOLVED", {})
    monkeypatch.setattr(design, "_WARM", {})
    reference = tuning.reference("p51d")
    for factor in (1.02, 1.03, 1.04):
        values = {"best_climb": factor * reference.design["best_climb"]}
        warm = design.solve("p51d", values)
        assert design._WARM["p51d"] is warm
    cold = tuning.design_row(reference, values, start=design.PRESET_ROWS["p51d"], fast=True)
    for name, value in cold.row.items():
        assert warm.row[name] == pytest.approx(value, rel=1e-3), name
    assert tuning.design_of(warm.row)["best_climb"] == pytest.approx(values["best_climb"],
                                                                     rel=2e-5)


def test_saved_presets_store_their_row_and_load_editable_or_fixed(tmp_path, monkeypatch):
    """A saved preset stores its inputs, the design-input table's version and its solved row.
    Under today's table it loads editable; under another it is a fixed ship (its stored row,
    the inputs measured from it) that the charts and the ship panel take as it is; a file in an
    old format (no row) or unreadable is not listed."""
    import json

    from lean.tool import app as server

    monkeypatch.setattr(server, "SAVED", tmp_path)
    client = TestClient(app)
    state = {"preset": "p51d", "values": {"g_limit": 7.0}, "time_scale": 2.0, "units": "game"}
    assert client.post("/api/saved", json={"name": "seven", "state": state}).status_code == 200
    record = json.loads((tmp_path / "seven.json").read_text())
    assert record["inputs_version"] == tuning.inputs_version()
    assert record["row"]["max_acceleration"] == pytest.approx(7.0 * card.GRAVITY)
    record["inputs_version"] = "another"
    (tmp_path / "old-table.json").write_text(json.dumps({**record, "name": "old-table"}))
    (tmp_path / "old-format.json").write_text(json.dumps({"name": "x", "state": state}))
    (tmp_path / "broken.json").write_text("{")
    listed = {r["name"]: r for r in client.get("/api/saved").json()["saved"]}
    assert set(listed) == {"seven", "old-table"}
    assert listed["seven"]["editable"] and not listed["old-table"]["editable"]
    assert listed["old-table"]["measured"]["g_limit"] == pytest.approx(7.0)
    fixed = {**record["row"], "hit_damage": 30.0}  # a row no solve would give from the inputs
    reply = client.post("/api/ship", json={"preset": "p51d", "fixed": fixed}).json()
    assert reply["row"] == pytest.approx(fixed) and reply["solve"]["iterations"] == 0
    png = client.post("/api/doghouse.png", json={"preset": "p51d", "fixed": fixed})
    assert png.status_code == 200


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
