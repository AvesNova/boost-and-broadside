"""The web tool: the envelope function matches the game's quasi-steady maths, the server answers
the envelope and ship endpoints and a design change, the sliders are the design inputs, saved
presets keep their row, and the sandbox socket advances a World. The tests of the solve (a moved
top speed or climb, warm starts, progress, failing fast, cancelling) use the documents profile:
under the card's (airframe and engine) nothing is solved."""

import math
import threading
import time

import numpy as np
import pytest
import torch

from lean.game import card, core, tuning
from lean.tool import design, session, units

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from lean.tool.app import app  # noqa: E402


def held_one_g_speed() -> float:
    """The card's type 0's lowest speed at which any stick level can hold 1 g (game units): the
    wing's maximum lift plus the thrust's normal part at the stall angle (below stall speed no
    level asks for more than the stall angle: Snap's deep angle closes to it, author
    2026-11-26) equal to 1 g, by bisection on the game's own constants."""
    p = core.per_type_constants(core.GameConfig(), torch.tensor([0]))
    low, high = 0.0, float(p.stall_speed[0])
    for _ in range(60):
        speed = 0.5 * (low + high)
        thrust = core.softmin(p.power / speed, p.static_thrust, p.thrust_softmin_exponent)
        normal = float((p.lift * speed * speed + thrust * torch.sin(p.stall))[0])
        low, high = (speed, high) if normal < card.GRAVITY else (low, speed)
    return high


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
    flown = tuning.performance(row)
    top = 1.25 * flown["top_speed"]  # the grid reaches past the top level speed, 2 m/s steps
    grid = tuning.envelope(row, np.arange(20.0, top, 2.0), np.linspace(0.0, 1.0, 101))
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
    # lines start above the lowest speed any level holds 1 g (a share of the stall speed: the
    # same at every time scale)
    stall = data["design"]["key"]["stall_speed"]
    floor = held_one_g_speed() / tuning.performance(tuning.row_of("p51d"))["stall_speed"]
    assert min(data["design"]["lowest"].values()) > floor * stall
    # the same design at 1x real time has the speeds and rates over the game's time factor
    at_game = client.post("/api/envelope", json={"preset": "p51d"}).json()
    assert data["design"]["key"]["top_speed"] == pytest.approx(
        at_game["design"]["key"]["top_speed"] / tuning.TIME_FACTOR, rel=1e-3
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
    ship = frames[-1]["ships"][0]
    assert ship["vision_range"] > 0 and 0 <= ship["vision_eccentricity"] < 1
    assert isinstance(ship["seen"], bool)


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


def test_the_sandbox_changes_the_hit_model_live():
    """{"hit_model": {...}} reaches the running World (no restart): the answer carries the
    sliders (core.HIT_MODEL) with the new value, the preset list is card.DEFLECTOR_PRESETS, and
    the World's next tick flies it; a reset keeps it."""
    client = TestClient(app)
    with client.websocket_connect("/api/sandbox") as socket:
        socket.send_json({"preset": "p51d", "time_scale": 2.0, "opponent": True})
        setup = socket.receive_json()
        assert [e["name"] for e in setup["hit_model"]] == [e.name for e in core.HIT_MODEL]
        assert setup["deflector_presets"] == card.DEFLECTOR_PRESETS
        socket.send_json({"hit_model": {"deflector_side": 7.5, "hit_radius": 15.0}})
        while (message := socket.receive_json())["type"] != "hit_model":
            pass
        values = {e["name"]: e["value"] for e in message["hit_model"]}
        assert values["deflector_side"] == 7.5 and values["hit_radius"] == 15.0
        socket.send_json({"stop": True})


def test_the_world_keeps_its_hit_model_across_a_reset():
    from lean.sandbox.world import World

    world = World(core.GameConfig(), opponent_on=False)
    world.set_hit_model(deflector_front=5.0)
    world.advance(0, 0, False)
    world.reset()
    assert core.hit_model_of(world.state.constants)["deflector_front"] == 5.0
    with pytest.raises(KeyError):
        world.set_hit_model(gun_range=1.0)


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
    # a negative g limit has no design (the solve's "impossible"); the stall speed of 400 m/s
    # used here before 2026-11-26 solves, and its 422 came from the chart's colour scale
    bad = client.post("/api/doghouse.png", json={"preset": "p51d", "values": {"g_limit": -1.0}})
    assert bad.status_code == 422 and bad.json()["error"]


def test_a_design_change_moves_the_design_and_the_reference_stays():
    """A moved g limit (the card's profile) turns harder; the reference is still the card's
    (its top speed is what the card's row flies: a readout under airframe and engine)."""
    reference = tuning.reference("p51d").design
    result = design.envelope("p51d", {"g_limit": 1.3 * reference["g_limit"]}, tuning.TIME_FACTOR)
    assert result["design"]["key"]["corner_rate"] > result["reference"]["key"]["corner_rate"]
    assert result["reference"]["key"]["top_speed"] == pytest.approx(
        tuning.performance(tuning.row_of("p51d"))["top_speed"], rel=1e-6
    )


def test_the_sliders_are_the_design_inputs():
    """Per profile (tuning.PROFILES), the page's sliders are its table, in order, each starting
    at the preset's reference value under it with its provenance and inside its range; the
    preset names its own profile (the card's: airframe and engine); the page keeps no list of its
    own, no profile's names and no Plotly."""
    from pathlib import Path

    preset = design.presets()["p51d"]
    assert preset["profile"] == tuning.REAL_AIRCRAFT["p51d"].profile == "airframe_and_engine"
    assert set(preset["profiles"]) == set(tuning.PROFILES) == set(design.profiles())
    page = Path(design.__file__).with_name("index.html").read_text()
    assert "plotly" not in page.lower()
    for profile, entries in preset["profiles"].items():
        assert [e["name"] for e in entries] == [s.name for s in tuning.PROFILES[profile].inputs]
        reference = tuning.reference("p51d", profile)
        for e in entries:
            assert e["value"] == pytest.approx(reference.design[e["name"]])
            assert e["low"] <= e["value"] <= e["high"] and e["group"] in tuning.GROUPS
            assert e["source"] and e["method"] in ("closed", "solved", "short circuit")
        assert {e["kind"] for e in entries} - {None} <= set(units.KINDS)
        assert not any(f'"{e["name"]}"' in page or f"'{e['name']}'" in page for e in entries)
        assert f'"{profile}"' not in page


STRUCTURE = next(c for c in tuning.PLAUSIBILITY if c.quantity == "structural_load")


def g_limit_at(reply: dict, factor: float) -> float:
    """A g limit `factor` times the structural check's bound, from a /api/ship reply's design
    (its g limit and implied structural load; the load is proportional to the g limit)."""
    return reply["flies"]["g_limit"] * factor * STRUCTURE.high / reply["implied"]["structural_load"]


def test_a_profile_is_chosen_per_request(monkeypatch):
    """A request names its profile: its inputs are that table's (a moved top speed under the
    documents is solved and flown; a readout is not an input and is refused), the warnings
    point at its inputs (a g limit a tenth above the structure's warns on the g limit, a
    tenth below does not), and an unknown profile is refused."""
    monkeypatch.setattr(design, "_SOLVED", {})
    client = TestClient(app)
    speed = 0.98 * tuning.reference("p51d", "documents").design["top_speed"]
    request = {"preset": "p51d", "profile": "documents", "values": {"top_speed": speed}}
    moved = client.post("/api/ship", json=request).json()
    assert moved["flies"]["top_speed"] == pytest.approx(speed, rel=1e-4)
    assert moved["solve"]["condition"] < 20.0
    inputs = {spec.name for spec in tuning.PROFILES["documents"].inputs}
    assert {w["input"] for w in moved["warnings"]} <= inputs
    for factor, warns in ((1.1, True), (0.9, False)):
        values = {"top_speed": speed, "g_limit": g_limit_at(moved, factor)}
        reply = client.post("/api/ship", json={**request, "values": values}).json()
        assert {w["input"] for w in reply["warnings"]} <= inputs
        assert ("g_limit" in {w["input"] for w in reply["warnings"]}) == warns, factor
    not_an_input = {**request, "values": {"sustained_speed": 100.0}}  # a readout, not an input
    assert client.post("/api/ship", json=not_an_input).status_code == 422
    unknown = client.post("/api/ship", json={**request, "profile": "nonesuch"})
    assert unknown.status_code == 422
    png = client.post("/api/doghouse.png", json=request)
    assert png.status_code == 200


def test_the_ship_panels_show_implied_physics_warnings_and_the_solved_row():
    """POST /api/ship: the implied physical quantities, the warnings (the structure's exactly
    when the implied load is above the check's bound: the P-51D's own and a g limit a tenth
    above the structure's), the solved card row with the solve's numbers; a moved design (a top
    speed, under the documents) is solved and flies its input; an unknown input is refused."""
    client = TestClient(app)
    own = client.post("/api/ship", json={"preset": "p51d"}).json()
    assert set(own["row"]) == set(card.CARD_FIELDS)
    assert own["implied"]["max_lift_coefficient"] == pytest.approx(tuning.P51D.max_lift_coefficient)
    over = own["implied"]["structural_load"] > STRUCTURE.high
    assert ("structural_load" in {w["quantity"] for w in own["warnings"]}) == over
    beyond = {"preset": "p51d", "values": {"g_limit": g_limit_at(own, 1.1)}}
    strained = client.post("/api/ship", json=beyond).json()
    assert "structural_load" in {w["quantity"] for w in strained["warnings"]}
    assert own["solve"]["iterations"] == 0
    top = 1.05 * tuning.reference("p51d", "documents").design["top_speed"]
    request = {"preset": "p51d", "profile": "documents", "values": {"top_speed": top}}
    moved = client.post("/api/ship", json=request).json()
    assert moved["flies"]["top_speed"] == pytest.approx(top, rel=1e-4)
    assert moved["solve"]["iterations"] > 0 and moved["solve"]["condition"] > 0.0
    bad = client.post("/api/ship", json={"preset": "p51d", "values": {"cruise_speed": 300.0}})
    assert bad.status_code == 422


def test_a_slider_drag_solves_warm_and_agrees_with_a_cold_solve(monkeypatch):
    """Successive slider values solve warm from the preset's last solve (design._WARM: its row
    and Jacobian) and fly the same design as a cold solve from the preset's row."""
    monkeypatch.setattr(design, "_SOLVED", {})
    monkeypatch.setattr(design, "_WARM", {})
    reference = tuning.reference("p51d", "documents")
    for factor in (1.02, 1.03, 1.04):
        values = {"best_climb": factor * reference.design["best_climb"]}
        warm = design.solve("p51d", values, profile="documents")
        assert design._WARM[("p51d", "documents")] is warm
    cold = tuning.design_row(reference, values, start=design.PRESET_ROWS["p51d"], fast=True)
    for name, value in cold.row.items():
        assert warm.row[name] == pytest.approx(value, rel=1e-3), name
    assert tuning.design_of(warm.row)["best_climb"] == pytest.approx(values["best_climb"], rel=2e-5)


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
    profile = tuning.REAL_AIRCRAFT["p51d"].profile  # the preset's own
    assert record["inputs_version"] == tuning.inputs_version(tuning.PROFILES[profile].inputs)
    assert record["profile"] == profile
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
    card as it stands, no stored numbers): every line is flown somewhere between the lowest speed
    any level holds 1 g (held_one_g_speed: since 2026-11-26 the thrust's normal part counts in
    the hold, so a line can start a little below stall speed) and the corner speed, and an
    independent random search of control sequences (flown.random_search) flies no line more than
    5 m/s below its measured start, so the search the charts use finds the lowest states near
    enough (3 m/s before 2026-11-24; with the stick levels as fractions of the top line the
    then-derived 0.88 level's start was 3.9 m/s above the random search's best)."""
    from lean.tool import flown

    lowest, key, _ = p51d_flown
    with tuning.flying(tuning.row_of("p51d")):
        floor = held_one_g_speed()
    for name, speed in lowest.items():
        assert floor < speed < key["corner_speed"], name
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
        # the envelope too: the most turn of any level, from the lowest of their starts
        assert set(drawn) == set(lowest) | {"envelope"}
        starts = {**lowest, "envelope": min(lowest[name] for name in ("sustain", "max", "snap"))}
        for name, line in drawn.items():
            x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata(), dtype=float)
            finite = np.isfinite(y)
            assert finite.any()
            assert x[finite].min() >= starts[name] / (tuning.TIME_FACTOR if real else 1.0) - 1e-6, (
                name
            )
        chart.plt.close(figure)


# --------------------------------------------------------------------------- progress, failing fast


def test_the_solve_reports_progress_and_the_result_is_the_same_without_watching():
    reference = tuning.reference("p51d", "documents")
    values = {"best_climb": 1.05 * reference.design["best_climb"]}
    start = design.PRESET_ROWS["p51d"]
    steps = []
    watched = tuning.design_row(
        reference,
        values,
        start=start,
        fast=True,
        progress=steps.append,
        patience=6,
        deadline=time.monotonic() + 30,
    )
    plain = tuning.design_row(reference, values, start=start, fast=True)
    assert steps and [s.iteration for s in steps] == list(range(len(steps)))
    first = steps[0]
    assert first.names and set(first.names) <= {spec.name for spec in tuning.SOLVED}
    assert len(first.residuals) == len(first.names) and first.step is None
    assert first.worst == max(abs(x) for x in first.residuals) and first.tolerance == 1e-5
    assert steps[-1].worst < steps[0].worst and steps[1].step is not None
    assert watched.row == plain.row and watched.iterations == plain.iterations


def test_an_impossible_design_fails_fast_naming_the_input(monkeypatch):
    monkeypatch.setattr(design, "TIME_LIMIT", 3.0)
    monkeypatch.setattr(design, "_SOLVED", {})
    monkeypatch.setattr(design, "_WARM", {})
    climb = tuning.reference("p51d", "documents").design["best_climb"]
    began = time.monotonic()
    with pytest.raises(tuning.SolveFailed) as raised:
        design.solve("p51d", {"best_climb": 20.0 * climb}, profile="documents")
    assert time.monotonic() - began < design.TIME_LIMIT + 1.5
    report = design.failure_report(raised.value, "p51d", "documents", "real_imperial", 2.0)
    assert report["reason"] in ("time", "stalled") and report["nearest"]
    worst = report["unmet"][0]  # the largest miss first, in the chosen units
    assert worst["name"] == "best_climb" and worst["unit"] == "ft/min"
    assert worst["achieved"] < worst["target"] and worst["error"] < -0.5
    assert worst["target"] == pytest.approx(
        20.0 * climb * units.factor("climb", "real_imperial", 2.0)
    )
    assert worst["label"] in report["message"]


def test_a_negative_or_flyless_design_is_impossible_at_once():
    began = time.monotonic()
    with pytest.raises(tuning.SolveFailed) as negative:
        design.solve("p51d", {"best_climb": -3.0}, profile="documents")
    with pytest.raises(tuning.SolveFailed) as flyless:
        design.solve("p51d", {"stall_speed": 400.0}, profile="documents")
    assert time.monotonic() - began < 1.0
    assert negative.value.reason == flyless.value.reason == "impossible"
    assert (
        "best_climb"
        in design.failure_report(negative.value, "p51d", "documents", "game", 2.0)["message"]
    )
    assert (
        "stall_speed"
        in design.failure_report(flyless.value, "p51d", "documents", "game", 2.0, ["stall_speed"])[
            "message"
        ]
    )


def test_the_failure_reaches_the_page_as_json_and_a_socket_message(monkeypatch):
    monkeypatch.setattr(design, "TIME_LIMIT", 2.0)
    client = TestClient(app)
    request = {
        "preset": "p51d",
        "profile": "documents",
        "values": {"best_climb": -3.0},
        "units": "real_metric",
    }
    bad = client.post("/api/ship", json=request)
    assert bad.status_code == 422 and bad.json()["failure"]["reason"] == "impossible"
    with client.websocket_connect("/api/solve") as socket:
        socket.send_json(request)
        message = socket.receive_json()
    assert message["type"] == "failed" and "best_climb" in message["error"]


def test_the_socket_streams_progress_then_done_and_a_reset_answers_at_once(monkeypatch):
    monkeypatch.setattr(design, "_SOLVED", {})
    monkeypatch.setattr(design, "_WARM", {})
    client = TestClient(app)
    climb = tuning.reference("p51d", "documents").design["best_climb"]
    with client.websocket_connect("/api/solve") as socket:
        moved = {"best_climb": 1.04 * climb}
        socket.send_json({"preset": "p51d", "profile": "documents", "values": moved})
        kinds = []
        while (message := socket.receive_json())["type"] == "progress":
            kinds.append(message)
        assert message["type"] == "done" and kinds
        assert {"iteration", "error", "residuals", "elapsed", "tolerance"} <= set(kinds[0])
    began = time.monotonic()
    with client.websocket_connect("/api/solve") as socket:  # Reset to preset: nothing moved
        socket.send_json({"preset": "p51d", "values": {}})
        assert socket.receive_json()["type"] == "done"
    assert time.monotonic() - began < 0.5


def test_a_cancelled_solve_stops_within_an_evaluation_and_keeps_the_warm_start(monkeypatch):
    monkeypatch.setattr(design, "_SOLVED", {})
    monkeypatch.setattr(design, "_WARM", {})
    monkeypatch.setattr(design, "TIME_LIMIT", 60.0)
    documents = tuning.reference("p51d", "documents").design
    warm = design.solve("p51d", {"best_climb": 1.03 * documents["best_climb"]}, profile="documents")
    kept = dict(design._WARM)
    cancel, outcome = threading.Event(), {}
    top = documents["top_speed"]

    def work():
        try:
            design.solve("p51d", {"top_speed": 5.0 * top}, profile="documents", cancel=cancel)
        except tuning.SolveCancelled:
            outcome["at"] = time.monotonic()

    thread = threading.Thread(target=work)
    thread.start()
    time.sleep(0.6)
    asked = time.monotonic()
    cancel.set()
    thread.join(5.0)
    assert not thread.is_alive() and outcome["at"] - asked < 0.5
    assert design._WARM == kept and design._WARM[("p51d", "documents")] is warm


def test_a_new_message_on_the_solve_socket_cancels_the_solve(monkeypatch):
    monkeypatch.setattr(design, "_SOLVED", {})
    monkeypatch.setattr(design, "_WARM", {})
    monkeypatch.setattr(design, "TIME_LIMIT", 60.0)
    top = tuning.reference("p51d", "documents").design["top_speed"]
    seen = []
    original = tuning.design_row

    def spy(*args, **kwargs):
        seen.append(kwargs["cancel"])
        return original(*args, **kwargs)

    monkeypatch.setattr(tuning, "design_row", spy)
    with TestClient(app).websocket_connect("/api/solve") as socket:
        moved = {"top_speed": 5.0 * top}
        socket.send_json({"preset": "p51d", "profile": "documents", "values": moved})
        assert socket.receive_json()["type"] == "progress"
        socket.send_text("next")  # the page's newer request
        asked = time.monotonic()
        while not seen[0].is_set() and time.monotonic() - asked < 2.0:
            time.sleep(0.01)
        assert seen[0].is_set() and time.monotonic() - asked < 0.5
