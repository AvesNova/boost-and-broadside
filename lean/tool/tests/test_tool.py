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
    assert (grid["ps"] * card.GRAVITY).ravel() == pytest.approx(energy_rate.numpy(), rel=1e-4, abs=1e-3)


def test_envelope_key_numbers_agree_with_performance():
    """The corner is the full-stick peak: the settled-flight corner of tuning.performance, to
    the grid's resolution; the best sustained turn is held (Ps >= 0) and at most the corner's."""
    row = tuning.row_of("p51d")
    grid = tuning.envelope(row, np.linspace(20.0, 470.0, 226), np.linspace(0.0, 1.0, 101))
    flown = tuning.performance(row)
    assert grid["corner_speed"] == pytest.approx(flown["corner_speed_flown"], abs=2.5)
    assert grid["corner_rate"] == pytest.approx(flown["corner_rate"], rel=1e-2)  # grid, and spawn against mid store
    assert grid["best_sustained_rate"] <= grid["corner_rate"]
    assert grid["tightest_sustained_radius"] <= grid["best_sustained_speed"] / grid["best_sustained_rate"] + 1e-6


def test_handles_change_moves_the_design_and_the_reference_stays():
    handles = tuning.handles_of(tuning.row_of("p51d"))
    result = design.envelope("p51d", {"g_limit": 1.3 * handles["g_limit"]}, tuning.TIME_FACTOR)
    assert result["design"]["key"]["corner_rate"] > result["reference"]["key"]["corner_rate"]
    assert result["reference"]["key"]["top_speed"] == pytest.approx(handles["cruise_speed"], rel=1e-4)


def test_server_answers_the_envelope_endpoint():
    client = TestClient(app)
    presets = client.get("/api/presets").json()
    assert "p51d" in presets["presets"]
    assert client.get("/").status_code == 200
    reply = client.post("/api/envelope", json={"preset": "p51d", "time_scale": 1.0})
    assert reply.status_code == 200
    data = reply.json()
    assert math.isfinite(data["design"]["key"]["corner_rate"])
    assert len(data["design"]["speeds"]) == len(data["design"]["full"])
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
    session.run(pipe, tuning.row_of("p51d"), tuning.TIME_FACTOR, opponent=False, realtime=False, max_ticks=30)
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

    response = TestClient(app).post("/api/doghouse.png", json={"preset": "p51d", "time_scale": 2.0, "units": "real"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
