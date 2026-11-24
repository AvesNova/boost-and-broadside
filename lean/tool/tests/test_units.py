"""The tool's units: one conversion module, round trips, the endpoints, the sliders' handles."""

import math

import pytest
from fastapi.testclient import TestClient

from lean.tool import design, units
from lean.tool.app import app


@pytest.mark.parametrize("system", units.SYSTEMS)
@pytest.mark.parametrize("time_scale", [1.0, 2.0, 3.0])
def test_conversions_round_trip(system, time_scale):
    for kind in units.KINDS:
        game = 123.456
        shown = units.to_display(kind, game, system, time_scale)
        assert units.from_display(kind, shown, system, time_scale) == pytest.approx(game, rel=1e-12)


def test_the_known_factors():
    assert units.factor("speed", "game", 2.0) == 1.0
    assert units.factor("speed", "real_metric", 2.0) == pytest.approx(0.5)
    assert units.factor("speed", "real_imperial", 2.0) == pytest.approx(0.5 * 2.236936292)
    assert units.factor("length", "real_imperial", 2.0) == pytest.approx(3.280839895)
    assert units.factor("climb", "real_imperial", 1.0) == pytest.approx(196.8503937)  # 1 m/s
    assert units.factor("acceleration", "real_metric", 2.0) == pytest.approx(0.25)
    assert units.factor("time", "real_metric", 2.0) == pytest.approx(2.0)
    for system in units.SYSTEMS:  # deg/s in all three
        assert units.label("angular_rate", system) == "deg/s"
    assert units.factor("angular_rate", "game", 2.0) == pytest.approx(180 / math.pi)
    assert units.system_of("real") == "real_metric" and units.system_of(None) == "game"
    with pytest.raises(ValueError):
        units.system_of("furlongs")


@pytest.mark.parametrize("system", units.SYSTEMS)
def test_slider_values_shown_convert_back_to_the_same_handle(system):
    """The page shows game * scale * k**-power and divides it back: the handle is unchanged."""
    described = units.descriptors()[system]
    for entry in design.inputs("p51d"):
        if entry["kind"] is None:
            continue
        d = described[entry["kind"]]
        scale = d["scale"] * 2.0 ** -d["time_power"]  # the page's scaleOf at 2x
        assert scale == pytest.approx(units.factor(entry["kind"], system, 2.0))
        shown = float(f"{entry['value'] * scale:.12g}")
        assert shown / scale == pytest.approx(entry["value"], rel=1e-9)


def test_every_unit_string_of_the_inputs_has_a_kind_or_is_plain():
    for entry in design.inputs("p51d"):
        assert (entry["kind"] is None) == (entry["unit"] not in units.UNIT_KINDS)
    assert "top level speed" in [e["label"] for e in design.inputs("p51d")]


def test_presets_carry_the_units():
    data = TestClient(app).get("/api/presets").json()["units"]
    assert data["order"] == list(units.SYSTEMS) and set(data["systems"]) == set(units.SYSTEMS)
    assert data["systems"]["real_imperial"]["speed"]["label"] == "mph"


@pytest.mark.parametrize("system", [*units.SYSTEMS, "real"])
def test_the_chart_endpoints_accept_every_system(system):
    client = TestClient(app)
    for url in ("/api/doghouse.png", "/api/sustained.png"):
        reply = client.post(url, json={"preset": "p51d", "time_scale": 2.0, "units": system})
        assert reply.status_code == 200 and reply.content[:4] == b"\x89PNG"


def test_an_unknown_system_is_an_error():
    reply = TestClient(app).post("/api/doghouse.png", json={"preset": "p51d", "units": "furlongs"})
    assert reply.status_code == 422
