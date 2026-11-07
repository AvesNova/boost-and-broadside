"""The baselines bench (`bench.baselines_core`): E is exact on constant acceleration along a
line, on a steady turn and on a constant nose-rate change; the event rule; R is
`DeltaCode.baseline`; the iterated forecasts."""

import numpy as np
import torch

from bench import baselines_core as core
from bnb.game import api
from bnb.models.yemong.delta import DeltaCode, state_dynamics
from bnb.models.yemong.state_code import MEANS, state_code_ranges

DT = 1.0 / 30.0


def _state(**values):
    base = dict(
        x=10.0, y=-20.0, vx=50.0, vy=0.0, heading=0.0, omega=0.0, attack=0.01,
        energy=0.5, shield=0.5, wait=0.0, cooldown=0.0, log_index=0.0, rate=0.12,
    )  # fmt: skip
    base.update(values)
    return {name: np.asarray(value, np.float64) for name, value in base.items()}


def _extrapolate(previous, state, shot=False, damage=False, has_previous=True):
    return core.extrapolate(
        previous, state, DT, np.asarray(shot), np.asarray(damage), np.asarray(has_previous)
    )


def _line(t, accel=12.0, direction=0.4):
    """A constant acceleration along a line: (state at time t)."""
    speed = 40.0 + accel * t
    distance = 40.0 * t + 0.5 * accel * t * t
    c, s = np.cos(direction), np.sin(direction)
    return _state(x=distance * c, y=distance * s, vx=speed * c, vy=speed * s, heading=direction)


def _turn(t, speed=70.0, rate=1.7, attack=0.05):
    """A steady turn at constant speed: the path turns at `rate`, the nose ahead by attack."""
    path = 0.2 + rate * t
    radius = speed / rate
    return _state(
        x=radius * np.sin(path),
        y=-radius * np.cos(path),
        vx=speed * np.cos(path),
        vy=speed * np.sin(path),
        heading=path + attack,
        omega=rate,
        attack=attack,
    )


def _assert_motion(predicted, truth, keys=("vx", "vy")):
    for key in keys:
        assert abs(float(predicted[key]) - float(truth[key])) < 1e-9, key


def test_extrapolation_is_exact_on_constant_acceleration_along_a_line():
    previous, state, truth = _line(0.0), _line(DT), _line(2 * DT)
    out, (arc_x, arc_y) = _extrapolate(previous, state)
    _assert_motion(out, truth, ("x", "y", "vx", "vy", "heading", "omega"))
    assert abs(float(arc_x) - float(truth["x"])) < 1e-9
    assert abs(float(arc_y) - float(truth["y"])) < 1e-9


def test_extrapolation_is_exact_on_a_steady_turn():
    previous, state, truth = _turn(0.0), _turn(DT), _turn(2 * DT)
    out, (arc_x, arc_y) = _extrapolate(previous, state)
    _assert_motion(out, truth, ("vx", "vy", "omega", "attack"))
    assert abs(float(core.wrap(out["heading"] - truth["heading"]))) < 1e-12
    assert abs(float(arc_x) - float(truth["x"])) < 1e-9
    assert abs(float(arc_y) - float(truth["y"])) < 1e-9
    # the literal x + v dt + a dt^2 / 2 is only second order on a circle
    literal = np.hypot(out["x"] - truth["x"], out["y"] - truth["y"])
    assert 1e-6 < float(literal) < 1e-2


def test_extrapolation_is_exact_on_a_constant_nose_rate_change():
    def at(t, w0=0.8, change=-5.0):
        return _state(heading=0.1 + w0 * t + 0.5 * change * t * t, omega=w0 + change * t)

    out, _ = _extrapolate(at(0.0), at(DT))
    truth = at(2 * DT)
    assert abs(float(out["heading"] - truth["heading"])) < 1e-12
    assert abs(float(out["omega"] - truth["omega"])) < 1e-12


def test_event_rule_uses_the_rule_rate_after_an_event():
    previous = _state(shield=0.30, energy=0.50, wait=0.0, cooldown=0.0)
    state = _state(shield=0.304, energy=0.497, wait=0.0, cooldown=0.0)
    quiet, _ = _extrapolate(previous, state)
    assert np.isclose(quiet["shield"], 0.308) and np.isclose(quiet["energy"], 0.494)
    # damage on t - 1 (the shield dropped, the wait reset): the rule (the wait counts down)
    hurt = _state(shield=0.2, wait=6.0)
    out, _ = _extrapolate(previous, hurt, damage=True)
    rule = core.rule(hurt, DT)
    assert out["shield"] == rule["shield"] and out["wait"] == rule["wait"] == 6.0 - DT
    # a reset seen in the wait alone counts as the event too
    out, _ = _extrapolate(previous, hurt, damage=False)
    assert out["wait"] == rule["wait"]
    # a shot on t - 1: energy persists, the cooldown counts down
    fired = _state(energy=0.47, cooldown=1.0)
    out, _ = _extrapolate(_state(energy=0.5, cooldown=0.0), fired, shot=True)
    assert out["energy"] == 0.47 and out["cooldown"] == 0.0
    # the wait's countdown continues and stops at 0
    out, _ = _extrapolate(_state(wait=0.05), _state(wait=0.05 - DT))
    assert out["wait"] == 0.0
    out, _ = _extrapolate(_state(wait=2.0), _state(wait=2.0 - DT))
    assert np.isclose(out["wait"], 2.0 - 2 * DT)


def test_extrapolation_falls_back_to_the_rule_without_a_previous_tick():
    previous, state = _line(0.0), _turn(DT)
    out, (arc_x, arc_y) = _extrapolate(previous, state, has_previous=False)
    rule = core.rule(state, DT)
    for name in core.STATE:
        assert out[name] == rule[name]
    assert arc_x == rule["x"] and arc_y == rule["y"]


def test_rule_is_the_delta_code_baseline():
    rules, types = api.rules_of(api.PRESETS["deathmatch"]), api.ship_types(torch.device("cpu"))
    ranges, dynamics = state_code_ranges(rules, types), state_dynamics(rules, types)
    code = DeltaCode(ranges, dynamics).double()
    rng = np.random.default_rng(0)
    n = 64
    ship_type = rng.integers(0, 2, n)
    max_shield = np.asarray(ranges.max_shield)[ship_type]
    max_energy = np.asarray(ranges.max_energy)[ship_type]
    values = dict(
        x=rng.normal(0, 500, n), y=rng.normal(0, 500, n),
        velocity_x=rng.normal(0, 80, n), velocity_y=rng.normal(0, 80, n),
        heading=rng.uniform(-np.pi, np.pi, n), nose_rate=rng.normal(0, 3, n),
        angle_of_attack=rng.normal(0, 0.1, n), shield_recharge_wait=rng.choice([0.0, 0.02, 3.0], n),
        shield=rng.uniform(0, 1, n) * max_shield, energy=rng.uniform(0, 1, n) * max_energy,
        gun_cooldown=rng.integers(0, 3, n) * DT, log_refractive_index=rng.normal(0, 0.1, n),
    )  # fmt: skip
    means = torch.tensor(np.stack([values[name] for name in MEANS], -1))
    expected = code.baseline(means, torch.tensor(ship_type)).numpy()
    expected = dict(zip(MEANS, expected.T, strict=True))
    rate = np.asarray(dynamics.shield_recharge_rate)[ship_type] / max_shield
    state = dict(
        x=values["x"], y=values["y"], vx=values["velocity_x"], vy=values["velocity_y"],
        heading=values["heading"], omega=values["nose_rate"], attack=values["angle_of_attack"],
        energy=values["energy"] / max_energy, shield=values["shield"] / max_shield,
        wait=values["shield_recharge_wait"], cooldown=np.round(values["gun_cooldown"] / DT),
        log_index=values["log_refractive_index"], rate=rate,
    )  # fmt: skip
    out = core.rule(state, DT)
    pairs = dict(x="x", y="y", vx="velocity_x", vy="velocity_y", omega="nose_rate",
                 attack="angle_of_attack", wait="shield_recharge_wait",
                 log_index="log_refractive_index")  # fmt: skip
    for mine, theirs in pairs.items():
        assert np.allclose(out[mine], expected[theirs], atol=1e-9), mine
    assert np.allclose(core.wrap(out["heading"] - expected["heading"]), 0.0, atol=1e-12)
    assert np.allclose(out["shield"] * max_shield, expected["shield"], atol=1e-9)
    assert np.allclose(out["energy"] * max_energy, expected["energy"], atol=1e-9)
    assert np.allclose(out["cooldown"] * DT, expected["gun_cooldown"], atol=1e-9)


def test_forecasts_dead_reckon_exactly_on_their_own_models():
    horizons = (3, 30)
    previous, state = _line(0.0), _line(DT)
    out = core.forecasts(previous, state, DT, np.asarray(True), horizons)
    for k in horizons:
        truth = _line((1 + k) * DT)
        x, y, vx, vy = out[k]["E"]
        assert np.allclose([x, y, vx, vy], [truth[n] for n in ("x", "y", "vx", "vy")], atol=1e-8)
        x, y, vx, vy = out[k]["CV"]
        assert np.allclose([vx, vy], [state["vx"], state["vy"]])
    # R on a turn with the nose along the path: the game's arc series, exact to its order
    previous, state = _turn(0.0, attack=0.0), _turn(DT, attack=0.0)
    out = core.forecasts(previous, state, DT, np.asarray(False), horizons)
    for k in horizons:
        truth = _turn((1 + k) * DT, attack=0.0)
        for forecast in ("R", "E"):  # E is R without a previous tick
            x, y, vx, vy = out[k][forecast]
            assert np.hypot(x - truth["x"], y - truth["y"]) < 1e-6 * k
            assert np.hypot(vx - truth["vx"], vy - truth["vy"]) < 1e-9 * k
