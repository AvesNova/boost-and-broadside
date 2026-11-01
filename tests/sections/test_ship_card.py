"""The ship card's acceptance (ship-card spec 7): round trip, independence, tick-rate
robustness, the cruise fix, infeasible sets, the chart tool, and every flown handle against
an independent flight of the game.

The solves are slow on the CPU (minutes): one batch solves every card these tests need
(ship_card_cases.cases(), shared by the tests in this module)."""

import math
import re

import pytest
import ship_card_cases as cases_module
import torch

from bnb.game import api
from bnb.game.handling import FLIGHT, FLOWN, NAMES, Handles, Infeasible, measure, solve
from bnb.game.handling.manoeuvres import batched, cruise_speed
from bnb.game.handling.solver import ERROR_FLOOR
from bnb.game.physics.flight import airframe_of, flight_tick
from bnb.game.state import Fields, Ships, replace

ACCEPTANCE = 0.005
pytestmark = pytest.mark.slow


def _error(name, measured, target):
    """Relative error, against the solver's floor for a flown handle near zero."""
    scale = target.abs().clamp_min(ERROR_FLOOR.get(name, 0.0))
    return ((measured - target) / scale).abs()


# ------------------------------------------------------------------ round trip

def test_round_trip_default_and_varied_cards():
    """handles -> card -> measured handles within 0.5%, for the default card's handles and
    8 varied cards (both types each)."""
    cases = cases_module.cases()
    rows = cases.round_trip
    assert cases.solution.converged[rows].all(), cases.solution.failures
    for name in NAMES:
        measured, target = getattr(cases.measured, name)[rows], getattr(cases.targets, name)[rows]
        error = _error(name, measured, target)
        assert error.max() < ACCEPTANCE, (name, error.max().item())


def test_varied_cards_span_the_envelope():
    """The varied cards' handles differ from the default's by more than the acceptance band
    in every flown handle (they are not one card)."""
    cases = cases_module.cases()
    for name in FLOWN:
        varied = getattr(cases.varied_handles, name)
        relative = (varied / getattr(cases.default, name)).log().abs()
        assert relative.max() > 0.05, name


# ------------------------------------------------------------------ independence

# Perturbations the solver cannot meet (ship-card-notes.md, "Independence"): holding the
# bleed rate and the other handles, the sustained turn cannot rise 5%; the solve drives the
# bleed exponent below 0.8 and still misses the climb (and fails, with that reason).
UNREACHABLE = {("sustained_turn_rate", 1.0)}


def test_independence():
    """Each flight handle by -5% and +5%, re-solved: it moves by the perturbation and every
    other measured handle stays within 0.5% of the default's; the UNREACHABLE ones fail
    with a reason instead."""
    cases = cases_module.cases()
    first = cases.independence.start
    for i, (moved, sign) in enumerate(cases.labels):
        row = first + i
        if (moved, sign) in UNREACHABLE:
            assert not cases.solution.converged[row].all()
            assert any(f.startswith(f"card {row} ") and "no convergence" in f
                       for f in cases.solution.failures)
            continue
        assert cases.solution.converged[row].all(), (moved, sign)
        for name in FLIGHT:
            base = getattr(cases.default, name)
            want = base * (1.0 + sign * cases_module.PERTURBATION) if name == moved else base
            error = _error(name, getattr(cases.measured, name)[row], want)
            assert error.max() < ACCEPTANCE, (moved, sign, name, error.max().item())


def test_conditioning_is_reported():
    cases = cases_module.cases()
    condition = cases.solution.condition
    assert condition.shape == cases.targets.ceiling.shape
    assert torch.isfinite(condition).all() and (condition >= 1.0).all()


# ------------------------------------------------------------------ tick rates

# The nose reversal of the fastest noses (0.15 s, under four ticks at 25 Hz) differs by up
# to 0.6% between 25 and 60 Hz, and by up to 0.54% from a 600 Hz flight at 25 Hz: the game's
# nose integrator, not the measurement (ship-card-notes.md, "Tick rates").
TICK_RATE_BAR = dict(nose_reversal_time=0.01)


def test_handles_agree_across_tick_rates():
    """The same parameters (the default card and the solved varied cards) give flight
    handles that agree within 0.5% at 25, 30 and 60 Hz."""
    cases = cases_module.cases()
    rows = cases.round_trip
    cards = cases_module.double(api.ShipTypes(**{k: v[rows]
                                                 for k, v in vars(cases.solution.types).items()}))
    by_rate = {30: Handles(**{name: getattr(cases.measured, name)[rows] for name in NAMES})}
    for rate in (25, 60):
        by_rate[rate] = measure(cards, cases_module.rules(rate))
    for name in FLIGHT:
        values = torch.stack([getattr(handles, name) for handles in by_rate.values()])
        spread = (values.amax(0) - values.amin(0)) / values.abs().mean(0)
        assert spread.max() < TICK_RATE_BAR.get(name, ACCEPTANCE), (name, spread.max().item())


def test_unsupported_weapon_cadence_is_refused():
    """The default fire interval (1/15 s) is not whole at 25 Hz: the solve refuses with the
    reason before flying."""
    cases = cases_module.cases()
    with pytest.raises(Infeasible, match="fire interval.*25 Hz"):
        solve(cases.default, cases_module.rules(25))


# ------------------------------------------------------------------ the cruise fix

# Solved cards whose cruise varies by more than 2% across the store (ship-card-notes.md,
# "The cruise fix"): the static-thrust cap shaves the empty store's engine power more than
# the full store's. Reported, not fixed (spec 2).
OVER_TWO_PERCENT = {3: "varied card 2", 10: "stall speed +5%", 11: "g limit -5%"}


def test_cruise_varies_little_across_the_store():
    """Cruise (the straight coasting equilibrium) across the store, empty to full: within 2%
    for the re-solved default card and every converged solved card but OVER_TWO_PERCENT,
    within 5% (the author's "too much") for those but the stall speed's +5% (whose bleed
    exponent fell below 0.9 and engine power tripled). The default card is reported."""
    cases = cases_module.cases()
    grid = torch.linspace(0.0, 1.0, 51, dtype=torch.float64)

    def variation(types):
        speeds = cruise_speed(cases.rules, types, types.max_energy[..., None] * grid)
        return (speeds.amax(-1) - speeds.amin(-1)) / speeds.mean(-1)

    solved = variation(cases_module.double(cases.solution.types))
    converged = cases.solution.converged.all(-1)
    over = {row for row in range(len(solved)) if converged[row] and solved[row].max() > 0.02}
    assert over == set(OVER_TWO_PERCENT), {row: solved[row].tolist() for row in over}
    for row in over - {10}:
        assert solved[row].max() < 0.05, (row, solved[row].tolist())
    default = variation(batched(api.ship_types(dtype=torch.float64)))
    print(f"default card's cruise variation: fighter {default[0, 0]:.2%}, "
          f"heavy {default[0, 1]:.2%}; re-solved {solved[0, 0]:.2%}, {solved[0, 1]:.2%}")
    assert default.max() < 0.02


# ------------------------------------------------------------------ infeasible sets

def test_infeasible_g_limit():
    cases = cases_module.cases()
    with pytest.raises(Infeasible, match="g limit"):
        solve(cases.default.replace(g_limit=torch.tensor([8.0, 0.9])), cases.rules)


def test_infeasible_cruise_below_stall():
    cases = cases_module.cases()
    with pytest.raises(Infeasible, match="cruise speed must be above the stall speed"):
        solve(cases.default.replace(cruise_speed=cases.default.stall_speed * 0.9), cases.rules)


def test_infeasible_sustained_turn_above_the_peak():
    cases = cases_module.cases()
    with pytest.raises(Infeasible, match="sustained turn rate must be below"):
        solve(cases.default.replace(sustained_turn_rate=cases.default.sustained_turn_rate * 3.0),
              cases.rules)


def test_infeasible_without_convergence():
    """One Newton iteration is not enough for the default card's handles (its first guess
    misses by more than 0.5%)."""
    cases = cases_module.cases()
    with pytest.raises(Infeasible, match="no convergence within 1 iterations"):
        solve(cases.default, cases.rules, iterations=1)


def test_infeasible_gun_range():
    cases = cases_module.cases()
    with pytest.raises(Infeasible, match="gun range"):
        solve(cases.default.replace(gun_range=cases.default.gun_range * 10.0), cases.rules)


# ------------------------------------------------------------------ independent flights

DEFAULT = api.ship_types(dtype=torch.float64)


def _one_ship(speed, energy, ship_type, angle=0.0, nose_rate=0.0):
    """One ship (1, 1), float64, along +x with the nose `angle` above the path."""
    def value(v):
        return torch.tensor([[float(v)]], dtype=torch.float64)

    zero, angle = value(0.0), value(angle)
    flag = torch.zeros((1, 1), dtype=torch.bool)
    integer = torch.zeros((1, 1), dtype=torch.int32)
    return Ships(x=zero, y=zero, heading_cos=torch.cos(angle), heading_sin=torch.sin(angle),
                 proper_speed=value(speed), angle_of_attack=angle,
                 angle_of_attack_cos=torch.cos(angle), angle_of_attack_sin=torch.sin(angle),
                 nose_rate=value(nose_rate), energy=value(energy), log_refractive_index=zero,
                 log_index_gradient_x=zero, log_index_gradient_y=zero, shield=zero,
                 shield_recharge_wait=zero, alive=~flag, fired=flag, spawned=flag,
                 respawn_wait=integer, gun_cooldown=integer, next_slot=integer,
                 ship_type=integer + ship_type)


FIELDS = Fields(x=torch.tensor([[1e9]], dtype=torch.float64),
                y=torch.zeros((1, 1), dtype=torch.float64),
                radius=torch.ones((1, 1), dtype=torch.float64),
                band_width=torch.ones((1, 1), dtype=torch.float64),
                level=torch.ones((1, 1), dtype=torch.int32))


class Pilot:
    """A plain loop over the game's flight tick for one ship."""

    def __init__(self, ship_type, tick_length=1.0 / 30.0):
        self.rules = cases_module.rules()
        self.airframe = airframe_of(DEFAULT, torch.tensor([[ship_type]], dtype=torch.int32))
        self.tick = tick_length

    def step(self, ship, stick=0.0, power=0.0):
        control = torch.tensor([[stick]], dtype=torch.float64), torch.tensor([[power]],
                                                                             dtype=torch.float64)
        return flight_tick(self.rules.flight, self.airframe, self.tick, ship, *control, FIELDS)

    def settle_nose(self, speed, energy, stick, ship_type, seconds=1.0):
        """The nose settled at a stick with the speed and store held (reset every tick)."""
        ship = _one_ship(speed, energy, ship_type)
        for _ in range(round(seconds / self.tick)):
            ship = self.step(ship, stick)
            ship = _one_ship(speed, energy, ship_type, float(ship.angle_of_attack),
                             float(ship.nose_rate))
        return ship


def _heading(ship):
    return math.atan2(float(ship.heading_sin), float(ship.heading_cos))


@pytest.mark.parametrize("ship_type", [0, 1])
def test_each_flown_handle_matches_an_independent_flight(ship_type):
    """The default card's handles from measure() against hand computations (direct) and
    plain one-ship flights of the game (flown), each measured its own way: long flights for
    the terminal states, a fine tick (1/3000 s) for the rates at a manoeuvre's start, the
    nose settled by flying with the speed held."""
    cases = cases_module.cases()
    measured = {name: float(getattr(cases.default, name)[ship_type]) for name in NAMES}
    gravity = cases.rules.flight.gravity
    lift, acceleration = float(DEFAULT.max_lift_coefficient[ship_type]), float(
        DEFAULT.max_acceleration[ship_type])
    full = float(DEFAULT.max_energy[ship_type])
    assert math.isclose(measured["stall_speed"], math.sqrt(gravity / lift), rel_tol=1e-12)
    assert math.isclose(measured["g_limit"], acceleration / gravity, rel_tol=1e-12)
    assert measured["ceiling"] == full
    assert math.isclose(measured["stall_angle"], math.radians(15.0), rel_tol=1e-12)
    assert measured["vision_range"] == 1024.0

    pilot, fine = Pilot(ship_type), Pilot(ship_type, 1.0 / 3000.0)
    # Cruise: coast straight at a full store for 200 s.
    ship = _one_ship(60.0, full, ship_type)
    for _ in range(6000):
        ship = pilot.step(ship)
    cruise = float(ship.proper_speed)
    assert math.isclose(measured["cruise_speed"], cruise, rel_tol=1e-6)
    # Climb: from empty at cruise until the store reaches 98%.
    ship, time, target = _one_ship(cruise, 0.0, ship_type), 0.0, 0.98 * full
    while float(ship.energy) < target:
        before, ship, time = float(ship.energy), pilot.step(ship), time + pilot.tick
    climb = time - pilot.tick * (float(ship.energy) - target) / (float(ship.energy) - before)
    assert math.isclose(measured["climb_time"], climb, rel_tol=1e-3)

    def start_rate(ship, stick=0.0, power=0.0, specific=False):
        after = fine.step(ship, stick, power)
        if specific:
            return (0.5 * float(after.proper_speed) ** 2 + float(after.energy)
                    - 0.5 * float(ship.proper_speed) ** 2 - float(ship.energy)) / fine.tick
        return (float(after.proper_speed) - float(ship.proper_speed)) / fine.tick

    stall = measured["stall_speed"]
    low_speed = start_rate(_one_ship(stall, 0.0, ship_type))
    assert math.isclose(measured["low_speed_acceleration"], low_speed, rel_tol=ACCEPTANCE)
    dive = start_rate(_one_ship(cruise, 0.5 * full, ship_type), power=-1.0)
    assert math.isclose(measured["dive_acceleration"], dive, rel_tol=ACCEPTANCE)
    zoom = start_rate(_one_ship(cruise, 0.5 * full, ship_type), power=1.0)
    assert math.isclose(measured["zoom_acceleration"], zoom, rel_tol=ACCEPTANCE)

    corner = math.sqrt(acceleration / lift)
    turning = 0.25 * full
    settled = fine.settle_nose(corner, turning, 1.0, ship_type)
    bleed = start_rate(settled, stick=1.0, specific=True)
    assert math.isclose(measured["bleed_rate"], bleed, rel_tol=ACCEPTANCE)

    # Sustained turn: 60 s at full stick from cruise, the store held below the knee; the
    # path's rate over the last 10 s.
    ship, path, samples = _one_ship(cruise, turning, ship_type), 0.0, []
    for tick in range(1800):
        after = replace(pilot.step(ship, 1.0), energy=ship.energy)
        turned = math.remainder(_heading(after) - _heading(ship), 2.0 * math.pi)
        path += turned - (float(after.angle_of_attack) - float(ship.angle_of_attack))
        ship = after
        if tick + 1 in (1500, 1800):
            samples.append(path)
    sustained = (samples[1] - samples[0]) / (300 * pilot.tick)
    assert math.isclose(measured["sustained_turn_rate"], sustained, rel_tol=1e-3)

    # Nose reversal at a fine tick, from a nose settled one way to 90% of the other way.
    reversal_pilot = Pilot(ship_type, 1.0 / 600.0)
    other_way = reversal_pilot.settle_nose(corner, turning, 1.0, ship_type)
    target = 0.9 * float(other_way.angle_of_attack)
    ship, time = reversal_pilot.settle_nose(corner, turning, -1.0, ship_type), 0.0
    while float(ship.angle_of_attack) < target:
        before, ship = float(ship.angle_of_attack), reversal_pilot.step(ship, 1.0)
        time += reversal_pilot.tick
    reversal = time - reversal_pilot.tick * (float(ship.angle_of_attack) - target) / (
        float(ship.angle_of_attack) - before)
    assert math.isclose(measured["nose_reversal_time"], reversal, rel_tol=ACCEPTANCE)


# ------------------------------------------------------------------ the chart tool

def _printed(text: str, label: str) -> list[float]:
    for line in text.splitlines():
        if line.strip().startswith(label):
            return [float(v) for v in re.findall(r"[-+]?\d[\d.]*(?:e[-+]?\d+)?", line[48:])]
    raise AssertionError(f"no row {label!r}")


def test_chart_tool_runs_and_its_shown_quantities_hold(tmp_path, capsys):
    pytest.importorskip("matplotlib", reason="the charts need the tools extra")
    from bnb.cli import ShipCard, ship_card
    from bnb.game.handling.chart import CHARTS

    ship_card(ShipCard(out=str(tmp_path)))
    text = capsys.readouterr().out
    for chart in CHARTS:
        assert (tmp_path / chart).stat().st_size > 10_000, chart
    for name in ("stall_speed", "cruise_speed", "sustained_turn_rate"):
        assert _printed(text, name)
    for t in (0, 1):
        lift = float(DEFAULT.max_lift_coefficient[t])
        acceleration = float(DEFAULT.max_acceleration[t])
        corner = _printed(text, "corner speed (m/s)")[t]
        assert math.isclose(corner, math.sqrt(acceleration / lift), rel_tol=1e-3)
        pilot = Pilot(t)
        full = float(DEFAULT.max_energy[t])
        ship = _one_ship(60.0, full, t)
        for _ in range(6000):
            ship = pilot.step(ship)
        cruise = float(ship.proper_speed)
        # Top speed: the peak of a straight boost from cruise with a full store.
        top = cruise
        for _ in range(1800):
            ship = pilot.step(ship, power=-1.0)
            top = max(top, float(ship.proper_speed))
        assert math.isclose(_printed(text, "top speed (m/s)")[t], top, rel_tol=1e-3)
        # Firing endurance coasting: fire at every chance until a shot is refused.
        ship, cost = _one_ship(cruise, full, t), float(DEFAULT.bullet_energy_cost[t])
        interval, wait, tick = int(DEFAULT.ticks_between_shots[t]), 0, 0
        while True:
            if wait == 0:
                if float(ship.energy) < cost:
                    break
                ship = _one_ship(float(ship.proper_speed), float(ship.energy) - cost, t,
                                 float(ship.angle_of_attack), float(ship.nose_rate))
                wait = interval
            ship, wait, tick = pilot.step(ship), max(wait - 1, 0), tick + 1
        endurance = _printed(text, "firing endurance, coasting (s)")[t]
        assert math.isclose(endurance, tick * pilot.tick, rel_tol=1e-3)
        # The coast datum at stick 0.4 (the store held below the knee): settled speed, turn
        # rate and gun offset after 120 s.
        ship, path, last = _one_ship(cruise, 0.25 * full, t), 0.0, []
        for k in range(3600):
            after = replace(pilot.step(ship, 0.4), energy=ship.energy)
            path += (math.remainder(_heading(after) - _heading(ship), 2.0 * math.pi)
                     - (float(after.angle_of_attack) - float(ship.angle_of_attack)))
            ship = after
            if k + 1 in (3570, 3600):
                last.append(path)
        rate = math.degrees((last[1] - last[0]) / (30 * pilot.tick))
        assert math.isclose(_printed(text, "stick 0.4: coast datum turn (deg/s)")[t], rate,
                            rel_tol=2e-3)
        assert math.isclose(_printed(text, "stick 0.4: coast datum speed (m/s)")[t],
                            float(ship.proper_speed), rel_tol=2e-3)
        assert math.isclose(_printed(text, "stick 0.4: gun offset (deg)")[t],
                            math.degrees(float(ship.angle_of_attack)), rel_tol=2e-3)
        # The instantaneous peak: the turn at its printed speed, nose settled, fine tick.
        speed = _printed(text, "at speed (m/s)")[t]
        fine = Pilot(t, 1.0 / 3000.0)
        settled = fine.settle_nose(speed, 0.25 * full, 1.0, t)
        after = fine.step(settled, 1.0)
        turned = (math.remainder(_heading(after) - _heading(settled), 2.0 * math.pi)
                  - (float(after.angle_of_attack) - float(settled.angle_of_attack)))
        peak = _printed(text, "instantaneous peak turn (deg/s)")[t]
        assert math.isclose(peak, math.degrees(turned / fine.tick), rel_tol=2e-3)
