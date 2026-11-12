"""The scripted agent's own time: off by default and then the agent exactly; when on, the
switches' rates and the memory's fade run in the ship's own time."""

import dataclasses
import pathlib
import sys

import pytest
import torch
from scripted_cases import CHARACTERISTICS, absent, observation, ship

from bnb.contracts.obs import GLOBAL, SHIP
from bnb.controllers.scripted_defaults import MODE_DEFAULTS
from bnb.controllers.scripted_memory import fresh_carry, remember, time_unit
from bnb.controllers.scripted_parameters import (
    ADDITIVE_BOUNDS,
    OWN_TIME,
    ScriptedParameters,
    Switch,
    in_own_time,
    logistic,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # `bench` is a top-level package of the repository
    sys.path.insert(0, str(ROOT))

from bench import scripted_search as search  # noqa: E402

pytestmark = pytest.mark.usefixtures("global_generator")
PARAMETERS = ScriptedParameters()
TURN = CHARACTERISTICS.turn_rate[:, 2]
AGILE, STIFF = int(TURN.argmax()), int(TURN.argmin())


def test_the_defaults_keep_seconds_in_every_mode():
    for mode, parameters in MODE_DEFAULTS.sets.items():
        assert parameters.time_unit_exponent == 0.0, mode


def test_own_time_scales_the_switches_and_the_fade():
    own_types = torch.tensor([[AGILE, STIFF]], dtype=torch.int32)
    assert torch.equal(time_unit(CHARACTERISTICS, own_types, PARAMETERS), torch.ones(1, 2))
    whole = dataclasses.replace(PARAMETERS, time_unit_exponent=1.0)
    units = time_unit(CHARACTERISTICS, own_types, whole)
    turn_time = CHARACTERISTICS.corner_speed / CHARACTERISTICS.max_acceleration
    assert torch.allclose(units, (turn_time / turn_time[0])[own_types.long()])
    assert float(time_unit(CHARACTERISTICS, torch.zeros(1, 1, dtype=torch.int32), whole)) == 1.0
    shape = Switch(8.0, 0.3, 0.06)
    assert in_own_time(shape, torch.ones(1)).rate.item() == 8.0
    assert in_own_time(shape, torch.full((1,), 2.0)).rate.item() == 4.0
    # a hidden enemy fades in the team's mean unit
    stiff_unit = float(units[0, 1])
    obs = observation([ship(ship_type=STIFF), absent(STIFF, 0)], [absent(), absent()])
    obs.ships[0, 2, SHIP["alive"]] = 1.0
    obs.ships[0, 2, SHIP["time_since_seen"]] = 1.0
    obs.globals[:, GLOBAL["reset"]] = 0.0
    carry = fresh_carry(1, 2, 2, torch.device("cpu"))
    carry = dataclasses.replace(
        carry,
        known=torch.tensor([[True, False]]),
        enemy_x=torch.tensor([[5e4, 0.0]]),  # far outside the own vision
    )
    reset = torch.zeros(1, dtype=torch.bool)
    _, recall = remember(carry, obs, reset, CHARACTERISTICS, whole, 1.0)
    fade = PARAMETERS.memory_fade_in_seconds
    expected = logistic(torch.tensor(1.0 / stiff_unit), fade)  # one own ship: its unit
    # the absent second own ship has the same type: the mean is the stiff unit
    assert float(recall.weight[0, 0]) == pytest.approx(float(expected), rel=1e-5)


def test_the_search_keeps_the_own_time_unless_asked_and_bounds_it():
    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        child = search.perturb(PARAMETERS, 1.0, generator)
        for name in OWN_TIME:
            assert getattr(child, name) == getattr(PARAMETERS, name), name
    moved = set()
    for _ in range(200):
        child = search.perturb(PARAMETERS, 1.0, generator, frozen=())
        for name, (low, high) in ADDITIVE_BOUNDS.items():
            assert low <= getattr(child, name) <= high, name
            moved |= {name} if getattr(child, name) != getattr(PARAMETERS, name) else set()
    assert moved == set(ADDITIVE_BOUNDS)
