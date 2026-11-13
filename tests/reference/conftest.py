"""The race acceptance tests (physics-race, branch v3-general) run against bnb.game.

The race tests speak the race interface (numpy in, numpy out, race observation layout);
`race_arm.RaceArm` translates. Goldens come from the race's float64 oracle, amended for two
later rules: in the step goldens, a ship alive at tick start that took no damage, gave an
input other than a quiet one, did not spawn and whose environment did not end keeps its
tick-start shield and observed shield (game version 3: no recharge on such a tick) and has
the delay as its recharge wait (game version 5: such an input restarts the wait).

Two later rules the arm plays as the race did, since the oracle has no part for them and
each has its own tests: the bullet damage (game version 6: the arm's card has explosive
share 1 and its core at the hit radius, so a hit does the race's flat damage times the
aspect; `race_arm.race_damage`) and the sight through the band (game version 14: the arm
sees with the band as outside; `race_arm.race_sight`).
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
GOLDEN = os.path.join(HERE, "golden")

from race_arm import RaceArm  # noqa: E402


@pytest.fixture(scope="session")
def arm():
    """bnb's game, or another implementation of the race interface: REFERENCE_ARM=<dir with
    arm.py defining make_arm()> (used to validate these tests against the float64 oracle and
    its deliberate mutants)."""
    path = os.environ.get("REFERENCE_ARM")
    if not path:
        return RaceArm()
    import importlib.util

    spec = importlib.util.spec_from_file_location("reference_arm", os.path.join(path, "arm.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_arm()


@pytest.fixture(autouse=True)
def _another_arm_may_draw(request):
    """Another implementation (REFERENCE_ARM) may build its modules from torch's default
    generator, which bnb's arm never does: under one, each test restores the generator."""
    if not os.environ.get("REFERENCE_ARM"):
        yield
        return
    with torch.random.fork_rng(devices=[]):
        yield


def golden(name: str) -> dict:
    """Load golden/<name>; "group/key" entries become nested dicts."""
    path = os.path.join(GOLDEN, name)
    if name.endswith(".json"):
        with open(path) as fh:
            return json.load(fh)
    out: dict = {}
    with np.load(path) as data:
        for key in data.files:
            if "/" in key:
                group, sub = key.split("/", 1)
                out.setdefault(group, {})[sub] = data[key]
            else:
                out[key] = data[key]
    return out
