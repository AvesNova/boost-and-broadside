"""The race acceptance tests (physics-race, branch v3-general) run against bnb.game.

The race tests speak the race interface (numpy in, numpy out, race observation layout);
`race_arm.RaceArm` translates. Goldens come from the race's float64 oracle.
"""

import json
import os
import sys

import numpy as np
import pytest

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
