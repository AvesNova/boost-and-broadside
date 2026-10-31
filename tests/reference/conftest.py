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
    return RaceArm()


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
