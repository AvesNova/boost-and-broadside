"""v3: games of any size (SPEC 1.2), at sizes that are not presets."""

import json
import os

import numpy as np
import pytest

from conftest import GOLDEN, golden
from helpers import compare, compare_dict

with open(os.path.join(GOLDEN, "sizes.json")) as fh:
    SIZES = json.load(fh)
NAMES = sorted(SIZES)


@pytest.mark.parametrize("name", NAMES)
def test_reset_at_custom_size(arm, name):
    g = golden(f"sizes_reset_{name}.npz")
    st = arm.reset(SIZES[name], g["uniforms"])
    compare_dict(st, g["expected"], rtol=1e-5, context=f"reset {name}: ")


@pytest.mark.parametrize("name", NAMES)
def test_step_at_custom_size(arm, name):
    g = golden(f"sizes_step_{name}.npz")
    size = SIZES[name]
    st, ev, obs = arm.step(size, g["state"], g["actions"], g["spread"], g["uniforms"])
    compare_dict(st, g["expected_state"], rtol=2e-5, max_bad=2e-3, context=f"step {name} state: ")
    compare_dict(ev, g["events"], rtol=2e-5, max_bad=2e-3, context=f"step {name} events: ")
    compare(obs["ships"], g["obs"]["ships"], "obs.ships", atol=5e-2, rtol=2e-5, max_bad=2e-3)
    compare(arm.chaser(size, g["state"]), g["chaser"], "chaser", max_bad=2e-3)
