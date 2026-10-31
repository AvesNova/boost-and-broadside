"""Rewards and Frontline equivalence tests: v2 against goldens from v1's code.

The goldens (golden/*.npz) were computed in float64 on the CPU from v1's own code at tag
v1-torch-final, with random weights and inputs. The tests speak the interface in
README.md; `rewards_arm.make_arm()` adapts it to `bnb`. REFERENCE_ARM=<dir with arm.py
defining make_arm()> runs the same tests against another implementation (used to validate
the tests against v1 itself and against deliberate mutants).
"""

import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rewards_arm import make_arm  # noqa: E402


@pytest.fixture(scope="session")
def arm():
    path = os.environ.get("REFERENCE_ARM")
    if path:
        spec = importlib.util.spec_from_file_location("rewards_reference_arm", os.path.join(path, "arm.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.make_arm()
    implementation = make_arm()
    if implementation is None:
        pytest.skip("the rewards and modes sections have not implemented rewards_arm.make_arm yet")
    return implementation
