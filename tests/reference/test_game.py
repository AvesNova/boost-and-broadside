"""Whole-tick behaviour against the reference: reset, step, observation, chaser (SPEC 2, 8-10)."""

import numpy as np
import pytest

from conftest import golden
from helpers import compare, compare_dict

PRESET_NAMES = ["duel", "squad", "battle"]


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_reset_matches_reference(arm, preset):
    g = golden(f"reset_{preset}.npz")
    st = arm.reset(preset, g["uniforms"])
    compare_dict(st, g["expected"], rtol=1e-5, context=f"reset {preset}: ")


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_step_matches_reference(arm, preset):
    """One tick from mid-game snapshots, including an episode end, due respawns and exposed ships.

    A small share of entries may differ where float32 lands on the other side of a
    threshold than float64 (a hit at exactly the hit radius, a sight line grazing a core).
    """
    g = golden(f"step_{preset}.npz")
    st, ev, obs = arm.step(preset, g["state"], g["actions"], g["spread"], g["uniforms"])
    compare_dict(st, g["expected_state"], rtol=2e-5, max_bad=2e-3, context=f"step {preset} state: ")
    compare_dict(ev, g["events"], rtol=2e-5, max_bad=2e-3, context=f"step {preset} events: ")
    compare(obs["ships"], g["obs"]["ships"], "obs.ships", atol=5e-2, rtol=2e-5, max_bad=2e-3)
    compare(obs["fields"], g["obs"]["fields"], "obs.fields", atol=1e-2, rtol=1e-6)
    compare(obs["globals"], g["obs"]["globals"], "obs.globals", atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_observe_matches_reference(arm, preset):
    g = golden(f"step_{preset}.npz")
    obs = arm.observe(preset, g["state"])
    compare(obs["ships"], g["obs_before"]["ships"], "obs.ships", atol=5e-2, rtol=2e-5, max_bad=2e-3)


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_chaser_matches_reference(arm, preset):
    g = golden(f"step_{preset}.npz")
    act = arm.chaser(preset, g["state"])
    compare(act, g["chaser"], "chaser", max_bad=2e-3)


def test_open_loop_rollout(arm):
    """30 ticks of recorded actions and noise from a reset; float32 drift stays small."""
    g = golden("rollout_squad.npz")
    st = arm.reset("squad", g["uniforms0"])
    for t in range(g["actions"].shape[0]):
        st, _, _ = arm.step("squad", st, g["actions"][t], g["spread"][t], g["uniforms"][t])
    exp = g["expected"]
    for key in ("x", "y"):
        compare(st[key], exp[key], key, atol=0.05, rtol=1e-4, max_bad=0.02, context="rollout: ")
    for key in ("w", "E", "shield"):
        compare(st[key], exp[key], key, atol=0.05 if key != "E" else 1.0, rtol=1e-3, max_bad=0.02,
                context="rollout: ")
    for key in ("alive", "ttl", "gun_wait", "t"):
        compare(st[key], exp[key], key, max_bad=0.02, context="rollout: ")
