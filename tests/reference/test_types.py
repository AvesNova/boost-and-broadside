"""v2: the heavy ship type (SPEC 1.1). Fields are moved out of play unless a test needs one."""

import math

import numpy as np
import pytest

from helpers import H, PRESETS, actions, aspect, far_fields, first_move, fresh, place, step  # noqa: F401
from race_presets import _DATA

HEAVY_COST = _DATA["types"]["heavy"]["bullet_cost"]
HEAVY_MUZZLE = _DATA["types"]["heavy"]["muzzle_speed"]


def rule(preset):
    N = PRESETS[preset]["num_ships"]
    i = np.arange(N)
    return ((i - (i >= N // 2) * (N // 2)) % 2).astype(np.int32)


@pytest.mark.parametrize("preset", ["duel", "squad", "battle"])
def test_types_follow_the_slot_rule(arm, preset):
    st = fresh(arm, preset, B=2)
    assert (st["ship_type"] == rule(preset)[None]).all()


def test_types_survive_respawn_and_reset(arm):
    st = far_fields(fresh(arm, "squad"))
    st["alive"][0, 1] = False
    st["respawn"][0, 1] = 1
    st["t"][0] = PRESETS["squad"]["episode_ticks"] - 2
    st, _, _ = step(arm, "squad", st, actions(1, 10))
    assert st["spawned"][0, 1] and (st["ship_type"][0] == rule("squad")).all()
    st, ev, _ = step(arm, "squad", st, actions(1, 10))
    assert ev["done"][0] and (st["ship_type"][0] == rule("squad")).all()


def squad(arm):
    st = far_fields(fresh(arm, "squad"))
    for i in range(10):
        place(st, i, -2000.0 + 60.0 * i, 0.0, 0.0)
    st["E"][:] = 4000.0
    st["shield"][:] = 100.0
    st["fired"][:] = False
    st["spawned"][:] = False
    return st


def test_heavy_gun_rate_cost_and_muzzle(arm):
    st = squad(arm)
    place(st, 1, 0.0, 0.0, 0.0)  # team 0, slot 1: a heavy
    calm, _, _ = step(arm, "squad", st, actions(1, 10))
    pattern = []
    for k in range(6):
        act = actions(1, 10)
        act[0, 1, 2] = 1
        st, _, _ = step(arm, "squad", st, act)
        pattern.append(bool(st["fired"][0, 1]))
        if k == 0:
            assert abs((calm["E"][0, 1] - st["E"][0, 1]) - HEAVY_COST) < 0.02
            assert abs(st["bx"][0, 1, 0] - first_move(100.0 + HEAVY_MUZZLE)) < 0.05
    assert pattern == [True, False, False, True, False, False]


def crossing_pair(arm, shooter, target, ahead=9.0, offset=0.0, target_speed=100.0):
    st = squad(arm)
    place(st, shooter, 0.0, 0.0, 0.0)
    place(st, target, ahead, offset, math.pi / 2, speed=target_speed)
    act = actions(1, 10)
    act[0, shooter, 2] = 1
    return st, act


def test_hit_radius_is_the_targets(arm):
    """A pass 9.5 m wide misses a fighter (8 m) and hits a heavy (11 m)."""
    st, act = crossing_pair(arm, 0, 5, offset=9.5, target_speed=1e-3)  # slot 0 of team 1: fighter
    _, ev, _ = step(arm, "squad", st, act)
    assert ev["damage"][0, 0, 5] == 0.0
    st, act = crossing_pair(arm, 0, 6, offset=9.5, target_speed=1e-3)  # slot 1 of team 1: heavy
    _, ev, _ = step(arm, "squad", st, act)
    assert ev["damage"][0, 0, 6] > 0.0


def test_damage_is_the_shooters(arm):
    st, act = crossing_pair(arm, 1, 5)  # a heavy shoots a fighter crossing its nose
    _, ev, _ = step(arm, "squad", st, act)
    assert abs(ev["damage"][0, 1, 5] - 16.0 * aspect(math.pi / 2)) < 0.15


def test_heavy_shield_and_spawn_energy(arm):
    st = squad(arm)
    st["alive"][0, 6] = False
    st["respawn"][0, 6] = 1
    st["shield"][0, 1] = 159.9
    new, _, _ = step(arm, "squad", st, actions(1, 10))
    assert new["shield"][0, 6] == 160.0 and new["E"][0, 6] == 5600.0
    assert abs(new["shield"][0, 1] - 160.0) < 1e-4, "a heavy recharges to its own maximum"
    assert new["shield"][0, 0] == 100.0, "a fighter stays at 100"


def test_type_column_is_never_masked(arm):
    st = squad(arm)
    place(st, 6, 2400.0, 0.0, 0.0)  # far out of anyone's sight
    obs = arm.observe("squad", st)["ships"]
    assert obs.shape[-1] == 17
    assert obs[0, 0, 6, 12] == 0.0 and obs[0, 0, 6, 16] == 1.0
    assert (obs[0, :, :, 16] == rule("squad")[None]).all()
