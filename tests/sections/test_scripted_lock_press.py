"""The scripted Frontline layer's capture-lock term (`scripted_frontline.lock_press`, the
doctrine's `lock_press`): off by default; the capture lock presses the attack or holds the
defense until a travel time before it ends."""

import types

import torch

from bnb.contracts.obs import ZONE, ZONE_COLUMNS
from bnb.controllers import scripted_doctrine
from bnb.controllers.scripted_doctrine import Doctrine, table
from bnb.controllers.scripted_frontline import LOCK_STRENGTHS, lock_press


def _fleet(x: float):
    """One view, two own ships at (x, 0) flying 100 m/s (corner speed 100 m/s), type 0."""
    return types.SimpleNamespace(
        own_type=torch.zeros((1, 2), dtype=torch.long),
        x=torch.full((1, 2), x),
        y=torch.zeros((1, 2)),
        speed=torch.full((1, 2), 100.0),
        corner_speed=torch.full((1, 2), 100.0),
    )


def _zones(lock_own: float = 0.0, lock_enemy: float = 0.0):
    """One zone at the origin, present, with these capture locks (s)."""
    zones = torch.zeros((1, 1, len(ZONE_COLUMNS)))
    zones[..., ZONE["present"]] = 1.0
    zones[..., ZONE["lock_own"]] = lock_own
    zones[..., ZONE["lock_enemy"]] = lock_enemy
    return zones


def _doctrines(**fields) -> torch.Tensor:
    return table([Doctrine(**fields)])


def test_the_default_doctrine_leaves_the_goal_alone():
    shift = lock_press(_fleet(1000.0), _zones(30.0, 30.0), _doctrines())
    assert torch.equal(shift, torch.zeros((1, 2)))
    assert "lock_press" in scripted_doctrine.WEIGHTS


def test_the_enemys_lock_presses_the_attack_until_a_travel_time_before_it_ends():
    doctrines = _doctrines(lock_press=1.0)
    # the ships are 1000 m (10 s at 100 m/s) from the zone the enemy lost
    pressing = lock_press(_fleet(1000.0), _zones(lock_enemy=15.0), doctrines)
    torch.testing.assert_close(pressing, torch.full((1, 2), -LOCK_STRENGTHS))
    ending = lock_press(_fleet(1000.0), _zones(lock_enemy=8.0), doctrines)
    assert torch.equal(ending, torch.zeros((1, 2)))


def test_the_own_lock_holds_the_defense_until_a_travel_time_before_it_ends():
    doctrines = _doctrines(lock_press=0.5)
    holding = lock_press(_fleet(1000.0), _zones(lock_own=15.0), doctrines)
    torch.testing.assert_close(holding, torch.full((1, 2), 0.5 * LOCK_STRENGTHS))
    ending = lock_press(_fleet(1000.0), _zones(lock_own=8.0), doctrines)
    assert torch.equal(ending, torch.zeros((1, 2)))
