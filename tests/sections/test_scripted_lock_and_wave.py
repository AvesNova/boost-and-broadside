"""The scripted Frontline layer's lock and wave terms (`scripted_frontline.lock_and_wave`,
the doctrine's `lock_press`, `wave_hold` and `wave_hold_seconds`): off by default, the
capture lock presses the attack or holds the defense until a travel time before it ends,
and a waiting respawn holds the defense until its wave."""

import types

import torch

from bnb.contracts.obs import ZONE, ZONE_COLUMNS
from bnb.controllers import scripted_doctrine
from bnb.controllers.scripted_doctrine import Doctrine, table
from bnb.controllers.scripted_frontline import LOCK_STRENGTHS, lock_and_wave


def _fleet(x: float, next_wave: float = 0.0):
    """One view, two own ships at (x, 0) flying 100 m/s (corner speed 100 m/s), type 0."""
    return types.SimpleNamespace(
        own_type=torch.zeros((1, 2), dtype=torch.long),
        x=torch.full((1, 2), x),
        y=torch.zeros((1, 2)),
        speed=torch.full((1, 2), 100.0),
        corner_speed=torch.full((1, 2), 100.0),
        next_wave=torch.tensor([next_wave]),
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


ALIVE = torch.ones((1, 2), dtype=torch.bool)


def test_the_default_doctrine_leaves_the_goal_alone():
    shift = lock_and_wave(_fleet(1000.0, 5.0), _zones(30.0, 30.0), ALIVE, _doctrines())
    assert torch.equal(shift, torch.zeros((1, 2)))
    assert "lock_press" in scripted_doctrine.WEIGHTS and "wave_hold" in scripted_doctrine.WEIGHTS


def test_the_enemys_lock_presses_the_attack_until_a_travel_time_before_it_ends():
    doctrines = _doctrines(lock_press=1.0)
    # the ships are 1000 m (10 s at 100 m/s) from the zone the enemy lost
    pressing = lock_and_wave(_fleet(1000.0), _zones(lock_enemy=15.0), ALIVE, doctrines)
    torch.testing.assert_close(pressing, torch.full((1, 2), -LOCK_STRENGTHS))
    ending = lock_and_wave(_fleet(1000.0), _zones(lock_enemy=8.0), ALIVE, doctrines)
    assert torch.equal(ending, torch.zeros((1, 2)))


def test_the_own_lock_holds_the_defense_until_a_travel_time_before_it_ends():
    doctrines = _doctrines(lock_press=0.5)
    holding = lock_and_wave(_fleet(1000.0), _zones(lock_own=15.0), ALIVE, doctrines)
    torch.testing.assert_close(holding, torch.full((1, 2), 0.5 * LOCK_STRENGTHS))
    ending = lock_and_wave(_fleet(1000.0), _zones(lock_own=8.0), ALIVE, doctrines)
    assert torch.equal(ending, torch.zeros((1, 2)))


def test_a_waiting_respawn_holds_the_defense_until_its_wave_lands():
    doctrines = _doctrines(wave_hold=1.0, wave_hold_seconds=12.0)
    one_dead = torch.tensor([[True, False]])
    soon = lock_and_wave(_fleet(1000.0, next_wave=6.0), _zones(), one_dead, doctrines)
    torch.testing.assert_close(soon, torch.full((1, 2), LOCK_STRENGTHS))
    far = lock_and_wave(_fleet(1000.0, next_wave=18.0), _zones(), one_dead, doctrines)
    assert torch.equal(far, torch.zeros((1, 2)))  # the wave is beyond the hold
    landed = lock_and_wave(_fleet(1000.0, next_wave=6.0), _zones(), ALIVE, doctrines)
    assert torch.equal(landed, torch.zeros((1, 2)))  # nobody waits: launch
