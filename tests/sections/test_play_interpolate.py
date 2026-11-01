"""Interpolation (`bnb.play.interpolate`): linear, the shortest way round, and no
interpolation across a teleport."""

import dataclasses
import math

import numpy as np
import pytest
import torch

from bnb.game.modes.rules import PRESETS
from bnb.play.interpolate import angle_lerp, interpolate
from bnb.play.loop import MatchLoop


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """One CPU thread: B = 1 ticks gain nothing from more, and lose much on a busy machine."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _pair():
    torch.manual_seed(0)
    loop = MatchLoop(PRESETS["deathmatch"], compile=False, pilot=None)
    for _ in range(20):
        loop.step()
    return loop.previous, loop.current


def test_positions_shield_and_energy_are_linear():
    previous, current = _pair()
    for alpha in (0.0, 0.25, 1.0):
        frame = interpolate(previous, current, alpha)
        for name in ("x", "y", "shield", "energy"):
            want = getattr(previous.ships, name) + alpha * (getattr(current.ships, name)
                                                            - getattr(previous.ships, name))
            assert np.allclose(getattr(frame.ships, name), want, atol=1e-4), name


def test_headings_take_the_shortest_way_round():
    start = np.array([math.radians(170.0), math.radians(-170.0), 0.5])
    end = np.array([math.radians(-170.0), math.radians(170.0), 0.7])
    middle = angle_lerp(start, end, 0.5)
    assert np.allclose(np.abs(middle[:2]), math.pi, atol=1e-6)  # through 180, not 0
    assert np.isclose(middle[2], 0.6)
    assert np.allclose(angle_lerp(start, end, 1.0)[2], 0.7)


def test_teleports_snap_to_the_new_state():
    previous, current = _pair()
    ships = current.ships
    spawned = ships.spawned.copy()
    spawned[0] = True
    moved = dataclasses.replace(ships, x=ships.x + 1000.0, spawned=spawned)
    frame = interpolate(previous, dataclasses.replace(current, ships=moved), 0.3)
    assert frame.ships.x[0] == moved.x[0]  # the spawned ship draws where it is now
    assert frame.ships.x[1] != moved.x[1]  # the others move between ticks
    reset = interpolate(previous, dataclasses.replace(current, ships=moved, reset=True), 0.3)
    assert np.array_equal(reset.ships.x, moved.x)


def test_a_fresh_bullet_draws_from_its_launch_point():
    torch.manual_seed(0)
    loop = MatchLoop(PRESETS["deathmatch"], compile=False, pilot=None)
    shoot = torch.tensor([[0, 0, 1]] * loop.rules.num_ships, dtype=torch.int32)
    for _ in range(4):
        loop.tick(shoot)
    previous, current = loop.previous, loop.current
    bullets = current.bullets
    fresh = bullets.active & (~previous.bullets.active
                              | (bullets.moves_left >= previous.bullets.moves_left))
    assert fresh.any()
    frame = interpolate(previous, current, 0.0)
    owner = bullets.owner[fresh]
    assert np.allclose(frame.bullets.x[fresh], previous.ships.x[owner])
    assert np.allclose(frame.bullets.y[fresh], previous.ships.y[owner])
    old = bullets.active & ~fresh
    assert np.allclose(frame.bullets.x[old], previous.bullets.x[old])
