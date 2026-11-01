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
    """A ship turning across the back (170 to -170 degrees) draws through 180, not 0;
    the flight path the same."""
    previous, current = _pair()
    moving = int(np.flatnonzero(current.ships.alive & ~current.ships.spawned
                                & previous.ships.alive)[0])

    def turned(ships, degrees: float):
        heading = ships.heading.copy()
        heading[moving] = math.radians(degrees)
        return dataclasses.replace(ships, heading=heading, flight_path=heading.copy())

    before = dataclasses.replace(previous, ships=turned(previous.ships, 170.0))
    after = dataclasses.replace(current, ships=turned(current.ships, -170.0))
    middle = interpolate(before, after, 0.5).ships
    assert np.isclose(abs(middle.heading[moving]), math.pi, atol=1e-6)
    assert np.isclose(abs(middle.flight_path[moving]), math.pi, atol=1e-6)
    quarter = interpolate(before, after, 0.25).ships
    assert np.isclose(quarter.heading[moving], math.radians(175.0), atol=1e-5)


def test_teleports_snap_to_the_new_state():
    previous, current = _pair()
    ships = current.ships
    spawned = ships.spawned.copy()
    spawned[0] = True
    moved = dataclasses.replace(ships, x=ships.x + 1000.0, spawned=spawned)
    frame = interpolate(previous, dataclasses.replace(current, ships=moved), 0.3)
    assert frame.ships.x[0] == moved.x[0]  # the spawned ship draws where it is now
    assert frame.ships.x[1] != moved.x[1]  # the others move between ticks


def _shooting(ticks: int):
    torch.manual_seed(0)
    loop = MatchLoop(PRESETS["deathmatch"], compile=False, pilot=None)
    shoot = torch.tensor([[0, 0, 1]] * loop.rules.num_ships, dtype=torch.int32)
    for _ in range(ticks):
        loop.tick(shoot)
    return loop.previous, loop.current


def test_a_bullet_fired_into_an_idle_slot_draws_from_its_launch_point():
    previous, current = _shooting(4)
    bullets = current.bullets
    new = bullets.active & ~previous.bullets.active
    assert new.any()
    owner = bullets.owner[new]
    for alpha in (0.0, 0.5):
        frame = interpolate(previous, current, alpha)
        want_x = previous.ships.x[owner] + alpha * (bullets.x[new] - previous.ships.x[owner])
        assert np.allclose(frame.bullets.x[new], want_x, atol=1e-3)


def test_a_bullet_in_flight_moves_linearly():
    previous, current = _shooting(4)
    flying = current.bullets.active & previous.bullets.active
    assert flying.any()
    frame = interpolate(previous, current, 0.5)
    want = 0.5 * (previous.bullets.x[flying] + current.bullets.x[flying])
    assert np.allclose(frame.bullets.x[flying], want, atol=1e-3)


def test_a_refired_slot_draws_from_its_launch_point():
    """A slot whose bullet was still flying fires again: the new bullet starts at its
    shooter, not on the line from the old bullet. Built from two real ticks: the slot of
    a bullet in flight gets a bullet fired on this tick (its position and its full count of
    moves taken from a bullet that was fired into an idle slot)."""
    previous, current = _shooting(4)
    before, after = previous.bullets, current.bullets
    flying = int(np.flatnonzero(before.active & after.active)[0])
    new = int(np.flatnonzero(after.active & ~before.active)[0])
    owner = after.owner[flying]
    launch_x, launch_y = previous.ships.x[owner], previous.ships.y[owner]
    offset_x = after.x[new] - previous.ships.x[after.owner[new]]
    offset_y = after.y[new] - previous.ships.y[after.owner[new]]
    x, y, moves_left = after.x.copy(), after.y.copy(), after.moves_left.copy()
    x[flying], y[flying] = launch_x + offset_x, launch_y + offset_y
    moves_left[flying] = after.moves_left[new]
    refired = dataclasses.replace(current, bullets=dataclasses.replace(
        after, x=x, y=y, moves_left=moves_left))
    for alpha in (0.0, 0.5):
        frame = interpolate(previous, refired, alpha)
        assert np.isclose(frame.bullets.x[flying], launch_x + alpha * offset_x, atol=1e-3)
        assert np.isclose(frame.bullets.y[flying], launch_y + alpha * offset_y, atol=1e-3)
