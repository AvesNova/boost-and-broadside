"""The render snapshot (`bnb.play.snapshot`): pure, complete, and its visibility is each
team's observation."""

import dataclasses
import math

import numpy as np
import pytest
import torch

from bnb.contracts import GameConfig
from bnb.contracts.obs import SHIP
from bnb.game.modes.rules import PRESETS
from bnb.play.snapshot import Snapshot, snapshot
from bnb.runtime.realtime import Realtime

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """One CPU thread: B = 1 ticks gain nothing from more, and lose much on a busy machine."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _match(config: GameConfig, fogged: bool = True, ticks: int = 40, seed: int = 0):
    torch.manual_seed(seed)
    match = Realtime(config, compile=False, fogged=fogged)
    for _ in range(ticks):
        match.step(torch.tensor([[1, 2, 1]] * match.rules.num_ships, dtype=torch.int32))
    return match


def _arrays(value):
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _arrays(getattr(value, field.name))
    elif isinstance(value, np.ndarray):
        yield value


def test_snapshot_is_pure_and_complete():
    match = _match(PRESETS["frontline"])
    rules = match.rules
    first = snapshot(rules, match.types, match.state, match.obs)
    again = snapshot(rules, match.types, match.state, match.obs)
    copies = [a.copy() for a in _arrays(first)]
    for a, b in zip(_arrays(first), _arrays(again)):
        assert np.array_equal(a, b)
    match.step(torch.zeros((rules.num_ships, 3), dtype=torch.int32))
    for a, b in zip(_arrays(first), copies):  # stepping on leaves the snapshot alone
        assert np.array_equal(a, b)
    N, K, M = rules.num_ships, rules.bullet_slots, rules.num_fields
    sizes = {"ships": N, "bullets": N * K, "fields": M, "zones": rules.num_zones}
    for part, size in sizes.items():
        group = getattr(first, part)
        for field in dataclasses.fields(group):
            values = getattr(group, field.name)
            assert values.shape[-1] == size, (part, field.name)
            assert np.isfinite(values.astype(float)).all(), (part, field.name)
    assert first.ships.visible.shape == (2, N)
    assert first.mode == "frontline" and first.zones.present.all()
    assert (0.0 <= first.ships.shield).all() and (first.ships.shield <= 1.0).all()
    assert (0.0 <= first.ships.energy).all() and (first.ships.energy <= 1.0 + 1e-6).all()
    assert {f.name for f in dataclasses.fields(Snapshot)} >= {"arena_radius", "clock", "fogged"}


def test_visibility_is_each_teams_observation_in_sim_order():
    match = _match(GameConfig(4, 12, 2600.0), ticks=60, seed=2)
    frame = snapshot(match.rules, match.types, match.state, match.obs)
    S = match.rules.ships_per_team
    for team in range(2):
        rows = match.obs.ships[0, team]
        order = [team * S + s for s in range(S)] + [(1 - team) * S + s for s in range(S)]
        seen = (rows[:, SHIP["visible"]] > 0.5).numpy()
        assert np.array_equal(frame.ships.visible[team][order], seen)
        assert frame.ships.visible[team][team * S : (team + 1) * S].all()  # own ships
    assert not frame.ships.visible.all()  # the scenario hides someone


def test_flight_path_and_angle_of_attack_agree_with_the_velocity():
    match = _match(PRESETS["deathmatch"], ticks=30)
    frame = snapshot(match.rules, match.types, match.state, match.obs)
    truth = match.obs.ships[0, 0]  # team 0's rows: its own ships, always shown
    velocity = torch.atan2(truth[:5, SHIP["velocity_y"]], truth[:5, SHIP["velocity_x"]]).numpy()
    turn = np.remainder(frame.ships.flight_path[:5] - velocity + math.pi, 2 * math.pi) - math.pi
    assert np.abs(turn).max() < 1e-4
