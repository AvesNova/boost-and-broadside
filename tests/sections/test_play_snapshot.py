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


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """One CPU thread: B = 1 ticks gain nothing from more, and lose much on a busy machine."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _match(config: GameConfig, fogged: bool = True, ticks: int = 40, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    match = Realtime(config, compiled=False, fogged=fogged, generator=generator)
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


def test_frontline_snapshot_shows_the_locks_the_waves_and_the_time_to_respawn():
    """A dead Frontline ship's time to respawn counts its team's wave (the minimum wait,
    then the first wave at or after it); the zones carry each team's capture lock."""
    match = _match(PRESETS["real_scale_frontline"], ticks=2)
    rules = match.rules
    tick = rules.tick_length
    interval = round(rules.modes[0].wave_seconds / tick)
    ships, mode = match.state.ships, match.state.mode
    alive, wait = ships.alive.clone(), ships.respawn_wait.clone()
    alive[0, :2], wait[0, :2] = False, torch.tensor([30, 200], dtype=wait.dtype)
    lock = torch.zeros_like(mode.lock)
    lock[0, 1, 3] = 90
    state = dataclasses.replace(
        match.state,
        ships=dataclasses.replace(ships, alive=alive, respawn_wait=wait),
        mode=dataclasses.replace(
            mode, lock=lock, wave_wait=torch.tensor([[100, 7]], dtype=mode.wave_wait.dtype)
        ),
    )
    frame = snapshot(rules, match.types, state, match.obs)
    # team 0's wave in 100 ticks: a ship ready in 30 waits for it; one ready in 200 for the
    # wave after (100 + interval >= 200)
    want = np.array([100, 100 + interval]) * tick
    np.testing.assert_allclose(frame.ships.respawn_wait[:2], want, rtol=1e-6)
    np.testing.assert_allclose(frame.next_wave, np.array([100, 7]) * tick, rtol=1e-6)
    assert frame.zones.lock.shape == (2, rules.num_zones)
    assert frame.zones.lock[1, 3] == pytest.approx(
        90 * tick
    ) and frame.zones.lock.sum() == pytest.approx(90 * tick)
    assert frame.zones.lock_length[0] == pytest.approx(45.0, abs=0.5)  # 19 s / sqrt(km), 5v5
