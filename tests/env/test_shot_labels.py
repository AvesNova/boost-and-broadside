"""Ghost bullets and counterfactual shot labels (env/shot_labels.py)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.constants import ShootActions
from boost_and_broadside.env import shot_labels
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.physics import (
    _handle_shooting,
    advance_bullets,
    resolve_collisions,
)
from boost_and_broadside.env.shot_labels import (
    ALLY,
    ENEMY,
    EX_CLOSEST_DSQ,
    EX_CLOSEST_SIDE,
    EX_EVENT_TAU,
    EX_HIT,
    EX_SNAP_EVENT,
    EX_TAU1,
    EX_TAU_FALLBACK,
    NO_ENCOUNTER_DSQ,
    NO_HIT,
    SNAP_DX,
    SNAP_DY,
    SNAP_INDEX,
    SNAP_VX,
)
from tests.conftest import make_state

# The Frontline rate: physics and decisions at 30 Hz.
CONFIG = replace(ShipConfig(), dt=1.0 / 30.0, bullet_spread=0.0)
DELAY = shot_labels.shot_label_delay(CONFIG)
COOLDOWN_TICKS = 3


# ---------------------------------------------------------------------------
# Hand-built scenarios: static ships, one shooter facing +x
# ---------------------------------------------------------------------------


def _labelled_state(positions: list[complex], teams: list[int]):
    state = make_state(num_envs=1, max_ships=len(positions), max_bullets=10, ship_config=CONFIG)
    n, k = len(positions), 10
    state.bullet_shot_record = torch.zeros(1, n, k, shot_labels.RECORD_DIM)
    state.shot_example = torch.zeros(1, n, shot_labels.EXAMPLE_DIM)
    state.ship_pos = torch.tensor([positions], dtype=torch.complex64)
    state.ship_team_id = torch.tensor([teams], dtype=torch.int32)
    # Only ship 0 ever launches: the others have no ghost cooldown to spare.
    state.ship_ghost_cooldown = torch.full((1, n), 1.0e9)
    state.ship_ghost_cooldown[0, 0] = 0.0
    return state


def _tick(state, generator, shoot: bool = False):
    shoot_action = torch.zeros(state.ship_pos.shape, dtype=torch.long)
    if shoot:
        shoot_action[0, 0] = int(ShootActions.SHOOT)
    _handle_shooting(state, shoot_action, CONFIG, generator)
    state.ship_ghost_cooldown[0, 1:] = 1.0e9
    state, trajectory = advance_bullets(state, CONFIG)
    shot_labels.advance(state, CONFIG)
    state, _ = resolve_collisions(state, CONFIG, trajectory=trajectory)
    return state


def _fly(state, *, shoot_first: bool = False, ticks: int = DELAY + 1):
    """Launch from ship 0 on tick 0, then run; returns ship 0's emitted examples."""
    generator = torch.Generator().manual_seed(0)
    emitted = []
    for tick in range(ticks):
        state = _tick(state, generator, shoot=shoot_first and tick == 0)
        # Hold every later ghost back so ship 0 has exactly one flight.
        state.ship_ghost_cooldown[0, 0] = 1.0e9
        if state.shot_example_valid[0, 0]:
            emitted.append((tick, state.shot_example[0, 0].clone()))
    return state, emitted


def test_delay_is_one_lifetime_of_moves_at_30_hz():
    assert DELAY == 29


def test_ghost_hits_enemy_without_damaging_it():
    state = _labelled_state([0j, 100 + 0j, 500 + 500j], [0, 1, 0])
    state, emitted = _fly(state)

    assert state.ship_health[0, 1] == CONFIG.max_health  # a ghost never damages
    assert not state.ship_is_shooting.any()
    assert state.ship_power[0, 0] == CONFIG.max_power  # nor costs power
    ((tick, example),) = emitted
    assert tick == DELAY
    assert example[EX_HIT] == ENEMY
    # 500 px/s at 30 Hz reaches 100 px within the sixth move.
    assert example[EX_EVENT_TAU] == 6
    assert example[EX_CLOSEST_DSQ] < CONFIG.collision_radius**2
    event = example[EX_SNAP_EVENT]
    assert 90.0 < event[SNAP_DX] < 120.0
    assert abs(event[SNAP_DY]) < 1e-3
    assert event[SNAP_VX] > 0.0
    assert event[SNAP_INDEX] == 1.0


def test_real_hit_damages_once_and_keeps_flying():
    # Two enemies on the line: a real bullet damages the first and passes the
    # second untouched, but its flight runs the whole lifetime.
    state = _labelled_state([0j, 100 + 0j, 200 + 0j], [0, 1, 1])
    state, emitted = _fly(state, shoot_first=True)

    # Damage scales with impact angle, so only that it landed is exact.
    assert state.ship_health[0, 1] < CONFIG.max_health
    assert state.ship_health[0, 2] == CONFIG.max_health
    ((tick, example),) = emitted
    assert tick == DELAY
    assert example[EX_HIT] == ENEMY
    assert example[EX_EVENT_TAU] == 6


def test_ally_in_the_line_of_fire_is_an_ally_hit():
    state = _labelled_state([0j, 300 + 0j, 100 + 0j], [0, 1, 0])
    _, emitted = _fly(state)
    ((_, example),) = emitted
    assert example[EX_HIT] == ALLY


def test_near_miss_records_closest_pass():
    state = _labelled_state([0j, 150 + 30j, 600 + 600j], [0, 1, 0])
    _, emitted = _fly(state)
    ((_, example),) = emitted
    assert example[EX_HIT] == NO_HIT
    assert example[EX_CLOSEST_SIDE] == ENEMY
    assert math.sqrt(example[EX_CLOSEST_DSQ]) == pytest.approx(30.0, abs=0.5)
    assert example[EX_SNAP_EVENT][SNAP_DX] == pytest.approx(150.0, abs=20.0)


def test_nothing_alive_to_pass_is_no_encounter():
    state = _labelled_state([0j, 300 + 0j], [0, 1])
    state.ship_alive[0, 1] = False
    _, emitted = _fly(state)
    ((_, example),) = emitted
    assert example[EX_HIT] == NO_HIT
    assert example[EX_CLOSEST_DSQ] == NO_ENCOUNTER_DSQ


def test_scheduled_snapshots_are_taken_at_their_ages():
    state = _labelled_state([0j, 600 + 600j], [0, 1])
    _, emitted = _fly(state)
    ((_, example),) = emitted
    for tau_index, snap in (
        (EX_TAU1, shot_labels.EX_SNAP1),
        (EX_TAU_FALLBACK, shot_labels.EX_SNAP_FALLBACK),
    ):
        tau = int(example[tau_index])
        assert 0 <= tau <= DELAY
        # Straight flight with drag: displacement grows with age, lateral zero.
        displacement = example[snap][SNAP_DX]
        if tau == 0:
            assert displacement == 0.0
        else:
            assert 0.0 < displacement <= CONFIG.bullet_speed * tau * CONFIG.dt + 1e-3
        assert abs(example[snap][SNAP_DY]) < 1e-3


def test_real_shot_inside_ghost_cooldown_replaces_the_ghost():
    state = _labelled_state([0j, 600 + 600j], [0, 1])
    generator = torch.Generator().manual_seed(0)
    state = _tick(state, generator)  # tick 0: ghost
    assert state.bullet_flying[0, 0].sum() == 1
    assert not state.bullet_active[0, 0].any()
    ghost_slot = int(state.bullet_flying[0, 0].int().argmax())
    state = _tick(state, generator, shoot=True)  # tick 1: real, same slot
    assert state.bullet_flying[0, 0].sum() == 1
    assert state.bullet_active[0, 0, ghost_slot]
    emitted = []
    for tick in range(2, DELAY + 5):
        state = _tick(state, generator)
        state.ship_ghost_cooldown[0, 0] = 1.0e9
        if state.shot_example_valid[0, 0]:
            emitted.append(tick)
    assert emitted == [1 + DELAY]  # only the real shot's flight completes


# ---------------------------------------------------------------------------
# Whole environment
# ---------------------------------------------------------------------------


def _env(shot_labels_on: bool, num_fields: int = 3) -> TensorEnv:
    env = TensorEnv(
        2,
        replace(ShipConfig(), dt=1.0 / 30.0),
        EnvConfig(
            num_ships=6,
            max_bullets=10,
            max_episode_steps=10_000,
            num_fields=num_fields,
            shot_labels=shot_labels_on,
        ),
        "cpu",
    )
    env.reset(seed=11)
    return env


def _random_actions(generator: torch.Generator, shape) -> torch.Tensor:
    power = torch.randint(0, 3, shape, generator=generator)
    turn = torch.randint(0, 5, shape, generator=generator)
    shoot = (torch.rand(shape, generator=generator) < 0.4).long()
    return torch.stack([power, turn, shoot], dim=-1)


GAMEPLAY_FIELDS = (
    "ship_pos",
    "ship_vel",
    "ship_attitude",
    "ship_health",
    "ship_power",
    "ship_cooldown",
    "ship_alive",
    "ship_is_shooting",
    "damage_matrix",
    "cumulative_damage_matrix",
)


def _real_bullets(state) -> list[list[list[complex]]]:
    """Each ship's active bullet positions, independent of slot order."""
    out = []
    for env in range(state.num_envs):
        ships = []
        for ship in range(state.max_ships):
            active = state.bullet_active[env, ship]
            positions = state.bullet_pos[env, ship][active].tolist()
            ships.append(sorted(positions, key=lambda z: (z.real, z.imag)))
        out.append(ships)
    return out


def _run(env: TensorEnv, ticks: int) -> list[dict]:
    generator = torch.Generator().manual_seed(5)
    trace = []
    shape = env.state.ship_pos.shape
    for _ in range(ticks):
        dones, truncated = env.tick(_random_actions(generator, shape))
        env.reset_envs(dones | truncated)
        snapshot = {name: getattr(env.state, name).clone() for name in GAMEPLAY_FIELDS}
        snapshot["bullets"] = _real_bullets(env.state)
        trace.append(snapshot)
    return trace


def test_ghosts_leave_gameplay_bit_identical():
    off = _run(_env(False), 150)
    labelled = _env(True)
    on = _run(labelled, 150)
    for tick, (a, b) in enumerate(zip(off, on, strict=True)):
        for name in GAMEPLAY_FIELDS:
            assert torch.equal(a[name], b[name]), f"{name} differs at tick {tick}"
        assert a["bullets"] == b["bullets"], f"real bullets differ at tick {tick}"
    assert any(t["damage_matrix"].sum() > 0 for t in on), "scenario never landed a hit"
    ghosts = labelled.state.bullet_flying & ~labelled.state.bullet_active
    assert ghosts.any(), "no ghost in flight"


def test_a_slot_is_only_reused_after_expiry_or_by_its_replacement():
    env = _env(True)
    generator = torch.Generator().manual_seed(9)
    shape = env.state.ship_pos.shape
    examples = 0
    for _ in range(4 * DELAY):
        flying = env.state.bullet_flying.clone()
        active = env.state.bullet_active.clone()
        age = env.state.bullet_age.clone()
        env.tick(_random_actions(generator, shape))
        # A slot launched this tick ends it at age one.
        relaunched = flying & (env.state.bullet_age == 1)
        replaced = relaunched & ~active & (age < COOLDOWN_TICKS)
        assert torch.equal(relaunched, replaced), "a live flight was overwritten"
        examples += int(env.state.shot_example_valid.sum())
    assert examples > 0


def test_emission_is_exactly_one_delay_after_launch():
    env = _env(True, num_fields=0)
    generator = torch.Generator().manual_seed(3)
    shape = env.state.ship_pos.shape
    launch_ticks: dict[tuple[int, int, int], int] = {}
    for tick in range(3 * DELAY):
        env.tick(_random_actions(generator, shape))
        launched = env.state.bullet_flying & (env.state.bullet_age == 1)
        for key in map(tuple, launched.nonzero().tolist()):
            launch_ticks[key] = tick
        for env_index, ship in env.state.shot_example_valid.nonzero().tolist():
            starts = [
                start
                for (e, s, _), start in launch_ticks.items()
                if (e, s) == (env_index, ship) and start == tick - DELAY
            ]
            assert len(starts) == 1


def test_reset_drops_flights_in_progress():
    env = _env(True, num_fields=0)
    shape = env.state.ship_pos.shape
    idle = torch.zeros((*shape, 3), dtype=torch.long)
    for _ in range(5):
        env.tick(idle)
    assert env.state.bullet_flying.any()
    env.reset_envs(torch.ones(2, dtype=torch.bool))
    assert not env.state.bullet_flying.any()
    # Hold every ship's ghost back; nothing launched before the reset emits.
    for _ in range(DELAY + 2):
        env.state.ship_ghost_cooldown.fill_(1.0e9)
        env.tick(idle)
        assert not env.state.shot_example_valid.any()


def test_labels_need_one_tick_decisions():
    with pytest.raises(ValueError, match="action_repeat"):
        EnvConfig(
            num_ships=2, max_bullets=10, max_episode_steps=10, action_repeat=2, shot_labels=True
        )


def test_is_shooting_observes_real_shots_and_never_ghosts():
    from boost_and_broadside.env.observation import ObsKey, observation_from_state

    env = _env(True, num_fields=0)
    shape = env.state.ship_pos.shape
    idle = torch.zeros((*shape, 3), dtype=torch.long)
    env.tick(idle)  # every ship launches a ghost
    assert (env.state.bullet_flying & ~env.state.bullet_active).any()
    obs = observation_from_state(env.state, env.ship_config)
    assert not obs[ObsKey.IS_SHOOTING].any()

    for _ in range(COOLDOWN_TICKS):
        env.tick(idle)
    shoot = idle.clone()
    shoot[0, 0, 2] = int(ShootActions.SHOOT)
    env.tick(shoot)
    obs = observation_from_state(env.state, env.ship_config)
    shooting = obs[ObsKey.IS_SHOOTING]
    assert shooting[0, 0]
    assert shooting.sum() == 1  # map tokens and idle ships never fire


def _full_ship(live_real: int):
    """Ship 0 with every slot flying: ``live_real`` live real bullets, the rest ghosts."""
    state = _labelled_state([0j, 600 + 600j], [0, 1])
    state.bullet_flying[0, 0] = True
    state.bullet_time[0, 0] = 0.5
    ages = torch.arange(3, 3 + 3 * 10, 3, dtype=torch.int32)
    state.bullet_age[0, 0] = ages
    state.bullet_active[0, 0, :live_real] = True
    return state


def test_a_ghost_takes_a_spent_slot_and_never_a_live_real_one():
    state = _full_ship(live_real=9)
    before = state.bullet_pos[0, 0].clone()
    _handle_shooting(
        state, torch.zeros((1, 2), dtype=torch.long), CONFIG, torch.Generator().manual_seed(0)
    )
    assert state.bullet_active[0, 0, :9].all()
    assert torch.equal(state.bullet_pos[0, 0, :9], before[:9])
    assert state.bullet_age[0, 0, 9] == 0  # the only ghost slot was relaunched


def test_with_every_slot_live_the_ghost_waits():
    state = _full_ship(live_real=10)
    before = state.bullet_pos[0, 0].clone()
    _handle_shooting(
        state, torch.zeros((1, 2), dtype=torch.long), CONFIG, torch.Generator().manual_seed(0)
    )
    assert torch.equal(state.bullet_pos[0, 0], before)
    assert state.ship_ghost_cooldown[0, 0] == 0.0  # still ready for the next tick


def test_a_real_shot_with_every_slot_live_overwrites_the_oldest_as_the_ring_would():
    state = _full_ship(live_real=10)
    shoot = torch.zeros((1, 2), dtype=torch.long)
    shoot[0, 0] = int(ShootActions.SHOOT)
    _handle_shooting(state, shoot, CONFIG, torch.Generator().manual_seed(0))
    oldest = 9
    assert state.bullet_age[0, 0, oldest] == 0
    assert (state.bullet_age[0, 0, :oldest] > 0).all()
