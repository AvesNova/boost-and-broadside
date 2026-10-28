"""Counterfactual shot labels, re-simulated from stored launch states (env/shot_labels.py)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.constants import ShootActions
from boost_and_broadside.env import shot_labels
from boost_and_broadside.env.env import TensorEnv
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
    FieldLayout,
    ShotLaunchState,
    launchable_rows,
    resimulate,
)

# The Frontline rate: physics and decisions at 30 Hz.
CONFIG = replace(ShipConfig(), dt=1.0 / 30.0, bullet_spread=0.0)
DELAY = shot_labels.shot_label_delay(CONFIG)


# ---------------------------------------------------------------------------
# Hand-built scenarios: static ships facing +x, held for a whole flight
# ---------------------------------------------------------------------------


def _static(
    positions: list[complex],
    teams: list[int],
    alive: list[bool] | None = None,
    rows: int = DELAY,
    num_fields: int = 0,
) -> tuple[ShotLaunchState, FieldLayout, torch.Tensor]:
    n = len(positions)
    alive_row = torch.tensor(alive if alive is not None else [True] * n)

    def held(row: torch.Tensor) -> torch.Tensor:
        return row.expand(rows, 1, n).clone()

    launch = ShotLaunchState(
        position=held(torch.tensor(positions, dtype=torch.complex64)),
        velocity=held(torch.zeros(n, dtype=torch.complex64)),
        attitude=held(torch.ones(n, dtype=torch.complex64)),
        local_index=held(torch.ones(n)),
        field_gradient=held(torch.zeros(n, dtype=torch.complex64)),
        alive=held(alive_row),
        team_id=held(torch.tensor(teams, dtype=torch.int32)),
    )
    fields = FieldLayout(
        torch.zeros(rows, 1, num_fields, dtype=torch.complex64),
        torch.zeros(rows, 1, num_fields),
        torch.zeros(rows, 1, num_fields),
        torch.ones(rows, 1, num_fields),
    )
    return launch, fields, torch.zeros(rows, 1, dtype=torch.bool)


def _first_shot(positions, teams, alive=None) -> torch.Tensor:
    """Ship 0's shot from row 0."""
    launch, fields, ends = _static(positions, teams, alive)
    examples, valid = resimulate(launch, fields, ends, CONFIG, torch.Generator().manual_seed(0))
    assert valid[0, 0, 0]
    return examples[0, 0, 0]


def test_delay_is_one_lifetime_of_moves_at_30_hz():
    assert DELAY == 29


def test_an_enemy_in_the_line_of_fire_is_hit():
    example = _first_shot([0j, 100 + 0j, 500 + 500j], [0, 1, 0])
    assert example[EX_HIT] == ENEMY
    # 500 px/s at 30 Hz reaches 100 px within the sixth move.
    assert example[EX_EVENT_TAU] == 6
    assert example[EX_CLOSEST_DSQ] < CONFIG.collision_radius**2
    event = example[EX_SNAP_EVENT]
    assert 90.0 < event[SNAP_DX] < 120.0
    assert abs(event[SNAP_DY]) < 1e-3
    assert event[SNAP_VX] > 0.0
    assert event[SNAP_INDEX] == 1.0


def test_the_first_hit_decides_the_shot():
    # Two enemies on the line: the shot is decided by the first one it reaches.
    example = _first_shot([0j, 100 + 0j, 200 + 0j], [0, 1, 1])
    assert example[EX_HIT] == ENEMY
    assert example[EX_EVENT_TAU] == 6


def test_ally_in_the_line_of_fire_is_an_ally_hit():
    example = _first_shot([0j, 300 + 0j, 100 + 0j], [0, 1, 0])
    assert example[EX_HIT] == ALLY


def test_near_miss_records_closest_pass():
    example = _first_shot([0j, 150 + 30j, 600 + 600j], [0, 1, 0])
    assert example[EX_HIT] == NO_HIT
    assert example[EX_CLOSEST_SIDE] == ENEMY
    assert math.sqrt(example[EX_CLOSEST_DSQ]) == pytest.approx(30.0, abs=0.5)
    assert example[EX_SNAP_EVENT][SNAP_DX] == pytest.approx(150.0, abs=20.0)


def test_nothing_alive_to_pass_is_no_encounter():
    example = _first_shot([0j, 300 + 0j], [0, 1], alive=[True, False])
    assert example[EX_HIT] == NO_HIT
    assert example[EX_CLOSEST_DSQ] == NO_ENCOUNTER_DSQ


def test_a_dead_ship_fires_nothing():
    launch, fields, ends = _static([0j, 100 + 0j], [0, 1], alive=[False, True])
    examples, valid = resimulate(launch, fields, ends, CONFIG)
    assert not valid[:, :, 0].any()
    assert not examples[:, :, 0].any()


def test_scheduled_snapshots_are_taken_at_their_ages():
    example = _first_shot([0j, 600 + 600j], [0, 1])
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


def test_a_given_age_and_spread_are_used_as_given():
    launch, fields, ends = _static([0j, 600 + 600j], [0, 1])
    taus = torch.full((DELAY, 1, 2, 2), 10.0)
    noise = torch.zeros((DELAY, 1, 2), dtype=torch.complex64)
    noise[..., 0] = 30j  # straight up at 30 px/s
    examples, _ = resimulate(launch, fields, ends, CONFIG, taus=taus, noise=noise)
    snap = examples[0, 0, 0, shot_labels.EX_SNAP1]
    assert examples[0, 0, 0, EX_TAU1] == 10
    assert snap[SNAP_DY] > 0.0  # the spread carried it off the axis


# ---------------------------------------------------------------------------
# Which rows can launch
# ---------------------------------------------------------------------------


def test_a_flight_must_fit_in_the_rollout():
    rows = launchable_rows(torch.zeros(40, 2, dtype=torch.bool), DELAY)
    assert rows[: 40 - DELAY + 1].all()
    assert not rows[40 - DELAY + 1 :].any()


def test_a_flight_must_not_cross_an_episode_end():
    ends = torch.zeros(80, 2, dtype=torch.bool)
    ends[40, 0] = True
    rows = launchable_rows(ends, DELAY)
    # Rows 40 - DELAY + 2 .. 40 would read the next episode's ships; the row
    # DELAY - 1 before the end reads its last pre-reset row and is kept.
    crossing = torch.arange(80)
    crossing = (crossing >= 40 - DELAY + 2) & (crossing <= 40)
    assert not rows[crossing, 0].any()
    assert rows[40 - DELAY + 1, 0]
    assert rows[41, 0]
    assert rows[: 80 - DELAY + 1, 1].all()


def test_chunking_does_not_change_the_labels():
    env = _env(num_fields=3)
    launch, fields, ends = _record(env, 60)
    taus = torch.randint(0, DELAY + 1, (*launch.position.shape, 2)).float()
    noise = torch.zeros(launch.position.shape, dtype=torch.complex64)
    whole = resimulate(launch, fields, ends, env.ship_config, taus=taus, noise=noise)
    pieces = resimulate(
        launch, fields, ends, env.ship_config, taus=taus, noise=noise, max_bullets_per_chunk=1
    )
    assert torch.equal(whole[0], pieces[0])
    assert torch.equal(whole[1], pieces[1])


# ---------------------------------------------------------------------------
# Whole environment
# ---------------------------------------------------------------------------


def _env(shot_labels_on: bool = True, num_fields: int = 3) -> TensorEnv:
    env = TensorEnv(
        2,
        CONFIG,
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


def _record(env: TensorEnv, ticks: int, seed: int = 5, trace: list | None = None):
    """Tick ``env`` and stack its launch copies as a rollout."""
    generator = torch.Generator().manual_seed(seed)
    shape = env.state.ship_pos.shape
    launches, layouts, ends = [], [], []
    for _ in range(ticks):
        dones, truncated = env.tick(_random_actions(generator, shape))
        launches.append(env.shot_launch)
        layouts.append(env.shot_fields)
        if trace is not None:
            trace.append(
                {
                    "shooting": env.state.ship_is_shooting.clone(),
                    "damage": env.state.damage_matrix.clone(),
                }
            )
        ends.append(dones | truncated)
        env.reset_envs(dones | truncated)
    return (
        ShotLaunchState(*(torch.stack(x) for x in zip(*launches, strict=True))),
        FieldLayout(*(torch.stack(x) for x in zip(*layouts, strict=True))),
        torch.stack(ends),
    )


def test_resimulated_real_shots_land_every_hit_the_environment_applies():
    env = _env(num_fields=3)
    trace: list[dict] = []
    launch, fields, ends = _record(env, 200, trace=trace)
    noise = torch.zeros(launch.position.shape, dtype=torch.complex64)
    examples, valid = resimulate(launch, fields, ends, CONFIG, noise=noise)
    shooting = torch.stack([t["shooting"] for t in trace])  # (T, B, N)
    damage = torch.stack([t["damage"] for t in trace])  # (T, B, N, N)
    team = launch.team_id
    real = valid & shooting
    assert real.sum() > 50

    # Every hit a real shot's re-simulation predicts, the environment applied
    # on that tick to a ship of that side.
    predicted = set()
    for t, b, i in real.nonzero().tolist():
        side = int(examples[t, b, i, EX_HIT])
        if side == NO_HIT:
            continue
        tick = t + int(examples[t, b, i, EX_EVENT_TAU]) - 1
        same = team[tick, b] == team[tick, b, i]
        hit_side = same if side == ALLY else ~same
        assert (damage[tick, b, i] > 0)[hit_side].any(), (t, b, i, side)
        predicted.add((tick, b, i))
    assert predicted, "no real shot hit anything"

    # And every hit the environment applied comes from some predicted shot,
    # wherever every real launch that could have made it is a labelled row.
    checked = 0
    for tick, b, i in (damage.sum(-1) > 0).nonzero().tolist():
        if tick < DELAY - 1:
            continue  # the shot may have left before row 0
        window = slice(tick - DELAY + 1, tick + 1)
        launched = shooting[window, b, i]
        if valid[window, b, i][launched].all():
            assert (tick, b, i) in predicted, (tick, b, i)
            checked += 1
    assert checked > 0


GAMEPLAY_FIELDS = (
    "ship_pos",
    "ship_vel",
    "ship_attitude",
    "ship_health",
    "ship_power",
    "ship_cooldown",
    "ship_alive",
    "ship_is_shooting",
    "bullet_pos",
    "bullet_active",
    "damage_matrix",
    "cumulative_damage_matrix",
)


def test_labels_leave_gameplay_bit_identical():
    def run(on: bool) -> list[dict]:
        env = _env(on)
        generator = torch.Generator().manual_seed(5)
        shape = env.state.ship_pos.shape
        trace = []
        for _ in range(150):
            dones, truncated = env.tick(_random_actions(generator, shape))
            env.reset_envs(dones | truncated)
            trace.append({name: getattr(env.state, name).clone() for name in GAMEPLAY_FIELDS})
        assert (env.shot_launch is not None) == on
        return trace

    off, on = run(False), run(True)
    for tick, (a, b) in enumerate(zip(off, on, strict=True)):
        for name in GAMEPLAY_FIELDS:
            assert torch.equal(a[name], b[name]), f"{name} differs at tick {tick}"
    assert any(t["damage_matrix"].sum() > 0 for t in on), "scenario never landed a hit"


def test_labels_need_one_tick_decisions():
    with pytest.raises(ValueError, match="action_repeat"):
        EnvConfig(
            num_ships=2, max_bullets=10, max_episode_steps=10, action_repeat=2, shot_labels=True
        )


def test_is_shooting_observes_real_shots():
    from boost_and_broadside.env.observation import ObsKey, observation_from_state

    env = _env(num_fields=0)
    shape = env.state.ship_pos.shape
    idle = torch.zeros((*shape, 3), dtype=torch.long)
    env.tick(idle)
    obs = observation_from_state(env.state, env.ship_config)
    assert not obs[ObsKey.IS_SHOOTING].any()

    shoot = idle.clone()
    shoot[0, 0, 2] = int(ShootActions.SHOOT)
    env.tick(shoot)
    obs = observation_from_state(env.state, env.ship_config)
    shooting = obs[ObsKey.IS_SHOOTING]
    assert shooting[0, 0]
    assert shooting.sum() == 1  # map tokens and idle ships never fire
