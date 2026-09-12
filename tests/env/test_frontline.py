"""Frontline state-transition and hazard contract tests."""

from dataclasses import replace

import pytest
import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import (
    EnvConfig,
    FrontlineConfig,
    MatchResult,
    ShipConfig,
    ZoneRole,
)
from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import (
    FRONTLINE_WORLD_SIZE,
    apply_frontline_tick,
    roles_from_front,
    zone_membership,
)
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.evaluation.match import evaluate_matchup


def _frontline(**overrides: float | int) -> FrontlineConfig:
    values: dict[str, float | int] = {
        "zone_radius": 330.0,
        "zone_ring_radius": 1200.0,
        "playable_radius": 2600.0,
        "capture_seconds": 20.0,
        "defense_damage_per_second": 2.0,
        "respawn_health": 25.0,
        "spawn_heal_per_second": 12.0,
        "enemy_spawn_damage_per_second": 8.0,
        "boundary_damage_per_second": 5.0,
        "boundary_damage_per_pixel_second": 0.05,
        "front_win_threshold": 5,
    }
    values.update(overrides)
    return FrontlineConfig(**values)


def _env(
    frontline: FrontlineConfig | None = None,
    *,
    num_envs: int = 1,
    num_ships: int = 4,
    max_steps: int = 600,
) -> TensorEnv:
    env = TensorEnv(
        num_envs,
        ShipConfig(world_size=FRONTLINE_WORLD_SIZE),
        EnvConfig(
            num_ships=num_ships,
            max_bullets=0,
            max_episode_steps=max_steps,
            frontline=frontline or _frontline(),
        ),
        "cpu",
    )
    env.reset(seed=7)
    return env


def _zone_index(env: TensorEnv, role: ZoneRole) -> int:
    return int((env.state.zone_roles[0] == int(role)).nonzero()[0, 0])


@pytest.mark.parametrize(
    ("front", "expected"),
    [
        (0, [2, 0, 1, 3, 4]),
        (1, [4, 2, 0, 1, 3]),
        (-1, [0, 1, 3, 4, 2]),
        (5, [2, 0, 1, 3, 4]),
        (-6, [0, 1, 3, 4, 2]),
    ],
)
def test_roles_are_derived_from_unwrapped_front(front: int, expected: list[int]) -> None:
    result = roles_from_front(torch.tensor([front]))
    assert result.tolist() == [expected]


def test_frontline_requires_design_world_size() -> None:
    with pytest.raises(ValueError, match="16384"):
        TensorEnv(
            1,
            ShipConfig(),
            EnvConfig(4, 0, 60, frontline=_frontline()),
            "cpu",
        )


def test_active_defense_zones_are_physically_adjacent() -> None:
    roles = roles_from_front(torch.arange(-12, 13))
    team0_index = (roles == int(ZoneRole.TEAM0_DEFENSE)).long().argmax(dim=1)
    team1_index = (roles == int(ZoneRole.TEAM1_DEFENSE)).long().argmax(dim=1)
    cyclic_separation = (team0_index - team1_index).abs()

    assert ((cyclic_separation == 1) | (cyclic_separation == 4)).all()


def test_reset_uses_one_toroidal_translation_for_map_geometry() -> None:
    env = _env(num_envs=4)
    offsets = env.state.zone_pos - env.state.map_center.unsqueeze(1)
    offsets.real = (offsets.real + 8192.0) % 16384.0 - 8192.0
    offsets.imag = (offsets.imag + 8192.0) % 16384.0 - 8192.0
    assert torch.allclose(offsets, offsets[:1].expand_as(offsets), atol=1e-3)


def test_initial_ships_spawn_inside_current_team_spawn() -> None:
    env = _env()
    membership = zone_membership(
        env.state.ship_pos,
        env.state.zone_pos,
        env.state.zone_radius,
        env.ship_config.world_size,
    )[0]
    spawn_roles = torch.where(
        env.state.ship_team_id[0] == 0,
        int(ZoneRole.TEAM0_SPAWN),
        int(ZoneRole.TEAM1_SPAWN),
    )
    in_spawn = (membership & (env.state.zone_roles[0] == spawn_roles.unsqueeze(1))).any(dim=1)
    assert in_spawn.all()


def test_team0_capture_advances_unwrapped_front_and_rotates_roles() -> None:
    config = _frontline(capture_seconds=1.0 / 60.0)
    env = _env(config)
    target = env.state.zone_pos[0, _zone_index(env, ZoneRole.TEAM1_DEFENSE)]
    team0 = env.state.ship_team_id[0] == 0
    env.state.ship_pos[0, team0] = target

    dones, _ = env.tick(torch.zeros((1, 4, 3), dtype=torch.long))

    assert env.state.front_position.item() == 1
    assert env.state.zone_roles.tolist() == [[4, 2, 0, 1, 3]]
    assert env.state.zone_capture_progress.count_nonzero().item() == 0
    assert not dones.item()


def test_simultaneous_capture_is_atomic_and_net_zero() -> None:
    config = _frontline(capture_seconds=1.0 / 60.0, defense_damage_per_second=0.0)
    env = _env(config)
    state = env.state
    team0 = state.ship_team_id[0] == 0
    team1 = ~team0
    state.ship_pos[0, team0] = state.zone_pos[0, _zone_index(env, ZoneRole.TEAM1_DEFENSE)]
    state.ship_pos[0, team1] = state.zone_pos[0, _zone_index(env, ZoneRole.TEAM0_DEFENSE)]
    health_before = state.ship_health.clone()
    position_before = state.ship_pos.clone()

    done = apply_frontline_tick(state, config, env.ship_config)

    assert state.simultaneous_capture.item()
    assert state.front_position.item() == 0
    assert state.front_delta.item() == 0
    assert state.zone_capture_progress.count_nonzero().item() == 0
    assert torch.equal(state.ship_health, health_before)
    assert torch.equal(state.ship_pos, position_before)
    assert not done.item()


@pytest.mark.parametrize(
    ("team0_count", "team1_count", "expected_direction"),
    [
        (4, 0, 1),
        (4, 2, 1),
        (1, 0, 1),
        (2, 1, 1),
        (0, 4, -1),
        (0, 2, -1),
        (1, 2, -1),
        (2, 4, -1),
        (2, 2, 0),
    ],
)
def test_capture_pressure_depends_only_on_which_team_has_more_ships(
    team0_count: int,
    team1_count: int,
    expected_direction: int,
) -> None:
    config = _frontline(capture_seconds=10.0, defense_damage_per_second=0.0)
    env = _env(config, num_ships=8)
    state = env.state
    defense_index = _zone_index(env, ZoneRole.TEAM1_DEFENSE)
    defense = state.zone_pos[0, defense_index]

    state.ship_alive.zero_()
    state.ship_pos.fill_(state.map_center[0])
    state.ship_team_id.zero_()
    if team0_count:
        state.ship_alive[0, :team0_count] = True
        state.ship_pos[0, :team0_count] = defense
    if team1_count:
        start = team0_count
        stop = start + team1_count
        state.ship_team_id[0, start:stop] = 1
        state.ship_alive[0, start:stop] = True
        state.ship_pos[0, start:stop] = defense
    state.zone_capture_progress[0, defense_index] = 0.5

    apply_frontline_tick(state, config, env.ship_config)

    expected_progress = 0.5 + expected_direction * env.ship_config.dt / 10.0
    assert state.zone_capture_direction[0, defense_index].item() == expected_direction
    assert state.zone_capture_progress[0, defense_index].item() == pytest.approx(expected_progress)


@pytest.mark.parametrize(
    ("front", "expected"),
    [
        (2, MatchResult.TEAM0_WIN),
        (-3, MatchResult.TEAM1_WIN),
        (0, MatchResult.DRAW),
    ],
)
def test_timeout_result_uses_unwrapped_front_sign(front: int, expected: MatchResult) -> None:
    env = _env(max_steps=1)
    env.state.front_position.fill_(front)

    _, truncated = env.tick(torch.zeros((1, 4, 3), dtype=torch.long))

    assert truncated.item()
    assert env.state.match_result.item() == int(expected)


def test_friendly_spawn_heals_and_hostile_spawn_damages() -> None:
    config = _frontline(
        spawn_heal_per_second=60.0,
        enemy_spawn_damage_per_second=60.0,
        defense_damage_per_second=0.0,
    )
    env = _env(config)
    state = env.state
    state.ship_team_id[0] = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    team0_spawn = state.zone_pos[0, _zone_index(env, ZoneRole.TEAM0_SPAWN)]
    state.ship_pos[0, 0] = team0_spawn
    state.ship_pos[0, 1] = team0_spawn
    state.ship_pos[0, 2:] = state.map_center[0]
    state.ship_health[0, :2] = 50.0

    apply_frontline_tick(state, config, env.ship_config)

    assert state.ship_health[0, 0].item() == pytest.approx(51.0)
    assert state.ship_health[0, 1].item() == pytest.approx(49.0)
    assert state.ship_spawn_healing[0, 0].item() == pytest.approx(1.0)
    assert state.ship_spawn_damage[0, 1].item() == pytest.approx(1.0)


def test_defense_hazard_uses_capture_membership() -> None:
    config = _frontline(defense_damage_per_second=60.0)
    env = _env(config)
    state = env.state
    defense = state.zone_pos[0, _zone_index(env, ZoneRole.TEAM0_DEFENSE)]
    state.ship_pos[0, 0] = defense
    state.ship_pos[0, 1:] = state.map_center[0]

    apply_frontline_tick(state, config, env.ship_config)

    assert state.ship_zone_damage[0, 0].item() == pytest.approx(1.0)
    assert state.ship_zone_damage[0, 1:].count_nonzero().item() == 0


def test_boundary_damage_increases_with_distance_outside() -> None:
    config = _frontline(boundary_damage_per_second=60.0, boundary_damage_per_pixel_second=0.06)
    env = _env(config)
    state = env.state
    radius = config.playable_radius
    state.ship_pos[0, 0] = state.map_center[0] + complex(radius + 100.0, 0.0)
    state.ship_pos[0, 1] = state.map_center[0] + complex(radius + 200.0, 0.0)

    apply_frontline_tick(state, config, env.ship_config)

    assert state.ship_boundary_damage[0, 1] > state.ship_boundary_damage[0, 0] > 0.0


def test_death_respawns_same_slot_at_current_spawn_with_low_health() -> None:
    env = _env()
    state = env.state
    slot = 0
    team_before = state.ship_team_id[0, slot].clone()
    state.front_position.fill_(2)
    state.zone_roles.copy_(roles_from_front(state.front_position))
    state.ship_health[0, slot] = 0.0
    state.ship_alive[0, slot] = False
    state.ship_combat_death[0, slot] = True

    apply_frontline_tick(state, env.env_config.frontline, env.ship_config)

    expected_role = ZoneRole.TEAM0_SPAWN if team_before.item() == 0 else ZoneRole.TEAM1_SPAWN
    membership = zone_membership(
        state.ship_pos,
        state.zone_pos,
        state.zone_radius,
        env.ship_config.world_size,
    )
    expected_zone = state.zone_roles[0] == int(expected_role)
    assert state.ship_team_id[0, slot] == team_before
    assert state.ship_alive[0, slot]
    assert state.ship_respawned[0, slot]
    assert state.ship_health[0, slot].item() == env.env_config.frontline.respawn_health
    assert (membership[0, slot] & expected_zone).any()


def test_front_lead_sets_authoritative_winner() -> None:
    config = replace(_frontline(capture_seconds=1.0 / 60.0), front_win_threshold=1)
    env = _env(config)
    target = env.state.zone_pos[0, _zone_index(env, ZoneRole.TEAM1_DEFENSE)]
    team0 = env.state.ship_team_id[0] == 0
    env.state.ship_pos[0, team0] = target

    dones, _ = env.tick(torch.zeros((1, 4, 3), dtype=torch.long))

    assert dones.item()
    assert env.state.match_result.item() == int(MatchResult.TEAM0_WIN)


def test_wrapper_reports_respawn_discontinuity_without_ending_episode() -> None:
    config = _frontline(
        defense_damage_per_second=0.0,
        enemy_spawn_damage_per_second=0.0,
        boundary_damage_per_second=6000.0,
        boundary_damage_per_pixel_second=0.0,
    )
    wrapper = YemongEnvWrapper(
        1,
        ShipConfig(world_size=FRONTLINE_WORLD_SIZE),
        EnvConfig(4, 0, 600, frontline=config),
        REWARDS,
        "cpu",
    )
    wrapper.reset(seed=3)
    slot = 0
    team_before = wrapper.state.ship_team_id[0, slot].clone()
    wrapper.state.ship_health[0, slot] = 1.0
    wrapper.state.ship_pos[0, slot] = wrapper.state.map_center[0] + complex(
        config.playable_radius + 10.0, 0.0
    )
    action = torch.zeros((1, 4, 3), dtype=torch.int32)
    action[0, slot] = torch.tensor([1, 3, 1])

    _, _, dones, truncated, info = wrapper.step(action)

    assert not (dones | truncated).item()
    assert not info["transition_contiguous"][0, slot]
    assert wrapper.state.ship_respawned[0, slot]
    assert wrapper.state.ship_team_id[0, slot] == team_before
    assert torch.equal(wrapper.state.prev_action[0, slot], action[0, slot].float())


def test_wrapper_can_hold_actual_terminal_state_for_frontend() -> None:
    wrapper = YemongEnvWrapper(
        1,
        ShipConfig(world_size=FRONTLINE_WORLD_SIZE),
        EnvConfig(4, 0, 1, frontline=_frontline()),
        REWARDS,
        "cpu",
    )
    wrapper.reset(seed=3)

    _, _, _, truncated, info = wrapper.step(
        torch.zeros((1, 4, 3), dtype=torch.int32),
        auto_reset=False,
    )

    assert truncated.item()
    assert wrapper.state.step_count.item() == 1
    assert wrapper.state.match_result.item() == info["match_result"].item()
    assert wrapper.state.match_result.item() == int(MatchResult.DRAW)


def test_scripted_frontline_match_uses_authoritative_timeout_result() -> None:
    ship_config = ShipConfig(world_size=FRONTLINE_WORLD_SIZE)
    env_config = EnvConfig(4, 0, 30, frontline=_frontline())
    scripted = ResolvedAgent(
        "scripted",
        StochasticScriptedAgent(ship_config, StochasticAgentConfig()),
    )

    team0_wins, team1_wins, draws, mean_length = evaluate_matchup(
        scripted,
        scripted,
        2,
        2,
        2,
        ship_config,
        env_config,
        "cpu",
    )

    assert (team0_wins, team1_wins, draws) == (0, 0, 2)
    assert mean_length == 30.0
