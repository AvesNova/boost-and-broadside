"""Focused tests for the Gate-1 frontline scripted-controller priorities."""

import pytest
import torch

from boost_and_broadside.agents.scripted_utils import select_targets
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import (
    FrontlineTendency,
    StochasticScriptedAgent,
)
from boost_and_broadside.config import ShipConfig, ZoneRole
from tests.conftest import make_state


def _frontline_state(*, teams: list[int] | None = None):
    teams = teams or [0, 0, 1, 1]
    config = ShipConfig(world_size=(16384.0, 16384.0))
    state = make_state(
        num_envs=1,
        max_ships=len(teams),
        max_bullets=0,
        ship_config=config,
    )
    state.ship_team_id[0] = torch.tensor(teams, dtype=torch.int32)
    state.zone_pos = torch.tensor(
        [[100.0 + 100.0j, 3000.0 + 100.0j, 6000.0 + 100.0j, 9000.0 + 100.0j, 12000.0 + 100.0j]],
        dtype=torch.complex64,
    )
    state.zone_radius = torch.full((1, 5), 200.0)
    state.zone_roles = torch.tensor(
        [
            [
                int(ZoneRole.NEUTRAL),
                int(ZoneRole.TEAM0_SPAWN),
                int(ZoneRole.TEAM0_DEFENSE),
                int(ZoneRole.TEAM1_DEFENSE),
                int(ZoneRole.TEAM1_SPAWN),
            ]
        ],
        dtype=torch.int8,
    )
    state.zone_capture_progress = torch.zeros((1, 5))
    state.zone_capture_direction = torch.zeros((1, 5), dtype=torch.int8)
    return config, state


def _targeting(agent: StochasticScriptedAgent, state):
    closest_dist, target_idx, has_target, _ = select_targets(state, agent.ship_config)
    return agent._frontline_targets(state, closest_dist, target_idx, has_target)


def _set_tendencies(
    agent: StochasticScriptedAgent,
    state,
    tendencies: list[FrontlineTendency],
    *,
    tie_attack: tuple[bool, bool] = (False, False),
) -> None:
    memory = agent._frontline_episode_memory(state)
    memory.tendencies[0] = torch.tensor(tendencies)
    memory.tie_attack[0] = torch.tensor(tie_attack)


def _bearing(source: complex, target: complex, world_size: tuple[float, float]) -> complex:
    width, height = world_size
    dx = (target.real - source.real + width / 2.0) % width - width / 2.0
    dy = (target.imag - source.imag + height / 2.0) % height - height / 2.0
    displacement = complex(dx, dy)
    if displacement == 0.0j:
        return 0.0j
    return displacement / abs(displacement)


def test_frontline_identity_draw_is_fifty_twenty_five_twenty_five() -> None:
    config = ShipConfig(world_size=(16384.0, 16384.0))
    state = make_state(
        num_envs=4096,
        max_ships=8,
        max_bullets=0,
        ship_config=config,
    )
    with torch.random.fork_rng():
        torch.manual_seed(20260909)
        agent = StochasticScriptedAgent(config, StochasticAgentConfig())
        tendencies = agent._frontline_episode_memory(state).tendencies.flatten()

    proportions = torch.bincount(tendencies, minlength=3).float() / tendencies.numel()
    assert proportions.tolist() == pytest.approx([0.5, 0.25, 0.25], abs=0.01)


def test_timid_low_health_disengages_from_nearby_enemy_until_fully_healed() -> None:
    config, state = _frontline_state()
    state.ship_pos[0] = torch.tensor(
        [1000.0 + 100.0j, 2000.0 + 100.0j, 1100.0 + 100.0j, 10000.0 + 100.0j]
    )
    state.ship_health[0, 0] = 29.0
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=500.0),
    )
    _set_tendencies(
        agent,
        state,
        [
            FrontlineTendency.TIMID,
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.DEFENSIVE,
            FrontlineTendency.TIMID,
        ],
    )

    distance, bearing, engage = _targeting(agent, state)

    assert not engage[0, 0]
    assert distance[0, 0].item() == pytest.approx(2000.0)
    assert bearing[0, 0].item() == pytest.approx(1.0 + 0.0j)

    state.ship_health[0, 0] = 99.0
    _, _, engage = _targeting(agent, state)
    assert not engage[0, 0]

    state.ship_health[0, 0] = config.max_health
    distance, bearing, engage = _targeting(agent, state)
    assert engage[0, 0]
    assert distance[0, 0].item() == pytest.approx(100.0)
    assert bearing[0, 0].item() == pytest.approx(1.0 + 0.0j)


def test_timid_at_thirty_percent_does_not_retreat() -> None:
    config, state = _frontline_state()
    state.ship_pos[0] = torch.tensor(
        [1000.0 + 100.0j, 2000.0 + 100.0j, 9000.0 + 100.0j, 10000.0 + 100.0j]
    )
    state.ship_health[0, 0] = 30.0
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=0.0),
    )
    _set_tendencies(
        agent,
        state,
        [
            FrontlineTendency.TIMID,
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.DEFENSIVE,
            FrontlineTendency.DEFENSIVE,
        ],
    )

    distance, bearing, engage = _targeting(agent, state)

    assert not engage[0, 0]
    assert distance[0, 0].item() == pytest.approx(4000.0)
    assert bearing[0, 0].item() == pytest.approx(1.0 + 0.0j)


def test_single_enemy_on_point_rallies_both_teams_without_defender_presence() -> None:
    config, state = _frontline_state(teams=[0, 0, 0, 1, 1, 1])
    state.ship_pos[0] = torch.tensor(
        [
            9000.0 + 100.0j,  # lone team-0 attacker in team 1's defense
            1000.0 + 100.0j,
            2000.0 + 100.0j,
            11000.0 + 100.0j,
            12000.0 + 100.0j,
            13000.0 + 100.0j,
        ]
    )
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=0.0),
    )
    _set_tendencies(
        agent,
        state,
        [FrontlineTendency.DEFENSIVE] * 3 + [FrontlineTendency.OFFENSIVE] * 3,
    )

    distance, bearing, engage = _targeting(agent, state)

    assert not engage.any()
    expected = torch.tensor(
        [
            _bearing(state.ship_pos[0, i].item(), 9000.0 + 100.0j, config.world_size)
            for i in range(6)
        ],
        dtype=torch.complex64,
    )
    assert torch.allclose(bearing[0], expected)
    assert distance[0, 0].item() == pytest.approx(0.0)


def test_healthy_ship_engages_nearby_enemy_over_objective_with_toroidal_bearing() -> None:
    config, state = _frontline_state()
    state.ship_pos[0] = torch.tensor(
        [16370.0 + 100.0j, 4000.0 + 100.0j, 10.0 + 100.0j, 10000.0 + 100.0j]
    )
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=500.0),
    )

    distance, bearing, engage = _targeting(agent, state)

    assert engage[0, 0]
    assert distance[0, 0].item() == pytest.approx(24.0)
    assert bearing[0, 0].item() == pytest.approx(1.0 + 0.0j)


def test_both_contested_points_restore_episode_tendencies_and_timid_majority() -> None:
    config, state = _frontline_state(teams=[0, 0, 0, 0, 1, 1, 1, 1])
    state.ship_pos[0] = torch.tensor(
        [
            9000.0 + 100.0j,  # team 0 contests team 1's defense
            1000.0 + 100.0j,
            2000.0 + 100.0j,
            3000.0 + 100.0j,
            6000.0 + 100.0j,  # team 1 contests team 0's defense
            11000.0 + 100.0j,
            12000.0 + 100.0j,
            13000.0 + 100.0j,
        ]
    )
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=0.0),
    )
    _set_tendencies(
        agent,
        state,
        [
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.DEFENSIVE,
            FrontlineTendency.TIMID,  # team-0 majority attacks
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.DEFENSIVE,
            FrontlineTendency.DEFENSIVE,
            FrontlineTendency.TIMID,  # team-1 majority defends
        ],
    )

    _, bearing, engage = _targeting(agent, state)

    expected_targets = [
        9000.0 + 100.0j,
        9000.0 + 100.0j,
        6000.0 + 100.0j,
        9000.0 + 100.0j,
        6000.0 + 100.0j,
        9000.0 + 100.0j,
        9000.0 + 100.0j,
        9000.0 + 100.0j,
    ]
    expected = torch.tensor(
        [
            _bearing(state.ship_pos[0, i].item(), target, config.world_size)
            for i, target in enumerate(expected_targets)
        ],
        dtype=torch.complex64,
    )
    assert not engage.any()
    assert torch.allclose(bearing[0], expected)


def test_neither_point_contested_uses_tendencies_and_stable_tie_break() -> None:
    config, state = _frontline_state(teams=[0, 0, 0, 0, 1, 1, 1, 1])
    state.ship_pos[0] = torch.tensor(
        [
            1000.0 + 100.0j,
            1400.0 + 100.0j,
            1800.0 + 100.0j,
            2200.0 + 100.0j,
            10000.0 + 100.0j,
            10400.0 + 100.0j,
            10800.0 + 100.0j,
            11200.0 + 100.0j,
        ]
    )
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=0.0),
    )
    _set_tendencies(
        agent,
        state,
        [
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.DEFENSIVE,
            FrontlineTendency.TIMID,
            FrontlineTendency.TIMID,
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.DEFENSIVE,
            FrontlineTendency.TIMID,
            FrontlineTendency.TIMID,
        ],
        tie_attack=(True, False),
    )

    distance, bearing, engage = _targeting(agent, state)

    expected = torch.tensor(
        [
            _bearing(state.ship_pos[0, i].item(), 5000.0 + 100.0j, config.world_size)
            for i in (0, 2, 3)
        ],
        dtype=torch.complex64,
    )
    defender_target = state.ship_pos[0, 1] + distance[0, 1] * bearing[0, 1]
    assert not engage.any()
    assert torch.allclose(bearing[0, [0, 2, 3]], expected)
    assert abs(defender_target.item() - (6000.0 + 100.0j)) == pytest.approx(220.0)


def test_offensive_wave_gathers_one_third_from_spawn_before_advancing() -> None:
    config, state = _frontline_state()
    state.ship_pos[0] = torch.tensor(
        [1000.0 + 100.0j, 2000.0 + 100.0j, 10000.0 + 100.0j, 11000.0 + 100.0j]
    )
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=0.0),
    )
    _set_tendencies(
        agent,
        state,
        [
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.DEFENSIVE,
            FrontlineTendency.DEFENSIVE,
        ],
    )

    _, bearing, engage = _targeting(agent, state)
    expected_rally = torch.tensor(
        [
            _bearing(state.ship_pos[0, i].item(), 5000.0 + 100.0j, config.world_size)
            for i in (0, 1)
        ]
    )
    assert not engage.any()
    assert torch.allclose(bearing[0, :2], expected_rally)

    state.ship_pos[0, 0] = 4900.0 + 100.0j
    state.ship_pos[0, 1] = 5000.0 + 100.0j
    _, bearing, _ = _targeting(agent, state)
    expected_attack = torch.tensor(
        [
            _bearing(state.ship_pos[0, i].item(), 9000.0 + 100.0j, config.world_size)
            for i in (0, 1)
        ]
    )
    assert torch.allclose(bearing[0, :2], expected_attack)

    state.ship_pos[0, 0] = 3000.0 + 100.0j
    state.ship_pos[0, 1] = 7000.0 + 100.0j
    state.ship_respawned[0, 0] = True
    _, bearing, _ = _targeting(agent, state)
    assert bearing[0, 1].item() == pytest.approx(-1.0 + 0.0j)


def test_idle_defender_patrol_waypoint_moves_around_safe_perimeter() -> None:
    config, state = _frontline_state()
    state.ship_pos[0] = torch.tensor(
        [1000.0 + 100.0j, 2000.0 + 100.0j, 10000.0 + 100.0j, 11000.0 + 100.0j]
    )
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=0.0),
    )
    _set_tendencies(
        agent,
        state,
        [
            FrontlineTendency.DEFENSIVE,
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.OFFENSIVE,
        ],
    )

    distance, bearing, engage = _targeting(agent, state)
    target = state.ship_pos[0, 0] + distance[0, 0] * bearing[0, 0]

    assert not engage[0, 0]
    assert abs(target.item() - (6000.0 + 100.0j)) == pytest.approx(220.0)
    assert target.imag.item() < 100.0

    state.ship_pos[0, 0] = 6000.0 + 1100.0j
    distance, bearing, engage = _targeting(agent, state)
    advanced_target = state.ship_pos[0, 0] + distance[0, 0] * bearing[0, 0]

    assert not engage[0, 0]
    assert abs(advanced_target.item() - (6000.0 + 100.0j)) == pytest.approx(220.0)
    assert advanced_target != pytest.approx(target)


def test_respawn_healing_rules_and_local_battle_priority() -> None:
    config, state = _frontline_state(teams=[0, 0, 1, 1])
    state.ship_pos[0] = torch.tensor(
        [3000.0 + 100.0j, 1000.0 + 100.0j, 1100.0 + 100.0j, 9000.0 + 100.0j]
    )
    state.ship_health[0, :2] = 25.0
    state.ship_respawned[0, :2] = True
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=500.0),
    )
    _set_tendencies(
        agent,
        state,
        [
            FrontlineTendency.TIMID,
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.DEFENSIVE,
            FrontlineTendency.DEFENSIVE,
        ],
    )

    distance, _, engage = _targeting(agent, state)

    assert not engage[0, 0]  # timid respawn healing overrides everything
    assert distance[0, 0].item() == pytest.approx(0.0)
    assert engage[0, 1]  # non-timid ships still prioritize a local battle
    assert distance[0, 1].item() == pytest.approx(100.0)

    state.ship_pos[0, 2] = 10000.0 + 100.0j
    state.ship_respawned.zero_()
    distance, bearing, engage = _targeting(agent, state)
    assert not engage[0, 1]
    assert distance[0, 1].item() == pytest.approx(2000.0)
    assert bearing[0, 1].item() == pytest.approx(1.0 + 0.0j)

    state.ship_health[0, 1] = config.max_health
    distance, _, _ = _targeting(agent, state)
    assert distance[0, 1].item() == pytest.approx(4000.0)


def test_contested_point_interrupts_non_timid_but_not_timid_respawn_healing() -> None:
    config, state = _frontline_state(teams=[0, 0, 1, 1])
    state.ship_pos[0] = torch.tensor(
        [3000.0 + 100.0j, 3000.0 + 100.0j, 6000.0 + 100.0j, 12000.0 + 100.0j]
    )
    state.ship_health[0, :2] = 25.0
    state.ship_respawned[0, :2] = True
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=0.0),
    )
    _set_tendencies(
        agent,
        state,
        [
            FrontlineTendency.TIMID,
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.DEFENSIVE,
            FrontlineTendency.DEFENSIVE,
        ],
    )

    distance, bearing, engage = _targeting(agent, state)

    assert not engage.any()
    assert distance[0, 0].item() == pytest.approx(0.0)
    assert distance[0, 1].item() == pytest.approx(3000.0)
    assert bearing[0, 1].item() == pytest.approx(1.0 + 0.0j)


def test_tendency_survives_respawn_and_healing_latch_clears_on_episode_reset() -> None:
    config, state = _frontline_state()
    agent = StochasticScriptedAgent(config, StochasticAgentConfig())
    _set_tendencies(
        agent,
        state,
        [
            FrontlineTendency.DEFENSIVE,
            FrontlineTendency.OFFENSIVE,
            FrontlineTendency.DEFENSIVE,
            FrontlineTendency.OFFENSIVE,
        ],
    )
    memory = agent._frontline_episode_memory(state)
    expected = memory.tendencies.clone()

    state.step_count.fill_(1)
    state.ship_health[0, 0] = 25.0
    state.ship_respawned[0, 0] = True
    _targeting(agent, state)
    assert memory.healing[0, 0]
    assert torch.equal(memory.tendencies, expected)

    state.step_count.zero_()
    state.ship_respawned.zero_()
    state.ship_health.fill_(config.max_health)
    _targeting(agent, state)
    assert not memory.healing.any()


def test_navigation_targets_never_shoot() -> None:
    config, state = _frontline_state()
    state.ship_pos[0] = torch.tensor(
        [1000.0 + 100.0j, 2000.0 + 100.0j, 10000.0 + 100.0j, 11000.0 + 100.0j]
    )
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=0.0),
    )

    actions, probs = agent.get_actions_and_probs(state)

    assert torch.equal(probs[..., -2], torch.ones_like(probs[..., -2]))
    assert torch.equal(probs[..., -1], torch.zeros_like(probs[..., -1]))
    assert not actions[..., 2].any()


def test_zero_zone_combat_mode_never_enters_frontline_path(monkeypatch) -> None:
    config = ShipConfig()
    state = make_state(num_envs=1, max_ships=2, max_bullets=0, ship_config=config)
    state.ship_team_id[0] = torch.tensor([0, 1], dtype=torch.int32)
    state.ship_pos[0] = torch.tensor([100.0 + 100.0j, 200.0 + 100.0j])
    agent = StochasticScriptedAgent(config, StochasticAgentConfig())

    def fail(_state):
        raise AssertionError("legacy combat mode entered the frontline controller")

    monkeypatch.setattr(agent, "_get_frontline_actions_and_probs", fail)
    actions, probs = agent.get_actions_and_probs(state)

    assert actions.shape == (1, 2, 3)
    assert probs.shape == (1, 2, 12)


@pytest.mark.parametrize(
    "overrides",
    [
        {"frontline_heal_health_fraction": -0.01},
        {"frontline_heal_health_fraction": 1.01},
        {"frontline_enemy_engage_distance": -1.0},
    ],
)
def test_frontline_scripted_config_rejects_invalid_values(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        StochasticAgentConfig(**overrides)
